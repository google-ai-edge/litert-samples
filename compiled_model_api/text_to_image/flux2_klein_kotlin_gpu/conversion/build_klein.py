"""FLUX.2-klein-4B single-stream block -> GPU-clean LiteRT graph (probe).

Mirrors the Z-Image playbook: reimplement one Flux2SingleTransformerBlock with
only GPU-delegate-friendly ops, prove it is bit-exact against the diffusers
reference, then convert and op-check.

Patches reused from Z-Image:
  * interleaved RoPE -> bake the even/odd de-interleave permutation into the Q/K
    row blocks of the FUSED to_qkv_mlp_proj (and into norm_q/norm_k weights), so
    RoPE becomes a contiguous half-split (SLICE) rather than a stride-2
    GATHER_ND. Exact: q.k is invariant to a shared channel permutation.
  * fp16-safe max-normalized RMSNorm / LayerNorm.
  * scaled_dot_product_attention -> manual matmul + softmax.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# klein-4B transformer config
DIM = 3072
N_HEADS = 24
HEAD_DIM = 128
HALF = HEAD_DIM // 2
MLP_RATIO = 3.0
EPS = 1e-6


def even_odd_perm():
    """Per-head channel permutation [0,2,..,126, 1,3,..,127] (evens, odds)."""
    return list(range(0, HEAD_DIM, 2)) + list(range(1, HEAD_DIM, 2))


def safe_rms(x, weight, eps=1e-6):
    """fp16-safe max-normalized RMSNorm over the last dimension.

    Args:
      x: Input tensor.
      weight: Per-channel scale.
      eps: Variance epsilon.

    Returns:
      The normalized tensor, same shape as x.
    """
    m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    xs = x / m
    r = torch.rsqrt(xs.pow(2).mean(dim=-1, keepdim=True) + (eps / (m * m)))
    return x * r / m * weight


def safe_rms_4d(x, weight, eps=1e-6):
    """safe_rms on a [B,S,H,hd] tensor, reshaped to 3D first.

    An amax over a 4D tensor trips litert-torch's NHWC layout pass with
    "NHWC node rewriter not found: amax". The reduction is over the last
    dimension either way, so the reshape is exact.

    Args:
      x: Input tensor shaped [batch, seq, heads, head_dim].
      weight: Per-channel scale over head_dim.
      eps: Variance epsilon.

    Returns:
      The normalized tensor, same shape as x.
    """
    b, s, h, d = x.shape
    return safe_rms(x.reshape(b, s * h, d), weight, eps).reshape(b, s, h, d)


def safe_ln_noaffine(x, eps=1e-6):
    """fp16-safe LayerNorm without affine parameters, max-normalized.

    Args:
      x: Input tensor.
      eps: Variance epsilon.

    Returns:
      The normalized tensor, same shape as x.
    """
    mu = x.mean(dim=-1, keepdim=True)
    d = x - mu
    m = d.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    ds = d / m
    variance = ds.pow(2).mean(dim=-1, keepdim=True)
    return ds * torch.rsqrt(variance + (eps / (m * m)))


def rope_halfsplit(x, cos, sin):
    """Half-split real RoPE.

    Args:
      x: Query or key shaped [B,S,H,hd], pre-permuted to evens then odds.
      cos: Cosine table shaped [1,S,1,hd//2].
      sin: Sine table shaped [1,S,1,hd//2].

    Returns:
      The rotated tensor, same shape as x.
    """
    x_e = x[..., :HALF]
    x_o = x[..., HALF:]
    return torch.cat([x_e * cos - x_o * sin, x_o * cos + x_e * sin], dim=-1)


class ExportKleinSingle(nn.Module):
    """GPU-clean reimplementation of one Flux2SingleTransformerBlock."""

    def __init__(self, block):
        super().__init__()
        attn = block.attn
        self.inner = attn.inner_dim
        self.mlp_hidden = attn.mlp_hidden_dim
        perm = even_odd_perm()

        # Bake the de-interleave permutation into the Q and K row blocks of the
        # fused to_qkv_mlp_proj weight (rows: [Q | K | V | MLP]).
        w = attn.to_qkv_mlp_proj.weight.detach().clone()
        full = torch.tensor(
            [h * HEAD_DIM + i for h in range(N_HEADS) for i in perm],
            dtype=torch.long)
        w[:self.inner] = w[:self.inner][full]
        w[self.inner:2 * self.inner] = w[self.inner:2 * self.inner][full]
        self.qkv_mlp = nn.Linear(w.shape[1], w.shape[0], bias=False)
        self.qkv_mlp.weight = nn.Parameter(w)

        perm_t = torch.tensor(perm, dtype=torch.long)
        norm_q = attn.norm_q.weight.detach()[perm_t].clone()
        norm_k = attn.norm_k.weight.detach()[perm_t].clone()
        self.register_buffer("nq_w", norm_q)
        self.register_buffer("nk_w", norm_k)
        self.to_out = attn.to_out

    def forward(self, hidden, cos, sin, shift, scale, gate):
        n = safe_ln_noaffine(hidden)
        n = (1 + scale) * n + shift

        h = self.qkv_mlp(n)
        qkv = h[..., : 3 * self.inner]
        mlp = h[..., 3 * self.inner:]
        b, s, _ = qkv.shape
        q = qkv[..., : self.inner].reshape(b, s, N_HEADS, HEAD_DIM)
        k = qkv[..., self.inner: 2 * self.inner].reshape(
            b, s, N_HEADS, HEAD_DIM)
        v = qkv[..., 2 * self.inner:].reshape(b, s, N_HEADS, HEAD_DIM)

        q = safe_rms_4d(q, self.nq_w, EPS)
        k = safe_rms_4d(k, self.nk_w, EPS)
        q = rope_halfsplit(q, cos, sin)
        k = rope_halfsplit(k, cos, sin)

        # manual SDPA over [B,H,S,hd]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        logits = q @ k.transpose(-1, -2) * (HEAD_DIM ** -0.5)
        w_attn = torch.softmax(logits, dim=-1)
        attn_out = (w_attn @ v).permute(0, 2, 1, 3).reshape(b, s, self.inner)

        g, u = mlp[..., : self.mlp_hidden], mlp[..., self.mlp_hidden:]
        mlp_out = (g * torch.sigmoid(g)) * u   # SiLU(gate) * up
        out = self.to_out(torch.cat([attn_out, mlp_out], dim=-1))
        return hidden + gate * out


def main():
    """Checks the GPU-clean block against diffusers, then converts it."""
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2SingleTransformerBlock)

    torch.manual_seed(0)
    block = Flux2SingleTransformerBlock(DIM, N_HEADS, HEAD_DIM, MLP_RATIO, EPS,
                                        bias=False).eval()
    # random but valid weights (bias=False everywhere except to_out)
    seq = 64
    hidden = torch.randn(1, seq, DIM) * 0.5
    # modulation params (what Flux2Modulation.split yields for one param set)
    shift = torch.randn(1, 1, DIM) * 0.1
    scale = torch.randn(1, 1, DIM) * 0.1
    gate = torch.randn(1, 1, DIM) * 0.1
    # Flux2Modulation.split yields the parameters in this order.
    temb_mod = torch.cat([shift, scale, gate], dim=-1)

    # interleaved cos/sin (repeat_interleave_real=True) for the reference
    ang = torch.randn(seq, HALF)
    cos_h, sin_h = torch.cos(ang), torch.sin(ang)
    cos_full = cos_h.repeat_interleave(2, dim=-1)
    sin_full = sin_h.repeat_interleave(2, dim=-1)

    with torch.no_grad():
        ref = block(hidden, None, temb_mod,
                    image_rotary_emb=(cos_full, sin_full))
    if isinstance(ref, tuple):
        ref = ref[-1]

    mine_mod = ExportKleinSingle(block).eval()
    cos = cos_h[None, :, None, :]
    sin = sin_h[None, :, None, :]
    with torch.no_grad():
        mine = mine_mod(hidden, cos, sin, shift, scale, gate)

    d = (ref - mine).abs()
    corr = torch.corrcoef(torch.stack([ref.flatten(), mine.flatten()]))[0, 1]
    print(f"[parity] ref {tuple(ref.shape)} vs gpu-clean: corr {corr:.8f}  "
          f"max|diff| {d.max():.2e}  range [{ref.min():.2f},{ref.max():.2f}]")

    if "--convert" in sys.argv:
        import litert_torch
        out = "klein_single_probe.tflite"
        inputs = (hidden, cos, sin, shift, scale, gate)
        litert_torch.convert(mine_mod, inputs).export(out)
        import os
        print(f"[convert] {out}  {os.path.getsize(out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
