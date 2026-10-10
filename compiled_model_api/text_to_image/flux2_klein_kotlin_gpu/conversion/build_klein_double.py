"""FLUX.2-klein-4B double-stream block -> GPU-clean LiteRT graph (probe).

`Flux2TransformerBlock` (5 of klein's 25 blocks) keeps separate image and text
streams and joins them inside attention: q/k/v are computed per stream, the text
tokens are concatenated in front of the image tokens, RoPE is applied to the
joint sequence, one attention runs over it, and the output is split back.

Same patch set as the single-stream probe:
  * baked even/odd de-interleave permutation, here into FOUR projections
    (to_q, to_k, add_q_proj, add_k_proj) and their four qk-norm weights, so the
    joint RoPE becomes a contiguous half-split. Exact: q.k is invariant to a
    shared channel permutation, and both streams share it.
  * fp16-safe max-normalized LayerNorm / RMSNorm (amax reduced in 3D).
  * scaled_dot_product_attention -> manual matmul + softmax.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_klein import (DIM, EPS, HEAD_DIM, MLP_RATIO, N_HEADS, even_odd_perm,
                         rope_halfsplit, safe_ln_noaffine, safe_rms_4d)


def _permute_rows(linear, full_perm):
    """Clones a Linear with its output rows permuted.

    Args:
      linear: The layer to clone.
      full_perm: Row index map over the full output dimension.

    Returns:
      A new nn.Linear whose rows follow full_perm.
    """
    w = linear.weight.detach()[full_perm].clone()
    new = nn.Linear(w.shape[1], w.shape[0], bias=linear.bias is not None)
    new.weight = nn.Parameter(w)
    if linear.bias is not None:
        new.bias = nn.Parameter(linear.bias.detach()[full_perm].clone())
    return new


class ExportKleinDouble(nn.Module):
    """GPU-clean reimplementation of one Flux2TransformerBlock."""

    def __init__(self, block):
        super().__init__()
        attn = block.attn
        self.inner = attn.inner_dim
        perm = even_odd_perm()
        full = torch.tensor(
            [h * HEAD_DIM + i for h in range(N_HEADS) for i in perm],
            dtype=torch.long)
        perm_t = torch.tensor(perm, dtype=torch.long)

        # Q/K of BOTH streams get the same channel permutation, since they
        # are concatenated and rotated together. V and the output projections
        # do not.
        self.to_q = _permute_rows(attn.to_q, full)
        self.to_k = _permute_rows(attn.to_k, full)
        self.to_v = attn.to_v
        self.add_q = _permute_rows(attn.add_q_proj, full)
        self.add_k = _permute_rows(attn.add_k_proj, full)
        self.add_v = attn.add_v_proj
        self.register_buffer(
            "nq_w", attn.norm_q.weight.detach()[perm_t].clone())
        self.register_buffer(
            "nk_w", attn.norm_k.weight.detach()[perm_t].clone())
        self.register_buffer(
            "naq_w", attn.norm_added_q.weight.detach()[perm_t].clone())
        self.register_buffer(
            "nak_w", attn.norm_added_k.weight.detach()[perm_t].clone())
        self.to_out = attn.to_out[0]
        self.to_add_out = attn.to_add_out
        self.ff_in, self.ff_out = block.ff.linear_in, block.ff.linear_out
        self.ffc_in = block.ff_context.linear_in
        self.ffc_out = block.ff_context.linear_out

    @staticmethod
    def _swiglu(x):
        g, u = x.chunk(2, dim=-1)
        return (g * torch.sigmoid(g)) * u

    def forward(self, hidden, encoder, cos, sin, mod_img, mod_txt):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = mod_img.chunk(6, dim=-1)
        (c_s_msa, c_sc_msa, c_g_msa,
         c_s_mlp, c_sc_mlp, c_g_mlp) = mod_txt.chunk(6, dim=-1)

        n_h = (1 + sc_msa) * safe_ln_noaffine(hidden) + s_msa
        n_e = (1 + c_sc_msa) * safe_ln_noaffine(encoder) + c_s_msa

        b, si, _ = n_h.shape
        st = n_e.shape[1]
        q = self.to_q(n_h).reshape(b, si, N_HEADS, HEAD_DIM)
        k = self.to_k(n_h).reshape(b, si, N_HEADS, HEAD_DIM)
        v = self.to_v(n_h).reshape(b, si, N_HEADS, HEAD_DIM)
        eq = self.add_q(n_e).reshape(b, st, N_HEADS, HEAD_DIM)
        ek = self.add_k(n_e).reshape(b, st, N_HEADS, HEAD_DIM)
        ev = self.add_v(n_e).reshape(b, st, N_HEADS, HEAD_DIM)

        q = safe_rms_4d(q, self.nq_w, EPS)
        k = safe_rms_4d(k, self.nk_w, EPS)
        eq = safe_rms_4d(eq, self.naq_w, EPS)
        ek = safe_rms_4d(ek, self.nak_w, EPS)

        q = rope_halfsplit(torch.cat([eq, q], dim=1), cos, sin)
        k = rope_halfsplit(torch.cat([ek, k], dim=1), cos, sin)
        v = torch.cat([ev, v], dim=1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        w = torch.softmax(q @ k.transpose(-1, -2) * (HEAD_DIM ** -0.5), dim=-1)
        out = (w @ v).permute(0, 2, 1, 3).reshape(b, st + si, self.inner)

        ctx_out = self.to_add_out(out[:, :st])
        img_out = self.to_out(out[:, st:])

        hidden = hidden + g_msa * img_out
        n_h = safe_ln_noaffine(hidden) * (1 + sc_mlp) + s_mlp
        hidden = hidden + g_mlp * self.ff_out(self._swiglu(self.ff_in(n_h)))

        encoder = encoder + c_g_msa * ctx_out
        n_e = safe_ln_noaffine(encoder) * (1 + c_sc_mlp) + c_s_mlp
        ff_context = self.ffc_out(self._swiglu(self.ffc_in(n_e)))
        encoder = encoder + c_g_mlp * ff_context
        return hidden, encoder


def main():
    """Checks the GPU-clean block against diffusers, then converts it."""
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2TransformerBlock)

    torch.manual_seed(0)
    block = Flux2TransformerBlock(DIM, N_HEADS, HEAD_DIM, MLP_RATIO, EPS,
                                  bias=False).eval()
    s_img, s_txt = 48, 16
    hidden = torch.randn(1, s_img, DIM) * 0.5
    encoder = torch.randn(1, s_txt, DIM) * 0.5
    mod_img = torch.randn(1, 1, DIM * 6) * 0.1
    mod_txt = torch.randn(1, 1, DIM * 6) * 0.1

    ang = torch.randn(s_txt + s_img, HEAD_DIM // 2)
    cos_h, sin_h = torch.cos(ang), torch.sin(ang)
    cos_full = cos_h.repeat_interleave(2, dim=-1)
    sin_full = sin_h.repeat_interleave(2, dim=-1)

    with torch.no_grad():
        # diffusers returns (encoder_hidden_states, hidden_states)
        ref_e, ref_h = block(hidden, encoder, mod_img, mod_txt,
                             image_rotary_emb=(cos_full, sin_full))

    mine_mod = ExportKleinDouble(block).eval()
    cos = cos_h[None, :, None, :]
    sin = sin_h[None, :, None, :]
    with torch.no_grad():
        my_h, my_e = mine_mod(hidden, encoder, cos, sin, mod_img, mod_txt)

    for tag, r, m in (("image", ref_h, my_h), ("text", ref_e, my_e)):
        corr = torch.corrcoef(torch.stack([r.flatten(), m.flatten()]))[0, 1]
        print(f"[parity] {tag} {tuple(r.shape)}: corr {corr:.8f}  "
              f"max|diff| {(r - m).abs().max():.2e}")

    if "--convert" in sys.argv:
        import os

        import litert_torch
        out = "klein_double_probe.tflite"
        litert_torch.convert(
            mine_mod, (hidden, encoder, cos, sin, mod_img, mod_txt)).export(out)
        print(f"[convert] {out}  {os.path.getsize(out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
