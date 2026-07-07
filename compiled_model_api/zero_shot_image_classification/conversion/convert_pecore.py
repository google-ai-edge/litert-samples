# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert timm Perception Encoder (PE-Core, base/patch16/224) image tower to a
GPU-clean LiteRT .tflite for the ML Drift GPU delegate.

PE-Core (Meta 2025, Apache-2.0) is a CLIP-style ViT image tower. timm exposes it
as `vit_pe_core_base_patch16_224` (weights `timm/vit_pe_core_base_patch16_224.fb`).

Walls re-authored here (all numerically verbatim, weights copied):
  * AttentionRope (x12): fused qkv -> 5D reshape head-split = the "C12" GPU wall.
    Decompose to separate q/k/v Linears, manual 4D (B,H,N,d) attention.
  * RoPE: PE-Core uses the *interleaved* layout (rotate_half=False) whose `rot()`
    does strided `x[...,::2]` -> GATHER_ND (GPU-banned). Fix = the proven
    even->odd channel permutation baked into q/k weights + `rotate_half`
    (slice+neg+concat, 4D) + constant half-layout cos/sin (const-folds to MUL/ADD).
    Permuting q AND k identically preserves q.k exactly, so attention is unchanged.
  * AttentionPoolLatent: fused kv -> 5D head-split. Decompose kv to k/v Linears.

I/O: input [1,3,224,224] NCHW float32, output [1,1024] L2-normalized image embedding.
Reproduces litert-community/PE-Core-base-patch16-224/pe_core_base_224_fp16.tflite.

    pip install litert-torch ai-edge-quantizer torch timm
    python convert_pecore.py [out_dir]
"""
import os
import sys
import types
import collections

# macOS guard: scipy's _propack .so fails to dlopen on some macOS builds, but
# litert-torch only needs scipy's maximum_flow, so stub the unused _propack
# module. Harmless elsewhere (litert-torch never imports _propack); must run
# before litert_torch is imported.
_pp = types.ModuleType("scipy.sparse.linalg._propack")
for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_pp, _nm, type("_D", (), {"__getattr__": lambda self, n: (lambda *a, **k: None)})())
sys.modules.setdefault("scipy.sparse.linalg._propack", _pp)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

MODEL = "vit_pe_core_base_patch16_224"
IMG = 224
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT_DIR, exist_ok=True)
FP32 = os.path.join(OUT_DIR, "pe_core_base_224.tflite")
FP16 = os.path.join(OUT_DIR, "pe_core_base_224_fp16.tflite")

BANNED = {"GATHER_ND", "GATHER", "TOPK_V2", "FLEX_ERF", "ERF", "BROADCAST_TO"}


# -------------------------------------------------- overflow-safe LayerNorm
class SafeLayerNorm(nn.Module):
    """LayerNorm whose variance reduction can't overflow fp16. The ML Drift GPU
    delegate computes the sum-of-squares reduction in fp16 even for an fp32 model;
    deep-ViT massive activations (|x|~50+) make `sum((x-mean)^2)` exceed fp16 max
    (65504) -> wrong normalization that compounds with depth (corr collapses to
    ~0.28 over 12 blocks). Scaling by `SC` before squaring (and undoing after)
    keeps the running sum in range -- mathematically identical to nn.LayerNorm."""
    SC = 0.03125  # 1/32: keeps sum((x-mean)*SC)^2 << 65504 for |x|<~290

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.weight, self.bias, self.eps = ln.weight, ln.bias, ln.eps

    def forward(self, x):
        xc = x - x.mean(-1, keepdim=True)
        xs = xc * self.SC
        var = (xs * xs).mean(-1, keepdim=True) / (self.SC * self.SC)
        return xc * torch.rsqrt(var + self.eps) * self.weight + self.bias


def patch_layernorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, SafeLayerNorm(child))
        else:
            patch_layernorm(child)


# ---------------------------------------------------------------- rope (clean)
def rope_rotate_half(x):
    # 4D-clean: slice halves, negate, concat. No strided slice, no >4D.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_half(x, cos, sin):
    # x: [B,H,N,d]; cos/sin: [1,1,N,d]
    return x * cos + rope_rotate_half(x) * sin


def _even_odd_perm(num_heads, head_dim):
    """Per-head index permutation [0,2,..,1,3,..] that maps the interleaved RoPE
    layout to the rotate-half layout (evens then odds within each head)."""
    perm = []
    for h in range(num_heads):
        base = h * head_dim
        perm += [base + i for i in range(0, head_dim, 2)]
        perm += [base + i for i in range(1, head_dim, 2)]
    return torch.tensor(perm, dtype=torch.long)


# ----------------------------------------------- AttentionRope -> 4D + clean rope
def _attn_rope_forward(self, x, rope=None, attn_mask=None, is_causal=False):
    B, N, C = x.shape
    H, d = self.num_heads, self.head_dim
    q = self.q_proj_d(x).reshape(B, N, H, d).transpose(1, 2)
    k = self.k_proj_d(x).reshape(B, N, H, d).transpose(1, 2)
    v = self.v_proj_d(x).reshape(B, N, H, d).transpose(1, 2)
    q, k = self.q_norm(q), self.k_norm(k)  # Identity for PE-Core
    npt = self.npt_
    cos, sin = self.cos_half, self.sin_half
    q = torch.cat([q[:, :, :npt, :], apply_half(q[:, :, npt:, :], cos, sin)], dim=2)
    k = torch.cat([k[:, :, :npt, :], apply_half(k[:, :, npt:, :], cos, sin)], dim=2)
    # SDPA lowers to a 3D batch-matmul with a MATERIALIZED transpose (adj_y=False),
    # which the GPU delegate accepts -- unlike explicit q@k.transpose (folds to
    # adj_y=True, rejected for non-constant RHS). Default scale = head_dim**-0.5.
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(B, N, self.attn_dim)
    out = self.norm(out)  # Identity (scale_norm off)
    return self.proj(out)


def reauthor_attn_rope(attn, cos_half, sin_half, npt):
    C = attn.qkv.in_features
    H, d = attn.num_heads, attn.head_dim
    w = attn.qkv.weight.data
    b = attn.qkv.bias.data if attn.qkv.bias is not None else None
    wq, wk, wv = w[:C], w[C:2 * C], w[2 * C:]
    perm = _even_odd_perm(H, d)
    has_b = b is not None
    q_proj = nn.Linear(C, C, bias=has_b)
    k_proj = nn.Linear(C, C, bias=has_b)
    v_proj = nn.Linear(C, C, bias=has_b)
    with torch.no_grad():
        q_proj.weight.copy_(wq[perm])   # permute OUTPUT channels (rows)
        k_proj.weight.copy_(wk[perm])
        v_proj.weight.copy_(wv)
        if has_b:
            q_proj.bias.copy_(b[:C][perm])
            k_proj.bias.copy_(b[C:2 * C][perm])
            v_proj.bias.copy_(b[2 * C:])
    attn.q_proj_d, attn.k_proj_d, attn.v_proj_d = q_proj, k_proj, v_proj
    attn.register_buffer("cos_half", cos_half[None, None])  # [1,1,N,d]
    attn.register_buffer("sin_half", sin_half[None, None])
    attn.npt_ = npt
    attn.forward = types.MethodType(_attn_rope_forward, attn)


# ----------------------------------------------- AttentionPoolLatent -> 4D
def _attn_pool_forward(self, x, attn_mask=None):
    # The pooling query is derived from a constant latent (latent_len=1). Both a
    # const@non-const BMM (rejected at compile) AND the reordered const-RHS BMM
    # (compiles but the GPU delegate MIS-COMPUTES it -> garbage embedding) fail, so
    # express the single-query attention as broadcast-multiply + reduce-sum, which
    # is exact and GPU-correct.
    B, N, C = x.shape
    H, d, L = self.num_heads, self.head_dim, self.latent_len
    k = self.k_norm(self.k_proj_d(x).reshape(B, N, H, d).transpose(1, 2))  # [B,H,N,d]
    v = self.v_proj_d(x).reshape(B, N, H, d).transpose(1, 2)               # [B,H,N,d]
    qc = self.q_const  # [H, L, d] constant, q_norm'd + scaled
    # Broadcast-multiply + reduce (no batch-matmul): exact for latent_len=1 and
    # avoids the const@non-const BMM that the GPU delegate mis-computes.
    scores = (qc.unsqueeze(0) * k).sum(dim=-1)        # [B, H, N]
    attn = scores.softmax(dim=-1).unsqueeze(-1)       # [B, H, N, 1]
    out = (attn * v).sum(dim=2).reshape(B, L, C)      # [B, L, C]
    out = self.proj(out)
    if self.mlp is not None:
        out = out + self.mlp(self.norm(out))
    if self.pool == "token":
        out = out[:, 0]
    elif self.pool == "avg":
        out = out.mean(1)
    return out


def reauthor_attn_pool(ap):
    assert ap.pos_embed is None, "attn_pool pos_embed not handled"
    C = ap.kv.in_features
    inner = ap.num_heads * ap.head_dim
    has_b = ap.kv.bias is not None
    k_proj = nn.Linear(C, inner, bias=has_b)
    v_proj = nn.Linear(C, inner, bias=has_b)
    with torch.no_grad():
        k_proj.weight.copy_(ap.kv.weight.data[:inner])
        v_proj.weight.copy_(ap.kv.weight.data[inner:])
        if has_b:
            k_proj.bias.copy_(ap.kv.bias.data[:inner])
            v_proj.bias.copy_(ap.kv.bias.data[inner:])
        H, d, L = ap.num_heads, ap.head_dim, ap.latent_len
        # constant query: q_norm(q(latent)) * scale  -> [H, L, d]
        ql = ap.q(ap.latent.expand(1, -1, -1)).reshape(1, L, H, d).transpose(1, 2)
        ql = ap.q_norm(ql) * ap.scale
    ap.k_proj_d, ap.v_proj_d = k_proj, v_proj
    ap.register_buffer("q_const", ql.reshape(H, L, d).detach())
    ap.forward = types.MethodType(_attn_pool_forward, ap)


# ------------------------------------------------------------------- wrapper
class PECoreImageEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel):
        m = self.m
        x = m.patch_embed(pixel)
        if x.dim() == 4:  # [B,Hg,Wg,C] -> [B,N,C]
            x = x.flatten(1, 2)
        cls = m.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        if m.pos_embed is not None:
            x = x + m.pos_embed
        x = m.norm_pre(x)
        for blk in m.blocks:
            x = blk(x)  # rope=None default; patched attn uses baked buffers
        x = m.norm(x)
        x = m.attn_pool(x)
        x = m.head(x)
        return F.normalize(x, dim=-1)


def build_half_cos_sin(m):
    """Half-layout constant cos/sin [N_patch, head_dim] from timm's interleaved rope."""
    emb = m.rope.get_embed()            # [N, 2*d] = cat(sin, cos)
    sin_emb, cos_emb = emb.chunk(2, -1)  # each [N, d] interleaved [s0,s0,s1,s1,...]
    s = sin_emb[:, ::2]                  # [N, d/2] = [s0,s1,...]
    c = cos_emb[:, ::2]
    sin_half = torch.cat([s, s], dim=-1)  # [N, d]
    cos_half = torch.cat([c, c], dim=-1)
    return cos_half.detach(), sin_half.detach()


def op_hist(path):
    """Static GPU-compat scan: read the op set straight from the .tflite flatbuffer."""
    from ai_edge_litert import schema_py_generated as schema
    with open(path, "rb") as f:
        model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
    hist = collections.Counter()
    over4d = 0
    for g in model.subgraphs:
        for op in g.operators:
            c = model.operatorCodes[op.opcodeIndex]
            code = max(c.builtinCode, c.deprecatedBuiltinCode)
            hist[c.customCode.decode() if c.customCode else names.get(code, str(code))] += 1
        over4d += sum(1 for t in g.tensors if t.shape is not None and len(t.shape) > 4)
    return hist, over4d


def tflite_run(path, x_nchw):
    """Single inference through the LiteRT CompiledModel API; returns the flat output."""
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(path)
    key = list(model.get_signature_list())[0]
    shp = list(next(iter(model.get_input_tensor_details(key).values()))["shape"])
    x = x_nchw if shp[1] == 3 else np.transpose(x_nchw, (0, 2, 3, 1)).copy()
    ins = model.create_input_buffers(0)
    outs = model.create_output_buffers(0)
    ins[0].write(np.ascontiguousarray(x, dtype=np.float32))
    model.run_by_index(0, ins, outs)
    n = (model.get_output_buffer_requirements(0, 0)["buffer_size"]
         // np.dtype(np.float32).itemsize)
    return outs[0].read(n, np.float32).astype("float64")


def main():
    torch.manual_seed(0)
    print(f"loading {MODEL} (pretrained, apache-2.0) ...")
    m = timm.create_model(MODEL, pretrained=True).eval()

    x = torch.randn(1, 3, IMG, IMG)
    with torch.no_grad():
        ref = F.normalize(m(x), dim=-1).numpy().flatten()  # original (interleaved rope, fused qkv)

    # ---- re-author in place ----
    cos_half, sin_half = build_half_cos_sin(m)
    npt = m.blocks[0].attn.num_prefix_tokens
    for blk in m.blocks:
        reauthor_attn_rope(blk.attn, cos_half, sin_half, npt)
    reauthor_attn_pool(m.attn_pool)
    patch_layernorm(m)  # GPU fp16 variance reduction overflows on deep-ViT outliers
    enc = PECoreImageEncoder(m).eval()

    with torch.no_grad():
        got = enc(x).numpy().flatten()
    corr = float(np.corrcoef(ref, got)[0, 1])
    maxd = float(np.abs(ref - got).max())
    print(f"EAGER parity (orig vs re-authored): corr {corr:.8f}  max|diff| {maxd:.3e}")
    assert corr > 0.9999, "re-authoring changed the math -- fix before convert"

    # ---- convert fp32 ----
    print("converting (litert_torch) ...")
    import litert_torch
    litert_torch.convert(enc, (x,)).export(FP32)

    hist, over4d = op_hist(FP32)
    bad = {k: v for k, v in hist.items() if k in BANNED}
    print(f"FP32 ops: {dict(sorted(hist.items(), key=lambda kv: -kv[1]))}")
    print(f"banned: {bad or 'NONE'} | >4D tensors: {over4d}")
    o = tflite_run(FP32, x.numpy())
    print(f"PARITY tflite(fp32) vs torch: corr {np.corrcoef(ref, o)[0,1]:.6f}")
    assert not bad and over4d == 0, "GPU blockers remain -- inspect op histogram"

    # ---- fp16 FLOAT_CASTING ----
    print("quantizing fp16 (FLOAT_CASTING) ...")
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT,
        ),
        algorithm_key=AlgorithmName.FLOAT_CASTING,
    )
    if os.path.exists(FP16):
        os.remove(FP16)
    qt = quantizer.Quantizer(float_model=FP32)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(FP16)

    s32, s16 = os.path.getsize(FP32) / 1e6, os.path.getsize(FP16) / 1e6
    print(f"SIZE fp32 {s32:.1f} MB -> fp16 {s16:.1f} MB ({s16/s32*100:.0f}%)")
    h16, o16d = op_hist(FP16)
    bad16 = {k: v for k, v in h16.items() if k in BANNED}
    print(f"FP16 banned: {bad16 or 'NONE'} | >4D: {o16d}")
    o16 = tflite_run(FP16, x.numpy())
    print(f"PARITY tflite(fp16) vs torch: corr {np.corrcoef(ref, o16)[0,1]:.6f}  "
          f"fp16-vs-fp32 corr {np.corrcoef(o, o16)[0,1]:.6f}")
    print("\nDONE:", FP16)


if __name__ == "__main__":
    main()
