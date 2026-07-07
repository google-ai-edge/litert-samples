# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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

"""Convert timm SigLIP2 (ViT-B/16, 224) image tower to a GPU-clean, GPU-correct
LiteRT .tflite for the ML Drift GPU delegate.

SigLIP2 (Google 2025, Apache-2.0) is a SOTA CLIP-style image tower. timm exposes
it as `vit_base_patch16_siglip_224.v2_webli` (93M params, conv patch-embed, NO
rope, NO cls token, attention-pool head). The text tower for zero-shot is
open_clip `ViT-B-16-SigLIP2` (same 768-d space, prompt "a photo of a {label}").

Re-authoring (all verbatim, weights copied, corr ~1.0 vs PyTorch) -- the same set
proven on PE-Core, minus the rope step (SigLIP2 has no rope):
  * Attention (x12): fused qkv -> 5D head-split (the GPU "C12" wall). Decompose to
    separate q/k/v, hand the 4D q/k/v to scaled_dot_product_attention (its lowering
    keeps the batch-matmul 3D with a materialized transpose -> GPU-resident).
  * AttentionPoolLatent: single constant-latent query -> a batch-matmul there is
    const@non-const (rejected / mis-computed). Express as broadcast-multiply +
    reduce-sum (exact for latent_len=1, GPU-correct).
  * LayerNorm -> overflow-safe LayerNorm: the delegate reduces the variance in fp16
    even for an fp32 graph; deep-ViT massive activations overflow fp16 (sum > 65504)
    -> wrong norm that compounds with depth while still reporting full residency.
    Scale-before-square keeps the sum in range.

I/O: input [1,3,224,224] NCHW float32 normalized to [-1,1] ((x/255-0.5)/0.5),
output [1,768] L2-normalized image embedding.
Reproduces litert-community/SigLIP2-base-patch16-224/siglip2_base_224_fp16.tflite.

    pip install litert-torch ai-edge-quantizer torch timm
    python convert_siglip2.py [out_dir]
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

MODEL = "vit_base_patch16_siglip_224.v2_webli"
IMG = 224
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT_DIR, exist_ok=True)
FP32 = os.path.join(OUT_DIR, "siglip2_base_224.tflite")
FP16 = os.path.join(OUT_DIR, "siglip2_base_224_fp16.tflite")

BANNED = {"GATHER_ND", "GATHER", "TOPK_V2", "FLEX_ERF", "ERF", "BROADCAST_TO"}


# -------------------------------------------------- overflow-safe LayerNorm
class SafeLayerNorm(nn.Module):
    """LayerNorm whose variance reduction can't overflow fp16 (see module docstring)."""
    SC = 0.03125  # 1/32

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


# ------------------------------------------ block Attention -> 4D (no rope)
def _attn_forward(self, x, *args, **kwargs):
    B, N, C = x.shape
    H, d = self.num_heads, self.head_dim
    q = self.q_proj_d(x).reshape(B, N, H, d).transpose(1, 2)
    k = self.k_proj_d(x).reshape(B, N, H, d).transpose(1, 2)
    v = self.v_proj_d(x).reshape(B, N, H, d).transpose(1, 2)
    q, k = self.q_norm(q), self.k_norm(k)  # Identity for SigLIP2
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(B, N, H * d)
    out = self.norm(out)  # Identity (scale_norm off)
    return self.proj(out)


def reauthor_attn(attn):
    C = attn.qkv.in_features
    w = attn.qkv.weight.data
    b = attn.qkv.bias.data if attn.qkv.bias is not None else None
    has_b = b is not None
    q_proj = nn.Linear(C, C, bias=has_b)
    k_proj = nn.Linear(C, C, bias=has_b)
    v_proj = nn.Linear(C, C, bias=has_b)
    with torch.no_grad():
        q_proj.weight.copy_(w[:C])
        k_proj.weight.copy_(w[C:2 * C])
        v_proj.weight.copy_(w[2 * C:])
        if has_b:
            q_proj.bias.copy_(b[:C])
            k_proj.bias.copy_(b[C:2 * C])
            v_proj.bias.copy_(b[2 * C:])
    attn.q_proj_d, attn.k_proj_d, attn.v_proj_d = q_proj, k_proj, v_proj
    attn.forward = types.MethodType(_attn_forward, attn)


# ------------------------------------------ AttentionPoolLatent -> broadcast-reduce
def _attn_pool_forward(self, x, attn_mask=None):
    B, N, C = x.shape
    H, d, L = self.num_heads, self.head_dim, self.latent_len
    k = self.k_norm(self.k_proj_d(x).reshape(B, N, H, d).transpose(1, 2))  # [B,H,N,d]
    v = self.v_proj_d(x).reshape(B, N, H, d).transpose(1, 2)               # [B,H,N,d]
    qc = self.q_const  # [H, L, d] constant, q_norm'd + scaled
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
        ql = ap.q(ap.latent.expand(1, -1, -1)).reshape(1, L, H, d).transpose(1, 2)
        ql = ap.q_norm(ql) * ap.scale
    ap.k_proj_d, ap.v_proj_d = k_proj, v_proj
    ap.register_buffer("q_const", ql.reshape(H, L, d).detach())
    ap.forward = types.MethodType(_attn_pool_forward, ap)


# ------------------------------------------------------------------- wrapper
class SigLIP2ImageEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel):
        m = self.m
        x = m.patch_embed(pixel)
        if x.dim() == 4:
            x = x.flatten(1, 2)
        if m.pos_embed is not None:
            x = x + m.pos_embed       # SigLIP2 has no cls token
        x = m.norm_pre(x)
        for blk in m.blocks:
            x = blk(x)
        x = m.norm(x)
        x = m.attn_pool(x)            # -> [B, 768]
        return F.normalize(x, dim=-1)


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
    m = timm.create_model(MODEL, pretrained=True, num_classes=0).eval()

    x = torch.randn(1, 3, IMG, IMG)
    with torch.no_grad():
        ref = F.normalize(m(x), dim=-1).numpy().flatten()  # original trunk embedding

    for blk in m.blocks:
        reauthor_attn(blk.attn)
    reauthor_attn_pool(m.attn_pool)
    patch_layernorm(m)
    enc = SigLIP2ImageEncoder(m).eval()

    with torch.no_grad():
        got = enc(x).numpy().flatten()
    corr = float(np.corrcoef(ref, got)[0, 1])
    print(f"EAGER parity (orig vs re-authored): corr {corr:.8f}  max|diff| {np.abs(ref-got).max():.3e}")
    assert corr > 0.9999, "re-authoring changed the math -- fix before convert"

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
