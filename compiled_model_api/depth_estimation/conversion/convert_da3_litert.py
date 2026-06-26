#!/usr/bin/env python3
"""Convert Depth Anything 3 (Small) monocular depth to a GPU-clean LiteRT .tflite.

Reproduces litert-community/Depth-Anything-3-Small/da3_small_gpu_fp16.tflite. DA3 is
NOT GPU-clean out of the box; this applies exact, weights-verbatim rewrites so the
DINOv2-RoPE backbone + DPT head ride the ML Drift GPU delegate:

  * checkpoint key-prefix fix (strip the leading "model.")
  * C14  RoPE data-dependent int max_position -> constant table
  * C12  fused-qkv 5D head-split -> separate q/k/v Linears, manual 4D attention
  * LayerScale folded into the preceding Linear (else the token dim is mis-laid-out
    on GPU: fully_connected {1,1,N,C} vs {N,1,1,C} -> compile fails)
  * pos_embed bicubic-resized and baked to a constant (interpolating a constant
    emits a runtime-less RESIZE_BILINEAR the delegate rejects)
  * ConvTranspose2d -> zero-stuff nearest-upsample + Conv2d (exact ~1e-7; Pixel 8a
    rejects TRANSPOSE_CONV)
  * camera-token in-place index-assign -> torch.cat (else SELECT_V2 with a broadcast
    'else' shape the delegate rejects)
  * DPT head resize align_corners=True -> False (banned) + drop the UV sincos
    pos-embed-again (BROADCAST_TO)

Measures depth corr vs the ORIGINAL (all-stock) model, op-checks the graph, then
FP16-quantizes. Needs the upstream DA3 source on the path (clone
github.com/ByteDance-Seed/Depth-Anything-3 and run this from its root so `src/` and
`depth_anything_3` import).

    pip install litert-torch ai-edge-quantizer torch timm safetensors huggingface_hub pillow
    python convert_da3_litert.py [image] [H] [W]   # H,W default to the shipped 896x504
"""
import os, sys, types, math, numpy as np, json, collections
class _Dummy:
    def __getattr__(self, n): return lambda *a, **k: None
_pp = types.ModuleType('scipy.sparse.linalg._propack')
for nm in ('_spropack', '_dpropack', '_cpropack', '_zpropack'): setattr(_pp, nm, _Dummy())
sys.modules['scipy.sparse.linalg._propack'] = _pp
for _n, _t in (("bool", bool), ("float", float), ("int", int), ("object", object), ("str", str)):
    if not hasattr(np, _n): setattr(np, _n, _t)
sys.path.insert(0, "src")
import torch, torch.nn as nn, torch.nn.functional as F, types as _ty
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from depth_anything_3.cfg import create_object
from depth_anything_3.model.dinov2.layers.attention import Attention
import depth_anything_3.model.dinov2.layers.rope as _rope

# C14: RoPE max_position = int(positions.max())+1 is data-dependent (torch.export aborts). Fix = constant
# 128 (>= the 33 a 518 grid needs); extra table rows are unused -> numerically identical to stock.
def _rope_fwd(self, tokens, positions):
    fd = tokens.size(-1) // 2
    c, s = self._compute_frequency_components(fd, 128, tokens.device, tokens.dtype)
    v, h = tokens.chunk(2, dim=-1)
    v = self._apply_1d_rope(v, positions[..., 0], c, s)
    h = self._apply_1d_rope(h, positions[..., 1], c, s)
    return torch.cat((v, h), dim=-1)
_rope.RotaryPositionEmbedding2D.forward = _rope_fwd

cfg = json.load(open(hf_hub_download("depth-anything/DA3-SMALL", "config.json")))
net = create_object(cfg["config"]).eval()
sd = load_file(hf_hub_download("depth-anything/DA3-SMALL", "model.safetensors"))
net.load_state_dict({(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}, strict=False)

class Mono(nn.Module):
    def __init__(s, n): super().__init__(); s.b, s.h = n.backbone, n.head
    def forward(s, x):
        H, W = x.shape[-2], x.shape[-1]
        f, _ = s.b(x.unsqueeze(1), cam_token=None)
        return s.h(f, H, W, patch_start_idx=0)["depth"]
m = Mono(net).eval()

# input H,W (multiples of 14). argv[2]=H, argv[3]=W. Default = the shipped 896x504
# (portrait native aspect; DA3 runs at the image's native aspect, no padding -- a
# square letterbox drops fidelity to corr ~0.977 as padding leaks through global attn).
H_IN = int(sys.argv[2]) if len(sys.argv) > 2 else 896
W_IN = int(sys.argv[3]) if len(sys.argv) > 3 else 504
S = H_IN  # back-compat alias used below for the square-ish paths
# argv[1] image gives a meaningful depth-corr check; without one a random tensor
# still validates eager parity + the GPU-clean op-check.
if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    img = Image.open(sys.argv[1]).convert("RGB").resize((W_IN, H_IN), Image.BILINEAR)
    a = (np.asarray(img, np.float32)/255.0 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    x_img = torch.from_numpy(np.transpose(a, (2,0,1))[None].astype(np.float32))
else:
    print("no image given -> random input (op-check + eager parity only; depth corr not meaningful)")
    x_img = torch.randn(1, 3, H_IN, W_IN)
with torch.no_grad():
    d_orig = m(x_img)[0,0].numpy()                 # ORIGINAL (all stock)

# ---- C12: qkv-decompose -> 4D manual attention ----
def _attn(self, x, pos=None, attn_mask=None):
    B,N,C = x.shape; H=self.num_heads; Hd=C//H
    q=self.q_lin(x).reshape(B,N,H,Hd).permute(0,2,1,3); k=self.k_lin(x).reshape(B,N,H,Hd).permute(0,2,1,3)
    v=self.v_lin(x).reshape(B,N,H,Hd).permute(0,2,1,3); q,k=self.q_norm(q),self.k_norm(k)
    if self.rope is not None and pos is not None: q=self.rope(q,pos); k=self.rope(k,pos)
    q=q*self.scale; attn=(q@k.transpose(-2,-1)).softmax(-1)
    return self.proj_drop(self.proj((attn@v).transpose(1,2).reshape(B,N,C)))
Attention.forward = _attn
for mod in net.modules():
    if isinstance(mod, Attention):
        C=mod.qkv.in_features; w=mod.qkv.weight; b=mod.qkv.bias
        for nm,sl in (("q_lin",slice(0,C)),("k_lin",slice(C,2*C)),("v_lin",slice(2*C,3*C))):
            lin=nn.Linear(C,C,bias=b is not None)
            with torch.no_grad():
                lin.weight.copy_(w[sl]); b is not None and lin.bias.copy_(b[sl])
            setattr(mod,nm,lin)

# ---- LayerScale bake (GPU FC-layout fix): fold ls1/ls2 gamma into attn.proj / mlp.fc2, ls->Identity.
# The LayerScale MUL (FC output [N,C] * gamma [C]) makes ML Drift mis-lay-out the token dim
# ({1,1,1025,384} vs {1025,1,1,384}) -> GPU compile fails. Baking eliminates the MUL. (MoGe's fix.)
def bake_layerscale(model):
    cnt = 0
    for block in model.modules():
        if hasattr(block, "ls1") and hasattr(getattr(block, "ls1"), "gamma") and hasattr(block, "attn"):
            g = block.ls1.gamma.data.squeeze()
            with torch.no_grad():
                block.attn.proj.weight.data.mul_(g.unsqueeze(1))
                if block.attn.proj.bias is not None: block.attn.proj.bias.data.mul_(g)
            block.ls1 = nn.Identity(); cnt += 1
        if hasattr(block, "ls2") and hasattr(getattr(block, "ls2"), "gamma") and hasattr(block, "mlp"):
            g = block.ls2.gamma.data.squeeze()
            last = None
            for ch in reversed(list(block.mlp.children())):
                if isinstance(ch, nn.Linear): last = ch; break
            if last is None and hasattr(block.mlp, "fc2"): last = block.mlp.fc2
            if last is not None:
                with torch.no_grad():
                    last.weight.data.mul_(g.unsqueeze(1))
                    if last.bias is not None: last.bias.data.mul_(g)
            block.ls2 = nn.Identity(); cnt += 1
    print(f"baked {cnt} LayerScale into Linear")
    return cnt
bake_layerscale(net)

# ---- C15: pos_embed BAKE (interpolating the constant pos_embed emits RESIZE_BILINEAR with 0 runtime
# inputs -> GPU rejects). Precompute the bilinear-resized pos_embed as a constant buffer -> no resize op. ----
GH, GW = H_IN // 14, W_IN // 14
for mod in net.modules():
    if hasattr(mod, "interpolate_pos_encoding") and hasattr(mod, "pos_embed"):
        with torch.no_grad():
            N = mod.pos_embed.shape[1] - 1; Mb = int(math.sqrt(N)); dim = mod.pos_embed.shape[-1]
            pe = mod.pos_embed.float(); cls = pe[:, 0]; patch = pe[:, 1:]
            patch = F.interpolate(patch.reshape(1, Mb, Mb, dim).permute(0, 3, 1, 2),
                                  size=(GH, GW), mode="bicubic", antialias=False)  # bicubic = match official (baked const)
            patch = patch.permute(0, 2, 3, 1).view(1, -1, dim)
            baked = torch.cat((cls.unsqueeze(0), patch), dim=1).to(mod.pos_embed.dtype)  # [1,1025,dim]
        mod.register_buffer("_baked_pos", baked)
        mod.interpolate_pos_encoding = _ty.MethodType(lambda self, x, w, h: self._baked_pos, mod)

# ---- SELECT_V2 fix: `x[:, :, 0] = cam_token` (in-place index assign) lowers to SELECT_V2 with a
# broadcast 'else' shape the GPU delegate rejects. Replace with an equivalent torch.cat (exact, GPU-clean).
# (alt_start must stay on — the camera-token / global-attn path DOES affect mono depth.) ----
import depth_anything_3.model.dinov2.vision_transformer as _vt
def _patched_gil(self, x, n=1, export_feat_layers=[], **kwargs):
    B, S, _, H, W = x.shape
    x = self.prepare_tokens_with_masks(x)
    output, total_block_len, aux_output = [], len(self.blocks), []
    blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)
    local_x = x
    for i, blk in enumerate(self.blocks):
        if i < self.rope_start or self.rope is None:
            g_pos, l_pos = None, None
        else:
            g_pos, l_pos = pos_nodiff, pos
        if self.alt_start != -1 and (i == self.alt_start - 1) and x.shape[1] >= _vt.THRESH_FOR_REF_SELECTION and kwargs.get("cam_token", None) is None:
            b_idx = _vt.select_reference_view(x, strategy=kwargs.get("ref_view_strategy", "saddle_balanced"))
            x = _vt.reorder_by_reference(x, b_idx); local_x = _vt.reorder_by_reference(local_x, b_idx)
        if self.alt_start != -1 and i == self.alt_start:
            if kwargs.get("cam_token", None) is not None:
                cam_token = kwargs.get("cam_token")
            else:
                ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                cam_token = torch.cat([ref_token, src_token], dim=1)
            x = torch.cat([cam_token.unsqueeze(2), x[:, :, 1:]], dim=2)   # was: x[:, :, 0] = cam_token
        if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
            x = self.process_attention(x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None))
        else:
            x = self.process_attention(x, blk, "local", pos=l_pos); local_x = x
        if i in blocks_to_take:
            out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
            if x.shape[1] >= _vt.THRESH_FOR_REF_SELECTION and self.alt_start != -1 and "b_idx" in locals():
                out_x = _vt.restore_original_order(out_x, b_idx)
            output.append((out_x[:, :, 0], out_x))
        if i in export_feat_layers:
            aux_output.append(x)
    return output, aux_output
_vt.DinoVisionTransformer._get_intermediate_layers_not_chunked = _patched_gil

# ---- TRANSPOSE_CONV (Pixel 8a reject): ConvTranspose2d(k=s,stride=s) -> bilinear-resize + 1x1 conv ----
# 1x1 conv weight = mean of the transposed kernel over its s*s positions (preserves channel mixing;
# spatial upsampling handled by bilinear). 1x1 conv commutes with bilinear so order is exact for the mix.
# EXACT GPU-clean equivalent: ConvTranspose2d(k=s,stride=s) == zero-stuff (nearest-upsample × top-left
# mask) + Conv2d(flipped weight). Matches the learned upsampler to ~1e-7 → depth stays as sharp as the
# original (a bilinear approx blurred it). Mask is a precomputed constant buffer (no index-assign/broadcast).
class ZeroStuffConvT(nn.Module):
    def __init__(self, ct, H, W):
        super().__init__(); self.s = ct.stride[0]; self.k = ct.kernel_size[0]
        self.register_buffer("w", ct.weight.flip(2, 3).transpose(0, 1).contiguous())
        self.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None else torch.zeros(ct.out_channels))
        s = self.s; mk = np.zeros((H*s, W*s), np.float32); mk[::s, ::s] = 1.0
        self.register_buffer("mask", torch.from_numpy(mk)[None, None])
    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]; s, k = self.s, self.k
        xn = F.interpolate(x, size=(H*s, W*s), mode="nearest")
        y = F.conv2d(xn * self.mask, self.w, bias=self.b, padding=k-1)
        return y[:, :, :H*s, :W*s]
# discover each ConvTranspose input size via a dry run, then swap with the exact equivalent
_ct_hw, _hooks = {}, []
for _nm, _mod in net.named_modules():
    if isinstance(_mod, nn.ConvTranspose2d):
        def _mk(nm):
            def _h(m, inp, out): _ct_hw[nm] = (inp[0].shape[-2], inp[0].shape[-1])
            return _h
        _hooks.append(_mod.register_forward_hook(_mk(_nm)))
with torch.no_grad(): m(x_img)
for _hk in _hooks: _hk.remove()
def swap_ct(module, prefix=""):
    for name, ch in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(ch, nn.ConvTranspose2d):
            H, W = _ct_hw[full]; setattr(module, name, ZeroStuffConvT(ch, H, W))
        else: swap_ct(ch, full)
swap_ct(net)

# ---- DPT head: align_corners=True -> False (banned RESIZE_BILINEAR) + drop _add_pos_embed expand (BROADCAST_TO) ----
import depth_anything_3.model.utils.head_utils as _hu
import depth_anything_3.model.dualdpt as _dd
import depth_anything_3.model.dpt as _dpt
_orig_ci = _hu.custom_interpolate
def _ci_no_ac(x, size=None, scale_factor=None, mode="bilinear", align_corners=True):
    return _orig_ci(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=False)
_hu.custom_interpolate = _ci_no_ac; _dd.custom_interpolate = _ci_no_ac; _dpt.custom_interpolate = _ci_no_ac
# head pos-embed-again (UV sincos, ratio 0.1): make_sincos broadcast emits BROADCAST_TO. Baking it as
# constants matches official ~0.0002 better but adds ~64 MB (full [1,C,H,W] per shape) — not worth it.
# Disable it (the ratio-0.1 UV refinement is negligible vs the size cost).
_n_pe = 0
for mod in net.modules():
    if isinstance(mod, _dd.DualDPT) and getattr(mod, "pos_embed", False):
        mod.pos_embed = False; _n_pe += 1
print(f"disabled head pos_embed on {_n_pe} DualDPT")

with torch.no_grad():
    d_clean = m(x_img)[0,0].numpy()                # FULLY GPU-CLEAN
corr = np.corrcoef(d_orig.flatten(), d_clean.flatten())[0,1]
print(f"depth corr (original vs full GPU-clean) = {corr:.6f}  mean-rel-diff = {np.abs(d_orig-d_clean).mean()/(np.abs(d_orig).mean()+1e-9)*100:.3f}%")

# ---- convert + op-check (canonical banned list; SELECT_V2 is NOT banned) ----
dummy = torch.rand(1, 3, H_IN, W_IN)
import litert_torch
litert_torch.convert(m.eval(), (dummy,)).export("da3_small_gpu.tflite")
from ai_edge_litert.interpreter import Interpreter
BANNED={'GATHER_ND','GATHER','TOPK_V2','PACK','SPLIT','FLEX_ERF','ERF','TRANSPOSE_CONV','BROADCAST_TO'}
it=Interpreter(model_path="da3_small_gpu.tflite"); it.allocate_tensors()
ops=collections.Counter(d.get('op_name','?') for d in it._get_ops_details())
bad={k:v for k,v in ops.items() if k in BANNED}
over=sum(1 for d in it.get_tensor_details() if len(d.get('shape',[]))>4)
print(f"op-check FP32: banned {bad or 'NONE'} | >4D {over} | GELU {ops.get('GELU',0)} | SELECT_V2 {ops.get('SELECT_V2',0)}")
print("VERDICT:", "GPU-CLEAN" if not bad and not over else "BLOCKERS REMAIN")
print("FP32 size %.1f MB" % (os.path.getsize("da3_small_gpu.tflite")/1e6))

# ---- FP16 (AI Edge Quantizer FLOAT_CASTING): half size, native on the GPU delegate,
# FP32 == FP16 here. This is the da3_small_gpu_fp16.tflite the app downloads. ----
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
FP16 = "da3_small_gpu_fp16.tflite"
if os.path.exists(FP16): os.remove(FP16)
qt = quantizer.Quantizer(float_model="da3_small_gpu.tflite")
qt.load_quantization_recipe(rm.get_quantization_recipe())
qt.quantize().export_model(FP16)
print("FP16 size %.1f MB -> %s" % (os.path.getsize(FP16)/1e6, FP16))
