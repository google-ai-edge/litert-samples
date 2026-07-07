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

"""Metric3D v2 (ViT-Small) -> LiteRT CompiledModel GPU.

Metric3D v2 = DINOv2 ViT-S/14 (+register tokens) encoder + RAFT-DPT decoder (4 iters),
producing absolute metric depth in meters. Fixed-shape export at 448x448 (= 32x32 patches
= 1024 tokens, the MoGe-2 token count that is device-verified on Pixel 8a GPU).

Re-authoring (all numerically-equivalent unless noted):
  ENCODER (reuse the MoGe-2 DINOv2 ViT-S suite):
    - fused qkv 5D reshape    -> 3 separate q/k/v Linear + manual 4D attention
    - nn.GELU                 -> x*sigmoid(1.702x)
    - LayerScale gamma        -> baked into attn.proj / mlp last Linear (eliminates MUL shape clash)
    - bicubic interpolate_pos -> baked constant pos-embed for the fixed 32x32 grid
  DECODER (RAFTDepthNormalDPT5):
    - convex upsample (6/7-D) -> fully-4D per-subpixel softmax-9 combine + nearest-stuff interleave
    - ConvTranspose2d (k2 s2) -> ZeroStuffConvT2d (RESIZE_NEAREST + MUL + CONV_2D; Mali rejects TRANSPOSE_CONV)
    - nn.GELU in DPT Readout  -> x*sigmoid(1.702x)
    - dynamic-scale interpolate / bicubic / align_corners=True -> fixed-size bilinear align_corners=False

Run: ~/clipconv/bin/python build_m3d.py [opcheck|parity|all]
"""
import load_m3d                       # stubs mmcv, patches inspect; provides load()
import sys
import os
import math
import types
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = 448                            # 32 patches * 14 ; divisible by 28 and by 4
PATCH = 14
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}


# ─── ZeroStuffConvT2d (shipped DAC/PP-OCR pattern) ───────────────────────────
class ZeroStuffConvT2d(nn.Module):
    """Exact GPU-clean ConvTranspose2d (k,s,p=0,op=0): nearest-upsample x stride zero-stuff
    mask + flipped conv2d + crop. RESIZE_NEAREST + MUL + CONV_2D (no TRANSPOSE_CONV)."""
    def __init__(s, ct, Hin, Win):
        super().__init__()
        s.s = ct.stride[0]
        s.k = ct.kernel_size[0]
        s.p = ct.padding[0]
        s.op = ct.output_padding[0]
        s.Hin = Hin
        s.Win = Win
        w = ct.weight.detach()                                     # (Cin, Cout, k, k)
        w = w.flip(2).flip(3).permute(1, 0, 2, 3).contiguous()     # (Cout, Cin, k, k)
        s.register_buffer("w", w)
        s.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None
                          else torch.zeros(ct.out_channels))
        mh = np.zeros((Hin * s.s, Win * s.s), np.float32)
        mh[::s.s, ::s.s] = 1.0
        s.register_buffer("mask", torch.from_numpy(mh)[None, None])

    def forward(s, x):
        xn = F.interpolate(x, size=(s.Hin * s.s, s.Win * s.s), mode="nearest") * s.mask
        y = F.conv2d(xn, s.w, bias=s.b, padding=s.k - 1)
        olH = (s.Hin - 1) * s.s + s.k - 2 * s.p + s.op
        olW = (s.Win - 1) * s.s + s.k - 2 * s.p + s.op
        return y[:, :, s.p:s.p + olH, s.p:s.p + olW]


# ─── interpolate patch: dynamic scale -> fixed size, bicubic->bilinear, ac=False ──
_orig_interp = F.interpolate

_KEEP_AC = os.environ.get("M3D_AC", "0") == "1"   # diagnostic: keep align_corners=True (not GPU-clean)

def _fixed_interp(input, size=None, scale_factor=None, mode="nearest",
                  align_corners=None, recompute_scale_factor=None, antialias=False):
    ac_in = align_corners
    if mode == "bicubic":
        mode = "bilinear"
    if scale_factor is not None and size is None:
        h, w = int(input.shape[-2]), int(input.shape[-1])
        sh, sw = (scale_factor if isinstance(scale_factor, (tuple, list)) else (scale_factor, scale_factor))
        size = (int(math.floor(h * sh)), int(math.floor(w * sw)))
        scale_factor = None
    if mode in ("bilinear", "trilinear"):
        align_corners = bool(ac_in) if _KEEP_AC else False   # GPU delegate rejects align_corners=True bilinear
    else:
        align_corners = None
    return _orig_interp(input, size=size, scale_factor=scale_factor, mode=mode,
                        align_corners=align_corners, recompute_scale_factor=None, antialias=False)


class _GELU(nn.Module):
    """GELU replacement; M3D_GELU=tanh uses the accurate tanh approx (POW-free), else sigmoid(1.702x)."""
    TANH = os.environ.get("M3D_GELU", "tanh") == "tanh"
    def forward(self, x):
        if _GELU.TANH:
            return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
        return x * torch.sigmoid(1.702 * x)


# ─── ENCODER patches (DINOv2 ViT-S, Metric3D's ViT_DINO_reg copy) ────────────
def patch_encoder(enc):
    import mono.model.backbones.ViT_DINO_reg as V

    # 1) GELU -> GPU-clean approx (Mlp act). Patch class + functional.
    n_gelu = 0
    for mod in enc.modules():
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.GELU):
                setattr(mod, cn, _GELU())
                n_gelu += 1

    # 2) decompose fused qkv -> q/k/v, manual 4D attention
    def decompose(attn):
        C = attn.qkv.in_features
        b = attn.qkv.bias
        attn.q_lin = nn.Linear(C, C, bias=b is not None)
        attn.k_lin = nn.Linear(C, C, bias=b is not None)
        attn.v_lin = nn.Linear(C, C, bias=b is not None)
        with torch.no_grad():
            w = attn.qkv.weight
            attn.q_lin.weight.copy_(w[:C])
            attn.k_lin.weight.copy_(w[C:2*C])
            attn.v_lin.weight.copy_(w[2*C:])
            if b is not None:
                attn.q_lin.bias.copy_(b[:C])
                attn.k_lin.bias.copy_(b[C:2*C])
                attn.v_lin.bias.copy_(b[2*C:])

    use_sdpa = os.environ.get("M3D_ATTN", "manual") == "sdpa"

    def attn_forward(self, x, attn_bias=None):
        B, N, C = x.shape
        h = self.num_heads
        hd = C // h
        q = self.q_lin(x).reshape(B, N, h, hd).permute(0, 2, 1, 3)
        k = self.k_lin(x).reshape(B, N, h, hd).permute(0, 2, 1, 3)
        v = self.v_lin(x).reshape(B, N, h, hd).permute(0, 2, 1, 3)
        if use_sdpa:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)

    n_attn = 0
    for mod in enc.modules():
        if isinstance(mod, V.Attention):
            decompose(mod)
            n_attn += 1
    V.Attention.forward = attn_forward
    V.MemEffAttention.forward = attn_forward   # ensure subclass uses 4D path too

    # 3) bake LayerScale gamma into the preceding Linear
    n_ls = 0
    for blk in enc.modules():
        if not (hasattr(blk, "ls1") and hasattr(blk, "ls2")):
            continue
        if isinstance(blk.ls1, V.LayerScale):
            g = blk.ls1.gamma.data.squeeze()
            with torch.no_grad():
                blk.attn.proj.weight.data.mul_(g.unsqueeze(1))
                if blk.attn.proj.bias is not None: blk.attn.proj.bias.data.mul_(g)
            blk.ls1 = nn.Identity()
            n_ls += 1
        if isinstance(blk.ls2, V.LayerScale):
            g = blk.ls2.gamma.data.squeeze()
            last = None
            for ch in reversed(list(blk.mlp.children())):
                if isinstance(ch, nn.Linear):
                    last = ch
                    break
            if last is not None:
                with torch.no_grad():
                    last.weight.data.mul_(g.unsqueeze(1))
                    if last.bias is not None: last.bias.data.mul_(g)
            blk.ls2 = nn.Identity()
            n_ls += 1

    # 4) bake the (otherwise-bicubic) interpolated pos-embed for the fixed grid
    hp = wp = INPUT // PATCH
    with torch.no_grad():
        dummy = torch.zeros(1, hp * wp + 1, enc.embed_dim)
        baked = enc.interpolate_pos_encoding(dummy, INPUT, INPUT).clone()   # [1, 1+hp*wp, C]
    enc.register_buffer("_baked_pos", baked)

    def prepare_tokens(self, x, masks=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self._baked_pos
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(B, -1, -1), x[:, 1:]), dim=1)
        return x

    type(enc).prepare_tokens_with_masks = prepare_tokens
    print(f"  encoder: GELU x{n_gelu}, qkv-decompose x{n_attn}, LayerScale-bake x{n_ls}, pos-embed baked {hp}x{wp}")


# ─── DECODER patches (RAFTDepthNormalDPT5) ───────────────────────────────────
def patch_decoder(dec):
    import mono.model.decode_heads.RAFTDepthNormalDPTDecoder5 as R

    # GELU in DPT Readout -> GPU-clean approx
    ng = 0
    for mod in dec.modules():
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.GELU):
                setattr(mod, cn, _GELU())
                ng += 1

    # kill the autocast-wrapped interpolate helpers -> plain fixed-size interpolate
    R.interpolate_float32 = lambda x, size=None, scale_factor=None, mode="nearest", align_corners=None: \
        _fixed_interp(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    R.interp = lambda x, dest: _fixed_interp(x, size=tuple(dest.shape[2:]), mode="bilinear", align_corners=True)

    # norm_normalize uses F.elu (-> SELECT). The initial normal feeds coords1 (active path),
    # so rewrite elu(k)+1+min_kappa as the exact SELECT-free identity exp(-relu(-k))+relu(k)+min_kappa.
    def norm_normalize_clean(norm_out):
        nx, ny, nz, kp = norm_out[:, 0:1], norm_out[:, 1:2], norm_out[:, 2:3], norm_out[:, 3:4]
        norm = torch.sqrt(nx * nx + ny * ny + nz * nz) + 1e-10
        kp = torch.exp(-F.relu(-kp)) + F.relu(kp) + 0.01
        return torch.cat([nx / norm, ny / norm, nz / norm, kp], dim=1)
    R.norm_normalize = norm_normalize_clean

    # ConvBlock's leading ReLU is on the residual input (no preceding conv to fuse into), so the
    # converter lowers it as where(x>0,x,0) -> SELECT. Make just that one a maximum (-> MAXIMUM op).
    # NOTE: the original uses nn.ReLU(inplace=True) on the FIRST act, which mutates x in place, so the
    # residual is relu(x)+convs(...), NOT x+convs(...). Replicate that exactly with xr as the base.
    def conv_block_forward(self, x):
        xr = torch.maximum(x, torch.zeros((), dtype=x.dtype))   # == in-place relu(x)
        out = self.conv1(xr)
        out = self.act(out)
        out = self.conv2(out)
        return xr + out
    R.ConvBlock.forward = conv_block_forward

    # ConvTranspose2d -> ZeroStuffConvT2d (detect input H,W by a probe forward)
    L = {}
    hks = []
    for n, mo in dec.named_modules():
        if isinstance(mo, nn.ConvTranspose2d):
            hks.append(mo.register_forward_pre_hook(
                (lambda nm: (lambda mod, i: L.__setitem__(nm, i[0].shape[-2:])))(n)))
    # run a probe through the full model below sets L; do it via the wrapper's first call.
    dec._zsct_pending = (L, hks)

    # convex upsample -> fully 4D
    factor = 2 ** dec.n_downsample             # 4
    fH = fW = INPUT // factor                  # flow resolution (112)
    Dn = 6                                      # flow channels (depth,conf,nx,ny,nz,kappa)
    # depth-to-space as a fixed ConvTranspose2d(D*r*r -> D, k=r, s=r), wrapped in ZeroStuffConvT2d.
    # ZeroStuff masks only stride-aligned positions (exact under any nearest convention) and the conv
    # kernel places each subpixel's in-block offset -> robust on the Mali delegate (unlike masking at
    # arbitrary in-block positions, which depends on RESIZE_NEAREST half-pixel semantics).
    d2s_ct = nn.ConvTranspose2d(Dn * factor * factor, Dn, factor, stride=factor, bias=False)
    with torch.no_grad():
        wt = torch.zeros(Dn * factor * factor, Dn, factor, factor)
        for i in range(factor):
            for j in range(factor):
                s = i * factor + j
                for d in range(Dn):
                    wt[s * Dn + d, d, i, j] = 1.0      # input ch (s,d) -> out ch d at in-block (i,j)
        d2s_ct.weight.copy_(wt)
    dec._d2s = ZeroStuffConvT2d(d2s_ct, fH, fW)

    def convex_upsample(self, flow, mask):
        N, D, H, W = flow.shape                # D=6, H=W=fH
        r = 2 ** self.n_downsample
        # 9 spatial neighbours via pad+slice (== F.unfold 3x3 tap order kh*3+kw)
        fp = F.pad(flow, (1, 1, 1, 1))
        nbr = [fp[:, :, di:di + H, dj:dj + W] for di in range(3) for dj in range(3)]  # 9 x [N,D,H,W]
        accs = []
        for s in range(r * r):
            # softmax over the 9 neighbours for this subpixel (mask ch = k*r*r + s)
            mk = torch.cat([mask[:, k * r * r + s: k * r * r + s + 1] for k in range(9)], dim=1)  # [N,9,H,W]
            msm = mk.softmax(dim=1)
            acc = msm[:, 0:1] * nbr[0]
            for k in range(1, 9):
                acc = acc + msm[:, k:k + 1] * nbr[k]               # [N,D,H,W]
            accs.append(acc)                                       # ordered s=i*r+j
        x = torch.cat(accs, dim=1)             # [N, D*r*r, H, W], channel = s*D + d
        return self._d2s(x)                    # [N, D, rH, rW]

    def bilinear_upsample(self, flow, mask):
        r = 2 ** self.n_downsample
        H, W = flow.shape[-2], flow.shape[-1]
        return _fixed_interp(flow, size=(H * r, W * r), mode="bilinear", align_corners=False)

    mode = os.environ.get("M3D_UPSAMPLE", "convex")
    dec.upsample_flow = types.MethodType(bilinear_upsample if mode == "bilinear" else convex_upsample, dec)
    print(f"  decoder: GELU x{ng}, upsample={mode} (factor {factor}, {fH}->{fH*factor}); ConvTranspose2d swap pending")


# ─── Static export wrapper ───────────────────────────────────────────────────
class M3DWrap(nn.Module):
    """Input [1,3,448,448] (canonical, ImageNet-normed inside encoder); output depth [1,1,448,448] meters."""
    def __init__(s, m):
        super().__init__()
        s.m = m
    def forward(s, img):
        return s.m({"input": img})[0]


def finalize_zsct(dec, wrap):
    """Run one probe forward to capture ConvTranspose2d input sizes, then swap to ZeroStuffConvT2d."""
    L, hks = dec._zsct_pending
    with torch.no_grad():
        wrap(torch.zeros(1, 3, INPUT, INPUT))
    for h in hks: h.remove()
    n = 0
    for name, mo in list(dec.named_modules()):
        if isinstance(mo, nn.ConvTranspose2d) and name in L:
            par = dec
            *pth, last = name.split(".")
            for q in pth: par = getattr(par, q)
            hh, ww = L[name]
            setattr(par, last, ZeroStuffConvT2d(mo, int(hh), int(ww)))
            n += 1
    del dec._zsct_pending
    print(f"  decoder: swapped {n} ConvTranspose2d -> ZeroStuffConvT2d")


def build():
    F.interpolate = _fixed_interp
    m = load_m3d.load()
    enc = m.depth_model.encoder
    dec = m.depth_model.decoder
    patch_encoder(enc)
    patch_decoder(dec)
    wrap = M3DWrap(m).eval()
    finalize_zsct(dec, wrap)
    return m, wrap


def opcheck(path, label):
    """Static GPU-compat scan: read the op set straight from the .tflite flatbuffer."""
    from ai_edge_litert import schema_py_generated as schema
    with open(path, "rb") as f:
        model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
    ops = collections.Counter()
    over = 0
    for g in model.subgraphs:
        for op in g.operators:
            c = model.operatorCodes[op.opcodeIndex]
            code = max(c.builtinCode, c.deprecatedBuiltinCode)
            ops[c.customCode.decode() if c.customCode else names.get(code, str(code))] += 1
        over += sum(1 for t in g.tensors if t.shape is not None and len(t.shape) > 4)
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(path)/1e6:.1f}MB",
          "GPU-CLEAN" if not bad and not over else "BLOCKERS")


def run_tflite(path, x):
    """Single inference through the LiteRT CompiledModel API; returns the flat fp32 output."""
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(path)
    inputs = model.create_input_buffers(0)
    outputs = model.create_output_buffers(0)
    inputs[0].write(np.ascontiguousarray(x, dtype=np.float32))
    model.run_by_index(0, inputs, outputs)
    n = model.get_output_buffer_requirements(0, 0)["buffer_size"] // np.dtype(np.float32).itemsize
    return outputs[0].read(n, np.float32)


def to_fp16(fp32, fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT), algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    q = quantizer.Quantizer(float_model=fp32)
    q.load_quantization_recipe(rm.get_quantization_recipe())
    q.quantize().export_model(fp16)
    return fp16


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    m, wrap = build()

    img = torch.randn(1, 3, INPUT, INPUT)
    with torch.no_grad():
        ref = wrap(img)
    print(f"re-authored forward: depth {tuple(ref.shape)} range [{ref.min():.3f},{ref.max():.3f}] m")

    if stage == "forward":
        return

    import litert_torch
    fp32 = os.path.join(HERE, "m3d.tflite")
    litert_torch.convert(wrap, (img,)).export(fp32)
    opcheck(fp32, "m3d")
    o = run_tflite(fp32, img.numpy()).reshape(tuple(ref.shape))
    corr = np.corrcoef(o.ravel(), ref.numpy().ravel())[0, 1]
    print(f"tflite vs torch: corr {corr:.6f} max|d| {np.abs(o - ref.numpy()).max():.3e}")

    if stage == "all":
        f16 = to_fp16(fp32, os.path.join(HERE, "m3d_fp16.tflite"))
        opcheck(f16, "m3d_fp16")


if __name__ == "__main__":
    main()
