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

"""GFPGAN v1.4 (blind face restoration, Apache-2.0) -> LiteRT CompiledModel GPU.

Crux: StyleGAN2 ModulatedConv2d uses a 5D weight (b,c_out,c_in,k,k) built at runtime from the
style vector + a grouped conv with that runtime weight -> BOTH banned on GPU (>4D tensor AND
conv filter must be a constant). Rewrite it to a mathematically EXACT 4D form (b=1 inference):

    modulate : conv(x, W*style) == conv(x * style_per_in_channel, W_const)   (conv is linear)
    demod    : rsqrt( sum_{i,k}(W[o,i,k]*s[i])^2 + eps )
             == rsqrt( (s^2) @ Wsq^T + eps ),   Wsq[o,i] = sum_k W[o,i,k]^2   (const matrix)

so the runtime filter disappears: a plain CONV_2D with a constant filter, an input channel-scale,
and a small (c_out x c_in) matmul + rsqrt for the per-out-channel demod. All tensors <=4D.
Everything else in the clean arch is already GPU-friendly (upsample = bilinear align_corners=False,
no GroupNorm, no ConvTranspose -> no ZeroStuff). noise -> stored buffers (randomize_noise=False)
so the graph is deterministic. Run: python build_gfpgan.py
"""
import sys
import os
import types
import importlib.util
import collections
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
GARCH = os.path.join(HERE, "GFPGAN", "gfpgan", "archs")


def _stub_basicsr():
    """The clean arch only needs basicsr for a registry decorator + weight-init (we load pretrained
    weights), so stub it instead of installing basicsr (breaks on modern torchvision)."""
    if "basicsr" in sys.modules:
        return

    class _Reg:
        def register(self, *a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]
            return lambda c: c

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        [setattr(m, k, v) for k, v in attrs.items()]
        sys.modules[name] = m
        return m

    _mod("basicsr")
    _mod("basicsr.utils")
    _mod("basicsr.utils.registry", ARCH_REGISTRY=_Reg())
    _mod("basicsr.archs")
    _mod("basicsr.archs.arch_util", default_init_weights=lambda *a, **k: None)


_stub_basicsr()


def _load_arch():
    """Load the two clean-arch files directly, bypassing gfpgan/__init__ (its scandir auto-import
    pulls in the custom-CUDA-op archs). Register a fake gfpgan.archs package so the relative import
    `from .stylegan2_clean_arch import ...` resolves."""
    for name, path in [("gfpgan", os.path.join(HERE, "GFPGAN", "gfpgan")), ("gfpgan.archs", GARCH)]:
        if name not in sys.modules:
            p = types.ModuleType(name)
            p.__path__ = [path]
            sys.modules[name] = p

    def _load(name, fn):
        spec = importlib.util.spec_from_file_location(name, os.path.join(GARCH, fn))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("gfpgan.archs.stylegan2_clean_arch", "stylegan2_clean_arch.py")
    gf = _load("gfpgan.archs.gfpganv1_clean_arch", "gfpganv1_clean_arch.py")
    return gf.GFPGANv1Clean, sys.modules["gfpgan.archs.stylegan2_clean_arch"].ModulatedConv2d


GFPGANv1Clean, ModulatedConv2d = _load_arch()
RES = 512
WEIGHTS = os.path.join(HERE, "gfpgan_models", "GFPGANv1.4.pth")
WURL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP", "RFFT2D", "FFT",
          "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM", "MIRROR_PAD"}


def mod_forward_4d(self, x, style):
    """GPU-friendly ModulatedConv2d.forward (b=1): no 5D, no runtime conv weight, and fp16-SAFE.

    The style vectors reach |s|~1000, so s^2 (and the demod sum Σ s^2·Wsq, which hits ~2.3e6)
    overflow fp16 on Mali -> rsqrt(inf)=0 -> the whole decoder collapses. Fix: divide the style by
    its per-image max (smax) first. StyleGAN2's demod is scale-covariant, so the smax cancels
    EXACTLY between the input modulation and the demod (demod_new = demod·smax); every intermediate
    stays in fp16 range. Mathematically identical (corr 1.0), numerically stable."""
    b, c, h, w = x.shape
    s = self.modulation(style)                       # (b, c_in), may reach |s|~1000
    if self.demodulate:
        # normalize style by its per-image max; smax cancels EXACTLY against the demod below
        # (max_pool because litert-torch lowers amax poorly)
        sa = s.abs()
        smax = F.max_pool2d(sa.view(b, 1, sa.shape[1], 1), kernel_size=(sa.shape[1], 1)).view(b, 1).clamp_min(1e-8)
        mods = s / smax                              # |mods| <= 1
    else:
        mods = s                                     # ToRGB (no demod): keep the raw style, no cancellation
    xs = x * mods.view(b, c, 1, 1)                    # modulation baked into the input
    if self.sample_mode == "upsample":
        xs = F.interpolate(xs, scale_factor=2, mode="bilinear", align_corners=False)
    elif self.sample_mode == "downsample":
        xs = F.interpolate(xs, scale_factor=0.5, mode="bilinear", align_corners=False)
    out = F.conv2d(xs, self.Wc, padding=self.padding)   # constant filter, b=1 -> plain conv, in range
    if self.demodulate:
        demod = torch.rsqrt((mods.pow(2) @ self.Wsq.t()) + self.eps)   # == original demod · smax, in range
        out = out * demod.view(b, self.out_channels, 1, 1)
    return out


def patch_modulated(model):
    n = 0
    for m in model.modules():
        if isinstance(m, ModulatedConv2d):
            wc = m.weight.detach().squeeze(0).contiguous()          # (c_out, c_in, k, k)
            m.register_buffer("Wsq", m.weight.detach().pow(2).sum(dim=[3, 4]).squeeze(0).contiguous())
            del m._parameters["weight"]                             # drop the dead 5D weight (else exported too)
            m.register_buffer("Wc", wc)
            n += 1
    ModulatedConv2d.forward = mod_forward_4d
    print(f"  patched {n} ModulatedConv2d -> 4D exact")


def build():
    if not os.path.exists(WEIGHTS):
        os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
        print("  downloading GFPGANv1.4.pth ...")
        urllib.request.urlretrieve(WURL, WEIGHTS)
    m = GFPGANv1Clean(out_size=RES, num_style_feat=512, channel_multiplier=2, decoder_load_path=None,
                      fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True,
                      narrow=1, sft_half=True).eval()
    sd = torch.load(WEIGHTS, map_location="cpu")
    sd = sd.get("params_ema", sd)
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"  loaded GFPGANv1.4; missing {len(miss)} unexpected {len(unexp)}; "
          f"params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    return m


class GFPGANWrap(nn.Module):
    """x in [-1,1] (1,3,512,512) -> restored image in [-1,1]. Deterministic (stored noise)."""
    def __init__(s, m):
        super().__init__()
        s.m = m
    def forward(s, x):
        img, _ = s.m(x, return_rgb=False, randomize_noise=False)
        return img


def opcheck(p, l):
    """Static GPU-compat scan: read the op set straight from the .tflite flatbuffer."""
    from ai_edge_litert import schema_py_generated as schema
    with open(p, "rb") as f:
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
    print(f"[{l}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB",
          "GPU-CLEAN" if not bad and not over else "BLOCKERS")


def run_tflite(p, x):
    """Single inference through the LiteRT CompiledModel API; returns the flat fp32 output."""
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(p)
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


if __name__ == "__main__":
    m = build()
    x = torch.rand(1, 3, RES, RES) * 2 - 1
    with torch.no_grad():
        ref = GFPGANWrap(m)(x)                     # original forward (5D modulated conv)
    print(f"forward: out {tuple(ref.shape)} range [{ref.min():.2f},{ref.max():.2f}]")
    patch_modulated(m)
    with torch.no_grad():
        ra = GFPGANWrap(m)(x)                       # patched 4D forward
    print(f"re-authored vs orig: corr {np.corrcoef(ref.numpy().ravel(), ra.numpy().ravel())[0,1]:.6f} "
          f"max|d| {(ref-ra).abs().max():.2e}")
    if (sys.argv[1] if len(sys.argv) > 1 else "all") == "forward":
        sys.exit()
    import litert_torch
    fp32 = os.path.join(HERE, "gfpgan.tflite")
    try:
        litert_torch.convert(GFPGANWrap(m).eval(), (x,)).export(fp32)
        opcheck(fp32, "gfpgan")
        o = run_tflite(fp32, x.numpy())
        print(f"tflite vs torch corr {np.corrcoef(o, ra.numpy().ravel())[0,1]:.6f}")
        to_fp16(fp32, os.path.join(HERE, "gfpgan_fp16.tflite"))
        opcheck(os.path.join(HERE, "gfpgan_fp16.tflite"), "gfpgan_fp16")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("CONVERT FAIL:", repr(e)[:200])
