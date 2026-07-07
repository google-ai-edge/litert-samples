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

"""Real-ESRGAN-General-x4v3 (SRVGGNetCompact) -> CompiledModel GPU. Pure CNN (both GPU gates).

Stock patch-free convert is NOT GPU-clean: PReLU->GREATER+SELECT+MUL (C6) and PixelShuffle->6D (C7).
Re-author:
  C6  PReLU       -> relu-form  relu(x) - a*relu(-x)         (exact, RELU/MUL/SUB, GPU-clean)
  C7  PixelShuffle-> sub-pixel == ConvTranspose(stride r) -> ZeroStuffConvT (TRANSPOSE_CONV-free, <=4D)
      [enabled with PS_FIX=1; default keeps nn.PixelShuffle to measure the >4D impact first]
Input 128x128 -> output 512x512 (x4). Tiny (~1.2M).

Run: ~/clipconv/bin/python convert_realesrgan.py [--nhwc]
"""
import sys
import os
import argparse
import collections
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class _D:
    def __getattr__(s, n):
        return lambda *a, **k: None
_p = types.ModuleType("scipy.sparse.linalg._propack")
for n in ("_spropack", "_dpropack", "_cpropack", "_zpropack"): setattr(_p, n, _D())
sys.modules["scipy.sparse.linalg._propack"] = _p

WEIGHTS = os.path.expanduser("~/Downloads/litert-upstream/realesrgan/realesr-general-x4v3.pth")
S = 128
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP", "PRELU"}
_PS_FIX = os.environ.get("PS_FIX") == "1"


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        return out + base


class PReLUClean(nn.Module):
    """PReLU(x) = relu(x) - a*relu(-x). Exact; RELU/MUL/SUB only (no GREATER/SELECT)."""
    def __init__(self, p):
        super().__init__()
        self.register_buffer("a", p.weight.detach().reshape(1, -1, 1, 1))
    def forward(self, x):
        return F.relu(x) - self.a * F.relu(-x)


class ZeroStuffConvT(nn.Module):
    """ConvTranspose2d(k=s,stride=s) == zero-stuff(nearest x top-left mask) + Conv2d(flipped w)."""
    def __init__(self, ct, H, W):
        super().__init__()
        self.s = ct.stride[0]
        self.k = ct.kernel_size[0]
        self.register_buffer("w", ct.weight.flip(2, 3).transpose(0, 1).contiguous())
        self.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None else torch.zeros(ct.out_channels))
        s = self.s
        mk = torch.zeros(H * s, W * s)
        mk[::s, ::s] = 1.0
        self.register_buffer("mask", mk[None, None])
    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        s, k = self.s, self.k
        xn = F.interpolate(x, size=(H * s, W * s), mode="nearest")
        return F.conv2d(xn * self.mask, self.w, bias=self.b, padding=k - 1)[:, :, :H * s, :W * s]


def subpixel_to_convT(conv, r):
    """Fold Conv(k=3,p=1) + PixelShuffle(r) into ConvTranspose2d(stride r) (Shi et al. sub-pixel==deconv),
    then ZeroStuffConvT. conv: Conv2d(F, C*r*r, 3, p=1). Returns ZeroStuffConvT producing [B,C,rH,rW]."""
    # Build an equivalent ConvTranspose2d(F, C, kernel=r, stride=r) is NOT exact for k=3 sub-pixel;
    # instead realize PixelShuffle exactly via ConvTranspose with a one-hot rearrange kernel.
    Fin = conv.in_channels
    Cout = conv.out_channels // (r * r)
    # PixelShuffle is a pure rearrange of conv's output -> express as ConvTranspose(Cr2->C, k=r, s=r)
    # with a fixed one-hot weight, applied to conv(x). w[oc, ic, i, j] = 1 iff ic == (oc? ...).
    ct = nn.ConvTranspose2d(conv.out_channels, Cout, kernel_size=r, stride=r, bias=False)
    w = torch.zeros(conv.out_channels, Cout, r, r)
    for c in range(Cout):
        for i in range(r):
            for j in range(r):
                w[c * r * r + i * r + j, c, i, j] = 1.0
    ct.weight.data.copy_(w)
    return ct


def reauthor(model):
    for i, layer in enumerate(model.body):
        if isinstance(layer, nn.PReLU):
            model.body[i] = PReLUClean(layer)
    if _PS_FIX:
        ct = subpixel_to_convT(model.body[-1], model.upscale)  # rearrange-only ConvT on conv output
        model.upsampler = ZeroStuffConvT(ct, S, S)             # input to upsampler is SxS (after final conv)
    return model


class Wrap(nn.Module):
    def __init__(self, m, nhwc=False):
        super().__init__()
        self.m = m
        self.nhwc = nhwc
    def forward(self, x):
        if self.nhwc:
            x = x.permute(0, 3, 1, 2).contiguous()
        return self.m(x)


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
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(path)/1e6:.1f}MB ops:{len(ops)}")
    print(f"[{label}] top:", dict(sorted(ops.items(), key=lambda kv: -kv[1])[:10]))


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhwc", action="store_true")
    args = ap.parse_args()
    m = SRVGGNetCompact()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu")["params"], strict=True)
    m.eval()
    img = torch.rand(1, S, S, 3) if args.nhwc else torch.rand(1, 3, S, S)
    with torch.no_grad():
        ref = Wrap(m, args.nhwc)(img)
    reauthor(m)
    with torch.no_grad():
        ra = Wrap(m, args.nhwc)(img)
    corr = np.corrcoef(ref.numpy().ravel(), ra.numpy().ravel())[0, 1]
    print(f"SRVGGNetCompact [{'nhwc' if args.nhwc else 'nchw'}] PS_FIX={_PS_FIX}: out {tuple(ref.shape)} | "
          f"re-auth vs orig corr {corr:.6f} max|d| {(ref-ra).abs().max():.2e}")

    tag = "realesr_x4v3" + ("_ps" if _PS_FIX else "") + ("_nhwc" if args.nhwc else "")
    import litert_torch
    litert_torch.convert(Wrap(m, args.nhwc).eval(), (img,)).export(f"{tag}.tflite")
    opcheck(f"{tag}.tflite", "realesr-fp32")
    to_fp16(f"{tag}.tflite", f"{tag}_fp16.tflite")
    opcheck(f"{tag}_fp16.tflite", "realesr-fp16")
    o = run_tflite(f"{tag}.tflite", img.numpy())
    print(f"tflite(fp32) vs torch-reauth corr {np.corrcoef(o, ra.numpy().ravel())[0,1]:.6f}")
    img.numpy().astype(np.float32).tofile(f"{tag}_input.bin")
    it2 = opcheck(f"{tag}_fp16.tflite", "fix") if False else None
    o.astype(np.float32).tofile(f"{tag}_expected.bin")


if __name__ == "__main__":
    main()
