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

"""YOLOX -> CompiledModel GPU: re-authoring + op-check + parity
(litert_torch path).

The only GPU blocker in YOLOX is the Focus stem: litert_torch lowers its
stride-2 space-to-depth slicing to GATHER_ND (ML Drift rejects it). Fix
= fold Focus + the following 3x3/s1 BaseConv into a single
numerically-exact 6x6/s2/p2 conv (one CONV_2D, no slice/gather).
Everything else (CSPDarknet+PAFPN+decoupled head) is already a clean
CNN.

Head is tapped raw (decode_in_inference=False): graph outputs
[B, n_anchors, 85] with sigmoid'd obj/cls + raw reg (grid-unit
cx,cy,w,h). Grid-decode (cx+grid)*stride, exp(wh)*stride and NMS run
host-side (Kotlin) -- the SSDLite raw-head pattern; this keeps
meshgrid/exp/gather/topk out of the graph.

Run: python build_yolox.py [yolox-s|yolox-tiny]
"""
import sys
import os
import math
import collections
import argparse
import types
import numpy as np
import torch
import torch.nn as nn

# --- workaround: broken scipy _propack dlopen on Darwin 27
# (unrelated to YOLOX) ---
class _D:
    def __getattr__(self, n):
        return lambda *a, **k: None
_pp = types.ModuleType("scipy.sparse.linalg._propack")
for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_pp, _nm, _D())
sys.modules["scipy.sparse.linalg._propack"] = _pp
# ------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "YOLOX"))
from yolox.exp.build import get_exp_by_name

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE",
          "SELECT", "SELECT_V2", "BROADCAST_TO", "POW", "TRANSPOSE_CONV",
          "CAST", "EMBEDDING_LOOKUP", "RFFT2D", "FFT", "STFT", "COMPLEX",
          "RFFT", "IRFFT", "CUMSUM"}

WEIGHTS = {"yolox-s": "yolox_s.pth", "yolox-tiny": "yolox_tiny.pth",
           "yolox-nano": "yolox_nano.pth", "yolox-m": "yolox_m.pth"}


class FoldedStem(nn.Module):
    """Focus(space-to-depth, cat order TL,BL,TR,BR) + 3x3/s1/p1 BaseConv
    -> one 6x6/s2/p2 conv.

    Exact: a 3x3 conv on the depth-to-space map equals a 6x6 stride-2
    conv on the input.
    W6[o, c, 2*kh+dr, 2*kw+dc] = Wstem[o, g*3+c, kh, kw], with
    g(TL)=(dr0,dc0) g(BL)=(dr1,dc0) g(TR)=(dr0,dc1) g(BR)=(dr1,dc1).
    pad=2 reproduces Focus' implicit zero-pad at both borders ->
    bit-for-bit boundaries.
    """
    def __init__(self, focus):
        super().__init__()
        # BaseConv(12, out, ksize=3, stride=1, bias=False)
        base = focus.conv
        W = base.conv.weight.data               # [out, 12, 3, 3]
        out_ch = W.shape[0]
        conv = nn.Conv2d(3, out_ch, kernel_size=6, stride=2, padding=2,
                         bias=False)
        W6 = torch.zeros(out_ch, 3, 6, 6, dtype=W.dtype)
        # TL, BL, TR, BR  -> (dr, dc)
        groups = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for g, (dr, dc) in enumerate(groups):
            for c in range(3):
                for kh in range(3):
                    for kw in range(3):
                        W6[:, c, 2 * kh + dr, 2 * kw + dc] = (
                            W[:, g * 3 + c, kh, kw])
        conv.weight.data.copy_(W6)
        self.conv, self.bn, self.act = conv, base.bn, base.act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLOXRaw(nn.Module):
    """eval-mode YOLOX with decode_in_inference=False -> raw heads
    [B, n_anchors, 85].

    nhwc=True: graph input is NHWC [B,H,W,3] (drop-in for Android Bitmap
    pixel writes / litert-samples convention); a single leading
    TRANSPOSE -> NCHW, GPU-delegated. Channel order stays BGR 0-255
    (YOLOX-native, no normalization)."""
    def __init__(self, m, nhwc=False):
        super().__init__()
        self.m = m
        self.nhwc = nhwc
    def forward(self, x):
        if self.nhwc:
            x = x.permute(0, 3, 1, 2).contiguous()
        return self.m(x)


def build(name):
    """Builds an eval-mode YOLOX model with raw (undecoded) head output.

    Args:
        name: YOLOX variant key in WEIGHTS (e.g. "yolox-s").

    Returns:
        A (model, size) tuple; size is the square test resolution.
    """
    exp = get_exp_by_name(name)
    exp.num_classes = 80
    model = exp.get_model().eval()
    ckpt = torch.load(WEIGHTS[name], map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.head.decode_in_inference = False
    H = W = exp.test_size[0]
    return model, H


def reauthor(model):
    """Swaps the Focus stem for the numerically-exact FoldedStem.

    Args:
        model: A YOLOX model from build().

    Returns:
        The same model with the folded 6x6/s2 stem installed.
    """
    stem = model.backbone.backbone.stem
    model.backbone.backbone.stem = FoldedStem(stem)
    return model


def opcheck(path, label):
    """Static GPU-compat scan of the op set in the .tflite flatbuffer.

    Args:
        path: Path to the .tflite file to scan.
        label: Tag prepended to the printed report lines.
    """
    from ai_edge_litert import schema_py_generated as schema
    with open(path, "rb") as f:
        model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items()
             if not k.startswith("_")}
    ops = collections.Counter()
    over = 0
    for g in model.subgraphs:
        for op in g.operators:
            c = model.operatorCodes[op.opcodeIndex]
            code = max(c.builtinCode, c.deprecatedBuiltinCode)
            ops[c.customCode.decode() if c.customCode
                else names.get(code, str(code))] += 1
        over += sum(1 for t in g.tensors
                    if t.shape is not None and len(t.shape) > 4)
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    flex = {k: v for k, v in ops.items()
            if "flex" in k.lower() or "custom" in k.lower()}
    gnd = ops.get("GATHER_ND", 0)
    print(f"[{label}] GATHER_ND:{gnd}  banned:{bad or 'NONE'}  "
          f"flex/custom:{flex or 'NONE'}  "
          f">4D:{over}  size {os.path.getsize(path)/1e6:.1f}MB  "
          f"ops:{len(ops)}")
    print(f"[{label}] top:",
          dict(sorted(ops.items(), key=lambda kv: -kv[1])[:10]))


def run_tflite(path, x):
    """Runs one inference through the LiteRT CompiledModel API.

    Args:
        path: Path to the .tflite model.
        x: Input array, written to the first input buffer as fp32.

    Returns:
        The flat fp32 array read from the first output buffer.
    """
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(path)
    inputs = model.create_input_buffers(0)
    outputs = model.create_output_buffers(0)
    inputs[0].write(np.ascontiguousarray(x, dtype=np.float32))
    model.run_by_index(0, inputs, outputs)
    n = (model.get_output_buffer_requirements(0, 0)["buffer_size"]
         // np.dtype(np.float32).itemsize)
    return outputs[0].read(n, np.float32)


def to_fp16(fp32, fp16):
    """fp16 weights via AI Edge Quantizer FLOAT_CASTING.

    Args:
        fp32: Source fp32 .tflite path.
        fp16: Destination fp16 .tflite path.

    Returns:
        The fp16 path.
    """
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16):
        os.remove(fp16)
    q = quantizer.Quantizer(float_model=fp32)
    q.load_quantization_recipe(rm.get_quantization_recipe())
    q.quantize().export_model(fp16)
    return fp16


def main():
    """Converts YOLOX to .tflite (fp32 + fp16) and checks GPU parity."""
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="yolox-s", choices=list(WEIGHTS))
    ap.add_argument("--nhwc", action="store_true",
                    help="NHWC graph input (drop-in for Android)")
    args = ap.parse_args()

    model, H = build(args.name)
    layout = "nhwc" if args.nhwc else "nchw"
    img = torch.randn(1, H, W := H, 3) if args.nhwc else torch.randn(1, 3, H, H)
    with torch.no_grad():
        ref = YOLOXRaw(model, args.nhwc)(img)
    print(f"{args.name} [{layout}]: input {tuple(img.shape)}  "
          f"raw head out {tuple(ref.shape)}")

    reauthor(model)
    with torch.no_grad():
        ra = YOLOXRaw(model, args.nhwc)(img)
    corr = np.corrcoef(ref.numpy().ravel(), ra.numpy().ravel())[0, 1]
    print(f"re-authored(folded stem) vs orig: corr {corr:.6f}  "
          f"max|d| {(ref-ra).abs().max():.2e}")

    tag = args.name.replace("yolox-", "yolox_") + ("_nhwc" if args.nhwc else "")
    fp32 = f"{tag}_raw.tflite"
    import litert_torch
    litert_torch.convert(YOLOXRaw(model, args.nhwc).eval(), (img,)).export(fp32)
    opcheck(fp32, f"{args.name}-fp32")
    o = run_tflite(fp32, img.numpy()).reshape(ra.shape)
    print(f"tflite(fp32) vs torch: corr "
          f"{np.corrcoef(o.ravel(), ra.numpy().ravel())[0,1]:.6f}  "
          f"max|d| {np.abs(o-ra.numpy()).max():.2e}")

    fp16 = f"{tag}_fp16.tflite"
    to_fp16(fp32, fp16)
    opcheck(fp16, f"{args.name}-fp16")
    o16 = run_tflite(fp16, img.numpy()).reshape(ra.shape)
    print(f"tflite(fp16) vs torch: corr "
          f"{np.corrcoef(o16.ravel(), ra.numpy().ravel())[0,1]:.6f}  "
          f"max|d| {np.abs(o16-ra.numpy()).max():.2e}")


if __name__ == "__main__":
    main()
