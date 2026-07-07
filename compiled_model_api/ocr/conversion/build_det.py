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

"""PP-OCRv5 mobile detection -> CompiledModel GPU. The DB head's 2 ConvTranspose2d (k2 s2) are the
only blocker (TRANSPOSE_CONV, Mali-rejected) -> replace with exact ZeroStuffConvT2d. Convert + fp16 + parity.
Run: ~/clipconv/bin/python build_det.py [stage]  stage in {opcheck,parity,all}
"""
import _stub_propack
import sys
import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PaddleOCR2Pytorch")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
from tools.infer.pytorchocr_utility import AnalysisConfig
from pytorchocr.base_ocr_v20 import BaseOCRV20
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
W_DET = os.path.join(HERE, "weights/ptocr_v5_mobile_det.safetensors")
Y_DET = os.path.join(REPO, "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml")
H, Wd = 640, 640
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}


class ZeroStuffConvT2d(nn.Module):
    """Exact GPU-clean ConvTranspose2d (k,s, p=0, op=0): 2D nearest-upsample x stride zero-stuff mask
    + flipped conv2d + crop. Mali rejects TRANSPOSE_CONV; this is RESIZE_NEAREST + MUL + CONV_2D."""
    def __init__(self, ct, Hin, Win):
        super().__init__()
        self.s = ct.stride[0]
        self.k = ct.kernel_size[0]
        self.p = ct.padding[0]
        self.op = ct.output_padding[0]
        self.Hin = Hin
        self.Win = Win
        w = ct.weight.detach()                       # (Cin, Cout, k, k)
        w = w.flip(2).flip(3).permute(1, 0, 2, 3).contiguous()   # (Cout, Cin, k, k)
        self.register_buffer("w", w)
        self.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None
                          else torch.zeros(ct.out_channels))
        mh = np.zeros((Hin * self.s, Win * self.s), np.float32)
        mh[::self.s, ::self.s] = 1.0
        self.register_buffer("mask", torch.from_numpy(mh)[None, None])

    def forward(self, x):
        xn = F.interpolate(x, size=(self.Hin * self.s, self.Win * self.s), mode="nearest") * self.mask
        y = F.conv2d(xn, self.w, bias=self.b, padding=self.k - 1)
        olH = (self.Hin - 1) * self.s + self.k - 2 * self.p + self.op
        olW = (self.Win - 1) * self.s + self.k - 2 * self.p + self.op
        return y[:, :, self.p:self.p + olH, self.p:self.p + olW]


class DetWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        o = self.net(x)
        return o["maps"] if isinstance(o, dict) else o


def build(swap):
    cfg = AnalysisConfig(W_DET, Y_DET)
    m = BaseOCRV20(cfg)
    m.net.load_state_dict(load_file(W_DET), strict=False)
    m.net.eval()
    if swap:
        L = {}
        hk = []
        for n, mo in m.net.named_modules():
            if isinstance(mo, nn.ConvTranspose2d):
                hk.append(mo.register_forward_pre_hook(
                    (lambda nm: (lambda mod, i: L.__setitem__(nm, i[0].shape[-2:])))(n)))
        with torch.no_grad():
            m.net(torch.randn(1, 3, H, Wd))
        for h in hk: h.remove()
        for name, mo in list(m.net.named_modules()):
            if isinstance(mo, nn.ConvTranspose2d) and name in L:   # skip training-only thresh branch
                par = m.net
                *pth, last = name.split(".")
                for q in pth: par = getattr(par, q)
                hh, ww = L[name]
                setattr(par, last, ZeroStuffConvT2d(mo, hh, ww))
    return m.net


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
    orig = build(swap=False)
    reauth = build(swap=True)
    x = torch.randn(1, 3, H, Wd)
    with torch.no_grad():
        a = DetWrap(orig)(x)
        b = DetWrap(reauth)(x)
    print(f"re-authored vs orig: corr {np.corrcoef(a.numpy().ravel(), b.numpy().ravel())[0,1]:.6f} "
          f"max|d| {(a-b).abs().max():.2e} shapes {tuple(a.shape)} {tuple(b.shape)}")

    import litert_torch
    fp32 = os.path.join(HERE, "ppocr_det.tflite")
    litert_torch.convert(DetWrap(reauth).eval(), (x,)).export(fp32)
    opcheck(fp32, "det")
    o = run_tflite(fp32, x.numpy())
    print(f"tflite vs torch: corr {np.corrcoef(o, b.numpy().ravel())[0,1]:.6f}")
    if stage == "all":
        to_fp16(fp32, os.path.join(HERE, "ppocr_det_fp16.tflite"))
        opcheck(os.path.join(HERE, "ppocr_det_fp16.tflite"), "det_fp16")


if __name__ == "__main__":
    main()
