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

"""PP-OCRv5 mobile recognition -> CompiledModel GPU.

Only blocker = SVTR attention's fused-QKV 5D reshape -> split q/k/v into
4D (C12 pattern, numerically identical). Convert + fp16 + parity.
CTC argmax + collapse is host-side.

Run: python build_rec.py [stage]
"""
import _stub_propack
import sys
import os
import types
import collections
import numpy as np
import torch
import torch.nn as nn
REPO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "PaddleOCR2Pytorch")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
from tools.infer.pytorchocr_utility import AnalysisConfig
from pytorchocr.base_ocr_v20 import BaseOCRV20
from pytorchocr.modeling.backbones.rec_svtrnet import Attention as SVTRAttention
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
W_REC = os.path.join(HERE, "weights/ptocr_v5_mobile_rec.safetensors")
Y_REC = os.path.join(REPO, "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml")
DICT = os.path.join(HERE, "weights/ppocrv5_dict.txt")
Hh, Ww = 48, 320
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE",
          "SELECT", "SELECT_V2", "BROADCAST_TO", "POW", "TRANSPOSE_CONV",
          "CAST", "EMBEDDING_LOOKUP", "RFFT2D", "FFT", "STFT", "COMPLEX",
          "RFFT", "IRFFT", "CUMSUM"}


def attn_4d(self, x):
    """SVTR Attention without the 5D fused-QKV reshape.

    Splits q/k/v into separate 4D (B, h, N, hd) tensors; numerically
    identical to the fused path.

    Args:
        self: The SVTR Attention module (bound via types.MethodType).
        x: Input tensor of shape (B, N, C).

    Returns:
        The projected attention output of shape (B, N, C).
    """
    b, N, C = x.shape
    h = self.num_heads
    hd = C // h
    q, k, v = self.qkv(x).split(C, dim=-1)
    q = q.reshape(b, N, h, hd).permute(0, 2, 1, 3) * self.scale
    k = k.reshape(b, N, h, hd).permute(0, 2, 1, 3)
    v = v.reshape(b, N, h, hd).permute(0, 2, 1, 3)
    attn = q.matmul(k.permute(0, 1, 3, 2))
    if self.mixer == 'Local':
        attn = attn + self.mask
    attn = torch.softmax(attn, dim=-1)
    x = attn.matmul(v).permute(0, 2, 1, 3).reshape(b, N, C)
    return self.proj(x)


class RecWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        o = self.net(x)
        if isinstance(o, dict):
            o = o.get("ctc", o.get("res", list(o.values())[0]))
        return o


def build(swap):
    """Builds the PP-OCRv5 mobile recognition net from the safetensors.

    Args:
        swap: If True, patch every SVTR Attention with the 4D re-authored
            forward.

    Returns:
        The eval-mode torch recognition network.
    """
    char_num = len(open(DICT, encoding="utf-8").read().splitlines()) + 2
    cfg = AnalysisConfig(W_REC, Y_REC, char_num=char_num)
    m = BaseOCRV20(cfg)
    m.net.load_state_dict(load_file(W_REC), strict=False)
    m.net.eval()
    if swap:
        for mo in m.net.modules():
            if isinstance(mo, SVTRAttention):
                mo.forward = types.MethodType(attn_4d, mo)
    return m.net


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
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} size "
          f"{os.path.getsize(path)/1e6:.1f}MB",
          "GPU-CLEAN" if not bad and not over else "BLOCKERS")


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
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
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
    """Converts, op-checks, and parity-tests the recognition model."""
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    orig = build(False)
    reauth = build(True)
    x = torch.randn(1, 3, Hh, Ww)
    with torch.no_grad():
        a = RecWrap(orig)(x)
        b = RecWrap(reauth)(x)
    print(f"re-authored vs orig: corr "
          f"{np.corrcoef(a.numpy().ravel(), b.numpy().ravel())[0,1]:.6f} "
          f"max|d| {(a-b).abs().max():.2e} shape {tuple(b.shape)}")
    import litert_torch
    fp32 = os.path.join(HERE, "ppocr_rec.tflite")
    litert_torch.convert(RecWrap(reauth).eval(), (x,)).export(fp32)
    opcheck(fp32, "rec")
    o = run_tflite(fp32, x.numpy())
    print(f"tflite vs torch: corr {np.corrcoef(o, b.numpy().ravel())[0,1]:.6f}")
    if stage == "all":
        to_fp16(fp32, os.path.join(HERE, "ppocr_rec_fp16.tflite"))
        opcheck(os.path.join(HERE, "ppocr_rec_fp16.tflite"), "rec_fp16")


if __name__ == "__main__":
    main()
