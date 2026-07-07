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

"""PP-OCRv5 mobile recognition (PPLCNetV3 + svtr CTC encoder + CTC head) GPU op-check.
The port drops the NRTR autoregressive branch -> pure CTC (no AR decoder). Run: ~/clipconv/bin/python probe_rec.py"""
import _stub_propack
import sys
import os
import collections
import numpy as np
import torch
import torch.nn as nn
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PaddleOCR2Pytorch")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
from tools.infer.pytorchocr_utility import AnalysisConfig
from pytorchocr.base_ocr_v20 import BaseOCRV20
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
W_REC = os.path.join(HERE, "weights/ptocr_v5_mobile_rec.safetensors")
Y_REC = os.path.join(REPO, "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml")
DICT = os.path.join(HERE, "weights/ppocrv5_dict.txt")
Hh, Ww = 48, 320
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}


class RecWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        o = self.net(x)
        if isinstance(o, dict):
            o = o.get("ctc", o.get("res", list(o.values())[0]))
        return o


def opcheck(path):
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
    print("  ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"  banned: {bad or 'NONE'} | >4D: {over} | size {os.path.getsize(path)/1e6:.1f}MB")


def main():
    char_num = len(open(DICT, encoding="utf-8").read().splitlines()) + 2  # + CTC blank + space
    cfg = AnalysisConfig(W_REC, Y_REC, char_num=char_num)
    m = BaseOCRV20(cfg)
    miss, unexp = m.net.load_state_dict(load_file(W_REC), strict=False)
    print(f"char_num={char_num} loaded rec: missing={len(miss)} unexpected={len(unexp)}")
    m.net.eval()
    x = torch.randn(1, 3, Hh, Ww)
    with torch.no_grad():
        y = RecWrap(m.net)(x)
    print(f"rec input (1,3,{Hh},{Ww}) -> logits {tuple(y.shape)}  (T x num_classes; CTC)")

    print("\n=== rec op-check ===")
    try:
        import litert_torch
        litert_torch.convert(RecWrap(m.net).eval(), (x,)).export("rec_raw.tflite")
        opcheck("rec_raw.tflite")
    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
