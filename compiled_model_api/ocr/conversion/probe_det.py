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

"""PP-OCRv5 mobile detection (DBNet: PPLCNetV4 + RepLKFPN + DB head) GPU op-check.
Run: ~/clipconv/bin/python probe_det.py [H W]
"""
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

W_DET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/ptocr_v5_mobile_det.safetensors")
Y_DET = os.path.join(REPO, "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml")
H = int(sys.argv[1]) if len(sys.argv) > 1 else 640
Wd = int(sys.argv[2]) if len(sys.argv) > 2 else 640
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}


class DetWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        o = self.net(x)
        return o["maps"] if isinstance(o, dict) else o


def main():
    cfg = AnalysisConfig(W_DET, Y_DET)
    m = BaseOCRV20(cfg)
    sd = load_file(W_DET)
    missing, unexpected = m.net.load_state_dict(sd, strict=False)
    print(f"loaded det: missing={len(missing)} unexpected={len(unexpected)}")
    m.net.eval()
    x = torch.randn(1, 3, H, Wd)
    with torch.no_grad():
        y = DetWrap(m.net)(x)
    print(f"det input (1,3,{H},{Wd}) -> maps {tuple(y.shape)}  range [{y.min():.3f},{y.max():.3f}]")

    print("\n=== det op-check ===")
    try:
        import litert_torch
        litert_torch.convert(DetWrap(m.net).eval(), (x,)).export("det_raw.tflite")
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path="det_raw.tflite")
        it.allocate_tensors()
        ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
        bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
        over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
        print("  ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
        print(f"  banned: {bad or 'NONE'} | >4D: {over} | size {os.path.getsize('det_raw.tflite')/1e6:.1f}MB")
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
