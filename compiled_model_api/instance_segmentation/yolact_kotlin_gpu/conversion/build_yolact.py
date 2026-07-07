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

"""Build GPU-compatible YOLACT ResNet50 (COCO instance segmentation) via litert-torch.

The graph emits the raw heads (loc / conf / mask coefficients / prototype masks);
box decode, NMS and the lincomb mask assembly run host-side in Kotlin. CUDA is
stubbed out so the CPU-only trace works, and device_count is faked to 2 so
yolact picks use_jit=False (the FPN stays a plain traceable nn.Module).

Only graph patch: ZeroPadMaxPool (-inf PADV2 -> 0-pad + unpadded maxpool).

Setup:
    pip install torch litert-torch huggingface_hub
    git clone https://github.com/dbolya/yolact.git   # into this directory

Run:
    python build_yolact.py
    # -> ../yolact.tflite + ../ref_in.npy / ../ref_out*.npy parity fixtures
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/yolact")
sys.argv = ['x']  # yolact modules read argv
torch.cuda.current_device = lambda: 0            # yolact calls this at import (CPU box)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 2              # -> use_jit=False so FPN is plain nn.Module (traceable)
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "map_location": "cpu"})
from data import set_cfg
set_cfg('yolact_resnet50_config')                # must run before importing Yolact
from yolact import Yolact


class ZeroPadMaxPool(nn.Module):
    """Explicit 0-pad + unpadded maxpool (Mali rejects the -inf PADV2 lowering)."""

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), value=0.0)
        return F.max_pool2d(x, 3, stride=2, padding=0)


class Wrap(nn.Module):
    """Expose the four raw heads as ordered tensors for conversion."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        d = self.n(x)
        return d['loc'], d['conf'], d['mask'], d['proto']


def main():
    pth = hf_hub_download("dbolya/yolact-resnet50", "yolact_resnet50_54_800000.pth")
    net = Yolact()
    net.load_weights(pth)
    net.eval()
    net.detect = lambda pred_outs, *a, **k: pred_outs   # bypass NMS -> raw dict

    for name, m in list(net.named_modules()):
        if isinstance(m, nn.MaxPool2d):
            p = net
            *path, last = name.split('.')
            for q in path:
                p = getattr(p, q)
            setattr(p, last, ZeroPadMaxPool())

    w = Wrap(net).eval()
    dummy = torch.randn(1, 3, 550, 550)
    with torch.no_grad():
        o = w(dummy)
    print("raw outs:", [tuple(t.shape) for t in o])
    np.save("../ref_in.npy", dummy.numpy())
    for i, t in enumerate(o):
        np.save(f"../ref_out{i}.npy", t.numpy())
    import litert_torch
    litert_torch.convert(w, (dummy,)).export("../yolact.tflite")
    print("saved %.1f MB" % (os.path.getsize("../yolact.tflite") / 1e6))


if __name__ == "__main__":
    main()
