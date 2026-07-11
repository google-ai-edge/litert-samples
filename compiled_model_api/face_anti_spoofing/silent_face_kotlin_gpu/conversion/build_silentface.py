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

"""Build GPU-compatible Silent-Face (MiniFASNetV2 liveness) via litert-torch.

Expects the minivision-ai/Silent-Face-Anti-Spoofing repository checked out next
to this script as Silent-Face-Anti-Spoofing/ (the released checkpoint ships in
its resources/anti_spoof_models/). Zero graph patches needed (PReLU is clean).

Run: python build_silentface.py
  # -> silentface.tflite ([1,3,80,80] -> [1,3] live/print/replay probs)
  #    + ref fixtures
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "Silent-Face-Anti-Spoofing")
sys.path.insert(0, REPO)
from src.model_lib.MiniFASNet import MiniFASNetV2
from src.utility import get_kernel


class Wrap(nn.Module):
    """Append the softmax so the graph emits calibrated probabilities."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return torch.softmax(self.n(x), dim=1)   # [1,3] live/print/replay probs


def main():
    """Builds silentface.tflite from the released MiniFASNetV2 weights."""
    kernel = get_kernel(80, 80)   # (5,5)
    net = MiniFASNetV2(conv6_kernel=kernel, num_classes=3).eval()
    # load weights (repo strips 'module.' prefix)
    sd = torch.load(
        os.path.join(REPO, "resources", "anti_spoof_models",
                     "2.7_80x80_MiniFASNetV2.pth"),
        map_location="cpu")
    if next(iter(sd)).startswith('module.'):
        sd = {k[7:]: v for k, v in sd.items()}
    print("load:", net.load_state_dict(sd, strict=False))

    w = Wrap(net).eval()
    dummy = torch.rand(1, 3, 80, 80)
    with torch.no_grad():
        o = w(dummy)
    print("out:", tuple(o.shape), "sum", round(float(o.sum()), 3))
    np.save("ref_in.npy", dummy.numpy())
    np.save("ref_out.npy", o.numpy())
    import litert_torch
    litert_torch.convert(w, (dummy,)).export("silentface.tflite")
    print("saved %.2f MB" % (os.path.getsize("silentface.tflite") / 1e6))


if __name__ == "__main__":
    main()
