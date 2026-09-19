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

"""Build the GPU-compatible PIDNet-S semantic segmentation model (Cityscapes).

Loads the trained PIDNet-S weights (from the ONNX mirror whose initializer names
match the original XuJiacong/PIDNet PyTorch keys), verifies they load 1:1, and
converts to a LiteRT CompiledModel-GPU .tflite with litert-torch. PIDNet is a
pure CNN with align_corners=False interpolation -> zero GPU patches needed.

Setup:
    pip install torch litert-torch onnx huggingface_hub
    git clone https://github.com/XuJiacong/PIDNet.git   # or set PIDNET_REPO

Run:
    PIDNET_REPO=./PIDNet python build_pidnet.py
    # -> pidnet_s.tflite  (30 MB, [1,3,1024,1024] -> [1,19,128,128],
    #    0 banned ops)
"""
import os
import sys

import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.environ.get("PIDNET_REPO", "PIDNet"))
import models.pidnet as P


class Wrap(nn.Module):
    """Reduce the multi-head training output to the single segmentation head."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        o = self.n(x)
        return o[0] if isinstance(o, (list, tuple)) else o


def main():
    """Loads the ONNX-mirror weights into PIDNet-S and exports .tflite."""
    onnx_path = hf_hub_download("oenpu/PIDNet_S_enlight_friendly_onnx",
                                "PIDNet_S_enlight_friendly.onnx")
    w = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy())
         for i in onnx.load(onnx_path).graph.initializer}

    net = P.get_pred_model("pidnet_s", 19).eval()
    sd = net.state_dict()
    matched = {k: w[k] for k in sd if k in w}
    assert len(matched) == len(sd), (
        f"only {len(matched)}/{len(sd)} weights matched")
    net.load_state_dict(matched, strict=False)
    print(f"loaded {len(matched)}/{len(sd)} weights")

    wrapped = Wrap(net).eval()
    dummy = torch.randn(1, 3, 1024, 1024)
    import litert_torch
    litert_torch.convert(wrapped, (dummy,)).export("pidnet_s.tflite")
    print("saved pidnet_s.tflite (%.1f MB)"
          % (os.path.getsize("pidnet_s.tflite") / 1e6))


if __name__ == "__main__":
    main()
