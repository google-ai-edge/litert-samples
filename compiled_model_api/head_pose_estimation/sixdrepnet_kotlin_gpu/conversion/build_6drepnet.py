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

"""Build GPU-compatible 6DRepNet (head pose estimation) via litert-torch.

Uses the deploy-mode (re-parameterized) RepVGG-B1g2 backbone with the
300W-LP/AFLW2000 weights from the osanseviero HF mirror. The graph emits the
raw 6D rotation representation; the Gram-Schmidt 6D -> rotation matrix ->
Euler angles decode runs host-side in Kotlin.

Setup: pip install torch litert-torch sixdrepnet huggingface_hub

Run: python build_6drepnet.py
  # -> 6drepnet.tflite ([1,3,224,224] -> [1,6]) + ref fixtures
"""
import os

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from sixdrepnet.model import SixDRepNet


class Wrap(nn.Module):
    """Backbone + GAP + 6D regression head only.

    Skips the in-graph rotation decode.
    """

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        x = self.n.layer0(x)
        x = self.n.layer1(x)
        x = self.n.layer2(x)
        x = self.n.layer3(x)
        x = self.n.layer4(x)
        x = self.n.gap(x)
        x = torch.flatten(x, 1)
        return self.n.linear_reg(x)   # 6D [1,6]


def main():
    """Converts 6DRepNet to 6drepnet.tflite and saves fixtures."""
    net = SixDRepNet(backbone_name='RepVGG-B1g2', backbone_file='', deploy=True,
                     pretrained=False).eval()
    sd = torch.load(
        hf_hub_download("osanseviero/6DRepNet_300W_LP_AFLW2000",
                        "model.pth"),
        map_location="cpu")
    sd = (sd.get('model_state_dict', sd.get('state_dict', sd))
          if isinstance(sd, dict) else sd)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    print("load:", net.load_state_dict(sd, strict=False))

    w = Wrap(net).eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        o = w(dummy)
    print("6D out:", tuple(o.shape))
    np.save("ref_in.npy", dummy.numpy())
    np.save("ref_out.npy", o.numpy())
    import litert_torch
    litert_torch.convert(w, (dummy,)).export("6drepnet.tflite")
    print("saved %.1f MB" % (os.path.getsize("6drepnet.tflite") / 1e6))


if __name__ == "__main__":
    main()
