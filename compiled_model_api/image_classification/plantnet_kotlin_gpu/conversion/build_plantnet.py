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

"""Build GPU-compatible PlantNet-300K ResNet18 (1081-species plant ID).

Converted with litert-torch. Only patch: ZeroPadMaxPool (ResNet stem
MaxPool(-inf PADV2) -> 0-pad + unpadded maxpool).

Run: python build_plantnet.py
  # -> plantnet.tflite  ([1,3,224,224] -> [1,1081] logits)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from huggingface_hub import hf_hub_download


class ZeroPadMaxPool(nn.Module):
    """Explicit 0-pad + unpadded maxpool (Mali rejects -inf PADV2 lowering)."""

    def forward(self, x):
        # exact: maxpool input is post-ReLU >= 0
        x = F.pad(x, (1, 1, 1, 1), value=0.0)
        return F.max_pool2d(x, kernel_size=3, stride=2, padding=0)


def main():
    """Loads PlantNet-300K ResNet18 weights and exports plantnet.tflite."""
    net = resnet18(num_classes=1081).eval()
    net.load_state_dict(
        torch.load(hf_hub_download("cpoisson/plantnet300k-resnet18",
                                   "plantnet_resnet18.pth"),
                   map_location="cpu", weights_only=False))
    net.maxpool = ZeroPadMaxPool()

    dummy = torch.randn(1, 3, 224, 224)
    import litert_torch
    litert_torch.convert(net, (dummy,)).export("plantnet.tflite")
    print("saved %.1f MB" % (os.path.getsize("plantnet.tflite") / 1e6))


if __name__ == "__main__":
    main()
