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

"""Build GPU-compatible Ultra-Fast-Lane-Detection (CULane ResNet18) via
litert-torch.

Expects the Ultra-Fast-Lane-Detection repository checked out next to this script
and the official CULane ResNet18 checkpoint saved as culane_official.pth
(from the UFLD release Google Drive).

Only patch: ZeroPadMaxPool (ResNet stem MaxPool(-inf PADV2) -> 0-pad +
unpadded maxpool).

Run: python build_ufld.py
  # -> ufld.tflite ([1,3,288,800] -> [1,201,18,4] row-anchor logits)
  #    + ref fixtures
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                + "/Ultra-Fast-Lane-Detection")
from model.model import parsingNet


class ZeroPadMaxPool(nn.Module):
    """Explicit 0-pad + unpadded maxpool.

    Mali rejects the -inf PADV2 lowering.
    """

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), value=0.0)
        return F.max_pool2d(x, 3, stride=2, padding=0)


def main():
    """Converts UFLD (CULane ResNet18) to ufld.tflite with fixtures."""
    # CULane: griding_num=200 -> cls_dim=(201,18,4)
    net = parsingNet(pretrained=False, backbone='18',
                     cls_dim=(201, 18, 4), use_aux=False).eval()
    pth = "culane_official.pth"
    sd = torch.load(pth, map_location="cpu")
    sd = sd.get('model', sd)
    sd = {k[len('module.'):] if k.startswith('module.') else k: v
          for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))

    for name, m in list(net.named_modules()):
        if isinstance(m, nn.MaxPool2d):
            p = net
            *path, last = name.split('.')
            for q in path:
                p = getattr(p, q)
            setattr(p, last, ZeroPadMaxPool())

    dummy = torch.randn(1, 3, 288, 800)
    with torch.no_grad():
        o = net(dummy)
    print("out:", tuple(o.shape))   # expect [1,201,18,4]
    np.save("ref_in.npy", dummy.numpy())
    np.save("ref_out.npy", o.numpy())
    import litert_torch
    litert_torch.convert(net, (dummy,)).export("ufld.tflite")
    print("saved %.1f MB" % (os.path.getsize("ufld.tflite") / 1e6))


if __name__ == "__main__":
    main()
