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

"""Build GPU-compatible DM-Count (crowd counting, density regression) via
litert-torch.

Expects the cvlab-stonybrook/DM-Count repository checked out next to this
script as DM-Count/ (its git-lfs checkout includes the MIT pretrained
UCF-QNRF checkpoint at pretrained_models/model_qnrf.pth).

Only graph change (exact): the mid-graph F.upsample_bilinear
(align_corners=True, banned as RESIZE_BILINEAR on the GPU delegate) is a
linear operator, so it is re-authored as two constant-matrix multiplies with
the constant on the RHS (lowers to FULLY_CONNECTED; the delegate rejects
BATCH_MATMUL with a constant LHS). Output matches PyTorch to fp32 rounding.

Run: python build_dmcount.py
  # -> dmcount.tflite ([1,3,512,512] -> [1,1,64,64] density map) + ref fixtures
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "DM-Count"))

SIZE = 512  # fixed input; the density map comes out at SIZE/8


def bilinear_matrix(n_in, n_out):
    """Returns the [n_out, n_in] align_corners=True bilinear resize matrix.

    Args:
        n_in: Input length along the resized axis.
        n_out: Output length along the resized axis.

    Returns:
        A [n_out, n_in] float32 numpy matrix mapping n_in samples to
        n_out samples with align_corners=True bilinear weights.
    """
    matrix = np.zeros((n_out, n_in), np.float32)
    scale = (n_in - 1) / (n_out - 1)
    for i in range(n_out):
        pos = i * scale
        lo = int(np.floor(pos))
        hi = min(lo + 1, n_in - 1)
        frac = np.float32(pos - lo)
        matrix[i, lo] += 1 - frac
        matrix[i, hi] += frac
    return matrix


def exact_upsample(x, size=None, scale_factor=None):
    """Exact align_corners=True 2x bilinear upsample as constant-RHS matmuls.

    Args:
        x: Input tensor [1, C, H, W].
        size: Unused; kept for F.upsample_bilinear signature parity.
        scale_factor: Unused; the upsample is always exactly 2x.

    Returns:
        The [1, C, 2H, 2W] align_corners=True bilinear upsample of x.
    """
    h, w = int(x.shape[2]), int(x.shape[3])
    # [h, 2h]
    a_h_t = torch.from_numpy(bilinear_matrix(h, 2 * h)).t().contiguous()
    # [w, 2w]
    a_w_t = torch.from_numpy(bilinear_matrix(w, 2 * w)).t().contiguous()
    y = x.transpose(2, 3) @ a_h_t   # [1,C,w,2h]
    y = y.transpose(2, 3) @ a_w_t   # [1,C,2h,w] @ [w,2w] -> [1,C,2h,2w]
    return y


F.upsample_bilinear = exact_upsample

# needs DM-Count/ on sys.path
from models import VGG, make_layers, cfg  # noqa: E402


class DensityHead(nn.Module):
    """Wraps DM-Count to emit only the raw density map (count = map sum)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        density, _ = self.net(x)   # [1,1,SIZE/8,SIZE/8]
        return density


def main():
    """Builds dmcount.tflite from the DM-Count UCF-QNRF checkpoint."""
    net = VGG(make_layers(cfg["E"]))
    checkpoint = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "DM-Count", "pretrained_models", "model_qnrf.pth")
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model = DensityHead(net).eval()

    dummy = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        out = model(dummy)
    print("out:", tuple(out.shape), "count", float(out.sum()))
    np.save("ref_in.npy", dummy.numpy())
    np.save("ref_out.npy", out.numpy())
    import litert_torch
    litert_torch.convert(model, (dummy,)).export("dmcount.tflite")
    print("saved %.1f MB" % (os.path.getsize("dmcount.tflite") / 1e6))


if __name__ == "__main__":
    main()
