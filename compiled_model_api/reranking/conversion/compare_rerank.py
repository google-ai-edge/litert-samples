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

"""Compare device probe_out_0.bin to ref_out.npy — reranker P(yes) score."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def pyes(v):
    """Softmax P(yes) from a [no, yes] logit pair.

    Args:
        v: Length-2 array of [no, yes] logits.

    Returns:
        The softmax probability of "yes" as a float.
    """
    a = v - v.max()
    e = np.exp(a)
    return float(e[1] / e.sum())


def main():
    """Compares the device output dump against the CPU reference."""
    L, real_len = 256, 0
    for line in open(os.path.join(HERE, "meta.txt")):
        if line.startswith("real_len="):
            real_len = int(line.split("=")[1])
        if line.startswith("L="):
            L = int(line.split("=")[1])

    ref = np.load(os.path.join(HERE, "ref_out.npy")).reshape(L, 2)
    dev = np.fromfile(
        sys.argv[1] if len(sys.argv) > 1
        else os.path.join(HERE, "probe_out_0.bin"),
        dtype="<f4").reshape(L, 2)

    p = real_len - 1
    sr, sd = pyes(ref[p]), pyes(dev[p])
    print(f"[device] P(yes)  ref={sr:.4f}  dev={sd:.4f}  |Δ|={abs(sr-sd):.5f}")
    print(f"  ref logits[no,yes]={ref[p]}  dev={dev[p]}")
    print("  VERDICT:", "fp16 OK" if abs(sr - sd) < 0.02 else "DEGRADED")


if __name__ == "__main__":
    main()
