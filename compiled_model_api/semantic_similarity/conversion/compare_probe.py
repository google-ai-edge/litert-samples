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

"""Compare device probe_out_0.bin to ref_out.npy (last-token embedding
parity)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = 128


def cos(a, b):
    """Cosine similarity between two 1-D vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The cosine similarity as a float.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    """Compares the device output dump against the CPU reference."""
    real_len = 8
    for line in open(os.path.join(HERE, "meta.txt")):
        if line.startswith("real_len="):
            real_len = int(line.strip().split("=")[1])

    # [L,1024]
    ref = np.load(os.path.join(HERE, "ref_out.npy")).reshape(L, -1)
    dev = np.fromfile(
        sys.argv[1] if len(sys.argv) > 1
        else os.path.join(HERE, "probe_out_0.bin"),
        dtype="<f4").reshape(L, -1)

    p = real_len - 1
    lt_cos = cos(ref[p], dev[p])
    print(f"[device parity] last-token(pos{p}) cos={lt_cos:.6f}")
    print(f"  ref  absmax={np.abs(ref[p]).max():.4f}  "
          f"dev absmax={np.abs(dev[p]).max():.4f}  "
          f"dev mean={dev[p].mean():.5f}")
    # per-position cos over real tokens (depth-collapse profile)
    cs = [cos(ref[i], dev[i]) for i in range(real_len)]
    print("  per-token cos (real tokens):", " ".join(f"{c:.3f}" for c in cs))
    print("  VERDICT:", "fp16 OK (>0.99)" if lt_cos > 0.99 else
          ("DEGRADED" if lt_cos > 0.9 else "fp16 WALL / collapse"))


if __name__ == "__main__":
    main()
