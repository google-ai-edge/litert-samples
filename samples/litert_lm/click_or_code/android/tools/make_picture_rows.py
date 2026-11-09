#!/usr/bin/env python3
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
"""Writes app/src/test/resources/picture_rows.json: what the JVM tests compare the app's code with.

- "stretch": the SHA-256 of the 1920 x 1088 picture that Pillow's BICUBIC resize makes from a test pattern of each
  size (the model card's script stretches every screenshot this way), desktop and phone shapes among them.
- "orient": the SHA-256 of the pattern turned by each EXIF orientation as Pillow's ImageOps.exif_transpose turns it.

  pip install pillow==13.0.0 numpy
  python tools/make_picture_rows.py
"""
import hashlib
import json
import os

import numpy as np
import PIL
from PIL import Image

H, W = 1088, 1920


def pattern(w, h):
  """The test pattern the JVM test draws too: r = 7x + 3y, g = x*x + 5y, b = x xor y (each mod 256)."""
  y, x = np.mgrid[0:h, 0:w].astype(np.int64)
  rgb = np.stack([(7 * x + 3 * y) % 256, (x * x + 5 * y) % 256, (x ^ y) % 256], axis=-1).astype(np.uint8)
  return Image.fromarray(rgb, "RGB")


def sha(img):
  return hashlib.sha256(np.asarray(img.convert("RGB")).tobytes()).hexdigest()


# ImageOps.exif_transpose's method for each orientation
TRANSPOSE = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
             5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
             8: Image.Transpose.ROTATE_90}


def main():
  sizes = [(1920, 1080), (1080, 2400), (2560, 1440), (1920, 1088), (800, 600), (390, 844), (13, 31), (6, 3)]
  rows = {"pillow": PIL.__version__, "stretch": [], "orient": []}
  for w, h in sizes:
    rows["stretch"].append({"w": w, "h": h, "sha256": sha(pattern(w, h).resize((W, H), Image.BICUBIC))})
  for o, op in TRANSPOSE.items():
    img = pattern(5, 3).transpose(op)
    rows["orient"].append({"w": 5, "h": 3, "orientation": o, "out": list(img.size), "sha256": sha(img)})
  out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "src", "test", "resources",
                     "picture_rows.json")
  os.makedirs(os.path.dirname(out), exist_ok=True)
  with open(out, "w") as f:
    json.dump(rows, f, indent=1)
    f.write("\n")
  print(f"wrote {os.path.normpath(out)}: {len(rows['stretch'])} stretch rows, {len(rows['orient'])} orient rows")


if __name__ == "__main__":
  main()
