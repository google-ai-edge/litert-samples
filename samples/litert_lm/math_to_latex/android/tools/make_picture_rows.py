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
"""Writes app/src/test/resources/picture_rows.json: what Pillow 13.0.0 and the model card's page script make of
synthetic pictures, which PicturesTest checks the app's Kotlin against, pixel for pixel (by SHA-256).

  pip install pillow==13.0.0
  python tools/make_picture_rows.py > app/src/test/resources/picture_rows.json

The reference functions below are those of the model card's page script, ocr_page.py
(https://github.com/john-rocky/hf-to-litertlm/blob/main/minicpmocr_work/ocr_page.py): sliced_grid, pad_square,
cells and pictures, with the default tile_above of 2.0.
"""
import hashlib
import json
import math
import sys

import PIL
from PIL import Image, ImageOps

G = 448
MAX_SLICES = 9
SCALE = 448


def sliced_grid(width, height, max_slice_nums=MAX_SLICES, scale_resolution=SCALE):
  log_ratio = math.log(width / height)
  ratio = width * height / (scale_resolution * scale_resolution)
  multiple = min(math.ceil(ratio), max_slice_nums)
  if multiple <= 1:
    return None
  best, min_error = [1, 1], float("inf")
  for num_slices in (multiple - 1, multiple, multiple + 1):
    if num_slices == 1 or num_slices > max_slice_nums:
      continue
    for rows in range(1, num_slices + 1):
      if num_slices % rows == 0:
        cols = num_slices // rows
        error = abs(log_ratio - math.log(cols / rows))
        if error < min_error or (error == min_error and rows > best[0]):
          best, min_error = [rows, cols], error
  return best


def pad_square(img, g=G):
  side = max(img.size)
  square = Image.new("RGB", (side, side), "white")
  square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
  return square.resize((g, g), Image.BICUBIC)


def cells(img, grid, g=G):
  rows, cols = grid
  w, h = img.size
  out = []
  for r in range(rows):
    for c in range(cols):
      box = (round(w * c / cols), round(h * r / rows), round(w * (c + 1) / cols), round(h * (r + 1) / rows))
      out.append(img.crop(box).resize((g, g), Image.BICUBIC))
  return out


def pictures(img, tile="auto", tile_above=2.0):
  grid = sliced_grid(img.width, img.height)
  big = img.width * img.height > tile_above * G * G
  if grid is None or tile == "never" or (tile == "auto" and not big):
    return [pad_square(img)], None
  return [pad_square(img)] + cells(img, grid), grid[1]


def pattern(w, h, seed):
  """The test picture PicturesTest makes too: a gradient with dark strokes and a 31-bit LCG's noise."""
  out = bytearray(w * h * 3)
  state = seed & 0x7FFFFFFF
  i = 0
  for y in range(h):
    for x in range(w):
      state = (state * 1103515245 + 12345) & 0x7FFFFFFF
      noise = (state >> 16) & 0x3F
      stroke = 170 if (x // 7 + y // 5) % 11 == 0 else 0
      for c in range(3):
        v = (x * (c + 1) * 3 + y * (3 - c) * 2 + noise) % 256
        out[i] = max(0, v - stroke)
        i += 1
  return Image.frombytes("RGB", (w, h), bytes(out))


def sha(img):
  return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()


def main():
  assert PIL.__version__ == "13.0.0", f"Pillow {PIL.__version__}: the rows are for 13.0.0"
  rows = []
  # Resizes: up, down square, wide, tall (the vertical pass first), and the two orders near their boundary.
  for (w, h, ow, oh, seed) in [(37, 23, 448, 448, 1), (900, 900, 448, 448, 2), (1200, 300, 448, 448, 3),
                               (300, 1300, 448, 448, 4), (500, 1000, 448, 448, 5), (1000, 600, 448, 448, 6),
                               (600, 1050, 448, 448, 7), (61, 448, 448, 448, 8)]:
    rows.append({"op": "resize", "w": w, "h": h, "ow": ow, "oh": oh, "seed": seed,
                 "sha256": sha(pattern(w, h, seed).resize((ow, oh), Image.BICUBIC))})
  for (w, h, seed) in [(600, 140, 11), (300, 800, 12), (448, 448, 13), (449, 449, 14), (95, 31, 15)]:
    rows.append({"op": "pad_square", "w": w, "h": h, "seed": seed, "sha256": sha(pad_square(pattern(w, h, seed)))})
  for (w, h, seed) in [(1240, 1754, 21), (1600, 500, 22), (640, 640, 23), (700, 600, 24)]:
    pics, slice_cols = pictures(pattern(w, h, seed))
    rows.append({"op": "prepare", "w": w, "h": h, "seed": seed, "slice_cols": slice_cols,
                 "sha256": [sha(p) for p in pics]})
  for (w, h) in [(448, 448), (449, 449), (640, 640), (1240, 1754), (1754, 1240), (1600, 500), (500, 1600),
                 (3000, 4000), (4032, 3024), (1000, 100), (100, 1000), (896, 896), (897, 897)]:
    rows.append({"op": "grid", "w": w, "h": h, "grid": sliced_grid(w, h)})
  for o in range(1, 9):
    im = pattern(5, 3, 31)
    exif = im.getexif()
    exif[0x0112] = o
    out = ImageOps.exif_transpose(im)
    rows.append({"op": "orient", "w": 5, "h": 3, "seed": 31, "orientation": o, "out_w": out.width,
                 "out_h": out.height, "sha256": sha(out)})
  json.dump({"generator": "tools/make_picture_rows.py", "pillow": PIL.__version__, "rows": rows}, sys.stdout, indent=1)
  sys.stdout.write("\n")


if __name__ == "__main__":
  main()
