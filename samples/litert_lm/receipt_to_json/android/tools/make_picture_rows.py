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

- "pad": the SHA-256 of the 768 x 1344 picture that pad_to below (the model card's ocr_xiaomi.py, verbatim) makes
  from a test pattern of each size. The sizes cover pictures taller and wider than the 4:7 canvas, and canvas sides
  that land half-way between two integers, which Python's round() sends to the even one.
- "orient": the SHA-256 of the pattern turned by each EXIF orientation as Pillow's ImageOps.exif_transpose turns it.
- "kie_prompt": the model card's key-information-extraction prompt for six keys (ocr_xiaomi.py's kie_prompt, verbatim).
- "kie_default_prompt": the same prompt for the app's default keys.
- "kie_schema_prompt": the same prompt for a list key and a value, with the schema written by json.dumps(schema,
  indent=4) (for plain keys it writes kie_prompt's text).

  pip install pillow==13.0.0 numpy
  python tools/make_picture_rows.py
"""
import hashlib
import json
import os

import numpy as np
import PIL
from PIL import Image

H, W = 1344, 768
KIE_HEAD = "Extract key information in the image"
PLAIN_KEYS = ["store_name", "date", "time", "subtotal", "tax", "total"]
DEFAULT_KEYS = ["store_name", "date", "tax", "total"]
LIST_SCHEMA = {"items": [{"name": "", "price": ""}], "total": ""}


def kie_prompt(keys):
  """The model card's key-information-extraction prompt, with one empty string per key."""
  body = "{\n" + ",\n".join(f'    "{k}": ""' for k in keys) + "\n}"
  return f"{KIE_HEAD}\n\nPlease output the key information in JSON format according to the following schema:\n{body}"


def kie_schema_prompt(schema):
  """The same prompt for any schema, written as json.dumps writes it (for plain keys: kie_prompt's text)."""
  return (f"{KIE_HEAD}\n\nPlease output the key information in JSON format according to the following schema:\n"
          f"{json.dumps(schema, indent=4)}")


def pad_to(img, h=H, w=W):
  """Centre img on a white canvas of aspect h:w that just holds it, then resize to w x h."""
  if img.height * w > img.width * h:      # taller than the canvas shape: keep the full height
    ch, cw = img.height, round(img.height * w / h)
  else:                                   # wider: keep the full width
    cw, ch = img.width, round(img.width * h / w)
  canvas = Image.new("RGB", (cw, ch), "white")
  canvas.paste(img, ((cw - img.width) // 2, (ch - img.height) // 2))
  return canvas.resize((w, h), Image.BICUBIC)


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
  sizes = [(576, 1029), (1800, 2808), (300, 200), (768, 1344), (1000, 1750), (402, 700), (10, 4), (6, 3), (13, 31)]
  assert kie_schema_prompt(dict.fromkeys(PLAIN_KEYS, "")) == kie_prompt(PLAIN_KEYS)
  rows = {"pillow": PIL.__version__, "pad": [], "orient": [], "kie_prompt": kie_prompt(PLAIN_KEYS),
          "kie_keys": PLAIN_KEYS, "kie_default_keys": DEFAULT_KEYS, "kie_default_prompt": kie_prompt(DEFAULT_KEYS),
          "kie_schema": LIST_SCHEMA, "kie_schema_prompt": kie_schema_prompt(LIST_SCHEMA)}
  for w, h in sizes:
    rows["pad"].append({"w": w, "h": h, "sha256": sha(pad_to(pattern(w, h)))})
  for o, op in TRANSPOSE.items():
    img = pattern(5, 3).transpose(op)
    rows["orient"].append({"w": 5, "h": 3, "orientation": o, "out": list(img.size), "sha256": sha(img)})
  out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "src", "test", "resources",
                     "picture_rows.json")
  os.makedirs(os.path.dirname(out), exist_ok=True)
  with open(out, "w") as f:
    json.dump(rows, f, indent=1)
    f.write("\n")
  print(f"wrote {os.path.normpath(out)}: {len(rows['pad'])} pad rows, {len(rows['orient'])} orient rows")


if __name__ == "__main__":
  main()
