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
"""Draws the example receipts of this sample (app/src/main/assets/examples) and writes their fields.

Every receipt is written for this sample: invented store names, an address in "Anytown" with ZIP 00000, phone numbers
in the 555-0100 to 555-0199 range that is kept for fiction, invented items and prices. They are US receipts whose date
is printed as MM/DD/YYYY or MM/DD/YY, with the day above 12. Fonts: DejaVu Sans Mono and DejaVu Sans Bold
(matplotlib's bundled copy; Bitstream Vera license with DejaVu changes in the public domain).

Two forms per receipt:
  <name>_scan.png   the receipt as a thermal printer lays it out (576 px wide, 80 mm at 203 dpi), on white
  <name>_photo.jpg  a photo-like copy: the same receipt drawn at twice the size on off-white paper, turned up to
                    2 degrees with a slight perspective, laid on a darker table, uneven light, blur, noise, JPEG
                    quality 80. It imitates a phone photo of a printed receipt; it is not a photo.

  pip install pillow==13.0.0 numpy matplotlib
  python tools/make_examples.py [--only bakery_photo,bike_shop_scan]

Writes the pictures and examples.json: the keys and, for each picture, the fields the app's default keys ask for
(store_name, tax and total as printed, date written as YYYY-MM-DD) and the date as printed.
"""
import argparse
import hashlib
import json
import os
import random

import matplotlib
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MPL_TTF = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
MONO = os.path.join(MPL_TTF, "DejaVuSansMono.ttf")
MONO_BOLD = os.path.join(MPL_TTF, "DejaVuSansMono-Bold.ttf")
SANS_BOLD = os.path.join(MPL_TTF, "DejaVuSans-Bold.ttf")

KEYS = ["store_name", "date", "tax", "total"]
COLS = 42          # characters per line of an 80 mm thermal printer at this font size
WIDTH = 576        # px: 80 mm at 203 dpi

def money(cents):
  return f"{cents // 100}.{cents % 100:02d}" if cents >= 0 else f"-{(-cents) // 100}.{(-cents) % 100:02d}"


# Each receipt: header lines, items (name, qty, unit cents) or discount lines (name, None, -cents),
# tax rate in basis points on the whole subtotal, date and time strings as printed, payment.
RECEIPTS = [
  dict(id="s3a", store="HILLTOP FARM STAND", addr=["14 Orchard Road", "Anytown, ST 00000", "Tel 555-0103"],
       date="09/14/2025", time="10:22 AM", tax_bp=500, pay=("VISA", "1184"), items=[
         ("SWEET CORN 6 EARS", 1, 450), ("HEIRLOOM TOMATOES", 1, 625), ("PEACHES 2LB", 1, 598),
         ("FRESH EGGS DOZEN", 1, 550), ("WILDFLOWER HONEY", 1, 1200), ("ZUCCHINI", 2, 125)]),
  dict(id="s3b", store="BAYSIDE BIKE SHOP", addr=["301 Shore Drive", "Anytown, ST 00000", "Tel 555-0111"],
       date="10/18/25", time="16:05", tax_bp=825, pay=("MASTERCARD", "6623"), items=[
         ("INNER TUBE 29IN", 2, 899), ("TIRE LEVERS SET", 1, 499), ("CHAIN LUBE", 1, 1099), ("BIKE BELL", 1, 1450),
         ("PATCH KIT", 1, 399), ("WATER BOTTLE", 1, 1299), ("TUNE UP BASIC", 1, 6500)]),
  dict(id="s3c", store="GOLDEN WOK KITCHEN", addr=["88 Lantern Way", "Anytown, ST 00000", "Tel 555-0126"],
       date="11/21/2025", time="7:36 PM", tax_bp=875, pay=("CASH", 6000), items=[
         ("KUNG PAO CHICKEN", 1, 1395), ("BEEF BROCCOLI", 1, 1495), ("FRIED RICE LARGE", 1, 895),
         ("EGG ROLLS", 3, 250), ("HOT SOUR SOUP", 1, 475)]),
  dict(id="s3d", store="MAPLEWOOD HARDWARE", addr=["9 Timber Lane", "Anytown, ST 00000", "Tel 555-0159"],
       date="10/16/25", time="08:57", tax_bp=700, pay=("DEBIT", "3051"), items=[
         ("DRYWALL SCREWS 1LB", 1, 799), ("SPACKLE TUB", 1, 649), ("PUTTY KNIFE", 1, 499), ("SANDING BLOCK", 2, 349),
         ("PAINT ROLLER KIT", 1, 1299), ("PAINTERS TAPE", 2, 599), ("STORE COUPON", None, -200)]),
  dict(id="s3e", store="SILVER SPOON BAKERY", addr=["52 Mill Street", "Anytown, ST 00000", "Tel 555-0168"],
       date="09/27/2025", time="9:12 AM", tax_bp=600, pay=("AMEX", "7790"), items=[
         ("CROISSANT BUTTER", 4, 325), ("SOURDOUGH ROUND", 1, 750), ("CINNAMON ROLL", 2, 395), ("LATTE LARGE", 2, 525),
         ("BLUEBERRY MUFFIN", 1, 350)]),
  dict(id="s3f", store="TALL PINES PET CENTER", addr=["600 Ridge Avenue", "Anytown, ST 00000", "Tel 555-0172"],
       date="11/19/25", time="13:48", tax_bp=650, pay=("VISA", "2468"), items=[
         ("DOG FOOD SALMON", 1, 3899), ("CAT TREATS TUNA", 3, 299), ("CHEW BONES LARGE", 1, 1149),
         ("LITTER CLUMPING", 1, 1799), ("FISH TANK FILTER", 1, 2199), ("PET BED MEDIUM", 1, 3499), ("CATNIP TOY", 2, 449),
         ("NAIL CLIPPERS", 1, 999)]),
]


def iso_date(us):
  """MM/DD/YYYY or MM/DD/YY -> YYYY-MM-DD."""
  m, d, y = us.split("/")
  return f"{int(y) + 2000 if len(y) == 2 else int(y):04d}-{int(m):02d}-{int(d):02d}"


def lines_of(r):
  """The receipt's lines as (text, style) with style in head / body / bold, and its fields as printed."""
  out = [(r["store"], "head")]
  out += [(a.center(COLS), "body") for a in r["addr"]]
  out.append(("", "body"))
  out.append((f"Date: {r['date']}".ljust(COLS - len(f"Time: {r['time']}")) + f"Time: {r['time']}", "body"))
  out.append((f"Receipt {int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 90000 + 10000}  Register 2".ljust(COLS), "body"))
  out.append(("-" * COLS, "body"))
  sub = 0
  for name, qty, cents in r["items"]:
    if qty is None:                                  # a discount line
      sub += cents
      out.append((("  " + name).ljust(COLS - 8) + money(cents).rjust(8), "body"))
    elif qty == 1:
      sub += cents
      out.append((name.ljust(COLS - 8) + money(cents).rjust(8), "body"))
    else:
      sub += qty * cents
      out.append((name.ljust(COLS), "body"))
      out.append((f"  {qty} @ {money(cents)}".ljust(COLS - 8) + money(qty * cents).rjust(8), "body"))
  tax = (sub * r["tax_bp"] + 5000) // 10000
  total = sub + tax
  out.append(("-" * COLS, "body"))
  out.append(("SUBTOTAL".ljust(COLS - 10) + money(sub).rjust(10), "body"))
  rate = f"{r['tax_bp'] / 100:.2f}".rstrip("0").rstrip(".")
  out.append((f"SALES TAX {rate}%".ljust(COLS - 10) + money(tax).rjust(10), "body"))
  out.append(("TOTAL".ljust(COLS - 10) + money(total).rjust(10), "bold"))
  out.append(("", "body"))
  kind, ref = r["pay"]
  if kind == "CASH":
    tendered = max(ref, (total + 999) // 1000 * 1000)   # never less than the total
    out.append(("CASH".ljust(COLS - 10) + money(tendered).rjust(10), "body"))
    assert tendered >= total, (r["id"], tendered, total)
    out.append(("CHANGE".ljust(COLS - 10) + money(tendered - total).rjust(10), "body"))
  else:
    out.append((f"{kind} ****{ref}".ljust(COLS - 10) + money(total).rjust(10), "body"))
    out.append(("APPROVED".ljust(COLS), "body"))
  out.append(("", "body"))
  n_items = sum(1 if q is None or q == 1 else q for _, q, _ in r["items"] if q is not None)
  out.append((f"ITEMS SOLD {n_items}".center(COLS), "body"))
  out.append(("THANK YOU - PLEASE COME AGAIN".center(COLS), "body"))
  printed = {"store_name": r["store"], "date": r["date"], "time": r["time"], "subtotal": money(sub),
             "tax": money(tax), "total": money(total)}
  return out, printed


def draw(lines, scale=1, paper=(255, 255, 255), ink=(20, 20, 20)):
  body = ImageFont.truetype(MONO, 22 * scale)
  bold = ImageFont.truetype(MONO_BOLD, 22 * scale)
  head = ImageFont.truetype(SANS_BOLD, 34 * scale)
  lh, hh, margin = 30 * scale, 52 * scale, 24 * scale
  h = margin * 2 + hh + lh * (len(lines) - 1) + 10 * scale
  img = Image.new("RGB", (WIDTH * scale, h), paper)
  d = ImageDraw.Draw(img)
  y = margin
  for text, style in lines:
    if style == "head":
      font, size = head, 34
      while d.textlength(text, font=font) > (WIDTH - 20) * scale:   # a long store name: smaller, never clipped
        size -= 1
        font = ImageFont.truetype(SANS_BOLD, size * scale)
      w = d.textlength(text, font=font)
      d.text(((WIDTH * scale - w) / 2, y + (34 - size) * scale // 2), text, font=font, fill=ink)
      y += hh
    else:
      d.text((10 * scale, y), text, font=bold if style == "bold" else body, fill=ink)
      y += lh
  return img


def perspective_coeffs(src, dst):
  """PIL's 8 perspective coefficients mapping output points dst to input points src."""
  import numpy as np
  a, b = [], []
  for (x, y), (u, v) in zip(dst, src):
    a.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
    a.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
  return list(np.linalg.solve(np.array(a, dtype=float), np.array(b, dtype=float)))


def photo_like(lines, seed):
  import numpy as np
  rng = random.Random(seed)
  rec = draw(lines, scale=2, paper=(246, 243, 233), ink=(38, 36, 40))
  rw, rh = rec.size
  W, H = 1800, max(2400, int(rh * 1.18))
  # table: a dark warm gradient with grain
  yy, xx = np.mgrid[0:H, 0:W]
  base = 70 + 30 * (xx / W) + 15 * np.sin(yy / 37.0)
  table = np.stack([base + 18, base + 6, base - 8], axis=-1)
  table += np.random.default_rng(seed).normal(0, 6, table.shape)
  bg = Image.fromarray(np.clip(table, 0, 255).astype("uint8"))
  # the receipt with a slight perspective (the top a little narrower), then turned up to 2 degrees
  inset = rw * rng.uniform(0.02, 0.04)
  src = [(0, 0), (rw, 0), (rw, rh), (0, rh)]
  dst = [(inset, 0), (rw - inset, 0), (rw, rh), (0, rh)]
  warped = rec.transform((rw, rh), Image.PERSPECTIVE, perspective_coeffs(src, dst), Image.BICUBIC,
                         fillcolor=(0, 0, 0))
  mask = Image.new("L", (rw, rh), 255).transform((rw, rh), Image.PERSPECTIVE, perspective_coeffs(src, dst),
                                                Image.BICUBIC, fillcolor=0)
  angle = rng.uniform(-2.0, 2.0)
  warped = warped.rotate(angle, resample=Image.BICUBIC, expand=True)
  mask = mask.rotate(angle, resample=Image.BICUBIC, expand=True)
  scale = min((W * 0.86) / warped.width, (H * 0.94) / warped.height)
  size = (int(warped.width * scale), int(warped.height * scale))
  warped, mask = warped.resize(size, Image.LANCZOS), mask.resize(size, Image.LANCZOS)
  shadow = Image.new("RGB", size, (25, 20, 18))
  x0, y0 = (W - size[0]) // 2 + rng.randint(-30, 30), (H - size[1]) // 2 + rng.randint(-20, 20)
  bg.paste(shadow, (x0 + 14, y0 + 18), mask.filter(ImageFilter.GaussianBlur(14)))
  bg.paste(warped, (x0, y0), mask)
  # uneven light: brighter towards one corner, darker towards the other
  arr = np.asarray(bg).astype("float32")
  cx, cy = rng.uniform(0.2, 0.8) * W, rng.uniform(0.1, 0.4) * H
  dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(W ** 2 + H ** 2)
  arr *= (1.10 - 0.38 * dist)[..., None]
  arr += np.random.default_rng(seed + 1).normal(0, 4.5, arr.shape)
  img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.1))
  return img


NAMES = {"s3a": "farm_stand", "s3b": "bike_shop", "s3c": "wok_kitchen", "s3d": "hardware", "s3e": "bakery",
         "s3f": "pet_center"}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  here = os.path.dirname(os.path.abspath(__file__))
  ap.add_argument("--out", default=os.path.join(here, "..", "app", "src", "main", "assets", "examples"))
  ap.add_argument("--only", default=None, help="comma-separated picture names, e.g. bakery_photo,bike_shop_scan")
  a = ap.parse_args()
  only = set(a.only.split(",")) if a.only else None
  os.makedirs(a.out, exist_ok=True)
  index = {"keys": KEYS, "examples": []}
  for i, r in enumerate(RECEIPTS):
    lines, printed = lines_of(r)
    fields = {"store_name": printed["store_name"], "date": iso_date(r["date"]), "tax": printed["tax"],
              "total": printed["total"]}
    for form, ext in (("scan", "png"), ("photo", "jpg")):
      name = f"{NAMES[r['id']]}_{form}"
      if only is not None and name not in only:
        continue
      path = os.path.join(a.out, f"{name}.{ext}")
      if form == "scan":
        draw(lines).save(path, format="PNG")
      else:
        photo_like(lines, seed=5000 + i).save(path, format="JPEG", quality=80)
      index["examples"].append({"file": os.path.basename(path), "fields": fields, "printed_date": r["date"],
                                "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()})
  with open(os.path.join(a.out, "examples.json"), "w") as fh:
    json.dump(index, fh, indent=1)
    fh.write("\n")
  print(f"wrote {len(index['examples'])} pictures and examples.json to {os.path.normpath(a.out)}")


if __name__ == "__main__":
  main()
