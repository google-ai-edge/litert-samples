# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Draws the showcase sign fixtures: test_assets/showcase/sign_*.jpg.

Not test fixtures: they feed integration_test/showcase_test.dart, which takes
screenshots of both demos. Two 1280x720 photo-like scenes:

- sign_do_not_feed.jpg: a yellow street sign on a post against a sky, reading
  "PLEASE DO NOT FEED THE AI";
- sign_wifi.jpg: a yellow sticky note on a wooden desk, next to a mug, reading
  "WIFI PASSWORD: gemma4".

    python3 tool/make_showcase_signs.py [--sign-font PATH] [--note-font PATH]

Needs Pillow (pip install pillow). The default fonts are macOS's DIN Condensed
Bold (the street sign) and Marker Felt (the handwritten note); elsewhere pass
any TrueType fonts, e.g. a condensed bold sans and a handwriting font. Other
fonts draw different images than the committed ones.
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
OUT = Path(__file__).resolve().parent.parent / "test_assets" / "showcase"
DEFAULT_SIGN_FONT = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"
DEFAULT_NOTE_FONT = "/System/Library/Fonts/MarkerFelt.ttc"


def noise(img: Image.Image, amount: int, seed: int) -> Image.Image:
    """Adds per-pixel grain so the scene reads as a photo, not a flat card."""
    rnd = random.Random(seed)
    px = img.load()
    for y in range(0, img.height, 2):
        for x in range(0, img.width, 2):
            r, g, b = px[x, y]
            n = rnd.randint(-amount, amount)
            c = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)))
            px[x, y] = c
            if x + 1 < img.width:
                px[x + 1, y] = c
    return img


def street_sign(sign_font: str) -> Image.Image:
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):  # sky gradient
        t = y / H
        d.line([(0, y), (W, y)], fill=(int(110 + 90 * t), int(160 + 70 * t), int(230 + 20 * t)))
    # Distant trees and a lawn.
    for i in range(14):
        x = i * 100 - 20
        d.ellipse([x, 470, x + 170, 640], fill=(48 + i % 3 * 8, 98 + i % 4 * 6, 52))
    d.rectangle([0, 600, W, H], fill=(88, 132, 66))
    # Post and sign, a little left of centre.
    d.rectangle([566, 300, 590, H], fill=(140, 142, 146))
    sign = Image.new("RGBA", (560, 300), (0, 0, 0, 0))
    s = ImageDraw.Draw(sign)
    s.rounded_rectangle([0, 0, 559, 299], radius=28, fill=(30, 30, 30))
    s.rounded_rectangle([10, 10, 549, 289], radius=22, fill=(250, 204, 21))
    s.rounded_rectangle([22, 22, 537, 277], radius=16, outline=(30, 30, 30), width=6)
    font_big = ImageFont.truetype(sign_font, 92)
    font_small = ImageFont.truetype(sign_font, 58)
    for text, font, y in [("PLEASE DO NOT", font_small, 38), ("FEED THE AI", font_big, 108)]:
        w = s.textlength(text, font=font)
        s.text(((560 - w) / 2, y), text, font=font, fill=(25, 25, 25))
    s.text((228, 214), "* * *", font=font_small, fill=(25, 25, 25))
    sign = sign.rotate(-3, resample=Image.BICUBIC, expand=True)
    img.paste(sign, (300, 70), sign)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return noise(img, 6, 1)


def sticky_note(note_font: str) -> Image.Image:
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    rnd = random.Random(7)
    for y in range(H):  # wooden desk: warm planks with grain lines
        plank = (y // 180) % 2
        base = (150, 104, 66) if plank else (138, 95, 60)
        d.line([(0, y), (W, y)], fill=base)
    for _ in range(260):
        y = rnd.randint(0, H)
        d.line([(0, y), (W, y + rnd.randint(-12, 12))], fill=(120, 82, 50), width=1)
    for y in (180, 360, 540):
        d.line([(0, y), (W, y)], fill=(95, 64, 40), width=3)
    # A mug on the right, seen from above.
    d.ellipse([930, 360, 1170, 600], fill=(240, 240, 236))
    d.ellipse([960, 390, 1140, 570], fill=(92, 58, 34))
    d.rounded_rectangle([1150, 450, 1215, 510], radius=24, outline=(240, 240, 236), width=16)
    # The note, with a soft shadow.
    note = Image.new("RGBA", (560, 520), (0, 0, 0, 0))
    n = ImageDraw.Draw(note)
    n.rectangle([0, 0, 559, 519], fill=(255, 236, 110))
    n.rectangle([0, 0, 559, 60], fill=(250, 226, 92))
    font = ImageFont.truetype(note_font, 74)
    font_big = ImageFont.truetype(note_font, 96)
    n.text((48, 110), "WIFI", font=font, fill=(30, 40, 90))
    n.text((48, 200), "PASSWORD:", font=font, fill=(30, 40, 90))
    n.text((48, 320), "gemma4", font=font_big, fill=(170, 30, 40))
    n.line([(48, 440), (430, 432)], fill=(170, 30, 40), width=6)
    shadow = Image.new("RGBA", note.size, (0, 0, 0, 90)).filter(ImageFilter.GaussianBlur(14))
    note = note.rotate(4, resample=Image.BICUBIC, expand=True)
    shadow = shadow.rotate(4, resample=Image.BICUBIC, expand=True)
    img.paste(shadow, (232, 112), shadow)
    img.paste(note, (210, 90), note)
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    return noise(img, 5, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draws test_assets/showcase/sign_*.jpg.")
    parser.add_argument(
        "--sign-font", default=DEFAULT_SIGN_FONT, help=f"street sign font (default: {DEFAULT_SIGN_FONT})"
    )
    parser.add_argument(
        "--note-font", default=DEFAULT_NOTE_FONT, help=f"sticky note font (default: {DEFAULT_NOTE_FONT})"
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    street_sign(args.sign_font).save(OUT / "sign_do_not_feed.jpg", quality=88)
    sticky_note(args.note_font).save(OUT / "sign_wifi.jpg", quality=88)
    print(f"wrote {OUT}/sign_do_not_feed.jpg, sign_wifi.jpg")


if __name__ == "__main__":
    main()
