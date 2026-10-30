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
"""Renders the app's example pages (app/src/main/assets/examples/*.png): two lecture handouts written for this
sample, typeset with MathJax 3.2.2 (SVG) in headless Chrome, text in DejaVu Serif, 1240 x 1754 pixels (A4 at 150
dpi). Also writes examples.json with every display formula's LaTeX and every paragraph's text, each with its box on
the page.

  python tools/make_examples.py --mathjax app/build/generated/mathjax/mathjax/tex-svg-full.js \
      --font /path/to/DejaVuSerif.ttf --font-italic /path/to/DejaVuSerif-Italic.ttf --font-bold /path/to/DejaVuSerif-Bold.ttf \
      --out app/src/main/assets/examples [--chrome "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]

The MathJax file is the one the build fetches (./gradlew :app:fetchMathJax). Chrome runs with a throwaway profile.
"""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from html import escape

PAGES = {
  "handout_attention": {
    "course": "CS 224 &middot; Deep Learning", "lecture": "Lecture 12",
    "title": "Scaled dot-product attention",
    "body": [
      ("p", r"Given queries \(Q \in \mathbb{R}^{n \times d_k}\), keys \(K \in \mathbb{R}^{m \times d_k}\) and values "
            r"\(V \in \mathbb{R}^{m \times d_v}\), attention weighs every value by how well its key matches the query:"),
      ("eq", "12.1", r"\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left( \frac{QK^{\top}}{\sqrt{d_{k}}} \right) V"),
      ("p", r"The softmax is taken row by row, so each row of weights sums to one:"),
      ("eq", "12.2", r"\mathrm{softmax}(\mathbf{z})_{i} = \frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}}"),
      ("p", r"Dividing by \(\sqrt{d_k}\) keeps the dot products from growing with the key size, where the softmax "
            r"would saturate. Multi-head attention runs \(h\) heads in parallel and joins them:"),
      ("eq", "12.3", r"\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_{1}, \ldots, \mathrm{head}_{h})\, W^{O}"),
      ("eq", "12.4", r"\mathrm{head}_{i} = \mathrm{Attention}(Q W_{i}^{Q},\, K W_{i}^{K},\, V W_{i}^{V})"),
      ("p", r"A language model trained with these layers minimizes the cross-entropy of the next token:"),
      ("eq", "12.5", r"\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_{\theta}(y_{t} \mid y_{<t}, x)"),
      ("p", r"Exercise. Show that scaling \(Q\) by a constant \(c > 0\) changes the attention weights only through "
            r"the temperature of the softmax."),
    ],
  },
  "handout_normal": {
    "course": "STAT 201 &middot; Probability and Statistics", "lecture": "Lecture 7",
    "title": "The normal distribution",
    "body": [
      ("p", r"A random variable \(X\) is normally distributed with mean \(\mu\) and standard deviation \(\sigma\), "
            r"written \(X \sim N(\mu, \sigma^{2})\), when its density is"),
      ("eq", "7.1", r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{(x-\mu)^{2}}{2\sigma^{2}} \right)"),
      ("p", r"The density is symmetric about \(\mu\), with inflection points at \(\mu \pm \sigma\). Any normal "
            r"variable can be standardized:"),
      ("eq", "7.2", r"Z = \frac{X - \mu}{\sigma} \sim N(0, 1)"),
      ("p", r"so every probability comes from one table of \(\Phi\), the standard normal distribution function:"),
      ("eq", "7.3", r"P(a \leq X \leq b) = \Phi\left( \frac{b - \mu}{\sigma} \right) - \Phi\left( \frac{a - \mu}{\sigma} \right)"),
      ("p", r"For the mean \(\bar{X}_{n}\) of \(n\) independent observations with finite variance, the central "
            r"limit theorem gives"),
      ("eq", "7.4", r"\sqrt{n}\, \frac{\bar{X}_{n} - \mu}{\sigma} \xrightarrow{d} N(0, 1)"),
      ("p", r"Example. Exam scores with \(\mu = 70\) and \(\sigma = 8\): the share above 86 is "
            r"\(1 - \Phi(2) \approx 0.023\)."),
    ],
  },
}

CSS = """
@font-face {{ font-family: Page; src: url("{font}"); }}
@font-face {{ font-family: Page; font-style: italic; src: url("{italic}"); }}
@font-face {{ font-family: Page; font-weight: bold; src: url("{bold}"); }}
html, body {{ margin: 0; background: #fff; }}
#page {{ width: 1240px; height: 1754px; box-sizing: border-box; padding: 110px 120px; font-family: Page;
        font-size: 23px; line-height: 1.55; color: #161616; }}
.head {{ display: flex; justify-content: space-between; font-size: 18px; color: #444; border-bottom: 1px solid #999;
        padding-bottom: 8px; }}
h1 {{ font-size: 36px; font-weight: bold; margin: 46px 0 26px; }}
p {{ margin: 18px 0; text-align: justify; }}
.eq {{ display: flex; align-items: center; margin: 22px 0; }}
.eq .m {{ flex: 1; text-align: center; font-size: 25px; }}
.eq .n {{ width: 90px; text-align: right; }}
"""


def html(page, mathjax, fonts):
  parts = [f'<div class="head"><span>{page["course"]}</span><span>{page["lecture"]}</span></div>',
           f'<h1>{page["title"]}</h1>']
  for item in page["body"]:
    if item[0] == "p":
      parts.append(f"<p>{escape(item[1], quote=False)}</p>")
    else:
      parts.append(f'<div class="eq" data-tag="{item[1]}"><div class="m">\\[{escape(item[2], quote=False)}\\]</div><div class="n">({item[1]})</div></div>')
  # Runs after MathJax's first typeset (startup.pageReady) and the fonts: every formula's and paragraph's box, into
  # the title.
  report = """async function report() {
  // Measure after the page's fonts are in: the screenshot run lays the text out with them.
  await document.fonts.ready;
  const box = r => [Math.floor(r.left), Math.floor(r.top), Math.ceil(r.right), Math.ceil(r.bottom)];
  const eqs = [...document.querySelectorAll('.eq')].map(e =>
    ({tag: e.dataset.tag, box: box(e.querySelector('.m mjx-container svg').getBoundingClientRect())}));
  const paras = [...document.querySelectorAll('p')].map(e => box(e.getBoundingClientRect()));
  document.title = JSON.stringify({eqs: eqs, paras: paras});
}"""
  return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS.format(**fonts)}</style>'
          f"<script>{report}\nwindow.MathJax={{svg:{{fontCache:'global'}},startup:{{pageReady:()=>"
          f"MathJax.startup.defaultPageReady().then(report)}}}};</script><script src=\"{mathjax}\"></script></head>"
          f'<body><div id="page">{"".join(parts)}</div></body></html>')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--mathjax", required=True)
  ap.add_argument("--font", required=True)
  ap.add_argument("--font-italic", required=True)
  ap.add_argument("--font-bold", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--chrome", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  fonts = {"font": "file://" + os.path.abspath(a.font), "italic": "file://" + os.path.abspath(a.font_italic),
           "bold": "file://" + os.path.abspath(a.font_bold)}
  meta = {"generator": "tools/make_examples.py", "mathjax": "3.2.2", "font": "DejaVu Serif", "pages": []}
  with tempfile.TemporaryDirectory() as tmp:
    for name, page in PAGES.items():
      src = os.path.join(tmp, f"{name}.html")
      open(src, "w").write(html(page, "file://" + os.path.abspath(a.mathjax), fonts))
      png = os.path.join(a.out, f"{name}.png")
      base = [a.chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={tmp}/profile",
              "--no-first-run", "--no-default-browser-check", "--force-device-scale-factor=1",
              "--window-size=1240,1754", "--virtual-time-budget=15000"]
      subprocess.run(base + [f"--screenshot={png}", "file://" + src], check=True, capture_output=True, timeout=180)
      dom = subprocess.run(base + ["--dump-dom", "file://" + src], check=True, capture_output=True, text=True,
                           timeout=180).stdout
      title = dom.split("<title>", 1)[1].split("</title>", 1)[0] if "<title>" in dom else "[]"
      found = json.loads(title.replace("&quot;", '"'))
      boxes = {b["tag"]: b["box"] for b in found["eqs"]}
      formulas = [{"tag": it[1], "latex": it[2], "box": boxes.get(it[1])} for it in page["body"] if it[0] == "eq"]
      texts = [it[1] for it in page["body"] if it[0] == "p"]
      paragraphs = [{"text": s, "box": b} for s, b in zip(texts, found["paras"])]
      meta["pages"].append({"file": f"{name}.png", "sha256": hashlib.sha256(open(png, "rb").read()).hexdigest(),
                            "formulas": formulas, "paragraphs": paragraphs})
      print(f"{png}: {len(formulas)} formulas, boxes {[f['box'] for f in formulas]}")
  json.dump(meta, open(os.path.join(a.out, "examples.json"), "w"), indent=1)


if __name__ == "__main__":
  main()
