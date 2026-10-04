#!/usr/bin/env python3
"""Local server for looking at the page against fixtures and failure modes.

Serves the Space's files from ../space. When it serves app.js it points the page's
five source URLs at the endpoints below, so the page under test is the shipped code
with only its URLs changed. The shipped page has no such switch, and nothing in this
directory goes to the Space. It serves index.html with a CSP that reaches this server
only, so a source it failed to rewrite is blocked instead of read live. The server
checks at startup that app.js and index.html still hold every line it rewrites, and
exits with status 1 when one is gone.

  python3 test/fixture_server.py 8766
  curl 'http://127.0.0.1:8766/__mode?hf=429'          # hf: the model list
  curl 'http://127.0.0.1:8766/__mode?card=404'        # card: the organization card
  curl 'http://127.0.0.1:8766/__mode?samples=one404'  # samples: the two sample tables
  curl 'http://127.0.0.1:8766/__mode?demos=empty'     # demos: the Demos collection
  curl 'http://127.0.0.1:8766/__mode?break=picks'     # break: the function that draws one part throws (fold: from the 13th row on)
  curl 'http://127.0.0.1:8766/__mode?timeout=1500'    # the page's fetch timeout in ms (0 = as shipped)

MODES below lists the values of each. A name or value it does not list is answered with 400. `hang` holds
an answer for 6 s, which is shorter than the shipped timeouts, so it fails a part only together with a
shorter `timeout`.

http://127.0.0.1:8766/__checks runs every mode in an iframe and prints PASS or FAIL per case.
"""
import json
import re
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"
FIX = ROOT / "test" / "fixtures"
MODES = {
    "hf": ("ok", "paged", "endless", "429", "500", "drop", "slow", "hang", "html", "hostile", "sparse", "empty", "quiet", "noscore", "nodate"),
    "card": ("ok", "404", "500", "slow", "hang", "html", "noheading", "noitems", "renamed", "loose", "hostile"),
    "samples": ("ok", "404", "500", "one404", "hang", "notable", "none", "hostile"),
    "demos": ("ok", "404", "500", "slow", "hang", "html", "noshape", "empty", "hostile"),
    "break": ("ok", "trending", "recent", "picks", "samples", "demos", "fold"),
}
MODE = {key: "ok" for key in MODES}
MODE["timeout"] = "0"

# The page's source URLs and the fixture endpoint each one is pointed at.
SOURCES = {
    '"https://huggingface.co/api/models"': '"/fx/hf/api/models"',
    '"https://huggingface.co/spaces/litert-community/README/raw/main/README.md"': '"/fx/card.md"',
    '"https://google-ai-edge.github.io/litert-samples/samples/litert/README.md"': '"/fx/samples_litert.md"',
    '"https://google-ai-edge.github.io/litert-samples/samples/litert_lm/README.md"': '"/fx/samples_litert_lm.md"',
    '"https://huggingface.co/api/collections/litert-community/demos-68efdb213c93efee10c2116f"': '"/fx/demos.json"',
}
TIMEOUT_LINE = "const TIMEOUT_MS = { models: 20000, card: 8000, samples: 8000, demos: 8000 };"
# The first line of the function that draws the rows of each part.
DRAWS = {
    "trending": "const trendingRow = m => ",
    "recent": "const recentRow = m => ",
    "picks": "function pickHtml(item) {",
    "samples": "function sampleHtml(s) {",
    "demos": "function demoHtml(d) {",
}

EVIL = "<img src=x onerror=window.__pwned=1>"
HOSTILE_MODEL = {
    "id": "litert-community/hostile-fields",
    "pipeline_tag": "<b>bold</b>",
    "downloads": 1, "likes": 1, "trendingScore": 99, "gated": False, "private": False,
    "createdAt": "2026-10-04T00:00:00.000Z",
    "siblings": [{"rfilename": "<i>x</i>.tflite"}],
    "cardData": {"base_model": [EVIL, "javascript:alert(1)", "../../x", "hostile-org/<b>b</b>", "hostile-org/still-a-repo"]},
}
# An id that is no repo id: the page leaves the entry out.
HOSTILE_ID = dict(HOSTILE_MODEL, id="litert-community/" + EVIL)
SPARSE = [
    {"id": "litert-community/no-fields"},
    {"id": "litert-community/odd-fields", "pipeline_tag": 7, "siblings": "none", "gated": "auto", "downloads": "many",
     "likes": None, "trendingScore": "high", "createdAt": None},
    {"id": "litert-community/odd-siblings", "trendingScore": 50, "createdAt": "2026-10-04T00:00:00.000Z",
     "siblings": [None, {}, {"rfilename": 3}, {"rfilename": "a.tflite"}]},
]
HOSTILE_CARD = """# Org

## Picks <script>window.__pwned=1</script>
* **[Good-Model](https://huggingface.co/litert-community/Laya-Multilingual-LiteRT)** (`Task`) A plain line.
* """ + EVIL + """ [x](javascript:alert(1)) [y](https://evil.example/a) [z](https://huggingface.co.evil.example/a) [w](https://huggingface.co@evil.example/a) <b>bold</b> <script>window.__pwned=1</script>
* [name & <b>co</b>](https://huggingface.co/litert-community/x"onmouseover="window.__pwned=1) `<i>code</i>` ![img](https://evil.example/p.png)

## Next
"""
# A picks section written loosely: a heading with "pick" above the one with "picks", a sentence before the
# list, numbered items, an item a comment hides, a rule, items under an item, a heading under the list,
# tags, and a link to a site the page does not link to.
LOOSE_CARD = """---
title: README
# a comment about the picks
---
# Org

## How to pick a model
- not a pick either

### Our picks
Chosen by the maintainers.

1. **[First](https://huggingface.co/litert-community/Laya-Multilingual-LiteRT)** (`Task`) One.<br>Second line of one.
<!--
1. [Hidden](https://huggingface.co/litert-community/laya-LiteRT) for now.
-->
2) <b>Second</b> on the <a href="https://example.com/x">site</a>, [docs](https://ai.google.dev/edge/litert), [elsewhere](https://example.com/y).

* * *

#### Older
- Series
    - [Third](https://huggingface.co/litert-community/PaddleOCR-VL-1.6)
    - `<fourth>` and last

### Next
- not a pick
"""
HOSTILE_SAMPLES = """# Samples

| Sample | Task | API | Platform | Model |
|---|---|---|---|---|
| [""" + EVIL + """](javascript:alert(1)) | x | x | x | [x](https://huggingface.co/litert-community/MobileNet-v2) |
| [`../../etc/`](../../etc/) | x | x | x | [x](https://huggingface.co/litert-community/MobileNet-v2) |
| [`ok_dir/`](ok_dir/) | """ + EVIL + """ | x | <b>phone</b> | [""" + EVIL + """](https://huggingface.co/litert-community/MobileNet-v2) |
"""
HOSTILE_DEMOS = {"items": [
    {"type": "space", "id": EVIL, "title": "bad id"},
    {"type": "model", "id": "someone/model", "title": "not a Space"},
    {"type": "space", "id": "someone/demo", "title": EVIL, "shortDescription": "<script>window.__pwned=1</script>", "likes": 1},
]}
HTML_PAGE = b"<!DOCTYPE html><html><body>Sign in</body></html>"


def page(html):
    """index.html with a CSP that lets the page reach this server only: a source URL the server failed to
    point at a fixture is then blocked, and its part fails, instead of reading the live source unseen."""
    found = re.findall(r"connect-src [^;]+;", html)
    if len(found) != 1:
        raise ValueError("index.html holds %d connect-src directives" % len(found))
    return html.replace(found[0], "connect-src 'self';")


def rewrite(src):
    """app.js with its sources pointed at the fixtures and the modes applied. Raises when a line is gone."""
    for needle in (*SOURCES, TIMEOUT_LINE, *DRAWS.values()):
        if src.count(needle) != 1:
            raise ValueError("app.js holds %d of: %s" % (src.count(needle), needle))
    for url, endpoint in SOURCES.items():
        src = src.replace(url, endpoint)
    if MODE["timeout"] != "0":
        t = int(MODE["timeout"])
        src = src.replace(TIMEOUT_LINE, "const TIMEOUT_MS = { models: %d, card: %d, samples: %d, demos: %d };" % (t, t, t, t))
    if MODE["break"] == "fold":
        # The rows of the Trending fold throw, after the first rows are made.
        line = DRAWS["trending"]
        src = src.replace(line, line + '{ if ((globalThis.fixtureRows = (globalThis.fixtureRows || 0) + 1) > 12) throw new Error("fixture break"); return whole(m); }; const whole = m => ')
    elif MODE["break"] != "ok":
        line = DRAWS[MODE["break"]]
        thrower = line + ' { throw new Error("fixture break"); }; const unused = m => ' if line.startswith("const") else line + ' throw new Error("fixture break");'
        src = src.replace(line, thrower)
    return src


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(SPACE), **kw)

    def log_message(self, fmt, *args):
        sys.stderr.write("[fx] " + fmt % args + "\n")

    def _send(self, code, body, ctype, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj, headers=None):
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8", headers)

    def _text(self, code, text):
        self._send(code, text.encode(), "text/markdown; charset=utf-8")

    def _html(self):
        self._send(200, HTML_PAGE, "text/html; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/__mode":
            return self._set_mode(q)
        if u.path == "/__checks":
            return self._send(200, (ROOT / "test" / "browser_checks.html").read_bytes(), "text/html; charset=utf-8")
        try:
            if u.path == "/app.js":
                return self._send(200, rewrite((SPACE / "app.js").read_text()).encode(), "text/javascript; charset=utf-8")
            if u.path in ("/", "/index.html"):
                return self._send(200, page((SPACE / "index.html").read_text()).encode(), "text/html; charset=utf-8")
        except ValueError as err:
            return self._json(500, {"error": str(err)})
        if u.path == "/fx/hf/api/models":
            return self._models(q)
        if u.path == "/fx/card.md":
            return self._card()
        if u.path in ("/fx/samples_litert.md", "/fx/samples_litert_lm.md"):
            return self._samples(u.path.rsplit("/", 1)[1])
        if u.path == "/fx/demos.json":
            return self._demos()
        return super().do_GET()

    def _set_mode(self, q):
        wanted = {k: v[0] for k, v in q.items()}
        for k, v in wanted.items():
            known = v.isdigit() if k == "timeout" else v in MODES.get(k, ())
            if not known:
                return self._json(400, {"error": "unknown mode %s=%s" % (k, v)})
        MODE.update(wanted)
        return self._json(200, MODE)

    def _models(self, q):
        mode = MODE["hf"]
        models = json.loads((FIX / "models.json").read_text())
        if mode in ("429", "500"):
            return self._json(int(mode), {"error": "fixture %s" % mode})
        if mode == "drop":
            self.connection.close()
            return None
        if mode == "slow":
            time.sleep(2)
        if mode == "hang":
            time.sleep(6)
        if mode == "html":
            return self._html()
        if mode == "empty":
            return self._json(200, [])
        if mode == "quiet":
            return self._json(200, [dict(m, trendingScore=0) for m in models])
        if mode == "noscore":
            return self._json(200, [{k: v for k, v in m.items() if k != "trendingScore"} for m in models])
        if mode == "nodate":
            return self._json(200, [{k: v for k, v in m.items() if k != "createdAt"} for m in models])
        if mode == "sparse":
            return self._json(200, SPARSE + models)
        if mode == "hostile":
            return self._json(200, [HOSTILE_ID, HOSTILE_MODEL] + models)
        if mode in ("paged", "endless"):
            page = int(q.get("page", ["0"])[0])
            chunk = models[page * 12:(page + 1) * 12] if mode == "paged" else models[:1]
            headers = {}
            if mode == "endless" or (page + 1) * 12 < len(models):
                headers["Link"] = '<http://%s/fx/hf/api/models?page=%d>; rel="next"' % (self.headers.get("Host"), page + 1)
            return self._json(200, chunk, headers)
        return self._json(200, models)

    def _card(self):
        mode = MODE["card"]
        if mode in ("404", "500"):
            return self._json(int(mode), {"error": "fixture %s" % mode})
        if mode == "slow":
            time.sleep(2)
        if mode == "hang":
            time.sleep(6)
        if mode == "html":
            return self._html()
        text = (FIX / "card.md").read_text()
        heading = "## 🌟 Community's Picks of the Week"
        if text.count(heading) != 1:
            return self._json(500, {"error": "the card fixture has no picks heading"})
        if mode == "noheading":
            text = text.replace(heading, "## 🌟 Chosen models")
        if mode == "renamed":
            text = text.replace(heading, "#### our pick")
        if mode == "noitems":
            text = "\n".join(line for line in text.split("\n") if not line.startswith(("* ", "  ")))
        if mode == "loose":
            text = LOOSE_CARD
        if mode == "hostile":
            text = HOSTILE_CARD
        return self._text(200, text)

    def _samples(self, name):
        mode = MODE["samples"]
        if mode in ("404", "500") or (mode == "one404" and name == "samples_litert.md"):
            return self._json(404 if mode == "one404" else int(mode), {"error": "fixture %s" % mode})
        if mode == "hang":
            time.sleep(6)
        text = (FIX / name).read_text()
        header = "| Sample | Task | API | Platform | Model |"
        if text.count(header) != 1:
            return self._json(500, {"error": "the sample fixture has no table header"})
        if mode == "notable":
            text = text.replace(header, "| Directory | Task | API | Platform | Weights |")
        if mode == "none":
            text = text.replace("https://huggingface.co/litert-community/", "https://huggingface.co/another-org/")
        if mode == "hostile":
            text = HOSTILE_SAMPLES
        return self._text(200, text)

    def _demos(self):
        mode = MODE["demos"]
        if mode in ("404", "500"):
            return self._json(int(mode), {"error": "fixture %s" % mode})
        if mode == "slow":
            time.sleep(2)
        if mode == "hang":
            time.sleep(6)
        if mode == "html":
            return self._html()
        if mode == "noshape":
            return self._json(200, {"error": "This collection does not exist"})
        if mode == "hostile":
            return self._json(200, HOSTILE_DEMOS)
        demos = json.loads((FIX / "demos.json").read_text())
        if mode == "empty":
            demos["items"] = []
        return self._json(200, demos)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    try:
        rewrite((SPACE / "app.js").read_text())
        page((SPACE / "index.html").read_text())
    except (OSError, ValueError) as err:
        sys.exit("fixture server: %s" % err)
    print(f"fixture server on http://127.0.0.1:{port}/ (root {ROOT})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
