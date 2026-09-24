#!/usr/bin/env python3
"""Local server for looking at the page against fixtures and failure modes.

Serves the Space's files from ../space. When it serves app.js it points the two
data sources at the endpoints below, so the page under test is the shipped code
with only its URLs changed. The shipped page has no such switch, and nothing in
this directory goes to the Space.

  python3 test/fixture_server.py 8765
  curl 'http://127.0.0.1:8765/__mode?hf=429'        # hf: ok|paged|429|500|drop|slow|hang|html|hostile|sparse|empty
  curl 'http://127.0.0.1:8765/__mode?board=broken'  # board: ok|404|broken|noshape|norows|renamed|hang|hostile|sparse|twins|orphan
  curl 'http://127.0.0.1:8765/__mode?tree=500'      # tree: ok|500
  curl 'http://127.0.0.1:8765/__mode?recipes=404'   # recipes: ok|404|hang|notable|hostile|orphan
  curl 'http://127.0.0.1:8765/__mode?timeout=1500'  # the page's fetch timeout in ms (0 = as shipped)

http://127.0.0.1:8765/__checks runs every mode in an iframe and prints PASS or FAIL per case.
"""
import json
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"
FIX = ROOT / "test" / "fixtures"
MODE = {"hf": "ok", "board": "ok", "tree": "ok", "recipes": "ok", "timeout": "0"}

HOSTILE = {
    "id": "litert-community/<img src=x onerror=window.__pwned=1>",
    "author": "litert-community",
    "pipeline_tag": "<b>bold</b>",
    "tags": ["\"><script>window.__pwned=1</script>"],
    "downloads": 1, "likes": 0, "gated": False, "private": False,
    "lastModified": "2026-09-01T00:00:00.000Z", "createdAt": "2026-09-01T00:00:00.000Z",
    "library_name": "litert",
    "siblings": [{"rfilename": "<i>x</i>.tflite"}],
    "cardData": {"base_model": ["<img src=x onerror=window.__pwned=1>", "javascript:alert(1)", "../../x", "hostile-org/<b>b</b>", "hostile-org/still-a-repo"]},
}
HOSTILE_RECIPES = """# Model recipes

## Model list

| Recipe | Task | Artifact | Source model | Converted weights |
|---|---|---|---|---|
| [<img src=x onerror=window.__pwned=1>](javascript:alert(1)) | x | x | x | [x](https://huggingface.co/litert-community/MiniCPM5-2B) |
| [`../../etc/`](../../etc/) | x | x | x | [x](https://huggingface.co/litert-community/MobileNet-v2) |
| [`a/b/`](a/b/) | x | x | x | [<b>x</b>](https://evil.example/litert-community/whisper-tiny) |
| [`ok/dir/`](ok/dir/) | x | x | x | [<img src=x onerror=window.__pwned=1>](https://huggingface.co/litert-community/U-2-Net) |
"""
SPARSE = [
    {"id": "litert-community/no-fields"},
    {"id": "litert-community/odd-fields", "pipeline_tag": 7, "siblings": "none", "gated": "auto", "downloads": "many",
     "lastModified": None},
    {"id": "litert-community/odd-siblings", "siblings": [None, {}, {"rfilename": 3}, {"rfilename": "a.tflite"}]},
]
EVIL = "<img src=x onerror=window.__pwned=1>"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(SPACE), **kw)

    def log_message(self, fmt, *args):
        sys.stderr.write("[fx] " + fmt % args + "\n")

    def _json(self, code, obj, headers=None):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/__mode":
            for k in MODE:
                if k in q:
                    MODE[k] = q[k][0]
            return self._json(200, MODE)
        if u.path == "/__checks":
            body = (ROOT / "test" / "browser_checks.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(body)
        if u.path == "/app.js":
            src = (SPACE / "app.js").read_text()
            assert '"https://huggingface.co/api/models"' in src
            src = src.replace('"https://huggingface.co/api/models"', '"/fx/hf/api/models"')
            src = src.replace(
                "https://google-ai-edge.github.io/litert-samples/benchmark/leaderboard/data/board.json",
                "/fx/board.json")
            assert '"https://google-ai-edge.github.io/litert-samples/models/README.md"' in src
            src = src.replace('"https://google-ai-edge.github.io/litert-samples/models/README.md"', '"/fx/recipes.md"')
            if MODE["timeout"] != "0":
                line = "const TIMEOUT_MS = { models: 20000, board: 8000, recipes: 8000, files: 10000 };"
                assert line in src
                t = int(MODE["timeout"])
                src = src.replace(line, "const TIMEOUT_MS = { models: %d, board: %d, recipes: %d, files: %d };" % (t, t, t, t))
            body = src.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(body)
        if u.path == "/fx/hf/api/models":
            return self._models(q)
        if u.path.startswith("/fx/hf/api/models/") and u.path.endswith("/tree/main"):
            if MODE["tree"] == "500":
                return self._json(500, {"error": "fixture 500"})
            return self._json(200, json.loads((FIX / "tree.json").read_text()))
        if u.path == "/fx/board.json":
            return self._board()
        if u.path == "/fx/recipes.md":
            return self._recipes()
        return super().do_GET()

    def _text(self, code, text):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _recipes(self):
        mode = MODE["recipes"]
        if mode == "404":
            return self._json(404, {"error": "not found"})
        if mode == "hang":
            time.sleep(6)
        text = (FIX / "recipes.md").read_text()
        if mode == "notable":
            text = text.replace("| Recipe | Task | Artifact | Source model | Converted weights |", "| Directory | Task | Artifact | Source model | Weights |")
        if mode == "hostile":
            text = HOSTILE_RECIPES
        if mode == "orphan":
            text = text.replace("litert-community/MiniCPM5-2B", "litert-community/not-in-the-org")
        return self._text(200, text)

    def _models(self, q):
        mode = MODE["hf"]
        models = json.loads((FIX / "models.json").read_text())
        if mode == "429":
            return self._json(429, {"error": "Rate limit reached"})
        if mode == "500":
            return self._json(500, {"error": "fixture 500"})
        if mode == "drop":
            self.connection.close()
            return None
        if mode == "slow":
            time.sleep(2)
        if mode == "hang":
            time.sleep(6)
        if mode == "html":
            body = b"<!DOCTYPE html><html><body>Sign in</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        if mode == "empty":
            return self._json(200, [])
        if mode == "sparse":
            return self._json(200, SPARSE + models)
        if mode == "hostile":
            return self._json(200, [HOSTILE] + models)
        if mode == "paged":
            page = int(q.get("page", ["0"])[0])
            chunk = models[page * 3:(page + 1) * 3]
            headers = {}
            if (page + 1) * 3 < len(models):
                host = self.headers.get("Host")
                headers["Link"] = f'<http://{host}/fx/hf/api/models?page={page + 1}>; rel="next"'
            return self._json(200, chunk, headers)
        return self._json(200, models)

    def _board(self):
        mode = MODE["board"]
        if mode == "404":
            return self._json(404, {"error": "not found"})
        if mode == "broken":
            return self._json(200, b'{"rows": [')
        if mode == "noshape":
            return self._json(200, {"hello": "world"})
        if mode == "slow":
            time.sleep(2)
        if mode == "hang":
            time.sleep(6)
        board = json.loads((FIX / "board.json").read_text())
        if mode == "hostile":
            for row in board["rows"] + board["lm_rows"]:
                for k in ("file", "device", "os", "delegate", "runtime_version", "date"):
                    row[k] = EVIL
            board["platforms"].append({"id": EVIL, "name": EVIL, "rows": 1})
        if mode == "sparse":
            keep = ("model", "file", "platform", "device", "accelerator", "status")
            board["rows"] = [{**{k: r[k] for k in keep}, "latency_ms": {}} for r in board["rows"]] + [{"model": "x"}, None, 5]
            board["lm_rows"] = [{**{k: r[k] for k in keep}, "metrics": {}} for r in board["lm_rows"]]
            board["platforms"] = None
            board.pop("generated_at", None)
        if mode == "norows":
            board["rows"], board["lm_rows"] = [], []
        if mode == "renamed":
            for row in board["rows"] + board["lm_rows"]:
                row["status"] = "ok"
        if mode == "twins":
            lm = board["lm_rows"]
            repeat = [dict(r, row_id=r["row_id"] + "/2", date="2026-09-20", metrics=dict(r["metrics"], decode_tok_s=r["metrics"]["decode_tok_s"] + 1)) for r in lm]
            other = [dict(r, row_id=r["row_id"] + "/os", device_id="caiman-36", os="Android 16 (API 36)") for r in lm]
            board["lm_rows"] = lm + repeat + other
        if mode == "orphan":
            row = dict(board["rows"][0])
            row["model"] = "someone-else/not-in-the-org"
            row["row_id"] = "orphan"
            board["rows"].append(row)
        return self._json(200, board)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"fixture server on http://127.0.0.1:{port}/ (root {ROOT})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
