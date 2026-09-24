# litert-community models page

A static page that lists every model in [litert-community](https://huggingface.co/litert-community) with its source model, its on-device benchmarks from the [leaderboard](../../benchmark/leaderboard/) in this repository ([LiteRT](https://github.com/google-ai-edge/litert) latency and memory, [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) tokens/s) and its conversion recipe under [`models/`](../../models/), where these exist. It fetches three sources when it loads, `https://huggingface.co/api/models?author=litert-community`, `benchmark/leaderboard/data/board.json` and `models/README.md`; there is no data file in this directory to update. The recipe links are parsed from the model list table in `models/README.md`; if that table changes shape, the page loads without them and says so. The Family and Size filters are read from names: Family is the leading letters of the source model's name (of the repo's name when the card names none), Size the parameter count a name carries (`0.6B`, `4B`, `270M`), so a model whose names carry no size is listed under "Size not in the name". Each model's details carry the [LiteRT CLI](https://github.com/google-ai-edge/LiteRT-CLI) lines that download, benchmark and run one `.tflite` file or `.litertlm` bundle of it, the one with the shortest name, on the machine at hand.

- [`space/`](space/): the page, `index.html`, `app.js`, `style.css` and the `README.md` card, the four files the Space needs.
- [`test/`](test/): the checks, which stay out of the Space; [`test/README.md`](test/README.md) lists the three commands.

To look at the page on your machine:

```bash
cd space && python3 -m http.server 8000
```

then open `http://localhost:8000/` (all three sources allow cross-origin requests, so any local server works).
