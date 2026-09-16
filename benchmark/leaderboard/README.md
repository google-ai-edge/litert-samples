# benchmark/leaderboard

A performance leaderboard of [LiteRT](https://github.com/google-ai-edge/litert) models on Android
devices: `index.html` renders `data/board.json`, one row per model file, device and accelerator,
measured with LiteRT's `benchmark_model` on Developer Device Platform (DDP) lab phones through
`litert benchmark --ddp`. The board starts with `.tflite` models on Android through DDP;
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) bundles are not on it. Paths follow
whatever layout you set: the page fetches `data/` relative to itself, and the driver in `../driver/`
takes `--data-dir`.

## What is on the board

A row is one (model repo, file) on one device and one accelerator, at the newest runtime version
measured. `data/measurements.jsonl` keeps every job, one line each; `data/board.json` is the
selection the page reads. Rows are ordered by median latency and compared only within one device,
accelerator and task, which the page's filters select. The page's "How the rows are measured"
section says what the binary ran and where each number comes from.

| Column | Meaning |
|---|---|
| Model | Hugging Face repo and the `.tflite` file; the task chip is the repo's pipeline tag |
| Device, Accelerator | DDP device id and its name from `matrix.yaml`; `cpu` or `gpu` with the delegate and nodes delegated N/M from the device log |
| Median, Avg, p95, Init, First inference, Footprint | `results.pb` (`tflite.tools.benchmark.BenchmarkResult`): latency in ms and overall memory footprint, shown in MB |
| Runs | inference runs the device completed; each row reports p95 and the run count from a single session |
| Runtime, Date | the `benchmark_model` release the CLI pins; the day the session outputs were pulled from the bucket |

## How a row is made

Rows are produced by a maintainer running the driver and committed here as data. The driver needs
`python3` with PyYAML, and `protoc` for the numbers from `results.pb`; new sessions also need the `litert` CLI with the `--ddp` target
(in review in [LiteRT-CLI](https://github.com/google-ai-edge/LiteRT-CLI); the `benchmark-on-ddp`
skill says how to install it until it lands), `gcloud auth application-default login`, and a
Google Cloud project with the Device Run API enabled, which the sessions are billed to.

1. Add the model to `../driver/matrix.yaml`: `repo`, `file`, `task`, `accelerators`. The devices every
   session runs on are listed once in the same file.

2. Print what would run, then run it without `--dry-run` (one DDP session per model and accelerator;
   the driver waits for each session, collects it, and rebuilds the board):

```bash
python3 ../driver/run_matrix.py --dry-run
```

3. Or turn sessions you already have into rows, then rebuild the board:

```bash
python3 ../driver/collect.py ~/.cache/litert-cli/ddp/session-fff9643f --model litert-community/MobileNet-v2
python3 ../driver/build_board.py
```

4. Look at the page, then commit `data/`:

```bash
python3 -m http.server 8000
```

`collect.py` decodes `results.pb` with `protoc` against LiteRT's `benchmark_result.proto`, fetched
once into `~/.cache/litert-samples-benchmark/`; without `protoc` it reads the results block and the
timing line from `logcat.txt` and marks the row `source: logcat`. A job with no results is reported
on stderr and not added. Pass `--runtime-version` when the CLI's pin differs from `matrix.yaml`.

## Tested on

The three rows in `data/`: `mobilenet_v2.tflite` from
[litert-community/MobileNet-v2](https://huggingface.co/litert-community/MobileNet-v2) on caiman-35
(Pixel 9 Pro, CPU and GPU) and pa3q-35 (Galaxy S25 Ultra, CPU), binary 2.2.0, collected on macOS with
Python 3.14 and protoc 34.1 on 2026-09-16. `run_matrix.py` was run in dry-run mode; the three sessions
were submitted by hand with `litert benchmark --ddp`, CPU on both devices and GPU on caiman-35.
