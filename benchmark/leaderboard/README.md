# benchmark/leaderboard

A performance leaderboard of [LiteRT](https://github.com/google-ai-edge/litert) models: `index.html`
renders `data/board.json`, one row per model file, platform, device and accelerator, measured with
LiteRT's `benchmark_model` at one pinned release. Android rows come from Developer Device Platform
(DDP) lab phones through `litert benchmark --ddp`; macOS rows from the same release's macOS binary,
run on a Mac by `run_local.py`; iOS rows from the app in `../ios/`, which builds the tool from source
and runs it on a phone. The board starts with `.tflite` models;
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) bundles are not on it. Paths follow whatever
layout you set: the page fetches `data/` relative to itself, and the drivers in `../driver/` take `--data-dir`.

## What is on the board

A row is one (model repo, file) on one platform, device and accelerator, at the newest runtime version
measured. `data/measurements.jsonl` keeps every job, one line each; `data/board.json` is the selection
the page reads. Rows are grouped by platform and, by default, ordered by median latency inside the group; a comparison
holds only within one platform, device, accelerator and task, which the page's filters select. The page's
"How the rows are measured" section says what the binary ran and where each number comes from.

| Column | Meaning |
|---|---|
| Model | Hugging Face repo and the `.tflite` file; the task chip is the repo's pipeline tag |
| Platform, Device | `android`, `macos` or `ios`; the device id with its name and OS, from `matrix.yaml` for DDP devices and from the machine itself for a Mac or an iPhone |
| Accelerator | `cpu` or `gpu` with the delegate the run's log names; nodes delegated N/M comes from the Android device log and reads n/a where a log has no such line |
| Median, Avg, p95, Init, First inference, Footprint | `results.pb` (`tflite.tools.benchmark.BenchmarkResult`): latency in ms and overall memory footprint, shown in MB |
| Runs | inference runs the run completed; each row reports p95 and the run count from a single session |
| Runtime, Date | the `benchmark_model` release; the row's Binary field names the build it ran (a bucket object, or the source tag for iOS). Date is the day the outputs were written or pulled |

## How a row is made

Rows are made by running a driver and are committed here as data. The drivers need `python3` with PyYAML,
and `protoc` for the numbers from `results.pb`; new DDP sessions also need the `litert` CLI with the `--ddp`
target (in review in [LiteRT-CLI](https://github.com/google-ai-edge/LiteRT-CLI); the `benchmark-on-ddp` skill
says how to install it until it lands), `gcloud auth application-default login`, and a Google Cloud project
with the Device Run API enabled, which the sessions are billed to. Rows from a Mac need only the Mac:
`run_local.py` fetches the macOS binary of the pinned release into `~/.cache/litert-samples-benchmark/`.

1. Add the model to `../driver/matrix.yaml`: `repo`, `file`, `task`, `accelerators`. The platforms, and the
   DDP devices every session runs on, are listed once in the same file.

2. Android: print what would run, then run it (one DDP session per model and accelerator; the driver waits
   for each session, collects it, and rebuilds the board):

```bash
python3 ../driver/run_matrix.py --dry-run
LITERT_GCP_PROJECT=your-project-id python3 ../driver/run_matrix.py
```

3. macOS: the same two steps on the Mac you are on (one run per model and accelerator, the device named
   from `system_profiler`; the driver collects the session and rebuilds the board):

```bash
python3 ../driver/run_local.py --dry-run
python3 ../driver/run_local.py
```

4. Or turn sessions you already have into rows, then rebuild the board. A session written by `../ios/run_ios.sh`
   on an iPhone goes in the same way:

```bash
python3 ../driver/collect.py ~/.cache/litert-cli/ddp/session-fff9643f --model litert-community/MobileNet-v2
python3 ../driver/build_board.py
```

5. Look at the page, then commit `data/`:

```bash
python3 -m http.server 8000
```

`collect.py` decodes `results.pb` with `protoc` against LiteRT's `benchmark_result.proto`, fetched once into
`~/.cache/litert-samples-benchmark/`; without `protoc` it reads the results block and the timing line from
the job's log and marks the row `source: log`. A job with no results is reported on stderr and not added.
Pass `--runtime-version` when the CLI's pin differs from `matrix.yaml`.

## Tested on

The rows in `data/`: `mobilenet_v2.tflite` from
[litert-community/MobileNet-v2](https://huggingface.co/litert-community/MobileNet-v2), binary 2.2.0.
Android, CPU and GPU on caiman-35 (Pixel 9 Pro) and pa3q-35 (Galaxy S25 Ultra): `run_matrix.py` submitted
the two sessions on 2026-09-16. macOS, CPU and GPU on a Mac Studio (M4 Max, macOS 27.0): `run_local.py` ran
both on 2026-09-17. iOS, CPU and GPU on an iPhone 17 Pro (iOS 27.0): the app in `../ios/` built from LiteRT
v2.2.0 (`145c7523f`), one session written by `run_ios.sh` on 2026-09-17. Every row was collected from those
session outputs and the board rebuilt on 2026-09-17, with Python 3.14 and protoc 34.1.
