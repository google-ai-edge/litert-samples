# benchmark/leaderboard

A performance leaderboard of [LiteRT](https://github.com/google-ai-edge/litert) models: `index.html` renders
`data/board.json`, one row per model file, platform, device and accelerator, measured with LiteRT's `benchmark_model`
at one pinned release: Android rows on Developer Device Platform (DDP) lab phones through `litert benchmark --ddp`, macOS
rows with the same release's macOS binary through `run_local.py`, iOS rows from the app in `../ios/`, which builds the tool
from source. A second table holds [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) rows: prefill and decode tokens/s
of a `.litertlm` bundle under LiteRT-LM's benchmark binary, run on a DDP phone through `litert benchmark <bundle>.litertlm --ddp`.
The page fetches `data/` relative to itself, and the drivers in `../driver/` take `--data-dir`.

## What is on the board

A row is one (model repo, file) on one platform, device and accelerator, at the newest runtime version measured.
`data/measurements.jsonl` (`measurements-lm.jsonl` for LiteRT-LM) keeps every job that wrote results, one line each, and
`data/board.json` is the selection the page reads, grouped by platform and, by default, ordered by median latency inside
the group (LiteRT-LM rows listed by model, device and backend); a comparison holds only within one platform, device, accelerator and task.

| Column | Meaning |
|---|---|
| Model | Hugging Face repo and the `.tflite` file; the task chip is the repo's pipeline tag |
| Platform, Device | `android`, `macos` or `ios`; the device id with its name and OS, from `matrix.yaml` for DDP devices and from the machine itself for a Mac or an iPhone |
| Accelerator | `cpu` or `gpu` with the delegate `runtime_info.pb` names (the log's name where a session has no such file); nodes delegated N/M counts the primary subgraph's nodes that delegate replaced, so a `gpu` row with a low N ran most of the graph on the CPU, and partitions is the runtime's count for the whole graph (the delegate's runs and the runs that stay off it); n/a where a session has no `runtime_info.pb` and its log no `Replacing N out of M` line |
| Median, Avg, p95, Init, First inference, Footprint, Runs | `results.pb` (`tflite.tools.benchmark.BenchmarkResult`): latency in ms, overall memory footprint in MB, and the inference runs completed; each row reports p95 and the run count from a single session |
| Runtime, Date | the `benchmark_model` release; the row's Binary field names the build it ran (a bucket object, or the source tag for iOS). Date is the day the outputs were written or pulled |
| Prefill, Decode, First token, Init, Tokens, Iterations (LiteRT-LM rows) | the `LitertLmMetricsList` the binary writes with `--metric_proto_file_path`: tokens/s and time to first token as the median over the run's iterations after the warm-up one (each iteration's values stay in the row), Init from the run's one engine creation, a cold start; Tokens is the prefill and decode count the run asked for, Iterations the count with the warm-up in parentheses; Runtime is the LiteRT-LM release, or `latest@<date>` with the object's sha256 in Binary, and Libraries the shared libraries pushed beside the binary; the row also keeps the bundle's sha256 from `provenance.txt`, since a bundle of the same name can change on the Hub |

## How a row is made

Rows are made by running a driver and are committed here as data. The drivers need `python3` with PyYAML and `protoc`
for the numbers from `results.pb` and `metrics.pb`; new DDP sessions also need the `litert` CLI of
[LiteRT-CLI](https://github.com/google-ai-edge/LiteRT-CLI) with the `--ddp` target (`pip install litert-cli-nightly`; the
0.2.0 release predates it), `gcloud auth application-default login`, and a Google Cloud project with the Device Run API
enabled, which the sessions are billed to. Rows from a Mac need only the Mac: `run_local.py` fetches the pinned macOS binary into `~/.cache/litert-samples-benchmark/`.

1. Add the model to `../driver/matrix.yaml` (`repo`, `file`, `task`, `accelerators`; `lm_models` for a bundle); the platforms and the DDP devices are listed once there.

2. Android: print what would run, then run it; `--only REPO` limits both to one repo (one DDP session per model
   and accelerator, counted by the dry run; the driver waits for each session, collects it, and rebuilds the board):

```bash
python3 ../driver/run_matrix.py --dry-run
LITERT_GCP_PROJECT=your-project-id python3 ../driver/run_matrix.py
```

3. macOS: the same two steps on the Mac you are on, `--only REPO` again for one repo (one run per model and
   accelerator, the device named from `system_profiler`; the driver collects each session and rebuilds the board):

```bash
python3 ../driver/run_local.py --dry-run
python3 ../driver/run_local.py
```

4. Or turn sessions you already have into rows, then rebuild the board; a session from `../ios/run_ios.sh` goes in the same way:

```bash
python3 ../driver/collect.py ~/.cache/litert-cli/ddp/session-fff9643f --model litert-community/MobileNet-v2
python3 ../driver/build_board.py
```

5. LiteRT-LM rows: one `litert benchmark` session per backend runs LiteRT-LM's benchmark binary on the phone (the CLI
   pushes the binary and the shared libraries of its bucket directory beside the bundle, removes the cache files an earlier
   session left, and pulls `metrics.pb`, `provenance.txt` and the logcat to `~/.cache/litert-cli/ddp/<session>/<backend>-<device>/`;
   the token counts and the five iterations are its defaults). Collect the sessions (the medians leave out the first
   iteration, `runtime_lm.warmup_iterations`), rebuild, look at the page, then commit `data/`:

```bash
LITERT_GCP_PROJECT=your-project-id litert benchmark qwen3_0_6b_mixed_int4.litertlm --ddp --device caiman-35 --cpu
LITERT_GCP_PROJECT=your-project-id litert benchmark qwen3_0_6b_mixed_int4.litertlm --ddp --device caiman-35 --gpu
python3 ../driver/collect_lm.py ~/.cache/litert-cli/ddp/session-dc8641bc ~/.cache/litert-cli/ddp/session-76a0ebe9 --model litert-community/Qwen3-0.6B --model-size-mb 497.66
python3 ../driver/build_board.py
python3 -m http.server 8000
```

`collect.py` decodes `results.pb` and `runtime_info.pb` with `protoc` against LiteRT's `benchmark_result.proto` and
`model_runtime_info.proto`, and `collect_lm.py` each job's `metrics.pb` against LiteRT-LM's `litert_lm_metrics.proto`,
all fetched once into `~/.cache/litert-samples-benchmark/`; without `protoc`, `collect.py` reads the results block and the
timing line from the job's log and marks the row `source: log`, and `collect_lm.py` stops. A job with no results is reported
on stderr and not added; pass `--runtime-version` when the CLI's pin differs from `matrix.yaml`. `collect_lm.py` also stops
when the binary that ran has another sha256 than `runtime_lm` names: a moved `latest` is a new row, so update `runtime_lm` first.

## Tested on

`benchmark_model` rows, all at release 2.2.0: the 21 files `matrix.yaml` lists, CPU and GPU, on caiman-35 (Pixel 9 Pro)
and pa3q-35 (Galaxy S25 Ultra) through `run_matrix.py` (42 sessions, 2026-09-18) and on a Mac Studio (M4 Max, macOS 27.0)
through `run_local.py` (42 runs, one session per file, 2026-09-18), every job with results; `mobilenet_v2.tflite` from
[litert-community/MobileNet-v2](https://huggingface.co/litert-community/MobileNet-v2) on an iPhone 17 Pro (iOS 27.0) through
the app in `../ios/` built from LiteRT v2.2.0 (`145c7523f`), one `run_ios.sh` session on 2026-09-17. LiteRT-LM rows:
`qwen3_0_6b_mixed_int4.litertlm` from [litert-community/Qwen3-0.6B](https://huggingface.co/litert-community/Qwen3-0.6B) on
caiman-35, CPU and GPU, one Device Run session on 2026-09-19 with the `latest` binary of the bucket's `litert_lm/` directory
as of 2026-09-18 (sha256 `adac974b…`) and the libraries `matrix.yaml` lists beside it, 1024/256 tokens, 5 iterations with the
first as warm-up. The board was rebuilt on 2026-09-19 with Python 3.14 and protoc 34.1. The two `litert benchmark` lines of step 5 were run on 2026-09-23 with litert-cli-nightly 0.3.0.dev20260922 against the bucket directory as published (sessions cc0825c2 and f264790a on caiman-35, CPU and GPU, both PASSED, the same binary sha256 as the rows above) and collected with `collect_lm.py` into a scratch data directory; the rows above are unchanged.
