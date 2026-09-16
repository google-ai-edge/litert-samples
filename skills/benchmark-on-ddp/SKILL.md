---
name: benchmark-on-ddp
description: Measure a LiteRT .tflite model on Developer Device Platform (DDP) lab phones with `litert benchmark --ddp` and turn the session into rows of the performance leaderboard under benchmark/leaderboard - one matrix entry per model, one session per accelerator, collect results.pb and the delegate line, rebuild board.json, check the page, commit the data. Use when a model needs latency and memory numbers per device and accelerator, or when a new benchmark_model release means re-measuring the board.
---

# Benchmark on DDP

A board row is done when three things hold:

1. it comes from the `results.pb` of a DDP job, with the delegate line (`N/M` nodes) from the same job's log,
2. it names the model file, the device id, the accelerator and the `benchmark_model` release,
3. `data/board.json` was rebuilt from `data/measurements.jsonl` and the page shows it.

A number without those is not a row. This skill covers `.tflite` models on [LiteRT](https://github.com/google-ai-edge/litert);
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) bundles are not covered. Every session consumes
device time billed to the Google Cloud project; the dry run prints every command it would run, so read it first.

## Before you start

- `python3` with PyYAML, and `protoc` on PATH (`brew install protobuf`, or `apt install protobuf-compiler`).
- For new sessions: `gcloud auth application-default login`, a project with the Device Run API enabled
  (`--gcp-project` or `LITERT_GCP_PROJECT`), and the `litert` CLI with the `--ddp` target. The target is in
  review in [LiteRT-CLI](https://github.com/google-ai-edge/LiteRT-CLI); until it lands, install the CLI from
  the review branch, after which `litert benchmark --help` lists `--ddp`:

```bash
pip install https://github.com/john-rocky/LiteRT-CLI/archive/refs/heads/benchmark-ddp-target.zip
```

## Loop

Work from the board folder, `cd benchmark/leaderboard` (or wherever the board lives); every command below
is relative to it, and the driver sits in `../driver/`.

**1. One model, one matrix entry.** In `../driver/matrix.yaml` add `repo`, `file` (one `.tflite`; a repo
with several variants gets one entry per variant you want on the board), `task` (the repo's pipeline tag)
and `accelerators`. Devices are listed once at the top of the file.

**2. Print the plan, then run it.**

```bash
python3 ../driver/run_matrix.py --dry-run --only litert-community/MobileNet-v2
LITERT_GCP_PROJECT=your-project-id python3 ../driver/run_matrix.py
```

The dry run prints one `litert benchmark … --ddp` line per accelerator, each followed by the `collect.py`
call that turns its session into rows. The second line submits each session, waits for it, pulls the job
outputs to `~/.cache/litert-cli/ddp/<session>/<job>/`, collects them and rebuilds the board; `--only <repo>` limits it to one model.

**3. Collect a session you already have.**

```bash
python3 ../driver/collect.py ~/.cache/litert-cli/ddp/session-fff9643f --model litert-community/MobileNet-v2
```

One row per job; a row with the same id replaces the earlier one, so a re-run is safe. A job with no results
is printed on stderr and skipped: read the tail of its `logcat.txt` before submitting it again. The row's
runtime version is `runtime.version` from the matrix unless you pass `--runtime-version` (the CLI's pin).

**4. Rebuild the board and look at the row.**

```bash
python3 ../driver/build_board.py
python3 -m http.server 8000
```

Open http://localhost:8000/ and click the new row. Check: nodes delegated reads `N/M` with `N = M` for a
graph that ran fully on the accelerator; a GPU row with Init far above Median is the delegate initializing
the graph, not a defect; `Numbers from: logcat` means `results.pb` could not be decoded, usually a missing `protoc`: install it and collect again.

**5. Commit the data.** `../driver/matrix.yaml`, `data/measurements.jsonl` and `data/board.json` in one commit; the page is static.

## Watch for

- **A new `benchmark_model` release is a new row, not an edit.** The row id carries the runtime version;
  the board shows the newest per model, device and accelerator, and earlier rows stay in `measurements.jsonl`.
- **Device names come from the matrix.** An id it does not list renders as the id; add it under `devices` first.
- **Never edit a row by hand.** Re-collect the session; the driver is the only writer.

## Tested on

macOS host, Python 3.14, protoc 34.1: the four rows in the repo (2026-09-16) came from `run_matrix.py` end to
end, two sessions on caiman-35 (Pixel 9 Pro) and pa3q-35 (Galaxy S25 Ultra), CPU and GPU, binary 2.2.0; the
collect and board steps were also run on their cached outputs.
