# Benchmark using Real Devices in Developer Device Platform

Two notebooks benchmark [LiteRT](https://github.com/google-ai-edge/litert) (`.tflite`) and
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (`.litertlm`) models on real phones in
[Developer Device Platform](https://docs.cloud.google.com/developer-device-platform/overview) (DDP), Google Cloud's managed
device lab, from a Colab runtime. Nothing is installed on your machine, and no device is attached to it.

| Notebook | Use it when |
| :--- | :--- |
| [`ddp_benchmark.ipynb`](ddp_benchmark.ipynb) | You want a number for one model on one device. One command per run. |
| [`ddp_benchmark_advanced.ipynb`](ddp_benchmark_advanced.ipynb) | You need a sweep across several devices and backends, a binary the CLI does not wrap, or control over every benchmark flag. |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-ai-edge/litert-samples/blob/main/benchmark/developer_device_platform/ddp_benchmark.ipynb)
&nbsp;&nbsp;`ddp_benchmark.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-ai-edge/litert-samples/blob/main/benchmark/developer_device_platform/ddp_benchmark_advanced.ipynb)
&nbsp;&nbsp;`ddp_benchmark_advanced.ipynb`

## What you need

A Google Cloud project with billing enabled and the Device Run API (`devicerun.googleapis.com`) turned on: DDP is in Preview
and each session is billed to that project. The [DDP quickstart](https://docs.cloud.google.com/developer-device-platform/quickstart)
covers the setup; the first cell of either notebook enables the API for the project you name and authenticates the runtime with
Colab auth, so a project id is the only input.

## `ddp_benchmark.ipynb` — the LiteRT CLI

Drives DDP through the [LiteRT CLI](https://github.com/google-ai-edge/LiteRT-CLI): `litert benchmark --ddp` uploads the model,
reserves a device, runs the prebuilt `benchmark_model` (LiteRT) or the LiteRT-LM benchmark binary on it, and prints the results.

1. **Environment setup** — set `ddp_gcp_project`, authenticate, enable the Device Run API.
2. **Benchmark with the LiteRT CLI** — `pip install litert-cli-nightly`, then, on the device set in `ddp_device`
   (`caiman-35` by default; `gcloud beta device-run devices list` shows the catalog):
   - `litert download litert-community/efficientnet_b1` and `litert benchmark ... --ddp --cpu`, then `--gpu`, which report
     latency (median, average, p95), init time and memory footprint per accelerator;
   - `litert download litert-community/Qwen3-0.6B` (a `.litertlm` bundle) and `litert benchmark ... --ddp --gpu`, which
     reports prefill and decode tokens/s and time to first token.

Both models come from [Hugging Face](https://huggingface.co/litert-community) and need no conversion. To benchmark your own
model, point `litert benchmark` at a local `.tflite` or `.litertlm` file instead.

## `ddp_benchmark_advanced.ipynb` — the `device-run` CLI directly

Skips the LiteRT CLI and calls `gcloud alpha device-run sessions submit android-executable` itself, so you choose the binary,
the flags and the files pushed to the device. It benchmarks
[Gemma 4 E2B](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) with
[`litert_lm_advanced_main`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/litert_lm_advanced_main.cc),
taking the binary and its shared libraries from the public release bucket at
`gs://litert/binaries/latest/android_arm64/litert_lm/`.

It shows the two ways to shape a DDP session:

1. **Direct execution** — the binary is the session executable. `--executable-args` carries its flags and
   `--executable-env-vars` sets `LD_LIBRARY_PATH`, which GPU runs need so the loader can find the OpenCL delegate that is
   `dlopen`ed at runtime. Simple, but `--executable-args` is session-level, so one session is one configuration.
2. **Wrapper script** — a shell script is the session executable instead, and it runs several configurations per device
   (CPU, then GPU). This is what lets a single session sweep backends, discard a warm-up pass, and cool the SoC between
   configurations so later runs are not throttled.

The notebook then pulls `logcat.txt` back from the session bucket — an `android-executable` job writes process output to
logcat rather than to a separate log file — parses prefill tokens/s, decode tokens/s, time to first token and peak memory,
confirms the OpenCL delegate actually loaded, and renders a comparison table and charts.

## Related

- [`../leaderboard/`](../leaderboard) — a leaderboard of results collected this way, plus the drivers in
  [`../driver/`](../driver) that run the same DDP sessions in batch from a workstation.
- [`../ios/`](../ios) — the same `benchmark_model` on a locally attached iPhone.
