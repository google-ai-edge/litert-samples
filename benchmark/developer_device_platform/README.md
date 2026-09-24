# Benchmark using Real Devices in Developer Device Platform

Notebooks to benchmark [LiteRT](https://github.com/google-ai-edge/litert) (`.tflite`) and
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (`.litertlm`) models on real phones in
[Developer Device Platform](https://docs.cloud.google.com/developer-device-platform/overview) (DDP), Google Cloud's managed
device lab, from a Colab runtime. Nothing is installed on your machine, and no device is attached to it.

| Notebook | What it is |
| :--- | :--- |
| [`litert_cli_benchmark.ipynb`](litert_cli_benchmark.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-ai-edge/litert-samples/blob/main/benchmark/developer_device_platform/litert_cli_benchmark.ipynb) | The easiest way to benchmark LiteRT runtimes. The [LiteRT CLI](https://github.com/google-ai-edge/LiteRT-CLI) drives DDP for you, with native support for LiteRT and LiteRT-LM models. |
| [`ddp_cli_benchmark.ipynb`](ddp_cli_benchmark.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-ai-edge/litert-samples/blob/main/benchmark/developer_device_platform/ddp_cli_benchmark.ipynb) | Uses the DDP CLI directly, with the LiteRT-LM runtime as the example. Extends easily across several devices, and to other benchmark binaries for other runtimes. |

## What you need

A Google Cloud project with billing enabled and the Device Run API (`devicerun.googleapis.com`) turned on: DDP is in Preview
and each session is billed to that project. The [DDP quickstart](https://docs.cloud.google.com/developer-device-platform/quickstart)
covers the setup; the first cell of either notebook enables the API for the project you name and authenticates the runtime with
Colab auth, so a project id is the only input.

## `litert_cli_benchmark.ipynb`: the LiteRT CLI

Drives DDP through the [LiteRT CLI](https://github.com/google-ai-edge/LiteRT-CLI): `litert benchmark --ddp` uploads the model,
reserves a device, runs the prebuilt `benchmark_model` (LiteRT) or the LiteRT-LM benchmark binary on it, and prints the results.

1. **Environment setup**: set `ddp_gcp_project`, authenticate, enable the Device Run API.
2. **Benchmark with the LiteRT CLI**: `pip install litert-cli-nightly`, then, on the device set in `ddp_device`
   (`caiman-35` by default; `gcloud beta device-run devices list` shows the catalog):
   - `litert download litert-community/efficientnet_b1` and `litert benchmark ... --ddp --cpu`, then `--gpu`, which report
     latency (median, average, p95), init time and memory footprint per accelerator;
   - `litert download litert-community/Qwen3-0.6B` (a `.litertlm` bundle) and `litert benchmark ... --ddp --gpu`, which
     reports prefill and decode tokens/s and time to first token.

Both models come from [Hugging Face](https://huggingface.co/litert-community) and need no conversion. To benchmark your own
model, point `litert benchmark` at a local `.tflite` or `.litertlm` file instead.

## `ddp_cli_benchmark.ipynb`: the DDP CLI directly

Calls `gcloud alpha device-run sessions submit android-executable` itself. Reach for it when you need one of these:

1. **Other runtimes.** LiteRT-LM is the example, but DDP runs any Android binary you hand it. The same notebook structure
   benchmarks another runtime by swapping the executable, its flags and the files pushed next to it.
2. **Lower-level control.** You set every flag, and you get back what the binary itself writes. Here that is a metrics
   protobuf from `--metric_proto_file_path`, plus each device's `logcat.txt`.
3. **Device fleets.** A single session fans the same run out across a list of devices, one job per device.

It benchmarks [Gemma 4 E2B](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) with
[`litert_lm_advanced_main`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/litert_lm_advanced_main.cc),
taking the binary and its shared libraries from the public release bucket at
`gs://litert/binaries/latest/android_arm64/litert_lm/`, in two ways:

1. **Direct call**: the binary is the session executable, running one configuration across several devices.
   `--executable-env-vars` sets the `LD_LIBRARY_PATH` that GPU runs need to load the OpenCL delegate.
2. **Wrapper script**: a shell script is the session executable and runs several configurations per device (CPU, then
   GPU), with a discarded warm-up pass and a cooldown between them so later runs are not throttled.

Both pull the metrics files back with `--paths-to-pull`, then tabulate prefill and decode tokens/s, time to first token and
peak memory, confirm the OpenCL delegate actually loaded, and chart the comparison.

## Related

- [`../leaderboard/`](../leaderboard): a leaderboard of results collected this way, plus the drivers in
  [`../driver/`](../driver) that run the same DDP sessions in batch from a workstation.
- [`../ios/`](../ios): the same `benchmark_model` on a locally attached iPhone.
