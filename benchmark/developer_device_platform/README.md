# benchmark/developer_device_platform

[`ddp_benchmark.ipynb`](ddp_benchmark.ipynb) benchmarks [LiteRT](https://github.com/google-ai-edge/litert) (`.tflite`) and
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (`.litertlm`) models on real phones in
[Developer Device Platform](https://docs.cloud.google.com/developer-device-platform/overview) (DDP), Google Cloud's managed
device lab, from a Colab runtime. The notebook drives DDP through the [LiteRT CLI](https://github.com/google-ai-edge/LiteRT-CLI):
`litert benchmark --ddp` uploads the model, reserves a device, runs the prebuilt `benchmark_model` (LiteRT) or the LiteRT-LM
benchmark binary on it, and prints the results. Nothing is installed on your machine, and no device is attached to it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-ai-edge/litert-samples/blob/main/benchmark/developer_device_platform/ddp_benchmark.ipynb)

## What you need

A Google Cloud project with billing enabled and the Device Run API (`devicerun.googleapis.com`) turned on: DDP is in Preview
and each session is billed to that project. The [DDP quickstart](https://docs.cloud.google.com/developer-device-platform/quickstart)
covers the setup; the notebook's first cell enables the API for the project you name and authenticates the runtime with
Colab auth, so a project id is the only input.

## What the notebook runs

1. **Environment setup** — set `ddp_gcp_project`, authenticate, enable the Device Run API.
2. **Benchmark with the LiteRT CLI** — `pip install litert-cli-nightly`, then, on the device set in `ddp_device`
   (`caiman-35` by default; `gcloud beta device-run devices list` shows the catalog):
   - `litert download litert-community/efficientnet_b1` and `litert benchmark ... --ddp --cpu`, then `--gpu`, which report
     latency (median, average, p95), init time and memory footprint per accelerator;
   - `litert download litert-community/Qwen3-0.6B` (a `.litertlm` bundle) and `litert benchmark ... --ddp --gpu`, which
     reports prefill and decode tokens/s and time to first token.

Both models come from [Hugging Face](https://huggingface.co/litert-community) and need no conversion. To benchmark your own
model, point `litert benchmark` at a local `.tflite` or `.litertlm` file instead.

## Related

- [`../leaderboard/`](../leaderboard) — a leaderboard of results collected this way, plus the drivers in
  [`../driver/`](../driver) that run the same DDP sessions in batch from a workstation.
- [`../ios/`](../ios) — the same `benchmark_model` on a locally attached iPhone.
