# LLaDA-8B Tensor API Implementation

Diffusion-LM denoise step for LLaDA-8B-Base, authored directly with the
[LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor)
(C++, no converter).

## This project demonstrates:

1.  One in-graph denoise step for a masked-diffusion LM: bidirectional
    backbone + ScatterNd-free top-k remasking (TopK → OneHot → Sum →
    SelectV2); the host only feeds ids back between steps.
2.  Loading the real LLaDA-8B-Base checkpoint: `--llada` selects the 8B
    architecture (32L, MHA 32/32, d_model 4096, rope 5e5, no qk-norm) and
    the sharded-safetensors loader maps the OLMo-style checkpoint names.
    Without `--llada`, a Qwen3-shaped GQA config runs smaller probes, and
    without `--weights` the step runs on synthetic weights.
3.  Composite options in an authored graph: `--attention=raw|sdpa`
    (`sdpa` keeps MHA shapes plain BSND and pre-folds GQA into batched
    MHA) and `--norms=raw|composite` (`odml.rms_norm`). For CPU-parity
    verification use `--attention=raw --norms=raw`.
4.  In-process weight quantization (`--weight_quant=int8|int4`; int4 =
    blockwise-32 with fp16 scales) and `--reuse_tflite` for on-device
    runs without the checkpoint.
5.  Explicit GPU precision control (`--gpu_precision=default|fp16|fp32`)
    — the Metal delegate default is fp16; label numbers accordingly.

## Prerequisites

1.  **Bazel**: installed via bazelisk (version pinned by `.bazelversion`).
2.  **Android NDK** 26+ for Android builds (tested with r27) and **ADB**
    for on-device runs.
3.  **Weights**: the GSAI-ML/LLaDA-8B-Base checkpoint (sharded
    safetensors) from Hugging Face, passed as a directory via `--weights`
    — not included in this repository. Synthetic-weight runs need no
    checkpoint.

## Build Instructions

All commands are run from the repository root (the workspace pulls LiteRT
`main` as `@litert_archive`).

```bash
# macOS (Apple silicon)
bazel build --config=macos_arm64 //models/llada/llada_8b/tensor_api:diffusion_main

# Android (the two extra flags mirror LiteRT's own android_arm64 config;
# needed until this repository's .bazelrc carries them)
ANDROID_NDK_HOME=<ndk> bazel build -c opt --config=android_arm64 \
  --incompatible_enable_cc_toolchain_resolution \
  --incompatible_enable_android_toolchain_resolution \
  //models/llada/llada_8b/tensor_api:diffusion_main
```

Runtime behavior was validated at LiteRT commit `a19d8fa` plus the
PR #8796 runner change; against current `main`, all sources compile and
the Android arm64 binary builds and links from this workspace. Please
tell us if anything else drifts and we will chase it.

## Run

```bash
# Synthetic smoke (no checkpoint needed)
diffusion_main --layers=4 --seq_len=256 --unmask_k=8 --accelerator=cpu

# Real LLaDA-8B weights
diffusion_main --llada --weights=<ckpt_dir> --accelerator=gpu \
  --gpu_precision=fp32
```

GPU runs on macOS need `libLiteRtMetalAccelerator.dylib` (shipped in the
target's runfiles) in the working directory.

## Measured highlights

*   Correctness: the C++ graph and an independent numpy reference produce
    identical per-step id trajectories over the full denoise schedule
    (real weights); int4 ids identical to fp32.
*   Full 32L LLaDA-8B runs on-device: Pixel 8a CPU int4 (6.29 GB model,
    memory-mapped) completes the schedule with ids identical to desktop.
*   M4 Max, 8L/vocab-4096 probe, ms/step: int8 81.6 vs int4 791.5 —
    XNNPACK blockwise-int4 fast kernels are GEMV-shape-only today, so
    full-sequence diffusion steps (M=64 GEMM) want per-channel int8 on
    CPU; int4 wins only for autoregressive decode shapes.
*   Metal fp32, S=1024: `--attention=sdpa` −3.8%/step (fp16 −2.5%),
    neutral at S=256.

## Findings ledger and evidence notes

The living findings ledger and per-campaign evidence notes are maintained
at
[john-rocky/litert-tensor-audio-examples](https://github.com/john-rocky/litert-tensor-audio-examples).
