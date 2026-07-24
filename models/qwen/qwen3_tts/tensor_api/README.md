# Qwen3-TTS Tensor API Implementation

Qwen3-TTS-12Hz-0.6B-Base implemented by authoring the graphs directly with
the [LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor)
(C++, no converter), plus a C++ reference host loop. Companion code to the
"Audio-LM Loop Design Note (Tensor API)".

## This project demonstrates:

1.  Authoring a full autoregressive TTS decode step as one Tensor API
    graph: `prefill`/`decode` signatures sharing weights, fixed-size KV
    caches as signature I/O (DynamicUpdateSlice at a runtime position,
    zero-copy output→input rebind), codebook-0 pick + 15-group code
    predictor + next-frame feedback aggregation in-graph — one signature
    call per 12.5 Hz frame. Both code predictor forms are build-time
    options (re-forward unroll for GPUs, KV-cached incremental for CPUs;
    ids bit-identical).
2.  The full speech_tokenizer decoder (RVQ + transformer + vocoder) as one
    Tensor API graph, with the causal 64+25 chunk schedule on the host.
3.  Composite adoption in an authored graph: `odml.runtime_bmm` (talker
    attention, `--attention=rbmm`) and `odml.rms_norm`
    (`--norms=composite`); `attn_bench` compares raw ops vs
    `odml.scaled_dot_product_attention` vs the `odml.runtime_bmm` +
    `odml.cache_update` pair at identical inputs, including multi-step
    feedback-loop runs (LiteRT PR #8796).
4.  Weight quantization paths: fp32, int8 per-channel, int4 blockwise-32,
    and pre-quantized (GPTQ) companion files.
5.  Sampling through a host-staged additive bias input: suppression, EOS
    gating, min-new-tokens, and exact temperature sampling (Gumbel
    noise-as-input) — no RNG op needed in-graph.
6.  Bit-exact verification against independent numpy references on the
    real checkpoint, plus an end-to-end text → wav smoke driver.

## Directory layout

*   `talker_step/` — talker decode-step graph, weights loader, host
    reference loop (`talker_main.cc`), and the attention micro-benchmark
    (`attn_bench.cc`).
*   `codec_decode/` — codec decoder graph, weights loader, and host
    chunking driver (`codec_main.cc`).
*   `verify/` — numpy references for both graphs, the GPTQ quantizer
    (emits the exact blockwise-32 layout the C++ loader consumes), speaker
    x-vector enrollment, and `audio_smoke.py` (text → wav end to end).

## Prerequisites

1.  **Bazel**: installed via bazelisk (version pinned by `.bazelversion`).
2.  **Android NDK** 26+ for Android builds (tested with r27) and **ADB**
    for on-device runs.
3.  **Python 3 + numpy** for `verify/` (torch additionally for the GPTQ
    and speaker-enrollment scripts).
4.  **Weights**: the Qwen3-TTS-12Hz-0.6B-Base checkpoint (safetensors)
    from Hugging Face — not included in this repository. `talker_main`
    runs without `--weights` using synthetic weights under the real
    tensor names (useful for smoke tests); `codec_main` requires the
    checkpoint.

## Build Instructions

All commands are run from the repository root (the workspace pulls LiteRT
`main` as `@litert_archive`).

```bash
# macOS (Apple silicon)
bazel build --config=macos_arm64 //models/qwen/qwen3_tts/tensor_api/talker_step:talker_main
bazel build --config=macos_arm64 //models/qwen/qwen3_tts/tensor_api/codec_decode:codec_main

# Android (the two extra flags mirror LiteRT's own android_arm64 config;
# needed until this repository's .bazelrc carries them)
ANDROID_NDK_HOME=<ndk> bazel build -c opt --config=android_arm64 \
  --incompatible_enable_cc_toolchain_resolution \
  --incompatible_enable_android_toolchain_resolution \
  //models/qwen/qwen3_tts/tensor_api/talker_step:talker_main
```

Runtime behavior was validated at LiteRT commit `a19d8fa` plus the
PR #8796 runner change; against current `main`, all sources compile and
the Android arm64 binaries build and link from this workspace. Please
tell us if anything else drifts and we will chase it.

## Run

```bash
# Talker: one signature call per frame, benchmark + optional dumps
talker_main --layers=28 --weights=<ckpt>/model.safetensors \
  --accelerator=gpu --gpu_buffer_storage=buffer

# Codec: codes -> 24 kHz waveform with production-style chunking
codec_main --weights=<ckpt>/speech_tokenizer/model.safetensors \
  --frames=128 --accelerator=cpu --dump_wav=/tmp/out.f32

# End to end (text -> wav), driving both binaries:
python3 verify/audio_smoke.py --help
```

GPU runs on macOS need `libLiteRtMetalAccelerator.dylib` (shipped in the
target's runfiles) in the working directory. The Metal delegate's default
compute precision is fp16; set `--gpu_precision` explicitly when comparing
against fp32 references. On Metal use `--gpu_buffer_storage=buffer` for
max_seq ≥ 1024 (texture storage silently falls back to CPU).

## Measured highlights

Frame budget = 80 ms (12.5 Hz).

*   Pixel 8a CPU, GPTQ int4 blockwise-32, KV-cached code predictor:
    99.6 ms/frame = RTF 1.24 (thermal-controlled).
*   M4 Max Metal, GPTQ int4 + composite stack (runtime_bmm + rms_norm):
    17.3 ms/frame = RTF 0.216; fp16 raw→composite 22.3 → 20.4 ms (−8.7%).
*   Codec on GPU: 17.6 ms (Mali) / 47.4 ms (Metal fp16) per 89-frame
    window.
*   KV-cache rebind by buffer handle is zero-copy and
    in-place-DUS-eligible; naive copy-rebind costs +24%/frame (M4 Max) /
    +35%/frame (Pixel 8a).
*   Quantization that survives listening tests: blockwise int4 (GPTQ ≥
    RTN); per-channel int8 collapses in long rollouts even where
    short-horizon id checks look healthy — gate quant recipes on full
    listenable rollouts.

## Findings ledger and evidence notes

The living findings ledger (delegate/composite observations with repro
commands), the standalone repro tools, and the per-campaign evidence notes
are maintained at
[john-rocky/litert-tensor-audio-examples](https://github.com/john-rocky/litert-tensor-audio-examples).
