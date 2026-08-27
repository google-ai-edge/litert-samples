# SAM 2.1 Video Tracking — Tensor API Implementation

SAM 2.1 Hiera-Tiny video tracking implemented by authoring the graphs
directly with the
[LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor)
(C++, no converter), plus a C++ reference host loop. Companion to the
`converted/` recipe in the parent directory (the litert-torch export of
the same four per-frame graphs) and to
[litert-tensor-vision-examples](https://github.com/john-rocky/litert-tensor-vision-examples)
(the same code as a LiteRT `tensor/examples/` overlay, with the findings
ledger).

## This project demonstrates:

1.  The full SAM 2.1 video stack authored as five signatures in ONE
    flatbuffer sharing weights: `encode` (Hiera encoder at the native
    1024, emitting the raw top-level feature map plus the two high-res
    skips), `memcond7` / `memcond2` (memory attention over a fixed bank
    of 7 or 2 spatial memory slots + 64 object-pointer tokens, unused
    entries masked additively — numerically identical to the reference's
    variable-length bank), `decode` (video mask decoder: sparse prompt
    and a no-memory scalar as inputs; all four mask tokens, IoU scores,
    object pointers and the object score out), `memorize` (mask
    downsampler + ConvNeXt fuser memory encoder with an occlusion
    input).
2.  Rotary position embedding without a RoPE op: SAM2's
    pairwise-interleaved RoPE is turned into the rotate-half form by
    permuting the q/k projection rows at weight-export time (q'.k' ==
    q.k exactly), with deinterleaved cos/sin tables baked as constants —
    no new op class anywhere in the video stack.
3.  A per-frame host loop (`sam2v_main.cc`) that mirrors the numpy
    specification in `verify/verify_video_1024.py`: bank bookkeeping,
    temporal position rows, object-pointer sine encoding, best-mask
    pick, no-object handling and mask_for_mem construction.
4.  Layer-for-layer verification tooling: an end-to-end chained-state
    parity harness against the `transformers` `Sam2VideoModel` streaming
    reference, per-graph isolation probes (each signature run on inputs
    captured from the HF modules themselves), and a block-by-block
    encoder mirror.

## Directory layout

*   `sam2_image/` — the image-path encoder/decoder library the video
    graphs build on (Hiera encoder, prompt encoder in-graph, mask
    decoder) plus its standalone 512 sample (`sam2_main.cc`) and PyTorch
    parity script.
*   `sam2_video/` — the video graphs (`sam2v_graph.cc`), the video-stack
    weight loader (`sam2v_weights.cc`), the host tracking loop
    (`sam2v_main.cc`), and `verify/` (fp32 weight export from
    `facebook/sam2.1-hiera-tiny`, HF streaming reference, per-frame
    compare, isolation probes).

## Prerequisites

1.  **Bazel** via bazelisk (version pinned by `.bazelversion`).
2.  **Python 3** with torch, transformers >= 5, safetensors, numpy and
    Pillow for `verify/`.
3.  **Weights**: `facebook/sam2.1-hiera-tiny` from Hugging Face — not
    included. `sam2_video/verify/export_weights_1024.py` produces the
    single fp32 safetensors file both binaries consume (the video stack
    included, conv layouts pre-permuted, RoPE bake self-checked in-run).
    `sam2v_main` runs without `--weights` using synthetic weights (shape
    and routing smoke test only).

## Build Instructions

All commands are run from the repository root (the workspace pulls LiteRT
`main` as `@litert_archive`).

```bash
# macOS (Apple silicon)
bazel build --config=macos_arm64 \
  //models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video:sam2v_main

# The image-path 512 sample
bazel build --config=macos_arm64 \
  //models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_main
```

Runtime behavior was validated at LiteRT commit `a19d8fa` plus the
PR #8796 runner change.

## Run

```bash
# One-time: weights + synthetic clip + HF reference
python3 sam2_video/verify/export_weights_1024.py --out sam2_tiny_1024_video.safetensors
python3 sam2_video/verify/verify_video_1024.py clip
python3 sam2_video/verify/verify_video_1024.py ref

# Track 10 frames on Metal, dump per-frame outputs, compare
sam2v_main --weights=sam2_tiny_1024_video.safetensors \
  --frames_file=<out>/frames.f32 --frames=10 --nmm=7 \
  --accelerator=gpu --gpu_precision=fp16 --gpu_buffer_storage=buffer \
  --dump_dir=<dir> [--bench_loops=3]
python3 sam2_video/verify/verify_video_1024.py compare --dump_dir=<dir> --nmm 7
```

GPU runs on macOS need `libLiteRtMetalAccelerator.dylib` (shipped in the
target's runfiles) in the working directory. The Metal delegate's default
compute precision is fp16; set `--gpu_precision` explicitly when
comparing against fp32 references, and use `--gpu_buffer_storage=buffer`
(texture storage silently falls back to CPU on these tensor sizes).

## Measured highlights

*   Parity vs the HF streaming reference (fp32, 10-frame synthetic clip,
    chained state): **min mask-IoU 1.0000** on CPU fp32 and Metal fp32
    for both bank sizes; 0.9950 at Metal fp16 over the chained loop.
*   M4 Max Metal fp16, per tracked frame end to end (host loop
    included): **94 ms** with the 7-slot bank, **64 ms** with 2 slots
    (encode 33.7 ms, memory attention 53.5 / 23.2 ms, decode 1.7 ms,
    memory encoder 1.4 ms). fp32: 117 / 78 ms. CPU fp32: 1505 ms.
*   The memory bank is host-side per-frame signature I/O (the memory
    attention itself is in-graph). Moving the bank in-graph as signature
    state (`odml.cache_update` + the PR #8796 feedback-loop runner) is
    the intended next step; it is currently blocked on Metal by the
    second-Run input-buffer re-registration failure ("The given buffer
    type is not supported") that also blocks the audio-side cache
    adoption.
