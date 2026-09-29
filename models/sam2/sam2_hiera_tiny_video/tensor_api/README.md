# SAM 2.1 Video Tracking — Tensor API Implementation

SAM 2.1 Hiera-Tiny video tracking implemented by authoring the graphs
directly with the
[LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor)
(C++, no converter), plus a C++ reference host loop. Companion to the
`converted/` recipe in the parent directory (the litert-torch export of
the same four per-frame graphs) and to
[litert-examples](https://github.com/google-ai-edge/litert-samples/edit/main/models/sam2/sam2_hiera_tiny_video/tensor_api/)
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
    decoder) plus its standalone 512 sample (`sam2_main.cc`, which also
    runs the frame pre/post-processing as `CreateLambdaRunner` graphs)
    and PyTorch parity script.
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

## Graph-level optimizations

The graph code applies these rewrites at build time. None of them changes
the math: outputs match the direct construction up to fp32 rounding, and
the CPU fp32 parity results are unchanged (see below).

*   **Image encoder (Hiera).** The 1/sqrt(d) attention scale is folded
    into the q rows of `qkv`. Windows stay 4-D (`[nH, nW*ws, ws, C]`),
    which drops the reshapes around `FullyConnected` and the q-pool
    `MaxPool2D`. `proj` runs after window unpartition. Single-head blocks
    skip the head transposes. When a map needs window padding, `qkv` runs
    on the unpadded map without its bias, and the bias is added after
    padding. Padded tokens therefore still carry `qkv = bias`, as in Hiera,
    which zero-pads the block input before `qkv`.
*   **Image mask decoder.** The attention scale is folded into `q_proj`.
*   **Memory attention.** The rotate-half RoPE sign is baked into the
    `sin` table, so the rotation is `x*cos + swap_halves(x)*sin`, with no
    `Neg` op. The attention scale is folded into the q projection.
*   **Video mask decoder.** The attention scale is folded into q. The
    duplicate `keys + key_pe` add is removed. The no-memory and no-mask
    biases are combined before the full-map add. The mask head's
    `BatchMatMul` uses rank-4 operands.
*   **Shared baked constants.** `s2v::ConstCache` is passed to every
    `Build*` call that goes into one `ModelFactory`. Constants derived from
    the weights at build time (pre-scaled projections, sign-baked RoPE
    tables, the dense PE grid) are then stored once, not once per
    signature.

| Signature (CPU fp32 flatbuffer) | Before | After |
|---|---|---|
| video `encode` (1024) / image `encode_image` (512) | 594 / 595 ops | 521 / 522 ops |
| image `decode_mask` | 276 ops | 269 ops |
| `memcond7` / `memcond2` | 346 / 346 ops | 313 / 313 ops |
| `decode` | 274 ops | 266 ops |
| `sam2_video.tflite` (1024, fp32) | 212.6 MB | 207.4 MB |

Parity after these rewrites (CPU fp32): `sam2_image/verify/sam2_torch_ref.py`
at 512 reports PARITY: PASS, with all correlations at 1.000000 and IoU scores
identical to the reference. `verify_video_1024.py compare` on the 10-frame
clip gives min mask-IoU 1.0000 and max|dmask| 0.008 for both bank sizes,
the same as before.

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
