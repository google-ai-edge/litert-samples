# Metric Depth with LiteRT — Metric3D v2 (on-device, fully-GPU)

An Android sample that runs [Metric3D v2](https://github.com/YvanYin/Metric3D) (CVPR/TPAMI 2024,
BSD-2-Clause) monocular **metric** (absolute, in-meters) depth end-to-end on device with the LiteRT
`CompiledModel` API. The app estimates depth on a bundled image at launch and on any image picked from the
gallery, rendering a depth colormap and the near/far metric range.

Unlike relative-depth models (MiDaS, Depth Anything), Metric3D predicts depth in **meters**. The DINOv2
ViT-S encoder **and** the RAFT-DPT decoder both run on the GPU delegate — no CPU/ONNX fallback.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| Metric3D v2 ViT-S | image[1,3,448,448] → depth[1,1,448,448] (meters) | **GPU** |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — tflite-vs-torch corr
**1.0**, depth corr **0.96** vs the original Metric3D pipeline (robust 0.96–0.98 across indoor 0.7–4 m /
mid 4–17 m / outdoor 11–200 m scenes). On a Pixel 8a (Tensor G3): **2447/2447** nodes on `LITERT_CL`
(full GPU residency), ~2.2 s one-time compile, **~44 ms / inference**, 78 MB.

The model outputs depth for a **canonical camera** (focal 1000 at the canonical resolution). For a
calibrated camera multiply by `fx / 1000` (the de-canonical transform); with no intrinsics the sample shows
the raw canonical-metric depth, which is already in meters.

## Re-authoring (litert-torch, parity corr 1.0)

Fixed 448×448 (= 32×32 patches = 1024 tokens). Encoder = the DINOv2 ViT-S suite (fused-QKV → 4D attention,
LayerScale folded into Linear, baked pos-embed). The RAFT-DPT decoder needs three fixes that **only the
on-device run reveals** (desktop fp16 stays at 0.9999):

- **Convex upsample → depth-to-space via `ZeroStuffConvT2d`** — 16 per-subpixel softmax-over-9-neighbour
  combines → `cat → ConvTranspose2d(96→6, k4, s4)` wrapped in `ZeroStuffConvT2d`. The naive "nearest-upsample
  + mask at the in-block offset" is exact on desktop but **0.57 on Mali** (`RESIZE_NEAREST` differs at
  non-stride positions); `ZeroStuffConvT2d` masks only stride-aligned positions and the conv kernel supplies
  the offset.
- **GELU → accurate tanh approximation** (POW-free). `x·sigmoid(1.702x)` collapses far-depth fidelity to
  **0.51** over the 0.1–200 m log-depth bins; tanh restores **0.96**.
- **`nn.ReLU(inplace=True)` mutates the DPT `ConvBlock` residual** (`relu(x)+convs`, not `x+convs`) —
  replicated exactly.

`Token2Feature`'s `ConvTranspose2d` → `ZeroStuffConvT2d`; `norm_normalize`'s `F.elu` → SELECT-free identity.
See [`conversion/`](conversion/).

## Run

1. Build the tflite with `conversion/build_m3d.py` (or get it from Hugging Face —
   [mlboydaisuke/Metric3D-v2-LiteRT](https://huggingface.co/mlboydaisuke/Metric3D-v2-LiteRT)).
2. Build/install the app, then push the model into its private storage:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-metric3d_fp16.tflite>
   ```
3. Launch the app. (The first launch fails with "Model not found" until the model is pushed.)

## Preprocessing

Center-crop to square, resize to 448×448, ImageNet normalize in 0–255 scale
`(px − [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]`, NCHW planar.

Upstream: [YvanYin/Metric3D](https://github.com/YvanYin/Metric3D) (BSD-2-Clause; DINOv2 backbone Apache-2.0).
