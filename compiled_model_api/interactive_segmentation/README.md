# LiteRT Interactive Segmentation Sample (SAM 2.1)

<p align="center"> <img src="https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Mask-Decoder/resolve/main/demo.gif" width="300" alt="Tap to segment demo (SAM 2.1 on LiteRT)"> </p>

This directory contains an Android **promptable "tap to segment"** sample demonstrating how to use LiteRT (Google's runtime for TensorFlow Lite) with the Compiled Model API. Tap any object in a photo and the app produces its segmentation mask in real time, using **[Segment Anything 2.1](https://huggingface.co/facebook/sam2.1-hiera-tiny) (Hiera-Tiny, Meta 2024)** split into the two halves of the SAM image path: a heavy image encoder on the **GPU** and a small mask decoder per tap.

## Overview

SAM 2 separates a **heavy image encoder** (run once per image) from a **lightweight mask decoder** (run once per click). This sample mirrors that split with two LiteRT models converted with [`litert-torch`](https://github.com/google-ai-edge/ai-edge-torch):

1.  **Image encoder** — runs **once** when you load a photo (~tens of ms). A Hiera hierarchical ViT + FPN that turns the RGB image into a multi-scale feature pyramid. This sample's encoder is the **decoder-ready** variant: it folds the decoder's `conv_s0` / `conv_s1` projections and the `no_memory` embedding into the graph, so its outputs feed the decoder directly.
2.  **Mask decoder** — runs **per tap** (a few ms). A 2-layer two-way (token↔image) transformer + a mask up-sampler that turns the cached features plus a point prompt into 3 candidate masks and their predicted IoU; the app overlays the highest-IoU one.

Both models are **GPU-clean** (LITERT_CL — no banned ops, no >4-D tensors, full residency: encoder 867/867, decoder 425/425 nodes) and **both run on the GPU**. Getting the decoder there took one non-obvious fix: written with the batch dim collapsed (rank-3 attention) it delegated fully and matched PyTorch on the host, yet returned silently wrong masks on device — see "[Residency ≠ correctness](#residency--correctness-the-rank-3-attention-miscompute)" below. The tiny **prompt encoder** (point → token, a sin/cos positional encoding) is computed **on the host** so the decoder graph stays sin/cos-free; its constants are bundled as a small binary asset (`prompt_encode_const.bin`). On CPU/desktop the two FP16 models reproduce the PyTorch reference masks at cosine ≈ **0.999999** (binary mask IoU ≈ **0.9996**); on the Pixel 8a GPU the end-to-end pipeline matches the CPU reference at correlation ≈ **0.9998**.

## Available Implementations

### 1. kotlin_cpu_gpu

A standard implementation utilizing the Compiled Model API with support for CPU and GPU delegates.

**Features:**
-   **Tap to segment**: load a photo, tap any object, get its mask instantly.
-   **Encode once, decode per tap**: the heavy encoder runs a single time; each tap only runs the light decoder, so subsequent clicks are near-instant.
-   **Multimask + IoU**: the decoder returns 3 candidate masks; the app shows the highest predicted IoU and the value in the settings sheet.
-   **Hardware Acceleration**: switch both models between CPU and GPU delegates at runtime.
-   **Both models GPU-clean**: full LITERT_CL residency (encoder 867/867, decoder 425/425), and both are numerically correct on the GPU.
-   **Jetpack Compose**: modern Android UI toolkit.

> Input is currently the system **Photo Picker** (Gallery). A live-camera "capture a frame, then tap" path is a natural follow-up.

## Technical Details

### Model Architecture
-   **Task**: promptable interactive segmentation (SAM 2 image path).
-   **Models** (converted with `litert-torch`, FP16, fetched from Hugging Face at build time):
    -   **Image encoder** (`sam2_image_encoder_fp16.tflite`, Hiera-Tiny + FPN, ~80 MB) — [litert-community/SAM2.1-Hiera-Tiny-Image-Encoder](https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Image-Encoder).
    -   **Mask decoder** (`sam2_mask_decoder_fp16.tflite`, two-way transformer, ~17 MB) — [litert-community/SAM2.1-Hiera-Tiny-Mask-Decoder](https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Mask-Decoder).
-   **Encoder I/O**: in `[1, 3, 1024, 1024]` NCHW float32 (resized to 1024², ImageNet-normalized); out `image_embeddings [1,256,64,64]`, `feat_s1 [1,64,128,128]`, `feat_s0 [1,32,256,256]`.
-   **Decoder I/O**: in `image_embeddings`, `sparse [1,2,256]`, `feat_s1`, `feat_s0` (bound **by index** in that order — `image_embeddings` and `feat_s1` share an element count); out `pred_masks [1,3,256,256]` (logits) and `iou_scores [1,3]`.
-   **Host prompt encode**: a positive click `(x, y)` in 1024-space → `c = 2·((x,y)+0.5)/1024 − 1`, `coord = 2π·(c · posmat)`, `token0 = [sin(coord), cos(coord)] + point_embed[1]`, `token1 = not_a_point` → `sparse = [[token0, token1]]`. Constants in `prompt_encode_const.bin` (768 little-endian floats: `posmat[2,128]`, `point_embed[1] 256`, `not_a_point 256`). Matches the upstream module to ~3.7e-7.
-   **Postprocess**: pick `argmax(iou)`, upsample the 256² logits to the image, threshold at 0.

### GPU compatibility notes
Making SAM 2 run fully on the ML Drift GPU delegate required verbatim (weights-exact) model-side rewrites — **no converter patch**. For the **encoder**: ≤4-D window partition/attention (the 6-D window reshape and 5-D fused-qkv are rejected), 3-D batched SDPA, baked positional embeddings, and an overflow-safe LayerNorm. For the **decoder**:
1.  **Two-way attention (×7) → 3-D batched SDPA** `[heads, N, d]` (a 4-D SDPA emits a `BROADCAST_TO`).
2.  **Mask up-sampler `ConvTranspose2d` (×2) → zero-stuff + `Conv2d`** — `TRANSPOSE_CONV` is rejected on device; the zero-insertion + convolution identity is numerically exact (not a bilinear approx).
3.  **Mask head kept ≤4-D** — the upstream `[1,1,4,256,256]` 5-D mask tensor is collapsed (batch and point-batch are 1).
4.  **Overflow-safe LayerNorm (×9)** — scale-before-square, the general deep-net-on-GPU fp16 fix.
5.  **Baked constants** — `image_positional_embeddings` and the no-mask dense prompt are baked buffers.
6.  **Static multimask slice** — `multimask_output=True` takes a fixed `[1:]` slice, avoiding the dynamic-stability `argmax` / `gather` / `where`.

Result: `banned ops = NONE`, `>4-D tensors = 0` for both models, full LITERT_CL residency on device; on CPU, end-to-end masks match PyTorch at cosine ≈ 0.999999.

### Residency ≠ correctness: the rank-3 attention miscompute

The decoder's first build wrote its attention with the **batch dim collapsed** — `q/k/v` shaped `[heads, N, d]` (rank 3). It compiled, delegated fully (358/358 LITERT_CL nodes) and matched PyTorch on the host, yet on the Pixel 8a GPU it returned **silently wrong masks**: correlation **0.265** against the CPU output, and a center-face tap that the CPU decoder segments at IoU ≈ 0.62 collapsed to IoU ≈ 0.10 with the mask on the background.

Device A/B ruled out the two obvious suspects: it is **not fp16 precision** (forcing fp32 GPU compute still gives correlation 0.473) and **not LayerNorm** (plain and overflow-safe LN give the same wrong result). Bisecting on device isolated the **attention rank** — making only the attention rank-4 fixes it, while the mask head's rank-2 matmul is innocent.

Keeping the leading batch dim (`[1, heads, N, d]`) is numerically identical on the host and restores correlation **0.9998** / binary-IoU **0.999** on the GPU, while running **~20 % faster** (6.8 ms vs 8.5 ms per tap). The encoder's rank-3 SDPA *is* GPU-correct on the same device, so a healthy sibling graph proves nothing: op gates, full delegation and desktop parity all pass while the output is garbage. Only a numeric GPU-vs-CPU check on device catches this.

### Key Dependencies
-   **LiteRT** (`com.google.ai.edge.litert:litert:2.1.3`)
-   **Jetpack Compose** (UI)

### Architecture Components
-   **`Sam2SegmentationHelper`**: loads both Compiled Models (CPU/GPU) and the prompt constants, runs the encoder once per image, does the host-side prompt encode, runs the decoder per tap, and builds the best-mask overlay bitmap.
-   **`MainActivity`**: main screen, image picker, and the settings sheet (CPU/GPU, timings, IoU).
-   **`SegmentationScreen`**: shows the image, maps a tap to 1024² model space, draws the mask overlay.
-   **`MainViewModel`**: manages UI state and bridges the UI and the Helper.

## Getting Started

1.  Open `kotlin_cpu_gpu/android` in Android Studio.
2.  Both models (`sam2_image_encoder_fp16.tflite`, `sam2_mask_decoder_fp16.tflite`) are downloaded into `app/src/main/assets/` at build time by `download_model.gradle`. The prompt-encoder constants (`prompt_encode_const.bin`) are bundled in `app/src/main/assets/`.
3.  Build and run the application on an Android device.
4.  Tap the **+** button to pick a photo. The encoder runs once (a spinner shows while it does).
5.  Tap any object in the image to segment it; the highest-IoU mask is overlaid.
6.  Use the settings sheet to switch the CPU / GPU delegate.

## License

The models are conversions of `facebook/sam2.1-hiera-tiny` (Meta), Apache-2.0. SAM 2 was trained on the SA-1B / SA-V datasets; see the [SAM 2 release](https://github.com/facebookresearch/sam2) for dataset and PII details.
