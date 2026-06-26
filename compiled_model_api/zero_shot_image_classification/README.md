# LiteRT Zero-Shot Image Classification Sample

This directory contains an Android zero-shot image classification sample demonstrating how to use
LiteRT (Google's runtime for TensorFlow Lite) with the Compiled Model API. Unlike standard image
classification, which is restricted to a fixed label set baked into the model, this sample classifies
the camera feed against an **open vocabulary** of natural-language labels — no task-specific training
required — and lets you switch at runtime between two state-of-the-art CLIP-style image towers:

-   **[Perception Encoder](https://ai.meta.com/research/publications/perception-encoder/)
    (PE-Core-B16-224, Meta 2025)** — a RoPE ViT, 1024-d embedding.
-   **[SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) (ViT-B/16, Google 2025)** —
    768-d embedding.

## Overview

Each model learns a shared embedding space for images and text. The on-device **image encoder** maps
a frame to an embedding; each candidate label is turned into a text embedding by the model's **text
encoder** (`"a photo of a {label}"`) and the predicted label is the one with the highest cosine
similarity to the image embedding.

Only the image encoder runs on device. Both are Vision Transformers converted with
[`litert-torch`](https://github.com/google-ai-edge/ai-edge-torch) and running **fully on the Compiled
Model API GPU delegate** (LITERT_CL, all ops, no CPU fallback) — and verified **numerically correct on
the GPU**, not just resident (see GPU compatibility notes). The text embeddings for the candidate
labels are **pre-computed on the host** and bundled as small binary assets (one per model), so no text
model runs at runtime. This makes the on-device cost identical to a single ViT forward pass while
keeping the label set fully customizable.

## Available Implementations

### 1. kotlin_cpu_gpu

A standard implementation utilizing the Compiled Model API with support for CPU and GPU delegates.

**Features:**
-   **Zero-shot, open-vocabulary**: Classifies into 96 diverse labels (animals, vehicles, food,
    scenes, …) with no retraining. Swap `labels.txt` / the `text_embeddings_*.bin` to change the set.
-   **Two SOTA models, switchable at runtime**: pick PE-Core or SigLIP 2 in the settings sheet.
-   **Real-time Inference**: Classifies objects from the camera feed (Camera and Gallery tabs).
-   **Hardware Acceleration**: Switch between CPU and GPU delegates at runtime.
-   **Vision Transformer fully on GPU**: both image encoders run at full GPU residency.
-   **Jetpack Compose**: Modern Android UI toolkit.

## Technical Details

### Model Architecture
-   **Task**: Zero-shot image classification (open-vocabulary).
-   **Models** (converted with `litert-torch`, FP16, fetched from Hugging Face at build time):
    -   **PE-Core-B16-224** (`pe_core_base_224_fp16.tflite`, ViT-B/16 + RoPE, 94M, 1024-d) —
        [litert-community/PE-Core-base-patch16-224](https://huggingface.co/litert-community/PE-Core-base-patch16-224).
    -   **SigLIP 2 (ViT-B/16, 224)** (`siglip2_base_224_fp16.tflite`, 93M, 768-d) —
        [litert-community/SigLIP2-base-patch16-224](https://huggingface.co/litert-community/SigLIP2-base-patch16-224).
-   **Input** (both): `[1, 3, 224, 224]` NCHW float32. Center-cropped to square, resized to 224×224,
    and normalized to `[-1, 1]` (`(pixel/255 - 0.5) / 0.5`).
-   **Output**: `[1, 1024]` (PE-Core) or `[1, 768]` (SigLIP 2) L2-normalized image embedding.
-   **Scoring**: cosine similarity against the pre-computed per-model text embeddings, followed by a
    temperature-scaled (logit scale ≈ 100) softmax.
-   **Format**: TensorFlow Lite (`.tflite`).
-   **Performance** (Pixel 8a, Mali-G615 GPU, full residency): PE-Core ~66 ms, SigLIP 2 ~60 ms / image.

### GPU compatibility notes
Making a RoPE ViT image tower run fully on the ML Drift GPU delegate **and produce correct output on
device** required four verbatim (weights-exact, output correlation ≈ 1.0 vs. PyTorch) model-side
rewrites — the first three for residency, the last for numerical correctness (full GPU residency does
**not** imply a correct result):
1.  **Fused-qkv → 4-D manual attention** — the fused `qkv` reshape emits a 5-D head-split the GPU
    delegate rejects; decompose into separate q/k/v projections. Self-attention uses
    `scaled_dot_product_attention`, whose lowering keeps the batch-matmul 3-D with a materialized
    transpose (both required for residency).
2.  **Interleaved 2D-RoPE → rotate-half** — the interleaved rotary uses a strided slice that lowers
    to `GATHER_ND` (GPU-banned); bake an even→odd channel permutation into the q/k weights (preserves
    q·k exactly) and apply the gather-free rotate-half form with constant cos/sin.
3.  **Attention-pool single-query attention → broadcast-multiply + reduce-sum** — the pooling query
    is a constant latent, so a batch-matmul there is `const @ non-const` (rejected, and the
    `const-RHS` reorder is mis-computed on device); `(q·k).sum` + softmax + `(attn·v).sum` is exact
    and GPU-correct.
4.  **Overflow-safe LayerNorm** — the delegate computes the LayerNorm variance reduction in fp16 even
    for an fp32 graph; deep-ViT massive activations make `sum((x-mean)²)` exceed the fp16 max (65504),
    corrupting normalization (output correlation collapses to ~0.28 over 12 blocks while still
    reporting full residency). Scaling by 1/32 before squaring keeps the sum in range.

**SigLIP 2** needs the same set **minus** the RoPE step (it has no rope and no class token) — rewrites
1, 3 and 4. The overflow-safe LayerNorm (#4) is a general deep-ViT-on-GPU fix, not model-specific.

### Key Dependencies
-   **LiteRT** (`com.google.ai.edge.litert:litert:2.1.3`)
-   **Jetpack Compose** (UI)
-   **CameraX** (Camera feed)

### Architecture Components
-   **`ZeroShotImageClassificationHelper`**: Loads the labels and pre-computed text embeddings,
    initializes the Compiled Model (CPU/GPU), runs the image encoder, and scores labels by cosine
    similarity.
-   **`MainActivity`**: Setup of the main screen and UI components.
-   **`CameraScreen`**: Composable for camera preview and the classification overlay.
-   **`MainViewModel`**: Manages UI state and communicates between the UI and the Helper.

## Getting Started

1.  Open `kotlin_cpu_gpu/android` in Android Studio.
2.  Both image encoders (`pe_core_base_224_fp16.tflite`, `siglip2_base_224_fp16.tflite`) are
    downloaded into `app/src/main/assets/` at build time by `download_model.gradle`. The labels and
    per-model pre-computed text embeddings (`labels.txt`, `text_embeddings_pecore.bin`,
    `text_embeddings_siglip2.bin`) are bundled in `app/src/main/assets/`.
3.  Build and run the application on an Android device.
4.  Point the camera at an object (or pick an image from the Gallery tab).
5.  Observe the predicted labels and confidence scores.
6.  Use the settings sheet to switch the model (PE-Core / SigLIP 2) and the CPU / GPU delegate.

### Customizing the label set
Regenerate `labels.txt` and the per-model `text_embeddings_*.bin` with your own labels using each
model's text encoder (`open_clip`'s `PE-Core-B-16` and `ViT-B-16-SigLIP2`, prompt
`"a photo of a {label}"`, L2-normalized), then replace the files in `app/src/main/assets/`. The label
count must match between `labels.txt` and each embeddings file.
