# RTMPose-s — top-down 2D human pose (on-device, fully-GPU)

A second pose model for this category: **RTMPose-s** (mmpose, CSPNeXt backbone + RTMCC/SimCC head), the SOTA real-time **top-down** estimator. Where the OpenPose sample in the parent directory is bottom-up (multi-person, PAF), RTMPose runs on a single centered person and predicts 17 COCO keypoints with a SimCC head — end-to-end on the LiteRT `CompiledModel` GPU.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| RTMPose-s | image[1,3,256,192] → simcc_x[1,17,384], simcc_y[1,17,512] | **GPU** |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — tflite-vs-torch corr **1.0**, device-vs-torch SimCC corr **0.999**, keypoints within **0.3 px**. On a Pixel 8a (Tensor G3): **256 / 256** nodes on `LITERT_CL` (full GPU residency), **~4 ms**, **11.1 MB** fp16. The SimCC argmax decode + skeleton run in the app.

## Two numerically-exact, on-device-only re-authorings

Both pass the desktop op-check and report full LITERT_CL residency, yet the device output was wrong until fixed (*residency ≠ correctness*):

1. **`ScaleNorm` (RMS norm) fp16 overflow → all-zero head.** The RTMCC `ScaleNorm` input reaches ≈ |274|, so its channel `Σ x²` ≈ 3.6M **overflows fp16 (max 65504)** on the Mali delegate (which reduces in fp16 even for an fp32 graph) → `norm = ∞` → `x/∞ = 0` → the whole head collapses to zero. Fix: scale `x` down by S=64 **before** squaring, then rescale (math-identical) — a SafeRMSNorm.
2. **GAU attention `act@act` BMM → broadcast-reduce.** The Gated Attention Unit's `q@kᵀ` and `kernel@v` are activation×activation batch-matmuls the Mali delegate mis-computes; at K=17 tokens the exact replacement is `(q[:,:,None,:]·k[:,None,:,:]).sum(-1)`.

See [`conversion/`](conversion/) for the full recipe (including the lightweight mmpose stub setup so `mmcv-lite` suffices — no compiled ops).

## Run

1. Build the tflite with `conversion/build_rtmpose.py` (downloads the RTMPose-s checkpoint), or use [litert-community/RTMPose-s-LiteRT](https://huggingface.co/litert-community/RTMPose-s-LiteRT).
2. Build/install the app and push the model:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-rtmpose_s_fp16.tflite>
   ```
3. Launch. (The first launch fails with "Model not found" until the model is pushed.)

**Preprocessing**: center-crop to 3:4, resize to 192×256, ImageNet 0-255 normalize (mean [123.675, 116.28, 103.53], std [58.395, 57.12, 57.375]), NCHW planar. Top-down — expects one roughly-centered person.

Upstream: [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) RTMPose-s (Apache-2.0).
