# Image Restoration with LiteRT — NAFNet deblur (on-device, fully-GPU)

An Android sample that runs [NAFNet](https://github.com/megvii-research/NAFNet) (Nonlinear Activation Free Network, ECCV 2022, MIT) image restoration end-to-end on device with the LiteRT `CompiledModel` API. NAFNet is a U-Net of **NAFBlocks** with **no activation functions at all** (SimpleGate = channel-split multiply), so the whole network is a clean CNN on the GPU delegate. This sample uses **NAFNet-GoPro-width32** (motion deblur): it restores a bundled blurry image at launch and any image picked from the gallery, showing input | restored.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| NAFNet-GoPro-width32 | image[1,3,256,256] (RGB [0,1]) → restored[1,3,256,256] | **GPU** |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — tflite-vs-torch corr **1.0**, **device-vs-torch corr 1.0** (numerically exact re-authoring). On a Pixel 8a (Tensor G3): **2179 / 2179** nodes on `LITERT_CL` (full GPU residency), ~42 ms @ 256×256, 38 MB.

## Re-authoring (litert-torch, parity corr 1.0)

Three numerically-exact GPU re-authorings. The headline is **SafeLayerNorm**: NAFNet's residual stream grows large (|x|≈175 at the bottleneck), so the LayerNorm channel reductions `Σ_c x` and `Σ_c (x−μ)²` (~15M) **overflow fp16 (max 65504)** on the Mali delegate (which computes in fp16 regardless of model dtype) — the output looks ~right (corr 0.98, input-dominated) but the learned residual is destroyed (corr 0.016) → a grid artifact. Fix: do the reductions in a down-scaled `x/S` domain (S=128) and rescale — exact. Plus the Simplified Channel Attention `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`, and the upsample `Conv2d(1×1)+PixelShuffle(2)` → Conv2d + depth-to-space `ZeroStuffConvT2d`. See [`conversion/`](conversion/).

## Run

1. Build the tflite with `conversion/build_nafnet.py` (downloads NAFNet-GoPro-width32.pth from HF `nyanko7/nafnet-models`), or get it from [litert-community/NAFNet-GoPro-width32-LiteRT](https://huggingface.co/litert-community/NAFNet-GoPro-width32-LiteRT).
2. Build/install the app, then push the model into its private storage:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-nafnet_fp16.tflite>
   ```
3. Launch the app. (The first launch fails with "Model not found" until the model is pushed.)

## Preprocessing

Center-crop to square, resize to 256×256, divide by 255 (RGB in [0,1]), NCHW planar. Output is the restored RGB image in [0,1].

Upstream: [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) (MIT).
