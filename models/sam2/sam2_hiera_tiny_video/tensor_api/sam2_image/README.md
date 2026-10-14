# SAM 2.1 Hiera-Tiny (`512x512`, `fp16`) Image Pipeline & Cross-Platform Benchmarks

This directory contains the C++ LiteRT Tensor API implementation of the SAM 2.1 Hiera-Tiny image segmentation pipeline (`encode_image` and `decode_mask` signatures, plus `CreateLambdaRunner` pre- and post-processing graphs) along with cross-platform benchmark harnesses for **macOS**, **iOS**, and **Android** (and the macOS Python MLX and iOS MLX-Swift baselines).

All benchmarks use the same deterministic fixture and timing protocol:
* **Input**: `512x512` synthetic image containing a centered white circle (`r = 120`, ImageNet-normalized) and a positive point prompt at `(256, 256)`.
* **Precision**: `fp16` on GPU (`fp32` on CPU).
* **Protocol**: 5 full-pipeline (`encode_image` + `decode_mask`) warmup runs followed by 20 timed runs per stage, reporting the median latency in milliseconds and verifying `iou_scores`, `object_score`, and the foreground pixel count (`2829 / 16384` on mask `[2]`).

---

## How the Benchmarks Work

### 1. LiteRT Tensor API (`sam2_main.cc` on macOS & Android, `ios/sam2_litert` on iOS)
1. **In-Memory Graph Authoring & Serialization (`sam2_main.cc`)**:
   - Loads `sam2_tiny_512.safetensors` via `sam2::LoadWeights` (`sam2_weights.cc`).
   - Authors the Hiera-Tiny image encoder (`sam2::BuildEncoder`) and prompt + mask decoder (`sam2::BuildDecoder`) directly with the C++ Tensor API (`sam2_graph.cc`), and serializes both signatures (`encode_image` and `decode_mask`) into a single two-signature `.tflite` flatbuffer (`sam2_image.tflite`).
2. **GPU Compilation & Pre/Post-Processing (`sam2_main.cc`)**:
   - Creates a shared `litert::Environment` and configures `litert::Options` with `HwAccelerators::kGpu`, `GpuOptions::Precision::kFp16` (`--gpu_precision=fp16`), `GpuOptions::BufferStorageType::kBuffer` (`--gpu_buffer_storage=buffer`), and `gpu_options->EnableBenchmarkMode(true)`.
   - Compiles the two-signature model on the Metal delegate (macOS) or OpenCL delegate (Android) via `LitertDynamicRunner::Create`, alongside the pre-processing (`CreatePreprocessRunner`: `RGBA8888 -> RGB -> fp32 -> 512x512 bilinear resize -> ImageNet normalize`) and post-processing (`CreatePostprocessRunner`: `mask logits -> frame-size bilinear upsample -> threshold`) `CreateLambdaRunner` graphs.
3. **Zero-Copy Buffer Hand-off**:
   - On macOS and Android (`sam2_main.cc`), the GPU output buffer of `pre_runner` (`pixels`) is bound directly as the input of `encode_image`; the three GPU output buffers of `encode_image` (`image_embeddings`, `feat_s1`, `feat_s0`) are bound directly as the inputs of `decode_mask`; and the `masks` output of `decode_mask` is bound directly as the input of `post_runner`.
   - On iOS (`ios/sam2_litert/Sources/LiteRT.swift` and `Sam2App.swift`), the app compiles `sam2_image.tflite` with `LiteRtCreateCompiledModel` (`kLiteRtHwAcceleratorGpu`), allocates managed Metal GPU `LiteRtTensorBuffer`s for signature 0 (`encode_image`) and signature 1 (`decode_mask`), and passes the three encoder output `LiteRtTensorBuffer`s directly into signature 1 without CPU copies.

### 2. MLX Baselines (`verify/mlx_512_bench.py` on macOS, `ios/sam2_mlx` on iOS)
1. **macOS Python MLX (`verify/mlx_512_bench.py`)**:
   - Instantiates `Sam2ImageSegmenter` from [`avbiswas/sam2-mlx`](https://github.com/avbiswas/sam2-mlx) configured for `image_size=512` (`pos_embed_hw=(128, 128)`) and loads the `fp16` `sam2_tiny_512.safetensors` weights.
   - Because MLX is lazy-evaluated, `encode()` calls `mx.eval(out["vision_features"], *out["high_res_features"])` and `decode(enc)` calls `mx.eval(out["low_res_masks"], out["ious"], out["object_score_logits"])` inside the timed block so Metal GPU execution completes before stopping the clock.
2. **iOS MLX-Swift (`ios/sam2_mlx`)**:
   - Implements the same `Sam2ImageSegmenter` (`Encoder.swift`, `Decoder.swift`) in Swift on `mlx-swift` (`MLX`, `MLXNN`, `MLXFast`), loads `sam2_tiny_512.safetensors` in `fp16`, and calls `eval(vision, highRes[0], highRes[1])` / `eval(masks, ious, obj)` inside each timed iteration (`Sam2MlxApp.swift`).

---

## Step 1: Export the `512x512` Weights (One-Time Setup)

From the repository root, generate `sam2_tiny_512.safetensors` (which re-composes the Hiera positional embedding at the `128x128` token grid from `facebook/sam2.1-hiera-tiny` and `avbiswas/sam2.1-hiera-tiny-mlx`):

```bash
python3 models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/ios/sam2_mlx/tools/make_512_weights.py \
  --size 512 \
  --out /tmp/sam2_tiny_512.safetensors
```

---

## Step 2: macOS Benchmark (LiteRT Tensor API on Metal vs. MLX)

### LiteRT Tensor API (Metal GPU, `fp16`)

Build and run `sam2_main` on Apple silicon macOS from the repository root:

```bash
bazel build --config=macos_arm64 -c opt \
  //models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_main

# Copy the Metal accelerator dylib into the working directory if needed:
cp -f bazel-bin/models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_main.runfiles/*/litert/c/libLiteRtMetalAccelerator.dylib . 2>/dev/null || true

./bazel-bin/models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_main \
  --weights=/tmp/sam2_tiny_512.safetensors \
  --tflite_path=/tmp/sam2_image.tflite \
  --accelerator=gpu \
  --gpu_precision=fp16 \
  --gpu_buffer_storage=buffer \
  --warmup=5 \
  --runs=20
```

This command also writes the two-signature flatbuffer (`/tmp/sam2_image.tflite` containing `encode_image` and `decode_mask`) used by the iOS LiteRT benchmark app below.

### MLX Baseline (macOS Python MLX, `fp16`)

Clone [`avbiswas/sam2-mlx`](https://github.com/avbiswas/sam2-mlx) and run [`verify/mlx_512_bench.py`](verify/mlx_512_bench.py):

```bash
git clone https://github.com/avbiswas/sam2-mlx /tmp/sam2-mlx

python3 models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/verify/mlx_512_bench.py \
  --sam2_mlx /tmp/sam2-mlx \
  --weights /tmp/sam2_tiny_512.safetensors \
  --size 512
```

---

## Step 3: iOS Benchmark (LiteRT Tensor API on Metal vs. MLX-Swift)

Both iOS benchmark harnesses under [`ios/`](ios/) use [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`) to generate an Xcode project that runs the 5-warmup + 20-timed-iteration benchmark on launch and prints the results both on screen and to the Xcode console (`SAM2BENCH` / `SAM2MLX`).

### LiteRT Tensor API on iOS (`ios/sam2_litert`)

1. Copy `/tmp/sam2_image.tflite` (exported by `sam2_main` in Step 2) into `ios/sam2_litert/Resources/`:
   ```bash
   cd models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/ios/sam2_litert
   mkdir -p Resources
   cp /tmp/sam2_image.tflite Resources/sam2_image.tflite
   xcodegen generate
   open SAM2.xcodeproj
   ```
2. In Xcode, select the **SAM2** scheme, set the build configuration to **Release**, select your physical iOS device, set your Signing Team under **Signing & Capabilities**, and press **Run (`⌘R`)**.

### MLX-Swift Baseline on iOS (`ios/sam2_mlx`)

1. Copy `/tmp/sam2_tiny_512.safetensors` (generated in Step 1) into `ios/sam2_mlx/Resources/`:
   ```bash
   cd models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/ios/sam2_mlx
   mkdir -p Resources
   cp /tmp/sam2_tiny_512.safetensors Resources/sam2_tiny_512.safetensors
   xcodegen generate
   open SAM2MLX.xcodeproj
   ```
2. In Xcode, select the **SAM2MLX** scheme, set the build configuration to **Release**, select your physical iOS device, set your Signing Team under **Signing & Capabilities**, and press **Run (`⌘R`)**.

---

## Step 4: Android Benchmark (LiteRT Tensor API on OpenCL GPU & CPU)

1. Cross-compile `sam2_main` for Android `arm64-v8a` from the repository root (`ANDROID_NDK_HOME` must point to an Android NDK r25+ installation):
   ```bash
   bazel build --config=android_arm64 -c opt \
     //models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_main
   ```

2. Push the binary, the OpenCL GPU accelerator shared library (`libLiteRtOpenClAccelerator.so`), and `/tmp/sam2_tiny_512.safetensors` to the connected Android device over `adb`:
   ```bash
   DEVICE_DIR=/data/local/tmp/sam2_bench
   adb shell "mkdir -p $DEVICE_DIR"

   adb push bazel-bin/models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_main "$DEVICE_DIR/"
   adb push bazel-bin/models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_main.runfiles/*/litert/c/libLiteRtOpenClAccelerator.so "$DEVICE_DIR/"
   adb push /tmp/sam2_tiny_512.safetensors "$DEVICE_DIR/"
   adb shell "chmod +x $DEVICE_DIR/sam2_main"
   ```

3. Run the benchmark on the Android GPU (`fp16`, OpenCL buffer storage) and CPU (`fp32` XNNPACK):
   ```bash
   # Android GPU (OpenCL fp16)
   adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ./sam2_main \
     --weights=$DEVICE_DIR/sam2_tiny_512.safetensors \
     --tflite_path=$DEVICE_DIR/sam2_image.tflite \
     --accelerator=gpu \
     --gpu_precision=fp16 \
     --gpu_buffer_storage=buffer \
     --warmup=5 \
     --runs=20"

   # Android CPU (XNNPACK fp32)
   adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ./sam2_main \
     --weights=$DEVICE_DIR/sam2_tiny_512.safetensors \
     --tflite_path=$DEVICE_DIR/sam2_image.tflite \
     --accelerator=cpu \
     --warmup=5 \
     --runs=20"
   ```
