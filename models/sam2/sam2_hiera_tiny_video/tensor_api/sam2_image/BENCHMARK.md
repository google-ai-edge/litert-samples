# SAM 2.1 Hiera-Tiny (`512x512`, `fp16`) Cross-Platform Benchmarks

This guide reproduces the SAM 2.1 Hiera-Tiny (`512x512`, `fp16`) image encoder and mask decoder latency benchmarks across **macOS**, **iOS**, and **Android** (plus the macOS Python MLX and iOS MLX-Swift baselines).

All benchmarks use the same deterministic fixture and timing protocol:
* **Input**: `512x512` synthetic image containing a centered white circle (`r = 120`, ImageNet-normalized) and a positive point prompt at `(256, 256)`.
* **Precision**: `fp16` on GPU.
* **Protocol**: 5 full-pipeline (`encode_image` + `decode_mask`) warmup runs followed by 20 timed runs per stage, reporting the median latency in milliseconds and verifying `iou_scores`, `object_score`, and the foreground pixel count (`2829 / 16384` on mask `[2]`).

---

## 1. Export the `512x512` Weights (One-Time Setup)

From the repository root, generate the `512x512` weights (`sam2_tiny_512.safetensors`, with the Hiera positional embedding re-composed at the `128x128` token grid):

```bash
python3 models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/ios/sam2_mlx/tools/make_512_weights.py \
  --size 512 \
  --out /tmp/sam2_tiny_512.safetensors
```

---

## 2. macOS Benchmark (LiteRT Tensor API on Metal vs. MLX)

### LiteRT Tensor API (Metal GPU, `fp16`)

Build and run `sam2_main` on Apple silicon macOS:

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

This command also writes the two-signature flatbuffer (`/tmp/sam2_image.tflite` with `encode_image` and `decode_mask` signatures) used by the iOS LiteRT benchmark app below.

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

## 3. iOS Benchmark (LiteRT Tensor API on Metal vs. MLX-Swift)

Both iOS benchmark harnesses under [`ios/`](ios/) use [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`) to generate an Xcode project that runs the 5-warmup + 20-timed-iteration benchmark on launch and prints the results both on screen and to the Xcode console (`SAM2BENCH` / `SAM2MLX`).

### LiteRT Tensor API on iOS (`ios/sam2_litert`)

1. Copy `/tmp/sam2_image.tflite` (exported by `sam2_main` in Section 2) into `ios/sam2_litert/Resources/`:
   ```bash
   cd models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/ios/sam2_litert
   mkdir -p Resources
   cp /tmp/sam2_image.tflite Resources/sam2_image.tflite
   xcodegen generate
   open SAM2.xcodeproj
   ```
2. In Xcode, select the **SAM2** scheme, set the build configuration to **Release**, select your physical iOS device, set your Signing Team under **Signing & Capabilities**, and press **Run (`⌘R`)**.

### MLX-Swift Baseline on iOS (`ios/sam2_mlx`)

1. Copy `/tmp/sam2_tiny_512.safetensors` (generated in Section 1) into `ios/sam2_mlx/Resources/`:
   ```bash
   cd models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/ios/sam2_mlx
   mkdir -p Resources
   cp /tmp/sam2_tiny_512.safetensors Resources/sam2_tiny_512.safetensors
   xcodegen generate
   open SAM2MLX.xcodeproj
   ```
2. In Xcode, select the **SAM2MLX** scheme, set the build configuration to **Release**, select your physical iOS device, set your Signing Team under **Signing & Capabilities**, and press **Run (`⌘R`)**.

---

## 4. Android Benchmark (LiteRT Tensor API on OpenCL GPU & CPU)

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
