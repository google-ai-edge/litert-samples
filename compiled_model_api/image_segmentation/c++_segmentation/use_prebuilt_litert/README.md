# LiteRT Android C++ Image Segmentation — Prebuilt SDK

This directory contains the **same** image-segmentation sample as
`build_from_source/`, but migrated from **Bazel** to **CMake** using the
prebuilt [LiteRT C++ SDK](https://ai.google.dev/edge/litert/next/android_cpp_sdk).
No LiteRT source tree is required.

## What Changed vs. `build_from_source`

| Aspect | build_from_source | use_prebuilt_litert |
|---|---|---|
| Build system | Bazel | CMake (NDK toolchain) |
| LiteRT dependency | Built from source | Prebuilt `litert_cc_sdk.zip` + `libLiteRt.so` |
| STB headers | Bazel `@stblib` | Bundled in `third_party/stb/` |
| Include paths | Bazel workspace | Flat relative paths |
| Deploy script arg | `<bazel-bin path>` | `<cmake build dir>` (e.g. `build/`) |

## Prerequisites

1. **Android NDK** r25c or later.
2. **CMake** ≥ 3.22 (`sudo apt-get install cmake` on Linux).
3. **ADB** installed and in `PATH` (for deployment).

---

## Build

### Step 1 — Get `libLiteRt.so` (one-time, ~5 MB)

Download it from the Google Maven AAR:

```bash
# Download the AAR for v2.1.1
wget -O litert.aar \
    "https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/2.1.1/litert-2.1.1.aar"

# Extract the arm64-v8a .so
unzip litert.aar "jni/arm64-v8a/libLiteRt.so" -d extracted/
```

> **Tip**: The AAR also contains `libLiteRtOpenClAccelerator.so` (GPU accelerator).
> Extract it the same way if you need GPU support:
> ```bash
> unzip litert.aar "jni/arm64-v8a/libLiteRtOpenClAccelerator.so" -d extracted/
> ```
> Place it alongside `libLiteRt.so` in `litert_cc_sdk/`.

### Step 2 — Run `build_prebuilt.sh`

The script **automatically downloads** `litert_cc_sdk.zip` (SDK headers + cmake files)
on first run, copies your `.so` in, and builds everything with CMake + NDK.

```bash
cd .../use_prebuilt_litert/

bash build_prebuilt.sh \
    --litert_version=2.1.1 \
    --ndk_path=/path/to/android-ndk \
    --litert_so=extracted/jni/arm64-v8a/libLiteRt.so
```

On subsequent runs the script detects the SDK is already extracted and skips downloads.

Binaries will be in `build/`:
```
build/cpp_segmentation_cpu
build/cpp_segmentation_gpu
build/cpp_segmentation_npu
```

## Deploy and Run on Device

Use `deploy_and_run_on_android.sh`, passing the cmake build directory (`build/`).
The script pushes the binary, libraries, shaders, and model to the device,
runs inference, and pulls `output_segmented.png` back automatically.

```bash
# CPU
./deploy_and_run_on_android.sh --accelerator=cpu --phone=s25 build/

# GPU (OpenCL)
./deploy_and_run_on_android.sh --accelerator=gpu --phone=s25 build/

# GPU with zero-copy GL buffers
./deploy_and_run_on_android.sh --accelerator=gpu --use_gl_buffers --phone=s25 build/

# NPU (Qualcomm S25 / SM8750, AOT compiled model)
./deploy_and_run_on_android.sh \
    --accelerator=npu --phone=s25 \
    --host_npu_lib=/path/to/qairt/lib \
    --host_npu_dispatch_lib=/path/to/dir/with/libLiteRtDispatch_Qualcomm.so \
    build/

# NPU JIT (no pre-compiled model needed)
./deploy_and_run_on_android.sh \
    --accelerator=npu --phone=s25 --jit \
    --host_npu_lib=/path/to/qairt/lib \
    --host_npu_dispatch_lib=/path/to/dir/with/libLiteRtDispatch_Qualcomm.so \
    build/

# MediaTek APU (dim9400, JIT)
./deploy_and_run_on_android.sh \
    --accelerator=npu --phone=dim9400 --jit \
    --host_npu_dispatch_lib=/path/to/dir/with/libLiteRtDispatch_MediaTek.so \
    build/
```

**`--phone` values**: `s24` (Snapdragon 8 Gen 3), `s25` (Snapdragon 8 Elite), `dim9400` (MediaTek Dimensity 9400)

The Qualcomm NPU requires:
- `libLiteRtDispatch_Qualcomm.so` from the [LiteRT NPU runtime libraries](https://github.com/google-ai-edge/LiteRT/releases/tag/v2.1.1) zip
- QAIRT SDK libraries (`libQnnHtp.so`, stub/skel `.so` files)

---

## Performance

| Processor    | Execution Type           | Inference | E2E    |
|:-------------|:-------------------------|----------:|-------:|
| CPU          | Sync                     | ~128 ms   | ~157 ms |
| GPU (OpenCL) | Sync                     | ~0.95 ms  | ~43 ms  |
| GPU          | Async + zero-copy buffer | —         | ~17 ms  |
| NPU (S25)    | AOT                      | —         | ~17 ms  |
| NPU (S25)    | JIT                      | —         | ~28 ms  |
| MediaTek APU | JIT                      | —         | ~9 ms   |
