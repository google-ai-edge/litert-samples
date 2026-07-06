# LiteRT Image Segmentation - iOS (Clean App)

An iOS application demonstrating real-time and static image segmentation using LiteRT's Compiled Model API. The app performs multi-class segmentation on a bundled test image, allowing easy verification of CPU (Builtin Kernels) and GPU (Metal) execution.

## Screenshot

<p align="center">
  <img src="img/inference.png" alt="Image Segmentation running on iPhone" width="300">
</p>

## Features

- **Backend Switching**: Select between CPU and GPU (Metal) directly in the UI.
- **XNNPACK Bypass**: Avoids delegate prepare allocation issues on iOS by running Builtin CPU Kernels.
- **GPU Acceleration**: Utilizes the dynamically loaded LiteRT Metal compiler plugin.
- **Static Verification**: Includes a bundled portrait sample image (`image.jpeg`) to immediately verify model compilation and inference on startup.
- **Performance Metrics Display**: Live measurements of pre-process, inference, and post-process execution times in milliseconds.

## Architecture

The app uses a Swift SwiftUI interface that bridges directly to a lightweight Objective-C++ wrapper (`LiteRTSegmenter.mm`) around LiteRT's C Compiled Model API.

| Component | File | Description |
|-----------|------|-------------|
| **LiteRTSegmenter** | `LiteRTSegmenter.mm` / `.h` | Objective-C++ bridge wrapping the LiteRT C API |
| **ContentView** | `ContentView.swift` | Single-page UI displaying original vs mask images, performance timing, and accelerator selection |
| **ImageSegmentationApp** | `ImageSegmentationApp.swift` | Swift application entry point |
| **CLiteRT.xcframework** | `CLiteRT.xcframework` | Precompiled LiteRT C framework |
| **libLiteRtMetalAccelerator.dylib** | `libLiteRtMetalAccelerator.dylib` | Precompiled Metal compiler plugin |

---

## Prerequisites & Setup

### 1. Xcode & Project Configuration
1. Open the project `ImageSegmentation.xcodeproj` in Xcode.
2. Select the **ImageSegmentation** target.
3. In **Signing & Capabilities**, select your **Personal Team** profile. The bundle identifier is configured to `com.aravindmurali.ImageSegmentation`.

### 2. Git LFS (Required for GPU Execution)
The Metal compiler plugin (`libLiteRtMetalAccelerator.dylib`) is stored in the LiteRT repository via **Git LFS**. Before running the app, install Git LFS and pull the actual binary slices (otherwise Xcode will package Git pointer text files, and `dlopen` will fail at runtime):
```bash
# Install Git LFS via Homebrew
brew install git-lfs

# Initialize LFS in your global Git config
git lfs install

# Navigate to your LiteRT submodule and pull the binary
cd path/to/LiteRT
git lfs pull
```

---

## How It Works

### CPU Backend (Builtin Kernels)
Due to a shape calculation overflow crash in the default XNNPACK delegate during the `RESIZE_NEAREST_NEIGHBOR` operator allocation, CPU compilation is configured to bypass XNNPACK. 

The wrapper uses the internal LiteRT CPU options API to set the kernel execution mode to builtin:
```objc
LrtCpuOptions* cpu_opts = nullptr;
if (LrtCreateCpuOptions(&cpu_opts) == kLiteRtStatusOk) {
    LrtSetCpuOptionsKernelMode(cpu_opts, kLiteRtCpuKernelModeBuiltin);
    // Serialize and add options to LiteRtOptions...
}
```

### GPU Backend (Metal)
To compilation-verify and execute operations on GPU:
1. **Fallback bitmask**: The compilation options set a combined bitmask of `kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu`. This instructs LiteRT to compile supported operators for Metal and fall back to CPU for unsupported operators.
2. **Dynamic Loading**: Since the GPU registry loads accelerator plugins dynamically at runtime, the compiler plugin `libLiteRtMetalAccelerator.dylib` is bundled under the app's `Frameworks/` directory and signed.
3. **Library Directory Tag**: During environment creation, we pass the bundle's private frameworks folder path as `kLiteRtEnvOptionTagRuntimeLibraryDir` so the loader can locate the plugin:
```objc
NSString *frameworksPath = [[NSBundle mainBundle] privateFrameworksPath];
LiteRtEnvOption env_options[1];
env_options[0].tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
env_options[0].value.type = kLiteRtAnyTypeString;
env_options[0].value.str_value = [frameworksPath UTF8String];
LiteRtStatus status = LiteRtCreateEnvironment(1, env_options, &_env);
```

---

## Model Information
* **Name**: `selfie_multiclass_256x256.tflite`
* **Input**: `1 x 256 x 256 x 3` (normalized float32 values in `[-1.0, 1.0]`)
* **Output**: `1 x 256 x 256 x 6` (float32 values representing probabilities across 6 target segmentation classes)
