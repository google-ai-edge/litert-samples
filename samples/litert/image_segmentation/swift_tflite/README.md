# TensorFlow Lite (Swift API) Image Segmentation - iOS

An iOS application demonstrating real-time and static multi-class image segmentation using the **TensorFlow Lite Swift API** (`TensorFlowLite` module integrated into the `LiteRT` Swift Package). The app supports switching at runtime between **3 hardware execution backends**:

1. **CPU (`XNNPACK`)** — Multi-threaded CPU execution (`Interpreter.Options.isXNNPackEnabled = true`, `threadCount = 4`).
2. **GPU (`MetalDelegate`)** — Apple GPU compute acceleration (`MetalDelegate` built with `--define=use_metal_delegate=1`).
3. **NPU / ANE (`CoreMLDelegate`)** — Apple Neural Engine / Core ML acceleration (`CoreMLDelegate` built with `--define=use_coreml_delegate=1`).

---

## Features

- **3-Way Backend Switching**: Select between **CPU (XNNPACK)**, **GPU (MetalDelegate)**, and **NPU (CoreMLDelegate)** directly in the expandable bottom sheet.
- **Real-Time Camera Stream**: Run live segmentation on the front/back camera feed.
- **Gallery Image Selection**: Run static segmentation on the bundled portrait (`image.jpeg`) or import custom images from the photo library.
- **Performance Metrics Display**: Live measurements of Pre-process, Inference, and Post-process execution times in milliseconds alongside `TensorFlowLite.Runtime.version`.

---

## Architecture

| Component | File | Description |
|-----------|------|-------------|
| **TFLiteSegmenter** | `ImageSegmentation/TFLiteSegmenter.swift` | Native Swift implementation utilizing `TensorFlowLite` (`Interpreter`, `Interpreter.Options`, `MetalDelegate`, `CoreMLDelegate`, `Tensor`) |
| **ContentView** | `ImageSegmentation/ContentView.swift` | Single-page SwiftUI interface displaying camera/gallery segmentation, latency telemetry, and 3-way delegate picker |
| **CameraManager** | `ImageSegmentation/CameraManager.swift` | Manages `AVFoundation` camera capture session and frame streams |
| **ImagePicker** | `ImageSegmentation/ImagePicker.swift` | Wraps `PHPickerViewController` in `UIViewControllerRepresentable` for photo library access |
| **ImageSegmentationApp** | `ImageSegmentation/ImageSegmentationApp.swift` | SwiftUI application entry point |

---

## Prerequisites & Setup

### 1. Building `TensorFlowLite_xcframework` and `TensorFlowLiteC_xcframework` with Delegates

By default, `//litert/swift:TensorFlowLite_xcframework` builds with no delegates (CPU/XNNPACK only). To enable both **`MetalDelegate`** and **`CoreMLDelegate`**, pass `--define=use_metal_delegate=1` and `--define=use_coreml_delegate=1`:

```bash
# Navigate to the LiteRT repository
cd path/to/LiteRT

# Build TensorFlowLite_xcframework and TensorFlowLiteC_xcframework with Metal and CoreML delegates
bazel build -c opt --config=ios \
  --define=use_metal_delegate=1 \
  --define=use_coreml_delegate=1 \
  //litert/swift:TensorFlowLite_xcframework \
  //litert/swift:TensorFlowLiteC_xcframework

# Copy the compiled xcframework archives into LiteRT/prebuilt/
mkdir -p prebuilt
cp -f bazel-bin/litert/swift/TensorFlowLite_xcframework.xcframework.zip prebuilt/TensorFlowLite.xcframework.zip
cp -f bazel-bin/litert/swift/TensorFlowLiteC_xcframework.xcframework.zip prebuilt/TensorFlowLiteC.xcframework.zip
```

### 2. Download the Model File (if not already present)
```bash
cd path/to/litert-samples/samples/litert/image_segmentation/swift_tflite
curl -L -o selfie_multiclass_256x256.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

### 3. Open in Xcode
1. Open `ImageSegmentation.xcodeproj` in Xcode.
2. Select the **ImageSegmentation** target and configure your **Signing & Capabilities** team.
3. Build and run on an iOS device or Simulator.

---

## How the 3 Backends Work in `TFLiteSegmenter.swift`

### 1. CPU Backend (`XNNPACK`)
Uses the built-in XNNPACK delegate via `Interpreter.Options`:
```swift
var options = Interpreter.Options()
options.threadCount = 4
options.isXNNPackEnabled = true
let interpreter = try Interpreter(modelPath: modelPath, options: options)
try interpreter.allocateTensors()
```

### 2. GPU Backend (`MetalDelegate`)
Uses `MetalDelegate` (`--define=use_metal_delegate=1`):
```swift
var metalOptions = MetalDelegate.Options()
metalOptions.isPrecisionLossAllowed = true
metalOptions.waitType = .passive
let metalDelegate = MetalDelegate(options: metalOptions)
let interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: [metalDelegate])
try interpreter.allocateTensors()
```

### 3. Core ML / Neural Engine Backend (`CoreMLDelegate`)
Uses `CoreMLDelegate` (`--define=use_coreml_delegate=1`), targeting `.neuralEngine` (Apple Neural Engine / NPU) first and falling back to `.all` on Simulator or non-ANE devices:
```swift
var coreMLOptions = CoreMLDelegate.Options()
coreMLOptions.enabledDevices = .neuralEngine
let coreMLDelegate = CoreMLDelegate(options: coreMLOptions) ?? {
    coreMLOptions.enabledDevices = .all
    return CoreMLDelegate(options: coreMLOptions)!
}()
let interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: [coreMLDelegate])
try interpreter.allocateTensors()
```
