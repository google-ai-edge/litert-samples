---
title: The LiteRT CompiledModel API
source: https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/litert-runtime/SKILL.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/litert-runtime/references/verify.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/compiled-model-app-scaffolding/SKILL.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/litert-compiled-model-migration/SKILL.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/on-device-verification/SKILL.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/runtime/README.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/core/README.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/swift/README.md
license: Apache-2.0
---

# The LiteRT CompiledModel API

This document describes the LiteRT CompiledModel API, the recommended way to run `.tflite` models with LiteRT 2.x: its core concepts, how to create a model and its tensor buffers in Kotlin, Swift, C++ and Python, how accelerators are selected, the threading and lifecycle rules, how it replaces the TensorFlow Lite Interpreter, and how to verify that a model's output on a device is correct.

## Core concepts of the LiteRT runtime

The LiteRT runtime manages the execution of models, including memory allocation, the dispatch of operations to hardware accelerators, and the synchronization of events. It is built around four core concepts.

- Accelerator: a hardware accelerator that can be used to speed up the execution of a model. The runtime provides a registry for accelerators, which allows developers to register their own custom accelerators.
- TensorBuffer: a block of memory that can be used to store tensor data. It can be backed by host memory, hardware buffers, or other types of memory.
- Event: used to synchronize the execution of operations on different hardware accelerators.
- CompiledModel: a model that has been optimized for a specific hardware platform. It is created from a Model and a set of compilation options.

Above them sits the Environment, the top-level object in the LiteRT runtime, responsible for managing the lifecycle of all other objects. Applications use the public APIs: the C API provides a low-level, ABI-stable interface, and the C++ API provides a higher-level, more convenient interface. The runtime and core internals are not part of the public API and can change without notice. Kotlin, Swift, Python and JavaScript bindings sit on top.

## Why CompiledModel replaces the TensorFlow Lite Interpreter

New inference code should use the CompiledModel API; the TensorFlow Lite `Interpreter` API and manual delegate creation are legacy, and TensorFlow Lite packages are in maintenance mode. The Kotlin migration path is described as a "like for like" baseline migration: the `Interpreter` becomes `CompiledModel`, and delegates are replaced by accelerator options.

Several legacy pieces have no LiteRT 2.x equivalent. NNAPI is deprecated since Android 15, and the LiteRT 2.x Java and Kotlin API has no NNAPI entry point; the replacement is `CompiledModel.Options(Accelerator.NPU)`. The Flex delegate (Select TF ops) ships only in the legacy `tensorflow-lite-select-tf-ops` artifact (68 MB for arm64 in 2.16.1) and in no LiteRT 2.x artifact, so a model with Select TF ops must be re-exported without them, or the custom math implemented as a C++ `litert::CustomOpKernel`. The legacy `org.tensorflow.lite.support` library (`ImageProcessor`, `ResizeOp`, `NormalizeOp`) must be removed, because it depends on `org.tensorflow:tensorflow-lite`, which would keep the legacy Interpreter runtime in the app next to LiteRT 2.x.

## Adding LiteRT to an Android app

The Android runtime is the standalone `com.google.ai.edge.litert:litert` AAR from Google Maven, for example version 2.2.0, added with `implementation("com.google.ai.edge.litert:litert:2.2.0")`. It pulls in `litert-api`, bundles the GPU accelerator, declares the GPU driver libraries in its own manifest, and sets `minSdk` 24; no second artifact and no manifest entry are needed, and no separate `litert-gpu` artifact exists for 2.x. As of September 2026 no LiteRT artifact for Google Play services is on Google Maven (only the `play-services-tflite` Interpreter artifacts exist).

Put the model at `app/src/main/assets/model.tflite` and set `androidResources { noCompress += "tflite" }` so the asset stays memory-mappable. A model too large to bundle is downloaded into `context.filesDir` and loaded from its file path. With AGP 9.x the build of version 2.2.0 stops at `processDebugMainManifest` because `litert` and `litert-api` both declare the namespace `com.google.ai.edge.litert`; the documented workaround is `android.uniquePackageNames=false` in `gradle.properties`, removed again once a LiteRT version builds without it.

## Creating a CompiledModel and its buffers in Kotlin

`CompiledModel.create(context.assets, assetName, CompiledModel.Options(Accelerator.CPU), env)` loads a model from assets, and `create(filePath, options, env)` loads it from a file; nothing takes a ByteBuffer or a mapped file. `Options` takes one or more of `Accelerator.CPU`, `GPU` and `NPU`, for example `Options(Accelerator.GPU, Accelerator.CPU)`. The optional `env` is `Environment.create(context)`, which enables the compiler cache, or `Environment.create(context, BuiltinNpuAcceleratorProvider(context))` for the NPU.

Buffers come from the model: `createInputBuffers()` and `createOutputBuffers()` return a `List<TensorBuffer>`, filled with `writeFloat`, `writeInt8`, `writeInt` or `writeLong` and read with `readFloat()` and friends. Signature-based models use `run(inputMap, outputMap, key)` with `Map<String, TensorBuffer>`. Tensor shapes are queried by name with `getInputTensorType(inputName).layout?.dimensions`.

```kotlin
val model = CompiledModel.create(context.assets, "model.tflite",
    CompiledModel.Options(Accelerator.GPU))
val inputs = model.createInputBuffers()
val outputs = model.createOutputBuffers()
inputs[0].writeFloat(input)
model.run(inputs, outputs)
val result = outputs[0].readFloat()
```

## How run() behaves on the CPU and the GPU

`run(inputs, outputs)` is synchronous on the CPU: it blocks until the outputs are ready. On the GPU it can return before the GPU finishes, and the `read*()` call waits; the output read is the synchronization point. There is no callback API, and the Kotlin API has no `runAsync`.

This matters for benchmarking. Time `run()` and `readFloat()` together; timing `run()` alone under-reports GPU work, and has reported a 4× GPU win that did not exist. A large readback time is usually the deferred compute, not the transfer. Per-call overhead also makes small per-step graphs a net GPU loss, with the crossover around hundreds of nodes per call.

The Swift API differs here: its `CompiledModel` runs inference synchronously with `run`, or dispatches it asynchronously with `dispatch` when supported. LiteRT V2 lists true asynchronous execution among the CompiledModel API's features.

## Choosing the CPU, GPU or NPU accelerator

`Accelerator.CPU` runs everywhere. `Accelerator.GPU` compiles the graph for the GPU; if `create` throws `LiteRtException` because an op is not supported by the GPU, the app can create the model again with `Accelerator.CPU`. `CompiledModel.Options()` with no accelerator compiles, but `create` then throws `LiteRtException`. GPU alone makes `create` throw when an op is not GPU-supported, whereas `Options(Accelerator.GPU, Accelerator.CPU)` permits partial delegation, like the legacy GPU delegate that left unsupported ops on the CPU. NPU-only options compile as NPU plus CPU, so unsupported ops fall back to the CPU.

The two LiteRT skills give complementary advice. The migration skill builds an explicit NPU, then GPU, then CPU fallback cascade, trying each `CompiledModel.create` and catching `LiteRtException`, and notes that the accelerator reported is the step that did not throw, not proof that every op runs there. The app-scaffolding skill asks for the strict accelerator a model was verified on: `Accelerator.GPU` fails compilation on an unsupported op instead of silently falling back, which it calls a feature; surface the failure as a visible error rather than hiding a 10× slowdown behind a CPU fallback.

## Threading, warm-up and resource lifecycle

One confined dispatcher owns the model. In Kotlin that is `Dispatchers.IO.limitedParallelism(1)` (or a single-thread executor); create, run and close the model only inside it, never on the main thread. `limitedParallelism(1)` serializes the calls; it does not pin them to one thread.

Buffers are created once with the model, reused for every inference, and closed. `CompiledModel`, `TensorBuffer` and `Environment` are `AutoCloseable`: close the buffers, then the model, then an `Environment` the app created. Forgetting the buffers leaks native memory even when the model itself is closed.

Warm up once at initialization. The first GPU inference includes shader compilation, so run one inference on dummy input right after create, and never quote a first-run number as latency.

When one app creates many `CompiledModel`s (pipelines, chunked models), create one `Environment` and pass it to every `CompiledModel.create` call: per-create GPU contexts leak, and a create-per-step loop will eventually take the process down (observed at about 20 creates). For stateful and multi-graph models, feed step N's output buffers as step N+1's inputs instead of copying state through the host.

## Preprocessing is part of the model contract

Preprocessing must follow the model's own input requirements: input size, mean and standard deviation, channel order and layout. A typical image example scales a bitmap to 224 by 224 and writes `(channel - mean) / std` into a `FloatArray` in NHWC, RGB order, scaled to the range -1 to 1 with mean and std of 127.5. A wrong mean or std, or RGB swapped for BGR, looks exactly like a broken model, and it is the first thing to diff against the model's export script when app output is subtly wrong.

For this reason the inference helper class owns pre- and post-processing, not the screen. The litert-samples shared helpers provide `CompiledModelRunner.kt` (the buffer lifecycle), `ImageTensor.kt` (bitmap to float tensor with mean/std, NCHW/NHWC, RGB/BGR and letterbox with coordinate mapping back), `AudioCapture.kt` (a 16 kHz mono `AudioRecord` loop), `RealtimeCameraPipeline.kt` (CameraX capture to pooled bitmaps) and `MathOps.kt` (softmax, argmax, IoU, NMS).

## Mapping legacy Interpreter calls to CompiledModel

The migration skill gives one-to-one replacements for the common legacy calls.

- `org.tensorflow.lite.Interpreter` becomes `com.google.ai.edge.litert.CompiledModel`, and `Interpreter(modelFile, options)` becomes `CompiledModel.create(context.assets, assetName, options, env)` or `create(filePath, options, env)`.
- `interpreter.run(input, output)` and `runForMultipleInputsOutputs` become `compiledModel.run(inputBuffers, outputBuffers)`; `runSignature(inputs, outputs, key)` becomes `run(inputMap, outputMap, key)`.
- `Interpreter.Options().setNumThreads(n)` becomes `options.cpuOptions = CompiledModel.CpuOptions(numThreads = n)` on a fresh `Options(Accelerator.CPU)`.
- `GpuDelegate()` becomes `CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)`, and `NnApiDelegate()` becomes `CompiledModel.Options(Accelerator.NPU)`.
- `interpreter.resizeInput(...)` has no Kotlin equivalent; the C++ API has `ResizeInputTensor`.
- In C++, `tensorflow/lite/interpreter.h` becomes `litert/cc/litert_compiled_model.h`, and `tensorflow/lite/c/c_api.h` becomes `litert/c/litert_compiled_model.h`.

## Using the CompiledModel API from Swift on iOS and macOS

The LiteRT Swift package provides `LiteRT`, a type-safe Swift wrapper over the LiteRT C API for loading, compiling and running models with hardware acceleration (CPU, GPU, NPU) on iOS and macOS, plus the legacy `TensorFlowLite` Swift API. It requires Xcode 15 or later and an app that targets iOS 15 or later, or macOS 12 or later. Developers add `https://github.com/google-ai-edge/LiteRT` as a package dependency on a release branch such as `release/2.3.0`, choose the `LiteRT` product and, to use the GPU, the `LiteRtMetalAccelerator` product.

The key types are `Environment` (runtime options and the registered accelerators; it locates the Metal accelerator automatically when the app links `LiteRtMetalAccelerator`), `CompiledModel` (loads from a file path, `Data` buffer or file descriptor and compiles it), `Options` (selects CPU, GPU or NPU, with accelerator settings such as `CpuOptions`), and `TensorBuffer` (host memory or a Metal `MTLBuffer`). The GPU path calls `options.setHardwareAccelerators([.gpu, .cpu])`, which runs on the Metal GPU and falls back to the CPU for unsupported ops; the CPU path can set the XNNPACK delegate kernel mode and a thread count. The sample iOS image segmentation app switches between the CPU (XNNPACK) and GPU (Metal) backends in its UI.

## C++ and zero-copy hardware buffers

A `TensorBuffer` can be backed by host memory or by hardware buffers. The runtime's buffer types include Android Hardware Buffers (AHWB), DMA-BUF, ION, FastRPC, OpenGL buffers and textures, and OpenCL memory, with tensor buffer requirements describing what a backend needs.

In Kotlin, tensor buffers come from the model and there is no `TensorBuffer.createFromAhwb`. AHardwareBuffer zero-copy interop is part of the C++ API (`litert::TensorBuffer::CreateFromAhwb`), so a camera pipeline that needs it runs the whole inference from C++; there is no supported way to hand a C++ buffer to Kotlin `run()`. For native Android modules, the C++ SDK is `litert_cc_sdk.zip` from the matching LiteRT GitHub release, with `libLiteRt.so` copied from the AAR, linked through the `litert_cc_api` CMake target.

## Python CompiledModel for checking a model on a workstation

The Python package `ai-edge-litert` exposes the same API: `CompiledModel.from_file("model.tflite", hardware_accel=HardwareAccelerator.GPU)`, or `.CPU`, where one argument switches the accelerator. Desktop Python runtimes exercise the CPU and XNNPACK only, which is exactly what makes them the right numerical reference for a device run. Compiling with `HardwareAccelerator.CPU | HardwareAccelerator.GPU` permits partial delegation and hides fallback; use the combined mode only to discover which ops fell back after a strict GPU compile fails.

## Verifying model output on the device

A device result is done when three things hold: the model compiles and runs on the accelerator you claim it runs on; the output matches the source model numerically and on a task-level gate; and the record names the device, the runtime version and the residency line.

The loop starts with reference dumps from the source model for fixed inputs, one real sample plus fixed-seed random input. Run the same inputs on the device with `Accelerator.CPU` first: if that already differs from the reference, the problem is the conversion, not the device. Then run the GPU or NPU and gate the output three ways: numeric (correlation and max absolute difference against the dump), task (argmax match, IoU, token match), and artifact (the mask, audio or text a person can judge). Record device, LiteRT version, accelerator, correlation, max abs diff and the task result, one row per device.

### Telling a real GPU run from a silent CPU fallback

The most common false positive is that everything runs and the numbers match perfectly. If the GPU output is bit-identical to the device CPU fp32 output, it almost certainly did not run on the GPU; a genuine GPU fp16 run drifts in the last digits. Read the delegate log before reading any numbers: a line such as "Replacing N out of M node(s) with delegate ... X partitions" tells you how much of the graph was delegated. If N is less than M, or X is more than 1, part of the graph runs on the CPU. Only when the N of M count is full and the outputs drift in the last digits are you looking at a real GPU run; decide whether partial CPU execution is acceptable before quoting any accuracy or latency number.

### Device-only failures and their causes

Some failures appear only on the device. A GPU compile that fails for a model that runs on the workstation is a whole-graph compile ceiling; split the model at a block boundary. A file that refuses to load at all hits the more than 2 GB FlatBuffer limit. NaN or garbage only on the GPU is usually an fp16 range break. A process that dies loading a large float model has hit a memory ceiling; quantize or split first. A deep transformer shipped as an fp16 graph can be bit-exact on desktop and noise on the device CPU, because Android ARM XNNPACK computes native fp16 while desktop XNNPACK upcasts to fp32; ship fp32 graphs for CPU inference on device. Graphs with fused-LSTM-style variable tensors are Interpreter-only, because the CompiledModel loader rejects variable tensors outright.

## Structuring an app around a verified model

The app-scaffolding skill sets three conditions for an app: it reproduces the model recipe's verification numbers on the verified accelerator before any UI exists; inference is confined and leak-free, with one dispatcher owning the model, every buffer closed, and benchmarks that include the readback; and the inference code is liftable into another app unchanged.

Its Android shape has four layers: a task helper that owns the CompiledModel, its buffers and pre- and post-processing with no Android UI types; a ViewModel state machine that drives the helper on a confined dispatcher; one immutable UiState data class; and a thin activity with Compose screens. Weights are never committed: bundleable models are fetched at build time into `assets/` with `noCompress` set, and models too big to bundle are staged into the app's private `filesDir`. Two UI traps masquerade as model bugs: an image composable that renders a 256-pixel output at native size inside a vertical scroll, and a "slow model" that is really work on the UI thread.
