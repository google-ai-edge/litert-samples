---
title: GPU and NPU acceleration in LiteRT
source: https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/README.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/tflite/delegates/gpu/README.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/python/tools/mixed_precision/README.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/COMPILER_PLUGIN.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/DISPATCH_API.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/JIT_COMPILATION.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/vendors/qualcomm/README.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/vendors/qualcomm/doc/HTP_INSTRUCTIONS.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/vendors/qualcomm/doc/QAIRT_SDK.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/vendors/mediatek/compiler/MediaTek_Neuro_Compiler.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/vendors/intel_openvino/README.md ; https://github.com/google-ai-edge/LiteRT/blob/bbb1312f51e37cab695f7ec3a3c74b8a4a9ca13a/litert/vendors/arm_vulkan_ml/README.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/litert-compiled-model-migration/SKILL.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/on-device-verification/SKILL.md ; https://github.com/google-ai-edge/litert-samples/blob/367ddbdee22a8d73fba6e519c4a3f18d19008b05/skills/gpu-clean-conversion/SKILL.md
license: Apache-2.0
---

# GPU and NPU acceleration in LiteRT

This document explains how LiteRT runs models on GPUs and NPUs: why GPUs suit neural networks, GPU precision and performance tips, how NPU support is built from compiler plugins and the Dispatch API, ahead-of-time versus just-in-time compilation, the Qualcomm, MediaTek, Intel and Arm integrations, how an Android app enables the NPU, and why acceleration results must be checked per device.

## The three kinds of accelerator in LiteRT

LiteRT executes a model on the CPU through XNNPack, on the GPU through ML Drift, or on supported TPUs and NPUs. In LiteRT V2 the Compiled Model API features automated accelerator selection, so no explicit delegates are needed. GPU acceleration uses OpenCL or OpenGL on Android, Metal on iOS, Metal or WebGPU on macOS, and WebGPU on Linux, Windows and the web. NPUs are reached through unified NPU acceleration, which gives seamless access to NPUs from major chipset providers through a single, consistent API; on Android the supported NPU vendors are Broadcom, Google Tensor, Intel, MediaTek, Qualcomm and S.LSI.

LiteRT is built with first-class support for heterogeneous graphs. Any operation not selected for an accelerator is left to the CPU or made available for acceleration on another backend.

## Why neural networks run well on mobile GPUs

GPUs are designed to have high throughput for massively parallelizable workloads. They are well suited to deep neural nets, which consist of a huge number of operators, each working on input tensors that can be easily divided into smaller workloads and carried out in parallel, typically resulting in lower latency. In the best scenario, inference on the GPU may run fast enough to become suitable for real-time applications where it was not before.

GPUs do their computation with 16-bit or 32-bit floating point numbers and, unlike CPUs, do not require quantization for optimal performance. Another benefit of GPU inference is power efficiency: GPUs carry out the computations in a very efficient and optimized way, so they consume less power and generate less heat than when the same task is run on the CPU.

## ML Drift and the legacy TensorFlow Lite GPU delegate

LiteRT V2 provides faster GPU acceleration via ML Drift, aimed at supporting generative AI inference. It leverages state-of-the-art GPU acceleration with new buffer interoperability that minimizes latency across various GPU buffer types.

The older TensorFlow Lite GPU backend was used through delegate APIs: the app created a GPU delegate and added it to the interpreter builder. That backend uses OpenGL ES 3.1 compute shaders or OpenCL on Android, and Metal shaders on iOS. Its documented op set in 16-bit and 32-bit float precision includes ADD, AVERAGE_POOL_2D, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, LOGISTIC, a basic LSTM, MAX_POOL_2D, MUL, PAD, PRELU, RELU, RELU6, RESHAPE, RESIZE_BILINEAR, SOFTMAX, STRIDED_SLICE, SUB and TRANSPOSE_CONV. In LiteRT 2.x, application code no longer creates delegates manually; it passes the GPU accelerator in the CompiledModel options.

## GPU precision, FP16 by default and FP32 when needed

GPU execution commonly runs in half precision. In the legacy delegate, setting `allow_precision_loss` to true lets the GPU perform FP16 calculation internally, which is faster; the default options may not be the fastest. By default the GPU delegate may run an entire model in FP16 precision, ignoring the FP32 tensors in the model, to optimize for speed. The LiteRT conversion guidance goes further: fp16 is used even for an fp32 graph, because the delegate reduces in fp16 regardless of tensor dtype, so anything that sums many large values (variance, sums of squares, multi-axis means) is a candidate for overflow.

To make the GPU respect FP32 sections, configure the accelerator to compile with FP32 precision. In the LiteRT C++ API that is `gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp32)`. The GPU then executes the FP16 parts of a mixed-precision model in FP16 and the selected sensitive parts in FP32, recovering accuracy while keeping most of the performance gains. Forcing fp32 rescues overflow-to-NaN cases only; it does not fix precision compounding, so "fp32 didn't help" does not exonerate fp16.

## GPU performance tips

Some operations that are trivial on the CPU can be high cost on the GPU. One class is the various forms of reshape operations, including BATCH_TO_SPACE, SPACE_TO_BATCH and SPACE_TO_DEPTH; if those ops were inserted into the network only for the architect's logical thinking, it is worth removing them for performance.

On the GPU, tensor data is sliced into 4 channels. A computation on a tensor of shape [B, H, W, 5] performs about the same as on [B, H, W, 8], but significantly worse than on [B, H, W, 4]. For the same reason, if the camera hardware supports image frames in RGBA, feeding that 4-channel input is significantly faster, because a memory copy from 3-channel RGB to 4-channel RGBX can be avoided. Re-training a classifier with a mobile-optimized network architecture is also a significant part of optimization for on-device inference.

## How LiteRT reaches NPUs, compiler plugins and dispatch

NPU support in LiteRT has two halves. A compiler plugin integrates a specific hardware accelerator with a compiler dependency into the framework: it takes portions of the model and converts them into a format the target hardware can execute, by calling the backend's compiler. The Dispatch API is the runtime analog: it executes the binary blobs generated by the plugin on the NPU.

The Dispatch API is intended to replace the existing TFLite delegate, and enables features the delegate did not support. Hardware buffers go through the standard `TensorBuffer` type, with buffer requirement handshaking through `TensorBufferRequirements`. The interface is ABI stable, being all C APIs. It handles both JIT and AOT compilation, with the NPU compiler interface standardized through the compiler plugin, and it supports asynchronous execution. In an app, the Dispatch API is used through the NPU accelerator within the `CompiledModel`, which internally creates a dispatch delegate.

## Compiler plugins, partitioning and compilation

The framework uses a compiler plugin in two phases. In partitioning, the plugin inspects the model graph and identifies subsets of operations that it supports and can efficiently accelerate on the target hardware; these subgraphs are marked for compilation and outlined. By default LiteRT groups all selected ops into the largest possible sub-DAGs. In compilation, the plugin uses its internal logic and possibly external toolchains to generate one or more hardware-specific bytecode modules for the partitions. The framework replaces the original subgraphs with custom operations that invoke the hardware driver, and embeds the compiled output in the `.tflite` model.

Model authors can also control placement explicitly with composite ops. Operations wrapped in `odml.npu_call` are automatically selected for compilation by the targeted compiler plugin, so a subgraph is treated as a single optimized unit on the NPU. Operations inside `odml.cpu_call` are shielded from all compiler plugins and run on the CPU, for example to preserve numerical precision or to use a CPU-only custom kernel.

## Ahead-of-time versus on-device NPU compilation

LiteRT can use compiler plugins for ahead-of-time (AOT) compilation through native tooling on a workstation, and for on-device compilation. On-device compilation is more flexible, fully internalized within the LiteRT runtime APIs, and only requires the management of a single backend-agnostic model. The AOT flow can unblock compilation when it is too resource intensive to run on-device, which may be the case with many contemporary large models.

The Qualcomm documentation names the modes concretely. AOT compiles on the host (x86 Linux) and never recompiles at load; it is the most common path. Real JIT compiles on the device at load time, in memory, without serialization or cache, so it recompiles on every load. On-device AOT compiles on the device on the first load and caches the result, so later loads skip recompilation. Choose JIT when you cannot pre-compile on a host. Host AOT compilation is done for a named target SoC, for example SM8850 for the Snapdragon 8 Elite Gen 5, and JIT instead pushes the original `.tflite` model plus the compiler plugin to the device.

## Just-in-time NPU compilation in memory

JIT compilation in LiteRT is an in-memory flow that lets NPU compiler plugins pass compiled executable handles directly to the dispatch runtime, bypassing serialization of compiled bytecode into the `.tflite` file and deserialization at startup. It is useful for reducing startup latency, since skipping serialization and deserialization significantly reduces the time to initialize the model, and for supporting backends whose vendor SDKs do not support serializing compiled graphs to disk.

JIT is typically enabled through vendor-specific options; on Qualcomm it is the `enable_just_in_time` option. Because JIT relies on in-memory handles that exist only during the lifetime of the compiler plugin and the runtime process, JIT compilation cannot be cached. When JIT handles are detected, LiteRT disables model caching for that run and logs "JIT execution handles detected. Disabling JIT model caching", even if a compilation cache directory is configured.

## Qualcomm NPUs through Qualcomm AI Engine Direct

The LiteRT Qualcomm integration targets Qualcomm AI Engine Direct (QAIRT), offloading models to Qualcomm NPUs, GPUs and DSPs. Its compiler plugin legalizes and compiles LiteRT graphs into QNN graphs, online and offline, and its Dispatch API manages execution of the compiled QNN graphs on device. The top-tier supported devices are the Snapdragon 8 Gen 5 (SM8850), the Snapdragon 8 Elite (SM8750) and the Snapdragon 8 Gen 3 (SM8650), with a full list in the repository's supported SoC table.

The plugin supports the QNN backends Htp, Dsp, Ir, Saver and Gpu, on Android, x86 Linux, x86 and aarch64 Windows, and aarch64 Linux IoT devices. With the HTP backend, the target's Hexagon architecture decides which libraries are needed; for example SM8850 is Hexagon V81, SM8750 is V79 and SM8650 is V75. There is also an LPAI (Low Power AI) backend for always-on embedded use cases.

## MediaTek, Intel and Arm integrations

The MediaTek Neuron compiler plugin is built as `libLiteRtCompilerPlugin_MediaTek.so`, which LiteRT loads during AOT compilation, and links against `libneuron_adapter.so` from the NeuroPilot SDK, selected per SDK version (v7, v8 or v9). Its compile-time options include a performance mode (LowPower, FastSingleAnswer, SustainedSpeed or TurboBoost, defaulting to SustainedSpeed), Gemma-specific compiler optimizations, L1 cache optimizations and an optimization hint.

Intel NPU support comes through OpenVINO; LiteRT bundles a pinned OpenVINO package through the `ai_edge_litert_sdk_intel` pip package, and Intel published "LiteRT Support for Intel NPUs via OpenVINO" in May 2026. On Windows the supported NPU vendor is Intel.

The Arm integration targets Arm Mali GPUs that support the ML Extensions for Vulkan. It is a work in progress: the compiler plugin only accepts the JIT flow, participating in partitioning only when `enable_just_in_time` is true, and the dispatch implementation does not yet execute models. It selects an operation only when all its tensors are Bool, 8 to 32-bit integers, Float16 or Float32, matching the TOSA PRO-INT and PRO-FLOAT profiles.

## Enabling the NPU from an Android app

In the Kotlin CompiledModel API the NPU is requested with `CompiledModel.Options(Accelerator.NPU)` and an environment created with `Environment.create(context, BuiltinNpuAcceleratorProvider(context))`, used when `BuiltinNpuAcceleratorProvider(context).isDeviceSupported()` is true. NPU-only options compile as NPU plus CPU, so unsupported ops fall back to the CPU. On a supported device the provider sets the dispatch and compiler-plugin library directories to the app's native library directory, and the context overload enables the compiler cache under `context.cacheDir`.

For JIT on Qualcomm, the vendor shared libraries are packaged in `app/src/main/jniLibs/arm64-v8a/`, for example `libLiteRtDispatch_Qualcomm.so`, `libLiteRtCompilerPlugin_Qualcomm.so`, `libQnnHtp.so` and `libQnnSystem.so`. The `litert` AAR's manifest already declares `libcdsprpc.so` as an optional native library, for Qualcomm FastRPC, together with the OpenCL and Google Tensor entries. For peak NPU speed with a Float32 model, AI Edge Quantizer can produce a full-integer INT8 model with the `static_wi8_ai8` recipe, which needs a few representative inputs for calibration. NNAPI, the older Android route to NPUs, is deprecated since Android 15 and has no entry point in LiteRT 2.x.

## Portability, why one device is not enough

GPU compilers differ per vendor, so a compile ceiling on one chip may not exist on another. One device proves correctness, not portability: "runs on Android GPUs" means a device matrix, recorded as one row per device, either on your own devices or on a farm service such as AI Edge Portal. Host checks exercise the host GPU, while the device has its own shader compiler, its own precision behavior and its own memory ceiling, and every documented device-only failure was hit by a model that had already passed on the host.

One runtime version also proves results only for that version. Ops have been dropped between minor versions, and mixing accelerator and core libraries across versions silently falls back to CPU. To classify a miscompute, run the same graph across precision, buffer-storage and backend options: a bit-identical wrong result across all of them places the bug in the shared graph-compilation layer and rules out precision and storage in one pass.
