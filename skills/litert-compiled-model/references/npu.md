# NPU with the CompiledModel API

LiteRT drives vendor NPUs through one interface: Google Tensor (ahead-of-time compiled models today), Qualcomm AI Engine Direct, MediaTek NeuroPilot, Samsung Exynos AI LiteCore and Intel OpenVINO (ahead-of-time and on-device compilation). Guide: https://ai.google.dev/edge/litert/next/npu

## What changes compared with CPU and GPU

- The model must be compiled for the NPU. Either ahead of time (AOT) with the LiteRT AOT compiler for each target SoC, delivered as a Google Play AI Pack, or on the device (JIT) from the plain `.tflite` in `assets/`, at a higher first-run cost. Google Tensor supports AOT only.
- The vendor runtime libraries ship with the app through Play Feature Delivery, one feature module per vendor and SoC generation (`litert_npu_runtime_libraries:qualcomm_runtime_v75`, `mediatek_runtime`, `google_tensor_runtime`, `samsung_runtime` in the sample), and their version must match the `litert` Maven version.
- NPU builds are arm64-v8a only: `ndk { abiFilters.add("arm64-v8a") }`. Qualcomm runtimes need `packaging { jniLibs { useLegacyPackaging = true } }`.

## Code

```kotlin
val provider = BuiltinNpuAcceleratorProvider(context)
val env = Environment.create(context, provider)
val options = if (provider.isDeviceSupported()) CompiledModel.Options(Accelerator.NPU) else CompiledModel.Options(Accelerator.GPU)
val model = CompiledModel.create(context.assets, "segmenter.tflite", options, env)
```

`BuiltinNpuAcceleratorProvider(context)` answers `isDeviceSupported()` from the SoC (Qualcomm, MediaTek, Samsung, Google Tensor on Android 16 and later) and points the environment at the bundled libraries. Decide the fallback order (NPU, then GPU, then CPU) in your own code and log which one is running; a failed NPU create must not look like a working app.

## Worked examples

- `samples/litert/image_segmentation/kotlin_npu/android`: AOT-compiled models delivered as an AI Pack.
- `samples/litert/image_segmentation/kotlin_npu/android_jit`: on-device compilation from the plain `.tflite`.
