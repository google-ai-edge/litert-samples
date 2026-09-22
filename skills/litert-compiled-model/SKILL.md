---
name: litert-compiled-model
description: Runs a .tflite model in an Android app with the LiteRT CompiledModel API (Kotlin) on CPU, GPU or NPU. Use this skill to add on-device inference for a classic ML model (vision, audio, embeddings), to choose the accelerator, to manage input and output buffers and their lifecycle, or to migrate code from the TensorFlow Lite Interpreter API.
license: Apache-2.0
metadata:
  author: Google LLC
  last-updated: '2026-09-22'
  keywords:
  - LiteRT
  - CompiledModel
  - tflite
  - GPU
  - NPU
---

This skill provides step-by-step guidance for running a `.tflite` model with the LiteRT CompiledModel API in Android apps. The CompiledModel API is the current LiteRT runtime API. The Interpreter API (`org.tensorflow.lite.Interpreter`, `GpuDelegate`, the `tensorflow-lite*` and `litert-gpu` 1.x artifacts) is maintained for backward compatibility only and is not used here.

## Prerequisites

- `minSdk` 23 or higher.
- One dependency in the app-level `build.gradle`: `implementation("com.google.ai.edge.litert:litert:2.2.0")` (Google Maven). The AAR bundles the runtime and the GPU accelerator (`libLiteRt.so`, `libLiteRtClGlAccelerator.so`); do not add `litert-gpu`, `tensorflow-lite` or `tensorflow-lite-gpu`.
- NPU needs the vendor runtime libraries in addition and a model compiled for that NPU; see [NPU](references/npu.md).
- Keep the model uncompressed in assets: `androidResources { noCompress += "tflite" }`.

## Detailed steps

### 1. Load the model and choose the accelerator

```kotlin
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

val model = CompiledModel.create(context.assets, "segmenter.tflite", CompiledModel.Options(Accelerator.GPU))
```

`CompiledModel.create(assetManager, assetName, options)` or `CompiledModel.create(filePath, options)`. `CompiledModel.Options(vararg accelerators)`; `Options.CPU` is the default. `Accelerator.GPU` compiles the whole graph for the GPU and throws `LiteRtException` when an op is unsupported; it does not fall back to the CPU silently. Catch the exception, show it, and only then decide on a CPU run: a silent fallback hides a 10x slowdown.

### 2. Create the buffers once, run, read

```kotlin
class Segmenter(context: Context) : AutoCloseable {
    private val model = CompiledModel.create(context.assets, "segmenter.tflite", CompiledModel.Options(Accelerator.GPU))
    private val inputs = model.createInputBuffers()
    private val outputs = model.createOutputBuffers()

    fun infer(input: FloatArray): FloatArray {
        inputs[0].writeFloat(input)
        model.run(inputs, outputs)
        return outputs[0].readFloat()
    }

    override fun close() {
        inputs.forEach { it.close() }
        outputs.forEach { it.close() }
        model.close()
    }
}
```

The buffers are fields created once with the model and closed before it; do not create buffers per inference. `run()` enqueues the work; the readback (`readFloat()`) is the synchronization point, so time `run` and `readFloat` together. A ready copy of this lifecycle with the sharp edges documented is `utilities/common/kotlin/CompiledModelRunner.kt` in litert-samples.

### 3. Lifecycle

- `CompiledModel`, `TensorBuffer` and `Environment` are `AutoCloseable`. Close every buffer, then the model; buffers leak native memory even after the model is closed.
- Create, run and close on one background dispatcher, for example `Dispatchers.IO.limitedParallelism(1)`; never on the main thread.
- Warm up: run one inference on dummy input right after `create`. The first GPU run includes shader compilation, so never report the first run as latency.
- Several models in one process: create one `Environment` with `Environment.create(context)` and pass it as the last argument of every `CompiledModel.create` call.

### 4. NPU (optional)

`Environment.create(context, BuiltinNpuAcceleratorProvider(context))` plus `CompiledModel.Options(Accelerator.NPU)`; the model must be compiled for that NPU, ahead of time or on the device. Follow [NPU](references/npu.md).

### 5. Migrate from the Interpreter API

Replace `org.tensorflow:tensorflow-lite*` and `com.google.ai.edge.litert:litert-gpu` with the single `litert` 2.x dependency; `Interpreter` becomes `CompiledModel`, `GpuDelegate` becomes `Accelerator.GPU`, `ByteBuffer` I/O becomes `TensorBuffer.writeFloat` and `readFloat`, and `NnApiDelegate` becomes `Accelerator.NPU`. Follow [migration](references/migration.md).

### 6. Verify on device

Before building UI, reproduce the model's reference output on the accelerator you ship, with the same preprocessing (mean/std, RGB/BGR, NHWC). Follow [verification](references/verify.md). A complete app in this shape is `samples/litert/image_segmentation/kotlin_cpu_gpu/android` in litert-samples.

## Troubleshooting

- `LiteRtException` at create with `Accelerator.GPU`: an op the GPU accelerator does not support. Run on CPU to confirm the model, then fix the conversion; do not paper over it with a fallback.
- Wrong output and no error: preprocessing. Diff mean/std, channel order and layout against the model's export script before suspecting the runtime.
- Slow first frame: no warm-up, or the model is created per frame.

## Links

- LiteRT for Android (API overview and versions): https://ai.google.dev/edge/litert/android
- LiteRT repository (Kotlin API sources under `litert/kotlin`): https://github.com/google-ai-edge/litert
- LiteRT-LM, the runtime for language models, when the model is an LLM: https://github.com/google-ai-edge/LiteRT-LM
