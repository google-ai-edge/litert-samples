# Migrating from the Interpreter API

## Dependencies

Remove `org.tensorflow:tensorflow-lite`, `tensorflow-lite-gpu`, `tensorflow-lite-support`, `tensorflow-lite-select-tf-ops`, and the 1.x `com.google.ai.edge.litert:litert-gpu` and `litert-support`. Add `com.google.ai.edge.litert:litert:2.2.0`.

Support-library preprocessing (`ImageProcessor`, `ResizeOp`, `NormalizeOp`) has no 2.x equivalent: scale with `androidx.core.graphics.scale` or `Bitmap.createScaledBitmap` and normalize into a `FloatArray` yourself. `utilities/common/kotlin/ImageTensor.kt` in litert-samples is a ready copy (mean/std, NCHW/NHWC, RGB/BGR, letterbox).

## API map

| Interpreter API | CompiledModel API |
|---|---|
| `org.tensorflow.lite.Interpreter` | `com.google.ai.edge.litert.CompiledModel` |
| `Interpreter(modelBuffer, Interpreter.Options().addDelegate(GpuDelegate()))` | `CompiledModel.create(context.assets, "model.tflite", CompiledModel.Options(Accelerator.GPU))` |
| `NnApiDelegate()` | `Accelerator.NPU` with an `Environment` (see npu.md) |
| `ByteBuffer` inputs and output arrays | `TensorBuffer` from `createInputBuffers()` / `createOutputBuffers()`, `writeFloat` / `readFloat` |
| `interpreter.run(input, output)` | `model.run(inputs, outputs)` |
| `interpreter.getInputTensor(0)` | `model.getInputTensorType(name)`, `model.getInputBufferRequirements(name)` |
| `interpreter.close()` | close every `TensorBuffer`, then `model.close()` |
| Flex delegate (`select-tf-ops`) | not available; re-export the model without TF ops (litert-torch) |
| C++: `tensorflow/lite/interpreter.h` | `litert/cc/litert_compiled_model.h` |

## Order of work

1. Like for like: same model, same preprocessing, `CompiledModel` on CPU. Compare the output with the old build on one fixed input (see verify.md).
2. Then the accelerator: `Accelerator.GPU`. A `LiteRtException` at create means an unsupported op in the model, not a code bug; fix the conversion, do not fall back silently.
3. Then lifecycle: buffers as fields, one dispatcher, warm-up.

The full procedure with a self-test template is `skills/litert-compiled-model-migration` in litert-samples.
