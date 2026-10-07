# Verify the model on the device

## Reference first

Run the same `.tflite` on the workstation on one fixed input and keep the output: the Python package `ai-edge-litert` has the same CompiledModel API (`CompiledModel.from_file(path)`, then write, run, read). The `qwen3_tts` recipe in litert-samples ships such dumps as `dump_*_ref.py`. This output is the truth for every device run.

## Device CPU, then the GPU

Run the same input with `Accelerator.CPU` first and compare with the reference; a difference here is in the model file, not in the device. Then run `Accelerator.GPU` and compare twice: the numbers (max abs diff, correlation) and the task result (argmax, mask, text). The GPU computes in fp16, so the last digits move; the task result should not.

## Tells

- First run slow: shader compilation. The constructor's warm-up takes it; never quote the first run.
- On the GPU, time `run` and `readFloat` together: the output read is where the app waits for the GPU.

## Device-only failures

| Symptom | Cause and fix |
|---|---|
| `create` with `Accelerator.GPU` throws `Failed to compile model` | an op the GPU does not support (logcat names it: `Following operations are not supported by GPU delegate`): run the model on the CPU, or re-export it without that op |
| NaN or garbage only on the GPU | an fp16 range break in the model: run it on the CPU, or re-export it fp16-safe |
| the process dies loading a large float model | not enough memory: a quantized or split model |

## Record

Device, LiteRT version, accelerator, max abs diff and the task result, one line per device in the app's README. The procedure for the conversion side is `skills/on-device-verification` in litert-samples.
