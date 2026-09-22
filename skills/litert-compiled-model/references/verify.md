# Verify the model on the device

## Reference first

Dump the source model's output for one real input and one fixed-seed random input on the workstation (a model recipe in litert-samples ships this as `dump_*_ref.py`). These dumps are the truth for every device run.

## Device CPU, then the accelerator

Run the same inputs with `Accelerator.CPU` first. If that already differs from the reference, the problem is the conversion, not the device. Then run `Accelerator.GPU` (or NPU) and compare three ways: numeric (correlation and max abs diff against the dump), task (argmax match, IoU, token match), artifact (the mask, audio or text a person can judge).

## Tells

- GPU output bit-identical to the device CPU output: the model almost certainly did not run on the GPU. A genuine GPU fp16 run drifts in the last digits.
- First run slow: shader compilation. Warm up once and never quote the first run.
- Timing `run()` alone: it enqueues. Time `run` and `readFloat` together.

## Device-only failures

| Symptom | Cause and fix |
|---|---|
| GPU create fails for a model that runs on the workstation | a whole-graph compile ceiling; split the model at a block boundary |
| NaN or garbage only on the GPU | an fp16 range break; an fp16-safe rewrite in the conversion, or keep that block on CPU |
| the process dies loading a large float model | a memory ceiling; quantize or split first |

## Record

Device, LiteRT version, accelerator, correlation, max abs diff and the task result, one row per device in the model's README. The full procedure is `skills/on-device-verification` in litert-samples.
