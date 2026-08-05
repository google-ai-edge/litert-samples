# KittenTTS nano → LiteRT conversion

Re-authors [KittenTTS nano 0.8](https://github.com/KittenML/KittenTTS) (StyleTTS2 + ISTFTNet +
mini-ALBERT, 15M params, Apache-2.0) in TF/Keras from the official ONNX weights and converts it
with the TFLite converter into three **dynamic-sequence-length** CPU graphs. The TF path is what
makes the dynamic length possible: the official converter emits fused dynamic-length TFLite LSTM
kernels for the model's five BiLSTMs, which the torch export path cannot keep dynamic (torch.export
specializes the time axis, and even conv-only graphs bake the trace length into internal RESHAPEs).

```
kitten_predictor.tflite : ids[1,N] i32, style[1,256], speed[1] -> d[1,N,256], t_en[1,N,128], durations[N] i32
host                    : en = repeat(d, durations), asr = repeat(t_en, durations)
kitten_prosody.tflite   : en[1,T,256], style[1,256] -> f0[1,2T], n[1,2T], har[1,120T+1,22]
kitten_vocoder.tflite   : asr[1,T,128], f0, n, har, style -> wav[1,600T] @ 24 kHz
```

## Workflow

Two Python environments, matching the two script groups:

1. **torch/ORT venv** (`onnxruntime`, `torch`, `phonemizer`, `espeakng-loader`, `kittentts`):

   ```bash
   # Download the model first: huggingface.co/KittenML/kitten-tts-nano-0.8 -> ../models/nano-0.8-fp32
   python make_goldens.py      # reference inputs/intermediates/waveform + full weight dump
   ```

2. **TF venv** (`tensorflow`, `tf-keras`, `ai-edge-litert`, `numpy`):

   ```bash
   python build_kitten_tflite.py         # fp32 graphs -> ../out/
   python build_kitten_tflite.py --fp16  # fp16 graphs (what the app uses, half the size)
   python check_stages.py                # per-module numeric check vs the goldens
   python check_decoder.py
   python verify_kitten_litert.py        # end-to-end: accuracy, dynamic lengths, streaming, RTF
   python export_voices.py               # voices.npz -> ../out/voices.bin for the app
   ```

`verify_kitten_litert.py` needs both stacks (LiteRT + the espeak frontend for extra sentences);
run it in the torch venv with `ai-edge-litert` installed.

## Verification

The reference model is stochastic (SineGen draws a random initial harmonic phase and additive
noise every run), so the fair bar is the ONNX's own run-to-run variability. On the golden
sentence (Mac M-series CPU):

| Comparison | log-mel corr | spec-conv |
| ---------- | ------------ | --------- |
| ONNX vs ONNX (two runs, same inputs) | 0.98327 | 0.0949 |
| **LiteRT fp32 vs ONNX (deterministic)** | **0.98388** | 0.1242 |
| LiteRT fp16 vs ONNX (deterministic) | 0.98231 | — |

The port sits inside the model's intrinsic noise floor; predicted durations are bit-identical to
the ONNX. (Raw-waveform correlation is not meaningful here: sub-0.5% f0 differences de-correlate
the waveform via accumulated sine phase while being inaudible and spectrally identical.) int8
dynamic-range quantization was tried and rejected (log-mel corr 0.913, durations change).

## Notes that generalize

- ONNX exporters value-deduplicate identical initializers *and* CSE-merge whole InstanceNorm
  nodes — map norm parameters by graph connectivity, not module name.
- `tf.nn.leaky_relu` defaults to α=0.2; torch defaults to 0.01.
- TFLite has no `ATAN` builtin — `atan2` is the supported route to the harmonic STFT phase.
- `tf_keras` (Keras 2) is required: Keras 3 leaves READ_VARIABLE resource ops in the graph.
- The deterministic SineGen fixes the reference's random initial phase / noise to zero (the
  reference produces a different waveform every run; zeroing selects one deterministic sample).
