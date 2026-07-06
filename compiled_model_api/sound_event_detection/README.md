# Sound Event Detection with LiteRT — PANNs CNN14 (AudioSet, on-device)

An Android sample that runs [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) **CNN14** (`Cnn14_mAP=0.431`, Apache-2.0 code / CC-BY-4.0 weights) general sound-event tagging on device with the LiteRT `CompiledModel` API. Given ~10 s of audio it predicts probabilities over the **527 [AudioSet](https://research.google.com/audioset/) classes** — speech, music, instruments, animals, vehicles, alarms, household sounds, and so on. AudioSet tagging is **multi-label**: several tags can be high at once. The app self-tests on a bundled clip on launch, then records 10 s from the mic and lists the top tags.

```
waveform[320000] (32 kHz) →[host: log-mel]→ logmel[1,1,1001,64] →[GPU: CNN14]→ probs[1,527] (sigmoid)
```

## Model

| Stage | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| log-mel front-end (Kotlin) | waveform[320000] → logmel[1,1,1001,64] | CPU |
| CNN14 body | logmel[1,1,1001,64] → probs[1,527] | **GPU** |

The CNN body loads with `CompiledModel.create(...)` on `Accelerator.GPU`. fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — fp16 tflite-vs-torch corr **1.000000**. On a Pixel 8a (Tensor G3): **45/45** nodes on `LITERT_CL`, **1 partition** (single graph, no CPU fallback); ~124 ms GPU + ~99 ms host log-mel ≈ **0.22 s** per 10 s clip; bundled-clip self-test → top tag "Speech" (matches PyTorch).

## Why the log-mel is host-side

PANNs builds its spectrogram with **torchlibrosa**, whose STFT is a *DFT-as-Conv1d* — so there is **no FFT op** and the whole raw-audio→tags graph is almost GPU-clean; the only blocker is the STFT centering reflect-pad (one `GATHER_ND`, removable via `pad_mode='constant'`, corr 1.0). **But** litert-torch lowers the giant 1024-tap DFT-conv incorrectly (fp32 tflite corr ≈ 0.19), and the power spectrum `|STFT|²` (~1e6) **overflows fp16 on Mali → NaN**. So the spectral front-end is computed on the CPU (matched to torchlibrosa exactly — host log-mel vs torch corr **1.000000**, max|d| 0.0017), and only the CNN body (`bn0` + 6 conv blocks + pooling + 2 FC + sigmoid — a pure CNN, corr **1.000000** in fp32 and fp16) rides the GPU. See [`conversion/`](conversion/).

## Run

1. Build the tflite with `conversion/build_panns.py` (or get it from Hugging Face — [litert-community/PANNs-CNN14-AudioSet-LiteRT](https://huggingface.co/litert-community/PANNs-CNN14-AudioSet-LiteRT)).
2. Build/install the app, then push the model into its private storage:
   ```bash
   ./kotlin_cpu_gpu/android/install_to_device.sh <dir-with-the-tflite>
   ```
3. Launch **Sound Event Detection** — it compiles the GPU shaders (~first launch), self-tests on the bundled clip, then the Record button captures 10 s from the mic and lists the top tags with a bar chart.

## Files

- `kotlin_cpu_gpu/android/` — the Android app (`AudioTagger.kt` = the GPU CNN + label decode, `MelSpectrogram.kt` = the host-side log-mel matched to torchlibrosa, `MainActivity.kt` = bundled-clip self-test + mic Record button).
- `conversion/` — the litert-torch conversion + host-mel validation script (`build_panns.py`).

Upstream: [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) — code Apache-2.0, weights CC-BY-4.0 ([Zenodo](https://zenodo.org/record/3987831)). AudioSet ontology © Google, CC-BY-4.0.
