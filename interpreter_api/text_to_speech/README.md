# Text-to-Speech (Kokoro-82M) — LiteRT

On-device text-to-speech with [Kokoro-82M](https://huggingface.co/litert-community/Kokoro-82M)
(StyleTTS2 + ISTFTNet) running on the LiteRT CPU runtime via the Interpreter API.

> ⚠️ **Preview — fixed-length, FP32, CPU.** This build is exported at a fixed phoneme
> length, so it reproduces one baked utterance. The neural graph runs everything up to the
> magnitude/phase spectrogram; the final iSTFT overlap-add runs on the host in Kotlin
> (~2 ms, no learned weights, numerically exact) to sidestep a converter constant-dedup bug.
> Variable-length / quantized is gated on converter fixes (dynamic-LSTM, litert-torch #1063).

## Model

| Component | Repo | File |
|---|---|---|
| TTS body | [litert-community/Kokoro-82M](https://huggingface.co/litert-community/Kokoro-82M) | `kokoro_82m_fixedlen_fp32.tflite` (~338 MB) |

The host-side iSTFT bases (`istft_Wr_f32.bin`, `istft_Wi_f32.bin`) and the baked demo input
(`ref_input_ids_i64.bin`, `ref_s_f32.bin`) ship in `android/app/src/main/assets/`. A neural
grapheme-to-phoneme front-end for free text is available separately at
[litert-community/Kokoro-G2P-en-US](https://huggingface.co/litert-community/Kokoro-G2P-en-US).

## Build & run

1. Build and install the app (this creates the app's data dir):
   ```
   cd android && ./gradlew :app:installDebug
   ```
2. Stage the model into the app's files dir (it is not bundled in the APK):
   ```
   ./scripts/install_model.sh
   ```
   This downloads the model from Hugging Face and pushes it onto the connected device.
3. Launch the app and tap **Synthesize**. It reports inference time, iSTFT time, and RTF.

## How it works

`KokoroTtsHelper` runs the `.tflite` graph (`input_ids` + voice style vector → magnitude /
phase spectrogram + predicted durations), then reconstructs the 24 kHz waveform with a
host-side iSTFT overlap-add (`filter_length = 20`, `hop = 5`, `freq_bins = 11`). In this FP32 CPU
preview it measures **RTF ≈ 1.8 (~0.55× realtime) on a Pixel 8a** — about 6.6 s of compute for
3.7 s of audio, i.e. slower than realtime; quantization is the planned next step. The app shows
the measured inference time, iSTFT time, and RTF live.
