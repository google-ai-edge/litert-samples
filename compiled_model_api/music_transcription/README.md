# Music Transcription with LiteRT — Basic Pitch (on-device audio-to-MIDI)

An Android sample that runs [Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify, ICASSP 2022) music transcription on device with the LiteRT `CompiledModel` API, **fully on the GPU — including the conv-based CQT front-end**: play an instrument (or sing, or pick a clip) and see the transcribed notes on a piano roll.

```
wav[1,43844] (2 s @ 22.05 kHz) →[GPU: 9-octave conv-CQT → NormalizedLog → harmonic stacking →
3-branch CNN]→ contour[172,264] + note[172,88] + onset[172,88] →[host]→ note events → piano roll
```

## Model

| Model | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| Basic Pitch nmp (0.84 MB fp32) | wav[1,43844] → 3 posteriorgrams | **GPU** |

Loads with `CompiledModel.create(...)` on `Accelerator.GPU`: **241 / 241 nodes on `LITERT_CL`** (full residency, 1 partition), **~4.4 ms per 2 s window** ([litert-community/Basic-Pitch-LiteRT](https://huggingface.co/litert-community/Basic-Pitch-LiteRT)). The torch re-implementation used for conversion is **bit-exact vs the official ONNX** (corr 1.000000); on-device note-event F1@0.5 is 0.98 vs the official model.

## Why it's GPU-clean — the re-authoring

Rebuilt from the official ONNX's constants (see [`conversion/`](conversion/)): the CQT2010v2 front-end is a 9-octave chain of shared 36×256-tap kernel convolutions with a lowpass stride-2 downsample between octaves — reflect padding becomes an anti-diagonal-constant `FULLY_CONNECTED` (no gather), PACK/stacking becomes concat + static slices. Two exact fp16-on-GPU fixes: a **post-log clamp** `clamp(10·log10(p+1e-10), min=-100)` (desktop no-op; recovers `log(0) = -inf` after the delegate's fp16 arithmetic flushes the 1e-10 floor) and the **per-bin CQT norm folded into per-octave kernel copies** (magnitude is linear in the kernel scale).

## Run it

1. Get the model: build with [`conversion/build_bp.py`](conversion/build_bp.py) or download `basicpitch.tflite` from [litert-community/Basic-Pitch-LiteRT](https://huggingface.co/litert-community/Basic-Pitch-LiteRT).
2. Install the app: `cd kotlin_cpu_gpu/android && ./gradlew :app:installDebug`
3. Push the model: `./install_to_device.sh <dir-with-the-tflite>`
4. **Record & transcribe** (play some notes) or pick a clip → piano roll.

## App architecture

The app is MVVM + Jetpack Compose (Material 2). `MainActivity` is a thin Compose host; all model and audio work lives in `MainViewModel`, which exposes a single immutable `UiState`. The piano roll is not a custom `View` — the note rects are drawn onto a fixed-size mutable `Bitmap` (`renderPianoRoll`, same rect math as the old `PianoRollView.onDraw`) and shown via a Compose `Image`. Recording asks for `RECORD_AUDIO` with a Compose `RequestPermission` launcher; clips are picked with a `GetContent("audio/*")` launcher.

| File | Role |
| :-- | :-- |
| `MainActivity.kt` | Compose host; RECORD_AUDIO permission launcher + `GetContent("audio/*")` picker |
| `MainViewModel.kt` | Owns `Transcriber`; mic capture, clip decode, transcription, piano-roll `Bitmap` render; one confined worker |
| `UiState.kt` | Immutable UI snapshot (`isModelReady`, `isProcessing`, `statusMessage`, `resultImage`, `errorMessage`) |
| `view/PianoRollScreen.kt` | Status header + Record / Pick buttons + result `Image` |
| `view/Theme.kt`, `view/Color.kt` | Compose Material 2 theme |
| `Transcriber.kt` | Basic Pitch `CompiledModel` GPU inference (unchanged) |
| `AudioDecoder.kt` | Decodes a picked clip to mono 22.05 kHz float PCM (unchanged) |

Upstream: [spotify/basic-pitch](https://github.com/spotify/basic-pitch) (Apache-2.0). Please cite Bittner et al., ICASSP 2022.
