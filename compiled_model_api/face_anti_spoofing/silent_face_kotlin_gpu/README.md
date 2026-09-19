# Silent-Face Anti-Spoofing (MiniFASNetV2) — Face liveness (LiteRT CompiledModel GPU)

Real-time **face liveness / anti-spoofing** running **fully on the LiteRT `CompiledModel`
GPU** delegate. [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
detects **presentation attacks** — a printed photo or a replayed screen shown to the camera
— so a live face passes and a fake is rejected. The anti-fraud building block for face login
/ e-KYC. Tiny (**1.85 MB**), ~5 ms/frame on a Pixel 8a.

- **Model:** [litert-community/Silent-Face-Anti-Spoofing-LiteRT](https://huggingface.co/litert-community/Silent-Face-Anti-Spoofing-LiteRT)
- **Weights:** [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) · Apache-2.0
- **Input:** `[1, 3, 80, 80]` NCHW, **BGR**, `x/255` (a face crop)
- **Output:** `[1, 3]` softmax — class 1 = live, 0 & 2 = spoof (print / replay)

## How it works

MiniFASNetV2 is a pure CNN → fully GPU-compatible (**168/168 nodes on the delegate, 1
partition**; device corr 1.0, ~5 ms) with **zero patches** (PReLU lowers to relu ops).
CPU-exact vs PyTorch (corr 1.0). Live score = `output[1]`; `argmax == 1` → live.

## Run

```bash
cd android
./gradlew :app:installDebug
```

The 1.85 MB `silentface.tflite` is bundled in `app/src/main/assets/`. The sample scores the
bundled face photo at launch and lets you pick another image from the gallery — since a photo
is itself a presentation attack, it is correctly flagged **SPOOF**; a live camera capture would
score **LIVE**. Adapt `MainViewModel.kt` to feed a detected face crop from the camera.

## App architecture

The UI is **MVVM + Jetpack Compose** (Compose Material). `LivenessDetector` owns the LiteRT
`CompiledModel`; `MainViewModel` runs it on a single confined worker
(`Dispatchers.Default.limitedParallelism(1)`), draws the colored verdict border onto a copy of
the input, and publishes an immutable `UiState`. `MainActivity` is a thin Compose host that
collects the state and renders `LivenessScreen`. On launch the bundled photo is scored; the
gallery picker feeds additional images.

| File | Role |
| --- | --- |
| `LivenessDetector.kt` | LiteRT `CompiledModel` (GPU) wrapper — face crop in, `[print, live, replay]` softmax + ms out |
| `MainViewModel.kt` | Owns the detector, runs inference off the main thread, annotates the result, exposes `UiState` |
| `UiState.kt` | Immutable snapshot: `resultImage`, `resultText`, `inferenceTimeMs`, model/processing/error flags |
| `MainActivity.kt` | Thin `ComponentActivity` host: view model + gallery picker + Compose theme |
| `view/LivenessScreen.kt` | Status header, verdict detail text, image picker, annotated result image |
| `view/Theme.kt`, `view/Color.kt` | Compose Material theme colors |
| `ImageUtils.kt` | Asset/gallery bitmap decoding (EXIF-oriented) helpers |

## Convert

See [`conversion/`](conversion/) — `build_silentface.py` uses the Apache-2.0 weights and
converts with litert-torch.
