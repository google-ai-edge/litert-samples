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

The 1.85 MB `silentface.tflite` is bundled in `app/src/main/assets/`. The sample runs on a
bundled face photo — since a photo is itself a presentation attack, it is correctly flagged
**SPOOF**; a live camera capture would score **LIVE**. Adapt `MainActivity.kt` to feed a
detected face crop from the camera.

## Convert

See [`conversion/`](conversion/) — `build_silentface.py` uses the Apache-2.0 weights and
converts with litert-torch.
