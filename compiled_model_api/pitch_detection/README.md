# Pitch Detection with LiteRT — CREPE (on-device, real-time tuner)

An Android sample that runs [CREPE](https://github.com/marl/crepe) monophonic pitch (f0) estimation
(MIT) on device with the LiteRT `CompiledModel` API. A 1024-sample (16 kHz) window → activations over
**360 pitch bins** (20 cents each, ~C1–B7); the host decodes them to a frequency and the nearest
musical note. The app is a **real-time tuner** — it listens to the mic and shows the live note and how
many cents flat/sharp you are.

```
frame[1,1024] (16 kHz, per-frame zero-mean/unit-var) →[GPU CNN]→ activations[1,360] →[host]→ Hz → note
```

## Model

| Stage | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| frame norm (Kotlin) | mic → frame[1,1024], zero-mean/unit-var | CPU |
| CREPE (full) | frame[1,1024] → activations[1,360] | **GPU** |
| decode (Kotlin) | 360 bins → Hz → note + cents | CPU |

The CNN loads with `CompiledModel.create(...)` on `Accelerator.GPU`. fp16, converted with
[litert-torch](https://github.com/google-ai-edge/litert) — fp16 tflite-vs-torch corr **1.000000**. On a
Pixel 8a (Tensor G3): **49/49** nodes on `LITERT_CL`, **1 partition** (single graph, no CPU fallback);
~75 ms/frame; self-test (synthesized 440 Hz) → A4, 440.4 Hz.

## Why it's GPU-clean — the simplest conversion, zero patches

The whole network is a **pure CNN**: 6× {zero-pad → `Conv2d` → ReLU → `BatchNorm` → `MaxPool`} +
permute/reshape (≤4D) + `Linear` + `sigmoid`. No banned ops (the asymmetric "same" padding is a constant
zero-pad → native `PAD`, not `GATHER`; no GELU / TransposeConv / dilated conv), and per-frame
normalization keeps activations ~O(1) so there is **no fp16-on-Mali precision issue** (banned NONE, >4D 0).
See [`conversion/`](conversion/).

## Decode

CREPE bin → cents: `cents = 20·bin + 1997.3794…`; `Hz = 10·2^(cents/1200)`. The pitch is the
activation-weighted average over ±4 bins around the peak (torchcrepe `weighted_argmax`); the peak
activation is the confidence (the tuner shows "—" below 0.5). Nearest note from `midi = 69 + 12·log2(Hz/440)`.

## Run

1. Build the tflite with `conversion/build_crepe.py` (or get it from Hugging Face —
   [litert-community/CREPE-pitch-LiteRT](https://huggingface.co/litert-community/CREPE-pitch-LiteRT)).
2. Build/install the app, then push the model into its private storage:
   ```bash
   ./kotlin_cpu_gpu/android/install_to_device.sh <dir-with-the-tflite>
   ```
3. Launch **Pitch Detection** — it self-tests on a synthesized 440 Hz tone (→ A4), then **Start tuner**
   opens the mic and shows the live note + cents gauge. Uses `AudioSource.UNPROCESSED` (no AGC/NS) for
   accurate pitch; the capture loop drains backlog each cycle so the reading stays current.

## Files

- `kotlin_cpu_gpu/android/` — the Android app (`PitchDetector.kt` = the GPU CNN + per-frame norm +
  bin→Hz decode, `MainActivity.kt` = 440 Hz self-test + live mic tuner).
- `conversion/` — the litert-torch conversion + parity script (`build_crepe.py`).

Upstream: [marl/crepe](https://github.com/marl/crepe) (ICASSP 2018, MIT); PyTorch weights via
[torchcrepe](https://github.com/maxrmorrison/torchcrepe) (MIT).
