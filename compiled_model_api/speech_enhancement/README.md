# Speech Enhancement with LiteRT — CMGAN (on-device noise suppression)

An Android sample that runs [CMGAN](https://github.com/ruizhecao96/CMGAN) (TASLP 2024) speech enhancement on device with the LiteRT `CompiledModel` API, **fully on the GPU**: record in a noisy place (or pick any audio/video clip) and A/B the original vs the denoised result.

```
wav (2 s chunks, host reflect-pad) →[GPU: DFT-conv STFT → mag^0.3 → dense encoder →
4× (time + freq conformer) → mask + complex decoders]→ enhanced compressed spectrogram
→[host: mag^(1/0.3) + iSTFT + overlap-add]→ denoised wav
```

## Model

| Model | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| CMGAN (1.83 M, VoiceBank-DEMAND) | wav[1,32400] → spec real+imag [1,1,321,201] | **GPU** |

Loads with `CompiledModel.create(...)` on `Accelerator.GPU`: **1 651 / 1 651 nodes on `LITERT_CL`** (full residency, 1 partition), **~20 ms per 2 s chunk** (RTF ≈ 0.01), fp16 4.2 MB ([litert-community/CMGAN-LiteRT](https://huggingface.co/litert-community/CMGAN-LiteRT)). SI-SNR +7.2 dB on a 6.6 dB noisy sample (PyTorch +9.6 dB), device-vs-PyTorch waveform corr 0.997. The **STFT and the power compression run inside the GPU graph** — the host does only reflect-padding, un-compression, inverse STFT and overlap-add ([`NoiseSuppressor.kt`](kotlin_cpu_gpu/android/app/src/main/java/com/google/ai/edge/examples/speech_enhancement/NoiseSuppressor.kt)).

## Why it's GPU-clean — the re-authoring

All numerically-equivalent (see [`conversion/`](conversion/)): the phase path **cancels algebraically** (`mask·mag·cos(∠x) ≡ mask·x_r` — no atan2/cos/sin in-graph); Shaw relative positional embedding (an Embedding lookup → GATHER) is baked to a constant for the fixed chunk and applied as a 2D `FULLY_CONNECTED` plus a pad/reshape **skew** realignment; the conformer's folded batches become batch-1 4D tensors (channel-LayerNorm per position, Linears as 1×1 convs, depthwise Conv1d as `(1,k)` Conv2d, heads folded into the 3D-BMM batch); `mag^0.3` → `exp(0.3·ln(·))`; `SPConvTranspose2d`'s 5-D view → an exact 4D reshape chain; InstanceNorm → safe spatial norm; eval-mode BatchNorm → constant scale/shift; all norm eps ≥ 1e-4 (fp16 min-normal on the delegate).

## Run it

1. Get the model: build with [`conversion/build_cmgan.py`](conversion/build_cmgan.py) or download `cmgan_fp16.tflite` from [litert-community/CMGAN-LiteRT](https://huggingface.co/litert-community/CMGAN-LiteRT).
2. Install the app: `cd kotlin_cpu_gpu/android && ./gradlew :app:installDebug`
3. Push the model: `./install_to_device.sh <dir-with-the-tflite>`
4. Record noisy audio (unprocessed mic source) or pick a clip, then A/B Noisy vs Enhanced.

Upstream: [ruizhecao96/CMGAN](https://github.com/ruizhecao96/CMGAN) (MIT), trained on VoiceBank-DEMAND. Please cite Cao et al. (Interspeech 2022 / TASLP 2024).
