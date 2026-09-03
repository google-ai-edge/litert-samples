# Audio Source Separation with LiteRT — TIGER-DnR (on-device)

An Android sample that runs [TIGER](https://github.com/JusperLee/TIGER) (ICASSP 2025) cinematic sound separation on device with the LiteRT `CompiledModel` API, **fully on the GPU**: pick a clip (movie scene, game, vlog — any audio/video file) or record with the mic, and the app splits it into **Dialogue / Sound effects / Music** stems you can play back individually.

```
clip →[decode+resample 44.1 kHz mono]→ 12.06 s chunks →[3 TIGER graphs, GPU]→ complex masks →[host iSTFT+OLA]→ 3 stems
```

## Models

| Stage | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| TIGER dialog / effect / music (~1.4 M params each) | wav[1,534016] (reflect-padded chunk) → spec real+imag [1,3,1025,1040] | **GPU** |
| iSTFT + overlap-add (Kotlin) | spec → stem waveform | CPU |

The three sibling graphs load with `CompiledModel.create(...)` on `Accelerator.GPU` (one at a time; each sweeps all chunks). fp16, 16.1 MB each, converted with litert-torch. On a Pixel 8a (Tensor G3): **23 974 / 23 974** nodes on `LITERT_CL` (1 partition, no CPU fallback), device-vs-PyTorch waveform corr **0.99987**, ~4.5 s per 12.06 s chunk per graph. The **STFT runs inside the graph** as a windowed-DFT `Conv1d`; per the DnR convention each graph contributes one stem (dialog = source 2, effect = source 1, music = source 0).

## Why it's GPU-clean — the re-authoring

TIGER has no RNN and no gather, but the reference implementation folds time/band axes into the batch (`(B·T, N, band)` `Conv1d`), builds 6-D mask tensors, and leans on `adaptive_avg_pool1d` / nearest `interpolate` — all numerically re-authored (see [`conversion/`](conversion/)): folded batches become 4D `(1,k)`-`Conv2d`; the chunk length is chosen so T=1040 divides by 16, making every pool a uniform `AVERAGE_POOL_2D` and every nearest resize an exact integer repeat; the non-uniform 57-band axis uses constant one-hot/averaging `FULLY_CONNECTED` matrices; attention runs as per-head batch-1 3D matmuls with 1/√d folded into Q; `PReLU` → `relu(x) − w·relu(−x)`; the 6-D mask view → static channel slices. Two fp16-on-GPU fixes are exact-equivalent: norm eps raised to 1e-4 (1e-8 underflows to 0 in fp16 → 0/0 NaN on silent bands) and the mask head rewritten without dim-1 broadcast MULs.

## Run it

1. Get the models: build with [`conversion/build_tiger.py`](conversion/build_tiger.py) or download `tiger_{dialog,effect,music}_fp16.tflite` from [litert-community/TIGER-DnR-LiteRT](https://huggingface.co/litert-community/TIGER-DnR-LiteRT).
2. Install the app: `cd kotlin_cpu_gpu/android && ./gradlew :app:installDebug`
3. Push the models: `./install_to_device.sh <dir-with-the-tflites>`
4. Pick a clip (or record 15 s) and play the separated stems (first 32 s are processed).

Upstream: [JusperLee/TIGER](https://github.com/JusperLee/TIGER) (MIT), weights [JusperLee/TIGER-DnR](https://huggingface.co/JusperLee/TIGER-DnR) (Apache-2.0), trained on the openly-built [DnR](https://github.com/darius522/dnr-utils) dataset.
