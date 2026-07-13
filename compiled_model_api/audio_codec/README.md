# Neural Audio Codec with LiteRT — Mimi (Kyutai 2024, on-device, hybrid GPU/CPU)

An Android sample that runs [**Mimi**](https://huggingface.co/kyutai/mimi) (the Kyutai/Moshi streaming neural audio **codec**, 24 kHz, 12.5 Hz frame rate) end-to-end on device with the LiteRT `CompiledModel` API: it round-trips a clip (encode → quantize → decode) and plays **original vs. reconstructed** so you can A/B by ear. No network.

> Mimi is a **codec**, not TTS — it compresses audio to discrete codes and reconstructs it (the codec backbone used inside Moshi and other audio LMs). It is the sibling of [DAC](https://github.com/descriptinc/descript-audio-codec); the interesting part on-device is the GPU/CPU placement split below.

## Models

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| enc_conv (SEANet encoder) | audio[1,1,L] → feat[1,512,Se] | **GPU** |
| enc_tx (encoder-transformer + downsample) | feat[1,Se,512] → emb[1,512,Tc] | CPU |
| dec_tx (upsample + decoder-transformer) | emb[1,512,Tc] → conv_in[1,512,seq] | CPU |
| deconly (SEANet decoder) | conv_in[1,512,seq] → audio[1,1,L] | **GPU** |
| RVQ encode/decode (32 codebooks) | codes ↔ emb | CPU |

All graphs are loaded with `CompiledModel.create(...)`; the accelerator (`GPU`/`CPU`) is chosen per graph. fp16 weights, converted with [litert-torch](https://github.com/google-ai-edge/litert) — per-graph tflite-vs-torch corr **1.000000**, full round-trip corr **1.0** (codec quality floor on device). On a Pixel 8a (Tensor G3): enc_conv `189/189` + deconly `220/220` on the GPU delegate (`LITERT_CL`), transformers on XNNPACK; encode ≈ 0.49 s · decode ≈ 0.18 s for 2 s → **RTF ≈ 0.35**.

## Why hybrid (GPU convs, CPU transformers)

Every op in all four graphs is GPU-clean (re-authored — see [`conversion/`](conversion/)), and the SEANet convs are **fp16-exact on Mali** (decoder-only fed the exact transformer output = 48 dB SNR). But the decoder transformer's residual stream reaches **|x| = 27**, where the Mali GPU delegate's internal fp16 compute loses precision — full-GPU decode drops to ~12 dB on real speech. The transformer behaves **identically standalone and fused** on device, so this is fp16 **precision**, not a fusion artifact. The transformers are tiny (8 layers × 512, seq ~50), so running them on CPU is trivial and exact, while the heavy convs stay on the GPU. The split RVQ (Euclidean argmin + int64 indices that Mali rejects) runs on CPU. This is the standard codec deployment split.

```
audio →[GPU enc_conv]→ feat →[CPU enc_tx]→ emb →[CPU RVQ.encode]→ codes
      →[CPU RVQ.decode]→ emb →[CPU dec_tx]→ conv_in →[GPU deconly]→ audio
```

## Re-authoring (litert-torch, parity ~1.0)

`nn.GELU`(erf)→tanh-GELU · `MimiRotaryEmbedding`→baked cos/sin + rotate_half · causal/sliding mask→baked additive bias · `MimiLayerScale`→bake into the preceding Linear · `ConvTranspose1d` (depthwise upsample)→grouped `ZeroStuffConvT1d` (no `TRANSPOSE_CONV`) · `MimiConv1d` causal pad→baked constant `F.pad` · `nn.ELU`→`relu(x)−relu(1−exp(min(x,0)))` (SELECT-free) · downsample replicate-pad→ SLICE+CONCAT. Scripts in [`conversion/`](conversion/): `build_mimi.py` (op-check + parity + the standalone/tapped C33 test graphs), `build_hybrid_graphs.py` (the 4-graph deployment split), `mimi_rvq_validate_export.py` (RVQ + `mimi_rvq.bin`).

## Run

1. Build the graphs + RVQ weights with `conversion/build_hybrid_graphs.py` + `conversion/mimi_rvq_validate_export.py` (or get them from Hugging Face — *litert-community/Mimi*).
2. Build/install the app, then push the models into its private storage:
   ```bash
   ./kotlin_cpu_gpu/android/install_to_device.sh <dir-with-the-files>
   ```
3. Launch **Mimi Codec** — it compiles the GPU shaders (~10 s first launch), round-trips, and plays.

## Files

- `kotlin_cpu_gpu/android/` — the Android app (Compose + MVVM): `MimiCodec.kt` = 2 GPU + 2 CPU `CompiledModel`s, `MimiRvq.kt` = split RVQ on CPU, `MainViewModel.kt` = round-trip + A/B `AudioTrack` playback, `MainActivity.kt`/`view/CodecScreen.kt` = the UI.
- `conversion/` — the litert-torch conversion + RVQ export scripts.

Upstream: [kyutai/mimi](https://huggingface.co/kyutai/mimi) (CC-BY-4.0).
