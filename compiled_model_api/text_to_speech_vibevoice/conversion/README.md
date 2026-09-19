# VibeVoice-Realtime-0.5B → LiteRT conversion

`build_vibevoice.py` produces the four `.tflite` graphs and three binary assets used by the Android
sample, from the [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) source repo and the
[VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) checkpoint, with
[litert-torch](https://github.com/google-ai-edge/litert). Each graph is re-authored to be GPU-clean
(`CompiledModel`), and conversions print eager-vs-tflite parity.

## Environment

```bash
# 1. The VibeVoice source (standalone modeling code) and the checkpoint
git clone https://github.com/microsoft/VibeVoice.git
huggingface-cli download microsoft/VibeVoice-Realtime-0.5B --local-dir VibeVoice-Realtime-0.5B

# 2. A Python 3.10 conversion env with the LiteRT toolchain + the repo pin
pip install "torch==2.6.0" "transformers==4.51.3" diffusers accelerate safetensors soundfile
pip install litert-torch ai-edge-litert ai-edge-quantizer      # graph conversion
```

## Run

```bash
python build_vibevoice.py --stage all \
    --src   ./VibeVoice \
    --ckpt  ./VibeVoice-Realtime-0.5B \
    --out   ./out
```

Individual stages: `--stage {decoder,head,backbone,assets}`.

This writes, in `--out`:

| File | What |
|---|---|
| `vv_base_lm_kv_fp32.tflite` | 4-layer Qwen2 text LM, one AR step (packed KV cache), fp32 |
| `vv_tts_lm_kv_fp32.tflite` | 20-layer Qwen2 TTS LM, one AR step, fp32 |
| `vv_diffhead_fp16.tflite` | 4-layer DDPM diffusion head, one denoise step, fp16 |
| `vv_decoder_fp32.tflite` | σ-VAE conv decoder (fixed 128 frames), fp32 |
| `embed_tokens.f16` | fp16 token-embedding table (host lookup) |
| `glue.f32` + `glue_layout.json` | acoustic connector + type embedding + EOS classifier weights (with offsets) |
| `voice_en-Emma_woman.bin` | precomputed prompt KV cache = the voice preset |

Push them with [`../kotlin_cpu_gpu/android/install_to_device.sh`](../kotlin_cpu_gpu/android/install_to_device.sh).
The BPE tokenizer's `vocab.json` + `merges.txt` are bundled in the app assets, not built here.

## fp16 vs fp32

The two Qwen2 LMs and the σ-VAE decoder are exported **fp32**; only the tiny diffusion head is
exported **fp16**. Android ARM XNNPACK computes an fp16 graph in fp16, which collapses the 20-layer
LM residual stream to noise, so the LMs must ship fp32. The diffusion head is small and stays fp16
on the GPU (with `GpuOptions.Precision.FP32` it is exact and much smaller). `export_fp16` handles
both cases.

## Re-authoring → GPU-clean

The graphs are re-authored so every op compiles under `CompiledModel` (see `build_vibevoice.py`):

- **Token embedding is `GATHER`** (banned) → looked up on the host from the mmapped
  `embed_tokens.f16`; the LMs take `inputs_embeds`.
- **Autoregressive KV cache** → packed into a 4D `[1, L*nkv, Pmax, 64]` tensor (all layers on
  dim 1 to stay ≤4D); the current token's key/value is concatenated at the tail and an additive
  mask blanks the padding slots, so there is no in-graph scatter. Keys are stored post-RoPE.
- **`scaled_dot_product_attention`** → manual matmul + softmax; GQA (14 Q / 2 KV heads) expanded
  by `cat` (no `BROADCAST_TO`).
- **RoPE** cos/sin computed on the host and fed per step (position-dependent → not a constant).
- **RMSNorm** → max-normalized safe form (`safe_rms`, an fp16 sum-of-squares overflow guard).
- **σ-VAE decoder `ConvTranspose1d`** → `ZeroStuffConvT1d` (nearest-upsample × zero-stuff mask +
  flipped conv), avoiding the Mali-rejected `TRANSPOSE_CONV`; ConvNeXt GELU → tanh-GELU.
- **Diffusion head** sinusoidal timestep embedding computed on the host (fed as `t_freq`);
  `chunk` → slicing (no `SPLIT`).

## Device placement — the σ-VAE decoder bug

All four graphs *compile* on the Pixel 8a Mali ML Drift GPU delegate, but the σ-VAE decoder is moved
to CPU because ML Drift **miscomputes** it at runtime. On-device probing with single-output
sub-graphs shows every individual op of a ConvNeXt block — conv, RMSNorm, depthwise conv, FFN
`linear → tanh-GELU → linear` — is bit-exact on the GPU, yet the assembled block miscomputes. The
divergence first appears at the LayerScale broadcast-multiply that closes the block (the same
multiply is correct earlier in the block), reproduces on OpenCL and OpenGL and under every
buffer-storage / precision / constant-sharing option, and persists at fp32 — so it is a
graph-assembly buffer/scheduling bug, not precision, a backend, or a kernel. Splitting the decoder
does not help (a single block already trips it), so the decoder runs on CPU, where it is bit-exact
with the reference. Everything else — the two LMs and the diffusion head — runs on the GPU at fp32
precision. The LMs need LiteRT ≥ 2.1.5 (this sample's pin): 2.1.3's Mali delegate rejected their
KV-step `FULLY_CONNECTED` weights shape, a delegate bug fixed in 2.1.5, where both LMs delegate
every node and the end-to-end waveform is bit-identical to the CPU reference. See the top-level
[`../README.md`](../README.md) for the full placement table.
