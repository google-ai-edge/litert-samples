# Matcha-TTS → LiteRT conversion

Scripts that produce the four `.tflite` graphs used by the Android sample, from the
official Matcha-TTS checkpoints, with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install --no-deps matcha-tts diffusers einops conformer deep-phonemizer
pip install litert-torch ai-edge-litert ai-edge-quantizer
# checkpoints (auto-downloaded by the matcha-tts CLI, or):
#   matcha_ljspeech.ckpt + generator_v1   (github.com/shivammehta25/Matcha-TTS-checkpoints v1.0)
#   openphonemizer_best_model.pt           (hf://openphonemizer/ckpt/best_model.pt)
```

## Run

```bash
python convert_final.py 512        # text encoder + CFM decoder + HiFi-GAN vocoder (fp16)
python convert_g2p_matcha.py       # DeepPhonemizer G2P (fp16)
```

Outputs `artifacts/`: `matcha_textenc_fp16.tflite`, `matcha_decoder_fp16.tflite`,
`matcha_vocoder_fp16.tflite`, `dp_g2p_matcha_fp16.tflite`, plus the host tables
(`emb.bin`, `g2p_dict.txt.gz`, `config.json`, `g2p_meta.json`).

## Files

| File | What |
|---|---|
| `build_matcha.py` | the re-authoring recipe (GroupNorm→4D, Mish→SELECT-free softplus, ConvTranspose1d→ZeroStuffConvT1d, diffusers Attention→manual additive-masked, SinusoidalPosEmb host-side) + real-weight conversion + per-graph parity (corr 1.0). |
| `convert_final.py` | converts + fp16-quantizes the three acoustic graphs; end-to-end waveform parity. |
| `convert_g2p_matcha.py` | converts the DeepPhonemizer (espeak-IPA) G2P to a fixed `[1,96]` graph. |
| `e2e_masked.py`, `e2e_matcha.py` | end-to-end host-orchestration parity (pad-to-max + runtime mask). |
| `kotlin_replica.py` | replicates the exact Android host logic in Python (validates the Kotlin port). |
| `probe_tx_standalone.py`, `probe_decoder_taps.py` | the on-device bisection that isolated the Mali ML Drift transformer-fusion bug (decoder → CPU). |

## Re-authoring → GPU-clean

Every graph converts GPU-clean (per-graph tflite-vs-torch corr **1.000000**; end-to-end
waveform corr ≥0.99). Fixed shapes (256 phonemes, 512 mel frames) with a runtime float mask
let one compiled graph handle any length. See `build_matcha.py` for the op-by-op recipe.
