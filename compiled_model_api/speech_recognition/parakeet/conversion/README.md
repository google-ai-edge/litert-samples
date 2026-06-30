# Parakeet (FastConformer-CTC) → LiteRT conversion

Produces `parakeet_ship_fp16.tflite` for the Parakeet speech-recognition sample, from NVIDIA's
[`parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) (CC-BY-4.0), with
[litert-torch](https://github.com/google-ai-edge/litert).

## Environment

NeMo and litert-torch cannot share a process (a jax/torch threading conflict), so conversion is split into
two processes, each ending with `os._exit(0)`. Use a Python 3.10 venv:

```bash
pip install "nemo_toolkit[asr]" litert-torch ai-edge-litert ai-edge-quantizer torch numpy soundfile sentencepiece
# download parakeet-tdt_ctc-110m.nemo from Hugging Face (nvidia/parakeet-tdt_ctc-110m) into this dir,
# extracted to model_weights.ckpt + model_config.yaml + the tokenizer files.
```

`_stub.py` (imported first in each script) stubs a broken scipy submodule on some platforms.

## Steps

```bash
python build_parakeet_A.py        # NeMo only: torch.save encoder+CTC modules -> parakeet_modules.pt
python build_parakeet_A2.py       # NeMo only: real-speech reference (mel/logits/ids) for parity
python extract_prep.py            # save the model's exact mel filterbank -> mel_fb.bin (an app asset)
python build_parakeet_ship.py     # litert-torch: load modules, apply GPU patches, pad to the 16 s window
                                  #   with the additive attention mask, convert -> parakeet_ship.tflite
python fp16_parakeet.py           # ai_edge_quantizer FLOAT_CASTING fp32->fp16 + op-check (banned NONE, >4D 0)
python validate_mel.py            # verify the host log-mel against NeMo's reference
```

`build_parakeet_ship.py` emits `parakeet_ship.tflite` (fp32); `fp16_parakeet.py` emits the fp16 file. Push
the fp16 model with `../kotlin_cpu_gpu/android/install_to_device.sh`.

## What the patches do

- `RelPositionMultiHeadAttention` → manual ≤4D matmuls (no SDPA, no cache); GLU → `a·sigmoid(b)` (SPLIT
  banned); BatchNorm folds; CTC `ConvASRDecoder` fused into the graph.
- Encoder length masking → GPU-clean additive attention bias + conv frame-mask, so a fixed 16 s window can
  be zero-padded without contaminating the real frames.
- **fp16-safe LayerNorm**: the subsampling front-end emits |x|≈7000, so the `var = mean(d²)·S²` rescaling
  overflows fp16 on Mali; the reduction stays in a down-scaled domain (`y = d/√(mean(d²)+ε)`, the scale
  cancels) — exact and overflow-free.
