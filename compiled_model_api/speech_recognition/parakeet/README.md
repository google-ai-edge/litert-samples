# Speech Recognition with LiteRT — Parakeet (FastConformer-CTC, on-device, fully-GPU)

An Android sample that runs **NVIDIA Parakeet** ([`parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m),
CC-BY-4.0) speech-to-text end-to-end on device with the LiteRT `CompiledModel` API. The 17-layer
FastConformer encoder and the CTC head run as a **single GPU graph**; only the log-mel front-end and the
greedy-CTC + SentencePiece decode are on the host. The app transcribes a bundled clip or live microphone
audio (up to 16 s).

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| FastConformer encoder + CTC | mel[1,80,1601] + mask[1,201] → logits[1,201,1025] | **GPU** |

fp16, ~226 MB, converted with [litert-torch](https://github.com/google-ai-edge/litert). On a Pixel 8a
(Tensor G3): the on-device transcript matches PyTorch exactly, real-frame logits corr **0.99997**,
**3105 / 3105** nodes on `LITERT_CL` (full GPU residency, 1 partition), ~**330 ms GPU + ~70 ms host mel ≈
0.4 s end-to-end** per 16 s window (device-app measured; the async `run()` enqueue returns in ~40 ms and the
GPU compute completes at output readback). Get the
model from [litert-community/Parakeet-tdt-ctc-110m-LiteRT](https://huggingface.co/litert-community/Parakeet-tdt-ctc-110m-LiteRT)
or build it with [`conversion/`](conversion/).

## Re-authoring (litert-torch)

To make the FastConformer GPU-compatible:

- `RelPositionMultiHeadAttention` re-authored as manual ≤4D matmuls (`q±pos_bias_{u,v}`, `matrix_ac`,
  `rel_shift(matrix_bd)`, softmax·v) — no SDPA, no cache. GLU → `a·sigmoid(b)` via slicing (SPLIT banned).
- BatchNorm folds; CausalConv1d uses symmetric zero-padding; the CTC `ConvASRDecoder` (Conv1d 512→1025) is
  fused into the graph.
- **Fixed window + masking.** `CompiledModel` needs static shapes, so audio ≤16 s is padded to a 1601-frame
  mel window; the encoder length masking is folded into the graph as a GPU-clean **additive attention bias**
  (`scores += (1-mask)·-3e4`) plus a **conv frame-mask**, so padded frames never affect the real output.
- **fp16-safe LayerNorm.** The subsampling front-end emits very large pre-norm activations (|x|≈7000); the
  usual `var = mean(d²)·S²` rescaling overflows fp16 on the Mali delegate (S²≈8e5, var≈2.5e7 > 65504),
  collapsing the norm → blank output. The fix keeps the whole reduction in a down-scaled domain and never
  rebuilds the large variance (`y = d/√(mean(d²)+ε)`; the scale cancels) — exact, fp16-safe at any
  magnitude.

The result is GPU-clean (no banned ops, all tensors ≤4D) and fully delegated to LITERT_CL.

## Preprocessing & decode

- **Host log-mel** matches NeMo's `AudioToMelSpectrogramPreprocessor`: preemphasis 0.97, center zero-pad,
  hann(400) in a 512-pt FFT, `|·|²`, slaney mel filterbank (`assets/mel_fb.bin`, the model's own buffer),
  `log(x + 2⁻²⁴)`, per-feature normalization. See `MelSpectrogram.kt`.
- **Decode** is greedy CTC over the real frames (argmax per frame, drop blank=1024 and repeats) + manual
  SentencePiece detokenize from `assets/tokens.txt` (`▁`→space). See `ParakeetAsr.kt`.

## Run

1. Get `parakeet_ship_fp16.tflite` from
   [litert-community/Parakeet-tdt-ctc-110m-LiteRT](https://huggingface.co/litert-community/Parakeet-tdt-ctc-110m-LiteRT),
   or build it with `conversion/build_parakeet_ship.py`.
2. Build/install the app and push the model:
   ```bash
   cd kotlin_cpu_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-parakeet_ship_fp16.tflite>
   ```
3. Launch **Parakeet ASR**: tap **Record** (auto-stops at 16 s) or **Transcribe sample** (bundled clip).
   (The first launch fails with "Model not found" until the model is pushed.)

Upstream: [NVIDIA NeMo / parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m)
(CC-BY-4.0).
