# SAM 2.1 (Hiera-Tiny) → LiteRT conversion

These scripts reproduce the two GPU-clean `.tflite` models the app fetches at build time
(`download_model.gradle`). They convert [`facebook/sam2.1-hiera-tiny`](https://huggingface.co/facebook/sam2.1-hiera-tiny)
(Meta, Apache-2.0) with [`litert-torch`](https://github.com/google-ai-edge/ai-edge-torch) — **model-side
re-authoring only, no converter patch**.

| Script | Produces | Notes |
|---|---|---|
| `convert_sam2.py --v2 --convert` | `sam2_tiny_image_encoder_v2_fp16.tflite` (~80 MB) | Hiera + FPN encoder, decoder-ready (folds `conv_s0`/`conv_s1` + `no_memory`). `--convert` (no `--v2`) makes the raw-FPN encoder. |
| `convert_sam2_decoder.py --convert` | `sam2_tiny_mask_decoder_fp16.tflite` (~17 MB) | Prompt-conditioned two-way transformer + mask up-sampler. |

```bash
pip install ai-edge-torch transformers ai-edge-litert ai-edge-quantizer torch
export SAM2_OUT=/tmp/sam2_out          # where the .tflite + reference tensors land
python convert_sam2.py --v2 --convert
python convert_sam2_decoder.py --convert
```

## GPU-clean re-authoring (each weights-exact)

**Encoder**: ≤4-D window partition/attention (the 6-D window reshape and 5-D fused-qkv are GPU-rejected),
3-D batched SDPA, baked positional embeddings, dropped constant sine FPN encodings, overflow-safe
LayerNorm. **Decoder**: 3-D batched SDPA (×7), `ConvTranspose2d` → exact zero-stuff + `Conv2d` (×2,
`TRANSPOSE_CONV` is rejected on device), mask head kept ≤4-D (no 5-D tensor), baked
`image_positional_embeddings` + no-mask prompt, static multimask slice. Both: `banned ops = NONE`,
`>4-D tensors = 0`, full LITERT_CL residency.

The point→token **prompt encoder** is computed host-side (sin/cos) so the decoder graph stays
sin/cos-free; its constants are dumped to `prompt_encode_const.bin` (also bundled in the app's assets).

## ⚠ Decoder on CPU (residency ≠ correctness)

Both models fully delegate to the GPU, but on the Pixel 8a the **decoder's GPU fp16 output is wrong**
(a face tap that the CPU decoder segments at IoU ≈ 0.62 collapses to ≈ 0.10 with the mask on the
background under the GPU delegate). The encoder's GPU output is fine, so the app runs the encoder on GPU
and the (tiny) decoder on CPU. It is not LayerNorm (plain vs. overflow-safe give the same wrong GPU
result); `PLAIN_LN=1 python convert_sam2_decoder.py --convert` reproduces that check.
