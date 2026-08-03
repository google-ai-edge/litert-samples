# Conversion recipe — Bonsai Image 4B → LiteRT

Reproduces the three published graphs from the upstream checkpoint (`prism-ml/bonsai-image-ternary-4B-unpacked`, downloaded automatically; set `BONSAI_SNAPSHOT` to use a local copy).

```bash
pip install torch "diffusers>=0.39" transformers litert-torch ai-edge-quantizer huggingface_hub

python export_dit.py                      # -> dit_fp32.tflite (14.4 GiB)
python quantize_dit.py                    # -> dit_int4b32.tflite (2.11 GiB); BLOCK=128 for b128
python fix_zero_block_scales.py dit_int4b32.tflite dit_int4b32.tflite   # zero-scale patch (required)

python export_textenc.py                  # -> textenc_fp32.tflite (11.6 GiB, top 9 layers pruned)
python quant_textenc.py                   # -> textenc_int4.tflite (1.68 GiB)
python quantize_weight_only.py textenc_fp32.tflite textenc_int8_weightonly.tflite 8 0   # optional variant

python export_vae.py                      # -> vae_dec_fp32.tflite (0.19 GiB)
```

## GPU-shaped DiT (for the macOS sample app)

The export above is CPU-shaped: `apply_rotary_emb` carries rank-5 reshapes and `Flux2PosEmbed` a rank>4 slice, which the GPU delegate does not run. `export_dit_gpu.py` re-exports the same weights in a GPU-clean form — rope tables precomputed once with the pipeline's constant position ids, and the interleaved rotation rewritten as a rank-4 matmul against a constant pair-swap matrix — verified in-process against the unpatched forward before export:

```bash
python export_dit_gpu.py                  # -> dit_gpu_fp32.tflite
python quantize_dit.py dit_gpu_fp32.tflite   # -> dit_gpu_int4b32.tflite
python fix_zero_block_scales.py dit_gpu_int4b32.tflite dit_gpu_int4b32.tflite
```

The result is the DiT the [macOS sample app](../../../../samples/litert/image_generation/macos) runs on the Metal accelerator.

## The three walls this recipe handles

1. **Flux2 RoPE builds its frequency table in float64**, which leaves a `tfl.pow` on f64 in the graph and kills conversion at legalization. `export_dit.py` forces the rope frequency dtype to float32 and verifies the patch numerically with real position ids (cost: max rel 2.9e-6). A verification with all-zero ids is vacuous — at position 0, cos=1/sin=0 for any dtype.
2. **Ternary sparsity produces all-zero quantization blocks**, whose min-max scale is 0, and XNNPACK refuses to prepare the model. `fix_zero_block_scales.py` patches those scales to the tensor's smallest nonzero scale — dequantization is unchanged because the block's values are all zero.
3. **The text encoder is a prompt embedder, not an LM.** The pipeline reads hidden states from layers (9, 18, 27) only; exporting that interface lets the converter prune the top 9 of 36 layers and the LM head automatically (4.02 B → 3.11 B).

The DiT's ternary weights ({-scale, 0, +scale} per 128-group) land in int4 block-32 as exactly {-7, 0, +7} — zero rounding decisions; the container is lossless for this model's weights.
