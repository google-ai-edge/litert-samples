# FLUX.2-klein conversion

Reproduces the twelve int8 LiteRT graphs and the staged host inputs the Android app
reads. Everything is written next to the scripts; nothing is committed.

```bash
pip install torch diffusers transformers litert-torch ai-edge-litert pillow numpy

python build_klein_enc.py --export        # ke_enc0/1/2   (Qwen3-4B, 3 x 912 MB)
python chunked_export_klein.py --export   # kc_prep, kc_double0/1, kc_single0-3, kc_final
python vae_deploy_klein.py --export       # kv_vae
python gen_prep_klein.py                  # ref_fp32.png + klein_bins/*.bin
python gen_verify_klein.py                # the device loop, on the host, tflite-only
```

Conversion needs roughly 40 GB of RAM: the fp32 `diffusers` pipeline plus one ~912 MB
export at a time.

| Script | What it does |
|---|---|
| `build_klein.py` | The single-stream block, re-authored GPU-clean. Bakes the RoPE even/odd de-interleave into the fused `to_qkv_mlp_proj` rows; safe RMSNorm / LayerNorm. |
| `build_klein_double.py` | The double-stream block. Same RoPE bake across `to_q`, `to_k`, `add_q_proj`, `add_k_proj` and the four qk-norm weights; joint RoPE over `concat(text, image)`. |
| `build_klein_dit.py` | Assembles the transformer from the two block types and verifies it against the stock `Flux2Transformer2DModel`. |
| `build_klein_real.py` | Loads the real weights and pins the fixed geometry: 256 px, 16x16 image tokens, 512 text tokens, `latent_ids`, RoPE axes. |
| `chunked_export_klein.py` | Splits the transformer into `kc_prep`, `kc_double0/1`, `kc_single0-3`, `kc_final` and exports each as an INTEGER-int8 graph. Checks the sequential composition against the monolithic model first. |
| `build_klein_enc.py` | The Qwen3-4B text encoder as three 9-layer chunks, tapped at layers 9 / 18 / 27. Holds the two device-only fixes: the concat-based `repeat_kv` and the head-expanded attention mask. |
| `vae_deploy_klein.py` | The VAE decoder, with `GroupNorm` replaced by a manual N-dimensional equivalent (the mid-block attention normalizes a 3D tensor). |
| `gen_prep_klein.py` | Runs the stock fp32 pipeline once: writes `ref_fp32.png` and every `.bin` the device needs — embeddings, mask, rotary tables, per-step timestep embedding, `dsigma`, initial latents, batch-norm statistics, and the two gather maps. |
| `gen_verify_klein.py` | The device loop, on the host, driven only by the twelve `.tflite` graphs. Reports PSNR against `ref_fp32.png`. |
| `generate_klein.py` | Quality gate: the stock pipeline with the fp32 transformer swapped for the int8 chunks, which isolates transformer quantization error from text-encoder quantization error. |
| `probe_enc_layer.py` | One encoder layer as an fp32 graph with taps at `q_rope`, `k_rope`, `key_rep`, the raw logits, the softmax probabilities, the attention output and the layer output. Reach for this when the device is wrong but nothing errors. |

## Why these rewrites

All are exact — no accuracy-changing approximations.

- **RoPE without `GATHER_ND`.** FLUX interleaves even and odd channels. Baking that
  permutation into the rows of `to_q` and `to_k` turns the rotation into a contiguous
  half-split. It is exact because `q · k` is invariant to a channel permutation applied
  to both.
- **`repeat_kv` as a `CONCATENATION`.** The stock grouped-query expansion builds a rank-5
  tensor, and the obvious 4D rewrite uses `expand`, which lowers to `BROADCAST_TO` — an
  operation the GPU delegate rejects outright.
- **The attention mask is pre-expanded to `[1, num_heads, S, S]`.** A broadcast `ADD` on a
  `BATCH_MATMUL` result is silently miscomputed by the GPU delegate.
- **`amax` on a 4D tensor** trips the converter's NHWC layout pass, so the norms reshape
  4D to 3D around it.
- **Safe RMSNorm / LayerNorm** (max-normalized) avoid fp16 overflow, and the runtime is
  asked for `GpuOptions(precision = FP32)` because the modulated blocks return NaN in
  fp16 regardless.
