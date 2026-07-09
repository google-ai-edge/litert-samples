# SAM 2.1 (Hiera-Tiny) → LiteRT conversion

These scripts reproduce the two GPU-clean `.tflite` models the app fetches at build time (`download_model.gradle`). They convert [`facebook/sam2.1-hiera-tiny`](https://huggingface.co/facebook/sam2.1-hiera-tiny) (Meta, Apache-2.0) with [`litert-torch`](https://github.com/google-ai-edge/ai-edge-torch) — **model-side re-authoring only, no converter patch**.

| Script | Produces | Notes |
|---|---|---|
| `convert_sam2.py --v2 --convert` | `sam2_tiny_image_encoder_v2_fp16.tflite` (~80 MB) | Hiera + FPN encoder, decoder-ready (folds `conv_s0`/`conv_s1` + `no_memory`). `--convert` (no `--v2`) makes the raw-FPN encoder. |
| `convert_sam2_decoder.py --convert` | `sam2_tiny_mask_decoder_v2_fp16.tflite` (~17 MB) | Prompt-conditioned two-way transformer + mask up-sampler. |

```bash
pip install ai-edge-torch transformers ai-edge-litert ai-edge-quantizer torch
export SAM2_OUT=/tmp/sam2_out          # where the .tflite + reference tensors land
python convert_sam2.py --v2 --convert
python convert_sam2_decoder.py --convert
```

## GPU-clean re-authoring (each weights-exact)

**Encoder**: ≤4-D window partition/attention (the 6-D window reshape and 5-D fused-qkv are GPU-rejected), 3-D batched SDPA, baked positional embeddings, dropped constant sine FPN encodings, overflow-safe LayerNorm. **Decoder**: rank-4 batched SDPA (×7, the batch dim is kept — see below), `ConvTranspose2d` → exact zero-stuff + `Conv2d` (×2, `TRANSPOSE_CONV` is rejected on device), mask head kept ≤4-D (no 5-D tensor), baked `image_positional_embeddings` + no-mask prompt, static multimask slice. Both: `banned ops = NONE`, `>4-D tensors = 0`, full LITERT_CL residency, and both run on the GPU.

The point→token **prompt encoder** is computed host-side (sin/cos) so the decoder graph stays sin/cos-free; its constants are dumped to `prompt_encode_const.bin` (also bundled in the app's assets).

## ⚠ Residency ≠ correctness: the rank-3 attention miscompute

The first decoder build wrote its attention with the **batch dim collapsed** — `q/k/v` shaped `[heads, N, d]` (rank 3). That graph compiled, delegated fully (358/358 LITERT_CL nodes, `banned ops = NONE`, `>4-D = 0`) and matched the PyTorch reference exactly on the host — yet on the Pixel 8a GPU it returned **silently wrong masks**: correlation **0.265** against the CPU output, and a face tap that the CPU decoder segments at IoU ≈ 0.62 collapsed to ≈ 0.10 with the mask on the background.

Two natural explanations were ruled out by device A/B:

- **Not fp16 precision.** Forcing fp32 GPU compute still gives correlation **0.473**.
- **Not LayerNorm.** Plain and overflow-safe LN produce the same wrong GPU result (`PLAIN_LN=1 python convert_sam2_decoder.py --convert` reproduces that check).

Bisecting the graph on device isolated the cause to the **attention rank**: making only the attention rank-4 fixes it, while the mask head's rank-2 matmul is innocent. Keeping the leading batch dim (`[1, heads, N, d]`) leaves the host numerics identical (eager cos 0.999999) and on the Pixel 8a GPU restores correlation **0.9998** / binary-IoU **0.999** against CPU, while running **~20 % faster** (6.8 ms vs 8.5 ms). Inputs and outputs are unchanged, so the fixed model is a drop-in replacement (`sam2_tiny_mask_decoder_v2_fp16.tflite`).

Note that the **encoder's** rank-3 SDPA *is* GPU-correct on the same device — a healthy sibling graph proves nothing. Op gates, full delegation and desktop parity all pass while the output is garbage: only a numeric GPU-vs-CPU comparison on device catches this class of bug.
