# CLIPSeg → LiteRT conversion

`build_clipseg.py` converts [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) (Apache-2.0) into three graphs with parity checks: CLIP **vision** and **text** encoders (GPU, fp16) and the **decoder** (CPU, fp32 — its small-head-dim attention fp16-miscomputes on the Mali GPU delegate). `probe_vision.py` holds the re-authored ViT encoder (qkv-3D-BMM attention, quick-GELU, baked interpolated pos-embed, `safe_ln`/`safe_ln_up`). Also exports the host assets (token-embedding table, text projection, tokenizer vocab/merges).

```bash
pip install transformers pillow
python build_clipseg.py    # -> clipseg_{vision,text}_fp16.tflite + clipseg_decoder.tflite + assets
```
