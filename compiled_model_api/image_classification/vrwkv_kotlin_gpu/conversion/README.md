# Vision-RWKV (VRWKV-S) → LiteRT conversion

Converts the OpenGVLab Apache-2.0 [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV) VRWKV-S ImageNet-1K classifier into a single graph that runs **entirely on the LiteRT `CompiledModel` GPU delegate**, then validates it against the fp32 PyTorch reference.

## Why it converts

VRWKV's token mixer is a bidirectional WKV — a CUDA kernel. The classifier processes a **fixed** 14×14 = 196-token grid, so the bidirectional WKV is exactly a per-channel decay-biased attention:

```
L[c,t,i] = k[c,i] − (spatial_decay[c]/T)·|t−i| + (spatial_first[c]/T)·δ(t,i)
y[c,t]   = Σ_i softmax_i(L[c,t,·]) · v[c,i]
```

— plain 4D `softmax` + `matmul`, no sequential scan.

## Key detail: runtime token-distance input

The `[C,T,T]` decay bias `w·dist` is `frozen_param × constant` → the converter **const-folds** it into a 59 MB flatbuffer constant per block (an unshippable 1.5 GB model that fp16 cannot shrink). The fix: feed the token-distance matrix `dist[t,i] = |t−i|` as a **runtime input** so the bias is computed live (a transient `[C,T,T]` tensor); `eye = relu(1 − dist)` derives the self-weight without a second input. The flatbuffer drops to 48 MB fp16.

Other re-authorings: VRWKV-S is **post-norm** (the norm follows the mixer; the LayerScale gamma is baked into that norm's affine params); q-shift is pad+slice+concat (≤4D).

## Files

- `build_vrwkv.py` — model reconstruction, parity (matrix Bi-WKV vs explicit oracle + real-image top-5), litert-torch conversion, fp16 cast (`ai_edge_quantizer` FLOAT_CASTING).
- `validate_vrwkv.py` — runs the fp16 tflite through the **CompiledModel API** and checks its top-1 and logits against fp32 PyTorch.

## Run

```bash
# deps: torch, numpy, pillow, litert-torch, ai-edge-quantizer, ai-edge-litert
# inputs: vrwkv_s_in1k_224.pth (OpenGVLab) + imagenet_classes.txt + dog.jpg next to the scripts
python build_vrwkv.py all      # parity -> convert -> fp16
python validate_vrwkv.py       # CompiledModel vs fp32 torch (top-1 + logits corr)
```

Expected: parity Bi-WKV corr 1.0000000 with a sensible top-5; validate top-1 identical to fp32 and logits corr ≈ 1.0. On a Pixel 8a the fp16 model runs 1371/1371 nodes on the GPU delegate (1 partition, ~28 ms), device top-1 matching desktop.
