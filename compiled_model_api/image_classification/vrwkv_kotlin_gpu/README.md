# Image Classification with LiteRT — Vision-RWKV (RWKV vision backbone on GPU)

An Android sample that classifies a photo with an **RWKV-style vision backbone running fully on the LiteRT `CompiledModel` GPU delegate**. [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV) (OpenGVLab, ICLR 2025, Apache-2.0) replaces softmax self-attention with a **bidirectional WKV** linear-attention scan — the vision counterpart of the RWKV language model. This runs the VRWKV-S ImageNet-1K classifier (80.1% top-1).

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| VRWKV-S | `image[1,3,224,224]` + `dist[1,1,196,196]` → `logits[1,1000]` | GPU (1371/1371 nodes, 1 partition) |

fp16 weights, 48 MB, ~28 ms/inference. Converted with [litert-torch](https://github.com/google-ai-edge/litert) — see [`conversion/`](conversion/).

## The Bi-WKV re-authoring

VRWKV's token mixer is a CUDA `bi_wkv` kernel. Because the token count is fixed (14×14 = 196), the bidirectional WKV is exactly a **per-channel decay-biased attention** over the token sequence:

```
L[c,t,i] = k[c,i] − (spatial_decay[c]/T)·|t−i| + (spatial_first[c]/T)·δ(t,i)
y[c,t]   = Σ_i softmax_i(L[c,t,·]) · v[c,i]
```

i.e. C independent `[T,T]` attention matrices → plain 4D `softmax` + `matmul`, **no sequential scan** (the matrix form is oracle-exact vs the explicit bidirectional sum, corr 1.0). Two details keep it GPU-clean and small:

- The `[C,T,T]` decay bias `w·dist` (frozen `w` × constant `dist`) gets **const-folded** into a 59 MB-per-block flatbuffer constant — an unshippable 1.5 GB model that fp16 cannot shrink. Feeding the token-distance matrix `dist[t,i] = |t−i|` as a **runtime input** (`eye = relu(1 − dist)`) keeps the bias a transient live tensor → **48 MB**.
- VRWKV-S is **post-norm** (norm after the mixer); the LayerScale gamma is baked into the following norm's affine params; q-shift is pad+slice+concat (≤4D).

## Pipeline

```
photo --host: resize-256 → center-crop-224 → ImageNet-norm (NCHW)--> image[1,3,224,224]
      --host: dist[t,i] = |t-i| (constant)-->                        dist[1,1,196,196]
                                        VRWKV-S (GPU) -> logits[1000] -> softmax -> top-5
```

`VrwkvClassifier.kt` does the preprocessing, builds the constant token-distance input, runs the CompiledModel, and returns the top-5 ImageNet labels.

## Verification

- Sequential Bi-WKV re-authoring is oracle-exact (matrix vs explicit bidirectional sum, corr 1.0000000).
- fp16 tflite vs fp32 PyTorch through the CompiledModel API: top-1 identical, logits corr 1.00000 (`conversion/validate_vrwkv.py`).
- On device (Pixel 8a): **1371/1371 nodes on the GPU delegate, 1 partition**, ~28 ms; device fp16 top-1 matches desktop fp32 on the bundled sample (`Samoyed` 79%, top-5 identical; logits corr 0.9989).

## Build & run

```bash
cd android
./gradlew :app:installDebug
```

The 48 MB `vrwkv_s_fp16.tflite` is downloaded from Hugging Face ([`litert-community/Vision-RWKV-S-LiteRT`](https://huggingface.co/litert-community/Vision-RWKV-S-LiteRT)) into the app assets at build time (`app/download_model.gradle`); the labels and a sample image are bundled. Pick a photo (or use the bundled sample) to get the top-5 ImageNet predictions.

## License

Model: Apache-2.0 (Vision-RWKV / OpenGVLab).
