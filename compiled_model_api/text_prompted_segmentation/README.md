# Text-Prompted Segmentation with LiteRT — CLIPSeg (on-device)

An Android sample that runs [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) (CVPR 2022) open-vocabulary segmentation on device: type what you want to segment ("a cat", "the sky", "a red car") and get a mask — no fixed class list. The CLIP text and vision encoders run on the LiteRT `CompiledModel` **GPU**; the tiny decoder runs on **CPU**.

```
prompt →[host BPE + emb lookup]→ [GPU text enc]→ EOT row @ text_proj → cond[512]
image  →[CLIP norm]→ [GPU vision enc]→ t3,t6,t9 →[CPU decoder + FiLM(cond)]→ logits → sigmoid mask
```

## Models

| Stage | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| CLIP text encoder | token-emb [1,77,512] → hidden [1,77,512] | **GPU** 761/761, ~8.7 ms |
| CLIP vision encoder | image [1,3,352,352] → t3,t6,t9 [1,485,768] | **GPU** 613/613, ~8.2 ms |
| decoder | t3,t6,t9,cond[512] → logits [1,352,352] | **CPU** (exact) |

End-to-end device-vs-PyTorch logits corr **0.99998**, mask IoU **0.9986** ([litert-community/CLIPSeg-rd64-LiteRT](https://huggingface.co/litert-community/CLIPSeg-rd64-LiteRT)).

## Why the decoder is on CPU

Both CLIP encoders are GPU-clean after re-authoring (qkv-3D-BMM attention, quick-GELU, baked interpolated pos-embed, `safe_ln_up`). The vision encoder's **484-token global attention survives fp16 on the Mali delegate (corr 0.998)** — a useful data point between CLIP-B/32 (49 tokens) and Depth-Anything-V2 (784 tokens). But an on-device bisection pinned a miscompute to the decoder's **first attention layer** (post-FiLM corr 0.999 → decoder-layer-0 0.864): the decoder uses **4 heads × head_dim 16**, and the small head dim is the fp16-fragile axis (the vision encoder's head_dim 64 survives). The decoder is only 3 layers of 64-dim, so running it on CPU is exact and fast — the same fp16-fragile-part-on-CPU split as the speaker-diarization sample. See [`conversion/`](conversion/).

## Run it

1. Build with [`conversion/build_clipseg.py`](conversion/build_clipseg.py) or download from [litert-community/CLIPSeg-rd64-LiteRT](https://huggingface.co/litert-community/CLIPSeg-rd64-LiteRT).
2. `cd kotlin_cpu_gpu/android && ./gradlew :app:installDebug`
3. `./install_to_device.sh <dir-with-the-artifacts>`
4. Pick an image, type a prompt, tap **Segment** → red mask overlay.

Upstream: [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) (Apache-2.0). Please cite Lüddecke & Ecker, CVPR 2022.
