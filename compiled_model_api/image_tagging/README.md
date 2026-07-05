# Image Tagging (RAM++) — LiteRT CompiledModel, GPU/CPU hybrid

[RAM++ (Recognize Anything Plus)](https://github.com/xinyu1205/recognize-anything) (Apache-2.0)
**open-vocabulary multi-label tagging**: a photo in, the tags it recognizes out — from a 4,585-tag
vocabulary, per-tag sigmoid, no fixed class head. Four graphs on the LiteRT `CompiledModel` API — the
Swin-L encoder **stages 0-2** and the Query2Label **tag head** run on the **GPU**; the **last Swin
stage** and the 479 MB frozen **tag bank** run on **CPU**.

## On-device (Pixel 8a, Tensor G3 — verified)

| graph | in → out | delegate |
|---|---|---|
| Swin stages 0-2 | image [1,3,384,384] → feat [1,144,1536] | **GPU** (corr 0.998) |
| Swin stage 3 + norm + proj | feat → image_embeds [1,145,512] | **CPU** (exact) |
| reweight (multi-grained) | cls [1,512] → tag queries [1,4585,768] | **CPU** |
| Query2Label tag head | queries + image_embeds → logits [1,4585] | **GPU** (corr 0.9987, ~270 ms) |

Sample photo → 14 tags in ~2 s, all correct.

```
image →[ImageNet norm]→ [GPU Swin 0-2]→ feat →[CPU Swin-3 + norm + proj]→ image_embeds[1,145,512]
       token0 = cls →[CPU reweight over the 4585×51 tag bank]→ queries[1,4585,768]
       (queries, image_embeds) →[GPU Q2L tag head]→ logits →[sigmoid + per-class threshold]→ tags
```

## Why the split — a Mali fp16 finding

The Swin-L encoder is fully GPU-convertible, but its **last stage miscomputes in fp16 on the Mali
delegate**. Per-stage bisect on-device: stage 0 = 0.9999, stage 1 = 0.9999, stage 2 = 0.9983,
**stage 3 = 0.709**. It is **not** head_dim (stage 2 shares head_dim 32) and **not** overflow (every
stage-3 value < 848 ≪ fp16 max 65504; a round-to-fp16-between-ops simulation reproduces the fp32
result at corr 0.99999997) — it is Mali's **fp16 matmul accumulation** in the deep, high-magnitude
(absmax 847) blocks. Those 2 blocks run on CPU; everything else stays on GPU.

## Build & run

```bash
# 1. get the graphs — from Hugging Face (litert-community/RAM-Plus-LiteRT) or ./conversion/
# 2. build + install the app
cd kotlin_cpu_gpu/android
./gradlew :app:installDebug
# 3. push the 4 tflites into the app's filesDir
./install_to_device.sh <dir-with-the-tflites>
```

Launch the app — it tags the bundled sample on start; tap **Pick image** for your own photos.

Model: [`litert-community/RAM-Plus-LiteRT`](https://huggingface.co/litert-community/RAM-Plus-LiteRT)
(~769 MB, 4 graphs — staged into the app's `filesDir`, never bundled in the APK). Conversion scripts
in [`conversion/`](conversion/). Upstream:
[xinyu1205/recognize-anything](https://github.com/xinyu1205/recognize-anything) (Apache-2.0).
