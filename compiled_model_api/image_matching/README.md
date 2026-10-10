# Image Matching with LiteRT — XFeat (on-device local features)

An Android sample that runs [XFeat](https://github.com/verlab/accelerated_features) (CVPR 2024) local feature extraction on device with the LiteRT `CompiledModel` API, **fully on the GPU**: pick two photos of the same scene (different angles) and see the matched keypoints — the building block for AR, panorama stitching, SLAM and image registration.

```
gray[1,1,480,640] (host instance-norm) →[GPU: XFeat CNN]→ feats[1,64,60,80] +
keypoints[1,65,60,80] + heatmap[1,1,60,80] →[host: cell-softmax decode + NMS +
descriptor sampling + mutual-nearest-neighbor]→ matches
```

## Model

| Model | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| XFeat (1.5 M, fp16 1.4 MB) | gray[1,1,480,640] → feats + keypoint logits + reliability | **GPU** |

Loads with `CompiledModel.create(...)` on `Accelerator.GPU`: **72 / 72 nodes on `LITERT_CL`** (full residency, 1 partition), **~0.4 ms** per image, device-vs-PyTorch corr **0.9999** ([litert-community/xfeat-litert](https://huggingface.co/litert-community/xfeat-litert)).

## Why it's GPU-clean — the re-authoring

Two numerically-equivalent rewrites (see [`conversion/`](conversion/)): the input grayscale **InstanceNorm moves host-side** (its H·W spatial reduction overflows fp16 on the delegate), and `_unfold2d(x, 8)` (space-to-depth via unfold → >4-D tensors / GATHER_ND) becomes an exact **one-hot `Conv2d(1, 64, kernel=8, stride=8)`**. Result: zero GATHER/SELECT/TOPK/CAST, all tensors ≤4-D. Keypoint decode (per-cell softmax over 64 positions + dustbin, weighted by the reliability heatmap), 5×5 NMS, bilinear descriptor sampling and mutual-nearest-neighbor matching (cosine ≥ 0.82) run in Kotlin ([`XFeatMatcher.kt`](kotlin_cpu_gpu/android/app/src/main/java/com/google/ai/edge/examples/image_matching/XFeatMatcher.kt)).

## Run it

1. Get the model: build with [`conversion/convert_xfeat.py`](conversion/convert_xfeat.py) or download `xfeat_fp16.tflite` from [litert-community/xfeat-litert](https://huggingface.co/litert-community/xfeat-litert).
2. Install the app: `cd kotlin_cpu_gpu/android && ./gradlew :app:installDebug`
3. Push the model: `./install_to_device.sh <dir-with-the-tflite>`
4. Pick two photos of the same scene → side-by-side match lines (green = confident).

Upstream: [verlab/accelerated_features](https://github.com/verlab/accelerated_features) (Apache-2.0). Please cite Potje et al., *XFeat: Accelerated Features for Lightweight Image Matching* (CVPR 2024).
