# XFeat → GPU-native LiteRT (conversion)

Re-authors **XFeat** (Accelerated Features, Apache-2.0, ~1.5M pure CNN) into a GPU-native LiteRT
`.tflite` with [`litert-torch`](https://github.com/google-ai-edge/ai-edge-torch). Pre-built model:
[`litert-community/xfeat-litert`](https://huggingface.co/litert-community/xfeat-litert) (downloaded by the app).

## Re-authoring (pure CNN, both GPU gates)
- Input gray + **InstanceNorm moved host-side** (graph takes normalized grayscale `[1,480,640,1]`);
  the InstanceNorm spatial reduction over H·W would overflow fp16 on the delegate.
- **`_unfold2d(x, 8)`** (space-to-depth via `unfold` → >4-D / GATHER_ND) → one-hot
  `Conv2d(1,64,k=8,s=8)` (exact, single CONV_2D). Result: zero GATHER/SELECT/TopK/Cast, no >4-D.

Pixel 8a: full `LITERT_CL` residency (72/72), ~0.4 ms, GPU vs CPU corr 0.9999.

## I/O
Input `[1,480,640,1]` NHWC grayscale (host InstanceNorm). Outputs `feats[1,64,60,80]`,
`keypoints[1,65,60,80]`, `heatmap[1,1,60,80]`. Keypoint decode + descriptor sampling + mutual-NN
matching run in the app (`XFeatHelper.kt`).

## Run
```bash
pip install torch litert-torch ai-edge-litert ai-edge-quantizer numpy
python convert_xfeat.py --nhwc
```
