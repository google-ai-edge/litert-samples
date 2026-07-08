# HSEmotion (EfficientNet-B0) → LiteRT conversion

Converts the Apache-2.0 [HSEmotion](https://github.com/av-savchenko/face-emotion-recognition) EfficientNet-B0 AffectNet emotion classifier into a graph that runs **entirely on the LiteRT `CompiledModel` GPU delegate**, then validates it against the fp32 PyTorch reference.

## Two hurdles

1. **Old-timm pickle.** The released `enet_b0_8_best_afew.pt` is a pickled model built with an old timm; its forward is broken under current timm (`DepthwiseSeparableConv` has no `conv_s2d`). Fix: lift the state dict into a fresh timm `tf_efficientnet_b0` (num_classes=8, remapping `classifier.0.*` → `classifier.*`). 358/360 tensors match by name and shape; the rest is the remapped classifier.

2. ⭐ **fp16 SqueezeExcite mean → NaN.** The SE block's global mean `x.mean((2,3))` over the 112×112 stem feature map is a single fp16 reduction whose partial sum overflows 65504, so the GPU delegate emits an **all-NaN** output (it reduces in fp16 even for an fp32 graph; the desktop fp16 CPU path is exact). Fix: a **hierarchical mean** — repeated `avg_pool2d` over equal-size tiling windows (≤ 49 elements each), which is mathematically identical but keeps every accumulation small.

## Files

- `build_hsemotion.py` — timm rebuild, safe-SE-mean patch, parity (a happy face → Happiness), litert-torch conversion, fp16 cast (`ai_edge_quantizer` FLOAT_CASTING).
- `validate_hsemotion.py` — runs the fp16 tflite through the **CompiledModel API** and checks its top-1 and logits against fp32 PyTorch.

## Run

```bash
# deps: torch, timm, numpy, pillow, litert-torch, ai-edge-quantizer, ai-edge-litert
# inputs: enet_b0_8_best_afew.pt (github release) + happy.jpg next to the scripts
python build_hsemotion.py all      # parity -> convert -> fp16
python validate_hsemotion.py       # CompiledModel vs fp32 torch (top-1 + logits corr)
```

Expected: parity shows a sensible emotion for the test face; validate top-1 identical to fp32 and logits corr ≈ 1.0. On a Pixel 8a the fp16 model runs 342/342 nodes on the GPU delegate (1 partition, ~2 ms), device top-1 matching desktop.
