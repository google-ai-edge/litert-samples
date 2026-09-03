# Metric3D v2 → LiteRT conversion

Scripts that produce the `metric3d_fp16.tflite` graph used by the Android sample, from the [YvanYin/Metric3D](https://github.com/YvanYin/Metric3D) `torch.hub` model (`metric3d_vit_small`), with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer
pip install torch numpy pillow mmengine timm
# metric3d_vit_small is auto-downloaded via torch.hub. load_m3d.py stubs `mmcv` (the hub model
# imports it but only needs mmengine) and guards an inspect.getsourcefile crash.
```

## Run

```bash
python build_m3d.py all          # re-author + op-check (banned NONE / >4D 0) + fp16 + tflite-vs-torch parity
python device_gate.py            # real-image torch-vs-tflite parity fixtures (optional)
```

`build_m3d.py all` emits `m3d.tflite` (fp32) and `m3d_fp16.tflite`; rename the fp16 file to `metric3d_fp16.tflite` and push it with `../kotlin_cpu_gpu/android/install_to_device.sh`.

## Files

| File | What |
|---|---|
| `build_m3d.py` | the full re-authoring recipe + op-check + fp16 + parity. Env toggles (`M3D_GELU`, `M3D_UPSAMPLE`, `M3D_AC`, `M3D_ATTN`) default to the shipped recipe. |
| `load_m3d.py` | mmcv-stub loader for the `torch.hub` Metric3D model. |
| `device_gate.py` | builds real-image fixtures (ImageNet-normalized) for on-device parity checks. |

## Recipe (all numerically-equivalent unless noted)

**Encoder — DINOv2 ViT-S/14 + register tokens:** fused-QKV 5D reshape → 3 q/k/v Linear + manual 4D attention; LayerScale γ baked into `attn.proj` / `mlp` last Linear; bicubic pos-embed baked to a constant.

**Decoder — RAFTDepthNormalDPT5:**
1. **Convex upsample (6/7-D) → depth-to-space `ZeroStuffConvT2d`** (16 softmax-over-9 subpixel combines → `cat [N,96,H,W]` → fixed `ConvTranspose2d(96→6,k4,s4)`). On Mali, masking at non-stride positions via `RESIZE_NEAREST` is wrong (corr 0.57) — `ZeroStuffConvT2d` masks only stride-aligned positions.
2. **`Token2Feature` `ConvTranspose2d` → `ZeroStuffConvT2d`** (Mali rejects `TRANSPOSE_CONV`).
3. **GELU → tanh approximation** (POW-free) — `x·sigmoid(1.702x)` is not accurate enough for the 0.1–200 m log-depth head (corr 0.51 → 0.96 with tanh).
4. **`norm_normalize` `F.elu` → SELECT-free** `exp(−relu(−k))+relu(k)+min_κ` (exact).
5. The DPT `ConvBlock`'s `nn.ReLU(inplace=True)` mutates the residual (`relu(x)+convs`) — replicated.

Result: banned ops NONE, all tensors ≤4D, fp16 78 MB, tflite-vs-torch corr 1.0.
