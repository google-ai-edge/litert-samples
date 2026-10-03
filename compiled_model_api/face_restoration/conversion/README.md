# Converting GFPGAN v1.4 to LiteRT (GPU)

`build_gfpgan.py` converts [GFPGAN v1.4](https://github.com/TencentARC/GFPGAN) (Apache-2.0) to a GPU-clean `.tflite` with **litert-torch** (NCHW preserved), verifies parity, and exports fp16.

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer diffusers torch pillow
python build_gfpgan.py          # -> gfpgan_fp16.tflite  (431 MB)
```

The face detector `yunet_fp16.tflite` (used for alignment) comes from the YuNet / [libfacedetection](https://github.com/ShiqiYu/libfacedetection) conversion.

## The one real re-authoring: StyleGAN2 `ModulatedConv2d`

Its original form is doubly GPU-incompatible — it builds a **5D** weight `(b,c_out,c_in,k,k)` at runtime from the style vector and convolves with that runtime filter (a GPU `CONV_2D` needs a **constant** filter). It is rewritten to a mathematically exact 4D form (verified corr `1.000000`):

- **modulation** — `conv(x, W·style) == conv(x · style_per_in_channel, W_const)` (conv is linear), so the style becomes an input channel-scale and the filter stays constant.
- **demodulation** — `rsqrt(Σ (W·style)² + eps) == rsqrt((style²) @ Wsqᵀ + eps)` where `Wsq[o,i] = Σ_k W[o,i,k]²` is a constant `(c_out × c_in)` matrix — a small matmul + `RSQRT`.

Additionally the style vectors reach `|s|~1000`, so the demod sum `Σ style²·Wsq` (≈2.3e6) overflows fp16 on Mali → `rsqrt(inf)=0` → the decoder collapses to a flat color. Normalizing the style by its per-image max before squaring keeps every fp16 intermediate in range; the scale cancels exactly against the demod, so the device output matches the desktop fp32 result.

Everything else in the clean arch is already GPU-friendly: upsampling is `RESIZE_BILINEAR` (`align_corners=False`, no `TRANSPOSE_CONV`), no `GroupNorm`, noise fixed to stored buffers (`randomize_noise=False`) for a deterministic graph. `basicsr` is stubbed and the clean arch loaded directly (bypassing the package `__init__`, which pulls in custom-CUDA-op archs).

Device (Pixel 8a, Tensor G3): `551/551` nodes on `LITERT_CL` (single partition, no CPU fallback), ~1.2 s/face, fp16 output identical to desktop fp32.
