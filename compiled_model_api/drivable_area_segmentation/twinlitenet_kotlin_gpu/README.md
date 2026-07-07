# TwinLiteNet — Drivable-area + lane segmentation (LiteRT CompiledModel GPU)

Real-time **drivable-area and lane-line segmentation** running **fully on the LiteRT
`CompiledModel` GPU** delegate. [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet)
(2023) is an ultra-light ESPNet-based network with two segmentation heads — the ADAS
"where can I drive" + "where are the lanes" building block. Only **3.1 MB**, ~44 ms/frame
on a Pixel 8a.

- **Model:** [litert-community/TwinLiteNet-LiteRT](https://huggingface.co/litert-community/TwinLiteNet-LiteRT)
- **Weights:** [chequanghuy/TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet) (BDD100K) · MIT
- **Input:** `[1, 3, 360, 640]` NCHW, RGB, `x/255`
- **Outputs:** two `[1, 2, 360, 640]` logit maps — drivable_area + lane_line (argmax over 2 classes)

## How it works

TwinLiteNet is a pure CNN, so the graph converts fully GPU-compatible (**270/270 nodes on
the delegate, 1 partition**; device corr 0.99997/0.99998 on the two heads, ~44 ms) with
one patch: the `ConvTranspose2d` upsamplers → ZeroStuffConvT2d (Mali rejects
`TRANSPOSE_CONV`). CPU-exact vs PyTorch (corr 1.0). Decode: `argmax` over the 2 classes of
each head → drivable-area mask + lane mask.

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
cd android
./install_to_device.sh <dir-with-twinlite.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample runs on a bundled dashcam frame and overlays the drivable area (green) + lane
lines (red). Adapt `MainActivity.kt` to feed live camera frames for a real-time demo.

## Convert

See [`conversion/`](conversion/) — `build_twinlite.py` uses the MIT weights and converts
with litert-torch.
