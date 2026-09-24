# YuNet → LiteRT conversion

Produces the `yunet_fp16.tflite` graph used by the Face Detection sample, from [YuNet](https://github.com/ShiqiYu/libfacedetection) (ShiqiYu/libfacedetection, BSD-3-Clause), with [litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch numpy pillow
git clone https://github.com/ShiqiYu/libfacedetection.train    # ships weights/yunet_n.pth + the model code
```

`build_yunet.py` expects `libfacedetection.train/` next to it (it imports `yunet_train.tasks.face.build_yunet` and loads `weights/yunet_n.pth`).

## Run

```bash
python build_yunet.py all          # op-check (banned NONE / >4D 0) + fp16 + tflite-vs-torch parity
python device_gate_yunet.py        # real-image torch-vs-tflite parity + face decode (optional)
```

`build_yunet.py all` emits `yunet.tflite` (fp32) and `yunet_fp16.tflite` (0.3 MB — the smallest model in the zoo); push the fp16 file with `../kotlin_cpu_gpu/android/install_to_device.sh`.

## Recipe (zero re-authoring)

Pure CNN (depthwise-separable `ConvDPUnit`) + a TFPN neck whose upsample is `F.interpolate(mode="nearest")` → `RESIZE_NEAREST_NEIGHBOR` (no transposed conv → no ZeroStuff), + non-padded `MaxPool2d` (no `-inf` pad → no `PADV2`). The head's per-stride `permute(0,2,3,1).reshape(B,-1,C)` (+ `.sigmoid()` on cls/obj) is wrapped so the model emits 12 decode-ready outputs (cls/obj/bbox/kps × strides {8,16,32}, output order identity). **Preprocessing = BGR, 0-255, NO normalization** (`Normalize(mean=0,std=1,to_rgb=False)`).

Decode (host-side): anchor-free priors `px=col·s, py=row·s` (offset 0), score=`cls·obj`, box = center + `exp(wh)·s`, 5 landmarks `kps·s+prior`, then NMS (IoU 0.45). Result: banned ops NONE, ≤4D, tflite-vs-torch corr 1.0, device-vs-torch corr 0.9999.
