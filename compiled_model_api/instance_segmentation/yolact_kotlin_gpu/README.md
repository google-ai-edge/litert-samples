# YOLACT-ResNet50 — Real-time instance segmentation (LiteRT CompiledModel GPU)

Per-object **instance segmentation** (COCO 80 classes) running **fully on the LiteRT `CompiledModel` GPU** delegate. [YOLACT](https://arxiv.org/abs/1904.02689) (ICCV 2019): the network (ResNet50 + FPN + protonet + heads) runs on the GPU; the lightweight decode (NMS + linear-combination masks) runs host-side. ~41 ms/graph on a Pixel 8a.

- **Model:** [litert-community/YOLACT-ResNet50-LiteRT](https://huggingface.co/litert-community/YOLACT-ResNet50-LiteRT) (`yolact.tflite` + `priors.bin`)
- **Weights:** [dbolya/yolact](https://github.com/dbolya/yolact) · MIT
- **Input:** `[1, 3, 550, 550]` NCHW, BGR, `(x - [103.94,116.78,123.68]) / [57.38,57.12,58.40]` (no /255)
- **Raw outputs:** `loc [1,19248,4]`, `conf [1,19248,81]`, `mask [1,19248,32]`, `proto [1,138,138,32]`

## How it works

Base YOLACT is a pure CNN, so the graph converts fully GPU-compatible (**138/138 nodes on the delegate, 1 partition**; device corr 0.99999–1.0 on all four raw outputs) with one patch: the ResNet50 stem `MaxPool2d(padding=1)` `-inf` PADV2 → 0-pad + unpadded maxpool (exact post-ReLU); the scripted FPN is made traceable by disabling YOLACT's JIT.

**Host-side decode** (`YolactSegmenter.kt`): SSD box-decode vs the baked `priors.bin` (19248 anchors, variances [0.1,0.2]) → per-class NMS (IoU 0.5, score 0.3) → lincomb masks `sigmoid(proto @ coeff)` cropped to each box.

## Run

```bash
# 1. Get the model + priors (build with ../conversion or download from Hugging Face)
cd android
./install_to_device.sh <dir-with-yolact.tflite-and-priors.bin>

# 2. Build & run
./gradlew :app:installDebug
```

The sample segments a bundled photo and draws colored instance masks + boxes + COCO labels. Adapt `MainActivity.kt` to feed live camera frames for a real-time demo.

## Convert

See [`conversion/`](conversion/) — `build_yolact.py` loads the MIT ResNet50 weights and converts with litert-torch.
