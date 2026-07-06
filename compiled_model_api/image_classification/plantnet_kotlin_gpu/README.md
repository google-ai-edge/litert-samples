# PlantNet-300K — Fine-grained plant identification (LiteRT CompiledModel GPU)

Identify **1081 plant species** from a photo, running **fully on the LiteRT `CompiledModel` GPU** delegate. A [PlantNet-300K](https://github.com/plantnet/PlantNet-300K) (NeurIPS 2021) ResNet18. ~16 ms/frame on a Pixel 8a.

- **Model:** [litert-community/PlantNet-300K-ResNet18-LiteRT](https://huggingface.co/litert-community/PlantNet-300K-ResNet18-LiteRT) · 47 MB
- **Weights:** [cpoisson/plantnet300k-resnet18](https://huggingface.co/cpoisson/plantnet300k-resnet18) · Apache-2.0
- **Input:** `[1, 3, 224, 224]` NCHW, RGB, ImageNet-normalized
- **Output:** `[1, 1081]` species logits (Latin names)

## How it works

Plain torchvision ResNet18 — a pure CNN. It converts to a fully GPU-compatible graph (**37/37 nodes on the delegate, 1 partition**; device corr 0.99999, top-1 match) with **one patch**: the ResNet stem `MaxPool2d(padding=1)` lowers to a PADV2 with `-inf` padding (`PADV2: src has wrong size` on the Mali delegate), replaced by an explicit 0-pad + unpadded maxpool (exact post-ReLU). CPU-exact vs PyTorch (corr 0.99999999999).

Preprocess: RGB, center-crop → resize 224×224, ImageNet normalize, NCHW. Postprocess: softmax + top-k. Class index → species via sorted PlantNet species-id order (`PlantNetLabels.kt`).

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
cd android
./install_to_device.sh <dir-with-plantnet.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample classifies a bundled plant photo and shows the top-5 species. Adapt `MainActivity.kt` to feed live camera frames for a real-time plant-ID demo.

## Convert

See [`conversion/`](conversion/) — `build_plantnet.py` loads the Apache-2.0 ResNet18 weights and converts with litert-torch.
