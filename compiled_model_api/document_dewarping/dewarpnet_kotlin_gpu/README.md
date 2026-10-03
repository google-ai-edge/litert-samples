# DewarpNet — Document unwarping / rectification (LiteRT CompiledModel GPU)

**Document dewarping** running **fully on the LiteRT `CompiledModel` GPU** delegate.
[DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) (ICCV 2019) flattens a
photographed, curved/folded document — the core of a document scanner. Two CNNs predict
a backward-mapping grid; the network runs on the GPU and the `grid_sample` unwarp is a
tiny host-side step. ~24 ms/frame on a Pixel 8a.

- **Model:** [litert-community/DewarpNet-LiteRT](https://huggingface.co/litert-community/DewarpNet-LiteRT)
- **Weights:** [cvlab-stonybrook/DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) (doc3d) · MIT
- **Input:** `[1, 3, 256, 256]` NCHW, **BGR**, `x/255`
- **Output:** `[1, 2, 128, 128]` backward-mapping grid (~`[-1,1]`)

## How it works

DewarpNet (WCNet UNet + BMNet DenseNet) is a pure CNN, so the graph converts fully
GPU-compatible (**371/371 nodes on the delegate, 1 partition**; device corr 0.999866,
~24 ms) with two exact patches: `ConvTranspose2d` → ZeroStuffConvT2d (Mali rejects
`TRANSPOSE_CONV`) and `Hardtanh(0,1)` → `relu(x)-relu(x-1)` (Mali rejects `RELU_0_TO_1`).
CPU-exact vs PyTorch (corr 0.9999999999).

**Host-side unwarp** (`DocumentDewarper.kt`): the model outputs a backward-mapping grid;
for each output pixel, bilinearly read the map (over 128×128), convert the `[-1,1]` coord
to a source pixel, and bilinearly sample the source — a `grid_sample` in Kotlin.

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
cd android
./install_to_device.sh <dir-with-dewarp.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample dewarps a bundled photo of a curved document and shows the flattened result.
Adapt `MainActivity.kt` to feed live camera frames for a real-time document scanner.

## Convert

See [`conversion/`](conversion/) — `build_dewarp.py` fetches the MIT weights and converts
with litert-torch.
