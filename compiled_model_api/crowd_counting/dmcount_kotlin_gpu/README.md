# DM-Count — Crowd counting (LiteRT CompiledModel GPU)

**Crowd counting** running **fully on the LiteRT `CompiledModel` GPU** delegate.
[DM-Count](https://github.com/cvlab-stonybrook/DM-Count) (NeurIPS 2020) regresses a person
**density map** whose sum is the crowd size — it counts hundreds of people where
detector-based counting saturates. ~79 ms/frame on a Pixel 8a.

- **Model:** [litert-community/DM-Count-Crowd-LiteRT](https://huggingface.co/litert-community/DM-Count-Crowd-LiteRT)
- **Weights:** [cvlab-stonybrook/DM-Count](https://github.com/cvlab-stonybrook/DM-Count) (UCF-QNRF) · MIT
- **Input:** `[1, 3, 512, 512]` NCHW, RGB, ImageNet-normalized
- **Output:** `[1, 1, 64, 64]` non-negative density map — `sum(map)` = person count

## How it works

DM-Count is a pure CNN (VGG19 + conv regression head) → fully GPU-compatible (**30/30 nodes
on the delegate, 1 partition**; device corr 0.9998–1.0, count within 0.4% of PyTorch). The
one graph change is exact: the mid-graph align_corners=True bilinear upsample (banned on the
delegate) is re-authored as two constant-matrix multiplies at conversion time. Decode:
`sum(map)` = count; normalize the map per frame for a heatmap overlay.

## Run

```bash
cd android
./install_to_device.sh <dir-with-dmcount.tflite>
./gradlew :app:installDebug
```

The sample counts the people in a bundled crowd photo and overlays the density heatmap.
Adapt `MainActivity.kt` to feed camera frames for live crowd monitoring.

## Convert

See [`conversion/`](conversion/) — `build_dmcount.py` loads the MIT UCF-QNRF weights and
converts with litert-torch; `validate_dmcount.py` checks GPU-op cleanliness and
CompiledModel parity.
