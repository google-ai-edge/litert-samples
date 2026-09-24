# 6DRepNet — Head pose estimation (LiteRT CompiledModel GPU)

Real-time **6-DoF head pose estimation** running **fully on the LiteRT `CompiledModel`
GPU** delegate. [6DRepNet](https://github.com/thohemp/6DRepNet) (ICIP 2022) regresses a
continuous 6D rotation from a face crop — yaw / pitch / roll for driver-monitoring, AR,
and attention. ~21 ms/frame on a Pixel 8a.

- **Model:** [litert-community/6DRepNet-HeadPose-LiteRT](https://huggingface.co/litert-community/6DRepNet-HeadPose-LiteRT)
- **Weights:** [thohemp/6DRepNet](https://github.com/thohemp/6DRepNet) (300W-LP) · MIT · RepVGG-B1g2
- **Input:** `[1, 3, 224, 224]` NCHW, RGB, ImageNet-normalized (a **face crop**)
- **Output:** `[1, 6]` continuous 6D rotation → Gram-Schmidt → yaw/pitch/roll (host-side)

## How it works

Deploy-mode RepVGG (plain convs + ReLU) is a pure CNN, so the graph converts fully
GPU-compatible (**36/36 nodes on the delegate, 1 partition**; device corr 0.9993, ~21 ms)
with **zero patches**. Host-side decode (`HeadPoseEstimator.kt`): Gram-Schmidt the 6D into
a 3×3 rotation matrix, then `pitch=atan2(R21,R22)`, `yaw=atan2(-R20,√(R00²+R10²))`,
`roll=atan2(R10,R00)`.

## App architecture

The Android app is **MVVM + Jetpack Compose** (Compose Material). `MainActivity` is a thin
Compose host; `MainViewModel` owns the `HeadPoseEstimator`, loads the model from `filesDir`,
runs inference on a single confined worker (`Dispatchers.Default.limitedParallelism(1)`), and
exposes an immutable `UiState`. The centered face crop, the 3D axis drawing (trig projection
onto a bitmap copy via `android.graphics.Canvas`), and the yaw/pitch/roll formatting all live
in the ViewModel; the screen just renders `resultImage` + `resultText`. A gallery picker lets
you run the model on your own photo; a bundled face still runs at launch.

### Files

| File | Role |
| --- | --- |
| `app/src/main/java/.../sixdrepnet/MainActivity.kt` | Thin Compose host: view model + gallery picker |
| `app/src/main/java/.../sixdrepnet/MainViewModel.kt` | Owns the estimator; crop + axis draw + status formatting; `UiState` |
| `app/src/main/java/.../sixdrepnet/UiState.kt` | Immutable screen state (result image, angle text, ms, errors) |
| `app/src/main/java/.../sixdrepnet/HeadPoseEstimator.kt` | LiteRT CompiledModel GPU inference + 6D→yaw/pitch/roll decode |
| `app/src/main/java/.../sixdrepnet/ImageUtils.kt` | Asset / gallery bitmap decoding helpers |
| `app/src/main/java/.../sixdrepnet/view/HeadPoseScreen.kt` | Status header, image picker, annotated result |
| `app/src/main/java/.../sixdrepnet/view/Theme.kt`, `view/Color.kt` | Compose theme |

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
cd android
./install_to_device.sh <dir-with-6drepnet.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample runs on a bundled face photo (centered crop) and draws the 3D head-pose axes.
Pick a photo from the gallery to pose your own image, or add a face detector and feed the
detected crop for full-frame / real-time head tracking.

## Convert

See [`conversion/`](conversion/) — `build_6drepnet.py` loads the MIT deploy weights and
converts with litert-torch.
