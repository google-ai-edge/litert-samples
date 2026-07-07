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

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
cd android
./install_to_device.sh <dir-with-6drepnet.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample runs on a bundled face photo (centered crop) and draws the 3D head-pose axes.
Add a face detector and feed the detected crop for full-frame / real-time head tracking.

## Convert

See [`conversion/`](conversion/) — `build_6drepnet.py` loads the MIT deploy weights and
converts with litert-torch.
