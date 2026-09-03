# Converting lightweight-OpenPose to LiteRT

Reproduces the pose model used by this sample.

```bash
pip install litert-torch ai-edge-quantizer torch
python convert_pose_litert.py . 256
```

Produces:

- `pose_256.tflite` — fp32 (~16.4 MB)
- `pose_256_fp16.tflite` — fp16 (~8.3 MB, **used by the app**)

The official weights (`checkpoint_iter_370000.pth`, Apache-2.0) are downloaded automatically.

## Why this is GPU-clean (and MoveNet isn't)

The exported graph returns **only the final-stage keypoint heatmaps**. The keypoint decode (argmax over each heatmap) is done in the Android app, **not** in the graph. So the graph is just convolutions + the refinement adds — it lowers entirely to GPU-clean builtins and runs **fully on the GPU delegate**.

MoveNet's official `.tflite` instead bakes the decode into the graph (`GATHER_ND`), which the ML Drift GPU delegate cannot execute — so it only partially offloads and falls back to the CPU. Moving the decode to app code is the fix.

```
CONV_2D, DEPTHWISE_CONV_2D, TRANSPOSE, ELU(→EXP/SUB/GREATER_EQUAL/SELECT), ADD, PAD, CONCATENATION
```

## Notes

- **Heatmaps-only output.** `PoseHeatmaps` wraps the model to return the final-stage heatmaps `[1, 19, H/8, W/8]` (18 keypoints + background); PAFs and intermediate stages are dropped.
- **Channel-last I/O** (`to_channel_last_io(..., args=[0], outputs=[0])`): NHWC in and out.
- **fp16** (AI Edge Quantizer `FLOAT_CASTING`): GPU-native.
- Verified on-device (Pixel 8a): the fp16 model compiles to **158/158 nodes on the LiteRT GPU delegate (LITERT_CL)** with no CPU fallback.
