# Ultralytics YOLO26 (`yolo26n`) — LiteRT detector for the Pi GPU

**Source:** Ultralytics YOLO26 — <https://docs.ultralytics.com/models/yolo26/>
(weights `yolo26n.pt`, AGPL-3.0 — <https://github.com/ultralytics/ultralytics>).

The detector runs on the Raspberry Pi's V3D GPU. YOLO26 is end-to-end (NMS-free)
by default, but that head lowers to INT64 gather/select ops the LiteRT GPU
delegate rejects — so the plain export runs only on the CPU. Exporting the
**raw head (`end2end=False`)** gives a `[1, 84, 8400]` graph that runs fully on
the GPU; the decode + NMS then run on the CPU (`emulator/detector.py`).

The model file is **not** checked in (`*.tflite` is git-ignored, like every
other model in this sample). Convert it once and drop it in place.

## Convert

```bash
pip install ultralytics litert-torch ai-edge-litert
```

```python
from ultralytics import YOLO

m = YOLO("yolo26n.pt")                 # downloads the Ultralytics weights
m.model.model[-1].end2end = False      # raw head → GPU-clean output
m.export(format="litert", imgsz=640)   # → yolo26n.tflite, output [1, 84, 8400]
```

Verify it runs fully on the GPU with this repo's `gpu-clean-conversion` toolkit
(put `utilities/` on `PYTHONPATH`):

```python
from litert_gpu_toolkit.checker import check_gpu_compatibility, print_report

print_report(check_gpu_compatibility("yolo26n.tflite"))
# expect: Status: VERIFIED (fully on GPU)
```

## Where to put it

Copy the converted file next to this README:

```
assets/yolo/yolo26n.tflite
```

The catalog (`emulator/models.py`) resolves it from this path, and
`demo/gpu_detect.py` serves it on the GPU. Nothing else needs changing.
