# OCR with LiteRT — PP-OCRv5 (on-device, fully GPU)

<p align="center"><img src="https://huggingface.co/litert-community/PP-OCRv5-LiteRT/resolve/main/hero.png" width="380" alt="PP-OCRv5 on-device OCR on a Pixel 8a"></p>


An Android sample that runs [PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) (PaddleOCR 2025) text detection + recognition end-to-end on device with the LiteRT `CompiledModel` API. It detects text regions in an image and reads each line, then overlays the boxes + recognized text.

The recognizer uses a **CTC head — no autoregressive decoder** — so both the detector and the recognizer run **fully on the GPU** delegate with no CPU/ONNX fallback. (A VLM-based OCR such as Florence-2 or GOT-OCR has an autoregressive decoder that must run on CPU; the classic CNN+CTC pipeline avoids that entirely.)

## Models

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| Detection (DBNet) | image[1,3,640,640] → prob map[1,1,640,640] | **GPU** |
| Recognition (SVTR + CTC) | line[1,3,48,320] → logits[1,T,18385] | **GPU** |

Both load with `CompiledModel.create(...)` on `Accelerator.GPU`. fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — per-graph tflite-vs-torch corr **1.000000**. On a Pixel 8a (Tensor G3): detector `777/777` + recognizer `827/827` on `LITERT_CL`, ~9 ms each; a 3-line image read 3/3 correct.

## Pipeline

```
image →[GPU detector]→ prob map → [CPU: threshold + connected components + unclip] → boxes
   → crop+resize each box →[GPU recognizer]→ CTC logits → [CPU: CTC greedy decode] → text
```

## Re-authoring (litert-torch, parity corr 1.0)

- **Detector DB-head `ConvTranspose2d`** → `ZeroStuffConvT2d`: 2D nearest-upsample × stride zero-stuff mask + flipped `conv2d` (`TRANSPOSE_CONV` is Mali-rejected; this is `RESIZE_NEAREST` + `MUL` + `CONV_2D`, numerically exact).
- **Recognizer SVTR attention** fused-QKV 5D reshape → split q/k/v into 4D (numerically identical).

See [`conversion/`](conversion/).

## Run

1. Build the two tflites with `conversion/build_det.py` + `conversion/build_rec.py` (or get them from Hugging Face — [litert-community/PP-OCRv5-LiteRT](https://huggingface.co/litert-community/PP-OCRv5-LiteRT)).
2. Build/install the app, then push the models into its private storage:
   ```bash
   ./kotlin_cpu_gpu/android/install_to_device.sh <dir-with-the-tflites>
   ```
3. Launch **PP-OCRv5** — it compiles the GPU shaders (~10 s first launch), detects + reads the text.

## Files

- `kotlin_cpu_gpu/android/` — `PpocrDetector.kt` (GPU detector + DB box postprocess), `PpocrRecognizer.kt` (GPU recognizer + CTC decode), `MainActivity.kt` (runs OCR on a bundled image, overlays boxes + text).
- `conversion/` — the litert-torch conversion scripts (`build_det.py`, `build_rec.py`).

Weights are converted from PaddleOCR via the [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch) port (Apache-2.0). Upstream: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0).
