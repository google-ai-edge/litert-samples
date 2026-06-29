# Object Detection with LiteRT — RT-DETRv2-S (on-device, fully-GPU)

An Android sample that runs **RT-DETRv2-S** ([lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR),
Baidu 2024, `PekingU/rtdetr_v2_r18vd`, Apache-2.0) end-to-end on device with the LiteRT `CompiledModel`
API. RT-DETRv2 is a real-time **transformer** detector (ResNet18-vd backbone + a hybrid AIFI/CCFM encoder
+ a plain deformable-attention DETR decoder). Both transformer graphs run on the GPU; only the topk
selection and a tiny per-token tail run on the CPU. The app detects objects in a bundled image and any
image picked from the gallery, drawing the boxes + COCO labels.

## Model (two-graph split)

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| Graph A — backbone + hybrid encoder + score head | image[1,3,640,640] → enc_class[1,8400,80], memory_raw[1,8400,256] | **GPU** ~6 ms |
| Graph B — two-stage combine + plain decoder + heads | (memory_raw, target[1,300,256], ref[1,300,4]) → boxes[1,300,4], logits[1,300,80] | **GPU** `704/704` |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — per-graph tflite-vs-torch
corr **1.0**. On a Pixel 8a (Tensor G3): both graphs fully on `LITERT_CL`; the device chain reproduces the
PyTorch detections **exactly** — COCO val *giraffe* image **7/7**, *cats* image **6/6**, every box at
**IoU 0.98–1.00** with matching class and score.

## How it splits (and why it's fully GPU)

RT-DETRv2 is a two-stage DETR. The query selection (top-300 proposals by class score) is `TOPK`/`GATHER`,
which have no GPU op — but the proposal grid is image-independent, so the model splits at exactly that point:

```
image →[GPU Graph A]→ enc_class, memory_raw
      →[host: top-300 by max class score; per-token tail on the 300 selected:
              target = enc_output(valid·memory_raw)   (Linear + LayerNorm)
              ref    = enc_bbox_head(target) + anchors (3-layer MLP)]
      →[GPU Graph B (memory_raw, target, ref)]→ boxes, logits
      →[host: sigmoid + threshold + cxcywh→xyxy + light NMS]→ detections
```

### The on-device gate — a Mali 3D-sequence fan-out bug

Both graphs convert GPU-clean and are fully `LITERT_CL`-resident, but a naïve Graph A (emitting
`enc_class`/`enc_coord`/`output_memory`/`memory_raw` together) silently produced wrong boxes on device —
large objects vanished while small ones stayed perfect. The cause is **not** an fp16-precision wall: it is
a Mali delegate bug where a **3-D *token* tensor `[1,N,256]`** (from `conv.flatten(2).transpose(1,2)`) that
is both a graph output and consumed by another node — or that fans out to several consumers — gets clobbered
on the longer branch. (4-D conv-map outputs with the same fan-out are fine.) Here `output_memory` fed both
the score head and the box head; the 3-layer box head lost, so its reference-box deltas collapsed to ~0 →
boxes shrank to the default anchor → large objects dropped. The corruption was masked in correlation by the
±1e4 baked anchors ("corr 1.0" while the real valid-row corr was 0.88).

**Fix:** Graph A emits only the two fp16-clean leaves — `enc_class` (the 1-layer score head survives) and
`memory_raw × 2` (the ×2 forces a separate, clean output buffer; exact in fp16, undone on the host). The
per-token tail (`enc_output` + `enc_bbox_head`) is moved to the host, in fp32, over only the 300 selected
tokens — exact, because per-token ops commute with the gather (`gather(f(x),i) = f(gather(x,i))`) — so the
reference boxes are perfect and every object survives. See [`conversion/`](conversion/) for the full
re-authoring (deformable `grid_sample` → tent-matmul, MSDeformAttn ≤4D, SafeLayerNorm, ResNet stem
zero-pad maxpool, baked sine pos-embed) and this fix.

## Output & decode

Graph B emits `boxes` (cxcywh, normalized `[0,1]`) and `logits` (80-way, contiguous COCO id `0–79`). The
host (`RtDetr.kt`) applies sigmoid + score threshold + `cxcywh→xyxy` + light NMS. Preprocessing: square
resize to 640×640, RGB, `[0,1]` rescale only (no ImageNet normalization), NCHW.

## Run

1. Build the two tflites + `host_params.bin` with [`conversion/`](conversion/) (`build_rtdetr_split.py` +
   `build_rtdetr_fix3.py` + `pack_assets.py`), or get the models from
   [litert-community/RT-DETRv2-S-LiteRT](https://huggingface.co/litert-community/RT-DETRv2-S-LiteRT).
2. Build/install the app and push the models:
   ```bash
   cd android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-the-two-tflites>
   ```
   (`test_image.jpg`, `coco_labels.txt`, `host_params.bin` are bundled.)
3. Launch **RT-DETRv2** — it compiles the GPU shaders (~1 s/graph on first launch), then detects objects.

## Files

| File | Description |
|------|-------------|
| `android/.../rt_detr_v2/RtDetr.kt` | Both GPU graphs on CompiledModel + host topk + per-token tail + decode + light NMS |
| `android/.../rt_detr_v2/MainActivity.kt` | Runs detection on a bundled image / gallery pick, overlays boxes + labels |
| `android/.../assets/coco_labels.txt` | 80-line COCO label table (contiguous class id 0–79) |
| `android/.../assets/host_params.bin` | `enc_output` + `enc_bbox_head` weights, valid mask, anchors (fp32) |
| `conversion/` | litert-torch conversion (2-graph split + the 3D-fan-out fix) + notes |

**Original model**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) (RT-DETRv2) ·
`PekingU/rtdetr_v2_r18vd` · [Apache-2.0](https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE)
