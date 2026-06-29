# Object Detection with LiteRT — D-FINE-S (on-device, fully-GPU)

An Android sample that runs **D-FINE-S** ([Peterande/D-FINE](https://github.com/Peterande/D-FINE),
USTC 2024, `ustc-community/dfine-small-coco`, Apache-2.0) — the SOTA real-time DETR — end-to-end on device
with the LiteRT `CompiledModel` API. D-FINE is a **transformer** detector (HGNetV2 backbone + a hybrid
AIFI/CCFM encoder + an **FDR** Fine-grained-Distribution-Refinement decoder). Both transformer graphs run
on the GPU; only the topk selection and a tiny per-token tail run on the CPU. This is a **still-image**
demo (bundled image + gallery pick): the GATHER-free deformable decoder over D-FINE's 8400 tokens / 80×80
levels is ~GPU-compute-bound (~450 ms/frame), so it is accurate but not real-time on-device. (The sibling
[RF-DETR](../rf_detr_kotlin_gpu) sample — one small 24×24 deformable level — is the real-time camera demo.)

## Model (two-graph split)

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| Graph A — HGNetV2 backbone + hybrid encoder + score head | image[1,3,640,640] → enc_class[1,8400,80], memory_raw[1,8400,256] | **GPU** `511/511` |
| Graph B — two-stage combine + FDR decoder + heads | (memory_raw, target[1,300,256], ref[1,300,4]) → boxes[1,300,4], logits[1,300,80] | **GPU** `850/850` |

fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — per-graph tflite-vs-torch
corr **1.0**. On a Pixel 8a (Tensor G3): both graphs fully on `LITERT_CL`; on a COCO val image (giraffe +
cows) the device chain reproduces the PyTorch detections at **IoU 0.99–1.00** with matching class and score.

## How it splits (and why it's fully GPU)

D-FINE is a two-stage DETR. The query selection (top-300 proposals by class score) is `TOPK`/`GATHER`,
which have no GPU op — but the proposal grid is image-independent, so the model splits at exactly that point:

```
image →[GPU Graph A]→ enc_class, memory_raw
      →[host: top-300 by max class score; per-token tail on the 300 selected:
              target = enc_output(valid·memory_raw)   (Linear + LayerNorm)
              ref    = enc_bbox_head(target) + anchors (3-layer MLP)]
      →[GPU Graph B (memory_raw, target, ref)]→ boxes, logits
      →[host: sigmoid + threshold + cxcywh→xyxy + light NMS]→ detections
```

### The on-device gate — a Mali 3D-sequence fan-out bug (NOT the FDR decoder)

A naïve Graph A (emitting `enc_class`/`enc_coord`/`output_memory`/`memory_raw` together) gave **0 detections**
on device, and it first looked like the FDR decoder collapsing in fp16. That was a **red herring.** The real
cause is a Mali delegate bug where a **3-D *token* tensor `[1,N,256]`** (from `conv.flatten(2).transpose(1,2)`)
that is both a graph output and consumed by another node — or that fans out to several consumers — is
silently clobbered on the longer branch (4-D conv-map outputs are fine). Here the raw `memory` output
(Graph B's cross-attention input) was garbage (device corr −0.02), so the decoder cross-attended to noise.

**Fix:** Graph A emits only the two fp16-clean leaves (`enc_class` + `memory_raw × 2`) and the per-token tail
(`enc_output` + `enc_bbox_head`) runs on the host, in fp32, over only the 300 selected tokens — exact, since
per-token ops commute with the gather. With clean memory the **FDR decoder is perfect** (correlation is not
the ship criterion — real-image detection IoU is). See [`conversion/`](conversion/) for the full re-authoring
(deformable `grid_sample` → tent-matmul, multi-level MSDeformAttn ≤4D, the FDR LQE `prob.topk` → iterative
max, `distance2bbox` `stack`→`cat`, baked AIFI sine pos-embed, SafeLayerNorm) and this fix.

## Output & decode

Graph B emits `boxes` (cxcywh, normalized `[0,1]`) and `logits` (80-way, contiguous COCO id `0–79`). The
host (`DFine.kt`) applies sigmoid + score threshold + `cxcywh→xyxy` + light NMS. Preprocessing: square
resize to 640×640, RGB, `[0,1]` rescale only (no ImageNet normalization), NCHW.

## Run

1. Build the two tflites + `host_params.bin` with [`conversion/`](conversion/) (`build_dfine_split.py` +
   `build_dfine_fix3.py` + `pack_assets.py`), or get the models from
   [litert-community/D-FINE-S-LiteRT](https://huggingface.co/litert-community/D-FINE-S-LiteRT).
2. Build/install the app and push the models:
   ```bash
   cd android
   ./gradlew :app:installDebug
   ./install_to_device.sh <dir-with-the-two-tflites>
   ```
   (`test_image.jpg`, `coco_labels.txt`, `host_params.bin` are bundled.)
3. Launch **D-FINE** — it compiles the GPU shaders (~1 s/graph on first launch), then detects objects.

## Files

| File | Description |
|------|-------------|
| `android/.../d_fine/DFine.kt` | Both GPU graphs on CompiledModel + host topk + per-token tail + decode + light NMS |
| `android/.../d_fine/MainActivity.kt` | Runs detection on a bundled image / gallery pick, overlays boxes + labels |
| `android/.../assets/coco_labels.txt` | 80-line COCO label table (contiguous class id 0–79) |
| `android/.../assets/host_params.bin` | `enc_output` + `enc_bbox_head` weights, valid mask, anchors (fp32) |
| `conversion/` | litert-torch conversion (2-graph split + the 3D-fan-out fix) + notes |

**Original model**: [Peterande/D-FINE](https://github.com/Peterande/D-FINE) ·
`ustc-community/dfine-small-coco` · [Apache-2.0](https://github.com/Peterande/D-FINE/blob/main/LICENSE)
