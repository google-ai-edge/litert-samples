# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-image end-to-end check of the RF-DETR-Seg Nano graphs.

Runs a photo through the full converted chain via the LiteRT CompiledModel
API (Graph A -> numpy host glue -> Graph B) and prints the detections. When
the `rfdetr` package is installed, also runs the official PyTorch model on
the same image and reports per-detection box IoU, mask IoU and class
agreement — the ship criterion (raw correlation alone does not prove the
decoded output is right).

Run (after build_rf_detr_seg_nano.py has emitted the artifacts):
  python verify_rf_detr_seg_nano.py photo.jpg [threshold]
"""

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
R, NPROP, NQ, NCLS, HID, MSK, GRID = 312, 676, 100, 91, 256, 78, 26
MEANS = np.array([0.485, 0.456, 0.406], np.float32)
STDS = np.array([0.229, 0.224, 0.225], np.float32)


def run_tflite(path, inputs):
  """Single inference through the CompiledModel API, buffers by size.

  Args:
    path: Path to the .tflite file.
    inputs: List of input arrays (any order, matched by element count).

  Returns:
    Dict {element_count: flat fp32 array} of the graph outputs.
  """
  from ai_edge_litert.compiled_model import CompiledModel
  model = CompiledModel.from_file(path)
  ins = model.create_input_buffers(0)
  outs = model.create_output_buffers(0)
  by_elems = {int(np.asarray(a).size): np.asarray(a) for a in inputs}
  for i in range(len(ins)):
    n = model.get_input_buffer_requirements(i)["buffer_size"] // 4
    ins[i].write(np.ascontiguousarray(by_elems[n].ravel(), np.float32))
  model.run_by_index(0, ins, outs)
  res = {}
  for j in range(len(outs)):
    n = model.get_output_buffer_requirements(j)["buffer_size"] // 4
    res[n] = outs[j].read(n, np.float32)
  return res


def host_select(enc_cls, delta, rp):
  """Proposal combine + topk-100 + gather + reparam (numpy port).

  Args:
    enc_cls: [676, 91] encoder class logits.
    delta: [676, 4] encoder box deltas.
    rp: [100, 4] learned refpoint_embed rows.

  Returns:
    refpoint [100, 4] for Graph B.
  """
  gy, gx = np.mgrid[0:GRID, 0:GRID]
  prop = np.stack([(gx + 0.5) / GRID, (gy + 0.5) / GRID], -1)
  prop = prop.reshape(NPROP, 2).astype(np.float32)
  cxcy = delta[:, :2] * 0.05 + prop
  wh = np.exp(delta[:, 2:]) * 0.05
  enc_coord = np.concatenate([cxcy, wh], -1)
  top = np.argsort(-enc_cls.max(-1))[:NQ]
  ts = enc_coord[top]
  ref = np.concatenate(
      [rp[:, :2] * ts[:, 2:] + ts[:, :2], np.exp(rp[:, 2:]) * ts[:, 2:]], -1)
  return ref.astype(np.float32)


def decode(boxes, logits, masks, thr, w0, h0):
  """sigmoid + threshold decode to pixel-space detections.

  Args:
    boxes: [100, 4] cxcywh in [0, 1].
    logits: [100, 91] class logits.
    masks: [100, 78, 78] raw mask logits.
    thr: Score threshold.
    w0: Original image width.
    h0: Original image height.

  Returns:
    List of (class_id, score, xyxy box, bool mask), best first.
  """
  score = 1.0 / (1.0 + np.exp(-logits.max(-1)))
  cls = logits.argmax(-1)
  keep = np.where((score > thr) & (cls > 0))[0]
  out = []
  for q in keep:
    cx, cy, bw, bh = boxes[q]
    box = np.array([(cx - bw / 2) * w0, (cy - bh / 2) * h0,
                    (cx + bw / 2) * w0, (cy + bh / 2) * h0])
    m = np.asarray(Image.fromarray(masks[q]).resize((w0, h0),
                                                    Image.BILINEAR)) > 0
    out.append((int(cls[q]), float(score[q]), box, m))
  return sorted(out, key=lambda d: -d[1])


def box_iou(a, b):
  """IoU of two xyxy boxes.

  Args:
    a: First box [x0, y0, x1, y1].
    b: Second box [x0, y0, x1, y1].

  Returns:
    Intersection-over-union in [0, 1].
  """
  x0, y0 = max(a[0], b[0]), max(a[1], b[1])
  x1, y1 = min(a[2], b[2]), min(a[3], b[3])
  inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
  ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
        - inter)
  return inter / ua if ua > 0 else 0.0


def torch_reference(x):
  """Official eager forward_export (no GPU patches).

  Args:
    x: [1, 3, 312, 312] normalized input.

  Returns:
    (coord, cls, masks) numpy arrays, or None when rfdetr/torch is not
    installed.
  """
  try:
    sys.path.insert(0, HERE)
    import tfm_compat  # noqa: F401
    import torch
    from rfdetr import RFDETRSegNano
  except ImportError as e:
    print(f"(skipping PyTorch reference: {e})")
    return None
  net = RFDETRSegNano().model.model.eval()
  net.export()
  with torch.no_grad():
    coord, cls, masks = net.forward_export(torch.from_numpy(x))
  return coord[0].numpy(), cls[0].numpy(), masks[0].numpy()


if __name__ == "__main__":
  if len(sys.argv) < 2:
    sys.exit("usage: verify_rf_detr_seg_nano.py photo.jpg [threshold]")
  thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
  pil = Image.open(sys.argv[1]).convert("RGB")
  w0, h0 = pil.size
  im = np.asarray(pil.resize((R, R), Image.BILINEAR), np.float32) / 255.0
  x = ((im - MEANS) / STDS).transpose(2, 0, 1)[None].astype(np.float32)

  clspos = np.fromfile(f"{HERE}/clspos.bin", np.float32)
  pospatch = np.fromfile(f"{HERE}/pospatch.bin", np.float32)
  rp = np.fromfile(f"{HERE}/refpoint_embed.bin", np.float32).reshape(NQ, 4)
  qf = np.fromfile(f"{HERE}/query_feat.bin", np.float32)

  a = run_tflite(f"{HERE}/rfdetrseg_graphA_fp16.tflite",
                 [x, clspos, pospatch])
  enc_cls = a[NPROP * NCLS].reshape(NPROP, NCLS)
  delta = a[NPROP * 4].reshape(NPROP, 4)
  mem = a[NPROP * HID] * 0.5  # graph outputs memory*2

  ref = host_select(enc_cls, delta, rp)
  b = run_tflite(f"{HERE}/rfdetrseg_graphB_fp16.tflite", [mem, ref, qf])
  dets = decode(b[NQ * 4].reshape(NQ, 4), b[NQ * NCLS].reshape(NQ, NCLS),
                b[NQ * MSK * MSK].reshape(NQ, MSK, MSK), thr, w0, h0)
  print(f"tflite   : {len(dets)} detections over thr={thr}")
  for c, s, box, m in dets:
    print(f"  COCO id {c:2d}  {s:.3f}  xyxy={np.round(box, 1)}  "
          f"mask px={int(m.sum())}")

  gold = torch_reference(x)
  if gold is not None:
    gdets = decode(gold[0], gold[1], gold[2], thr, w0, h0)
    print(f"pytorch  : {len(gdets)} detections over thr={thr}")
    for gc, gs, gbox, gm in gdets:
      best = max(dets, key=lambda d: box_iou(gbox, d[2]), default=None)
      if best is None:
        print(f"  COCO id {gc:2d}  {gs:.3f}  UNMATCHED")
        continue
      bi = box_iou(gbox, best[2])
      mi = float((gm & best[3]).sum()) / max(float((gm | best[3]).sum()), 1.0)
      ok = "OK" if gc == best[0] else f"MISMATCH({gc} vs {best[0]})"
      print(f"  COCO id {gc:2d}  {gs:.3f}: box IoU {bi:.4f}  "
            f"mask IoU {mi:.4f}  class {ok}")
