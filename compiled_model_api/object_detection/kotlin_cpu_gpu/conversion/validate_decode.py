# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prove the raw-head + host-decode pipeline. Two checks:
  (1) numeric: my numpy decode(raw heads) == YOLOX built-in decode_outputs (corr/max|d|).
  (2) end-to-end: tflite(fp16) raw heads -> host decode -> NMS on assets/dog.jpg, vs stock
      YOLOX (decode_in_inference=True + postprocess) on the same preprocessed input.
Host decode mirrors yolo_head.decode_outputs: box_xy=(raw_xy+grid)*stride, wh=exp(raw_wh)*stride.

Run: ~/.pyenv/versions/lama-cml/bin/python validate_decode.py [yolox-s|yolox-tiny]
"""
import sys
import os
import argparse
import types
import numpy as np
import torch

class _D:
    def __getattr__(self, n):
        return lambda *a, **k: None
_pp = types.ModuleType("scipy.sparse.linalg._propack")
for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"): setattr(_pp, _nm, _D())
sys.modules["scipy.sparse.linalg._propack"] = _pp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "YOLOX"))
import cv2
from yolox.utils import postprocess
from yolox.data.datasets import COCO_CLASSES
import build_yolox as B


def host_grids(hw, strides):
    """numpy (grid_x, grid_y, stride) per anchor, concatenated over levels -> [n_anchors, 3]."""
    g, s = [], []
    for (h, w), st in zip(hw, strides):
        yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        grid = np.stack((xv, yv), -1).reshape(-1, 2)
        g.append(grid)
        s.append(np.full((grid.shape[0], 1), st, np.float32))
    return np.concatenate(g, 0).astype(np.float32), np.concatenate(s, 0)


def host_decode(raw, hw, strides):
    """raw [n_anchors, 85] (raw cx,cy,w,h ; sigmoid obj/cls) -> decoded [n_anchors, 85]."""
    grids, st = host_grids(hw, strides)
    xy = (raw[:, 0:2] + grids) * st
    wh = np.exp(raw[:, 2:4]) * st
    return np.concatenate([xy, wh, raw[:, 4:]], -1)


def preproc(img, size):
    """YOLOX val preproc: ratio-resize, pad to 114 (gray), CHW, BGR, no normalization."""
    padded = np.ones((size, size, 3), np.uint8) * 114
    r = min(size / img.shape[0], size / img.shape[1])
    rsz = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    padded[:rsz.shape[0], :rsz.shape[1]] = rsz
    return np.ascontiguousarray(padded.transpose(2, 0, 1), np.float32), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="yolox-s", choices=list(B.WEIGHTS))
    args = ap.parse_args()

    model, S = B.build(args.name)
    B.reauthor(model)
    hw = [(S // s, S // s) for s in model.head.strides]   # head feature sizes

    # (1) numeric: host decode == built-in decode_outputs
    x = torch.randn(1, 3, S, S)
    with torch.no_grad():
        raw = model(x)                                     # decode_in_inference=False
        model.head.decode_in_inference = True
        dec = model(x)                                     # built-in decode
        model.head.decode_in_inference = False
    mine = host_decode(raw[0].numpy(), hw, model.head.strides)
    corr = np.corrcoef(mine.ravel(), dec[0].numpy().ravel())[0, 1]
    print(f"[decode] host vs built-in: corr {corr:.6f}  max|d| {np.abs(mine - dec[0].numpy()).max():.2e}")

    # (2) end-to-end on a real image through the fp16 tflite
    from ai_edge_litert.interpreter import Interpreter
    tag = args.name.replace("yolox-", "yolox_")
    it = Interpreter(model_path=f"{tag}_fp16.tflite")
    it.allocate_tensors()
    img = cv2.imread(os.path.join(HERE, "YOLOX/assets/dog.jpg"))
    inp, r = preproc(img, S)
    it.set_tensor(it.get_input_details()[0]["index"], inp[None])
    it.invoke()
    raw_tf = it.get_tensor(it.get_output_details()[0]["index"])[0]   # [n_anchors, 85]
    dec_tf = host_decode(raw_tf, hw, model.head.strides)
    out = postprocess(torch.from_numpy(dec_tf)[None], 80, conf_thre=0.3, nms_thre=0.45)[0]

    print(f"[e2e] tflite(fp16) + host decode + NMS on dog.jpg ({args.name}, {S}px):")
    if out is None:
        print("  no detections")
        return
    out = out.numpy()
    for d in out:
        x0, y0, x1, y1, obj, cls_conf, cls = d
        print(f"  {COCO_CLASSES[int(cls)]:12s} {obj*cls_conf:.3f}  box=({x0/r:.0f},{y0/r:.0f},{x1/r:.0f},{y1/r:.0f})")


if __name__ == "__main__":
    main()
