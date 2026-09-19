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

"""End-to-end check of a converted TIPSv2-B/14 DPT .tflite (CompiledModel).

Runs an image through the converted graph and writes a four-panel png
(input | depth | normals | ADE20K segmentation). When
build_tipsv2_b14_dpt.py has saved the official reference (ref_*.npy), also
reports per-output correlation / argmax agreement against it.

Run:
  python verify_tipsv2_b14_dpt.py tipsv2_b14_dpt_fp16.tflite image.jpg
"""

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = 448
SEG_RES = 256
N_CLS = 150


def preprocess(path):
  """Loads an image as [1, 3, 448, 448] float32 in [0, 1] (no mean/std).

  Args:
    path: Path to the image file.

  Returns:
    (input array [1, 3, 448, 448], the resized RGB uint8 image [448, 448, 3]).
  """
  im = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
  arr = np.asarray(im, np.float32) / 255.0
  return arr.transpose(2, 0, 1)[None], np.asarray(im)


def run(model_path, x):
  """Runs the graph through the CompiledModel API.

  Args:
    model_path: Path to the converted .tflite.
    x: Input array [1, 3, 448, 448] float32 in [0, 1].

  Returns:
    (depth [448, 448] metres, normals [3, 448, 448], seg [150, 256, 256]).
  """
  from ai_edge_litert.compiled_model import CompiledModel
  model = CompiledModel.from_file(model_path)
  inputs = model.create_input_buffers(0)
  outputs = model.create_output_buffers(0)
  inputs[0].write(np.ascontiguousarray(x, np.float32))
  model.run_by_index(0, inputs, outputs)
  depth = outputs[0].read(IMG * IMG, np.float32).reshape(IMG, IMG)
  normals = outputs[1].read(3 * IMG * IMG, np.float32).reshape(3, IMG, IMG)
  seg = outputs[2].read(N_CLS * SEG_RES * SEG_RES, np.float32)
  return depth, normals, seg.reshape(N_CLS, SEG_RES, SEG_RES)


def label_palette():
  """Deterministic 150-colour palette for the segmentation panel.

  Returns:
    uint8 array [150, 3].
  """
  rng = np.random.RandomState(0)
  return rng.randint(40, 255, size=(N_CLS, 3)).astype(np.uint8)


def render(rgb, depth, normals, seg, out_path):
  """Writes input | depth | normals | segmentation side by side.

  Args:
    rgb: Resized input image [448, 448, 3] uint8.
    depth: Depth [448, 448] in metres.
    normals: Unit normals [3, 448, 448].
    seg: Segmentation logits [150, 256, 256].
    out_path: Output png path.
  """
  disp = 1.0 / np.clip(depth, 1e-3, None)
  lo, hi = np.percentile(disp, 2), np.percentile(disp, 98)
  n = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
  depth_rgb = np.stack([n, 1 - np.abs(2 * n - 1), 1 - n], -1)  # near = red
  depth_rgb = (depth_rgb * 255).astype(np.uint8)
  normals_rgb = ((normals.transpose(1, 2, 0) + 1) * 127.5)
  normals_rgb = normals_rgb.clip(0, 255).astype(np.uint8)
  labels = seg.argmax(0)
  seg_rgb = np.asarray(Image.fromarray(label_palette()[labels]).resize(
      (IMG, IMG), Image.NEAREST))
  Image.fromarray(np.concatenate([rgb, depth_rgb, normals_rgb, seg_rgb], 1)
                  ).save(out_path)
  print(f"wrote {out_path}")


def main():
  """Runs the converted graph on an image and reports parity."""
  if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)
  model_path, image = sys.argv[1], sys.argv[2]
  x, rgb = preprocess(image)
  depth, normals, seg = run(model_path, x)
  labels = seg.argmax(0)
  hist = np.bincount(labels.ravel(), minlength=N_CLS)
  print(f"depth {depth.min():.2f}..{depth.max():.2f} m | normals |n| "
        f"{np.linalg.norm(normals, axis=0).mean():.3f} | seg top ids "
        f"{np.argsort(-hist)[:5].tolist()}")
  render(rgb, depth, normals, seg, os.path.join(HERE, "verify_out.png"))
  ref = os.path.join(HERE, "ref_depth.npy")
  if os.path.exists(ref):
    import torch
    import torch.nn.functional as F
    rd = np.load(ref)[0, 0]
    rn = np.load(os.path.join(HERE, "ref_normals.npy"))[0]
    rs = np.load(os.path.join(HERE, "ref_seg.npy"))[0]
    s448 = F.interpolate(torch.from_numpy(seg)[None], size=(IMG, IMG),
                         mode="bilinear", align_corners=False)[0].numpy()

    def c(a, b):
      return np.corrcoef(a.ravel(), b.ravel())[0, 1]

    print(f"vs official: depth corr {c(depth, rd):.6f} max|d| "
          f"{np.abs(depth - rd).max():.4f} m | normals corr "
          f"{c(normals, rn):.6f} | seg argmax agree "
          f"{(s448.argmax(0) == rs.argmax(0)).mean():.4f}")


if __name__ == "__main__":
  main()
