#!/usr/bin/env python3
"""MLX (python) SAM2 hiera-tiny baseline on the same fixture/protocol as
the Tensor API harness: white circle (r = size/4 - 8), center positive
point, fp16, warm 5 + 20 timed runs, medians.

Usage:
    python mlx_512_bench.py --sam2_mlx /path/to/avbiswas-sam2-mlx \
        --weights /path/to/sam2_tiny_512.safetensors [--size 512]

Requires: mlx, numpy, opencv-python-headless (sam2-mlx import dependency).
The weights must be the 512-baked set (pos_embed_full re-composed at the
size/4 token grid) — see the README's Weights section.
"""
import argparse
import statistics
import sys
import time
from dataclasses import replace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sam2_mlx", required=True,
                    help="Path to an avbiswas/sam2-mlx checkout")
parser.add_argument("--weights", required=True,
                    help="sam2_tiny_512.safetensors (MLX naming, 512-baked)")
parser.add_argument("--size", type=int, default=512)
args = parser.parse_args()
SIZE = args.size

sys.path.insert(0, f"{args.sam2_mlx}/src")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_sam.config import model_config_for_name  # noqa: E402
from mlx_sam.models.segmenter import Sam2ImageSegmenter  # noqa: E402

config = model_config_for_name("sam2.1_hiera_tiny")
config = replace(config, hiera=replace(
    config.hiera, pos_embed_hw=(SIZE // 4, SIZE // 4), image_size=SIZE))
model = Sam2ImageSegmenter(config=config)
model.set_image_size(SIZE)
weights = mx.load(args.weights)
model.load_weights(list(weights.items()), strict=False)
mx.eval(model.parameters())

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
arr = np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)
cx = cy = SIZE // 2
r2 = (SIZE // 4 - 8) ** 2
yy, xx = np.mgrid[0:SIZE, 0:SIZE]
circle = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r2).astype(np.float32)
for c in range(3):
    arr[0, c] = (circle - mean[c]) / std[c]
pixels = mx.array(arr).astype(mx.float16)

coords = mx.array([[[SIZE / 2.0, SIZE / 2.0]]])
labels = mx.array([[1]], dtype=mx.int32)


def encode():
    out = model.encode_image(pixels)
    mx.eval(out["vision_features"], *out["high_res_features"])
    return out


def decode(enc):
    out = model.predict_from_encoded(enc, coords, labels)
    mx.eval(out["low_res_masks"], out["ious"], out["object_score_logits"])
    return out


for _ in range(5):
    decode(encode())

enc_times = []
for _ in range(20):
    t = time.perf_counter()
    encode()
    enc_times.append((time.perf_counter() - t) * 1000)
enc0 = encode()
dec_times = []
for _ in range(20):
    t = time.perf_counter()
    decode(enc0)
    dec_times.append((time.perf_counter() - t) * 1000)

out = decode(enc0)
ious = np.array(out["ious"]).flatten()
masks = np.array(out["low_res_masks"].astype(mx.float32))
masks3 = masks.reshape(-1, masks.shape[-2], masks.shape[-1])[-3:]
best = int(ious[-3:].argmax())
fg = int((masks3[best] > 0).sum())
print(f"MLX python {mx.__version__} fp16 {SIZE}x{SIZE}")
print(f"enc_median={statistics.median(enc_times):.2f}ms "
      f"dec_median={statistics.median(dec_times):.2f}ms (runs=20)")
print(f"iou_scores={np.round(ious[-3:], 4).tolist()} "
      f"object_score={float(np.array(out['object_score_logits']).flatten()[0]):.2f}")
print(f"best mask=[{best}] fg={fg}/{masks3[best].size}")
