#!/usr/bin/env python3
# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Verifies a Sam2Pipeline (ModelChain) run dumped by sam2_chain_main (native)
or by the browser e2e test (same dump layout):

  <dump>/meta.json               size, width, height, frames, nmm, prompts, effect, stroke
  <dump>/pixels_f{t}.f32         preprocess output [S,S,3] (the encoder's input)
  <dump>/mask_o{k}_f{t}.f32      object k low-res mask logits [S/4,S/4]
  <dump>/score_o{k}_f{t}.f32     object score, iou
  <dump>/rgb_f{t}.f32            composite [H,W,3]

Checks
  preprocess  pixels vs a numpy reference of the graph (crop, average pool,
              bilinear half-pixel resize, ImageNet normalization) run on the
              raw RGBA clip
  masks       every object / frame vs HF Sam2VideoModel run on the SAME
              preprocessed frames with the same prompts and memory size
              (one HF session per object, from its prompt frame)
  composite   rgb vs a numpy reference of the composite graph fed the dumped
              masks and the raw frame
  hf_preproc  (informational) masks vs HF run on HF's own preprocessing of
              the raw frames

  python verify_chain.py --dump DIR --rgba clip.rgba [--png] [--min_iou 0.99]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # export_weights.py

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PALETTE = np.array([[76, 141, 255], [255, 158, 44], [62, 207, 142],
                    [240, 82, 156], [170, 110, 255], [250, 204, 21]], np.float32) / 255
GREEN = np.array([0, 177, 64], np.float32) / 255
K = 6


def parse_prompts(spec):
    out = []
    for item in filter(None, spec.split('|')):
        head, pts = item.split(':')
        obj, frame = map(int, head.split('@'))
        out.append((obj, frame, [tuple(map(float, p.split(','))) for p in pts.split(';')]))
    return out


def describe(pts):
    """'2 clicks', 'box + 1 click' (labels 2 / 3 are a box's corners)."""
    clicks = sum(1 for *_, l in pts if int(l) in (0, 1))
    box = any(int(l) == 2 for *_, l in pts)
    c = f'{clicks} click{"s" if clicks != 1 else ""}'
    return f'box + {c}' if box and clicks else 'box' if box else c


def bilinear(x, oh, ow):
    """TFLite RESIZE_BILINEAR, half_pixel_centers=True (= torch align_corners=False). x: [H,W,C]."""
    ih, iw = x.shape[:2]

    def taps(o, i):
        s = (np.arange(o, dtype=np.float64) + 0.5) * (i / o) - 0.5
        f = np.floor(s)
        lo = np.clip(f, 0, i - 1).astype(int)
        hi = np.clip(np.ceil(s), 0, i - 1).astype(int)
        return lo, hi, (s - lo).astype(np.float32)

    y0, y1, fy = taps(oh, ih)
    x0, x1, fx = taps(ow, iw)
    fy = fy[:, None, None]
    fx = fx[None, :, None]
    a, b = x[y0][:, x0], x[y0][:, x1]
    c, d = x[y1][:, x0], x[y1][:, x1]
    return (a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + d * fy * fx).astype(np.float32)


def preprocess_ref(rgba, S):
    x = rgba[..., :3].astype(np.float32) / 255.0
    H, W = x.shape[:2]
    ky, kx = max(1, H // S), max(1, W // S)
    if ky > 1 or kx > 1:
        h, w = H // ky, W // kx
        x = x[:h * ky, :w * kx].reshape(h, ky, w, kx, 3).mean(axis=(1, 3))
    x = bilinear(x, S, S)
    return (x - MEAN) / STD


def composite_ref(rgba, masks, fx):
    """masks: list of K [m,m] logits (or None). fx: [fill, ring, stroke, matte, spot]."""
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    H, W = rgb.shape[:2]
    m = next(mm.shape[0] for mm in masks if mm is not None)
    stack = np.stack([mm if mm is not None else np.full((m, m), -1024, np.float32) for mm in masks], -1)
    L = bilinear(stack, H, W)
    P = np.pad(L, ((1, 1), (1, 1), (0, 0)), mode='edge')
    gx = 0.5 * (P[1:-1, 2:] - P[1:-1, :-2])
    gy = 0.5 * (P[2:, 1:-1] - P[:-2, 1:-1])
    d = np.clip(L / np.maximum(np.sqrt(gx * gx + gy * gy), 1e-3), -1000, 1000)
    cov = np.clip(d + 0.5, 0, 1)
    fill, ring, stroke, matte, spot = fx[:5]
    sa_all = ring * np.clip(0.5 * stroke + 0.5 - np.abs(d), 0, 1)
    fa_all = fill * cov
    over = rgb.copy()
    for k in range(K):
        sa, fa = sa_all[..., k:k + 1], fa_all[..., k:k + 1]
        under = fa * (1 - sa)
        over = under * PALETTE[k] + sa + over * (1 - (sa + under))
    u = cov.max(-1, keepdims=True)
    a, b = 1 - 0.85, 0.35
    M = b * np.array([[0.2126 + 0.7874 * a, 0.7152 - 0.7152 * a, 0.0722 - 0.0722 * a],
                      [0.2126 - 0.2126 * a, 0.7152 + 0.2848 * a, 0.0722 - 0.0722 * a],
                      [0.2126 - 0.2126 * a, 0.7152 - 0.7152 * a, 0.0722 + 0.9278 * a]], np.float32)
    dim = rgb @ M.T
    bg = dim * spot + GREEN * (1 - spot)
    mat = rgb * u + bg * (1 - u)
    return mat * matte + over * (1 - matte)


def effect_params(effect, stroke):
    if effect == 'overlay':
        return [0.5, 0.95 if stroke > 0 else 0.0, stroke, 0.0, 0.0]
    return [0.0, 0.0, stroke, 1.0, 1.0 if effect == 'spotlight' else 0.0]


def hf_masks(S, nmm, prompts, frames_nchw, T):
    """HF Sam2VideoModel, one streaming session per object from its prompt frame."""
    import torch
    from transformers import Sam2VideoInferenceSession
    from export_weights import model_at
    model = model_at(S)
    model.num_maskmem = nmm
    out = {}
    for obj, f0, pts in prompts:
        sess = Sam2VideoInferenceSession(video_height=S, video_width=S, dtype=torch.float32)
        oi = sess.obj_id_to_idx(1)
        coords = [[min(max(x * S, 0), S - 1), min(max(y * S, 0), S - 1)] for x, y, _ in pts]
        sess.add_point_inputs(oi, 0, {'point_coords': torch.tensor([[coords]], dtype=torch.float32),
                                      'point_labels': torch.tensor([[[int(l) for _, _, l in pts]]], dtype=torch.int32)})
        sess.obj_with_new_inputs = [1]
        with torch.no_grad():
            for t in range(f0, T):
                o = model(inference_session=sess, frame=torch.from_numpy(frames_nchw(t)))
                out[(obj, t)] = o.pred_masks.numpy().reshape(S // 4, S // 4)
                out[(obj, t, 'score')] = float(o.object_score_logits.reshape(-1)[0])
    return out


def iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--rgba', required=True, help='raw uint8 RGBA clip the run consumed')
    ap.add_argument('--min_iou', type=float, default=0.99,
                    help='per frame: IoU >= this, or at most --max_xor pixels differ (tiny objects)')
    ap.add_argument('--max_xor', type=int, default=3)
    ap.add_argument('--min_mean_iou', type=float, default=0.99)
    ap.add_argument('--png', action='store_true', help='write composites as PNG')
    ap.add_argument('--hf_preproc', action='store_true', help='also compare vs HF own preprocessing')
    ap.add_argument('--tag', default='')
    ap.add_argument('--ref_cache', default='',
                    help='directory for a cached HF reference computed from the numpy preprocess reference '
                         '(one HF run shared by every native / browser run of the same clip + prompts)')
    a = ap.parse_args()
    meta = json.load(open(f'{a.dump}/meta.json'))
    S, W, H, T, nmm = meta['size'], meta['width'], meta['height'], meta['frames'], meta['nmm']
    prompts = parse_prompts(meta['prompts'])
    clip = np.fromfile(a.rgba, np.uint8).reshape(-1, H, W, 4)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"{'  ok ' if cond else '  FAIL'} {msg}")
        ok = ok and cond

    # ---- preprocess
    worst = 0.0
    for t in range(T):
        got = np.fromfile(f'{a.dump}/pixels_f{t}.f32', np.float32).reshape(S, S, 3)
        worst = max(worst, float(np.abs(got - preprocess_ref(clip[t], S)).max()))
    check(worst < 0.05, f'preprocess graph vs numpy reference, {T} frames: max |diff| {worst:.2e} (normalized units)')

    # ---- masks vs HF on the same frames
    def pixels(t):
        return np.fromfile(f'{a.dump}/pixels_f{t}.f32', np.float32).reshape(S, S, 3).transpose(2, 0, 1)[None].copy()

    if a.ref_cache:
        import hashlib
        key = hashlib.sha1(f"{S}|{nmm}|{T}|{meta['prompts']}|{os.path.getsize(a.rgba)}".encode()).hexdigest()[:12]
        path = f'{a.ref_cache}/hf_ref_{S}_nmm{nmm}_{key}.npz'
        if os.path.exists(path):
            z = np.load(path)
            ref = {(int(k.split('_')[1]), int(k.split('_')[2])): z[k] for k in z.files}
            print(f'  (HF reference from cache {os.path.basename(path)})')
        else:
            os.makedirs(a.ref_cache, exist_ok=True)
            ref = hf_masks(S, nmm, prompts, lambda t: preprocess_ref(clip[t], S).transpose(2, 0, 1)[None].copy(), T)
            np.savez_compressed(path, **{f'm_{k[0]}_{k[1]}': v for k, v in ref.items() if len(k) == 2})
            print(f'  (HF reference computed and cached: {os.path.basename(path)})')
    else:
        ref = hf_masks(S, nmm, prompts, pixels, T)
    for obj, f0, pts in prompts:
        ious, fg, xors = [], [], []
        for t in range(f0, T):
            m = np.fromfile(f'{a.dump}/mask_o{obj}_f{t}.f32', np.float32).reshape(S // 4, S // 4)
            ious.append(iou(m, ref[(obj, t)]))
            fg.append(int((m > 0).sum()))
            xors.append(int(((m > 0) != (ref[(obj, t)] > 0)).sum()))
        ious, xors = np.array(ious), np.array(xors)
        frame_ok = (ious >= a.min_iou) | (xors <= a.max_xor)
        check(bool(frame_ok.all()) and ious.mean() >= a.min_mean_iou,
              f'object {obj} ({describe(pts)} @ frame {f0}) vs HF, frames {f0}..{T - 1}: '
              f'IoU min {ious.min():.4f} mean {ious.mean():.4f}, max {xors.max()} px differ, '
              f'mask {min(fg)}..{max(fg)} px of {(S // 4) ** 2}')

    # ---- composite
    fx = effect_params(meta['effect'], meta['stroke'])
    for f in sorted(int(n[5:-4]) for n in os.listdir(a.dump) if n.startswith('rgb_f')):
        got = np.fromfile(f'{a.dump}/rgb_f{f}.f32', np.float32).reshape(H, W, 3)
        masks = []
        for k in range(K):
            p = f'{a.dump}/mask_o{k}_f{f}.f32'
            masks.append(np.fromfile(p, np.float32).reshape(S // 4, S // 4) if os.path.exists(p) else None)
        if all(m is None for m in masks):
            continue
        want = composite_ref(clip[f], masks, fx)
        diff = np.abs(got - want)
        check(diff.mean() < 2e-3 and np.quantile(diff, 0.999) < 0.05,
              f'composite frame {f} ({meta["effect"]}) vs numpy reference: mean |diff| {diff.mean():.2e}, '
              f'99.9% < {np.quantile(diff, 0.999):.2e}')
        if a.png:
            from PIL import Image
            Image.fromarray((np.clip(got, 0, 1) * 255 + 0.5).astype(np.uint8)).save(f'{a.dump}/rgb_f{f}.png')

    # ---- informational: HF's own preprocessing of the raw frames
    if a.hf_preproc:
        from transformers import Sam2VideoProcessor
        proc = Sam2VideoProcessor.from_pretrained('facebook/sam2.1-hiera-tiny')

        def hf_pixels(t):
            v = proc.video_processor(videos=[clip[t, ..., :3]], size={'height': S, 'width': S}, return_tensors='pt')
            return v['pixel_values_videos'][0].numpy()

        ref2 = hf_masks(S, nmm, prompts, hf_pixels, T)
        for obj, f0, _ in prompts:
            ious = [iou(np.fromfile(f'{a.dump}/mask_o{obj}_f{t}.f32', np.float32).reshape(S // 4, S // 4),
                        ref2[(obj, t)]) for t in range(f0, T)]
            print(f'  info object {obj}: IoU vs HF with its own preprocessing: min {min(ious):.4f} mean {np.mean(ious):.4f}')
    print(f"VERIFY {a.tag} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
