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
"""qwen25vl_vision.py — Qwen2.5-VL-72B-Instruct's vision tower as two static graphs for LiteRT-LM's fast_vlm contract.

  VISION_ENCODER : image NHWC [1,G,G,3] in [0,1] -> [1,(G/28)^2,5120]  (ViT + merger ln_q + the 2x2 merge)
  VISION_ADAPTER : [1,N,5120] -> [1,N,8192]                            (merger MLP: Linear, GELU, Linear)

G is a multiple of 112 px (a window = 8x8 patches = 4x4 merge units). The tokens stay in raster patch order; window
blocks attend within each window, batched by reshape / transpose (no gather); blocks 7, 15, 23, 31 attend over the
whole image. HF orders the sequence window-major with merge units contiguous; attention is permutation-equivariant
and each token keeps its own 2D RoPE angles, so raster order with per-window batching is the same computation. The
patch embedding (Conv3d over a temporally duplicated 2x14x14 patch) becomes a stride-14 Conv2d with the two temporal
kernels summed; the CLIP mean / std normalization is in the graph. The 2x2 merge is four strided slices stacked on
the channel axis in the (dy, dx) order HF's merger reads. RoPE cos / sin come from HF's own rotary module.

Subcommands:
  selftest --config SNAP --layers 4 --g 448   random-weight HF tower vs the static module (torch, fp32)
  ref      --src SNAP --image PNG --g 448 --out ref.npz     HF reference through the HF image processor
  export   --src SNAP --g 448 --out-dir DIR [--quant dynamic_wi8_afp32]
  check    --enc E.tflite --adp A.tflite --ref ref.npz      CompiledModel CPU vs the reference
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

WINDOW = 8  # patches per window side (window_size 112 px / patch 14)
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def window_order(gh, gw):
    nh, nw = gh // WINDOW, gw // WINDOW
    return torch.arange(gh * gw).reshape(nh, WINDOW, nw, WINDOW).permute(0, 2, 1, 3).reshape(-1)


class StaticQwenVisionEncoder(nn.Module):
    def __init__(self, vision, g):
        super().__init__()
        cfg = vision.config
        self.p = cfg.patch_size
        self.gh = self.gw = g // self.p
        assert g % (WINDOW * self.p) == 0, f"G={g} must be a multiple of {WINDOW * self.p}"
        self.nh, self.nw = self.gh // WINDOW, self.gw // WINDOW
        self.dim = cfg.hidden_size
        self.heads = cfg.num_heads
        self.hd = self.dim // self.heads
        self.full = set(cfg.fullatt_block_indexes)
        # raster (h, w) positions -> HF's rotary module -> window order
        hh, ww = torch.meshgrid(torch.arange(self.gh), torch.arange(self.gw), indexing="ij")
        pos = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
        with torch.no_grad():
            cos, sin = vision.rotary_pos_emb(torch.zeros(1), pos)
        order = window_order(self.gh, self.gw)
        self.register_buffer("cos", cos.float()[order].clone(), persistent=False)
        self.register_buffer("sin", sin.float()[order].clone(), persistent=False)
        w = vision.patch_embed.proj.weight.detach().float()  # [D, 3, 2, 14, 14]
        self.register_buffer("conv_w", w.sum(2).contiguous(), persistent=False)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 1, 1, 3), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 1, 1, 3), persistent=False)
        self.blocks = vision.blocks
        self.ln_q = vision.merger.ln_q

    def to_windows(self, x):
        x = x.reshape(self.nh, WINDOW, self.nw, WINDOW * self.dim).permute(0, 2, 1, 3)
        return x.reshape(self.gh * self.gw, self.dim)

    def from_windows(self, x):
        x = x.reshape(self.nh, self.nw, WINDOW, WINDOW * self.dim).permute(0, 2, 1, 3)
        return x.reshape(self.gh * self.gw, self.dim)

    def attention(self, attn, x, windowed):
        s = x.shape[0]
        qkv = attn.qkv(x).reshape(s, 3, self.heads, self.hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        cos, sin = self.cos[:, None, :], self.sin[:, None, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        nwin = self.nh * self.nw if windowed else 1
        L = s // nwin
        q = q.reshape(nwin, L, self.heads, self.hd).permute(0, 2, 1, 3)
        k = k.reshape(nwin, L, self.heads, self.hd).permute(0, 2, 1, 3)
        v = v.reshape(nwin, L, self.heads, self.hd).permute(0, 2, 1, 3)
        probs = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (self.hd ** -0.5), dim=-1)
        o = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(s, self.dim)
        return attn.proj(o)

    def forward(self, image):
        x = (image - self.mean) / self.std
        x = F.conv2d(x.permute(0, 3, 1, 2), self.conv_w, stride=self.p)  # [1, D, gh, gw]
        x = self.to_windows(x.permute(0, 2, 3, 1).reshape(self.gh * self.gw, self.dim))
        for i, blk in enumerate(self.blocks):
            x = x + self.attention(blk.attn, blk.norm1(x), windowed=i not in self.full)
            x = x + blk.mlp(blk.norm2(x))
        x = self.ln_q(self.from_windows(x)).reshape(self.gh, self.gw, self.dim)
        parts = [x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]]
        n = (self.gh // 2) * (self.gw // 2)
        return torch.cat(parts, dim=-1).reshape(1, n, 4 * self.dim)


class StaticQwenVisionAdapter(nn.Module):
    def __init__(self, merger):
        super().__init__()
        self.mlp = merger.mlp

    def forward(self, feats):
        return self.mlp(feats)


def build_hf(src=None, layers=None, seed=0, cfg_dir=None):
    from transformers import AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
    cfg_dir = src or cfg_dir
    if not cfg_dir:
        raise SystemExit("pass --config DIR (a directory holding the model's config.json, e.g. the HF snapshot)")
    vcfg = AutoConfig.from_pretrained(cfg_dir).vision_config
    if layers:
        vcfg.depth = layers
        vcfg.fullatt_block_indexes = [layers - 1]
    vcfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    vision = Qwen2_5_VisionTransformerPretrainedModel(vcfg).float().eval()
    if src:
        from safetensors import safe_open
        wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
        sd = vision.state_dict()
        by_file = {}
        for name in sd:
            by_file.setdefault(wm["visual." + name], []).append(name)
        for fname, names in by_file.items():
            with safe_open(os.path.join(src, fname), framework="pt") as f:
                for name in names:
                    with torch.no_grad():
                        sd[name].copy_(f.get_tensor("visual." + name).float())
        print(f"loaded {len(sd)} vision tensors")
    else:
        with torch.no_grad():
            for _, p in vision.named_parameters():
                p.copy_(torch.randn_like(p) * (0.1 if p.dim() == 1 else p[0].numel() ** -0.5))
    return vcfg, vision


def hf_pixels(image_u8, vcfg):
    """The image processor's layout for a GxG image (no resize): [N, 3*2*14*14] in merge-block order, grid_thw."""
    p, t, m = vcfg.patch_size, vcfg.temporal_patch_size, vcfg.spatial_merge_size
    x = torch.from_numpy(image_u8).float() / 255.0
    x = ((x - torch.tensor(MEAN)) / torch.tensor(STD)).permute(2, 0, 1)[None]  # [1, 3, H, W]
    x = x.repeat(t, 1, 1, 1)  # temporal duplicate: [t, 3, H, W]
    gh, gw = x.shape[2] // p, x.shape[3] // p
    x = x.reshape(1, t, 3, gh // m, m, p, gw // m, m, p)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(gh * gw, 3 * t * p * p), torch.tensor([[1, gh, gw]])


def cmd_selftest(a):
    vcfg, vision = build_hf(layers=a.layers, seed=a.seed, cfg_dir=a.config)
    for g in [int(x) for x in a.g.split(",")]:
        img = np.random.default_rng(1).integers(0, 256, size=(g, g, 3), dtype=np.uint8)
        pv, grid = hf_pixels(img, vcfg)
        with torch.no_grad():
            ref = vision(hidden_states=pv, grid_thw=grid).pooler_output
            enc = StaticQwenVisionEncoder(vision, g).eval()
            adp = StaticQwenVisionAdapter(vision.merger).eval()
            got = adp(enc(torch.from_numpy(img).float()[None] / 255.0))[0]
        d = (got - ref).abs().max().item()
        rel = ((got - ref).norm() / ref.norm()).item()
        print(f"G={g} layers={a.layers} out {tuple(got.shape)} max|d| {d:.3e} rel {rel:.3e}")
    return 0


def cmd_ref(a):
    vcfg, vision = build_hf(src=a.src, layers=a.layers, seed=a.seed)
    from PIL import Image
    if a.image:
        img = np.asarray(Image.open(a.image).convert("RGB"), dtype=np.uint8)
        assert img.shape[:2] == (a.g, a.g), img.shape
    else:
        img = np.random.default_rng(1).integers(0, 256, size=(a.g, a.g, 3), dtype=np.uint8)
    if a.src and a.image:  # the processor's own pixel values (resize to the same size is the identity here)
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(a.src)
        out = proc(images=[Image.fromarray(img)], return_tensors="pt")
        pv, grid = out["pixel_values"].float(), out["image_grid_thw"]
        assert grid.tolist() == [[1, a.g // 14, a.g // 14]], grid.tolist()
        pv_mine, _ = hf_pixels(img, vcfg)
        print(f"processor vs in-tool pixels: max|d| {(pv - pv_mine).abs().max().item():.3e}")
    else:
        pv, grid = hf_pixels(img, vcfg)
    with torch.no_grad():
        o = vision(hidden_states=pv, grid_thw=grid)
    np.savez_compressed(a.out, image=img, out=o.pooler_output.numpy())
    print(f"ref out {tuple(o.pooler_output.shape)} -> {a.out}")
    return 0


def cmd_export(a):
    import litert_torch
    vcfg, vision = build_hf(src=a.src, layers=a.layers, seed=a.seed)
    enc = StaticQwenVisionEncoder(vision, a.g).eval()
    adp = StaticQwenVisionAdapter(vision.merger).eval()
    os.makedirs(a.out_dir, exist_ok=True)
    n = (a.g // 28) ** 2
    tag = f"g{a.g}" + (f"_l{a.layers}" if a.layers else "")
    for name, mod, sample in (("vision_encoder", enc, (torch.rand(1, a.g, a.g, 3),)),
                              ("vision_adapter", adp, (torch.randn(1, n, 4 * vcfg.hidden_size),))):
        path = os.path.join(a.out_dir, f"{name}_{tag}_fp32.tflite")
        with torch.no_grad():
            litert_torch.convert(mod, sample).export(path)
        print(f"wrote {path} {os.path.getsize(path)} B")
        for recipe in [r for r in a.quant.split(",") if r and r != "none"]:
            from litert_torch.generative.export_hf.core import export_lib
            short = {"dynamic_wi8_afp32": "wi8"}.get(recipe, recipe)
            qpath = os.path.join(a.out_dir, f"{name}_{tag}_{short}.tflite")
            os.replace(export_lib.quantize_model(path, recipe), qpath)
            print(f"wrote {qpath} {os.path.getsize(qpath)} B ({recipe})")
    return 0


def run_tflite(path, x):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    model = CompiledModel.from_file(path, hardware_accel=HardwareAccelerator.CPU)
    sig = list(model.get_signature_list().keys())[0]
    (iname, _), = model.get_input_tensor_details(sig).items()
    (oname, odet), = model.get_output_tensor_details(sig).items()
    ib = model.create_input_buffer_by_name(sig, iname)
    ib.write(np.ascontiguousarray(x.astype(np.float32)))
    ob = model.create_output_buffer_by_name(sig, oname)
    model.run_by_name(sig, {iname: ib}, {oname: ob})
    return np.array(ob.read(int(np.prod(odet["shape"])), np.float32)).reshape(odet["shape"])


def cmd_check(a):
    ref = np.load(a.ref)
    got = run_tflite(a.adp, run_tflite(a.enc, ref["image"].astype(np.float32)[None] / 255.0))[0]
    want = ref["out"]
    rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    cos = float(np.mean(np.sum(got * want, -1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1))))
    res = {"enc": a.enc, "adp": a.adp, "ref": a.ref, "max_abs": float(np.abs(got - want).max()), "rel_l2": rel,
           "mean_cos": cos}
    print(f"encoder+adapter max|d| {res['max_abs']:.3e} rel L2 {rel:.3e} mean cos {cos:.6f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--config", required=True, help="a directory holding the model's config.json (the HF snapshot)")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--g", default="448")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_selftest)
    for name, fn in (("ref", cmd_ref), ("export", cmd_export)):
        p = sub.add_parser(name)
        p.add_argument("--src")
        p.add_argument("--layers", type=int)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--g", type=int, default=448)
        if name == "ref":
            p.add_argument("--image")
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--out-dir", required=True)
            p.add_argument("--quant", default="none")
        p.set_defaults(fn=fn)
    p = sub.add_parser("check")
    p.add_argument("--enc", required=True)
    p.add_argument("--adp", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
