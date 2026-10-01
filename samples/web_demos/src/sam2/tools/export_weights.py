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

"""Derive the 512x512 weight file for the Tensor API SAM2 video graphs from the
1024 export (export_weights_1024.py in litert-samples).

All learned weights are resolution independent and copied unchanged. Only the
tables baked for the input resolution are recomputed, by the HF model itself
configured for 512 (image_size, backbone feature sizes, RoPE grid):

  trunk.pos_embed_full        [1,128,128,96]  (1024: [1,256,256,96])
  tables.rope_cos / rope_sin  [1024,256]      deinterleaved, 32x32 grid
  tables.vision_pos_scaled    [1024,256]      0.1 * top-level sine PE
  tables.mem_pos              [1024,64]
  tables.track_sparse         [2,256]         (resolution independent; recomputed as a check)

Self-check: the same procedure at 1024 must reproduce the 1024 file's tables,
so the recipe (not just the shapes) is validated before writing 512.

  python tools/export_weights.py --size 384 --base $ARTIFACTS/sam2_tiny_1024_video.safetensors \
      --out $ARTIFACTS/sam2_tiny_384_video.safetensors
"""
import argparse

import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from transformers import Sam2VideoConfig, Sam2VideoModel

CKPT = "facebook/sam2.1-hiera-tiny"
HD = 256


def model_at(size):
    cfg = Sam2VideoConfig.from_pretrained(CKPT)
    g = size // 16
    cfg.image_size = size
    cfg.prompt_encoder_config.image_size = size
    cfg.mask_decoder_config.image_size = size
    cfg.vision_config.backbone_config.image_size = [size, size]
    cfg.vision_config.backbone_feature_sizes = [[4 * g, 4 * g], [2 * g, 2 * g], [g, g]]
    cfg.memory_attention_rope_feat_sizes = [g, g]
    return Sam2VideoModel.from_pretrained(CKPT, config=cfg).eval()


def tables(model, size):
    g, tg = size // 16, size // 4
    out = {}
    with torch.no_grad():
        bb = model.vision_encoder.backbone
        out["trunk.pos_embed_full"] = bb._get_pos_embed((tg, tg)).detach().contiguous().clone()
        ma = model.memory_attention
        if hasattr(ma.rotary_emb, "rope_embeddings_cos"):
            cos = ma.rotary_emb.rope_embeddings_cos.detach().clone()
            sin = ma.rotary_emb.rope_embeddings_sin.detach().clone()
        else:
            cos, sin = ma.rotary_emb(torch.zeros(1), ma.position_ids)
        cos, sin = cos.reshape(g * g, HD), sin.reshape(g * g, HD)
        assert torch.equal(cos[:, 0::2], cos[:, 1::2]), "expected pairwise-equal"
        half = lambda c: torch.cat([c[:, 0::2], c[:, 0::2]], dim=-1).contiguous()
        out["tables.rope_cos"] = half(cos)
        out["tables.rope_sin"] = half(sin)
        ve = model.vision_encoder(torch.randn(1, 3, size, size), return_dict=True)
        vp = ve.fpn_position_encoding[-1]
        assert vp.shape == (1, HD, g, g), vp.shape
        out["tables.vision_pos_scaled"] = (0.1 * vp).reshape(HD, g * g).T.contiguous()
        mp = model.memory_encoder.position_encoding((1, 64, g, g), torch.device("cpu"), torch.float32)
        out["tables.mem_pos"] = mp.reshape(64, g * g).T.contiguous()
        ts, _ = model.prompt_encoder(input_points=torch.zeros(1, 1, 1, 2),
                                     input_labels=-torch.ones(1, 1, 1, dtype=torch.int32),
                                     input_boxes=None, input_masks=None)
        out["tables.track_sparse"] = ts.reshape(2, HD).contiguous()
    return {k: np.ascontiguousarray(v.numpy().astype(np.float32)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()

    base = load_file(a.base)
    check = tables(model_at(1024), 1024)
    for k, v in check.items():
        d = float(np.abs(v - base[k]).max())
        print(f"  1024 recipe check {k:28s} {tuple(v.shape)} max|d|={d:.2e}")
        assert v.shape == base[k].shape and d < 1e-4, k

    new = tables(model_at(a.size), a.size)
    out = dict(base)
    for k, v in new.items():
        print(f"  {a.size} {k:28s} {tuple(base[k].shape)} -> {tuple(v.shape)}")
        out[k] = v
    assert np.abs(out["tables.track_sparse"] - base["tables.track_sparse"]).max() < 1e-5
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
