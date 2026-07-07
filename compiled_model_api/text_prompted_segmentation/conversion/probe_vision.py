# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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

"""Phase-1 probe: re-authored CLIP ViT-B/16 vision encoder @352 (484+1 tokens) -> fp16 device
corr. This decides CLIPSeg feasibility (between CLIP-B/32 49-token OK and DA-V2 784-token wall)."""
import os
import sys
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 352
EPS = 1e-4
SAFE = 16.0


def safe_ln(x, w, b):
    # vision variant (large-variance inputs): scaled-sum + eps 1e-4 (device-proven 0.998)
    m = x.mean(dim=-1, keepdim=True)
    d = (x - m) * (1.0 / SAFE)
    v = (d * d).mean(dim=-1, keepdim=True)
    return (x - m) / torch.sqrt(v * (SAFE * SAFE) + EPS) * w + b


def safe_ln_up(x, w, b, up=8.0, eps=1e-5):
    # small-variance variant (CLIP text / CLIPSeg decoder): up-scale x so the eps lands in
    # fp16-normal range (var(8x) + 64*eps = var*64 + 6.4e-4), exact vs torch to ~1e-5 relative.
    y = x * up
    m = y.mean(dim=-1, keepdim=True)
    d = (y - m) * (1.0 / SAFE)
    v = (d * d).mean(dim=-1, keepdim=True)
    return (y - m) / torch.sqrt(v * (SAFE * SAFE) + eps * up * up) * w + b


class VisionGPU(nn.Module):
    """Functional re-author of CLIPSegVisionTransformer, batch-1, taps at 3/6/9 + final."""

    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        emb = vm.embeddings
        # bake pos-embed interpolated 14x14 -> 22x22 (bicubic, align_corners=False as HF)
        pe = emb.position_embedding.weight                     # [197, 768]
        cls_pe, patch_pe = pe[:1], pe[1:]
        g = int(patch_pe.shape[0] ** 0.5)
        p2 = patch_pe.T.reshape(1, -1, g, g)
        p2 = F.interpolate(p2, size=(SIZE // 16, SIZE // 16), mode="bicubic", align_corners=False)
        p2 = p2.reshape(768, -1).T
        self.register_buffer("pos", torch.cat([cls_pe, p2], 0).unsqueeze(0))  # [1, 485, 768]

    def forward(self, x):
        vm = self.vm
        emb = vm.embeddings
        p = F.conv2d(x, emb.patch_embedding.weight, None, stride=16)   # [1, 768, 22, 22]
        p = p.flatten(2).transpose(1, 2)                                # [1, 484, 768]
        cls = emb.class_embedding.view(1, 1, -1)
        h = torch.cat([cls.expand(1, -1, -1), p], dim=1) + self.pos     # [1, 485, 768]
        h = safe_ln(h, vm.pre_layrnorm.weight, vm.pre_layrnorm.bias)
        taps = []
        for li, layer in enumerate(vm.encoder.layers):
            r = h
            y = safe_ln(h, layer.layer_norm1.weight, layer.layer_norm1.bias)
            a = layer.self_attn
            q = F.linear(y, a.q_proj.weight, a.q_proj.bias) * a.scale
            k = F.linear(y, a.k_proj.weight, a.k_proj.bias)
            v = F.linear(y, a.v_proj.weight, a.v_proj.bias)
            B, N, C = q.shape
            H = a.num_heads
            d = C // H
            q3 = q.reshape(N, H, d).permute(1, 0, 2)                    # [12, 485, 64]
            k3 = k.reshape(N, H, d).permute(1, 0, 2)
            v3 = v.reshape(N, H, d).permute(1, 0, 2)
            att = torch.matmul(q3, k3.transpose(1, 2))                  # [12, 485, 485]
            att = F.softmax(att, dim=-1)
            o = torch.matmul(att, v3)                                   # [12, 485, 64]
            o = o.permute(1, 0, 2).reshape(1, N, C)
            h = r + F.linear(o, a.out_proj.weight, a.out_proj.bias)
            r = h
            y = safe_ln(h, layer.layer_norm2.weight, layer.layer_norm2.bias)
            y = F.linear(y, layer.mlp.fc1.weight, layer.mlp.fc1.bias)
            y = y * torch.sigmoid(1.702 * y)                            # quick_gelu
            h = r + F.linear(y, layer.mlp.fc2.weight, layer.mlp.fc2.bias)
            if li in (3, 6, 9):                             # CLIPSeg: hidden_states[i+1] (+1 for embeddings)
                taps.append(h)
        return taps[0], taps[1], taps[2], h


def main():
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    m = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval()
    vm = m.clip.vision_model
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    from PIL import Image
    img = Image.open(os.path.join(HERE, "cats.jpg"))
    x = proc(images=img, return_tensors="pt")["pixel_values"]
    print("input", tuple(x.shape))

    with torch.no_grad():
        ref = vm(x, interpolate_pos_encoding=True, output_hidden_states=True)
        refs = [ref.hidden_states[i] for i in (3, 6, 9)] + [ref.hidden_states[12]]
        g = VisionGPU(vm).eval()
        mine = g(x)
    for nm, a, b in zip(["tap3", "tap6", "tap9", "final"], mine, refs):
        c = np.corrcoef(a.numpy().ravel(), b.numpy().ravel())[0, 1]
        print(f"[torch] {nm}: corr {c:.7f} absmax {a.abs().max():.1f}")

    import litert_torch
    sys.path.insert(0, os.path.expanduser("~/Downloads/meeting/cmgan-work"))
    from build_cmgan import opcheck, to_fp16
    fp32 = os.path.join(HERE, "clipseg_vis.tflite")
    litert_torch.convert(g, (x,)).export(fp32)
    it, clean = opcheck(fp32, "fp32")
    if clean:
        fp16 = to_fp16(fp32, os.path.join(HERE, "clipseg_vis_fp16.tflite"))
        opcheck(fp16, "fp16")
        x.numpy().astype(np.float32).tofile(os.path.join(HERE, "vis_input.bin"))
        np.save(os.path.join(HERE, "vis_ref.npy"),
                np.concatenate([t.numpy().ravel() for t in mine]))
        print("wrote fp16 + fixtures")


if __name__ == "__main__":
    main()
