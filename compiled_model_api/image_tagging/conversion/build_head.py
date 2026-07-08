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

"""Phase 1b: re-author RAM++ reweight + Query2Label tagging head.

GPU-clean, torch parity vs reference logits. Fed with REF image_embeds
so head correctness is isolated from the Swin encoder.

Usage: python build_head.py [parity|convert]
"""
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn

from ram_load import load_ram_plus
from build_swin import tanh_gelu, corr
WORK = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WORK, "out")
os.makedirs(OUT, exist_ok=True)
REF = np.load(os.path.join(WORK, "ref", "ref_demo1.npz"))

def safe_ln_mod(x, ln, eps_floor=1e-4):
    """fp16-safe LayerNorm with an eps floor.

    Args:
        x: Input tensor [..., C].
        ln: The nn.LayerNorm module supplying weight/bias/eps.
        eps_floor: Minimum epsilon used inside the rsqrt.

    Returns:
        The normalized tensor, same shape as x.
    """
    mu = x.mean(-1, keepdim=True)
    dd = x - mu
    var = (dd * dd).mean(-1, keepdim=True)
    eps = max(float(ln.eps), eps_floor)
    return dd * torch.rsqrt(var + eps) * ln.weight + ln.bias

class TagHeadGPU(nn.Module):
    """Graph B (GPU): 2-layer Query2Label cross-attention head.

    queries[1,C,768] + image_embeds [1,145,512] -> logits[1,C].
    tanh-GELU + safe LayerNorm (fp16-safe).
    """
    def __init__(self, model):
        super().__init__()
        self.m = model
    def forward(self, h, image_embeds):
        th = self.m.tagging_head
        for layer in th.encoder.layer:
            ca = layer.crossattention
            s = ca.self
            nH, hd = s.num_attention_heads, s.attention_head_size
            Nq = h.shape[1]
            Nk = image_embeds.shape[1]
            q = s.query(h).reshape(1, Nq, nH, hd).permute(0, 2, 1, 3)
            k = s.key(image_embeds).reshape(1, Nk, nH, hd).permute(0, 2, 1, 3)
            v = (s.value(image_embeds)
                 .reshape(1, Nk, nH, hd).permute(0, 2, 1, 3))
            att = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(hd),
                                dim=-1)
            ctx = (att @ v).permute(0, 2, 1, 3).reshape(1, Nq, nH * hd)
            h = safe_ln_mod(ca.output.dense(ctx) + h, ca.output.LayerNorm)
            ff = tanh_gelu(None, layer.intermediate.dense(h))
            h = safe_ln_mod(layer.output.dense(ff) + h, layer.output.LayerNorm)
        return self.m.fc(h).squeeze(-1)

class Reweight(nn.Module):
    """Graph R (CPU): cls[1,512] -> reweighted tag queries[1,C,768].

    Bakes the frozen 479MB label_embed (4585*51 x 512). Multi-grained:
    softmax over 51 descriptions/class.
    """
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.num_class = model.num_class
        # store the tag bank as fp16 (one copy); cast at runtime so it
        # is baked ONCE at 240MB
        self.register_buffer("label_embed_h", model.label_embed.detach().half())
        self.scale = float(model.reweight_scale.exp())
    def forward(self, cls):
        cls_n = cls / cls.norm(dim=-1, keepdim=True)
        le = self.label_embed_h.float()  # single fp16 const + runtime cast
        # lpi: [1,C,51]
        lpi = (self.scale * cls_n @ le.t()).view(1, self.num_class, -1)
        wn = torch.softmax(lpi, dim=2)
        le_r = le.view(self.num_class, -1, 512)  # [C,51,512]
        rw = (wn[0].unsqueeze(-1) * le_r).sum(dim=1).unsqueeze(0)  # [1,C,512]
        return torch.relu(self.m.wordvec_proj(rw))

class RamHead(nn.Module):
    """reweight(cls) + 2-layer Q2L cross-attention head -> logits[1,4585]."""
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.num_class = model.num_class
        self.register_buffer("label_embed", model.label_embed.detach())
        self.scale = float(model.reweight_scale.exp())

    def reweight(self, cls):
        # cls [1,512]
        cls_n = cls / cls.norm(dim=-1, keepdim=True)
        le = self.label_embed  # [C*51, 512]
        # lpi: [1,C,51]
        lpi = (self.scale * cls_n @ le.t()).view(1, self.num_class, -1)
        wn = torch.softmax(lpi, dim=2)
        reshaped = le.view(self.num_class, -1, 512)  # [C,51,512]
        # rw: [1,C,512]
        rw = (wn[0].unsqueeze(-1) * reshaped).sum(dim=1).unsqueeze(0)
        return torch.relu(self.m.wordvec_proj(rw))  # [1,C,768]

    def forward(self, image_embeds, cls):
        h = self.reweight(cls)  # [1,C,768]
        th = self.m.tagging_head
        for layer in th.encoder.layer:
            ca = layer.crossattention
            s = ca.self
            nH, hd = s.num_attention_heads, s.attention_head_size
            Nq = h.shape[1]
            Nk = image_embeds.shape[1]
            q = s.query(h).reshape(1, Nq, nH, hd).permute(0, 2, 1, 3)
            k = s.key(image_embeds).reshape(1, Nk, nH, hd).permute(0, 2, 1, 3)
            v = (s.value(image_embeds)
                 .reshape(1, Nk, nH, hd).permute(0, 2, 1, 3))
            att = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(hd),
                                dim=-1)
            ctx = (att @ v).permute(0, 2, 1, 3).reshape(1, Nq, nH * hd)
            h = safe_ln_mod(ca.output.dense(ctx) + h, ca.output.LayerNorm)
            ff = tanh_gelu(None, layer.intermediate.dense(h))
            h = safe_ln_mod(layer.output.dense(ff) + h, layer.output.LayerNorm)
        return self.m.fc(h).squeeze(-1)  # [1,C]

def main():
    """Runs the requested mode: parity, full, convert, or reweight."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "parity"
    model = load_ram_plus(384)
    head = RamHead(model).eval()
    image_embeds = torch.from_numpy(REF["image_embeds"])  # [1,145,512]
    cls = image_embeds[:, 0, :]  # [1,512]
    with torch.no_grad():
        logits = head(image_embeds, cls).numpy()
    ref = REF["logits"]
    print("logits", logits.shape, "corr vs ref:", corr(logits, ref),
          "maxabsdiff:", float(np.abs(logits - ref).max()))
    # tag agreement (the ship metric): compare thresholded predicted-tag sets
    thr = np.load(os.path.join(WORK, "ref", "class_threshold.npy"))
    def tags(l):
        return set(np.where(1/(1+np.exp(-l[0])) > thr)[0].tolist())
    ta, tb = tags(logits), tags(ref)
    print(f"tags mine={len(ta)} ref={len(tb)} inter={len(ta & tb)} "
          f"jaccard={len(ta&tb)/max(1,len(ta|tb)):.4f}")

    if mode == "full":
        # true end-to-end: re-authored Swin -> reweight -> taghead, on
        # the real image
        from build_swin import patch_gpu_clean, SwinEncoder
        patch_gpu_clean(model)
        enc = SwinEncoder(model).eval()
        image = torch.from_numpy(REF["image"])
        with torch.no_grad():
            ie = enc(image)  # my image_embeds
            logits_full = head(ie, ie[:, 0, :]).numpy()
        print("[full pipeline] logits corr vs ref:", corr(logits_full, ref),
              "maxabsdiff:", float(np.abs(logits_full - ref).max()))
        thr = np.load(os.path.join(WORK, "ref", "class_threshold.npy"))
        tl = __import__("json").load(
            open(os.path.join(WORK, "ref", "tag_list.json")))
        def tagset(l):
            return set(np.where(1/(1+np.exp(-l[0])) > thr)[0].tolist())
        ta, tb = tagset(logits_full), tagset(ref)
        print(f"[full] tags mine={len(ta)} ref={len(tb)} "
              f"inter={len(ta&tb)} jaccard={len(ta&tb)/max(1,len(ta|tb)):.4f}")
        print("  missed:", [tl[i] for i in sorted(tb - ta)])
        print("  extra :", [tl[i] for i in sorted(ta - tb)])
        return

    if mode == "convert":
        import litert_torch
        sys.path.insert(
            0, os.path.expanduser("~/Downloads/meeting/cmgan-work"))
        from build_cmgan import opcheck, to_fp16
        # Graph B = tagging head alone (GPU candidate):
        # queries[1,C,768] + image_embeds[1,145,512] -> logits
        with torch.no_grad():
            queries = head.reweight(cls)  # [1,C,768]
        tagb = TagHeadGPU(model).eval()
        with torch.no_grad():
            lb = tagb(queries, image_embeds).numpy()
        print("[graphB] logits corr vs ref:", corr(lb, ref))
        fp32 = os.path.join(OUT, "ram_taghead.tflite")
        litert_torch.convert(tagb, (queries, image_embeds)).export(fp32)
        _, cleanB = opcheck(fp32, "taghead fp32")
        if cleanB:
            b16 = to_fp16(fp32, os.path.join(OUT, "ram_taghead_fp16.tflite"))
            opcheck(b16, "taghead fp16")
            # device fixtures for the tag-head GPU probe:
            # th_queries [1,4585,768], th_iemb [1,145,512],
            # th_ref [1,4585]
            queries.numpy().astype("<f4").tofile(
                os.path.join(OUT, "th_queries.bin"))
            image_embeds.numpy().astype("<f4").tofile(
                os.path.join(OUT, "th_iemb.bin"))
            ref.astype("<f4").tofile(os.path.join(OUT, "th_ref.bin"))
            print("wrote taghead fp16 + device fixtures "
                  "(th_queries/th_iemb/th_ref)")

    if mode == "reweight":
        import litert_torch
        sys.path.insert(
            0, os.path.expanduser("~/Downloads/meeting/cmgan-work"))
        from build_cmgan import opcheck, to_fp16
        rw = Reweight(model).eval()
        with torch.no_grad():
            q = rw(cls).numpy()
        print("[reweight] queries", q.shape, "corr vs head.reweight:",
              corr(q, head.reweight(cls).detach().numpy()))
        fp32 = os.path.join(OUT, "ram_reweight.tflite")
        print("converting reweight (bakes 479MB label_embed) ...")
        litert_torch.convert(rw, (cls,)).export(fp32)
        # CPU graph — GPU-cleanliness not required
        opcheck(fp32, "reweight fp32")
        to_fp16(fp32, os.path.join(OUT, "ram_reweight_fp16.tflite"))

if __name__ == "__main__":
    main()
