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

# RF-DETR Nano 2-graph split for CompiledModel GPU (final ship path).
#   Graph A (GPU) = backbone(encoder+projector) + flatten + proposal-grid + enc heads
#                   -> enc_class[1,576,91], enc_coord[1,576,4], memory[1,576,256]   (NO topk/gather/mask)
#   host (Kotlin)  = scores=max(enc_class,-1) -> topk-300 -> gather enc_coord -> refpoint_embed_ts[1,300,4]
#   Graph B (GPU) = two-stage reparam combine + decoder(query_feat, memory, refpoints) + bbox/class heads
#                   -> boxes[1,300,4], logits[1,300,91]
# Importing build_rfdetr_full applies every GPU patch (grid_sample tent-matmul, sine-embed bake,
# MSDeformAttn <=4D forward, windowed-DINOv2 backbone, tanh-GELU, torch.export friction fixes).
import sys
import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import build_rfdetr_full as FULL          # <- applies ALL patches on import (also pulls in build_rfdetr_bb)
B = FULL.B
R = int(os.environ.get("RF_RES", "384"))
HERE = os.path.dirname(os.path.abspath(__file__))
BANNED = B.BANNED
NQ = 300
NCLS = 91
HID = 256
GH = GW = R // 16                          # 24x24 single deformable level

# ---- fp16-SAFE projector LayerNorm (NAFNet SafeLayerNorm) -------------------------------------------
# The MultiScaleProjector fuses 4 backbone feature maps; its ConvX outputs reach |x|~440 pre-norm, so the
# channel reduction sum_c (x-mu)^2 (256 * 440^2) OVERFLOWS fp16 (max 65504) on the Mali delegate (which
# computes in fp16 regardless of model dtype) -> garbage (device memory corr collapsed 1.0 -> 0.58).
# Fix: do the reductions in a down-scaled domain (x/S) so the sums stay < 65504, then rescale var and
# (x-mu) back -> numerically EXACT (LayerNorm is scale-invariant). The native channels-first LN here
# permutes->F.layer_norm->permute; we replace it with the same math, channel-dim reduce, down-scaled.
from rfdetr.models.backbone import projector as _PROJ
_LN_S = 128.0
def _safe_ln_proj_forward(self, x):
    xs = x * (1.0 / _LN_S)                              # down-scale before any reduction
    mu = xs.mean(1, keepdim=True)                       # sum_c (x/S): fp16-safe
    d = xs - mu                                         # (x - mu)/S
    var = (d * d).mean(1, keepdim=True) * (_LN_S * _LN_S)   # sum_c (x/S)^2 safe, then rescale to true var
    d = d * _LN_S                                       # back to (x - mu)
    y = d * torch.rsqrt(var + self.eps)                    # exact original LayerNorm
    return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]
_PROJ.LayerNorm.forward = _safe_ln_proj_forward

# ---- fp16-SAFE nn.LayerNorm (channels-last, ADAPTIVE scale) ----------------------------------------
# The first decoder layer's self-attention output reaches |x|~1068 (FFN linear2 ~58); the residual into
# norm1/norm3 then overflows fp16 (256 * 1068^2 >> 65504) on Mali -> garbage. But a FIXED large down-scale
# squashes the small-magnitude norms (the final `norm` sees ~8 -> precision loss, logits corr 0.88->0.32).
# Fix: pick S per row from the input's own max so sum_c (x/S)^2 ~ C*64 < 65504 -> S = max(1, amax/8).
# For small |x| (amax<8) S=1 (native, no precision loss); for the 1068 spike S~134 (safe). S cancels
# algebraically -> EXACT (desktop corr stays 1.0). amax/MAX/clamp are all GPU-clean ops.
def _safe_ln_forward(self, x):
    amax = x.abs().amax(-1, keepdim=True)
    S = (amax * (1.0 / 8.0)).clamp(min=1.0)
    xs = x / S
    mu = xs.mean(-1, keepdim=True)
    d = xs - mu
    var = (d * d).mean(-1, keepdim=True) * (S * S)
    d = d * S
    y = d * torch.rsqrt(var + self.eps)
    if self.elementwise_affine:
        y = y * self.weight + self.bias
    return y
nn.LayerNorm.forward = _safe_ln_forward


def build_net():
    """Build RFDETR-Nano with export wiring + GELU/pos bakes; return the raw LWDETR net."""
    wrap = FULL.build()                    # net.export(), interpolate_pos bake, GELU->tanh across net.modules()
    return wrap.net, wrap


def _gen_proposals_nomask(memory, h, w):
    """gen_encoder_output_proposals for bbox_reparam (unsigmoid=False), single level, NO validity masking.
    For a 24x24 grid every proposal lies in (0.01,0.99) -> output_proposals_valid is all-True, so the
    original masked_fill is a no-op; we drop it (and the SELECT_V2/GATHER it would emit). Grid is a
    constant (image-independent), so the host needs no validity mask either."""
    bs = memory.shape[0]
    gy, gx = torch.meshgrid(torch.linspace(0, h - 1, h, dtype=torch.float32),
                            torch.linspace(0, w - 1, w, dtype=torch.float32), indexing="ij")
    grid = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], -1)            # h,w,2
    scale = torch.tensor([w, h], dtype=torch.float32).reshape(1, 1, 1, 2)
    cxcy = (grid.unsqueeze(0) + 0.5) / scale                              # 1,h,w,2
    wh = torch.ones_like(cxcy) * 0.05                                     # lvl=0 -> 0.05*2^0
    proposals = torch.cat((cxcy, wh), -1).reshape(bs, -1, 4)             # bs,hw,4
    return memory, proposals.to(memory.dtype)


class GraphA(nn.Module):
    """image -> enc_class, enc_coord, memory."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.tr = net.transformer
        self.bb0 = net.backbone[0]            # Backbone (encoder=DinoV2 + projector=MultiScaleProjector)
        # Bake the fixed proposal grid as a constant buffer (image-independent) so meshgrid never
        # enters the traced graph (it would lower to BROADCAST_TO, a Mali blocker).
        _, prop = _gen_proposals_nomask(torch.zeros(1, GH * GW, 1), GH, GW)
        self.register_buffer("proposals", prop, persistent=False)          # [1, 576, 4]

    def forward(self, x):
        raw = self.bb0.encoder(x)             # DinoV2.forward -> list of feature maps [B,384,24,24]
        feats = self.bb0.projector(raw)       # MultiScaleProjector -> [ [B,256,24,24] ]
        src = feats[0]
        memory = src.flatten(2).transpose(1, 2)                          # bs,hw,c
        output_memory = memory
        output_proposals = self.proposals                                  # baked constant
        om = self.tr.enc_output_norm[0](self.tr.enc_output[0](output_memory))  # enc_output + norm
        enc_class = self.tr.enc_out_class_embed[0](om)                      # bs,hw,91
        delta = self.tr.enc_out_bbox_embed[0](om)                          # bs,hw,4 (reparam delta)
        cxcy = delta[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2]
        wh = delta[..., 2:].exp() * output_proposals[..., 2:]
        enc_coord = torch.cat([cxcy, wh], -1)                          # bs,hw,4
        return enc_class, enc_coord, memory


class GraphB(nn.Module):
    """(memory, refpoint_embed_ts) -> boxes, logits."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.tr = net.transformer
        self.register_buffer("ss", torch.tensor([[GH, GW]], dtype=torch.long), persistent=False)
        self.register_buffer("lsi", torch.tensor([0], dtype=torch.long), persistent=False)

    def forward(self, memory, refpoint_embed_ts):
        bs = memory.shape[0]
        qf = self.net.query_feat.weight[:NQ]                               # baked learned query feats
        rp = self.net.refpoint_embed.weight[:NQ]                           # baked learned refpoints (zero-init)
        tgt = qf.unsqueeze(0).expand(bs, -1, -1).contiguous()
        rp = rp.unsqueeze(0).expand(bs, -1, -1).contiguous()
        ts = refpoint_embed_ts
        cxcy = rp[..., :2] * ts[..., 2:] + ts[..., :2]                 # two-stage reparam combine
        wh = rp[..., 2:].exp() * ts[..., 2:]
        refpoint = torch.cat([cxcy, wh], -1)
        dec = self.tr.decoder(tgt, memory, memory_key_padding_mask=None, pos=None,
                           refpoints_unsigmoid=refpoint, level_start_index=self.lsi,
                           spatial_shapes=self.ss, valid_ratios=None)
        hs, ref = dec[:2]                                              # export: hs[bs,300,256], ref=refpoint
        delta = self.net.bbox_embed(hs)
        bcxcy = delta[..., :2] * ref[..., 2:] + ref[..., :2]
        bwh = delta[..., 2:].exp() * ref[..., 2:]
        boxes = torch.cat([bcxcy, bwh], -1)
        logits = self.net.class_embed(hs)
        return boxes, logits


def host_select(enc_class, enc_coord):
    """topk-300 by max-class-score + gather coords (the host glue, also done in Kotlin)."""
    scores = enc_class.amax(-1)                                         # bs,hw
    idx = scores.topk(NQ, dim=1).indices                               # bs,300
    ts = torch.gather(enc_coord, 1, idx.unsqueeze(-1).expand(-1, -1, 4))
    return ts, idx


def opcheck(p, l):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=p)
    it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    print(f"[{l}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB",
          "GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it, (not bad and not over)


def run_tflite(path, inputs):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    ins = it.get_input_details()
    # match inputs to interpreter slots by shape
    for d in ins:
        shp = tuple(d["shape"])
        matched = None
        for arr in inputs:
            if tuple(arr.shape) == shp:
                matched = arr
                break
        assert matched is not None, f"no input for slot {d['name']} shape {shp}"
        it.set_tensor(d["index"], matched.astype(d["dtype"]))
    it.invoke()
    return [it.get_tensor(od["index"]) for od in it.get_output_details()]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    net, wrap = build_net()
    x = torch.randn(1, 3, R, R) * 0.5
    ga, gb = GraphA(net).eval(), GraphB(net).eval()

    with torch.no_grad():
        ref_coord, ref_cls = wrap(x)                                   # torch full reference
        ec, eco, mem = ga(x)
        ts, idx = host_select(ec, eco)
        sb, sl = gb(mem, ts)
    print("ref     :", tuple(ref_coord.shape), tuple(ref_cls.shape))
    print("graphA  :", tuple(ec.shape), tuple(eco.shape), tuple(mem.shape))
    print("graphB  :", tuple(sb.shape), tuple(sl.shape))
    cb = np.corrcoef(sb.numpy().ravel(), ref_coord.numpy().ravel())[0, 1]
    cl = np.corrcoef(sl.numpy().ravel(), ref_cls.numpy().ravel())[0, 1]
    print(f"split-vs-torch corr  boxes {cb:.6f}  logits {cl:.6f}")
    if cmd == "forward":
        sys.exit()

    import litert_torch
    # ---- Graph A ----
    pa = f"{HERE}/rfA.tflite"
    litert_torch.convert(ga, (x,)).export(pa)
    ita, okA = opcheck(pa, "rfA")
    oa = run_tflite(pa, [x.numpy().astype(np.float32)])
    # order outputs by shape
    def by_shape(outs, shp):
        return next(o for o in outs if tuple(o.shape) == shp)
    ta_ec = by_shape(oa, tuple(ec.shape))
    ta_eco = by_shape(oa, tuple(eco.shape))
    ta_mem = by_shape(oa, tuple(mem.shape))
    print(f"  A enc_class corr {np.corrcoef(ta_ec.ravel(), ec.numpy().ravel())[0,1]:.6f}")
    print(f"  A enc_coord corr {np.corrcoef(ta_eco.ravel(), eco.numpy().ravel())[0,1]:.6f}")
    print(f"  A memory    corr {np.corrcoef(ta_mem.ravel(), mem.numpy().ravel())[0,1]:.6f}")

    # ---- Graph B ---- (feed torch memory + torch ts so B is tested in isolation)
    pb = f"{HERE}/rfB.tflite"
    litert_torch.convert(gb, (mem, ts)).export(pb)
    itb, okB = opcheck(pb, "rfB")
    ob = run_tflite(pb, [mem.numpy().astype(np.float32), ts.numpy().astype(np.float32)])
    tb_boxes = by_shape(ob, tuple(sb.shape))
    tb_logits = by_shape(ob, tuple(sl.shape))
    print(f"  B boxes  corr {np.corrcoef(tb_boxes.ravel(), sb.numpy().ravel())[0,1]:.6f}")
    print(f"  B logits corr {np.corrcoef(tb_logits.ravel(), sl.numpy().ravel())[0,1]:.6f}")

    # ---- end-to-end tflite chain vs torch reference ----
    ts_t, _ = host_select(torch.from_numpy(ta_ec), torch.from_numpy(ta_eco))
    ob2 = run_tflite(pb, [ta_mem.astype(np.float32), ts_t.numpy().astype(np.float32)])
    e2e_b = by_shape(ob2, tuple(sb.shape))
    e2e_l = by_shape(ob2, tuple(sl.shape))
    print(f"  E2E boxes  corr {np.corrcoef(e2e_b.ravel(), ref_coord.numpy().ravel())[0,1]:.6f}")
    print(f"  E2E logits corr {np.corrcoef(e2e_l.ravel(), ref_cls.numpy().ravel())[0,1]:.6f}")

    if cmd == "fp16" or cmd == "all":
        B.to_fp16(pa, f"{HERE}/rfA_fp16.tflite")
        opcheck(f"{HERE}/rfA_fp16.tflite", "rfA_fp16")
        B.to_fp16(pb, f"{HERE}/rfB_fp16.tflite")
        opcheck(f"{HERE}/rfB_fp16.tflite", "rfB_fp16")
        # device-probe artifacts
        x.numpy().astype(np.float32).tofile(f"{HERE}/rfA_in.bin")
        mem.numpy().astype(np.float32).tofile(f"{HERE}/rfB_in_memory.bin")
        ts.numpy().astype(np.float32).tofile(f"{HERE}/rfB_in_ts.bin")
        np.save(f"{HERE}/rfA_ec.npy", ec.numpy())
        np.save(f"{HERE}/rfA_eco.npy", eco.numpy())
        np.save(f"{HERE}/rfB_boxes.npy", sb.numpy())
        np.save(f"{HERE}/rfB_logits.npy", sl.numpy())
        print("saved fp16 + device-probe artifacts")
