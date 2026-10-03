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

"""RF-DETR-Seg Nano (roboflow/rf-detr 1.9.3) -> LiteRT CompiledModel GPU.

Two-graph split of a DETR-family instance segmenter (DINOv2-S/12 backbone,
deformable-attention decoder, ConvNeXt-style mask head - 33.6M params,
Apache-2.0):

  Graph A (GPU)  image[1,3,312,312] + clspos[1,1,384] + pospatch[1,676,384]
                 -> enc_class[1,676,91], enc_delta[1,676,4], memory*2
  host           /2 -> proposal-grid combine -> topk-100 by max class score
                 -> gather -> reparam with refpoint_embed -> refpoint[1,100,4]
  Graph B (GPU)  (memory, refpoint, query_feat[1,100,256])
                 -> boxes[1,100,4], logits[1,100,91], masks[1,100,78,78]

The cls+pos embedding, patch pos-embed and decoder query embedding are fed as
runtime INPUTS (emitted as .bin artifacts): the GPU delegate silently
mis-executes compute chains that consume large baked-constant tensors (fp32
and fp16 flatbuffers return identical wrong numbers), so no big constant may
live inside the graph. See README.md for the full re-authoring table.

Setup (see README.md for details):
  pip install litert-torch ai-edge-litert ai-edge-quantizer
  pip install torch torchvision supervision pyDeprecate transformers
  pip install rfdetr==1.9.3 --no-deps   # weights auto-download on first run

Run:
  python build_rf_detr_seg_nano.py {forward,fp16,all}
"""

import collections
import math
import os
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tfm_compat  # noqa: F401  transformers 4.57 <-> 5.x shims (no-op on 5+)

# Untraceable private op -> constant (the resolution is fixed at trace time).
torch._shape_as_tensor = lambda t: torch.tensor(
    list(t.shape), dtype=torch.long
)
torch._assert = lambda *a, **k: None

R = int(os.environ.get("RF_RES", "312"))
NQ, NCLS, HID = 100, 91, 256
GH = GW = R // 12  # 26x26 single deformable level
MH = MW = R // 4  # 78x78 mask grid (mask_downsample_ratio=4)
BANNED = {
    "GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT",
    "SELECT_V2", "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST",
    "EMBEDDING_LOOKUP", "RFFT2D", "FFT", "STFT", "COMPLEX", "CUMSUM",
    "MIRROR_PAD",
}

# ---- backbone SDPA -> manual rank-4 matmuls ---------------------------------
# (the delegate mis-executes rank-3 batched matmuls, and SDPA has no lowering)
from rfdetr.models.backbone import dinov2_with_windowed_attn as D


def _sdpa_manual(self, hidden_states, output_attentions=False):
  """Manual rank-4 SDPA replacement (same weights, exact).

  Args:
    hidden_states: [B, N, C] token sequence.
    output_attentions: Unused, kept for signature compatibility.

  Returns:
    (context [B, N, C], None) like the transformers original.
  """
  q = self.transpose_for_scores(self.query(hidden_states))
  k = self.transpose_for_scores(self.key(hidden_states))
  v = self.transpose_for_scores(self.value(hidden_states))
  scale = 1.0 / (q.shape[-1] ** 0.5)
  s = torch.matmul(q, k.transpose(-1, -2)) * scale
  a = torch.softmax(s, dim=-1)
  c = torch.matmul(a, v).permute(0, 2, 1, 3).contiguous()
  c = c.view(c.size()[:-2] + (self.all_head_size,))
  return c, None


D.Dinov2WithRegistersSdpaSelfAttention.forward = _sdpa_manual


# ---- SafeLayerNorm v2 for nn.LayerNorm (channels-last) ----------------------
# Adaptive per-row down-scale, and NEVER reconstruct the large variance: the
# scale cancels in y = d/sqrt(var), so every intermediate stays
# O(1)..O(amax) -- fp16-safe at any magnitude.
def _safe_ln_forward(self, x):
  """fp16-safe LayerNorm forward (see the comment above).

  Args:
    x: [..., C] channels-last input.

  Returns:
    The normalized tensor, numerically equal to nn.LayerNorm.
  """
  amax = x.abs().amax(-1, keepdim=True)
  s = (amax * (1.0 / 8.0)).clamp(min=1.0)
  xs = x / s
  mu = xs.mean(-1, keepdim=True)
  d = xs - mu
  var = (d * d).mean(-1, keepdim=True)
  y = d * torch.rsqrt(var + self.eps)
  if self.elementwise_affine:
    y = y * self.weight + self.bias
  return y


nn.LayerNorm.forward = _safe_ln_forward

# ---- projector channels-first LayerNorm -> same v2 math, via a 3D detour ----
# litert-torch's NHWC layout pass has no rewriter for amax on layout-tracked
# 4D tensors, so drop to [B, HW, C] (3D leaves the conv-layout domain) before
# the adaptive reduction; the math is unchanged.
from rfdetr.models.backbone import projector as _PROJ


def _safe_ln_proj_forward(s, x):
  """fp16-safe channels-first LayerNorm via a 3D detour.

  Args:
    s: The projector LayerNorm module (weight/bias/eps).
    x: [B, C, H, W] input.

  Returns:
    The normalized [B, C, H, W] tensor.
  """
  b, c, h, w = x.shape
  t = x.reshape(b, c, h * w).transpose(1, 2)  # [B, HW, C]
  amax = t.abs().amax(-1, keepdim=True)
  scale = (amax * (1.0 / 8.0)).clamp(min=1.0)
  xs = t / scale
  mu = xs.mean(-1, keepdim=True)
  d = xs - mu
  var = (d * d).mean(-1, keepdim=True)
  y = d * torch.rsqrt(var + s.eps) * s.weight + s.bias
  return y.transpose(1, 2).reshape(b, c, h, w)


_PROJ.LayerNorm.forward = _safe_ln_proj_forward


# ---- grid_sample -> GATHER/CAST-free bilinear (tent weights + matmul) -------
# Numerically exact (MAE ~1e-8) including zeros-padding out-of-bounds.
def _gs(input, grid, mode="bilinear", padding_mode="zeros",
        align_corners=None):
  """grid_sample as tent-weight matmuls (exact, GATHER/CAST-free).

  Args:
    input: [N, C, H, W] value map.
    grid: [N, Hg, Wg, 2] normalized sampling grid.
    mode: Only "bilinear" is supported (the rfdetr use).
    padding_mode: Only "zeros" is supported. OOB taps get weight 0.
    align_corners: Same semantics as F.grid_sample.

  Returns:
    [N, C, Hg, Wg] sampled map.
  """
  n, c, h, w = input.shape
  hg, wg = grid.shape[1], grid.shape[2]
  if align_corners:
    ix = (grid[..., 0] + 1) * (w - 1) / 2
    iy = (grid[..., 1] + 1) * (h - 1) / 2
  else:
    ix = (grid[..., 0] + 1) * w / 2 - 0.5
    iy = (grid[..., 1] + 1) * h / 2 - 0.5
  ix = ix.reshape(n, hg * wg, 1)
  iy = iy.reshape(n, hg * wg, 1)
  xs = torch.arange(w, dtype=input.dtype).reshape(1, 1, w)
  ys = torch.arange(h, dtype=input.dtype).reshape(1, 1, h)
  wx = torch.relu(1 - (ix - xs).abs())
  wy = torch.relu(1 - (iy - ys).abs())
  wm = (wy.unsqueeze(-1) * wx.unsqueeze(-2)).reshape(n, 1, hg * wg, h * w)
  # Rank-4 BMM: the delegate mis-executes rank-3 batched matmuls.
  out = torch.matmul(input.reshape(n, 1, c, h * w), wm.transpose(-1, -2))
  return out.reshape(n, c, hg, wg)


F.grid_sample = _gs
from rfdetr.utilities import tensors as _T

_T._bilinear_grid_sample = (
    lambda input, grid, padding_mode="zeros", align_corners=False:
    _gs(input, grid, padding_mode=padding_mode, align_corners=align_corners)
)
from rfdetr.models.ops.functions import ms_deform_attn_func as _MSF

_MSF._bilinear_grid_sample = _T._bilinear_grid_sample

# ---- MSDeformAttn re-authored <=4D (n_levels=1), tent-matmul sampler --------
import rfdetr.models.ops.modules.ms_deform_attn as _MSMOD


def _msda_forward(self, query, reference_points, input_flatten,
                  input_spatial_shapes, input_level_start_index,
                  input_padding_mask=None, input_spatial_shapes_hw=None,
                  **kw):
  """MSDeformAttn.forward re-authored <=4D for the single-level case.

  Args:
    query: [bs, len_q, d_model] queries.
    reference_points: [bs, len_q, 1, 2 or 4] normalized references.
    input_flatten: [bs, H*W, d_model] flattened value map.
    input_spatial_shapes: Unused tensor form of (H, W).
    input_level_start_index: Unused (single level).
    input_padding_mask: Optional [bs, H*W] padding mask.
    input_spatial_shapes_hw: Python list [(H, W)] (traceable form).
    **kw: Ignored extras from the caller.

  Returns:
    [bs, len_q, d_model] attended output.
  """
  bs = query.shape[0]
  len_q = query.shape[1]
  nh, npnt, dm = self.n_heads, self.n_points, self.d_model
  hd = dm // nh
  h, w = input_spatial_shapes_hw[0] if input_spatial_shapes_hw else (GH, GW)
  value = self.value_proj(input_flatten)  # [bs, HW, dm]
  if input_padding_mask is not None:
    value = value.masked_fill(input_padding_mask[..., None], 0.0)
  so = self.sampling_offsets(query).view(bs, len_q, nh, npnt * 2)
  so = so.permute(0, 2, 1, 3).reshape(bs * nh, len_q, npnt, 2)
  aw = self.attention_weights(query).view(bs, len_q, nh, npnt)
  aw = torch.softmax(aw, -1).permute(0, 2, 1, 3)
  aw = aw.reshape(bs * nh, 1, len_q, npnt)
  ref = reference_points[:, :, 0, :]  # squeeze n_levels=1
  rxy = ref[..., :2].unsqueeze(1).repeat(1, nh, 1, 1)
  rxy = rxy.reshape(bs * nh, len_q, 1, 2)
  if ref.shape[-1] == 4:
    rwh = ref[..., 2:].unsqueeze(1).repeat(1, nh, 1, 1)
    rwh = rwh.reshape(bs * nh, len_q, 1, 2)
    loc = rxy + so / npnt * rwh * 0.5
  else:
    norm = torch.tensor([w, h], dtype=value.dtype).reshape(1, 1, 1, 2)
    loc = rxy + so / norm
  val = value.transpose(1, 2).reshape(bs * nh, hd, h, w)
  sampled = _gs(val, 2 * loc - 1, padding_mode="zeros", align_corners=False)
  out = (sampled * aw).sum(-1).reshape(bs, dm, len_q).transpose(1, 2)
  return self.output_proj(out)


_MSMOD.MSDeformAttn.forward = _msda_forward

# ---- sine pos-embed: bake dim_t (no POW/FLOOR_DIV), interleave via reshape --
import rfdetr.models.transformer as _TR

_DIMT = {}


def _gen_sine(pos_tensor, dim=128):
  """Sine position embedding without POW/FLOOR_DIV/strided GATHER_ND.

  Args:
    pos_tensor: [B, N, 2 or 4] normalized box coordinates.
    dim: Embedding dim per coordinate pair.

  Returns:
    [B, N, 2*dim or 4*dim] sine embedding, matching the original.
  """
  scale = 2 * math.pi
  if dim not in _DIMT:
    dt = torch.arange(dim, dtype=torch.float32)
    _DIMT[dim] = (10000.0 ** (2 * (dt // 2) / dim)).detach()
  dim_t = _DIMT[dim]

  def il(emb):
    p = emb[:, :, None] * scale / dim_t
    pr = p.reshape(p.shape[0], p.shape[1], dim // 2, 2)
    return torch.stack((pr[..., 0].sin(), pr[..., 1].cos()), -1).flatten(2)

  pos_x = il(pos_tensor[:, :, 0])
  pos_y = il(pos_tensor[:, :, 1])
  if pos_tensor.size(-1) == 2:
    return torch.cat((pos_y, pos_x), dim=2)
  return torch.cat(
      (pos_y, pos_x, il(pos_tensor[:, :, 2]), il(pos_tensor[:, :, 3])), dim=2
  )


_TR.gen_sineembed_for_position = _gen_sine

# ---- seg-head DepthwiseConvBlock: plain F.conv2d + the LN 3D detour ---------
# Its permute(0,2,3,1) LayerNorm sits on a layout-tracked 4D tensor (the amax
# rewriter gap again), so the norm runs on a [B, HW, C] view instead.
from rfdetr.models.heads import segmentation as _SEG


def _dwblock_forward(self, x):
  """DepthwiseConvBlock forward with plain conv2d and the LN 3D detour.

  Args:
    x: [B, C, H, W] input feature map.

  Returns:
    [B, C, H, W] output (residual included).
  """
  inp = x
  x = F.conv2d(x, self.dwconv.weight, self.dwconv.bias, self.dwconv.stride,
               self.dwconv.padding, self.dwconv.dilation, self.dwconv.groups)
  b, c, h, w = x.shape
  x = x.reshape(b, c, h * w).transpose(1, 2)  # [B, HW, C] 3D
  x = self.norm(x)
  x = self.pwconv1(x)
  x = self.act(x)
  if self.gamma is not None:
    x = self.gamma * x
  x = x.transpose(1, 2).reshape(b, c, h, w)
  return x + inp


_SEG.DepthwiseConvBlock.forward = _dwblock_forward


class TanhGelu(nn.Module):
  """tanh-approximation GELU (ERF has no GPU lowering; corr 0.99999)."""

  def forward(self, x):
    return 0.5 * x * (
        1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x))
    )


class ManualMha(nn.Module):
  """Rank-4 replacement for nn.MultiheadAttention (batch_first self-attn).

  torch MHA lowers to rank-3 BMMs, which the delegate silently mis-executes;
  same weights, manual head-split matmuls, exact.
  """

  def __init__(self, mha):
    super().__init__()
    self.h = mha.num_heads
    self.e = mha.embed_dim
    self.in_w = nn.Parameter(mha.in_proj_weight.data.clone())
    self.in_b = nn.Parameter(mha.in_proj_bias.data.clone())
    self.out = mha.out_proj

  def forward(self, q, k, v, attn_mask=None, key_padding_mask=None,
              need_weights=False):
    e, h = self.e, self.h
    hd = e // h
    b, n, _ = q.shape
    qp = F.linear(q, self.in_w[:e], self.in_b[:e])
    qp = qp.reshape(b, n, h, hd).permute(0, 2, 1, 3)
    kp = F.linear(k, self.in_w[e:2 * e], self.in_b[e:2 * e])
    kp = kp.reshape(b, -1, h, hd).permute(0, 2, 1, 3)
    vp = F.linear(v, self.in_w[2 * e:], self.in_b[2 * e:])
    vp = vp.reshape(b, -1, h, hd).permute(0, 2, 1, 3)
    a = torch.softmax(
        torch.matmul(qp, kp.transpose(-1, -2)) * (hd ** -0.5), -1
    )
    c = torch.matmul(a, vp).permute(0, 2, 1, 3).reshape(b, n, e)
    return self.out(c), None


def build_net():
  """Loads RFDETRSegNano and applies the weight-level GPU patches."""
  from rfdetr import RFDETRSegNano
  m = RFDETRSegNano()
  net = m.model.model.eval()
  net.export()
  bb = None
  for mod in net.modules():
    if (hasattr(mod, "encoder") and hasattr(mod.encoder, "layer")
        and hasattr(mod, "embeddings")):
      bb = mod
      break
  emb = bb.embeddings
  c = emb.cls_token.shape[-1]
  nreg = getattr(emb.config, "num_register_tokens", 0)
  n = GH * GW + 1
  _pos = emb.interpolate_pos_encoding(torch.zeros(1, n, c), R, R).detach()
  emb.interpolate_pos_encoding = lambda e, h, w, _p=_pos: _p
  for mod in net.modules():
    for cn, ch in list(mod.named_children()):
      if isinstance(ch, nn.GELU) or type(ch).__name__ in (
          "GELUActivation", "QuickGELUActivation"):
        setattr(mod, cn, TanhGelu())
  # LayerScale bake (exact): the delegate mis-executes `h + lambda*f(h)` (a
  # broadcast-const MUL feeding a residual ADD; device corr 0.62 vs 0.9999
  # for `h + f(h)`). Fold lambda into the preceding Linear and drop the op.
  nbaked = 0
  for mod in net.modules():
    if (hasattr(mod, "layer_scale1") and hasattr(mod, "attention")
        and hasattr(mod, "mlp")):
      lam1 = mod.layer_scale1.lambda1.data
      mod.attention.output.dense.weight.data.mul_(lam1[:, None])
      mod.attention.output.dense.bias.data.mul_(lam1)
      mod.layer_scale1 = nn.Identity()
      lam2 = mod.layer_scale2.lambda1.data
      mod.mlp.fc2.weight.data.mul_(lam2[:, None])
      mod.mlp.fc2.bias.data.mul_(lam2)
      mod.layer_scale2 = nn.Identity()
      nbaked += 1
  print(f"  LayerScale baked into dense/fc2 weights in {nbaked} layers")
  nmha = 0
  for mod in net.modules():
    for cn, ch in list(mod.named_children()):
      if isinstance(ch, nn.MultiheadAttention):
        setattr(mod, cn, ManualMha(ch))
        nmha += 1
  print(f"  nn.MultiheadAttention -> rank-4 ManualMha in {nmha} layers")
  nwin = emb.config.num_windows
  params = sum(p.numel() for p in net.parameters()) / 1e6
  print(f"  RF-DETR-Seg-Nano: {params:.1f}M params; num_windows={nwin} "
        f"registers={nreg} dec_registers={net.transformer.num_registers} "
        f"seg_blocks={len(net.segmentation_head.blocks)}")
  assert nwin == 1, "SegNano expected num_windows=1 (all-global attention)"
  assert net.transformer.num_registers == 0, "decoder registers not handled"
  assert nreg == 0, "backbone register tokens not handled by this split"
  clspos = (emb.cls_token + _pos[:, :1]).detach().clone()
  pospatch = _pos[:, 1:].detach().clone()
  return net, bb, clspos, pospatch


def build_proposals(h, w):
  """gen_encoder_output_proposals for bbox_reparam, single level, no mask.

  For a 26x26 grid every proposal lies in (0.019, 0.981), so the
  (0.01, 0.99) validity mask is all-True and the masked_fill is a no-op;
  the grid is image-independent -> a host-side constant.

  Args:
    h: Grid height.
    w: Grid width.

  Returns:
    [1, h*w, 4] proposal boxes (cxcywh, wh = 0.05).
  """
  gy, gx = torch.meshgrid(
      torch.linspace(0, h - 1, h, dtype=torch.float32),
      torch.linspace(0, w - 1, w, dtype=torch.float32), indexing="ij")
  grid = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], -1)
  scale = torch.tensor([w, h], dtype=torch.float32).reshape(1, 1, 1, 2)
  cxcy = (grid.unsqueeze(0) + 0.5) / scale
  wh = torch.ones_like(cxcy) * 0.05
  return torch.cat((cxcy, wh), -1).reshape(1, -1, 4)


class GraphA(nn.Module):
  """(image, clspos, pospatch) -> enc_class, enc_delta, memory*2.

  The position embedding (and the cls token, pre-added into clspos) is
  HOST-FED: the delegate mis-executes compute chains that consume large
  baked constants, so no big constant may live inside the graph.
  """

  def __init__(self, net, inner):
    super().__init__()
    self.tr = net.transformer
    self.bb0 = net.backbone[0]
    self.inner = inner  # WindowedDinov2WithRegistersBackbone

  def forward(self, x, clspos, pospatch):
    emb = self.inner.embeddings
    pe = emb.patch_embeddings(x)  # [1, 676, 384]
    h = torch.cat((clspos, pe + pospatch), dim=1)
    hs_all = [h]
    for l in self.inner.encoder.layer:
      o = l(h)
      h = o[0] if isinstance(o, tuple) else o
      hs_all.append(h)
    feats = []
    for stage, hstate in zip(self.inner.stage_names, hs_all):
      if stage in self.inner.out_features:
        if self.inner.config.apply_layernorm:
          hstate = self.inner.layernorm(hstate)
        hstate = hstate[:, 1:]  # strip cls (num_register_tokens=0)
        hstate = hstate.reshape(1, GH, GW, -1).permute(0, 3, 1, 2)
        feats.append(hstate.contiguous())
    src = self.bb0.projector(feats)[0]  # MultiScaleProjector(P4)
    memory = src.flatten(2).transpose(1, 2)  # [1, 676, 256]
    om = self.tr.enc_output_norm[0](self.tr.enc_output[0](memory))
    enc_class = self.tr.enc_out_class_embed[0](om)
    delta = self.tr.enc_out_bbox_embed[0](om)  # token-pointwise -> pre-topk
    # Raw delta out; the proposal-grid combine moves to the HOST (the same
    # baked-const rule). memory is consumed (enc_output) AND a graph output
    # -> the delegate zeroes the output copy ([1,N,C] output-and-consumed
    # bug). x2 forces a separate buffer (exact in fp16); the host halves.
    return enc_class, delta, memory * 2.0


class GraphB(nn.Module):
  """(memory, refpoint, query_feat) -> boxes, logits, masks.

  query_feat is a HOST-fed input (not a baked const) and the two-stage
  reparam combine runs on the host (baked refpoint_embed otherwise).
  """

  def __init__(self, net):
    super().__init__()
    self.net = net
    self.tr = net.transformer
    self.seg = net.segmentation_head
    self.register_buffer(
        "ss", torch.tensor([[GH, GW]], dtype=torch.long), persistent=False)
    self.register_buffer(
        "lsi", torch.tensor([0], dtype=torch.long), persistent=False)

  def forward(self, memory, refpoint, query_feat):
    tgt = query_feat
    dec = self.tr.decoder(
        tgt, memory, memory_key_padding_mask=None, pos=None,
        refpoints_unsigmoid=refpoint, level_start_index=self.lsi,
        spatial_shapes=self.ss, spatial_shapes_hw=[(GH, GW)],
        valid_ratios=None)
    hs, ref = dec[:2]  # export mode: hs[1, 100, 256]
    delta = self.net.bbox_embed(hs)
    bcxcy = delta[..., :2] * ref[..., 2:] + ref[..., :2]
    bwh = delta[..., 2:].exp() * ref[..., 2:]
    boxes = torch.cat([bcxcy, bwh], -1)
    logits = self.net.class_embed(hs)
    # Seg mask branch: rebuild the projector map from memory (memory IS
    # srcs[0] flattened), then a rank-4 matmul instead of the einsum.
    src = memory.transpose(1, 2).reshape(1, HID, GH, GW)
    sf = F.interpolate(src, size=(MH, MW), mode="bilinear",
                       align_corners=False)
    for blk in self.seg.blocks:
      sf = blk(sf)
    sfp = self.seg.spatial_features_proj(sf)  # [1, 256, 78, 78]
    q = self.seg.query_features_proj(self.seg.query_features_block(hs))
    m4 = torch.matmul(q.unsqueeze(1), sfp.reshape(1, 1, HID, MH * MW))
    masks = (m4 + self.seg.bias).reshape(1, NQ, MH, MW)
    return boxes, logits, masks


def host_select(enc_class, enc_delta, proposals, rp):
  """Host glue: proposal combine + topk-100 + gather + two-stage reparam.

  All per-token elementwise math plus a topk -- exact off-GPU. The Kotlin
  runner ports this function 1:1.

  Args:
    enc_class: [1, 676, 91] encoder class logits.
    enc_delta: [1, 676, 4] encoder box deltas.
    proposals: [1, 676, 4] proposal grid from build_proposals().
    rp: [1, 100, 4] learned refpoint_embed rows.

  Returns:
    (refpoint [1,100,4], ts [1,100,4], topk indices).
  """
  cxcy = enc_delta[..., :2] * proposals[..., 2:] + proposals[..., :2]
  wh = enc_delta[..., 2:].exp() * proposals[..., 2:]
  enc_coord = torch.cat([cxcy, wh], -1)
  scores = enc_class.amax(-1)
  idx = scores.topk(NQ, dim=1).indices
  ts = torch.gather(enc_coord, 1, idx.unsqueeze(-1).expand(-1, -1, 4))
  rcxcy = rp[..., :2] * ts[..., 2:] + ts[..., :2]
  rwh = rp[..., 2:].exp() * ts[..., 2:]
  refpoint = torch.cat([rcxcy, rwh], -1)
  return refpoint, ts, idx


def opcheck(path, label):
  """Static GPU-compat scan: reads the op set from the .tflite flatbuffer.

  Args:
    path: Path to the .tflite flatbuffer.
    label: Log label prefix.

  Returns:
    True when no banned op or >4D tensor is found.
  """
  from ai_edge_litert import schema_py_generated as schema
  with open(path, "rb") as f:
    model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
  names = {v: k for k, v in vars(schema.BuiltinOperator).items()
           if not k.startswith("_")}
  ops = collections.Counter()
  over = 0
  for g in model.subgraphs:
    for op in g.operators:
      c = model.operatorCodes[op.opcodeIndex]
      code = max(c.builtinCode, c.deprecatedBuiltinCode)
      key = c.customCode.decode() if c.customCode else names.get(
          code, str(code))
      ops[key] += 1
    over += sum(1 for t in g.tensors
                if t.shape is not None and len(t.shape) > 4)
  bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
  print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
  print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} "
        f"size {os.path.getsize(path)/1e6:.1f}MB",
        "GPU-CLEAN" if not bad and not over else "BLOCKERS")
  return not bad and not over


def run_tflite(path, inputs):
  """Single inference through the LiteRT CompiledModel API.

  Buffers are matched by float element count (every tensor in both graphs
  has a distinct size), the same capacity-matching the Android runner uses,
  so the signature slot order does not matter.

  Args:
    path: Path to the .tflite file.
    inputs: List of input arrays (any order).

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
    assert n in by_elems, f"no input with {n} elements for slot {i}"
    ins[i].write(np.ascontiguousarray(by_elems[n].ravel(), np.float32))
  model.run_by_index(0, ins, outs)
  res = {}
  for j in range(len(outs)):
    n = model.get_output_buffer_requirements(j)["buffer_size"] // 4
    res[n] = outs[j].read(n, np.float32)
  return res


def to_fp16(fp32, fp16):
  """Casts weights to fp16 with ai-edge-quantizer (compute stays float).

  Args:
    fp32: Path to the fp32 .tflite input.
    fp16: Path for the fp16 .tflite output.

  Returns:
    The fp16 path.
  """
  from ai_edge_quantizer import quantizer
  from ai_edge_quantizer import recipe_manager
  from ai_edge_quantizer.recipe import AlgorithmName
  from ai_edge_quantizer.recipe import qtyping
  rm = recipe_manager.RecipeManager()
  rm.add_quantization_config(
      regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      op_config=qtyping.OpQuantizationConfig(
          weight_tensor_config=qtyping.TensorQuantizationConfig(
              num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
          compute_precision=qtyping.ComputePrecision.FLOAT),
      algorithm_key=AlgorithmName.FLOAT_CASTING)
  if os.path.exists(fp16):
    os.remove(fp16)
  q = quantizer.Quantizer(float_model=fp32)
  q.load_quantization_recipe(rm.get_quantization_recipe())
  q.quantize().export_model(fp16)
  return fp16


def stats(name, a, b):
  """Prints correlation and max abs diff between two arrays.

  Args:
    name: Log label.
    a: First array-like.
    b: Second array-like.

  Returns:
    (corr, max_abs_diff).
  """
  a = np.asarray(a).ravel()
  b = np.asarray(b).ravel()
  c = np.corrcoef(a, b)[0, 1]
  md = np.abs(a - b).max()
  print(f"  {name}: corr {c:.6f}  max|diff| {md:.4e}")
  return c, md


if __name__ == "__main__":
  cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
  net, inner, clspos, pospatch = build_net()
  x = torch.randn(1, 3, R, R) * 0.5
  ga, gb = GraphA(net, inner).eval(), GraphB(net).eval()
  proposals = build_proposals(GH, GW)
  rp = net.refpoint_embed.weight[:NQ].unsqueeze(0).detach().clone()
  qf0 = net.query_feat.weight[:NQ].unsqueeze(0).detach().clone()

  with torch.no_grad():
    ref_coord, ref_cls, ref_masks = net.forward_export(x)  # torch reference
    ec, ed, mem2 = ga(x, clspos, pospatch)
    mem = mem2 * 0.5  # invert the x2 output trick
    refpoint, ts, idx = host_select(ec, ed, proposals, rp)
    sb, sl, sm = gb(mem, refpoint, qf0)
  print("ref     :", tuple(ref_coord.shape), tuple(ref_cls.shape),
        tuple(ref_masks.shape))
  print("graphA  :", tuple(ec.shape), tuple(ed.shape), tuple(mem.shape))
  print("graphB  :", tuple(sb.shape), tuple(sl.shape), tuple(sm.shape))
  stats("split-vs-torch boxes ", sb, ref_coord)
  stats("split-vs-torch logits", sl, ref_cls)
  stats("split-vs-torch masks ", sm, ref_masks)
  if cmd == "forward":
    sys.stdout.flush()
    sys.exit()

  import litert_torch
  pa = f"{HERE}/rfdetrseg_graphA.tflite"
  litert_torch.convert(ga, (x, clspos, pospatch)).export(pa)
  ok_a = opcheck(pa, "graphA")
  oa = run_tflite(pa, [x.numpy(), clspos.numpy(), pospatch.numpy()])
  ta_ec = oa[ec.numel()].reshape(ec.shape)
  ta_ed = oa[ed.numel()].reshape(ed.shape)
  ta_mem = oa[mem.numel()].reshape(mem.shape) * 0.5
  stats("A enc_class", ta_ec, ec.numpy())
  stats("A enc_delta", ta_ed, ed.numpy())
  stats("A memory   ", ta_mem, mem.numpy())

  pb = f"{HERE}/rfdetrseg_graphB.tflite"
  litert_torch.convert(gb, (mem, refpoint, qf0)).export(pb)
  ok_b = opcheck(pb, "graphB")
  ob = run_tflite(pb, [mem.numpy(), refpoint.numpy(), qf0.numpy()])
  stats("B boxes ", ob[sb.numel()].reshape(sb.shape), sb.numpy())
  stats("B logits", ob[sl.numel()].reshape(sl.shape), sl.numpy())
  stats("B masks ", ob[sm.numel()].reshape(sm.shape), sm.numpy())

  ref_t, _, _ = host_select(
      torch.from_numpy(np.asarray(ta_ec)),
      torch.from_numpy(np.asarray(ta_ed)), proposals, rp)
  ob2 = run_tflite(pb, [np.asarray(ta_mem, np.float32),
                        ref_t.numpy(), qf0.numpy()])
  stats("E2E boxes ", ob2[sb.numel()].reshape(sb.shape), ref_coord.numpy())
  stats("E2E logits", ob2[sl.numel()].reshape(sl.shape), ref_cls.numpy())
  stats("E2E masks ", ob2[sm.numel()].reshape(sm.shape), ref_masks.numpy())

  if cmd in ("fp16", "all"):
    to_fp16(pa, f"{HERE}/rfdetrseg_graphA_fp16.tflite")
    opcheck(f"{HERE}/rfdetrseg_graphA_fp16.tflite", "graphA_fp16")
    to_fp16(pb, f"{HERE}/rfdetrseg_graphB_fp16.tflite")
    opcheck(f"{HERE}/rfdetrseg_graphB_fp16.tflite", "graphB_fp16")
    # Host-side runtime artifacts (published next to the .tflite files):
    # raw little-endian float32, shapes as commented.
    clspos.numpy().astype(np.float32).tofile(f"{HERE}/clspos.bin")
    pospatch.numpy().astype(np.float32).tofile(f"{HERE}/pospatch.bin")
    qf0.numpy().astype(np.float32).tofile(f"{HERE}/query_feat.bin")
    rp.numpy().astype(np.float32).tofile(f"{HERE}/refpoint_embed.bin")
    print("saved fp16 graphs + host-constant .bin artifacts")
  sys.stdout.flush()
