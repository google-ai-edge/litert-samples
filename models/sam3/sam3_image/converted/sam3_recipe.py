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

"""SAM 3 image side -> CompiledModel GPU: model loading + re-authoring recipe.

SAM 3 (facebook/sam3) detects and segments every instance matching a text
phrase. This module builds the detector from the released checkpoint and
re-authors it so that all three graphs lower to LiteRT and run on the GPU
delegates (ML Drift on Android, Metal on Apple). Every rewrite here is
numerically exact with respect to the stock modules -- weight permutations,
reshapes and algebraically identical rewrites only. `convert_sam3.py` asserts
that with a per-graph parity check before it exports anything.

Three graphs, one flat float input and one flat float output each:

  vision  image (1,3,1008,1008), (x/255-0.5)/0.5, NCHW RGB
          -> [fpn288 | fpn144 | fpn72]  (ViT-L/14 trunk + detection neck)
  text    token embeddings (1,32,1024), looked up on the host
          -> text memory (1, 32*256)
  head    [fpn288 | fpn144 | fpn72 | text_mem(32*256) | text_pad(32)]
          -> [logits(200) | boxes(200*4) | presence(1) | masks(200*288*288)]

Why each rewrite exists (all of these are silent failures without it):

  * >4-D ViT attention. The stock trunk builds 5-D/6-D/8-D tensors (fused qkv
    head split, window partition, tiled absolute position, interleaved RoPE).
    The GPU delegates reject >4-D tensors, and the converter itself mis-lowers
    them: a raw export of the trunk reaches only corr 0.607 against PyTorch
    while the re-authored one reaches corr 1.00000. See `patch_vit_4d`.
  * Interleaved RoPE. The (2p, 2p+1) pair rotation is folded into the rows of
    the qkv projection so the rotation becomes a contiguous half-split. The
    same permutation is applied to q and k, and q.k is invariant under a shared
    permutation of the channel axis, so this is exact.
  * Window partition. reshape -> transpose -> reshape -> transpose. The cheaper
    order-swap trick used for window attention elsewhere is NOT valid here,
    because RoPE is position dependent inside the window; the extra transpose
    restores the exact layout.
  * fp16 accumulators. This ViT-L residual stream reaches |x| ~ 300 and the
    CLIP-L text residual stream reaches |x| ~ 1.2e3, so sum((x-mean)^2)
    overflows fp16 (65504) inside the delegate even for an fp32 graph.
    `SafeLayerNorm` scales before squaring; LayerNorm is scale invariant and
    eps is scaled to match, so it stays exact.
  * Rank-3 activations that fan out. A rank-3 [1, N, C] tensor feeding several
    consumers is mis-executed by the GPU delegates: on device this made all 200
    decoder logits identical while boxes and masks stayed correct. The decoder
    is therefore re-implemented batch-first and returns rank-4, and the text
    encoder runs on a rank-4 (1,1,L,E) activation.
  * Masked softmax. A SOFTMAX fed directly by an elementwise op with a
    broadcast constant (an additive causal or padding mask) is mis-executed by
    the Metal delegate. The form used here computes the max over valid keys
    only and multiplies inside the exponent; see `masked_softmax`.
  * Banned ops. ConvTranspose2d -> zero-stuff + Conv2d, clamp(0,1) ->
    relu(x) - relu(x-1), strided slices in the sine embedding -> stack of the
    64 distinct frequencies, GroupNorm -> 4-D manual with hierarchical means.

The video tracker built on the same trunk is out of scope here; it lives in the
model zoo linked from the README.
"""

import math
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT = 32
IMAGE_SIZE = 1008
FPN_SIZES = ((288, 288), (144, 144), (72, 72))
NUM_QUERIES = 200
MASK_SIZE = 288

# Additive mask value. exp(NEG - max) is 0 in both fp32 and fp16 and never
# produces a NaN row, unlike -inf.
NEG = -1e4

# Ops the GPU delegates reject. `convert_sam3.py` fails the build if any of
# these survives in an exported graph.
GPU_BANNED_OPS = frozenset({
    "GATHER_ND", "GATHER", "SELECT", "SELECT_V2", "NOT_EQUAL", "EQUAL",
    "GREATER", "LESS", "TOPK_V2", "CAST", "PACK", "SPLIT", "TRANSPOSE_CONV",
    "ARG_MAX", "ARG_MIN", "WHERE", "CUMSUM", "SCATTER_ND", "UNIQUE",
    "NON_MAX_SUPPRESSION_V4", "NON_MAX_SUPPRESSION_V5",
})

# fp16 weights on the matmul-class ops only. Quantizing every constant makes
# small buffers (position encodings, query embeddings, masks) fp16 too, and a
# DEQUANTIZE feeding an elementwise op -- or one shared by several consumers --
# is refused by the Metal delegate at compile time.
FP16_RECIPE = [
    {
        "regex": ".*",
        "operation": op,
        "algorithm_key": "float_casting",
        "op_config": {
            "weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"}
        },
    }
    for op in ("FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D")
] + [
    {
        # Two tiny MLPs are applied once per decoder layer, so a single fp16
        # weight would feed six FULLY_CONNECTED consumers through one
        # DEQUANTIZE. Keep them fp32.
        "regex": ".*boxRPB_embed.*",
        "operation": "FULLY_CONNECTED",
        "algorithm_key": "no_quantize",
        "op_config": {},
    }
]


def _install_triton_stub():
  """Stubs sam3.model.edt, which imports triton unconditionally.

  The EDT kernel it guards is CUDA-only and unused by inference on CPU, so a
  stub module lets the conversion environment stay CPU-only.
  """
  try:
    import triton  # noqa: F401  (only probing availability)
    return
  except ImportError:
    pass
  stub = types.ModuleType("sam3.model.edt")
  stub.edt_triton = None
  sys.modules["sam3.model.edt"] = stub


_install_triton_stub()


class SafeLayerNorm(nn.Module):
  """LayerNorm that cannot overflow an fp16 variance accumulator.

  The delegates accumulate reductions in fp16 even when the graph is fp32, so
  sum((x - mean)^2) overflows once |x| grows past a few hundred. Scaling the
  input before squaring keeps the accumulator small; LayerNorm is invariant to
  that scale as long as eps is scaled by the same factor squared, so the
  rewrite is exact.

  Args:
    layer_norm: the nn.LayerNorm module to replace.
    scale: pre-multiplier applied to the input. 1/16 for the head, 1/32 for
      the ViT trunk (|x| ~ 300), 1/64 for the CLIP text tower (|x| ~ 1.2e3).
  """

  def __init__(self, layer_norm, scale):
    super().__init__()
    self.scale = scale
    self.eps = layer_norm.eps * scale * scale
    self.register_buffer("w", layer_norm.weight.detach().clone())
    self.register_buffer("b", layer_norm.bias.detach().clone())

  def forward(self, x):
    """Normalizes over the last dimension."""
    xs = x * self.scale
    mu = xs.mean(-1, keepdim=True)
    d = xs - mu
    var = (d * d).mean(-1, keepdim=True)
    return d * torch.rsqrt(var + self.eps) * self.w + self.b


# ----------------------------------------------------------------- softmax


def masked_softmax(scores, keep):
  """Softmax over the last dim restricted to the keys where keep == 1.

  The obvious form, softmax(scores + (1 - keep) * NEG), is mis-executed by the
  Metal delegate whenever an elementwise op with a broadcast constant feeds
  SOFTMAX directly. Taking the max over the valid keys only and multiplying
  inside the exponent avoids that pattern and is fp16-safe as well: the
  exponent is always <= 0 and masked keys contribute exp(0) which the trailing
  multiply removes. Rewriting the same math with clamp, min or relu brings the
  mis-execution back, so keep this form.

  Args:
    scores: (..., Lq, Lk) attention logits.
    keep: float mask broadcastable to scores, 1.0 for a valid key.

  Returns:
    Attention weights with the same shape as scores.
  """
  m = (scores * keep + (1.0 - keep) * NEG).max(dim=-1, keepdim=True).values
  e = torch.exp((scores - m) * keep) * keep
  return e / e.sum(dim=-1, keepdim=True)


def biased_softmax(scores, bias):
  """Softmax of scores + bias without an ADD that has two consumers.

  Used for the log-scale relative position bias of the decoder's image
  cross-attention. Written so that the sum feeding the reduce is consumed
  once, and so that the exponent stays <= 1.

  Args:
    scores: (..., Lq, Lk) attention logits.
    bias: additive bias broadcastable to scores.

  Returns:
    Attention weights with the same shape as scores.
  """
  m = (scores + bias).max(dim=-1, keepdim=True).values
  e = torch.exp((scores - m) + bias)
  return e / e.sum(dim=-1, keepdim=True)


# --------------------------------------------------------------- ViT trunk


def _deinterleave_rows(weight, num_heads, head_dim, bias=None):
  """Reorders output rows so even channels precede odd ones inside each head.

  Folding this permutation into the q and k rows of the fused qkv projection
  turns the interleaved (2p, 2p+1) RoPE rotation into a contiguous half-split.
  Applying the same permutation to q and k leaves q.k unchanged.

  Args:
    weight: (C_out, ...) projection weight slice for q or k.
    num_heads: number of attention heads.
    head_dim: channels per head.
    bias: matching bias slice, or None.

  Returns:
    A (permuted_weight, permuted_bias) tuple; permuted_bias is None if bias
    was None.
  """
  channels = num_heads * head_dim
  index = (
      torch.arange(channels)
      .view(num_heads, head_dim // 2, 2)
      .permute(0, 2, 1)
      .reshape(channels)
  )
  return weight[index], (bias[index] if bias is not None else None)


def window_partition_4d(x, window):
  """Partitions (B, H, W, C) into windows using <=4-D ops only.

  Args:
    x: (B, H, W, C) feature map; H and W must be multiples of window.
    window: window side length.

  Returns:
    A ((B*nH*nW, window, window, C) tensor, (H, W)) tuple.
  """
  b, h, w, c = x.shape
  n_h, n_w = h // window, w // window
  x = x.reshape(b * n_h, window, w, c)
  x = x.transpose(1, 2)
  x = x.reshape(b * n_h * n_w, window, window, c)
  return x.transpose(1, 2).contiguous(), (h, w)


def window_unpartition_4d(windows, window, pad_hw, hw):
  """Inverse of `window_partition_4d`.

  Args:
    windows: (B*nH*nW, window, window, C) tensor.
    window: window side length.
    pad_hw: padded size, unused here because 1008/14 = 72 is a multiple of 24.
    hw: the (H, W) returned by `window_partition_4d`.

  Returns:
    The (B, H, W, C) feature map.
  """
  del pad_hw  # 72 x 72 with window 24 never pads.
  h, w = hw
  c = windows.shape[-1]
  n_h, n_w = h // window, w // window
  b = windows.shape[0] // (n_h * n_w)
  x = windows.transpose(1, 2)
  x = x.reshape(b * n_h, w, window, c)
  x = x.transpose(1, 2)
  return x.reshape(b, h, w, c)


def _vit_attention_4d(self, x):
  """ViT attention with <=4-D tensors, half-split RoPE and explicit softmax.

  Args:
    x: Activation shaped (batch, height, width, channels).

  Returns:
    The attention output, shaped like `x`.
  """
  b, h, w, c = x.shape
  length = h * w
  num_heads = self.num_heads
  head_dim = c // num_heads
  qkv = self.qkv(x.reshape(b, length, c))
  q = qkv[:, :, :c].reshape(b, length, num_heads, head_dim).transpose(1, 2)
  k = (
      qkv[:, :, c:2 * c]
      .reshape(b, length, num_heads, head_dim)
      .transpose(1, 2)
  )
  v = qkv[:, :, 2 * c:].reshape(b, length, num_heads, head_dim).transpose(1, 2)
  if self.use_rope:
    cos, sin = self.rope_cos, self.rope_sin
    half = head_dim // 2
    q1, q2 = q[..., :half], q[..., half:]
    q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], -1)
    k1, k2 = k[..., :half], k[..., half:]
    k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], -1)
  q = q * (1.0 / math.sqrt(head_dim))
  k_t = k.transpose(-2, -1)
  chunks = getattr(self, "q_chunks", 1)
  if chunks > 1 and length % chunks == 0:
    # Exact query chunking. The global blocks attend 5184 queries over 5184
    # keys; materializing that score tensor at once needs 860 MB in fp16,
    # so bound it by slicing the query axis.
    chunk = length // chunks
    outs = []
    for i in range(chunks):
      scores = torch.matmul(q[:, :, i * chunk:(i + 1) * chunk], k_t)
      outs.append(torch.matmul(torch.softmax(scores, dim=-1), v))
    o = torch.cat(outs, 2)
  else:
    o = torch.matmul(torch.softmax(torch.matmul(q, k_t), dim=-1), v)
  o = o.transpose(1, 2).reshape(b, h, w, c)
  return self.proj(o)


def _vit_forward_4d(self, x):
  """Trunk forward with the tiled absolute position baked into a buffer.

  Args:
    x: Image batch shaped (batch, 3, 1008, 1008).

  Returns:
    A single-element list holding the NCHW trunk features.
  """
  x = self.patch_embed(x)
  x = x + self.pos_baked
  x = self.ln_pre(x)
  for block in self.blocks:
    x = block(x)
  x = self.ln_post(x)
  return [x.permute(0, 3, 1, 2)]


@torch.no_grad()
def patch_vit_4d(vit, safe_ln=True, global_chunks=9):
  """Re-authors the ViT-L/14 trunk in <=4-D, in place.

  Bakes the tiled absolute position embedding, folds the interleaved RoPE into
  the qkv rows, replaces the head split and window partition with 4-D reshapes
  and transposes, and swaps SDPA for an explicit matmul softmax.

  Args:
    vit: the loaded `sam3.model.vitdet` trunk.
    safe_ln: replace ln_pre, norm1 and norm2 with `SafeLayerNorm`.
    global_chunks: query chunks used by the four global-attention blocks.
      Exact for any divisor of 5184; 9 keeps the score tensor near 96 MB.

  Returns:
    The same trunk, patched.
  """
  from sam3.model import vitdet

  assert not vit.retain_cls_token and vit.pos_embed is not None
  img_hw = (
      vit.blocks[-1].attn.input_size
      if vit.blocks[-1].window_size == 0
      else None
  )
  assert img_hw is not None, "expects the last block to be global (72x72)"
  pos = vitdet.get_abs_pos(
      vit.pos_embed,
      vit.pretrain_use_cls_token,
      tuple(img_hw),
      vit.retain_cls_token,
      tiling=vit.tile_abs_pos,
  )
  vit.register_buffer("pos_baked", pos.detach().clone(), persistent=False)
  if safe_ln:
    vit.ln_pre = SafeLayerNorm(vit.ln_pre, 1.0 / 32)
    for block in vit.blocks:
      block.norm1 = SafeLayerNorm(block.norm1, 1.0 / 32)
      block.norm2 = SafeLayerNorm(block.norm2, 1.0 / 32)
  for block in vit.blocks:
    attn = block.attn
    assert attn.use_rope and attn.use_rope_real
    assert not attn.use_rel_pos and not attn.use_ve_rope
    num_heads = attn.num_heads
    channels = attn.qkv.in_features
    head_dim = channels // num_heads
    w = attn.qkv.weight.data
    b = attn.qkv.bias.data if attn.qkv.bias is not None else None
    w_q, b_q = _deinterleave_rows(
        w[:channels], num_heads, head_dim,
        b[:channels] if b is not None else None)
    w_k, b_k = _deinterleave_rows(
        w[channels:2 * channels], num_heads, head_dim,
        b[channels:2 * channels] if b is not None else None)
    attn.qkv.weight.data = torch.cat([w_q, w_k, w[2 * channels:]], 0)
    if b is not None:
      attn.qkv.bias.data = torch.cat([b_q, b_k, b[2 * channels:]], 0)
    attn.register_buffer(
        "rope_cos",
        attn.freqs_cis_real.detach().clone()[None, None],
        persistent=False,
    )
    attn.register_buffer(
        "rope_sin",
        attn.freqs_cis_imag.detach().clone()[None, None],
        persistent=False,
    )
    attn.q_chunks = global_chunks if block.window_size == 0 else 1
    attn.forward = types.MethodType(_vit_attention_4d, attn)
  vitdet.window_partition = window_partition_4d
  vitdet.window_unpartition = window_unpartition_4d
  vit.forward = types.MethodType(_vit_forward_4d, vit)
  return vit


class ZeroStuffConvT(nn.Module):
  """ConvTranspose2d(k=2, s=2) as nearest upsample + stride mask + Conv2d.

  TRANSPOSE_CONV is rejected by the ML Drift delegate. Zero-stuffing the input
  and convolving with the flipped, transposed kernel is the same operation.

  Args:
    conv_t: the nn.ConvTranspose2d module to replace.
    in_hw: input side length (square feature maps only).
  """

  def __init__(self, conv_t, in_hw):
    super().__init__()
    self.stride = conv_t.stride[0]
    self.kernel = conv_t.kernel_size[0]
    self.out_hw = in_hw * self.stride
    self.register_buffer(
        "weight",
        conv_t.weight.detach().flip(2, 3).transpose(0, 1).contiguous(),
    )
    self.bias = conv_t.bias
    mask = torch.zeros(1, 1, self.out_hw, self.out_hw)
    mask[:, :, ::self.stride, ::self.stride] = 1.0
    self.register_buffer("mask", mask)

  def forward(self, x):
    """Upsamples by `stride` and convolves."""
    up = F.interpolate(x, size=(self.out_hw, self.out_hw), mode="nearest")
    out = F.conv2d(
        up * self.mask, self.weight, self.bias, padding=self.kernel - 1
    )
    return out[:, :, :self.out_hw, :self.out_hw]


def patch_neck(neck):
  """Replaces every ConvTranspose2d of the detection neck, in place.

  Args:
    neck: `det.backbone.vision_backbone`, whose three neck heads share one
      layout.
  """
  heads = (
      neck.convs,
      getattr(neck, "interactive_convs", None),
      getattr(neck, "propagation_convs", None),
  )
  for convs in heads:
    if convs is None:
      continue
    conv4x, conv2x = convs[0], convs[1]
    if isinstance(conv4x.dconv_2x2_0, ZeroStuffConvT):
      continue
    conv4x.dconv_2x2_0 = ZeroStuffConvT(conv4x.dconv_2x2_0, 72)
    conv4x.dconv_2x2_1 = ZeroStuffConvT(conv4x.dconv_2x2_1, 144)
    conv2x.dconv_2x2 = ZeroStuffConvT(conv2x.dconv_2x2, 72)


# --------------------------------------------------------------- attention


def _mask_to_additive(mask, dtype):
  """Converts a bool mask (True = masked) to a float additive mask.

  Args:
    mask: Bool or float mask, or None.
    dtype: Floating dtype the returned mask is cast to.

  Returns:
    The additive float mask, or None when `mask` is None.

  Raises:
    TypeError: If `mask` is neither bool nor floating point.
  """
  if mask is None:
    return None
  if mask.dtype == torch.bool:
    return mask.to(dtype) * NEG
  if mask.is_floating_point():
    return mask
  raise TypeError(f"unsupported mask dtype {mask.dtype}")


def mha_core_bf(module, query, key, value, key_padding_mask=None,
                attn_mask=None, attn_bias=None):
  """Batch-first 4-D attention using a MultiheadAttention module's weights.

  Works for both sam3's own MultiheadAttention and torch's
  nn.MultiheadAttention; they share the attribute layout. Patching only
  the former silently leaves torch's instances (the decoder self-attention
  and text cross-attention) running SDPA, which lowers to >4-D tensors.

  Args:
    module: the attention module supplying weights and head count.
    query: (B, Lq, E) queries.
    key: (B, Lk, E) keys.
    value: (B, Lk, E) values.
    key_padding_mask: (B, Lk) bool or float indicator, 1.0 marks padding.
    attn_mask: bool mask or float additive bias, (Lq, Lk), (B*H, Lq, Lk) or
      (B, H, Lq, Lk).
    attn_bias: extra additive bias broadcastable to (B, H, Lq, Lk).

  Returns:
    (B, Lq, E) attention output.
  """
  b, len_q, embed = query.shape
  len_k = key.shape[1]
  num_heads = module.num_heads
  head_dim = embed // num_heads
  if module._qkv_same_embed_dim:
    w = module.in_proj_weight
    bias = module.in_proj_bias
    q = F.linear(query, w[:embed], None if bias is None else bias[:embed])
    k = F.linear(
        key,
        w[embed:2 * embed],
        None if bias is None else bias[embed:2 * embed],
    )
    v = F.linear(
        value, w[2 * embed:], None if bias is None else bias[2 * embed:]
    )
  else:
    bias = module.in_proj_bias
    q = F.linear(
        query, module.q_proj_weight,
        None if bias is None else bias[:embed])
    k = F.linear(
        key, module.k_proj_weight,
        None if bias is None else bias[embed:2 * embed])
    v = F.linear(
        value, module.v_proj_weight,
        None if bias is None else bias[2 * embed:])
  q = q.reshape(b, len_q, num_heads, head_dim).transpose(1, 2)
  k = k.reshape(b, len_k, num_heads, head_dim).transpose(1, 2)
  v = v.reshape(b, len_k, num_heads, head_dim).transpose(1, 2)
  scores = torch.matmul(
      q * (1.0 / math.sqrt(head_dim)), k.transpose(-2, -1)
  )
  bias_term = None
  if attn_mask is not None:
    additive = _mask_to_additive(attn_mask, scores.dtype)
    if additive.dim() == 2:
      additive = additive.reshape(1, 1, len_q, len_k)
    elif additive.dim() == 3:
      if additive.shape[0] == b * num_heads:
        additive = additive.reshape(-1, num_heads, len_q, len_k)
      else:
        additive = additive.reshape(1, 1, len_q, len_k)
    bias_term = additive
  if attn_bias is not None:
    bias_term = attn_bias if bias_term is None else bias_term + attn_bias
  if len_k == 1:
    # A single key makes softmax identically 1, and the converter would
    # otherwise emit DIV(e, e), which the GPU delegates refuse.
    o = v.expand(b, num_heads, len_q, head_dim)
  else:
    if key_padding_mask is not None:
      assert bias_term is None
      keep = 1.0 - key_padding_mask.to(scores.dtype).reshape(
          b, 1, 1, len_k
      )
      attn = masked_softmax(scores, keep)
    elif bias_term is not None:
      attn = biased_softmax(scores, bias_term)
    else:
      attn = torch.softmax(scores, dim=-1)
    o = torch.matmul(attn, v)
  o = o.transpose(1, 2).reshape(b, len_q, embed)
  return module.out_proj(o)


def mha_forward_4d(self, query, key, value, key_padding_mask=None,
                   need_weights=False, attn_mask=None,
                   average_attn_weights=True, attn_bias=None):
  """Drop-in MultiheadAttention.forward for eval-mode inference.

  Args:
    query: Query activation.
    key: Key activation.
    value: Value activation.
    key_padding_mask: Per-key padding mask, or None.
    need_weights: Ignored; weights are never returned.
    attn_mask: Additive or bool attention mask, or None.
    average_attn_weights: Ignored; weights are never returned.
    attn_bias: Additive bias broadcast over the score matrix, or None.

  Returns:
    An (output, None) tuple; attention weights are never returned.
  """
  del need_weights, average_attn_weights  # Weights are never requested.
  assert not self.training
  seq_first = not self.batch_first
  if seq_first:
    query = query.transpose(0, 1)
    key = key.transpose(0, 1)
    value = value.transpose(0, 1)
  o = mha_core_bf(
      self, query, key, value, key_padding_mask, attn_mask, attn_bias
  )
  if seq_first:
    o = o.transpose(0, 1)
  return o, None


def text_attention_4d(self, q_x, k_x=None, v_x=None, attn_mask=None):
  """Drop-in ResidualAttentionBlock.attention for the CLIP text tower.

  Runs on a rank-4 (1, 1, L, E) activation because rank-3 [1, L, C] tensors
  that fan out to several consumers are mis-computed by the GPU delegates. The
  causal mask is a constant `causal_keep` buffer installed by
  `apply_text_patches`.

  Args:
    q_x: Token activation, rank 3 or rank 4.
    k_x: Ignored; this block is self-attention only.
    v_x: Ignored; this block is self-attention only.
    attn_mask: Ignored; the causal mask is baked in.

  Returns:
    The attention output, rank-matched to `q_x`.
  """
  del k_x, v_x, attn_mask  # Self-attention with a baked causal mask.
  attn = self.attn
  x4 = q_x if q_x.dim() == 4 else q_x.unsqueeze(1)
  b, _, length, embed = x4.shape
  num_heads = attn.num_heads
  head_dim = embed // num_heads
  w, bias = attn.in_proj_weight, attn.in_proj_bias
  q = F.linear(x4, w[:embed], bias[:embed])
  q = q.reshape(b, length, num_heads, head_dim).transpose(1, 2)
  k = F.linear(x4, w[embed:2 * embed], bias[embed:2 * embed])
  k = k.reshape(b, length, num_heads, head_dim).transpose(1, 2)
  v = F.linear(x4, w[2 * embed:], bias[2 * embed:])
  v = v.reshape(b, length, num_heads, head_dim).transpose(1, 2)
  scores = torch.matmul(
      q * (1.0 / math.sqrt(head_dim)), k.transpose(-2, -1)
  )
  attn_w = masked_softmax(scores, self.causal_keep)
  o = torch.matmul(attn_w, v).transpose(1, 2).reshape(b, 1, length, embed)
  o = attn.out_proj(o)
  return o if q_x.dim() == 4 else o.squeeze(1)


# --------------------------------------------------------------- head ops


def inverse_sigmoid_clean(x, eps=1e-3):
  """inverse_sigmoid without RELU_0_TO_1.

  clamp(x, 0, 1) lowers to RELU_0_TO_1, which one shipped runtime version
  mis-executes; relu(x) - relu(x - 1) is the same function.

  Args:
    x: input tensor.
    eps: lower bound applied before the log, as in the stock implementation.

  Returns:
    log(x / (1 - x)) with the stock clamping behaviour.
  """
  x = F.relu(x) - F.relu(x - 1.0)
  x1 = torch.clamp(x, min=eps)
  x2 = torch.clamp(1 - x, min=eps)
  return torch.log(x1 / x2)


class GroupNorm4d(nn.Module):
  """nn.GroupNorm on NCHW with 4-D tensors and hierarchical means.

  A single reduce over the ~2.6 M elements of one group overflows the
  delegate's fp16 accumulator, so the mean is taken per row and then per
  group. Scaling before the square protects the variance the same way
  `SafeLayerNorm` does.

  Args:
    group_norm: the nn.GroupNorm module to replace.
    scale: pre-multiplier applied to the input.
  """

  def __init__(self, group_norm, scale=1.0 / 8):
    super().__init__()
    self.groups = group_norm.num_groups
    self.channels = group_norm.num_channels
    self.eps = group_norm.eps
    self.scale = scale
    self.register_buffer(
        "w", group_norm.weight.detach().clone().view(1, -1, 1, 1)
    )
    self.register_buffer(
        "b", group_norm.bias.detach().clone().view(1, -1, 1, 1)
    )

  def forward(self, x):
    """Normalizes each group of an NCHW tensor."""
    n, c, h, w = x.shape
    xs = (x * self.scale).reshape(n, self.groups, (c // self.groups) * h, w)
    mu = xs.mean(3, keepdim=True).mean(2, keepdim=True)
    d = xs - mu
    var = (d * d).mean(3, keepdim=True).mean(2, keepdim=True)
    eps = self.eps * self.scale * self.scale
    y = (d * torch.rsqrt(var + eps)).reshape(n, c, h, w)
    return y * self.w + self.b


def mask_predictor_forward_4d(self, obj_queries, pixel_embed):
  """MaskPredictor.forward as a plain matmul, rank-4 aware.

  Keeping the decoder output rank-4 is what stops the delegate from
  corrupting the third consumer of the query tensor.

  Args:
    obj_queries: (B, Q, C) or (1, B, Q, C) decoder outputs.
    pixel_embed: (B, C, H, W) pixel decoder output.

  Returns:
    (B, Q, H, W) mask logits.
  """
  assert pixel_embed.dim() == 4
  b, c, h, w = pixel_embed.shape
  q = self.mask_embed(obj_queries)
  if q.dim() == 4:
    q = q.reshape(1, b, -1, c)
    prod = torch.matmul(q, pixel_embed.reshape(1, b, c, h * w))
    return prod.reshape(b, -1, h, w)
  prod = torch.matmul(q, pixel_embed.reshape(b, c, h * w))
  return prod.reshape(b, -1, h, w)


def rpb_matrix_4d(self, reference_boxes, feat_size, batch_first=False):
  """TransformerDecoder._get_rpb_matrix with 4-D broadcasting only.

  Args:
    reference_boxes: (nq, bs, 4) boxes, or (bs, nq, 4) when batch_first.
    feat_size: unused; the graph is fixed to the 72x72 grid of 1008/14.
    batch_first: whether reference_boxes is (bs, nq, 4).

  Returns:
    (bs, num_heads, nq, H*W) relative position bias.
  """
  del feat_size  # Pinned to 72 x 72 by the fixed input size.
  from sam3.model.box_ops import box_cxcywh_to_xyxy

  h = w = 72
  boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
  if not batch_first:
    boxes_xyxy = boxes_xyxy.transpose(0, 1)
  bs, nq, _ = boxes_xyxy.shape
  boxes = boxes_xyxy.reshape(bs * nq, 1, 4)
  ys = torch.cat([boxes[:, :, 1:2], boxes[:, :, 3:4]], -1)
  xs = torch.cat([boxes[:, :, 0:1], boxes[:, :, 2:3]], -1)
  dy = (self.rpb_coords_h - ys).view(bs, nq, h, 2)
  dx = (self.rpb_coords_w - xs).view(bs, nq, w, 2)
  if self.boxRPB in ["log", "both"]:
    dx_log = dx * 8
    dx_log = torch.sign(dx_log) * torch.log2(torch.abs(dx_log) + 1.0) / 3.0
    dy_log = dy * 8
    dy_log = torch.sign(dy_log) * torch.log2(torch.abs(dy_log) + 1.0) / 3.0
    if self.boxRPB == "log":
      dx, dy = dx_log, dy_log
    else:
      dx = torch.cat([dx, dx_log], -1)
      dy = torch.cat([dy, dy_log], -1)
  emb_x = self.boxRPB_embed_x(dx)
  emb_y = self.boxRPB_embed_y(dy)
  num_heads = emb_x.shape[-1]
  emb_y = emb_y.permute(0, 3, 1, 2).reshape(bs * num_heads, nq, h, 1)
  emb_x = emb_x.permute(0, 3, 1, 2).reshape(bs * num_heads, nq, 1, w)
  return (emb_y + emb_x).reshape(bs, num_heads, nq, h * w)


def _sine_dim_t_half(num_feats=128):
  """Builds the 64 distinct sine frequencies as a constant tensor.

  Args:
    num_feats: Positional-encoding width; half of it are distinct frequencies.

  Returns:
    A float32 tensor of `num_feats // 2` frequency divisors.
  """
  d = np.arange(num_feats // 2, dtype=np.float64)
  return torch.tensor((10000.0 ** (2 * d / num_feats)).astype(np.float32))


_SINE_DIM_T_HALF = _sine_dim_t_half()


def gen_sineembed_clean(pos_tensor, num_feats=256):
  """Sine position embedding without strided slices, POW or FLOOR_DIV.

  The stock implementation indexes dim_t with a stride of 2, which lowers to
  GATHER_ND. Because dim_t[2i] == dim_t[2i+1], the interleaved
  [sin(v/dim_t[0]), cos(v/dim_t[1]), ...] sequence is exactly a stack of
  (sin, cos) over the 64 distinct frequencies.

  Args:
    pos_tensor: (nq, bs, 2) or (nq, bs, 4) reference points.
    num_feats: embedding width per coordinate.

  Returns:
    (nq, bs, num_feats) or (nq, bs, 2*num_feats) embedding.
  """
  assert num_feats % 2 == 0
  half = num_feats // 2
  assert _SINE_DIM_T_HALF.numel() == half // 2
  dim_t = _SINE_DIM_T_HALF.to(pos_tensor.device)
  scale = 2 * math.pi

  def embed(value):
    p = value[:, :, None] / dim_t
    return torch.stack((p.sin(), p.cos()), dim=3).flatten(2)

  x = embed(pos_tensor[:, :, 0] * scale)
  y = embed(pos_tensor[:, :, 1] * scale)
  if pos_tensor.size(-1) == 2:
    return torch.cat((y, x), dim=2)
  w = embed(pos_tensor[:, :, 2] * scale)
  h = embed(pos_tensor[:, :, 3] * scale)
  return torch.cat((y, x, w, h), dim=2)


def dot_prod_mean_pool_text(self, prompt, prompt_mask):
  """DotProductScoring.mean_pool_text without boolean indexing.

  Args:
    prompt: (seq, bs, C) prompt tokens.
    prompt_mask: (bs, seq) float or bool mask, 1.0 marks padding.

  Returns:
    (bs, C) mean over the valid tokens.
  """
  del self  # Bound as a method; no state is used.
  is_valid = (1.0 - prompt_mask.to(prompt.dtype)).permute(1, 0)[..., None]
  num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
  return (prompt * is_valid).sum(dim=0) / num_valid


def geometry_encode_bf(geometry_encoder, cls_token, memory_bf, pos_bf):
  """Runs the geometry encoder's cross-attention layers batch-first.

  Args:
    geometry_encoder: `det.geometry_encoder`.
    cls_token: (1, 1, C) class token.
    memory_bf: (1, H*W, C) image features.
    pos_bf: (1, H*W, C) position encoding.

  Returns:
    (1, 1, C) encoded prompt token.
  """
  g = cls_token
  for layer in geometry_encoder.encode:
    normed = layer.norm1(g)
    g = g + mha_core_bf(layer.self_attn, normed, normed, normed)
    normed = layer.norm2(g)
    g = g + mha_core_bf(
        layer.cross_attn_image, normed, memory_bf + pos_bf, memory_bf
    )
    normed = layer.norm3(g)
    g = g + layer.linear2(layer.activation(layer.linear1(normed)))
  return geometry_encoder.encode_norm(g)


def decoder_forward_bf(dec, tgt, memory, pos, ref, text, text_pad):
  """DETR decoder re-implemented batch-first, returning rank-4 queries.

  The stock layout is sequence-first (n, 1, C) with n > 1 in dim 0, which the
  GPU delegates mis-execute for broadcast elementwise ops. Every activation
  here is (1, n, C), and the returned queries are rank-4 so that the score,
  box and mask heads do not fan out from a rank-3 tensor.

  Args:
    dec: `det.transformer.decoder`.
    tgt: (1, nq, C) initial queries.
    memory: (1, H*W, C) encoder output.
    pos: (1, H*W, C) position encoding.
    ref: (1, nq, 4) initial reference boxes in cxcywh.
    text: (1, L+1, C) prompt tokens.
    text_pad: (1, L+1) float padding indicator.

  Returns:
    A (queries (1, 1, nq, C), reference boxes (1, nq, 4),
    presence logit (1, 1)) tuple.
  """
  output = tgt
  presence = dec.presence_token.weight[None]
  normed = output
  ref_in = ref
  for layer in dec.layers:
    sine = gen_sineembed_clean(ref, dec.d_model)
    query_pos = dec.ref_point_head(sine)
    rpb = rpb_matrix_4d(dec, ref, (72, 72), batch_first=True)
    rpb = torch.cat([torch.zeros_like(rpb[:, :, :1, :]), rpb], 2)
    t = torch.cat([presence, output], 1)
    pos_all = torch.cat([torch.zeros_like(presence), query_pos], 1)
    q = t + pos_all
    t = t + mha_core_bf(layer.self_attn, q, q, t)
    t = layer.norm2(t)
    t = layer.catext_norm(
        t
        + mha_core_bf(
            layer.ca_text, t + pos_all, text, text,
            key_padding_mask=text_pad)
    )
    t = layer.norm1(
        t
        + mha_core_bf(
            layer.cross_attn, t + pos_all, memory + pos, memory,
            attn_bias=rpb)
    )
    t = layer.forward_ffn(t)
    presence, output = t[:, :1], t[:, 1:]
    normed = dec.norm(output)
    ref_in = ref
    ref = torch.sigmoid(
        dec.bbox_embed(normed) + inverse_sigmoid_clean(ref)
    )
  presence_logit = dec.presence_token_head(
      dec.presence_token_out_norm(presence)
  ).reshape(1, 1)
  return normed.unsqueeze(0), ref_in, presence_logit


# ------------------------------------------------------------ apply patches


def apply_text_patches(det, safe_ln_scale=1.0 / 64):
  """Re-authors the CLIP text tower in place.

  Args:
    det: the detector.
    safe_ln_scale: `SafeLayerNorm` scale for the text tower.

  Returns:
    The number of residual blocks patched.
  """
  encoder = det.backbone.language_backbone.encoder
  if safe_ln_scale:
    for block in encoder.transformer.resblocks:
      block.ln_1 = SafeLayerNorm(block.ln_1, safe_ln_scale)
      block.ln_2 = SafeLayerNorm(block.ln_2, safe_ln_scale)
    encoder.ln_final = SafeLayerNorm(encoder.ln_final, safe_ln_scale)
  length = encoder.context_length
  causal = encoder.attn_mask[:length, :length].detach().clone()
  keep = (causal == 0).float().view(1, 1, length, length)
  for block in encoder.transformer.resblocks:
    block.register_buffer("causal_keep", keep, persistent=False)
    block.attention = types.MethodType(text_attention_4d, block)
  return len(encoder.transformer.resblocks)


def apply_head_patches(det):
  """Re-authors the fusion encoder, decoder and segmentation head in place.

  Args:
    det: the detector.

  Returns:
    The number of attention modules patched.
  """
  from sam3.model import decoder as decoder_mod
  from sam3.model import maskformer_segmentation as ms
  from sam3.model import model_misc
  from sam3.model import sam3_image
  from sam3.model.model_misc import DotProductScoring
  from sam3.model.model_misc import MultiheadAttention

  patched = 0
  for module in det.modules():
    if isinstance(module, (MultiheadAttention, nn.MultiheadAttention)):
      module.forward = types.MethodType(mha_forward_4d, module)
      patched += 1
  model_misc.inverse_sigmoid = inverse_sigmoid_clean
  decoder_mod.inverse_sigmoid = inverse_sigmoid_clean
  sam3_image.inverse_sigmoid = inverse_sigmoid_clean
  decoder_mod.gen_sineembed_for_position = gen_sineembed_clean
  decoder_mod.TransformerDecoder._get_rpb_matrix = rpb_matrix_4d
  dec = det.transformer.decoder
  coords = (torch.arange(72, dtype=torch.float32) / 72).view(1, 72, 1)
  dec.register_buffer("rpb_coords_h", coords, persistent=False)
  dec.register_buffer("rpb_coords_w", coords.clone(), persistent=False)
  DotProductScoring.mean_pool_text = dot_prod_mean_pool_text
  ms.MaskPredictor.forward = mask_predictor_forward_4d
  pixel_decoder = det.segmentation_head.pixel_decoder
  pixel_decoder.norms = nn.ModuleList(
      [GroupNorm4d(g) for g in pixel_decoder.norms]
  )
  roots = (
      det.transformer,
      det.geometry_encoder,
      det.segmentation_head,
      det.dot_prod_scoring,
  )
  for root in roots:
    for _, module in list(root.named_modules()):
      for child_name, child in list(module.named_children()):
        if type(child) is nn.LayerNorm:
          setattr(module, child_name, SafeLayerNorm(child, 1.0 / 16))
  return patched


# ------------------------------------------------------------ graph wrappers


class VisionFlat(nn.Module):
  """Image -> flat [fpn288 | fpn144 | fpn72] detection features.

  Used for both the stock reference and the exported graph: the neck patch is
  applied in place, so the same wrapper covers both.
  """

  def __init__(self, det):
    super().__init__()
    self.neck = det.backbone.vision_backbone

  def forward(self, x):
    """Runs the trunk and the detection neck head."""
    sam3_out = self.neck(
        x,
        need_sam3_out=True,
        need_interactive_out=False,
        need_propagation_out=False,
    )[0]
    return torch.cat(
        [getattr(f, "tensors", f).flatten(1) for f in sam3_out], 1
    )


class TextFlatStock(nn.Module):
  """Token ids (1, 32) -> [text memory (32*256) | padding mask (32)].

  Reference only. The exported graph takes embeddings instead, because
  EMBEDDING_LOOKUP does not run on the GPU delegates.
  """

  def __init__(self, det):
    super().__init__()
    self.text = det.backbone.language_backbone

  def forward(self, tokens):
    """Encodes token ids into the flat text memory."""
    encoder = self.text.encoder
    x = encoder.token_embedding(tokens)
    x = x + encoder.positional_embedding[:CONTEXT]
    x = encoder.transformer(
        x, attn_mask=encoder.attn_mask[:CONTEXT, :CONTEXT]
    )
    x = encoder.ln_final(x)
    memory = self.text.resizer(x)
    pad = (tokens == 0).to(memory.dtype)
    return torch.cat([memory.flatten(1), pad], 1)


class TextFlat4d(nn.Module):
  """Token embeddings (1, 32, 1024) -> text memory (1, 32*256)."""

  def __init__(self, det):
    super().__init__()
    self.text = det.backbone.language_backbone
    self.length = self.text.encoder.context_length

  def forward(self, emb):
    """Encodes host-looked-up embeddings, staying rank-4 throughout."""
    encoder = self.text.encoder
    x = emb + encoder.positional_embedding[:self.length]
    x = x.unsqueeze(1)
    x = encoder.transformer(x, attn_mask=None)
    x = encoder.ln_final(x)
    return self.text.resizer(x).flatten(1)


class _HeadFlatBase(nn.Module):
  """Shared input unpacking for the head wrappers."""

  def __init__(self, det, sizes=FPN_SIZES):
    super().__init__()
    self.det = det
    self.sizes = list(sizes)
    self.counts = [256 * h * w for h, w in self.sizes]
    self.length = det.backbone.language_backbone.encoder.context_length
    pos_enc = det.backbone.vision_backbone.position_encoding
    self.pos = nn.ParameterList([
        nn.Parameter(
            pos_enc(torch.zeros(1, 256, h, w)).detach(),
            requires_grad=False,
        )
        for h, w in self.sizes
    ])

  def _split(self, flat):
    """Splits the flat input into feature maps, text memory and padding."""
    offset = 0
    fpn = []
    for (h, w), count in zip(self.sizes, self.counts):
      fpn.append(flat[:, offset:offset + count].reshape(1, 256, h, w))
      offset += count
    text_mem = flat[:, offset:offset + self.length * 256]
    text_mem = text_mem.reshape(1, self.length, 256)
    offset += self.length * 256
    text_pad = flat[:, offset:offset + self.length]
    return fpn, text_mem, text_pad


class HeadFlatStock(_HeadFlatBase):
  """Reference head that calls the stock `forward_grounding`."""

  def forward(self, flat):
    """Produces [logits | boxes | presence | masks]."""
    from sam3.model.data_misc import FindStage
    from sam3.model.geometry_encoders import Prompt

    fpn, text_mem, text_pad = self._split(flat)
    device = flat.device
    backbone_out = {
        "backbone_fpn": fpn,
        "vision_pos_enc": list(self.pos),
        "language_features": text_mem.transpose(0, 1),
        "language_mask": text_pad > 0.5,
    }
    find_input = FindStage(
        img_ids=torch.tensor([0], device=device),
        text_ids=torch.tensor([0], device=device),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    geometric_prompt = Prompt(
        box_embeddings=torch.zeros(0, 1, 4, device=device),
        box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
    )
    out = self.det.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_input,
        find_target=None,
        geometric_prompt=geometric_prompt,
    )
    return torch.cat([
        out["pred_logits"].reshape(1, -1),
        out["pred_boxes"].reshape(1, -1),
        out["presence_logit_dec"].reshape(1, -1),
        out["pred_masks"].reshape(1, -1),
    ], 1)


class HeadFlat4d(_HeadFlatBase):
  """GPU-clean head for text-only prompting (no boxes, no points).

  Re-implements `forward_grounding` without the tensor-index gathers of the
  stock path, batch-first and rank-4 where the delegates require it.
  """

  def __init__(self, det, sizes=FPN_SIZES):
    super().__init__(det, sizes)
    dec = det.transformer.decoder
    self.register_buffer(
        "query_embed", dec.query_embed.weight.detach().clone(),
        persistent=False)
    self.register_buffer(
        "ref_boxes0",
        dec.reference_points.weight.detach().clone().sigmoid(),
        persistent=False)
    # With a text-only prompt the geometry class token up to the final
    # projection is a constant; the cross-attention layers that follow stay
    # in the graph because they read the image features.
    geometry_encoder = det.geometry_encoder
    with torch.no_grad():
      cls = geometry_encoder.cls_embed.weight.view(
          1, 1, geometry_encoder.d_model
      )
      if geometry_encoder.final_proj is not None:
        cls = geometry_encoder.norm(geometry_encoder.final_proj(cls))
    self.register_buffer("geo_pre", cls.detach().clone(), persistent=False)

  def forward(self, flat):
    """Produces [logits | boxes | presence | masks]."""
    det = self.det
    length = self.length
    fpn, text_mem, text_pad = self._split(flat)
    device = flat.device
    pos = list(self.pos)
    # num_feature_levels == 1: the encoder, geometry encoder and decoder all
    # see the 72x72 level only.
    feat_sizes = [tuple(pos[-1].shape[-2:])]
    img_feats = [fpn[-1].flatten(2).permute(2, 0, 1)]
    img_pos = [pos[-1].flatten(2).permute(2, 0, 1)]
    feat_bf = fpn[-1].flatten(2).transpose(1, 2)
    pos_bf0 = pos[-1].flatten(2).transpose(1, 2)
    # The delegates refuse ops whose inputs are all constant, and the
    # converter does not fold them, so tie the constant-rooted tensors to a
    # runtime zero derived from the input. (x * 0 would be folded away.)
    zero = torch.clamp(flat[:, :1], min=0.0, max=0.0)
    geometry_encoder = det.geometry_encoder
    g = self.geo_pre + zero
    if geometry_encoder.encode is not None:
      g = geometry_encode_bf(geometry_encoder, g, feat_bf, pos_bf0)
    prompt = torch.cat([text_mem, g], 1)
    prompt_mask = torch.cat(
        [text_pad, torch.zeros(1, 1, device=device)], 1
    )
    prompt_sf = prompt.transpose(0, 1)
    memory = det.transformer.encoder(
        src=img_feats.copy(),
        src_key_padding_mask=None,
        src_pos=img_pos.copy(),
        prompt=prompt_sf,
        prompt_pos=torch.zeros_like(prompt_sf),
        prompt_key_padding_mask=prompt_mask,
        feat_sizes=feat_sizes,
        encoder_extra_kwargs=None,
    )
    enc_hs = memory["memory"].transpose(0, 1)
    pos_bf = memory["pos_embed"].transpose(0, 1)
    dec = det.transformer.decoder
    tgt = self.query_embed.unsqueeze(0) + zero
    ref0 = self.ref_boxes0.unsqueeze(0) + zero
    hs4, ref_in, presence = decoder_forward_bf(
        dec, tgt, enc_hs, pos_bf, ref0, prompt, prompt_mask
    )
    outputs_class = det.dot_prod_scoring(hs4, prompt_sf, prompt_mask)
    anchor = dec.bbox_embed(hs4)
    boxes = (
        inverse_sigmoid_clean(ref_in.unsqueeze(0)) + anchor
    ).sigmoid()
    prob_presence = presence.sigmoid().reshape(1, 1, 1, 1)
    logits = inverse_sigmoid_clean(
        outputs_class.sigmoid() * prob_presence
    )
    logits = torch.clamp(logits, min=-10.0, max=10.0)
    seg_head = det.segmentation_head
    embed = enc_hs
    if seg_head.cross_attend_prompt is not None:
      normed = seg_head.cross_attn_norm(embed)
      embed = embed + mha_core_bf(
          seg_head.cross_attend_prompt, normed, prompt, prompt,
          key_padding_mask=prompt_mask)
    embed = embed.transpose(1, 2).reshape(1, 256, *self.sizes[-1])
    pixel_embed = seg_head.pixel_decoder([fpn[0], fpn[1], embed])
    instance = seg_head.instance_seg_head(pixel_embed)
    masks = seg_head.mask_predictor(hs4, instance)
    return torch.cat([
        logits.reshape(1, -1),
        boxes.reshape(1, -1),
        presence.reshape(1, -1),
        masks.reshape(1, -1),
    ], 1)


# ------------------------------------------------------------ model loading


def _apply_export_shims():
  """Applies the CPU and export shims the stock code needs.

  None of these change the numerics at 1008x1008: they move CUDA-only
  allocations to CPU, replace a fused CUDA kernel with its plain form, pin one
  shape that torch.export would otherwise treat as unbacked, and short-circuit
  the geometry paths that a text-only prompt calls with empty tensors.
  """
  import sam3.model.geometry_encoders as geometry_encoders
  import sam3.model.vitdet as vitdet
  from sam3 import model_builder
  from sam3.model.decoder import TransformerDecoder
  from sam3.model.position_encoding import PositionEmbeddingSine

  # PositionEmbeddingSine(precompute_resolution=...) allocates on CUDA; the
  # lazy per-size cache computes the same constants on CPU.
  model_builder._create_position_encoding = (
      lambda precompute_resolution=None: PositionEmbeddingSine(
          num_pos_feats=256,
          normalize=True,
          scale=None,
          temperature=10000,
          precompute_resolution=None,
      )
  )
  original_get_coords = TransformerDecoder._get_coords
  TransformerDecoder._get_coords = staticmethod(
      lambda h, w, device: original_get_coords(
          h, w, "cpu" if device == "cuda" else device
      )
  )
  # The ViT MLP uses a bf16 CUDA fused addmm; the plain form is identical.
  vitdet.addmm_act = lambda act, linear, x: act()(linear(x))
  # The geometry encoder stages tensors with pin_memory(), a CUDA-only op.
  torch.Tensor.pin_memory = lambda self, *args, **kwargs: self

  original_roi_align = geometry_encoders.torchvision.ops.roi_align

  def roi_align_empty_ok(inp, boxes, output_size, *args, **kwargs):
    """roi_align that returns an empty tensor for zero boxes."""
    if isinstance(boxes, (list, tuple)):
      if all(b.shape[0] == 0 for b in boxes):
        return inp.new_zeros(0, inp.shape[1], output_size, output_size)
    return original_roi_align(inp, boxes, output_size, *args, **kwargs)

  geometry_encoders.torchvision.ops.roi_align = roi_align_empty_ok
  original_grid_sample = geometry_encoders.torch.nn.functional.grid_sample

  def grid_sample_empty_ok(inp, grid, *args, **kwargs):
    """grid_sample that returns an empty tensor for zero points."""
    if grid.shape[1] == 0:
      return inp.new_zeros(
          inp.shape[0], inp.shape[1], 0, grid.shape[2]
      )
    return original_grid_sample(inp, grid, *args, **kwargs)

  geometry_encoders.torch.nn.functional.grid_sample = grid_sample_empty_ok
  original_concat = geometry_encoders.concat_padded_sequences

  def concat_padded_empty_ok(seq1, mask1, seq2, mask2, return_index=False):
    """concat_padded_sequences short-circuited when one side is empty.

    The stock implementation uses a data-dependent scatter. With an empty
    side it is the identity on the other side, which is all a text-only
    prompt (zero points, zero boxes, one class token) needs.
    """
    if seq1.shape[0] == 0 or seq2.shape[0] == 0:
      if seq1.shape[0] == 0:
        seq, mask = seq2, mask2
      else:
        seq, mask = seq1, mask1
      if return_index:
        index = torch.arange(seq2.shape[0], device=seq.device)
        index = index[None].repeat(seq.shape[1], 1)
        return seq, mask, index
      return seq, mask
    return original_concat(seq1, mask1, seq2, mask2, return_index)

  geometry_encoders.concat_padded_sequences = concat_padded_empty_ok


def build_detector(ckpt_path=None, verbose=True):
  """Builds the SAM 3 detector and loads the `detector.` weights.

  Args:
    ckpt_path: path to the released checkpoint. The file also carries the
      video tracker under a `tracker.` prefix, which is ignored here.
    verbose: print the checkpoint key accounting.

  Returns:
    The detector in eval mode, with the export shims applied.
  """
  import pkg_resources
  from sam3 import model_builder
  from sam3.model.decoder import TransformerDecoder
  from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
  from sam3.model.vl_combiner import SAM3VLBackboneTri

  _apply_export_shims()
  bpe_path = pkg_resources.resource_filename(
      "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
  )
  tri_neck = model_builder._create_multiplex_tri_backbone(
      compile_mode=None, use_fa3=False, use_rope_real=True
  )
  text_encoder = model_builder._create_text_encoder(bpe_path)
  backbone = SAM3VLBackboneTri(
      scalp=0, visual=tri_neck, text=text_encoder
  )
  detector = Sam3MultiplexDetector(
      num_feature_levels=1,
      backbone=backbone,
      transformer=model_builder._create_sam3_transformer(use_fa3=False),
      segmentation_head=model_builder._create_segmentation_head(
          use_fa3=False),
      semantic_segmentation_head=None,
      input_geometry_encoder=model_builder._create_geometry_encoder(),
      use_early_fusion=True,
      use_dot_prod_scoring=True,
      dot_prod_scoring=model_builder._create_dot_product_scoring(),
      supervise_joint_box_scores=True,
      is_multiplex=True,
  )
  if ckpt_path:
    started = time.time()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
      ckpt = ckpt["model"]
    prefix = "detector."
    weights = {
        k[len(prefix):]: v
        for k, v in ckpt.items()
        if k.startswith(prefix)
    }
    missing, unexpected = detector.load_state_dict(weights, strict=False)
    missing = [
        k for k in missing
        if "freqs_cis" not in k and "attn_mask" not in k
    ]
    if verbose:
      print(
          f"[ckpt] detector.* keys={len(weights)} "
          f"missing={len(missing)} unexpected={len(unexpected)} "
          f"({time.time() - started:.0f}s)"
      )
    assert not missing, f"missing detector weights: {missing[:10]}"
  # The ViT attention keeps a complex freqs_cis buffer next to the real and
  # imaginary pair actually used; litert-torch cannot serialize complex64.
  for module in detector.modules():
    freqs = getattr(module, "freqs_cis", None)
    if isinstance(freqs, torch.Tensor) and freqs.is_complex():
      assert getattr(module, "use_rope_real", False)
      module._buffers["freqs_cis"] = torch.zeros(1)
  # The log-RPB feature size arrives as a 0-d tensor, which turns the
  # coordinate grid into unbacked symbols under torch.export. The graph is
  # fixed at 1008 -> 72x72, so pin it.
  original_rpb = TransformerDecoder._get_rpb_matrix
  TransformerDecoder._get_rpb_matrix = (
      lambda self, boxes, feat_size: original_rpb(self, boxes, (72, 72))
  )
  detector.eval()
  return detector


def preprocess_image(path, size=IMAGE_SIZE):
  """Loads an image as the (1, 3, size, size) tensor the vision graph takes.

  Args:
    path: image file path.
    size: square side length; the graph is fixed at 1008.

  Returns:
    A (tensor, (width, height)) tuple with the original image size.
  """
  from PIL import Image

  image = Image.open(path).convert("RGB")
  original_size = image.size
  resized = image.resize((size, size), Image.BILINEAR)
  array = np.asarray(resized).astype(np.float32) / 255.0
  x = torch.from_numpy(array).permute(2, 0, 1)[None]
  return (x - 0.5) / 0.5, original_size


def decode_detections(head_out, threshold=0.5):
  """Turns the flat head output into scores, boxes and mask logits.

  Args:
    head_out: flat head output, (1, 1001 + 200*288*288) or 1-D.
    threshold: keep queries whose score exceeds this.

  Returns:
    A (scores, boxes, masks, kept_indices) tuple. Scores are
    sigmoid(logit) * sigmoid(presence), boxes are cxcywh normalized to the
    resized image, and masks are 288x288 logits.
  """
  flat = np.asarray(head_out).reshape(-1)
  logits = flat[:NUM_QUERIES]
  boxes = flat[NUM_QUERIES:NUM_QUERIES * 5].reshape(NUM_QUERIES, 4)
  presence = flat[NUM_QUERIES * 5]
  masks = flat[NUM_QUERIES * 5 + 1:].reshape(
      NUM_QUERIES, MASK_SIZE, MASK_SIZE
  )
  scores = (1.0 / (1.0 + np.exp(-logits))) * (
      1.0 / (1.0 + np.exp(-presence))
  )
  kept = np.where(scores > threshold)[0]
  return scores, boxes, masks, kept
