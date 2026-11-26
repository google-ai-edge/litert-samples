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

"""Out-of-tree litert-torch `ModelExtension` for LFM3-1.7B on Google Tensor G6.

build_lfm3_1_7b_g6.py hands it to the stock HF exporter
(`--extension_modules`) for a split-cache export, the form LiteRT-LM's NPU
executor runs; any other export stops in `patch_model`. It subclasses the
CPU/GPU recipe's `Lfm3Extension` (../converted/lfm3_extension.py) and keeps
its `build_llm_metadata`: the EOS id as the only stop, and earlier assistant
turns rendered without their thought. Its own hooks:

  * `get_export_defaults`: the Jinja template, which the NPU preset turns off
    and which renders LFM3's thought channel, and LiteRT-LM's handwritten NPU
    sampler, masks and cache update.
  * `patch_model`: the conv layers on the split cache, and the forms the
    Pixel 11 compiles and computes right in its 16-bit floats:
    - every RMSNorm as plain ops on x / max|x| (the Pixel 11's on-device
      compiler fails to import the `odml.rms_norm` composite, and 16-bit
      floats lose the squares of small activations);
    - `Lfm3Attention.forward` with the per-head gate spread by a matmul (the
      on-device compiler scrambles the gate's `unsqueeze(-1)`), the queries of
      a prefill longer than 26 tokens in blocks of 26 (it rejects 32, 64 and
      128), or in one block in a sliding layer whose cache is its window, and
      a sliding layer's decode mask floored (it turns LiteRT-LM's -1e9 into
      NaN over a 256-slot cache);
    - the remote module's `rotate_half` as a matmul (it rejects the half-width
      slices of a 128-wide head);
    - `lm_head` with a copy of its weight times 8 fed by the final norm with
      its weight over 8 (untied from the input embedding; same logits), and
      each MLP's `w1` and `w3` with copies of their weights times 4 fed by its
      `ffn_norm` with its weight over 4.

Remote-code classes are distinct objects per checkpoint folder, so modules are
matched by class name.
"""

import contextlib
import functools
import math
import os
import sys

import numpy as np
import torch
from transformers import modeling_outputs

from litert_torch.backend.torch_tfl import _ops as _  # Registers torch.ops.tfl.*
from litert_torch.generative.custom_ops import bmm_4d as bmm_lib
from litert_torch.generative.export_hf.experimental.composites import short_conv as short_conv_composite
from litert_torch.generative.export_hf.model_ext import extension

_cpu_gpu = extension.load_extension_file(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                 'converted', 'lfm3_extension.py'))

# Queries per attention block of a longer prefill. The Pixel 11's on-device
# compiler takes LFM3's attention against 2,048 + 128 keys at up to 30
# queries, not at 32; the bundle checked on a Pixel 11 uses 26. Against a
# sliding layer's 256-slot cache it takes all 128 at once.
_QUERY_BLOCK = 26
# LiteRT-LM's -1e9 is -inf in the TPU's 16-bit floats, which the Pixel 11 turns
# into NaN in a sliding layer's decode attention over a 256-slot cache.
_MASK_FLOOR = -3e4
# The TPU keeps the sum of an int8 fully connected layer before its per-row
# scale in 16-bit floats; lm_head and the MLP's w1 and w3 come closest to 65,504.
_LM_HEAD_SCALE = 8.0
_FFN_SCALE = 4.0


class Lfm3LiteRTShortConv(torch.nn.Module):
  """LFM3 gated short conv reading and writing the LiteRT cache's conv state."""

  def __init__(self, original: torch.nn.Module):
    super().__init__()
    self.layer_idx = original.layer_idx
    self.in_proj = original.in_proj
    self.conv = original.conv
    self.out_proj = original.out_proj

  def forward(
      self, hidden_states, past_key_values=None, attention_mask=None, **kwargs
  ):
    """`attention_mask` is the `[1, T]` bool of real slots, or None (all real)."""
    del kwargs
    layer = past_key_values.layers[self.layer_idx]
    if attention_mask is None and hidden_states.shape[1] == 1:
      # Decode: one product with the stacked taps and one sum. `short_conv`
      # adds the taps one by one over six one-row slices of the state, which
      # the Pixel 11's on-device compiler fails on in pieces of 7 or 8 layers
      # ("Does not support overlap yet").
      b, c, x_proj = self.in_proj(hidden_states).chunk(3, dim=-1)
      padded = torch.cat([layer.conv_states, b * x_proj], dim=1)
      # The depthwise weight [hidden, 1, L] as channels-last [1, L, hidden].
      taps = self.conv.weight.squeeze(1).transpose(0, 1).unsqueeze(0)
      # `tfl.sum`: the converter drops the kept axis of `Tensor.sum` ahead of
      # a mul, and the Pixel 11 was checked with the axis kept.
      conv_out = torch.ops.tfl.sum(padded * taps, [1], True)
      if self.conv.bias is not None:
        conv_out = conv_out + self.conv.bias
      layer.conv_states = padded[:, 1:, :]
      return self.out_proj(c * conv_out)
    num_valid_tokens = None
    if attention_mask is not None:
      num_valid_tokens = (
          attention_mask.to(torch.float32).sum().to(torch.int32).reshape(1)
      )
    y, layer.conv_states = short_conv_composite.short_conv(
        in_proj_out=self.in_proj(hidden_states),
        conv_state=layer.conv_states,
        conv_weight=self.conv.weight,
        conv_bias=self.conv.bias,
        num_valid_tokens=num_valid_tokens,
    )
    return self.out_proj(y)


class PlainRMSNorm(torch.nn.Module):
  """RMSNorm without the `odml.rms_norm` composite, safe in 16-bit floats.

  The Pixel 11's TPU computes in 16-bit floats, where the squares of LFM3's
  small activations (about 1e-3 after a branch, before the sandwich norms)
  underflow to 0. Dividing by the largest magnitude m first keeps the squares
  near 1: x / sqrt(mean(x^2) + eps) = y / sqrt(mean(y^2) + eps / m^2), y = x / m.
  """

  def __init__(self, weight: torch.nn.Parameter, eps: float):
    super().__init__()
    self.weight = weight
    self.variance_epsilon = eps

  def forward(self, hidden_states):
    dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    # The 1e-4 floor keeps m a normal 16-bit float; any m > 0 gives the same norm.
    m = x.abs().amax(-1, keepdim=True) + 1e-4
    y = x / m
    eps_term = (math.sqrt(self.variance_epsilon) / m).pow(2)
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps_term)
    return self.weight * y.to(dtype)


def _attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
  """`Lfm3Attention.forward` with the per-head gate spread by a matmul.

  The HF forward broadcasts the gate `[b, t, heads]` over each head with
  `unsqueeze(-1)`; the Pixel 11's on-device compiler scrambles that reshape to
  `[b, t, heads, 1]`. A matmul with a constant 0/1 matrix copies each head's
  gate to its `head_dim` channels instead.
  """
  if hidden_states.shape[1] > _QUERY_BLOCK:
    return _query_block_attention_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        **kwargs,
    )
  modeling = sys.modules[type(self).__module__]
  input_shape = hidden_states.shape[:-1]
  hidden_shape = (*input_shape, -1, self.head_dim)
  query_states = self.q_norm(
      self.q_proj(hidden_states).view(*hidden_shape)
  ).transpose(1, 2)
  key_states = self.k_norm(
      self.k_proj(hidden_states).view(*hidden_shape)
  ).transpose(1, 2)
  value_states = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)
  cos, sin = position_embeddings
  query_states, key_states = modeling.apply_rotary_pos_emb(
      query_states, key_states, cos, sin
  )
  if past_key_values is not None:
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx
    )
  if self.sliding_window and hidden_states.shape[1] == 1:
    attention_mask = torch.maximum(
        attention_mask, torch.full((), _MASK_FLOOR, dtype=attention_mask.dtype)
    )
  attention_interface = modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
      self.config._attn_implementation, modeling.eager_attention_forward  # pylint: disable=protected-access
  )
  attn_output, attn_weights = attention_interface(
      self,
      query_states,
      key_states,
      value_states,
      attention_mask,
      dropout=0.0,
      scaling=self.scaling,
      sliding_window=self.sliding_window,
      **kwargs,
  )
  gate = _spread_gate(self, hidden_states)
  attn_output = attn_output.reshape(*input_shape, -1) * gate
  return self.o_proj(attn_output), attn_weights


def _spread_gate(self, hidden_states):
  """The sigmoid gate `[b, t, heads]` copied to each head's channels."""
  heads = self.gate_proj.out_features
  spread = np.kron(np.eye(heads), np.ones((1, self.head_dim)))
  return torch.sigmoid(self.gate_proj(hidden_states)) @ torch.from_numpy(
      spread.astype(np.float32)
  ).to(hidden_states.dtype)


def _query_block_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values,
    **kwargs,
):
  """`_attention_forward` of a split-cache prefill longer than 26 tokens.

  The Pixel 11's on-device compiler rejects LFM3's attention against 2,048 + T
  keys at 32, 64 and 128 queries ("Reshape input/output slice mismatch",
  "Sharding failed: Invalid shards") and a reshape of a slice of the queries.
  So each block of 26 queries computes them from its own rows of
  `hidden_states` and attends to the S cached and all T new keys, a score block
  `[1, kv_heads, groups * 26, S + T]`; K and V still come from all T tokens. A
  sliding layer whose cache is its window takes all T queries in one block.
  """
  if (kwargs.get('k_ts_idx'), kwargs.get('v_ts_idx')) != (2, 3):
    raise ValueError('lfm3_g6_extension: query blocks need k_ts_idx=2 and'
                     ' v_ts_idx=3.')
  modeling = sys.modules[type(self).__module__]
  t, head_dim = hidden_states.shape[1], self.head_dim
  groups = self.num_key_value_groups
  # The remote `apply_rotary_pos_emb` (float32), applied to K and Q apart.
  cos, sin = (x.unsqueeze(1).float() for x in position_embeddings)
  key_states = self.k_norm(
      self.k_proj(hidden_states).view(1, t, -1, head_dim)
  ).transpose(1, 2)
  key_states = key_states * cos + modeling.rotate_half(key_states) * sin
  value_states = (
      self.v_proj(hidden_states).view(1, t, -1, head_dim).transpose(1, 2)
  )
  (key_past, key), (value_past, value) = past_key_values.update(
      key_states, value_states, self.layer_idx
  )
  size = t if key_past.shape[2] == self.sliding_window else _QUERY_BLOCK
  outputs = []
  for i in range(0, t, size):
    rows = slice(i, i + size)
    block = hidden_states[:, rows]
    query = self.q_norm(
        self.q_proj(block).view(*block.shape[:-1], -1, head_dim)
    ).transpose(1, 2)
    query = query * cos[:, :, rows] + modeling.rotate_half(query) * sin[
        :, :, rows
    ]
    mask = attention_mask[:, :, rows]
    # `split_cache_attention` of this block, which would split the scores at
    # the query count instead of the T new keys.
    _, heads, b, _ = query.shape
    query = query.reshape(1, heads // groups, groups * b, head_dim)
    query = query * self.scaling
    logits = torch.cat(
        [bmm_lib.bmm_4d(query, key_past), bmm_lib.bmm_4d(query, key)], dim=-1
    )
    probs = torch.softmax(logits + torch.cat([mask] * groups, dim=-2), dim=-1)
    encoded = bmm_lib.bmm_4d(probs[..., :-t], value_past) + bmm_lib.bmm_4d(
        probs[..., -t:], value
    )
    outputs.append(encoded.reshape(1, heads, b, head_dim).permute(0, 2, 1, 3))
  gate = _spread_gate(self, hidden_states)
  attn_output = torch.cat(outputs, dim=1).reshape(1, t, -1) * gate
  return self.o_proj(attn_output), None


def rotate_half_by_matmul(x: torch.Tensor) -> torch.Tensor:
  """`rotate_half` as a matmul with a constant signed permutation.

  The Pixel 11's on-device compiler rejects the half-width slices of a
  128-wide head ("slice ... larger than allocated").
  """
  d = x.shape[-1]
  half = np.arange(d // 2)
  r = np.zeros((d, d), dtype=np.float32)
  r[half + d // 2, half] = -1.0  # out[:half] = -x[half:]
  r[half, half + d // 2] = 1.0  # out[half:] = x[:half]
  return x @ torch.from_numpy(r).to(x.dtype)


def _real_slots_from_split_mask(mask: torch.Tensor) -> torch.Tensor:
  """`[1, T]` bool: the slots of a split-cache mask that may see themselves.

  The last T columns of a split-cache mask `[1, 1, T, S + T]` are this call's
  tokens; LiteRT-LM masks the column of every padding slot, so the diagonal of
  that block is unmasked exactly on the real slots. A strided slice reads the
  diagonal (`torch.eye` lowers to `STABLEHLO_IOTA`, which the Pixel 11's
  LiteRT runtime does not register) of the comparison, not of the raw mask:
  LiteRT-LM's -1e9 is -inf in the TPU's 16-bit floats, and the raw read gave
  all-zero conv states there.
  """
  t = mask.shape[2]
  visible = mask[0, 0, :, -t:] > -1.0
  return visible.reshape(-1)[:: t + 1].unsqueeze(0)


def _scaled_linear(linear: torch.nn.Linear, factor: float) -> torch.nn.Linear:
  """`linear` with a copy of its weight times `factor`, for an input over it."""
  out = torch.nn.Linear(
      linear.in_features, linear.out_features, bias=False, device='meta'
  )
  out.weight = torch.nn.Parameter(
      linear.weight.detach() * factor, requires_grad=False
  )
  out.bias = linear.bias
  return out


def _model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    cache_position=None,
    valid_mask=None,
    **kwargs,
):
  """`Lfm3Model.forward` for export: the exportable supplies masks and cache."""
  # The exportable's `valid_mask` reads the same diagonal with `>= 0`; the conv
  # layers keep `_real_slots_from_split_mask`, as in the bundle checked on a
  # Pixel 11.
  del use_cache, cache_position, valid_mask
  hidden_states = (
      self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
  )
  if not isinstance(attention_mask, dict):
    attention_mask = {'full_attention': attention_mask}
  real_slots = None  # Decode: the one token is real.
  if hidden_states.shape[1] > 1:
    real_slots = _real_slots_from_split_mask(attention_mask['full_attention'])
  for i, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    layer_type = self.config.layer_types[i]
    if layer_type == 'conv':
      hidden_states = layer(
          hidden_states,
          attention_mask=real_slots,
          past_key_values=past_key_values,
      )
      continue
    hidden_states = layer(
        hidden_states,
        attention_mask=attention_mask.get(
            layer_type, attention_mask['full_attention']
        ),
        position_embeddings=self.rotary_emb(
            hidden_states, position_ids, layer_type
        ),
        position_ids=position_ids,
        past_key_values=past_key_values,
        **kwargs,
    )
  return modeling_outputs.BaseModelOutputWithPast(
      last_hidden_state=self.embedding_norm(hidden_states),
      past_key_values=past_key_values,
  )


@contextlib.contextmanager
def patch_lfm3_model(model, export_config):
  """Applies the G6 export patches to every LFM3 module under `model`."""
  if not getattr(export_config, 'split_cache', False):
    raise NotImplementedError(
        'lfm3_g6_extension: split-cache exports only (--split_cache=True);'
        ' the CPU/GPU export is ../converted/lfm3_extension.py')
  if any(p.is_meta for p in model.parameters()):
    # Converter V2 loads meta weights from the checkpoint by name, which drops
    # the in-memory rescales below.
    raise ValueError(
        'lfm3_g6_extension: the norm, lm_head and MLP weights are rescaled in'
        ' memory, which use_v2 (meta-device weights) would drop; export with'
        ' use_v2=False.')
  replaced = []  # (parent, attribute, original module)
  forwards = []  # modules given an instance `forward`
  for parent in list(model.modules()):
    for name, child in list(parent.named_children()):
      kind = type(child).__name__
      if kind == 'Lfm3ShortConv':
        setattr(parent, name, Lfm3LiteRTShortConv(child))
        replaced.append((parent, name, child))
      elif kind in ('Lfm3RMSNorm', 'RMSNorm'):
        weight = child.weight
        scale = {'embedding_norm': _LM_HEAD_SCALE, 'ffn_norm': _FFN_SCALE}
        if name in scale:
          weight = torch.nn.Parameter(
              weight.detach() / scale[name], requires_grad=False
          )
        setattr(parent, name, PlainRMSNorm(weight, child.variance_epsilon))
        replaced.append((parent, name, child))
      elif name == 'lm_head':
        setattr(parent, name, _scaled_linear(child, _LM_HEAD_SCALE))
        replaced.append((parent, name, child))
      elif type(parent).__name__ == 'Lfm3MLP' and name in ('w1', 'w3'):
        setattr(parent, name, _scaled_linear(child, _FFN_SCALE))
        replaced.append((parent, name, child))
  functions = []  # (remote modeling module, original rotate_half)
  for module in model.modules():
    kind = type(module).__name__
    if kind == 'Lfm3Attention':
      module.forward = functools.partial(_attention_forward, module)
      forwards.append(module)
    elif kind == 'Lfm3Model':
      module.forward = functools.partial(_model_forward, module)
      forwards.append(module)
      modeling = sys.modules[type(module).__module__]
      if modeling.rotate_half is not rotate_half_by_matmul:
        functions.append((modeling, modeling.rotate_half))
        modeling.rotate_half = rotate_half_by_matmul
  if not any(type(m).__name__ == 'Lfm3Model' for m in forwards):
    raise ValueError('lfm3_g6_extension: no Lfm3Model under the exported'
                     ' module.')
  print(f'lfm3_g6_extension: {len(replaced)} modules replaced,'
        f' {len(forwards)} forwards')
  try:
    yield
  finally:
    for modeling, original in functions:
      modeling.rotate_half = original
    for module in forwards:
      del module.forward
    for parent, name, original in reversed(replaced):
      setattr(parent, name, original)


@extension.register_model_extension()
class Lfm3G6Extension(_cpu_gpu.Lfm3Extension):
  """LFM3 split-cache export for LiteRT-LM's NPU executor on Tensor G6."""

  def get_export_defaults(self, config_dict, task=None, target='cpu_gpu'):
    del config_dict, task, target
    # LiteRT-LM's NPU executor runs its handwritten sampler, masks and cache
    # update only when the bundle asks for them; the bundle checked on a
    # Pixel 11 asks for all three.
    return extension.ExportDefaults(export_kwargs={
        'use_jinja_template': True,
        'enable_neon_for_npu_greedy_sampling': True,
        'use_hw_masking_for_npu': True,
        'use_hw_cache_update_for_npu': True,
    })

  def patch_model(self, model, export_config):
    return patch_lfm3_model(model, export_config)
