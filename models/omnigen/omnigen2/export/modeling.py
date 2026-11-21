# Copyright 2026 The LiteRT Torch Authors.
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
"""OmniGen2 (OmniGen2Transformer2DModel + Qwen2.5-VL-3B + AutoencoderKL) modules.

PyTorch modeling definitions matching the exact state_dict keys and numerical
math of `OmniGen2/OmniGen2`, reusing common components from `transformers` and
`export_hf`:
- `mllm`: Frozen `Qwen2.5-VL-3B-Instruct` text backbone (`OmniGen2TextEncoder`)
  built on `transformers.Qwen2Model` with GPU-compatible 4D `repeat_kv` from
  `bonsai_flux2` and eager attention (`434` text keys after filtering frozen
  vision tower `visual.*` keys).
- `transformer`: OmniGen2-specific Lumina2-style `OmniGen2Transformer2DModel`
  (`582` keys: 32 main layers + 2-layer `noise_refiner`, `ref_image_refiner`,
  `context_refiner` + 3D Omni-RoPE + `LuminaLayerNormContinuous`), reusing
  `modeling_flux2` RoPE and timestep embedding modules.
- `vae`: 16-channel `OmniGen2AutoencoderKL` (`244` keys,
  `scaling_factor=0.3611`, `shift_factor=0.1159`, no BN / `quant_conv`)
  subclassing `modeling_flux2.AutoencoderKLFlux2`.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
import types
from typing import Any

from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import bonsai_flux2
from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import modeling_flux2
import safetensors
import torch
from torch import nn
import torch.nn.functional as F
import transformers

CONDITIONING_EMBED_DIM = 1024
TIMESTEP_CHANNELS = 256
MAX_REF_IMAGES = 5


def _resolve_safetensors_files(directory: str) -> list[str]:
  """Resolves single or sharded safetensors file paths inside `directory`."""
  for index_name in (
      'diffusion_pytorch_model.safetensors.index.json',
      'model.safetensors.index.json',
  ):
    index_path = os.path.join(directory, index_name)
    if os.path.exists(index_path):
      with open(index_path, 'r') as f:
        index_data = json.load(f)
      shard_files = sorted(set(index_data.get('weight_map', {}).values()))
      return [os.path.join(directory, shard) for shard in shard_files]

  for single_name in (
      'diffusion_pytorch_model.safetensors',
      'model.safetensors',
  ):
    single_path = os.path.join(directory, single_name)
    if os.path.exists(single_path):
      return [single_path]
  return []


def load_safetensors_state_dict(directory: str) -> dict[str, torch.Tensor]:
  """Loads single or sharded safetensors weights from `directory` as float32."""
  return modeling_flux2._load_safetensors_state_dict(directory)  # pylint: disable=protected-access


def _read_safetensors_keys(directory: str) -> set[str]:
  """Reads tensor key names from safetensors headers without loading weights."""
  files = _resolve_safetensors_files(directory)
  if not files:
    raise FileNotFoundError(f'No safetensors files found in {directory}.')
  keys: set[str] = set()
  for file_path in files:
    with safetensors.safe_open(file_path, framework='pt') as handle:
      keys.update(handle.keys())
  return keys


def build_omnigen2_3d_rope_cos_sin(
    ids: torch.Tensor,
    axes_dim_rope: tuple[int, int, int] = (40, 40, 40),
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Computes 3D Omni-RoPE `(cos, sin)` tensors of shape `[seq_len, head_dim]`.

  Matches `OmniGen2RotaryPosEmbed` in `omnigen2/models/transformers/repo.py`
  where each axis `a` of dimension `d_a` uses
  `modeling_flux2.get_1d_rotary_pos_embed(d_a, ...)` with `theta=10000.0` and
  complex pair rotation (`use_real=False`).

  Args:
    ids: Position ID tensor of shape `[seq_len, 3]` with `(id, h, w)`
      coordinates.
    axes_dim_rope: Per-axis rotary dimensions `(d_id, d_h, d_w)`.
    theta: RoPE base frequency (`10000.0` for OmniGen2).

  Returns:
    Tuple `(cos, sin)` each of shape `[seq_len, sum(axes_dim_rope)]`.
  """
  cos_parts = []
  sin_parts = []
  for axis_idx, axis_dim in enumerate(axes_dim_rope):
    c, s = modeling_flux2.get_1d_rotary_pos_embed(
        axis_dim, ids[:, axis_idx], theta=theta
    )
    cos_parts.append(c)
    sin_parts.append(s)
  return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def configure_qwen2_for_gpu_export(module: nn.Module) -> None:
  """Configures `transformers.Qwen2Model` for eager attention and 4D repeat_kv."""
  modeling_qwen2 = sys.modules.get('transformers.models.qwen2.modeling_qwen2')
  if modeling_qwen2 is None:
    qwen2_mod = getattr(getattr(transformers, 'models', None), 'qwen2', None)
    modeling_qwen2 = getattr(qwen2_mod, 'modeling_qwen2', None)
  if modeling_qwen2 is not None and hasattr(modeling_qwen2, 'repeat_kv'):
    setattr(
        modeling_qwen2,
        'repeat_kv',
        bonsai_flux2._repeat_kv_4d,  # pylint: disable=protected-access
    )

  if hasattr(module, 'config') and module.config is not None:
    setattr(module.config, '_attn_implementation', 'eager')
  for sub in module.modules():
    if hasattr(sub, 'config') and sub.config is not None:
      setattr(sub.config, '_attn_implementation', 'eager')


class OmniGen2TextEncoder(nn.Module):
  """Frozen `Qwen2.5-VL-3B-Instruct` (`mllm`) text conditioning model for OmniGen2.

  Delegates the transformer backbone (`self.model`) to
  `transformers.Qwen2Model`,
  which is architecturally identical to `Qwen2_5_VLTextModel` for 1D text inputs
  (`model.embed_tokens.weight`, `model.layers.0..35.*`, `model.norm.weight` —
  434 keys) while avoiding 3D mRoPE `expand` ops during LiteRT export.
  """

  def __init__(
      self,
      vocab_size: int = 151936,
      hidden_size: int = 2048,
      intermediate_size: int = 11008,
      num_hidden_layers: int = 36,
      num_attention_heads: int = 16,
      num_key_value_heads: int = 2,
      rms_norm_eps: float = 1e-6,
      rope_theta: float = 1000000.0,
      max_position_embeddings: int = 32768,
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.head_dim = hidden_size // num_attention_heads
    self.rms_norm_eps = rms_norm_eps
    self.rope_theta = rope_theta

    qwen2_cfg_cls: Any = getattr(transformers, 'Qwen2Config')
    qwen2_model_cls: Any = getattr(transformers, 'Qwen2Model')
    qwen_config = qwen2_cfg_cls(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        use_sliding_window=False,
        use_cache=False,
        tie_word_embeddings=False,
        attn_implementation='eager',
    )
    self.config = qwen_config
    self.model: nn.Module = qwen2_model_cls(qwen_config)
    configure_qwen2_for_gpu_export(self.model)

  def forward(
      self,
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    b, s = input_ids.shape
    mask_val = -1e9
    q_idx = torch.arange(s, dtype=torch.int32, device=input_ids.device).view(
        1, 1, s, 1
    )
    k_idx = torch.arange(s, dtype=torch.int32, device=input_ids.device).view(
        1, 1, 1, s
    )
    causal = (k_idx > q_idx).to(torch.float32) * mask_val
    if attention_mask is not None:
      pad_mask = (
          torch.ones_like(attention_mask, dtype=torch.float32)
          - attention_mask.to(torch.float32)
      ).view(b, 1, 1, s) * mask_val
      mask_4d = causal + pad_mask
    else:
      mask_4d = causal

    outputs = self.model(
        input_ids=input_ids,
        attention_mask={
            'full_attention': mask_4d,
            'sliding_attention': mask_4d,
        },
        use_cache=False,
    )
    hidden_states = outputs.last_hidden_state
    if attention_mask is not None:
      hidden_states = hidden_states * attention_mask.to(
          hidden_states.dtype
      ).unsqueeze(-1)
    return hidden_states

  @classmethod
  def normalize_mllm_state_dict(
      cls,
      state_dict: dict[str, torch.Tensor],
  ) -> dict[str, torch.Tensor]:
    """Normalizes an `mllm` state_dict for `load_state_dict(strict=True)`.

    Verifies that every key belongs to either the text backbone (`model.*` or
    `model.language_model.*`) or the frozen Qwen2.5-VL vision tower / LM head
    (`visual.*`, `model.visual.*`, `lm_head.*`), raising `ValueError` on any
    unrecognized key so zero unknown checkpoint keys are silently ignored.

    Args:
      state_dict: Raw state dictionary loaded from `mllm` safetensors.

    Returns:
      Filtered state dictionary matching `OmniGen2TextEncoder`'s `model.*` keys.
    """
    normalized: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    for k, v in state_dict.items():
      if k.startswith('model.language_model.'):
        normalized['model.' + k[len('model.language_model.') :]] = v
      elif k.startswith('model.model.'):
        normalized['model.' + k[len('model.model.') :]] = v
      elif k.startswith(('visual.', 'model.visual.', 'lm_head.')):
        continue
      elif k.startswith('model.'):
        normalized[k] = v
      elif k.startswith(('embed_tokens.', 'layers.', 'norm.')):
        normalized['model.' + k] = v
      else:
        unexpected.append(k)
    if unexpected:
      raise ValueError(
          f'Unrecognized keys in OmniGen2 mllm state_dict: {unexpected[:10]}'
      )
    return normalized

  @classmethod
  def _read_init_kwargs(cls, model_dir: str) -> dict[str, Any]:
    config_path = os.path.join(model_dir, 'config.json')
    init_kwargs: dict[str, Any] = {}
    if os.path.exists(config_path):
      with open(config_path, 'r') as f:
        cfg = json.load(f)
      text_cfg = cfg.get('text_config', cfg)
      valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {
          'self'
      }
      for k in valid_params:
        if k in text_cfg:
          init_kwargs[k] = text_cfg[k]
        elif k in cfg:
          init_kwargs[k] = cfg[k]
    return init_kwargs

  @classmethod
  def from_pretrained(
      cls,
      pretrained_model_name_or_path: str,
      subfolder: str | None = 'mllm',
      torch_dtype: torch.dtype = torch.float32,
      use_random_weights: bool = False,
  ) -> OmniGen2TextEncoder:
    """Loads `OmniGen2TextEncoder` from an `OmniGen2/OmniGen2` checkpoint."""
    model_dir = pretrained_model_name_or_path
    if subfolder:
      candidate_dir = os.path.join(pretrained_model_name_or_path, subfolder)
      if os.path.exists(candidate_dir):
        model_dir = candidate_dir
    init_kwargs = cls._read_init_kwargs(model_dir)
    model = cls(**init_kwargs)
    if not use_random_weights:
      raw_sd = load_safetensors_state_dict(model_dir)
      if not raw_sd:
        raise FileNotFoundError(
            'No safetensors weights found for OmniGen2TextEncoder in '
            f'{model_dir}.'
        )
      norm_sd = cls.normalize_mllm_state_dict(raw_sd)
      model.load_state_dict(norm_sd, strict=True)
    return model.to(torch_dtype)


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
  """Combined timestep and caption embedding (`time_caption_embed`) in OmniGen2."""

  def __init__(
      self,
      hidden_size: int = 2520,
      text_feat_dim: int = 2048,
      norm_eps: float = 1e-5,
      timestep_scale: float = 1000.0,
  ):
    super().__init__()
    self.time_proj = modeling_flux2.Timesteps(
        num_channels=TIMESTEP_CHANNELS,
        flip_sin_to_cos=True,
        downscale_freq_shift=0.0,
        scale=int(timestep_scale),
    )
    self.timestep_embedder = modeling_flux2.TimestepEmbedding(
        in_channels=TIMESTEP_CHANNELS,
        time_embed_dim=min(hidden_size, CONDITIONING_EMBED_DIM),
    )
    self.caption_embedder = nn.Sequential(
        nn.RMSNorm(text_feat_dim, eps=norm_eps),
        nn.Linear(text_feat_dim, hidden_size, bias=True),
    )

  def forward(
      self,
      timestep: torch.Tensor,
      text_hidden_states: torch.Tensor,
      dtype: torch.dtype,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    time_proj = self.time_proj(timestep)
    time_embed = self.timestep_embedder(time_proj.to(dtype=dtype))
    caption_embed = self.caption_embedder(text_hidden_states)
    return time_embed, caption_embed


class LuminaRMSNormZero(nn.Module):
  """AdaLN-Zero RMSNorm (`norm1` on modulated OmniGen2TransformerBlock)."""

  def __init__(
      self,
      embedding_dim: int = 2520,
      norm_eps: float = 1e-5,
  ):
    super().__init__()
    self.silu = nn.SiLU()
    self.linear = nn.Linear(
        min(embedding_dim, CONDITIONING_EMBED_DIM),
        4 * embedding_dim,
        bias=True,
    )
    self.norm = nn.RMSNorm(embedding_dim, eps=norm_eps)

  def forward(
      self,
      x: torch.Tensor,
      emb: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    emb = self.linear(self.silu(emb))
    scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
    x = self.norm(x) * (1.0 + scale_msa[:, None])
    return x, gate_msa, scale_mlp, gate_mlp


class LuminaLayerNormContinuous(nn.Module):
  """Continuous conditioning output layer norm (`norm_out` in OmniGen2)."""

  def __init__(
      self,
      embedding_dim: int = 2520,
      conditioning_embedding_dim: int = CONDITIONING_EMBED_DIM,
      elementwise_affine: bool = False,
      eps: float = 1e-6,
      bias: bool = True,
      out_dim: int | None = None,
  ):
    super().__init__()
    self.silu = nn.SiLU()
    self.linear_1 = nn.Linear(
        conditioning_embedding_dim, embedding_dim, bias=bias
    )
    self.norm = nn.LayerNorm(
        embedding_dim, eps=eps, elementwise_affine=elementwise_affine
    )
    self.linear_2 = (
        nn.Linear(embedding_dim, out_dim, bias=bias)
        if out_dim is not None
        else None
    )

  def forward(
      self,
      x: torch.Tensor,
      conditioning_embedding: torch.Tensor,
  ) -> torch.Tensor:
    emb = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
    x = self.norm(x) * (1.0 + emb)[:, None, :]
    if self.linear_2 is not None:
      x = self.linear_2(x)
    return x


class LuminaFeedForward(nn.Module):
  """SwiGLU feed-forward network (`linear_1`, `linear_2`, `linear_3`) in OmniGen2."""

  def __init__(
      self,
      dim: int = 2520,
      inner_dim: int | None = None,
      multiple_of: int = 256,
      ffn_dim_multiplier: float | None = None,
  ):
    super().__init__()
    if inner_dim is None:
      inner_dim = 4 * dim
    if ffn_dim_multiplier is not None:
      inner_dim = int(ffn_dim_multiplier * inner_dim)
    inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)
    self.inner_dim = inner_dim

    self.linear_1 = nn.Linear(dim, inner_dim, bias=False)
    self.linear_2 = nn.Linear(inner_dim, dim, bias=False)
    self.linear_3 = nn.Linear(dim, inner_dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_2(F.silu(self.linear_1(x)) * self.linear_3(x))


class OmniGen2Attention(nn.Module):
  """GQA self-attention with RMSNorm Q/K and 3D Omni-RoPE (`attn` in OmniGen2)."""

  def __init__(
      self,
      hidden_size: int = 2520,
      num_attention_heads: int = 21,
      num_kv_heads: int = 7,
      eps: float = 1e-5,
  ):
    super().__init__()
    self.heads = num_attention_heads
    self.kv_heads = num_kv_heads
    self.head_dim = hidden_size // num_attention_heads
    self.num_kv_groups = num_attention_heads // num_kv_heads
    self.scale = 1.0 / math.sqrt(self.head_dim)

    self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
    self.to_k = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
    self.to_v = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
    self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
    self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
    self.to_out = nn.ModuleList([
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.Dropout(0.0),
    ])

  def forward(
      self,
      hidden_states: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    b, s, _ = hidden_states.shape
    q = self.to_q(hidden_states).view(b, s, self.heads, self.head_dim)
    k = self.to_k(hidden_states).view(b, s, self.kv_heads, self.head_dim)
    v = self.to_v(hidden_states).view(b, s, self.kv_heads, self.head_dim)

    q = self.norm_q(q)
    k = self.norm_k(k)

    q = modeling_flux2.apply_rotary_emb(q, (cos, sin), sequence_dim=1)
    k = modeling_flux2.apply_rotary_emb(k, (cos, sin), sequence_dim=1)

    q = q.transpose(1, 2)
    k = bonsai_flux2._repeat_kv_4d(k.transpose(1, 2), self.num_kv_groups)  # pylint: disable=protected-access
    v = bonsai_flux2._repeat_kv_4d(v.transpose(1, 2), self.num_kv_groups)  # pylint: disable=protected-access

    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    if attention_mask is not None:
      if attention_mask.dtype == torch.bool:
        scores = scores.masked_fill(~attention_mask.view(b, 1, 1, s), -1e9)
      elif attention_mask.ndim == 2:
        scores = (
            scores
            + (1.0 - attention_mask.to(scores.dtype)).view(b, 1, 1, s) * -1e9
        )
      else:
        scores = scores + attention_mask.to(scores.dtype)
    probs = F.softmax(scores.float(), dim=-1).to(v.dtype)
    out = (
        torch.matmul(probs, v)
        .transpose(1, 2)
        .reshape(b, s, self.heads * self.head_dim)
    )
    out = self.to_out[0](out)
    out = self.to_out[1](out)
    return out


class OmniGen2TransformerBlock(nn.Module):
  """Transformer block for `context_refiner`, `noise_refiner`, `ref_image_refiner`, and `layers`."""

  def __init__(
      self,
      hidden_size: int = 2520,
      num_attention_heads: int = 21,
      num_kv_heads: int = 7,
      multiple_of: int = 256,
      ffn_dim_multiplier: float | None = None,
      norm_eps: float = 1e-5,
      modulation: bool = True,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.modulation = modulation

    if modulation:
      self.norm1 = LuminaRMSNormZero(hidden_size, norm_eps=norm_eps)
    else:
      self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)

    self.attn = OmniGen2Attention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        eps=norm_eps,
    )
    self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
    self.ffn_norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
    self.feed_forward = LuminaFeedForward(
        dim=hidden_size,
        multiple_of=multiple_of,
        ffn_dim_multiplier=ffn_dim_multiplier,
    )
    self.ffn_norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)

  def forward(
      self,
      hidden_states: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      temb: torch.Tensor | None = None,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if self.modulation:
      if temb is None:
        raise ValueError('temb must be provided when modulation is enabled.')
      norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(
          hidden_states, temb
      )
      attn_output = self.attn(
          norm_hidden_states,
          cos=cos,
          sin=sin,
          attention_mask=attention_mask,
      )
      hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(
          attn_output
      )
      mlp_output = self.feed_forward(
          self.ffn_norm1(hidden_states) * (1.0 + scale_mlp.unsqueeze(1))
      )
      hidden_states = hidden_states + gate_mlp.unsqueeze(
          1
      ).tanh() * self.ffn_norm2(mlp_output)
    else:
      norm_hidden_states = self.norm1(hidden_states)
      attn_output = self.attn(
          norm_hidden_states,
          cos=cos,
          sin=sin,
          attention_mask=attention_mask,
      )
      hidden_states = hidden_states + self.norm2(attn_output)
      mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
      hidden_states = hidden_states + self.ffn_norm2(mlp_output)
    return hidden_states


class OmniGen2Transformer2DModel(nn.Module):
  """OmniGen2 Transformer 2D Model matching all 582 `OmniGen2/OmniGen2` state_dict keys."""

  def __init__(
      self,
      patch_size: int = 2,
      in_channels: int = 16,
      out_channels: int | None = None,
      hidden_size: int = 2520,
      num_layers: int = 32,
      num_refiner_layers: int = 2,
      num_attention_heads: int = 21,
      num_kv_heads: int = 7,
      multiple_of: int = 256,
      ffn_dim_multiplier: float | None = None,
      norm_eps: float = 1e-5,
      axes_dim_rope: tuple[int, int, int] | list[int] = (40, 40, 40),
      axes_lens: tuple[int, int, int] | list[int] = (1024, 1664, 1664),
      text_feat_dim: int = 2048,
      timestep_scale: float = 1000.0,
      rope_theta: float = 10000.0,
  ):
    super().__init__()
    head_dim = hidden_size // num_attention_heads
    if head_dim != sum(axes_dim_rope):
      raise ValueError(
          f'hidden_size // num_attention_heads ({head_dim}) '
          f'must equal sum(axes_dim_rope) ({sum(axes_dim_rope)}).'
      )
    self.patch_size = patch_size
    self.in_channels = in_channels
    self.out_channels = out_channels or in_channels
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_refiner_layers = num_refiner_layers
    self.num_attention_heads = num_attention_heads
    self.num_kv_heads = num_kv_heads
    self.multiple_of = multiple_of
    self.ffn_dim_multiplier = ffn_dim_multiplier
    self.norm_eps = norm_eps
    self.axes_dim_rope = (
        int(axes_dim_rope[0]),
        int(axes_dim_rope[1]),
        int(axes_dim_rope[2]),
    )
    self.axes_lens = (
        int(axes_lens[0]),
        int(axes_lens[1]),
        int(axes_lens[2]),
    )
    self.text_feat_dim = text_feat_dim
    self.timestep_scale = timestep_scale
    self.rope_theta = rope_theta
    self.temb_dim = min(hidden_size, CONDITIONING_EMBED_DIM)
    self.packed_channels = patch_size * patch_size * in_channels

    self.config = types.SimpleNamespace(
        patch_size=patch_size,
        in_channels=in_channels,
        out_channels=self.out_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_refiner_layers=num_refiner_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        multiple_of=multiple_of,
        ffn_dim_multiplier=ffn_dim_multiplier,
        norm_eps=norm_eps,
        axes_dim_rope=self.axes_dim_rope,
        axes_lens=self.axes_lens,
        text_feat_dim=text_feat_dim,
        timestep_scale=timestep_scale,
        rope_theta=rope_theta,
    )

    self.x_embedder = nn.Linear(
        in_features=patch_size * patch_size * in_channels,
        out_features=hidden_size,
        bias=True,
    )
    self.ref_image_patch_embedder = nn.Linear(
        in_features=patch_size * patch_size * in_channels,
        out_features=hidden_size,
        bias=True,
    )
    self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
        hidden_size=hidden_size,
        text_feat_dim=text_feat_dim,
        norm_eps=norm_eps,
        timestep_scale=timestep_scale,
    )
    self.noise_refiner = nn.ModuleList([
        OmniGen2TransformerBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            modulation=True,
        )
        for _ in range(num_refiner_layers)
    ])
    self.ref_image_refiner = nn.ModuleList([
        OmniGen2TransformerBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            modulation=True,
        )
        for _ in range(num_refiner_layers)
    ])
    self.context_refiner = nn.ModuleList([
        OmniGen2TransformerBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            modulation=False,
        )
        for _ in range(num_refiner_layers)
    ])
    self.layers = nn.ModuleList([
        OmniGen2TransformerBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            modulation=True,
        )
        for _ in range(num_layers)
    ])
    self.norm_out = LuminaLayerNormContinuous(
        embedding_dim=hidden_size,
        conditioning_embedding_dim=min(hidden_size, CONDITIONING_EMBED_DIM),
        elementwise_affine=False,
        eps=1e-6,
        bias=True,
        out_dim=patch_size * patch_size * self.out_channels,
    )
    self.image_index_embedding = nn.Parameter(
        torch.randn(MAX_REF_IMAGES, hidden_size)
    )

  def embed_cap(self, text_hidden_states: torch.Tensor) -> torch.Tensor:
    """Projects text conditioning features (`time_caption_embed.caption_embedder`)."""
    return self.time_caption_embed.caption_embedder(text_hidden_states)

  def embed_x(
      self,
      packed_latents: torch.Tensor,
      timestep: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Projects patchified image latents and computes timestep conditioning `temb`."""
    x_emb = self.x_embedder(packed_latents)
    t_proj = self.time_caption_embed.time_proj(timestep)
    temb = self.time_caption_embed.timestep_embedder(
        t_proj.to(dtype=packed_latents.dtype)
    )
    return x_emb, temb

  def embed_and_refine_ref_image(
      self,
      packed_ref_latents: torch.Tensor,
      ref_cos: torch.Tensor,
      ref_sin: torch.Tensor,
      temb: torch.Tensor,
      image_index: int = 0,
  ) -> torch.Tensor:
    """Embeds and refines a patchified reference image latent sequence."""
    ref_hidden = self.ref_image_patch_embedder(packed_ref_latents)
    ref_hidden = ref_hidden + self.image_index_embedding[image_index].view(
        1, 1, -1
    )
    for layer in self.ref_image_refiner:
      ref_hidden = layer(ref_hidden, cos=ref_cos, sin=ref_sin, temb=temb)
    return ref_hidden

  def forward(
      self,
      latents: torch.Tensor,
      timestep: torch.Tensor,
      text_hidden_states: torch.Tensor,
      text_attention_mask: torch.Tensor | None = None,
      ref_image_hidden_states: list[list[torch.Tensor]] | None = None,
  ) -> torch.Tensor:
    """Full reference-compatible forward pass of `OmniGen2Transformer2DModel`."""
    b, c, h, w = latents.shape
    if b != 1:
      raise ValueError(f'OmniGen2Transformer2DModel expects batch=1, got {b}.')
    p = self.patch_size
    grid_h, grid_w = h // p, w // p
    img_tokens = grid_h * grid_w
    seq_len = text_hidden_states.shape[1]

    if text_attention_mask is None:
      text_attention_mask = torch.ones(
          (1, seq_len), dtype=torch.bool, device=latents.device
      )
    active_cap_len = int(text_attention_mask[0].sum().item())

    temb, cap_hidden = self.time_caption_embed(
        timestep, text_hidden_states, latents.dtype
    )

    cap_ids = torch.zeros(
        (seq_len, 3), dtype=torch.float32, device=latents.device
    )
    if active_cap_len > 0:
      arange_cap = torch.arange(
          active_cap_len, dtype=torch.float32, device=latents.device
      )
      cap_ids[:active_cap_len] = arange_cap.unsqueeze(-1).expand(
          active_cap_len, 3
      )
    cap_cos, cap_sin = build_omnigen2_3d_rope_cos_sin(
        cap_ids, axes_dim_rope=self.axes_dim_rope, theta=self.rope_theta
    )
    if active_cap_len < seq_len:
      cap_cos = cap_cos.clone()
      cap_sin = cap_sin.clone()
      cap_cos[active_cap_len:] = 0.0
      cap_sin[active_cap_len:] = 0.0
    cap_cos_4d = cap_cos.unsqueeze(0).unsqueeze(2)
    cap_sin_4d = cap_sin.unsqueeze(0).unsqueeze(2)

    cap_mask_4d = (1.0 - text_attention_mask.to(latents.dtype)).view(
        1, 1, 1, seq_len
    ) * -1e9
    for layer in self.context_refiner:
      cap_hidden = layer(
          cap_hidden,
          cos=cap_cos_4d,
          sin=cap_sin_4d,
          attention_mask=cap_mask_4d,
      )

    pe_shift = active_cap_len
    ref_tokens_list: list[torch.Tensor] = []
    ref_cos_list: list[torch.Tensor] = []
    ref_sin_list: list[torch.Tensor] = []
    if ref_image_hidden_states and ref_image_hidden_states[0]:
      for j, ref_img in enumerate(ref_image_hidden_states[0]):
        _, ref_h, ref_w = ref_img.shape
        ref_gh, ref_gw = ref_h // p, ref_w // p
        packed_ref = (
            ref_img.unsqueeze(0)
            .view(1, c, ref_gh, p, ref_gw, p)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(1, ref_gh * ref_gw, p * p * c)
        )
        rh = torch.arange(ref_gh, dtype=torch.float32, device=latents.device)
        rw = torch.arange(ref_gw, dtype=torch.float32, device=latents.device)
        r_grid_h, r_grid_w = torch.meshgrid(rh, rw, indexing='ij')
        r_t = torch.full_like(r_grid_h, float(pe_shift))
        ref_ids = torch.stack([r_t, r_grid_h, r_grid_w], dim=-1).reshape(-1, 3)
        r_cos, r_sin = build_omnigen2_3d_rope_cos_sin(
            ref_ids, axes_dim_rope=self.axes_dim_rope, theta=self.rope_theta
        )
        r_cos_4d = r_cos.unsqueeze(0).unsqueeze(2)
        r_sin_4d = r_sin.unsqueeze(0).unsqueeze(2)
        ref_refined = self.embed_and_refine_ref_image(
            packed_ref, r_cos_4d, r_sin_4d, temb, image_index=j
        )
        ref_tokens_list.append(ref_refined)
        ref_cos_list.append(r_cos_4d)
        ref_sin_list.append(r_sin_4d)
        pe_shift += max(ref_gh, ref_gw)

    packed_x = (
        latents.view(1, c, grid_h, p, grid_w, p)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .view(1, img_tokens, p * p * c)
    )
    ih = torch.arange(grid_h, dtype=torch.float32, device=latents.device)
    iw = torch.arange(grid_w, dtype=torch.float32, device=latents.device)
    i_grid_h, i_grid_w = torch.meshgrid(ih, iw, indexing='ij')
    i_t = torch.full_like(i_grid_h, float(pe_shift))
    img_ids = torch.stack([i_t, i_grid_h, i_grid_w], dim=-1).reshape(-1, 3)
    img_cos, img_sin = build_omnigen2_3d_rope_cos_sin(
        img_ids, axes_dim_rope=self.axes_dim_rope, theta=self.rope_theta
    )
    img_cos_4d = img_cos.unsqueeze(0).unsqueeze(2)
    img_sin_4d = img_sin.unsqueeze(0).unsqueeze(2)

    x_hidden = self.x_embedder(packed_x)
    for layer in self.noise_refiner:
      x_hidden = layer(x_hidden, cos=img_cos_4d, sin=img_sin_4d, temb=temb)

    joint_parts = [cap_hidden] + ref_tokens_list + [x_hidden]
    joint_cos_parts = [cap_cos_4d] + ref_cos_list + [img_cos_4d]
    joint_sin_parts = [cap_sin_4d] + ref_sin_list + [img_sin_4d]

    joint_hidden = torch.cat(joint_parts, dim=1)
    joint_cos = torch.cat(joint_cos_parts, dim=1)
    joint_sin = torch.cat(joint_sin_parts, dim=1)

    extra_tokens = joint_hidden.shape[1] - seq_len
    joint_mask_2d = torch.cat(
        [
            text_attention_mask.to(latents.dtype),
            torch.ones(
                (1, extra_tokens), dtype=latents.dtype, device=latents.device
            ),
        ],
        dim=1,
    )
    joint_mask_4d = (torch.ones_like(joint_mask_2d) - joint_mask_2d).view(
        1, 1, 1, -1
    ) * -1e9

    for layer in self.layers:
      joint_hidden = layer(
          joint_hidden,
          cos=joint_cos,
          sin=joint_sin,
          temb=temb,
          attention_mask=joint_mask_4d,
      )

    out_hidden = self.norm_out(joint_hidden, temb)
    img_out = out_hidden[:, -img_tokens:, :]
    return (
        img_out.view(1, grid_h, grid_w, p, p, self.out_channels)
        .permute(0, 5, 1, 3, 2, 4)
        .contiguous()
        .view(1, self.out_channels, h, w)
    )

  @classmethod
  def _read_init_kwargs(cls, model_dir: str) -> dict[str, Any]:
    config_path = os.path.join(model_dir, 'config.json')
    init_kwargs: dict[str, Any] = {}
    if os.path.exists(config_path):
      with open(config_path, 'r') as f:
        cfg = json.load(f)
      valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {
          'self'
      }
      init_kwargs = {k: v for k, v in cfg.items() if k in valid_params}
    return init_kwargs

  @classmethod
  def from_pretrained(
      cls,
      pretrained_model_name_or_path: str,
      subfolder: str | None = None,
      torch_dtype: torch.dtype = torch.float32,
      use_random_weights: bool = False,
      **kwargs: Any,
  ) -> OmniGen2Transformer2DModel:
    """Loads `OmniGen2Transformer2DModel` from a local directory with `strict=True`."""
    del kwargs
    model_dir = (
        os.path.join(pretrained_model_name_or_path, subfolder)
        if subfolder
        else pretrained_model_name_or_path
    )
    init_kwargs = cls._read_init_kwargs(model_dir)
    state_dict: dict[str, torch.Tensor] = {}
    if not use_random_weights:
      state_dict = load_safetensors_state_dict(model_dir)
      if not state_dict:
        raise FileNotFoundError(
            'No safetensors weights found for OmniGen2Transformer2DModel in'
            f' {model_dir}.'
        )
    model = cls(**init_kwargs)
    if state_dict:
      model.load_state_dict(state_dict, strict=True)
    return model.to(torch_dtype)


class OmniGen2AutoencoderKL(modeling_flux2.AutoencoderKLFlux2):
  """16-channel AutoencoderKL for OmniGen2, reusing `AutoencoderKLFlux2` without BN.

  Matches `OmniGen2/OmniGen2/vae` (`latent_channels=16`,
  `scaling_factor=0.3611`, `shift_factor=0.1159`, `use_quant_conv=False`,
  `use_post_quant_conv=False`, 244 state_dict tensors).
  """

  def __init__(
      self,
      *args: Any,
      down_block_types: tuple[str, ...] | list[str] | None = None,
      up_block_types: tuple[str, ...] | list[str] | None = None,
      block_out_channels: tuple[int, ...] | list[int] = (128, 256, 512, 512),
      latent_channels: int = 16,
      scaling_factor: float = 0.3611,
      shift_factor: float = 0.1159,
      use_quant_conv: bool = False,
      use_post_quant_conv: bool = False,
      **kwargs: Any,
  ):
    if down_block_types is None or len(down_block_types) != len(
        block_out_channels
    ):
      down_block_types = ('DownEncoderBlock2D',) * len(block_out_channels)
    if up_block_types is None or len(up_block_types) != len(block_out_channels):
      up_block_types = ('UpDecoderBlock2D',) * len(block_out_channels)
    super().__init__(
        *args,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        latent_channels=latent_channels,
        use_quant_conv=use_quant_conv,
        use_post_quant_conv=use_post_quant_conv,
        **kwargs,
    )
    self.config.scaling_factor = scaling_factor
    self.config.shift_factor = shift_factor
    del self.bn

  @classmethod
  def _read_init_kwargs(cls, model_dir: str) -> dict[str, Any]:
    config_path = os.path.join(model_dir, 'config.json')
    init_kwargs: dict[str, Any] = {}
    if os.path.exists(config_path):
      with open(config_path, 'r') as f:
        cfg = json.load(f)
      valid_params = (
          set(
              inspect.signature(
                  modeling_flux2.AutoencoderKLFlux2.__init__
              ).parameters.keys()
          )
          | set(inspect.signature(cls.__init__).parameters.keys())
      ) - {'self', 'args', 'kwargs'}
      init_kwargs = {k: v for k, v in cfg.items() if k in valid_params}
    return init_kwargs

  @classmethod
  def from_pretrained(
      cls,
      pretrained_model_name_or_path: str,
      subfolder: str | None = None,
      torch_dtype: torch.dtype = torch.float32,
      use_random_weights: bool = False,
      **kwargs: Any,
  ) -> OmniGen2AutoencoderKL:
    """Loads `OmniGen2AutoencoderKL` from a local directory with `strict=True`."""
    del kwargs
    model_dir = (
        os.path.join(pretrained_model_name_or_path, subfolder)
        if subfolder
        else pretrained_model_name_or_path
    )
    init_kwargs = cls._read_init_kwargs(model_dir)
    state_dict: dict[str, torch.Tensor] = {}
    if not use_random_weights:
      state_dict = load_safetensors_state_dict(model_dir)
      if not state_dict:
        raise FileNotFoundError(
            'No safetensors weights found for OmniGen2AutoencoderKL in'
            f' {model_dir}.'
        )
    model = cls(**init_kwargs)
    if state_dict:
      model.load_state_dict(state_dict, strict=True)
    return model.to(torch_dtype)


def verify_checkpoint_keys(model_dir: str) -> dict[str, int]:
  """Verifies 100% of checkpoint safetensors keys across mllm, transformer, and vae.

  Uses `meta`-device instantiation and safetensors header inspection so full
  weights are not materialized in RAM. Raises `ValueError` on any missing or
  unconsumed/unrecognized key.

  Args:
    model_dir: Root directory of an OmniGen2 checkpoint.

  Returns:
    Mapping from component name (`mllm`, `transformer`, `vae`) to verified key
    count.
  """
  mllm_sub = 'mllm'
  if not os.path.exists(os.path.join(model_dir, 'mllm')) and os.path.exists(
      os.path.join(model_dir, 'text_encoder')
  ):
    mllm_sub = 'text_encoder'
  mllm_dir = os.path.join(model_dir, mllm_sub)
  dit_dir = os.path.join(model_dir, 'transformer')
  vae_dir = os.path.join(model_dir, 'vae')

  mllm_raw_keys = _read_safetensors_keys(mllm_dir)
  placeholder = torch.empty(0, device='meta')
  mllm_norm = OmniGen2TextEncoder.normalize_mllm_state_dict(
      {k: placeholder for k in mllm_raw_keys}
  )
  with torch.device('meta'):
    te_meta = OmniGen2TextEncoder(
        **OmniGen2TextEncoder._read_init_kwargs(mllm_dir)  # pylint: disable=protected-access
    )
    dit_meta = OmniGen2Transformer2DModel(
        **OmniGen2Transformer2DModel._read_init_kwargs(dit_dir)  # pylint: disable=protected-access
    )
    vae_meta = OmniGen2AutoencoderKL(
        **OmniGen2AutoencoderKL._read_init_kwargs(vae_dir)  # pylint: disable=protected-access
    )

  counts: dict[str, int] = {}
  for name, expected, actual in (
      ('mllm', set(te_meta.state_dict().keys()), set(mllm_norm.keys())),
      (
          'transformer',
          set(dit_meta.state_dict().keys()),
          _read_safetensors_keys(dit_dir),
      ),
      (
          'vae',
          set(vae_meta.state_dict().keys()),
          _read_safetensors_keys(vae_dir),
      ),
  ):
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
      raise ValueError(
          f'OmniGen2 checkpoint key mismatch in {name}: '
          f'missing={missing[:10]} unexpected={unexpected[:10]}'
      )
    counts[name] = len(expected)
  return counts
