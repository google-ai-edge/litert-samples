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
"""Z-Image-Turbo (S3-DiT + AutoencoderKL) PyTorch modules for LiteRT export.

Matches the exact state_dict structure of `Tongyi-MAI/Z-Image-Turbo`
(`ZImageTransformer2DModel` and `AutoencoderKL`) while keeping all intermediate
tensors rank <= 4 and using real-valued 3D axial RoPE (`cos`/`sin`) for mobile
GPU compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import json
import math
import os
import types
from typing import Any, TypeVar

from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import modeling_flux2
import torch
from torch import nn
import torch.nn.functional as F

ADALN_EMBED_DIM = 256
_ModuleT = TypeVar('_ModuleT', bound=nn.Module)


def _load_module_from_pretrained(
    cls: type[_ModuleT],
    pretrained_model_name_or_path: str,
    *,
    param_classes: Sequence[type[Any]],
    subfolder: str | None = None,
    torch_dtype: torch.dtype = torch.float32,
    use_random_weights: bool = False,
    state_dict_override: Mapping[str, torch.Tensor] | None = None,
    config_override: Mapping[str, Any] | None = None,
) -> _ModuleT:
  """Loads `cls` from a local checkpoint directory or state_dict override."""
  model_dir = (
      os.path.join(pretrained_model_name_or_path, subfolder)
      if subfolder
      else pretrained_model_name_or_path
  )
  cfg: Mapping[str, Any] = {}
  if config_override is not None:
    cfg = config_override
  else:
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
      with open(config_path, 'r') as f:
        cfg = json.load(f)
  valid_params: set[str] = set()
  for param_cls in param_classes:
    valid_params |= set(inspect.signature(param_cls.__init__).parameters.keys())
  valid_params -= {'self', 'args', 'kwargs'}
  init_kwargs = {k: v for k, v in cfg.items() if k in valid_params}
  state_dict: Mapping[str, torch.Tensor] = {}
  if not use_random_weights:
    if state_dict_override is not None:
      state_dict = state_dict_override
    else:
      state_dict = modeling_flux2._load_safetensors_state_dict(model_dir)  # pylint: disable=protected-access
    if not state_dict:
      raise FileNotFoundError(
          f'No safetensors weights found for {cls.__name__} in {model_dir}.'
      )
  model = cls(**init_kwargs)
  if state_dict:
    model.load_state_dict(state_dict, strict=True)
  return model.to(torch_dtype)


class ZImageTimestepEmbedder(nn.Module):
  """Sinusoidal timestep embedder with 2-layer MLP (`t_embedder`)."""

  def __init__(
      self,
      out_size: int,
      mid_size: int | None = None,
      frequency_embedding_size: int = 256,
  ):
    super().__init__()
    if mid_size is None:
      mid_size = out_size
    self.mlp = nn.Sequential(
        nn.Linear(frequency_embedding_size, mid_size, bias=True),
        nn.SiLU(),
        nn.Linear(mid_size, out_size, bias=True),
    )
    self.frequency_embedding_size = frequency_embedding_size

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    t_freq = modeling_flux2.get_timestep_embedding(
        t,
        self.frequency_embedding_size,
        flip_sin_to_cos=True,
        downscale_freq_shift=0.0,
    )
    return self.mlp(t_freq.to(t.dtype))


class ZImageFeedForward(nn.Module):
  """SwiGLU feed-forward network (`w1`, `w2`, `w3`) matching Z-Image."""

  def __init__(self, dim: int, hidden_dim: int):
    super().__init__()
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageAttention(nn.Module):
  """Self-attention module with RMSNorm Q/K normalization and 3D RoPE."""

  def __init__(
      self,
      dim: int,
      n_heads: int,
      norm_eps: float = 1e-5,
      qk_norm: bool = True,
  ):
    super().__init__()
    self.heads = n_heads
    self.head_dim = dim // n_heads
    self.to_q = nn.Linear(dim, dim, bias=False)
    self.to_k = nn.Linear(dim, dim, bias=False)
    self.to_v = nn.Linear(dim, dim, bias=False)
    self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
    self.norm_q = nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None
    self.norm_k = nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None

  def forward(
      self,
      x: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
  ) -> torch.Tensor:
    b, s, _ = x.shape
    q = self.to_q(x).view(b, s, self.heads, self.head_dim)
    k = self.to_k(x).view(b, s, self.heads, self.head_dim)
    v = self.to_v(x).view(b, s, self.heads, self.head_dim)

    if self.norm_q is not None:
      q = self.norm_q(q)
    if self.norm_k is not None:
      k = self.norm_k(k)

    q = modeling_flux2.apply_rotary_emb(q, (cos, sin), sequence_dim=1)
    k = modeling_flux2.apply_rotary_emb(k, (cos, sin), sequence_dim=1)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scale = 1.0 / math.sqrt(self.head_dim)
    key_norm = cos[:, :, :, :2].abs().sum(dim=-1, keepdim=True) + sin[
        :, :, :, :2
    ].abs().sum(dim=-1, keepdim=True)
    attn_bias = torch.where(key_norm > 0, 0.0, -1e4).permute(0, 2, 3, 1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale + attn_bias
    probs = F.softmax(scores.float(), dim=-1).to(v.dtype)
    out = torch.matmul(probs, v).transpose(1, 2).reshape(b, s, -1)
    return self.to_out[0](out)


class ZImageTransformerBlock(nn.Module):
  """Single-stream transformer block for Z-Image refiner and main layers."""

  def __init__(
      self,
      layer_id: int,
      dim: int,
      n_heads: int,
      n_kv_heads: int,
      norm_eps: float = 1e-5,
      qk_norm: bool = True,
      modulation: bool = True,
  ):
    super().__init__()
    del layer_id, n_kv_heads
    hidden_dim = int(dim / 3 * 8)
    self.dim = dim
    self.modulation = modulation

    self.attention = ZImageAttention(
        dim=dim,
        n_heads=n_heads,
        norm_eps=norm_eps,
        qk_norm=qk_norm,
    )
    self.feed_forward = ZImageFeedForward(dim=dim, hidden_dim=hidden_dim)

    self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
    self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
    self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
    self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)

    if modulation:
      self.adaLN_modulation = nn.ModuleList([  # pylint: disable=invalid-name
          nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
      ])

  def forward(
      self,
      x: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      adaln_input: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if self.modulation:
      if adaln_input is None:
        raise ValueError('adaln_input is required when modulation is enabled.')
      scale_msa, gate_msa, scale_mlp, gate_mlp = (
          self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
      )
      gate_msa = gate_msa.tanh()
      gate_mlp = gate_mlp.tanh()
      scale_msa = 1.0 + scale_msa
      scale_mlp = 1.0 + scale_mlp

      attn_out = self.attention(
          self.attention_norm1(x) * scale_msa,
          cos=cos,
          sin=sin,
      )
      x = x + gate_msa * self.attention_norm2(attn_out)
      x = x + gate_mlp * self.ffn_norm2(
          self.feed_forward(self.ffn_norm1(x) * scale_mlp)
      )
    else:
      attn_out = self.attention(self.attention_norm1(x), cos=cos, sin=sin)
      x = x + self.attention_norm2(attn_out)
      x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
    return x


class ZImageFinalLayer(nn.Module):
  """Final AdaLN + linear projection layer for Z-Image (`all_final_layer`)."""

  def __init__(self, hidden_size: int, out_channels: int):
    super().__init__()
    self.norm_final = nn.LayerNorm(
        hidden_size, elementwise_affine=False, eps=1e-6
    )
    self.linear = nn.Linear(hidden_size, out_channels, bias=True)
    self.adaLN_modulation = nn.Sequential(  # pylint: disable=invalid-name
        nn.SiLU(),
        nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
    )

  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    scale = 1.0 + self.adaLN_modulation(c).unsqueeze(1)
    x = self.norm_final(x) * scale
    return self.linear(x)


class ZImageTransformer2DModel(nn.Module):
  """Z-Image-Turbo S3-DiT model matching `Tongyi-MAI/Z-Image-Turbo` weights."""

  def __init__(
      self,
      all_patch_size: tuple[int, ...] | list[int] = (2,),
      all_f_patch_size: tuple[int, ...] | list[int] = (1,),
      in_channels: int = 16,
      dim: int = 3840,
      n_layers: int = 30,
      n_refiner_layers: int = 2,
      n_heads: int = 30,
      n_kv_heads: int = 30,
      norm_eps: float = 1e-5,
      qk_norm: bool = True,
      cap_feat_dim: int = 2560,
      rope_theta: float = 256.0,
      t_scale: float = 1000.0,
      axes_dims: tuple[int, int, int] | list[int] = (32, 48, 48),
      axes_lens: tuple[int, int, int] | list[int] = (1536, 512, 512),
  ):
    super().__init__()
    self.all_patch_size = tuple(all_patch_size)
    self.all_f_patch_size = tuple(all_f_patch_size)
    self.in_channels = in_channels
    self.dim = dim
    self.n_layers = n_layers
    self.n_refiner_layers = n_refiner_layers
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.norm_eps = norm_eps
    self.qk_norm = qk_norm
    self.cap_feat_dim = cap_feat_dim
    self.rope_theta = rope_theta
    self.t_scale = t_scale
    self.axes_dims = (
        int(axes_dims[0]),
        int(axes_dims[1]),
        int(axes_dims[2]),
    )
    self.axes_lens = (
        int(axes_lens[0]),
        int(axes_lens[1]),
        int(axes_lens[2]),
    )
    self.adaln_dim = min(dim, ADALN_EMBED_DIM)
    self.config = types.SimpleNamespace(
        all_patch_size=self.all_patch_size,
        all_f_patch_size=self.all_f_patch_size,
        in_channels=in_channels,
        dim=dim,
        n_layers=n_layers,
        n_refiner_layers=n_refiner_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        norm_eps=norm_eps,
        qk_norm=qk_norm,
        cap_feat_dim=cap_feat_dim,
        rope_theta=rope_theta,
        t_scale=t_scale,
        axes_dims=self.axes_dims,
        axes_lens=self.axes_lens,
    )

    self.t_embedder = ZImageTimestepEmbedder(self.adaln_dim, mid_size=1024)

    all_x_embedder = {}
    all_final_layer = {}
    for patch_size, f_patch_size in zip(
        self.all_patch_size, self.all_f_patch_size
    ):
      key = f'{patch_size}-{f_patch_size}'
      packed_channels = f_patch_size * patch_size * patch_size * in_channels
      all_x_embedder[key] = nn.Linear(packed_channels, dim, bias=True)
      all_final_layer[key] = ZImageFinalLayer(dim, packed_channels)

    self.all_x_embedder = nn.ModuleDict(all_x_embedder)
    self.all_final_layer = nn.ModuleDict(all_final_layer)

    self.noise_refiner = nn.ModuleList([
        ZImageTransformerBlock(
            1000 + layer_id,
            dim,
            n_heads,
            n_kv_heads,
            norm_eps,
            qk_norm,
            modulation=True,
        )
        for layer_id in range(n_refiner_layers)
    ])
    self.context_refiner = nn.ModuleList([
        ZImageTransformerBlock(
            layer_id,
            dim,
            n_heads,
            n_kv_heads,
            norm_eps,
            qk_norm,
            modulation=False,
        )
        for layer_id in range(n_refiner_layers)
    ])
    self.cap_embedder = nn.Sequential(
        nn.RMSNorm(cap_feat_dim, eps=norm_eps),
        nn.Linear(cap_feat_dim, dim, bias=True),
    )
    self.layers = nn.ModuleList([
        ZImageTransformerBlock(
            layer_id,
            dim,
            n_heads,
            n_kv_heads,
            norm_eps,
            qk_norm,
            modulation=True,
        )
        for layer_id in range(n_layers)
    ])

    self.x_pad_token = nn.Parameter(torch.zeros((1, dim)))
    self.cap_pad_token = nn.Parameter(torch.zeros((1, dim)))

  def embed_cap(
      self,
      cap_feats: torch.Tensor,
      cap_pad_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Projects caption features and replaces padded positions with `cap_pad_token`."""
    if cap_pad_mask is None:
      cap_pad_mask = cap_feats.abs().sum(dim=-1, keepdim=True) == 0
    elif cap_pad_mask.ndim == 2:
      cap_pad_mask = cap_pad_mask.unsqueeze(-1)
    cap_emb = self.cap_embedder(cap_feats)
    pad_tok = self.cap_pad_token.view(1, 1, -1).expand_as(cap_emb)
    return torch.where(cap_pad_mask, pad_tok, cap_emb)

  def embed_x(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
      patch_key: str = '2-1',
      x_pad_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Projects image patches and timesteps, applying `x_pad_token` if masked."""
    x_emb = self.all_x_embedder[patch_key](x)
    if x_pad_mask is not None:
      if x_pad_mask.ndim == 2:
        x_pad_mask = x_pad_mask.unsqueeze(-1)
      pad_tok = self.x_pad_token.view(1, 1, -1).expand_as(x_emb)
      x_emb = torch.where(x_pad_mask, pad_tok, x_emb)
    adaln_input = self.t_embedder(t * self.t_scale)
    return x_emb, adaln_input

  @classmethod
  def from_pretrained(
      cls,
      pretrained_model_name_or_path: str,
      subfolder: str | None = None,
      torch_dtype: torch.dtype = torch.float32,
      use_random_weights: bool = False,
      state_dict_override: Mapping[str, torch.Tensor] | None = None,
      config_override: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> ZImageTransformer2DModel:
    """Loads `ZImageTransformer2DModel` from a local directory or state dict."""
    del kwargs
    return _load_module_from_pretrained(
        cls,
        pretrained_model_name_or_path,
        param_classes=(cls,),
        subfolder=subfolder,
        torch_dtype=torch_dtype,
        use_random_weights=use_random_weights,
        state_dict_override=state_dict_override,
        config_override=config_override,
    )


class ZImageAutoencoderKL(modeling_flux2.AutoencoderKLFlux2):
  """AutoencoderKL for Z-Image, reusing Flux2 AutoencoderKL without BN."""

  def __init__(
      self,
      *args: Any,
      latent_channels: int = 16,
      scaling_factor: float = 0.3611,
      shift_factor: float = 0.1159,
      use_quant_conv: bool = False,
      use_post_quant_conv: bool = False,
      **kwargs: Any,
  ):
    super().__init__(
        *args,
        latent_channels=latent_channels,
        use_quant_conv=use_quant_conv,
        use_post_quant_conv=use_post_quant_conv,
        **kwargs,
    )
    self.config.scaling_factor = scaling_factor
    self.config.shift_factor = shift_factor
    del self.bn

  @classmethod
  def from_pretrained(
      cls,
      pretrained_model_name_or_path: str,
      subfolder: str | None = None,
      torch_dtype: torch.dtype = torch.float32,
      use_random_weights: bool = False,
      state_dict_override: Mapping[str, torch.Tensor] | None = None,
      config_override: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> ZImageAutoencoderKL:
    del kwargs
    return _load_module_from_pretrained(
        cls,
        pretrained_model_name_or_path,
        param_classes=(modeling_flux2.AutoencoderKLFlux2, cls),
        subfolder=subfolder,
        torch_dtype=torch_dtype,
        use_random_weights=use_random_weights,
        state_dict_override=state_dict_override,
        config_override=config_override,
    )
