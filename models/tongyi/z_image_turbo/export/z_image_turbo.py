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
"""Z-Image-Turbo sharded model wrappers and export pipeline for LiteRT."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from typing import Any

import litert_torch
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import quantization
from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import bonsai_flux2
from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import modeling_flux2
import safetensors
import torch
from torch import nn
import transformers

from models.tongyi.z_image_turbo import z_image_turbo_pb2
from models.tongyi.z_image_turbo.export import modeling
from litert_lm_builder.runtime.proto import text2image_metadata_pb2

ExportableModuleConfig = exportable_module_config.ExportableModuleConfig

Z_IMAGE_TURBO_EXTRA_KWARGS = (
    'max_seq_len',
    'default_steps',
)
exportable_module_config.register_extra_kwargs_keys(Z_IMAGE_TURBO_EXTRA_KWARGS)

DEFAULT_SEQ_LEN = 64
DEFAULT_IMG_SIZE = 256
DEFAULT_IN_CHANNELS = 16
DEFAULT_PATCH_SIZE = 2
DEFAULT_DIM = 3840
DEFAULT_PROMPT_DIM = 2560
DEFAULT_STEPS = 8
DEFAULT_VAE_SCALING_FACTOR = 0.3611
DEFAULT_VAE_SHIFT_FACTOR = 0.1159
DEFAULT_AXES_DIMS = (32, 48, 48)
DEFAULT_ROPE_THETA = 256.0
DEFAULT_SHIFT = 3.0
DEFAULT_QUANTIZATION_RECIPE = 'dynamic_wi4b32_afp32'
NUM_MAIN_SHARDS = 6
SEQ_MULTI_OF = 32

# Standard Text2ImageMetadata TFLiteModel section types supported by mainline
# `litertlm_builder` and `LitertLmLoader`.
Z_IMAGE_TURBO_STANDARD_SECTION_TYPES: dict[str, str] = {
    'text_encoder': 'tf_lite_text_encoder',
    'dit_single_block_0': 'tf_lite_diffusion_transformer_single_block_0',
    'dit_single_block_1': 'tf_lite_diffusion_transformer_single_block_1',
    'dit_single_block_2': 'tf_lite_diffusion_transformer_single_block_2',
    'dit_single_block_3': 'tf_lite_diffusion_transformer_single_block_3',
    'dit_final': 'tf_lite_diffusion_transformer_final',
    'vae_decoder': 'tf_lite_image_decoder',
}

# Z-Image-Turbo-specific shards packed as named `GenericBinaryData` sections
# when bundling `.litertlm` so zero upstream LiteRT-LM proto or loader
# edits are required.
Z_IMAGE_TURBO_GENERIC_BINARY_SHARDS: frozenset[str] = frozenset({
    'dit_cap_embedder',
    'dit_context_refiner',
    'dit_x_embedder',
    'dit_noise_refiner',
    'dit_single_block_4',
    'dit_single_block_5',
})


def build_z_image_cap_ids(seq_len: int) -> torch.Tensor:
  """Builds 3D axial position IDs `(t, h, w)` for caption tokens `[seq_len, 3]`.

  Matches `ZImageTransformer2DModel.create_coordinate_grid` for captions:
  `t = 1 .. seq_len`, `h = 0`, `w = 0`.

  Args:
    seq_len: Caption sequence length.

  Returns:
    Float32 tensor of shape `[seq_len, 3]`.
  """
  ids = torch.zeros((seq_len, 3), dtype=torch.float32)
  ids[:, 0] = torch.arange(1, seq_len + 1, dtype=torch.float32)
  return ids


def build_z_image_img_ids(
    grid_h: int,
    grid_w: int,
    seq_len: int,
) -> torch.Tensor:
  """Builds 3D axial position IDs `(t, h, w)` for image patch tokens.

  Matches `ZImageTransformer2DModel.create_coordinate_grid` for image patches:
  `t = seq_len + 1`, `h = 0 .. grid_h - 1`, `w = 0 .. grid_w - 1`.

  Args:
    grid_h: Patch grid height (`latent_h // patch_size`).
    grid_w: Patch grid width (`latent_w // patch_size`).
    seq_len: Caption sequence length.

  Returns:
    Float32 tensor of shape `[grid_h * grid_w, 3]`.
  """
  h = torch.arange(grid_h, dtype=torch.float32)
  w = torch.arange(grid_w, dtype=torch.float32)
  grid_h_t, grid_w_t = torch.meshgrid(h, w, indexing='ij')
  t = torch.full_like(grid_h_t, float(seq_len + 1))
  return torch.stack([t, grid_h_t, grid_w_t], dim=-1).reshape(-1, 3)


def compute_z_image_padded_cap_len(valid_tokens: int, seq_len: int) -> int:
  """Rounds `valid_tokens` up to a multiple of `SEQ_MULTI_OF` (32), clamped."""
  if seq_len <= 0:
    return 0
  if valid_tokens <= 0:
    return seq_len
  rounded = ((valid_tokens + SEQ_MULTI_OF - 1) // SEQ_MULTI_OF) * SEQ_MULTI_OF
  return min(seq_len, max(1, rounded))


def _compute_axial_rope(
    ids: torch.Tensor,
    axes_dims: tuple[int, int, int],
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Computes 3D axial RoPE `(cos, sin)` via `modeling_flux2`."""
  parts = [
      modeling_flux2.get_1d_rotary_pos_embed(
          axis_dim, ids[:, axis_idx], theta=theta
      )
      for axis_idx, axis_dim in enumerate(axes_dims)
  ]
  return (
      torch.cat([p[0] for p in parts], dim=-1),
      torch.cat([p[1] for p in parts], dim=-1),
  )


def build_z_image_rope(
    seq_len: int,
    grid_h: int,
    grid_w: int,
    axes_dims: tuple[int, int, int] = DEFAULT_AXES_DIMS,
    theta: float = DEFAULT_ROPE_THETA,
    active_cap_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Precomputes 3D RoPE `(cap_cos, cap_sin, img_cos, img_sin)` tensors.

  Args:
    seq_len: Caption sequence length.
    grid_h: Patch grid height.
    grid_w: Patch grid width.
    axes_dims: Per-axis rotary dimensions `(d_t, d_h, d_w)`.
    theta: RoPE base frequency (`256.0` for Z-Image).
    active_cap_len: Optional active caption length (multiple of 32). Positions
      at `>= active_cap_len` are zeroed out so attention masks them out, and
      image temporal coordinate is set to `active_cap_len + 1`.

  Returns:
    Tuple `(cap_cos, cap_sin, img_cos, img_sin)` with shapes
    `[1, seq_len, 1, head_dim]` and `[1, grid_h * grid_w, 1, head_dim]`.
  """
  eff_cap_len = (
      seq_len
      if active_cap_len is None
      else min(seq_len, max(1, active_cap_len))
  )
  cap_ids = build_z_image_cap_ids(seq_len)
  img_ids = build_z_image_img_ids(grid_h, grid_w, eff_cap_len)
  cap_cos, cap_sin = _compute_axial_rope(
      cap_ids, axes_dims=axes_dims, theta=theta
  )
  if eff_cap_len < seq_len:
    cap_cos = cap_cos.clone()
    cap_sin = cap_sin.clone()
    cap_cos[eff_cap_len:] = 0.0
    cap_sin[eff_cap_len:] = 0.0
  img_cos, img_sin = _compute_axial_rope(
      img_ids, axes_dims=axes_dims, theta=theta
  )
  return (
      cap_cos.unsqueeze(0).unsqueeze(2),
      cap_sin.unsqueeze(0).unsqueeze(2),
      img_cos.unsqueeze(0).unsqueeze(2),
      img_sin.unsqueeze(0).unsqueeze(2),
  )


def compute_z_image_sigmas(
    steps: int,
    image_tokens: int = 256,
    shift: float = DEFAULT_SHIFT,
    use_dynamic_shifting: bool = False,
) -> list[float]:
  """Computes the shifted FlowMatch Euler sigma schedule for Z-Image-Turbo.

  Args:
    steps: Number of Euler denoising steps.
    image_tokens: Number of patchified image tokens (`grid_h * grid_w`).
    shift: Static time shift factor (`3.0` for `Tongyi-MAI/Z-Image-Turbo`).
    use_dynamic_shifting: Whether to use resolution-dependent `exp(mu)` shifting
      (`False` for `Tongyi-MAI/Z-Image-Turbo`).

  Returns:
    List of `steps + 1` sigma values starting at `1.0` and ending at `0.0`.
  """
  if steps <= 0:
    return []
  m = (1.15 - 0.5) / (4096.0 - 256.0)
  b = 0.5 - m * 256.0
  mu = image_tokens * m + b
  exp_mu = math.exp(mu)

  sigmas: list[float] = []
  for i in range(steps):
    if steps == 1:
      lin = 1.0
    else:
      lin = 1.0 - float(i) * (1.0 - 1.0 / float(steps)) / float(steps - 1)
    if lin >= 1.0:
      sigmas.append(1.0)
    elif use_dynamic_shifting:
      sigmas.append(exp_mu / (exp_mu + (1.0 / lin - 1.0)))
    else:
      sigmas.append((shift * lin) / (1.0 + (shift - 1.0) * lin))
  sigmas.append(0.0)
  return sigmas


def pack_z_image_latents(
    z: torch.Tensor,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> torch.Tensor:
  """Patchifies NCHW latents `[B, C, H, W]` into `[B, (H/p)*(W/p), p*p*C]`."""
  b, c, h, w = z.shape
  p = patch_size
  h_t = h // p
  w_t = w // p
  z = z.view(b, c, h_t, p, w_t, p)
  z = z.permute(0, 2, 4, 3, 5, 1).contiguous()
  return z.view(b, h_t * w_t, p * p * c)


def unpack_z_image_latents(
    packed: torch.Tensor,
    latent_h: int,
    latent_w: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    in_channels: int = DEFAULT_IN_CHANNELS,
) -> torch.Tensor:
  """Unpatchifies `[B, (H/p)*(W/p), p*p*C]` into NCHW `[B, C, H, W]`."""
  b = packed.shape[0]
  p = patch_size
  h_t = latent_h // p
  w_t = latent_w // p
  z = packed.view(b, h_t, w_t, p, p, in_channels)
  z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
  return z.view(b, in_channels, latent_h, latent_w)


def _list_safetensors_files(directory: str) -> list[str]:
  """Returns ordered safetensors files in `directory` (index or single file)."""
  for index_name in (
      'diffusion_pytorch_model.safetensors.index.json',
      'model.safetensors.index.json',
  ):
    index_path = os.path.join(directory, index_name)
    if os.path.exists(index_path):
      with open(index_path, 'r') as f:
        index_data = json.load(f)
      shard_names = sorted(set(index_data.get('weight_map', {}).values()))
      return [os.path.join(directory, name) for name in shard_names]
  for single_name in (
      'diffusion_pytorch_model.safetensors',
      'model.safetensors',
  ):
    single_path = os.path.join(directory, single_name)
    if os.path.exists(single_path):
      return [single_path]
  return sorted(
      os.path.join(directory, name)
      for name in os.listdir(directory)
      if name.endswith('.safetensors')
  )


def _read_safetensors_keys(directory: str) -> set[str]:
  """Reads the union of all tensor keys from safetensors files in `directory`."""
  keys: set[str] = set()
  for path in _list_safetensors_files(directory):
    with safetensors.safe_open(path, framework='pt', device='cpu') as f:
      keys.update(f.keys())
  return keys


def _extract_tokenizer_framing_ids(
    model_dir: str,
) -> tuple[int | None, list[int], list[int]]:
  r"""Extracts Qwen3 chat-template framing token IDs via `transformers`.

  Loads the checkpoint tokenizer with `transformers.AutoTokenizer` (or
  `PreTrainedTokenizerFast` when only `tokenizer.json` is present) and looks up
  `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, newline (`\n` or ByteLevel
  `\u010a`), and `assistant`. Returns `(None, [], [])` if the tokenizer does
  not define all required tokens so C++ falls back to canonical Qwen3 defaults.

  Args:
    model_dir: Root checkpoint directory.

  Returns:
    Tuple `(pad_token_id, prefix_token_ids, suffix_token_ids)`.
  """
  tok_dir = os.path.join(model_dir, 'tokenizer')
  if not os.path.isdir(tok_dir):
    tok_dir = model_dir
  tok_json_path = os.path.join(tok_dir, 'tokenizer.json')
  if not os.path.exists(tok_json_path):
    return None, [], []

  if os.path.exists(os.path.join(tok_dir, 'tokenizer_config.json')):
    tokenizer: Any = transformers.AutoTokenizer.from_pretrained(tok_dir)
  else:
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=tok_json_path
    )
  vocab = tokenizer.get_vocab()

  pad_id = vocab.get('<|endoftext|>')
  im_start = vocab.get('<|im_start|>')
  im_end = vocab.get('<|im_end|>')
  newline = vocab.get('\n')
  if newline is None:
    newline = vocab.get('\u010a')
  assistant = vocab.get('assistant')

  if None in (pad_id, im_start, im_end, newline, assistant):
    return None, [], []
  assert (
      pad_id is not None
      and im_start is not None
      and im_end is not None
      and newline is not None
      and assistant is not None
  )
  return (
      int(pad_id),
      [int(im_start)],
      [int(im_end), int(newline), int(im_start), int(assistant), int(newline)],
  )


def _pack_z_image_metadata(
    *,
    seq_len: int,
    img_size: int,
    in_channels: int,
    patch_size: int,
    dim: int,
    prompt_dim: int,
    default_steps: int,
    vae_scaling_factor: float,
    vae_shift_factor: float,
    axes_dims: tuple[int, ...] | list[int],
    rope_theta: float = DEFAULT_ROPE_THETA,
    shift: float = DEFAULT_SHIFT,
    use_dynamic_shifting: bool = False,
    pad_token_id: int | None = None,
    prefix_token_ids: list[int] | None = None,
    suffix_token_ids: list[int] | None = None,
) -> text2image_metadata_pb2.Text2ImageMetadata:
  """Packs a `ZImageTurbo` proto into `Text2ImageMetadata.custom_model`."""
  z_proto = z_image_turbo_pb2.ZImageTurbo(
      seq_len=seq_len,
      img_size=img_size,
      in_channels=in_channels,
      patch_size=patch_size,
      dim=dim,
      prompt_dim=prompt_dim,
      default_steps=default_steps,
      vae_scaling_factor=vae_scaling_factor,
      vae_shift_factor=vae_shift_factor,
      axes_dims=[int(d) for d in axes_dims],
      rope_theta=rope_theta,
      shift=shift,
      use_dynamic_shifting=use_dynamic_shifting,
  )
  if pad_token_id is not None:
    z_proto.pad_token_id = pad_token_id
  if prefix_token_ids:
    z_proto.prefix_token_ids.extend(int(t) for t in prefix_token_ids)
  if suffix_token_ids:
    z_proto.suffix_token_ids.extend(int(t) for t in suffix_token_ids)

  metadata = text2image_metadata_pb2.Text2ImageMetadata()
  metadata.image_gen_model_type.custom_model.Pack(z_proto)
  return metadata


def verify_checkpoint_keys(
    model_dir: str,
    dit: modeling.ZImageTransformer2DModel,
    vae: modeling.ZImageAutoencoderKL,
    text_encoder_lm: nn.Module | None = None,
) -> None:
  """Verifies that every safetensors key in `model_dir` is consumed.

  Supports both Diffusers split-directory checkpoints (`transformer/`, `vae/`,
  `text_encoder/`) and single-file prefixed checkpoints (`transformer.*`,
  `vae.*`, `text_encoder.*`). Raises `ValueError` on missing subdirectories,
  missing keys, or unconsumed keys.

  Args:
    model_dir: Checkpoint root directory.
    dit: Instantiated `ZImageTransformer2DModel`.
    vae: Instantiated `ZImageAutoencoderKL`.
    text_encoder_lm: Optional instantiated Qwen3 causal LM or base model. When
      `None`, skips `text_encoder` key verification.

  Raises:
    ValueError: If a required component subdirectory or safetensors file is
      missing, or if any expected key is missing or checkpoint key unconsumed.
  """
  dit_expected = set(dit.state_dict().keys())
  vae_expected = set(vae.state_dict().keys())
  te_expected: set[str] | None = (
      set(text_encoder_lm.state_dict().keys())
      if text_encoder_lm is not None
      else None
  )

  dit_dir = os.path.join(model_dir, 'transformer')
  vae_dir = os.path.join(model_dir, 'vae')
  te_dir = os.path.join(model_dir, 'text_encoder')
  is_split_layout = (
      os.path.isdir(dit_dir) or os.path.isdir(vae_dir) or os.path.isdir(te_dir)
  )

  if is_split_layout:
    for component, subdir, expected in (
        ('transformer', dit_dir, dit_expected),
        ('vae', vae_dir, vae_expected),
    ):
      if not os.path.isdir(subdir):
        raise ValueError(
            f'Missing expected Z-Image-Turbo subdirectory: {subdir}'
        )
      found = _read_safetensors_keys(subdir)
      if not found:
        raise ValueError(
            f'No safetensors weights found for {component} in {subdir}.'
        )
      missing = sorted(expected - found)
      unexpected = sorted(found - expected)
      if missing or unexpected:
        raise ValueError(
            f'Checkpoint key mismatch for {component} in {subdir}: '
            f'missing={missing[:5]} ({len(missing)} total), '
            f'unexpected={unexpected[:5]} ({len(unexpected)} total).'
        )
    if te_expected is not None:
      if not os.path.isdir(te_dir):
        raise ValueError(
            f'Missing expected Z-Image-Turbo subdirectory: {te_dir}'
        )
      found_te = _read_safetensors_keys(te_dir)
      if not found_te:
        raise ValueError(
            f'No safetensors weights found for text_encoder in {te_dir}.'
        )
      # Support both `Qwen3Model` (`embed_tokens.*`) and `Qwen3ForCausalLM`
      # (`model.embed_tokens.*` + optional tied `lm_head.weight`).
      allowed_te = set(te_expected)
      optional_te: set[str] = set()
      if getattr(
          getattr(text_encoder_lm, 'config', None), 'tie_word_embeddings', False
      ):
        allowed_te.add('lm_head.weight')
        optional_te.add('lm_head.weight')
      unconsumed_te = sorted(found_te - allowed_te)
      missing_te = sorted((te_expected - optional_te) - found_te)
      if unconsumed_te or missing_te:
        raise ValueError(
            f'Checkpoint key mismatch for text_encoder in {te_dir}: '
            f'missing={missing_te[:5]} ({len(missing_te)} total), '
            f'unexpected={unconsumed_te[:5]} ({len(unconsumed_te)} total).'
        )
    return

  # Single-file / prefixed layout in `model_dir`.
  found_all = _read_safetensors_keys(model_dir)
  if not found_all:
    raise ValueError(f'No safetensors weights found in {model_dir}.')
  expected_all = {f'transformer.{k}' for k in dit_expected} | {
      f'vae.{k}' for k in vae_expected
  }
  optional_all: set[str] = set()
  if te_expected is not None:
    expected_all |= {f'text_encoder.{k}' for k in te_expected}
    if getattr(
        getattr(text_encoder_lm, 'config', None), 'tie_word_embeddings', False
    ):
      optional_all.add('text_encoder.lm_head.weight')
      expected_all.add('text_encoder.lm_head.weight')
  unexpected_all = sorted(found_all - expected_all)
  missing_all = sorted((expected_all - optional_all) - found_all)
  if unexpected_all or missing_all:
    raise ValueError(
        f'Single-file checkpoint key mismatch in {model_dir}: '
        f'missing={missing_all[:5]} ({len(missing_all)} total), '
        f'unexpected={unexpected_all[:5]} ({len(unexpected_all)} total).'
    )


class ZImagePromptEmbedder(nn.Module):
  """Wraps `bonsai_flux2.PromptEmbedder` to extract masked `layers[-2]`."""

  def __init__(self, qwen_model: nn.Module):
    super().__init__()
    self.embedder = bonsai_flux2.PromptEmbedder(qwen_model, layers=(-2,))

  def forward(
      self,
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
  ) -> torch.Tensor:
    hidden = self.embedder(input_ids, attention_mask)
    return hidden * attention_mask.to(hidden.dtype).unsqueeze(-1)


class ZImageCapEmbedderShard(nn.Module):
  """Caption embedder shard (`z_embc`): RMSNorm + Linear + `cap_pad_token`."""

  def __init__(self, dit: modeling.ZImageTransformer2DModel):
    super().__init__()
    self.dit = dit

  def forward(self, cap_feats: torch.Tensor) -> torch.Tensor:
    return self.dit.embed_cap(cap_feats)


class ZImageContextRefinerShard(nn.Module):
  """Context refiner shard (`z_refc`): unmodulated transformer layers."""

  def __init__(self, dit: modeling.ZImageTransformer2DModel):
    super().__init__()
    self.context_refiner = dit.context_refiner

  def forward(
      self,
      cap: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
  ) -> torch.Tensor:
    for block in self.context_refiner:
      cap = block(cap, cos=cos, sin=sin)
    return cap


class ZImageXEmbedderShard(nn.Module):
  """Image patch + timestep embedder shard (`z_embx`)."""

  def __init__(
      self,
      dit: modeling.ZImageTransformer2DModel,
      patch_key: str = '2-1',
  ):
    super().__init__()
    self.dit = dit
    self.patch_key = patch_key

  def forward(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self.dit.embed_x(x, t, patch_key=self.patch_key)


class ZImageNoiseRefinerShard(nn.Module):
  """Noise refiner shard (`z_refx`): AdaLN-modulated image refiner layers."""

  def __init__(self, dit: modeling.ZImageTransformer2DModel):
    super().__init__()
    self.noise_refiner = dit.noise_refiner

  def forward(
      self,
      x: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      adaln_input: torch.Tensor,
  ) -> torch.Tensor:
    for block in self.noise_refiner:
      x = block(x, cos=cos, sin=sin, adaln_input=adaln_input)
    return x


class ZImageMainBlocksShard(nn.Module):
  """Shard of unified S3-DiT main transformer blocks (`zc_main0`..`zc_main5`)."""

  def __init__(
      self,
      dit: modeling.ZImageTransformer2DModel,
      start_idx: int,
      end_idx: int,
  ):
    super().__init__()
    self.layers = nn.ModuleList(dit.layers[start_idx:end_idx])

  def forward(
      self,
      unified: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      adaln_input: torch.Tensor,
  ) -> torch.Tensor:
    for block in self.layers:
      unified = block(unified, cos=cos, sin=sin, adaln_input=adaln_input)
    return unified


class ZImageFinalShard(nn.Module):
  """Final projection shard (`zc_final`): slices image tokens and projects."""

  def __init__(
      self,
      dit: modeling.ZImageTransformer2DModel,
      img_tokens: int,
      patch_key: str = '2-1',
  ):
    super().__init__()
    self.final_layer = dit.all_final_layer[patch_key]
    self.img_tokens = img_tokens

  def forward(
      self,
      unified: torch.Tensor,
      adaln_input: torch.Tensor,
  ) -> torch.Tensor:
    img = unified[:, : self.img_tokens, :]
    return self.final_layer(img, adaln_input)


def _build_text_encoder_config(
    cfg_dict: Mapping[str, Any],
) -> transformers.PretrainedConfig:
  """Builds a `transformers` config from a dict via `AutoConfig.for_model`."""
  model_type = str(cfg_dict.get('model_type', 'qwen3'))
  kwargs = {k: v for k, v in cfg_dict.items() if k != 'model_type'}
  return transformers.AutoConfig.for_model(model_type, **kwargs)


def _load_text_encoder_lm(
    model_dir: str,
    *,
    has_split_dirs: bool,
    root_config: Mapping[str, Any],
    single_file_sd: Mapping[str, torch.Tensor] | None,
    use_random_weights: bool,
) -> nn.Module:
  """Loads the Qwen3 text encoder causal LM via `transformers` `Auto*` APIs."""
  text_enc_dir = os.path.join(model_dir, 'text_encoder')
  if has_split_dirs:
    if os.path.isdir(text_enc_dir):
      cfg_path = os.path.join(text_enc_dir, 'config.json')
      if os.path.exists(cfg_path):
        qwen_config = transformers.AutoConfig.from_pretrained(
            text_enc_dir, torch_dtype=torch.float32
        )
      else:
        qwen_config = _build_text_encoder_config(
            bonsai_flux2._DEFAULT_QWEN3_CONFIG  # pylint: disable=protected-access
        )
      if use_random_weights:
        causal_lm = transformers.AutoModelForCausalLM.from_config(qwen_config)
      else:
        causal_lm = transformers.AutoModelForCausalLM.from_pretrained(
            text_enc_dir, config=qwen_config, torch_dtype=torch.float32
        )
      return causal_lm.to(torch.float32).eval()
    if not use_random_weights:
      raise ValueError(
          f'Missing expected Z-Image-Turbo subdirectory: {text_enc_dir}'
      )

  if not use_random_weights and not single_file_sd:
    raise ValueError(
        'Missing single-file checkpoint weights for text_encoder in'
        f' {model_dir}.'
    )
  te_cfg_dict = root_config.get(
      'text_encoder',
      bonsai_flux2._DEFAULT_QWEN3_CONFIG,  # pylint: disable=protected-access
  )
  qwen_config = _build_text_encoder_config(te_cfg_dict)
  causal_lm = transformers.AutoModelForCausalLM.from_config(qwen_config)
  if not use_random_weights and single_file_sd is not None:
    te_sd = {
        k[len('text_encoder.') :]: v
        for k, v in single_file_sd.items()
        if k.startswith('text_encoder.')
    }
    causal_lm.load_state_dict(te_sd, strict=True)
  return causal_lm.to(torch.float32).eval()


class ZImageTurboModel(nn.Module):
  """Composite holder for Z-Image-Turbo sharded exportable submodels."""

  def __init__(
      self,
      text_encoder: nn.Module,
      dit: modeling.ZImageTransformer2DModel,
      vae: modeling.ZImageAutoencoderKL,
      seq_len: int = DEFAULT_SEQ_LEN,
      img_size: int = DEFAULT_IMG_SIZE,
      default_steps: int = DEFAULT_STEPS,
      num_main_shards: int = NUM_MAIN_SHARDS,
      text_encoder_lm: nn.Module | None = None,
  ):
    super().__init__()
    self.text_encoder = ZImagePromptEmbedder(text_encoder)
    self.text_encoder_lm = text_encoder_lm
    self.dit = dit
    self.vae = vae
    self.vae_decoder = bonsai_flux2.VaeDecoderWrapper(vae)
    self.seq_len = seq_len
    self.img_size = img_size
    self.default_steps = default_steps
    self.num_main_shards = num_main_shards

    patch_size = dit.all_patch_size[0]
    f_patch_size = dit.all_f_patch_size[0]
    patch_key = f'{patch_size}-{f_patch_size}'
    latent_size = img_size // 8
    grid_size = latent_size // patch_size
    self.img_tokens = grid_size * grid_size
    self.packed_channels = (
        f_patch_size * patch_size * patch_size * dit.in_channels
    )

    self.dit_cap_embedder = ZImageCapEmbedderShard(dit)
    self.dit_context_refiner = ZImageContextRefinerShard(dit)
    self.dit_x_embedder = ZImageXEmbedderShard(dit, patch_key=patch_key)
    self.dit_noise_refiner = ZImageNoiseRefinerShard(dit)

    n_layers = len(dit.layers)
    shard_count = max(1, min(num_main_shards, n_layers))
    layers_per_shard = (n_layers + shard_count - 1) // shard_count
    main_shards = []
    for i in range(shard_count):
      start_idx = i * layers_per_shard
      end_idx = min(n_layers, (i + 1) * layers_per_shard)
      if start_idx < end_idx:
        main_shards.append(ZImageMainBlocksShard(dit, start_idx, end_idx))
    self.dit_main_shards = nn.ModuleList(main_shards)
    self.num_main_shards = len(self.dit_main_shards)

    self.dit_final = ZImageFinalShard(
        dit, img_tokens=self.img_tokens, patch_key=patch_key
    )

  @classmethod
  def from_pretrained(
      cls,
      model_dir: str,
      seq_len: int = DEFAULT_SEQ_LEN,
      img_size: int = DEFAULT_IMG_SIZE,
      default_steps: int = DEFAULT_STEPS,
      num_main_shards: int = NUM_MAIN_SHARDS,
      use_random_weights: bool = False,
  ) -> ZImageTurboModel:
    """Loads `ZImageTurboModel` from a split-dir or single-file checkpoint."""
    has_split_dirs = any(
        os.path.isdir(os.path.join(model_dir, sub))
        for sub in ('transformer', 'vae', 'text_encoder')
    )
    single_file_sd: dict[str, torch.Tensor] | None = None
    root_config: dict[str, Any] = {}
    if not has_split_dirs:
      root_cfg_path = os.path.join(model_dir, 'config.json')
      if os.path.exists(root_cfg_path):
        with open(root_cfg_path, 'r') as f:
          root_config = json.load(f)
      if not use_random_weights:
        single_file_sd = modeling_flux2._load_safetensors_state_dict(model_dir)  # pylint: disable=protected-access

    causal_lm = _load_text_encoder_lm(
        model_dir,
        has_split_dirs=has_split_dirs,
        root_config=root_config,
        single_file_sd=single_file_sd,
        use_random_weights=use_random_weights,
    )
    qwen_model = getattr(causal_lm, 'model', causal_lm)

    dit_override: dict[str, torch.Tensor] | None = None
    if single_file_sd is not None:
      dit_override = {
          k[len('transformer.') :]: v
          for k, v in single_file_sd.items()
          if k.startswith('transformer.')
      }
    dit = modeling.ZImageTransformer2DModel.from_pretrained(
        model_dir,
        subfolder='transformer' if has_split_dirs else None,
        torch_dtype=torch.float32,
        use_random_weights=use_random_weights,
        state_dict_override=dit_override,
        config_override=(
            root_config.get('transformer') if not has_split_dirs else None
        ),
    ).eval()

    vae_override: dict[str, torch.Tensor] | None = None
    if single_file_sd is not None:
      vae_override = {
          k[len('vae.') :]: v
          for k, v in single_file_sd.items()
          if k.startswith('vae.')
      }
    vae = modeling.ZImageAutoencoderKL.from_pretrained(
        model_dir,
        subfolder='vae' if has_split_dirs else None,
        torch_dtype=torch.float32,
        use_random_weights=use_random_weights,
        state_dict_override=vae_override,
        config_override=root_config.get('vae') if not has_split_dirs else None,
    ).eval()

    if not use_random_weights:
      verify_checkpoint_keys(model_dir, dit, vae, causal_lm)

    return cls(
        text_encoder=qwen_model,
        dit=dit,
        vae=vae,
        seq_len=seq_len,
        img_size=img_size,
        default_steps=default_steps,
        num_main_shards=num_main_shards,
        text_encoder_lm=causal_lm,
    ).eval()


class ZImageTurbo(bonsai_flux2.BonsaiFlux2):
  """ImageGenModel exporter for `Tongyi-MAI/Z-Image-Turbo`."""

  def __init__(
      self,
      model_path: str,
      export_config: ExportableModuleConfig | None = None,
  ):
    self._scheduler_config: dict[str, Any] = {}
    super().__init__(model_path, export_config)

  def _load_configs(self) -> None:
    super()._load_configs()
    sched_cfg_path = os.path.join(
        self.model_dir, 'scheduler', 'scheduler_config.json'
    )
    if os.path.exists(sched_cfg_path):
      with open(sched_cfg_path, 'r') as f:
        self._scheduler_config = json.load(f)
    if not self._dit_config or not self._vae_config:
      root_cfg_path = os.path.join(self.model_dir, 'config.json')
      if os.path.exists(root_cfg_path):
        with open(root_cfg_path, 'r') as f:
          root_cfg = json.load(f)
        if not self._dit_config and isinstance(
            root_cfg.get('transformer'), dict
        ):
          self._dit_config = root_cfg['transformer']
        if not self._vae_config and isinstance(root_cfg.get('vae'), dict):
          self._vae_config = root_cfg['vae']

  def _get_max_seq_len(self, export_config: ExportableModuleConfig) -> int:
    if 'max_seq_len' in export_config.extra_kwargs or (
        export_config.prefill_lengths and export_config.prefill_lengths != [128]
    ):
      return super()._get_max_seq_len(export_config)
    return DEFAULT_SEQ_LEN

  def _get_default_steps(self, export_config: ExportableModuleConfig) -> int:
    if 'default_steps' in export_config.extra_kwargs:
      return int(export_config.extra_kwargs['default_steps'])
    return DEFAULT_STEPS

  def _export_single_tflite(
      self,
      module: nn.Module,
      sample_inputs: tuple[torch.Tensor, ...],
      output_dir: str,
      name: str,
      quantize: bool,
      quantization_recipe: str | None,
      keep_temporary_files: bool,
      default_quant_fn: Any = None,
      bundle_as_generic_binary: bool = False,
  ) -> str:
    """Exports a single PyTorch module shard to TFLite and optionally quantizes it."""
    fp32_path = os.path.join(output_dir, f'{name}_fp32.tflite')
    litert_torch.convert(module.eval(), sample_inputs).export(fp32_path)
    final_path = fp32_path
    if quantize and default_quant_fn is not None:
      recipe = (quantization_recipe or '').strip()
      if not recipe or recipe == DEFAULT_QUANTIZATION_RECIPE:
        q_path = os.path.join(output_dir, f'{name}_quant.tflite')
        default_quant_fn(fp32_path, q_path)
        final_path = q_path
      else:
        final_path = quantization.maybe_quantize_model(fp32_path, recipe)
      if (
          not keep_temporary_files
          and final_path != fp32_path
          and os.path.exists(fp32_path)
      ):
        os.remove(fp32_path)
    if bundle_as_generic_binary:
      bin_path = f'{final_path}.bin'
      os.replace(final_path, bin_path)
      return bin_path
    return final_path

  def export(self, export_config: ExportableModuleConfig) -> dict[str, str]:
    """Exports Z-Image-Turbo text encoder, 10 S3-DiT shards, and VAE decoder."""
    output_dir = (
        (
            export_config.work_dir
            if export_config.bundle_litert_lm
            else export_config.output_dir
        )
        or export_config.output_dir
        or export_config.work_dir
    )
    if not output_dir:
      raise ValueError('Either output_dir or work_dir must be specified.')
    os.makedirs(output_dir, exist_ok=True)

    seq_len = self._get_max_seq_len(export_config)
    img_size = export_config.t2i_output_image_size or DEFAULT_IMG_SIZE
    default_steps = self._get_default_steps(export_config)
    if img_size <= 0 or img_size % 16 != 0:
      raise ValueError(
          't2i_output_image_size must be a positive multiple of 16, got'
          f' {img_size}.'
      )

    composite = ZImageTurboModel.from_pretrained(
        self.model_dir,
        seq_len=seq_len,
        img_size=img_size,
        default_steps=default_steps,
        num_main_shards=NUM_MAIN_SHARDS,
        use_random_weights=export_config.use_random_weights,
    )
    self._dit_config = vars(composite.dit.config)
    self._vae_config = vars(composite.vae.config)

    quantize = self._should_quantize(export_config)
    quant_recipe = export_config.quantization_recipe
    keep_temp = bool(export_config.keep_temporary_files)
    bundle_litert_lm = bool(export_config.bundle_litert_lm)
    artifacts: dict[str, str] = {}

    def _export_shard(
        name: str,
        module: nn.Module,
        sample_inputs: tuple[torch.Tensor, ...],
        quant_fn: Any = None,
    ) -> str:
      return self._export_single_tflite(
          module,
          sample_inputs,
          output_dir,
          name,
          quantize,
          quant_recipe,
          keep_temp,
          default_quant_fn=quant_fn,
          bundle_as_generic_binary=(
              bundle_litert_lm and name in Z_IMAGE_TURBO_GENERIC_BINARY_SHARDS
          ),
      )

    # 1. Text encoder (`z_te`)
    te_inputs = (
        torch.ones(1, seq_len, dtype=torch.int32),
        torch.ones(1, seq_len, dtype=torch.int32),
    )
    artifacts['text_encoder'] = _export_shard(
        'text_encoder',
        composite.text_encoder,
        te_inputs,
        quant_fn=bonsai_flux2.quantize_text_encoder,
    )

    dim = composite.dit.dim
    cap_feat_dim = composite.dit.cap_feat_dim
    head_dim = dim // composite.dit.n_heads
    adaln_dim = composite.dit.adaln_dim
    img_tokens = composite.img_tokens
    total_tokens = img_tokens + seq_len
    packed_ch = composite.packed_channels

    # 2. Caption embedder (`z_embc`)
    artifacts['dit_cap_embedder'] = _export_shard(
        'dit_cap_embedder',
        composite.dit_cap_embedder,
        (torch.zeros(1, seq_len, cap_feat_dim, dtype=torch.float32),),
        quant_fn=bonsai_flux2.quantize_dit,
    )

    # 3. Context refiner (`z_refc`)
    artifacts['dit_context_refiner'] = _export_shard(
        'dit_context_refiner',
        composite.dit_context_refiner,
        (
            torch.zeros(1, seq_len, dim, dtype=torch.float32),
            torch.ones(1, seq_len, 1, head_dim, dtype=torch.float32),
            torch.zeros(1, seq_len, 1, head_dim, dtype=torch.float32),
        ),
        quant_fn=bonsai_flux2.quantize_dit,
    )

    # 4. Image patch + timestep embedder (`z_embx`)
    artifacts['dit_x_embedder'] = _export_shard(
        'dit_x_embedder',
        composite.dit_x_embedder,
        (
            torch.zeros(1, img_tokens, packed_ch, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
        ),
        quant_fn=bonsai_flux2.quantize_dit,
    )

    # 5. Noise refiner (`z_refx`)
    artifacts['dit_noise_refiner'] = _export_shard(
        'dit_noise_refiner',
        composite.dit_noise_refiner,
        (
            torch.zeros(1, img_tokens, dim, dtype=torch.float32),
            torch.ones(1, img_tokens, 1, head_dim, dtype=torch.float32),
            torch.zeros(1, img_tokens, 1, head_dim, dtype=torch.float32),
            torch.zeros(1, adaln_dim, dtype=torch.float32),
        ),
        quant_fn=bonsai_flux2.quantize_dit,
    )

    # 6. Unified S3-DiT main blocks (`zc_main0`..`zc_main5`)
    main_inputs = (
        torch.zeros(1, total_tokens, dim, dtype=torch.float32),
        torch.ones(1, total_tokens, 1, head_dim, dtype=torch.float32),
        torch.zeros(1, total_tokens, 1, head_dim, dtype=torch.float32),
        torch.zeros(1, adaln_dim, dtype=torch.float32),
    )
    for idx, shard in enumerate(composite.dit_main_shards):
      shard_name = f'dit_single_block_{idx}'
      artifacts[shard_name] = _export_shard(
          shard_name,
          shard,
          main_inputs,
          quant_fn=bonsai_flux2.quantize_dit,
      )

    # 7. Final layer (`zc_final`)
    artifacts['dit_final'] = _export_shard(
        'dit_final',
        composite.dit_final,
        (
            torch.zeros(1, total_tokens, dim, dtype=torch.float32),
            torch.zeros(1, adaln_dim, dtype=torch.float32),
        ),
        quant_fn=bonsai_flux2.quantize_dit,
    )

    # 8. VAE decoder (`zvae`)
    latent_size = img_size // 8
    latent_channels = int(
        getattr(composite.vae.config, 'latent_channels', DEFAULT_IN_CHANNELS)
    )
    artifacts['vae_decoder'] = _export_shard(
        'vae_decoder',
        composite.vae_decoder,
        (
            torch.zeros(
                1,
                latent_channels,
                latent_size,
                latent_size,
                dtype=torch.float32,
            ),
        ),
        quant_fn=None,
    )

    return artifacts

  def get_image_gen_metadata(
      self, export_config: ExportableModuleConfig
  ) -> text2image_metadata_pb2.Text2ImageMetadata:
    """Builds `Text2ImageMetadata` packing `ZImageTurbo` into `custom_model`."""
    seq_len = self._get_max_seq_len(export_config)
    img_size = export_config.t2i_output_image_size or DEFAULT_IMG_SIZE
    default_steps = self._get_default_steps(export_config)
    in_channels = int(self._dit_config.get('in_channels', DEFAULT_IN_CHANNELS))
    all_patch = self._dit_config.get('all_patch_size', (DEFAULT_PATCH_SIZE,))
    patch_size = int(all_patch[0]) if all_patch else DEFAULT_PATCH_SIZE
    dim = int(self._dit_config.get('dim', DEFAULT_DIM))
    prompt_dim = int(self._dit_config.get('cap_feat_dim', DEFAULT_PROMPT_DIM))
    vae_scale = float(
        self._vae_config.get('scaling_factor', DEFAULT_VAE_SCALING_FACTOR)
    )
    vae_shift = float(
        self._vae_config.get('shift_factor', DEFAULT_VAE_SHIFT_FACTOR)
    )
    axes_dims = self._dit_config.get('axes_dims', DEFAULT_AXES_DIMS)
    rope_theta = float(self._dit_config.get('rope_theta', DEFAULT_ROPE_THETA))
    shift = float(self._scheduler_config.get('shift', DEFAULT_SHIFT))
    use_dynamic_shifting = bool(
        self._scheduler_config.get('use_dynamic_shifting', False)
    )
    pad_token_id, prefix_token_ids, suffix_token_ids = (
        _extract_tokenizer_framing_ids(self.model_dir)
    )
    return _pack_z_image_metadata(
        seq_len=seq_len,
        img_size=img_size,
        in_channels=in_channels,
        patch_size=patch_size,
        dim=dim,
        prompt_dim=prompt_dim,
        default_steps=default_steps,
        vae_scaling_factor=vae_scale,
        vae_shift_factor=vae_shift,
        axes_dims=axes_dims,
        rope_theta=rope_theta,
        shift=shift,
        use_dynamic_shifting=use_dynamic_shifting,
        pad_token_id=pad_token_id,
        prefix_token_ids=prefix_token_ids,
        suffix_token_ids=suffix_token_ids,
    )
