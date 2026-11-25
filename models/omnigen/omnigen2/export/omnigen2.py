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
"""OmniGen2 sharded model wrappers and LiteRT-LM export pipeline."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any

import litert_torch
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import quantization
from litert_torch.generative.export_hf.core.image_gen import image_gen_model
from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import bonsai_flux2
import numpy as np
import torch
from torch import nn

from models.omnigen.omnigen2.export import modeling
from models.omnigen.omnigen2.runner import omnigen2_pb2
from litert_lm_builder.runtime.proto import text2image_metadata_pb2

ExportableModuleConfig = exportable_module_config.ExportableModuleConfig

DEFAULT_SEQ_LEN = 64
DEFAULT_IMG_SIZE = 256
DEFAULT_IN_CHANNELS = 16
DEFAULT_PATCH_SIZE = 2
DEFAULT_HIDDEN_SIZE = 2520
DEFAULT_PROMPT_DIM = 2048
DEFAULT_STEPS = 28
DEFAULT_TEXT_GUIDANCE_SCALE = 4.0
DEFAULT_VAE_SCALING_FACTOR = 0.3611
DEFAULT_VAE_SHIFT_FACTOR = 0.1159
DEFAULT_TIMESTEP_SCALE = 1000.0
DEFAULT_AXES_DIM_ROPE = (40, 40, 40)
DEFAULT_ROPE_THETA = 10000.0
DEFAULT_DYNAMIC_TIME_SHIFT = True
NUM_MAIN_SHARDS = 6

OMNIGEN2_EXTRA_KWARGS = (
    'image_size',
    'seq_len',
    'steps',
    'text_guidance_scale',
)
exportable_module_config.register_extra_kwargs_keys(OMNIGEN2_EXTRA_KWARGS)

# OmniGen2-specific sharded TFLite subgraphs that are not part of the closed
# `Text2ImageMetadata.TfLiteModelType` enum. When bundling into `.litertlm`,
# these artifacts are emitted with a `.bin` suffix so `package_image_gen_model`
# packs them as named `GenericBinaryData` sections read zero-copy in C++ via
# `ModelResources::GetGenericBinaryDataBuffer`.
GENERIC_BINARY_SHARD_NAMES = frozenset({
    'dit_cap_embedder',
    'dit_context_refiner',
    'dit_x_embedder',
    'dit_noise_refiner',
    'dit_single_block_4',
    'dit_single_block_5',
})


def _quantize_channelwise_int8(fp32_path: str, out_path: str) -> str:
  """Quantizes a TFLite model with channelwise INT8 (`dynamic_wi8_afp32`).

  OmniGen2's `hidden_size = 2520` and `kv_dim = 840` are not divisible by 32,
  so channelwise INT8 quantization is used instead of blockwise-32 INT4.

  Args:
    fp32_path: Path to the float32 `.tflite` model.
    out_path: Destination path for the quantized `.tflite` model.

  Returns:
    `out_path`.
  """
  q_tmp = quantization.maybe_quantize_model(fp32_path, 'dynamic_wi8_afp32')
  if q_tmp != out_path:
    shutil.move(q_tmp, out_path)
  bonsai_flux2.fix_zero_block_scales(out_path)
  return out_path


def _pack_omnigen2_metadata(
    *,
    seq_len: int,
    img_size: int,
    in_channels: int,
    patch_size: int,
    hidden_size: int,
    prompt_dim: int,
    default_steps: int,
    vae_scaling_factor: float,
    vae_shift_factor: float,
    timestep_scale: float,
    axes_dim_rope: tuple[int, ...] | list[int],
    dynamic_time_shift: bool,
    text_guidance_scale: float,
) -> text2image_metadata_pb2.Text2ImageMetadata:
  """Packs OmniGen2 parameters into `Text2ImageMetadata.image_gen_model_type.custom_model`."""
  omnigen2_proto = omnigen2_pb2.OmniGen2(
      seq_len=seq_len,
      img_size=img_size,
      in_channels=in_channels,
      patch_size=patch_size,
      hidden_size=hidden_size,
      prompt_dim=prompt_dim,
      default_steps=default_steps,
      vae_scaling_factor=vae_scaling_factor,
      vae_shift_factor=vae_shift_factor,
      timestep_scale=timestep_scale,
      axes_dim_rope=[int(d) for d in axes_dim_rope],
      dynamic_time_shift=dynamic_time_shift,
      text_guidance_scale=text_guidance_scale,
  )
  metadata = text2image_metadata_pb2.Text2ImageMetadata()
  metadata.image_gen_model_type.custom_model.Pack(omnigen2_proto)
  return metadata


def build_omnigen2_cap_ids(
    seq_len: int,
    active_cap_len: int | None = None,
) -> torch.Tensor:
  """Builds 3D Omni-RoPE position IDs `(id, h, w)` for caption tokens `[seq_len, 3]`."""
  eff_len = (
      seq_len
      if active_cap_len is None
      else min(seq_len, max(0, active_cap_len))
  )
  ids = torch.zeros((seq_len, 3), dtype=torch.float32)
  if eff_len > 0:
    pos = torch.arange(eff_len, dtype=torch.float32)
    ids[:eff_len] = pos.unsqueeze(-1).expand(eff_len, 3)
  return ids


def build_omnigen2_img_ids(
    grid_h: int,
    grid_w: int,
    pe_shift: int,
) -> torch.Tensor:
  """Builds 3D Omni-RoPE position IDs `(id, h, w)` for image patch tokens."""
  h = torch.arange(grid_h, dtype=torch.float32)
  w = torch.arange(grid_w, dtype=torch.float32)
  grid_h_t, grid_w_t = torch.meshgrid(h, w, indexing='ij')
  t = torch.full_like(grid_h_t, float(pe_shift))
  return torch.stack([t, grid_h_t, grid_w_t], dim=-1).reshape(-1, 3)


def build_omnigen2_rope(
    seq_len: int,
    grid_h: int,
    grid_w: int,
    axes_dim_rope: tuple[int, int, int] = DEFAULT_AXES_DIM_ROPE,
    theta: float = DEFAULT_ROPE_THETA,
    active_cap_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Precomputes 3D Omni-RoPE `(cap_cos, cap_sin, img_cos, img_sin)` tensors."""
  eff_cap_len = (
      seq_len
      if active_cap_len is None
      else min(seq_len, max(1, active_cap_len))
  )
  cap_ids = build_omnigen2_cap_ids(seq_len, active_cap_len=eff_cap_len)
  img_ids = build_omnigen2_img_ids(grid_h, grid_w, pe_shift=eff_cap_len)
  cap_cos, cap_sin = modeling.build_omnigen2_3d_rope_cos_sin(
      cap_ids, axes_dim_rope=axes_dim_rope, theta=theta
  )
  if eff_cap_len < seq_len:
    cap_cos = cap_cos.clone()
    cap_sin = cap_sin.clone()
    cap_cos[eff_cap_len:] = 0.0
    cap_sin[eff_cap_len:] = 0.0
  img_cos, img_sin = modeling.build_omnigen2_3d_rope_cos_sin(
      img_ids, axes_dim_rope=axes_dim_rope, theta=theta
  )
  return (
      cap_cos.unsqueeze(0).unsqueeze(2),
      cap_sin.unsqueeze(0).unsqueeze(2),
      img_cos.unsqueeze(0).unsqueeze(2),
      img_sin.unsqueeze(0).unsqueeze(2),
  )


def build_omnigen2_masks(
    seq_len: int,
    img_tokens: int,
    active_cap_len: int | None = None,
    neg_inf: float = -1e9,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Builds 4D additive attention masks `(cap_mask, joint_mask)` for OmniGen2."""
  eff_cap_len = (
      seq_len
      if active_cap_len is None
      else min(seq_len, max(1, active_cap_len))
  )
  cap_mask = torch.zeros((1, 1, 1, seq_len), dtype=torch.float32)
  if eff_cap_len < seq_len:
    cap_mask[:, :, :, eff_cap_len:] = neg_inf
  joint_mask = torch.zeros((1, 1, 1, seq_len + img_tokens), dtype=torch.float32)
  if eff_cap_len < seq_len:
    joint_mask[:, :, :, eff_cap_len:seq_len] = neg_inf
  return cap_mask, joint_mask


def compute_omnigen2_timesteps(
    steps: int,
    latent_tokens: int = 1024,
    dynamic_time_shift: bool = DEFAULT_DYNAMIC_TIME_SHIFT,
) -> list[float]:
  """Computes the OmniGen2 FlowMatch Euler timestep schedule in `[0.0, 1.0]`."""
  if steps <= 0:
    raise ValueError(f'steps must be positive, got {steps}.')
  t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[:-1]
  if dynamic_time_shift and latent_tokens > 0:
    m = np.float32(np.sqrt(latent_tokens) / 40.0)
    t = t / (m - m * t + t)
  schedule = [float(x) for x in t]
  schedule.append(1.0)
  return schedule


def pack_omnigen2_latents(
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


def unpack_omnigen2_latents(
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


class OmniGen2CapEmbedderShard(nn.Module):
  """Caption embedder shard: `time_caption_embed.caption_embedder`."""

  def __init__(self, dit: modeling.OmniGen2Transformer2DModel):
    super().__init__()
    self.dit = dit

  def forward(self, text_hidden_states: torch.Tensor) -> torch.Tensor:
    return self.dit.embed_cap(text_hidden_states)


class OmniGen2ContextRefinerShard(nn.Module):
  """Context refiner shard: 2 unmodulated `context_refiner` layers."""

  def __init__(self, dit: modeling.OmniGen2Transformer2DModel):
    super().__init__()
    self.context_refiner = dit.context_refiner

  def forward(
      self,
      cap: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      mask: torch.Tensor,
  ) -> torch.Tensor:
    for block in self.context_refiner:
      cap = block(cap, cos=cos, sin=sin, attention_mask=mask)
    return cap


class OmniGen2XEmbedderShard(nn.Module):
  """Image patch + timestep embedder shard."""

  def __init__(self, dit: modeling.OmniGen2Transformer2DModel):
    super().__init__()
    self.dit = dit

  def forward(
      self,
      x: torch.Tensor,
      t: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self.dit.embed_x(x, t)


class OmniGen2NoiseRefinerShard(nn.Module):
  """Noise refiner shard: 2 modulated `noise_refiner` layers."""

  def __init__(self, dit: modeling.OmniGen2Transformer2DModel):
    super().__init__()
    self.noise_refiner = dit.noise_refiner

  def forward(
      self,
      x: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      temb: torch.Tensor,
  ) -> torch.Tensor:
    for block in self.noise_refiner:
      x = block(x, cos=cos, sin=sin, temb=temb)
    return x


class OmniGen2RefImageEmbedderShard(nn.Module):
  """Reference image patch embedder (`ref_image_patch_embedder`)."""

  def __init__(self, dit: modeling.OmniGen2Transformer2DModel):
    super().__init__()
    self.ref_image_patch_embedder = dit.ref_image_patch_embedder

  def forward(self, ref_x: torch.Tensor) -> torch.Tensor:
    return self.ref_image_patch_embedder(ref_x)


class OmniGen2RefImageRefinerShard(nn.Module):
  """Reference image refiner shard (`image_index_embedding` + `ref_image_refiner`)."""

  def __init__(self, dit: modeling.OmniGen2Transformer2DModel):
    super().__init__()
    self.ref_image_refiner = dit.ref_image_refiner
    self.image_index_embedding = dit.image_index_embedding

  def forward(
      self,
      ref_x: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      temb: torch.Tensor,
      image_index: torch.Tensor,
  ) -> torch.Tensor:
    idx_emb = torch.nn.functional.embedding(
        image_index.reshape(-1), self.image_index_embedding
    ).view(1, 1, -1)
    ref_x = ref_x + idx_emb
    for block in self.ref_image_refiner:
      ref_x = block(ref_x, cos=cos, sin=sin, temb=temb)
    return ref_x


class OmniGen2MainBlocksShard(nn.Module):
  """Shard of joint OmniGen2 main transformer blocks (`dit_single_block_0..5`)."""

  def __init__(
      self,
      dit: modeling.OmniGen2Transformer2DModel,
      start_idx: int,
      end_idx: int,
  ):
    super().__init__()
    self.layers = nn.ModuleList(dit.layers[start_idx:end_idx])

  def forward(
      self,
      joint: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      mask: torch.Tensor,
      temb: torch.Tensor,
  ) -> torch.Tensor:
    for block in self.layers:
      joint = block(joint, cos=cos, sin=sin, temb=temb, attention_mask=mask)
    return joint


class OmniGen2FinalShard(nn.Module):
  """Final projection shard: slices last `img_tokens` and runs `norm_out`."""

  def __init__(
      self,
      dit: modeling.OmniGen2Transformer2DModel,
      img_tokens: int,
  ):
    super().__init__()
    self.norm_out = dit.norm_out
    self.img_tokens = img_tokens

  def forward(
      self,
      joint: torch.Tensor,
      temb: torch.Tensor,
  ) -> torch.Tensor:
    img = joint[:, -self.img_tokens :, :]
    return self.norm_out(img, temb)


OmniGen2VaeDecoderShard = bonsai_flux2.VaeDecoderWrapper


class OmniGen2Model(nn.Module):
  """Composite holder for OmniGen2 submodels (`mllm`, `transformer`, `vae`) and shards."""

  def __init__(
      self,
      text_encoder: modeling.OmniGen2TextEncoder,
      dit: modeling.OmniGen2Transformer2DModel,
      vae: modeling.OmniGen2AutoencoderKL,
      seq_len: int = DEFAULT_SEQ_LEN,
      img_size: int = DEFAULT_IMG_SIZE,
      default_steps: int = DEFAULT_STEPS,
      num_main_shards: int = NUM_MAIN_SHARDS,
      dynamic_time_shift: bool = DEFAULT_DYNAMIC_TIME_SHIFT,
      text_guidance_scale: float = DEFAULT_TEXT_GUIDANCE_SCALE,
  ):
    super().__init__()
    self.text_encoder = text_encoder
    self.dit = dit
    self.vae = vae
    self.vae_decoder = OmniGen2VaeDecoderShard(vae)
    self.seq_len = seq_len
    self.img_size = img_size
    self.default_steps = default_steps
    self.num_main_shards = num_main_shards
    self.dynamic_time_shift = dynamic_time_shift
    self.text_guidance_scale = text_guidance_scale

    patch_size = dit.patch_size
    latent_size = img_size // 8
    grid_size = latent_size // patch_size
    self.img_tokens = grid_size * grid_size
    self.packed_channels = dit.packed_channels

    self.dit_cap_embedder = OmniGen2CapEmbedderShard(dit)
    self.dit_context_refiner = OmniGen2ContextRefinerShard(dit)
    self.dit_x_embedder = OmniGen2XEmbedderShard(dit)
    self.dit_noise_refiner = OmniGen2NoiseRefinerShard(dit)
    self.dit_ref_image_embedder = OmniGen2RefImageEmbedderShard(dit)
    self.dit_ref_image_refiner = OmniGen2RefImageRefinerShard(dit)

    n_layers = len(dit.layers)
    shard_count = max(1, min(num_main_shards, n_layers))
    layers_per_shard = (n_layers + shard_count - 1) // shard_count
    main_shards = []
    for i in range(shard_count):
      start_idx = i * layers_per_shard
      end_idx = min(n_layers, (i + 1) * layers_per_shard)
      if start_idx < end_idx:
        main_shards.append(OmniGen2MainBlocksShard(dit, start_idx, end_idx))
    self.dit_main_shards = nn.ModuleList(main_shards)
    self.num_main_shards = len(self.dit_main_shards)

    self.dit_final = OmniGen2FinalShard(dit, img_tokens=self.img_tokens)

  @classmethod
  def from_pretrained(
      cls,
      model_dir: str,
      seq_len: int = DEFAULT_SEQ_LEN,
      img_size: int = DEFAULT_IMG_SIZE,
      default_steps: int = DEFAULT_STEPS,
      num_main_shards: int = NUM_MAIN_SHARDS,
      use_random_weights: bool = False,
      text_guidance_scale: float = DEFAULT_TEXT_GUIDANCE_SCALE,
  ) -> OmniGen2Model:
    """Loads `OmniGen2Model` from an `OmniGen2/OmniGen2` checkpoint directory."""
    mllm_sub = 'mllm'
    if not os.path.exists(os.path.join(model_dir, 'mllm')) and os.path.exists(
        os.path.join(model_dir, 'text_encoder')
    ):
      mllm_sub = 'text_encoder'
    text_encoder = modeling.OmniGen2TextEncoder.from_pretrained(
        model_dir,
        subfolder=mllm_sub,
        torch_dtype=torch.float32,
        use_random_weights=use_random_weights,
    ).eval()

    dit_sub = (
        'transformer'
        if os.path.exists(os.path.join(model_dir, 'transformer'))
        else None
    )
    dit = modeling.OmniGen2Transformer2DModel.from_pretrained(
        model_dir,
        subfolder=dit_sub,
        torch_dtype=torch.float32,
        use_random_weights=use_random_weights,
    ).eval()

    vae_sub = 'vae' if os.path.exists(os.path.join(model_dir, 'vae')) else None
    vae = modeling.OmniGen2AutoencoderKL.from_pretrained(
        model_dir,
        subfolder=vae_sub,
        torch_dtype=torch.float32,
        use_random_weights=use_random_weights,
    ).eval()

    dynamic_time_shift = DEFAULT_DYNAMIC_TIME_SHIFT
    sched_cfg_path = os.path.join(
        model_dir, 'scheduler', 'scheduler_config.json'
    )
    if os.path.exists(sched_cfg_path):
      with open(sched_cfg_path, 'r') as f:
        sched_cfg = json.load(f)
      dynamic_time_shift = bool(
          sched_cfg.get('dynamic_time_shift', DEFAULT_DYNAMIC_TIME_SHIFT)
      )

    return cls(
        text_encoder=text_encoder,
        dit=dit,
        vae=vae,
        seq_len=seq_len,
        img_size=img_size,
        default_steps=default_steps,
        num_main_shards=num_main_shards,
        dynamic_time_shift=dynamic_time_shift,
        text_guidance_scale=text_guidance_scale,
    ).eval()

  def get_image_gen_metadata(
      self,
      export_config: ExportableModuleConfig | None = None,
  ) -> text2image_metadata_pb2.Text2ImageMetadata:
    """Builds `Text2ImageMetadata` packing `omnigen2_pb2.OmniGen2`."""
    del export_config
    return _pack_omnigen2_metadata(
        seq_len=self.seq_len,
        img_size=self.img_size,
        in_channels=self.dit.in_channels,
        patch_size=self.dit.patch_size,
        hidden_size=self.dit.hidden_size,
        prompt_dim=self.dit.text_feat_dim,
        default_steps=self.default_steps,
        vae_scaling_factor=float(
            getattr(
                self.vae.config, 'scaling_factor', DEFAULT_VAE_SCALING_FACTOR
            )
        ),
        vae_shift_factor=float(
            getattr(self.vae.config, 'shift_factor', DEFAULT_VAE_SHIFT_FACTOR)
        ),
        timestep_scale=self.dit.timestep_scale,
        axes_dim_rope=self.dit.axes_dim_rope,
        dynamic_time_shift=self.dynamic_time_shift,
        text_guidance_scale=self.text_guidance_scale,
    )


class OmniGen2(image_gen_model.ImageGenModel):
  """ImageGenModel exporter for `OmniGen2/OmniGen2`."""

  def __init__(
      self,
      model_path: str,
      export_config: ExportableModuleConfig | None = None,
  ):
    super().__init__(model_path, export_config)
    self._checkpoint_dir = self.model_dir
    # `export_lib.py` loads `AutoTokenizer` from `model.model_dir/tokenizer`
    # or `model.model_dir`. Official `OmniGen2/OmniGen2` checkpoints store
    # tokenizer files under `processor/`, so point `self.model_dir` there when
    # `processor/` exists and `tokenizer/` does not, while reading all configs
    # and weights from `self._checkpoint_dir`.
    processor_dir = os.path.join(self._checkpoint_dir, 'processor')
    tokenizer_dir = os.path.join(self._checkpoint_dir, 'tokenizer')
    if os.path.isdir(processor_dir) and not os.path.isdir(tokenizer_dir):
      self.model_dir = processor_dir

    self._dit_config: dict[str, Any] = {}
    self._vae_config: dict[str, Any] = {}
    self._scheduler_config: dict[str, Any] = {}
    self._load_configs()

  def _load_configs(self) -> None:
    dit_cfg_path = os.path.join(
        self._checkpoint_dir, 'transformer', 'config.json'
    )
    if os.path.exists(dit_cfg_path):
      with open(dit_cfg_path, 'r') as f:
        self._dit_config = json.load(f)
    vae_cfg_path = os.path.join(self._checkpoint_dir, 'vae', 'config.json')
    if os.path.exists(vae_cfg_path):
      with open(vae_cfg_path, 'r') as f:
        self._vae_config = json.load(f)
    sched_cfg_path = os.path.join(
        self._checkpoint_dir, 'scheduler', 'scheduler_config.json'
    )
    if os.path.exists(sched_cfg_path):
      with open(sched_cfg_path, 'r') as f:
        self._scheduler_config = json.load(f)

  def _get_max_seq_len(self, export_config: ExportableModuleConfig) -> int:
    if 'seq_len' in export_config.extra_kwargs:
      return int(export_config.extra_kwargs['seq_len'])
    if 'max_seq_len' in export_config.extra_kwargs:
      return int(export_config.extra_kwargs['max_seq_len'])
    if export_config.prefill_lengths and export_config.prefill_lengths != [128]:
      return int(export_config.prefill_lengths[0])
    return DEFAULT_SEQ_LEN

  def _get_image_size(self, export_config: ExportableModuleConfig) -> int:
    if 'image_size' in export_config.extra_kwargs:
      return int(export_config.extra_kwargs['image_size'])
    return export_config.t2i_output_image_size or DEFAULT_IMG_SIZE

  def _get_default_steps(self, export_config: ExportableModuleConfig) -> int:
    if 'steps' in export_config.extra_kwargs:
      return int(export_config.extra_kwargs['steps'])
    return DEFAULT_STEPS

  def _get_text_guidance_scale(
      self, export_config: ExportableModuleConfig
  ) -> float:
    if 'text_guidance_scale' in export_config.extra_kwargs:
      return float(export_config.extra_kwargs['text_guidance_scale'])
    return DEFAULT_TEXT_GUIDANCE_SCALE

  def _should_quantize(self, export_config: ExportableModuleConfig) -> bool:
    recipe = export_config.quantization_recipe
    if not recipe or str(recipe).strip().lower() in (
        'none',
        'null',
        'false',
        '',
    ):
      return False
    return True

  def _export_single_tflite(
      self,
      module: nn.Module,
      sample_inputs: tuple[torch.Tensor, ...],
      output_dir: str,
      name: str,
      quantize: bool,
      keep_temporary_files: bool,
      bundle_litert_lm: bool,
      quant_fn: Any = None,
  ) -> str:
    """Exports a single PyTorch module shard to TFLite and optionally quantizes it."""
    fp32_path = os.path.join(output_dir, f'{name}_fp32.tflite')
    litert_torch.convert(module.eval(), sample_inputs).export(fp32_path)
    out_path = fp32_path
    if quantize and quant_fn is not None:
      q_path = os.path.join(output_dir, f'{name}_quant.tflite')
      quant_fn(fp32_path, q_path)
      if not keep_temporary_files and os.path.exists(fp32_path):
        os.remove(fp32_path)
      out_path = q_path

    if bundle_litert_lm and name in GENERIC_BINARY_SHARD_NAMES:
      bin_path = f'{out_path}.bin'
      shutil.move(out_path, bin_path)
      return bin_path
    return out_path

  def export(self, export_config: ExportableModuleConfig) -> dict[str, str]:
    """Exports OmniGen2 text encoder, DiT shards, and VAE decoder."""
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
    img_size = self._get_image_size(export_config)
    default_steps = self._get_default_steps(export_config)
    text_guidance_scale = self._get_text_guidance_scale(export_config)
    if img_size <= 0 or img_size % 16 != 0:
      raise ValueError(
          't2i_output_image_size must be a positive multiple of 16, got'
          f' {img_size}.'
      )

    if not export_config.use_random_weights:
      modeling.verify_checkpoint_keys(self._checkpoint_dir)

    composite = OmniGen2Model.from_pretrained(
        self._checkpoint_dir,
        seq_len=seq_len,
        img_size=img_size,
        default_steps=default_steps,
        num_main_shards=NUM_MAIN_SHARDS,
        use_random_weights=export_config.use_random_weights,
        text_guidance_scale=text_guidance_scale,
    )
    if len(composite.dit_main_shards) != NUM_MAIN_SHARDS:
      raise ValueError(
          f'OmniGen2 requires {NUM_MAIN_SHARDS} main DiT block shards '
          f'(num_layers >= {NUM_MAIN_SHARDS}), got '
          f'{len(composite.dit_main_shards)}.'
      )
    self._dit_config = vars(composite.dit.config)
    self._vae_config = vars(composite.vae.config)

    quantize = self._should_quantize(export_config)
    keep_temp = bool(export_config.keep_temporary_files)
    bundle_lm = bool(export_config.bundle_litert_lm)
    artifacts: dict[str, str] = {}

    # 1. Text encoder (`tf_lite_text_encoder`)
    te_inputs = (
        torch.ones(1, seq_len, dtype=torch.int32),
        torch.ones(1, seq_len, dtype=torch.int32),
    )
    artifacts['text_encoder'] = self._export_single_tflite(
        composite.text_encoder,
        te_inputs,
        output_dir,
        'text_encoder',
        quantize,
        keep_temp,
        bundle_lm,
        quant_fn=_quantize_channelwise_int8,
    )

    hidden_size = composite.dit.hidden_size
    text_feat_dim = composite.dit.text_feat_dim
    head_dim = hidden_size // composite.dit.num_attention_heads
    temb_dim = composite.dit.temb_dim
    img_tokens = composite.img_tokens
    total_tokens = seq_len + img_tokens
    packed_ch = composite.packed_channels

    # 2. Caption embedder (GenericBinaryData: `dit_cap_embedder`)
    artifacts['dit_cap_embedder'] = self._export_single_tflite(
        composite.dit_cap_embedder,
        (torch.zeros(1, seq_len, text_feat_dim, dtype=torch.float32),),
        output_dir,
        'dit_cap_embedder',
        quantize,
        keep_temp,
        bundle_lm,
        quant_fn=_quantize_channelwise_int8,
    )

    # 3. Context refiner (GenericBinaryData: `dit_context_refiner`)
    artifacts['dit_context_refiner'] = self._export_single_tflite(
        composite.dit_context_refiner,
        (
            torch.zeros(1, seq_len, hidden_size, dtype=torch.float32),
            torch.ones(1, seq_len, 1, head_dim, dtype=torch.float32),
            torch.zeros(1, seq_len, 1, head_dim, dtype=torch.float32),
            torch.zeros(1, 1, 1, seq_len, dtype=torch.float32),
        ),
        output_dir,
        'dit_context_refiner',
        quantize,
        keep_temp,
        bundle_lm,
        quant_fn=_quantize_channelwise_int8,
    )

    # 4. Image patch + timestep embedder (GenericBinaryData: `dit_x_embedder`)
    artifacts['dit_x_embedder'] = self._export_single_tflite(
        composite.dit_x_embedder,
        (
            torch.zeros(1, img_tokens, packed_ch, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
        ),
        output_dir,
        'dit_x_embedder',
        quantize,
        keep_temp,
        bundle_lm,
        quant_fn=_quantize_channelwise_int8,
    )

    # 5. Noise refiner (GenericBinaryData: `dit_noise_refiner`)
    artifacts['dit_noise_refiner'] = self._export_single_tflite(
        composite.dit_noise_refiner,
        (
            torch.zeros(1, img_tokens, hidden_size, dtype=torch.float32),
            torch.ones(1, img_tokens, 1, head_dim, dtype=torch.float32),
            torch.zeros(1, img_tokens, 1, head_dim, dtype=torch.float32),
            torch.zeros(1, temb_dim, dtype=torch.float32),
        ),
        output_dir,
        'dit_noise_refiner',
        quantize,
        keep_temp,
        bundle_lm,
        quant_fn=_quantize_channelwise_int8,
    )

    # 6. Joint main blocks (`dit_single_block_0..3` as TFLiteModel,
    #    `dit_single_block_4..5` as GenericBinaryData)
    main_inputs = (
        torch.zeros(1, total_tokens, hidden_size, dtype=torch.float32),
        torch.ones(1, total_tokens, 1, head_dim, dtype=torch.float32),
        torch.zeros(1, total_tokens, 1, head_dim, dtype=torch.float32),
        torch.zeros(1, 1, 1, total_tokens, dtype=torch.float32),
        torch.zeros(1, temb_dim, dtype=torch.float32),
    )
    for idx, shard in enumerate(composite.dit_main_shards):
      artifacts[f'dit_single_block_{idx}'] = self._export_single_tflite(
          shard,
          main_inputs,
          output_dir,
          f'dit_single_block_{idx}',
          quantize,
          keep_temp,
          bundle_lm,
          quant_fn=_quantize_channelwise_int8,
      )

    # 7. Final layer (`tf_lite_diffusion_transformer_final`)
    artifacts['dit_final'] = self._export_single_tflite(
        composite.dit_final,
        (
            torch.zeros(1, total_tokens, hidden_size, dtype=torch.float32),
            torch.zeros(1, temb_dim, dtype=torch.float32),
        ),
        output_dir,
        'dit_final',
        quantize,
        keep_temp,
        bundle_lm,
        quant_fn=_quantize_channelwise_int8,
    )

    # 8. VAE decoder (`tf_lite_image_decoder`)
    latent_size = img_size // 8
    latent_channels = int(
        getattr(composite.vae.config, 'latent_channels', DEFAULT_IN_CHANNELS)
    )
    artifacts['vae_decoder'] = self._export_single_tflite(
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
        output_dir,
        'vae_decoder',
        quantize,
        keep_temp,
        bundle_lm,
        quant_fn=None,
    )

    return artifacts

  def get_image_gen_metadata(
      self, export_config: ExportableModuleConfig
  ) -> text2image_metadata_pb2.Text2ImageMetadata:
    """Builds `Text2ImageMetadata` packing `omnigen2_pb2.OmniGen2`."""
    seq_len = self._get_max_seq_len(export_config)
    img_size = self._get_image_size(export_config)
    default_steps = self._get_default_steps(export_config)
    text_guidance_scale = self._get_text_guidance_scale(export_config)
    in_channels = int(self._dit_config.get('in_channels', DEFAULT_IN_CHANNELS))
    patch_size = int(self._dit_config.get('patch_size', DEFAULT_PATCH_SIZE))
    hidden_size = int(self._dit_config.get('hidden_size', DEFAULT_HIDDEN_SIZE))
    prompt_dim = int(self._dit_config.get('text_feat_dim', DEFAULT_PROMPT_DIM))
    timestep_scale = float(
        self._dit_config.get('timestep_scale', DEFAULT_TIMESTEP_SCALE)
    )
    vae_scale = float(
        self._vae_config.get('scaling_factor', DEFAULT_VAE_SCALING_FACTOR)
    )
    vae_shift = float(
        self._vae_config.get('shift_factor', DEFAULT_VAE_SHIFT_FACTOR)
    )
    axes_dim_rope = self._dit_config.get('axes_dim_rope', DEFAULT_AXES_DIM_ROPE)
    dynamic_time_shift = bool(
        self._scheduler_config.get(
            'dynamic_time_shift', DEFAULT_DYNAMIC_TIME_SHIFT
        )
    )
    return _pack_omnigen2_metadata(
        seq_len=seq_len,
        img_size=img_size,
        in_channels=in_channels,
        patch_size=patch_size,
        hidden_size=hidden_size,
        prompt_dim=prompt_dim,
        default_steps=default_steps,
        vae_scaling_factor=vae_scale,
        vae_shift_factor=vae_shift,
        timestep_scale=timestep_scale,
        axes_dim_rope=axes_dim_rope,
        dynamic_time_shift=dynamic_time_shift,
        text_guidance_scale=text_guidance_scale,
    )
