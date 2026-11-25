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
"""PyTorch export wrappers and strict weight loaders for Janus-Pro.

Reuses upstream `transformers.models.janus` architecture modules
(`JanusVisionModel`, `JanusVisionAlignerMLP`, `JanusVQVAEAlignerMLP`,
`JanusVQVAEHead`, `JanusVQVAEVectorQuantizer`, `JanusVQVAEDecoder`),
`export_hf.core.external_emb.LiteRTExportableModuleForEmbedder`, and
`export_hf`'s shared `weights_utils` helpers while mapping `deepseek-ai`
checkpoint keys (`vision_model.vision_tower.*`, `aligner.*`, `gen_embed.*`,
`gen_aligner.*`, `gen_head.*`, `gen_vision_model.*`, `language_model.*`) with
strict parameter population and key-consumption verification.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import os

from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
from litert_torch.generative.export_hf.model_ext.minicpm_o import weights_utils
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.janus import configuration_janus
from transformers.models.janus import modeling_janus

# Re-export shared `export_hf` helpers.
get_weight = weights_utils.get_weight
make_embedding = weights_utils.make_embedding
maybe_quantize = weights_utils.maybe_quantize
verify_all_parameters_populated = weights_utils.verify_all_parameters_populated

# Checkpoint key prefixes in `deepseek-ai/Janus-Pro-1B` / `Janus-Pro-7B` that
# belong to training-only or unused submodules (`ignore_head=True` in the
# SigLIP vision tower and the VQ-16 encoder / codebook counter in
# `gen_vision_model`) and are legitimately unused during inference.
IGNORED_CHECKPOINT_KEY_PREFIXES: tuple[str, ...] = (
    "gen_vision_model.encoder.",
    "gen_vision_model.quant_conv.",
    "gen_vision_model.quantize.codebook_used",
    "vision_model.vision_tower.attn_pool.",
)


def copy_weight(
    target: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    *keys: str,
    strict: bool = True,
    populated: set[int] | None = None,
    consumed_keys: set[str] | None = None,
) -> None:
  """Copies the first matching checkpoint tensor in `keys` into `target`."""
  matched_key: str | None = None
  for key in keys:
    in_mem = key in state_dict
    shard_path = weight_map.get(key)
    in_shard = bool(shard_path and os.path.exists(shard_path))
    if not in_mem and not in_shard:
      continue
    if strict:
      matched_key = key
      break
    raw = weights_utils.get_weight(state_dict, weight_map, key)
    if raw is not None and raw.shape == target.shape:
      matched_key = key
      break

  if matched_key is not None:
    weights_utils.copy_weight(
        target,
        state_dict,
        weight_map,
        matched_key,
        strict=strict,
        populated=populated,
    )
    if consumed_keys is not None:
      consumed_keys.add(matched_key)
    return

  weights_utils.copy_weight(
      target,
      state_dict,
      weight_map,
      *keys,
      strict=strict,
      populated=populated,
  )


def verify_checkpoint_keys_consumed(
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    consumed_keys: set[str],
    module_name: str,
    prefixes: Sequence[str] | None = None,
) -> None:
  """Raises ValueError if any non-ignored checkpoint key was not consumed."""
  all_keys = set(state_dict) | set(weight_map)
  unconsumed: list[str] = []
  prefix_tuple = tuple(prefixes) if prefixes is not None else None
  for key in sorted(all_keys):
    if key.startswith(IGNORED_CHECKPOINT_KEY_PREFIXES):
      continue
    if prefix_tuple is not None and not key.startswith(prefix_tuple):
      continue
    if key not in consumed_keys:
      unconsumed.append(key)
  if unconsumed:
    raise ValueError(
        f"Unconsumed checkpoint keys in {module_name}: {unconsumed}"
    )


def _copy_module_weights(
    module: nn.Module,
    prefix: str,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    strict: bool,
    populated: set[int],
    consumed: set[str],
) -> None:
  """Copies weights for `module` by iterating through its named parameters."""
  for name, param in module.named_parameters():
    copy_weight(
        param,
        state_dict,
        weight_map,
        f"{prefix}.{name}",
        strict=strict,
        populated=populated,
        consumed_keys=consumed,
    )


class JanusProVisionEncoderModule(nn.Module):
  """Fused `JanusVisionModel` (SigLIP) + `JanusVisionAlignerMLP` wrapper."""

  def __init__(
      self,
      vision_hidden_size: int = 1024,
      vision_intermediate_size: int = 4096,
      vision_num_layers: int = 24,
      vision_num_heads: int = 16,
      image_size: int = 384,
      patch_size: int = 16,
      llm_hidden_size: int = 2048,
  ):
    super().__init__()
    self.vision_hidden_size = vision_hidden_size
    self.image_size = image_size
    self.patch_size = patch_size
    self.grid_size = image_size // patch_size
    self.num_patches = self.grid_size * self.grid_size
    self.llm_hidden_size = llm_hidden_size

    mlp_ratio = vision_intermediate_size / vision_hidden_size
    if int(vision_hidden_size * mlp_ratio) != vision_intermediate_size:
      mlp_ratio = (vision_intermediate_size + 0.5) / vision_hidden_size
    vision_cfg = configuration_janus.JanusVisionConfig(
        hidden_size=vision_hidden_size,
        num_hidden_layers=vision_num_layers,
        num_attention_heads=vision_num_heads,
        image_size=image_size,
        patch_size=patch_size,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        mlp_ratio=mlp_ratio,
        attention_bias=True,
        projection_dim=llm_hidden_size,
        use_qk_norm=False,
        depth=2,
        num_image_tokens=self.num_patches,
    )
    self.vision_model = modeling_janus.JanusVisionModel(vision_cfg)
    self.vision_model.set_attn_implementation("eager")
    self.aligner = modeling_janus.JanusVisionAlignerMLP(vision_cfg)

  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    vision_outputs = self.vision_model(pixel_values=pixel_values)
    return self.aligner(vision_outputs.last_hidden_state)


def load_vision_encoder_weights(
    mod: JanusProVisionEncoderModule,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    strict: bool = True,
) -> set[str]:
  """Loads `vision_model.vision_tower.*` and `aligner.*` weights into `mod`."""
  populated: set[int] = set()
  consumed: set[str] = set()
  vt = "vision_model.vision_tower"

  _copy_module_weights(
      mod.vision_model.embeddings.patch_embedding,
      f"{vt}.patch_embed.proj",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )

  pos_key = f"{vt}.pos_embed"
  pos_target = mod.vision_model.embeddings.position_embedding.weight
  raw_pos = weights_utils.get_weight(state_dict, weight_map, pos_key)
  if raw_pos is not None:
    expected_3d = (1, mod.num_patches, mod.vision_hidden_size)
    if raw_pos.shape == expected_3d:
      with torch.no_grad():
        pos_target.copy_(
            raw_pos.squeeze(0).to(
                dtype=pos_target.dtype, device=pos_target.device
            )
        )
      populated.add(pos_target.data_ptr())
      consumed.add(pos_key)
    elif (
        raw_pos.ndim == 3
        and raw_pos.shape[0] == 1
        and raw_pos.shape[2] == mod.vision_hidden_size
    ):
      src_patches = int(raw_pos.shape[1])
      src_grid = int(math.isqrt(src_patches))
      if src_grid * src_grid == src_patches:
        pos_2d = (
            raw_pos.to(dtype=torch.float32)
            .reshape(1, src_grid, src_grid, mod.vision_hidden_size)
            .permute(0, 3, 1, 2)
        )
        resized = F.interpolate(
            pos_2d,
            size=(mod.grid_size, mod.grid_size),
            mode="bilinear",
            align_corners=False,
        )
        resized = resized.permute(0, 2, 3, 1).reshape(
            mod.num_patches, mod.vision_hidden_size
        )
        with torch.no_grad():
          pos_target.copy_(
              resized.to(dtype=pos_target.dtype, device=pos_target.device)
          )
        populated.add(pos_target.data_ptr())
        consumed.add(pos_key)
      elif strict:
        raise ValueError(
            f"Non-square pos_embed shape {tuple(raw_pos.shape)} cannot be"
            f" interpolated to {expected_3d}."
        )
    elif strict:
      raise ValueError(
          f"Weight shape mismatch for '{pos_key}': checkpoint has"
          f" {tuple(raw_pos.shape)}, target expects {expected_3d}."
      )
  elif strict:
    raise KeyError(f"Missing checkpoint weight: '{pos_key}'.")

  for idx, layer in enumerate(mod.vision_model.encoder.layers):
    assert isinstance(layer, modeling_janus.JanusVisionEncoderLayer)
    bp = f"{vt}.blocks.{idx}"
    _copy_module_weights(
        layer.layer_norm1,
        f"{bp}.norm1",
        state_dict,
        weight_map,
        strict,
        populated,
        consumed,
    )
    attn = layer.self_attn
    qkv_w_key = f"{bp}.attn.qkv.weight"
    qkv_b_key = f"{bp}.attn.qkv.bias"
    prev_w = attn.q_proj.weight.data_ptr() in populated
    prev_b = (
        attn.q_proj.bias is not None
        and attn.q_proj.bias.data_ptr() in populated
    )
    weights_utils.copy_qkv_in_proj(
        attn.q_proj,
        attn.k_proj,
        attn.v_proj,
        state_dict,
        weight_map,
        qkv_w_key,
        qkv_b_key,
        strict=strict,
        populated=populated,
    )
    if not prev_w and attn.q_proj.weight.data_ptr() in populated:
      consumed.add(qkv_w_key)
    if (
        not prev_b
        and attn.q_proj.bias is not None
        and attn.q_proj.bias.data_ptr() in populated
    ):
      consumed.add(qkv_b_key)
    _copy_module_weights(
        attn.projection_layer,
        f"{bp}.attn.proj",
        state_dict,
        weight_map,
        strict,
        populated,
        consumed,
    )
    _copy_module_weights(
        layer.layer_norm2,
        f"{bp}.norm2",
        state_dict,
        weight_map,
        strict,
        populated,
        consumed,
    )
    _copy_module_weights(
        layer.mlp,
        f"{bp}.mlp",
        state_dict,
        weight_map,
        strict,
        populated,
        consumed,
    )

  _copy_module_weights(
      mod.vision_model.post_layernorm,
      f"{vt}.norm",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )
  _copy_module_weights(
      mod.aligner.fc1,
      "aligner.layers.0",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )
  _copy_module_weights(
      mod.aligner.hidden_layers[0],
      "aligner.layers.2",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )

  if strict:
    weights_utils.verify_all_parameters_populated(
        mod, populated, "vision_encoder"
    )
    verify_checkpoint_keys_consumed(
        state_dict,
        weight_map,
        consumed,
        "vision_encoder",
        prefixes=("vision_model.", "aligner."),
    )
  return consumed


def build_text_embedder(
    vocab_size: int = 102400, llm_hidden_size: int = 2048
) -> external_emb_module.LiteRTExportableModuleForEmbedder:
  """Builds an `export_hf` `LiteRTExportableModuleForEmbedder` for text IDs."""
  return external_emb_module.LiteRTExportableModuleForEmbedder(
      weights_utils.make_embedding(vocab_size, llm_hidden_size)
  )


def load_text_embedder_weights(
    mod: external_emb_module.LiteRTExportableModuleForEmbedder,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    strict: bool = True,
) -> set[str]:
  """Loads `language_model.model.embed_tokens.weight` into `mod`."""
  assert isinstance(mod.model, nn.Embedding)
  populated: set[int] = set()
  consumed: set[str] = set()
  copy_weight(
      mod.model.weight,
      state_dict,
      weight_map,
      "language_model.model.embed_tokens.weight",
      "model.embed_tokens.weight",
      strict=strict,
      populated=populated,
      consumed_keys=consumed,
  )
  if strict:
    weights_utils.verify_all_parameters_populated(
        mod, populated, "text_embedder"
    )
  return consumed


class JanusProVqImageEmbedderModule(nn.Module):
  """VQ image token embedder (`gen_embed` + `JanusVQVAEAlignerMLP`)."""

  def __init__(
      self,
      vq_codebook_size: int = 16384,
      vq_codebook_embed_dim: int = 8,
      llm_hidden_size: int = 2048,
  ):
    super().__init__()
    self.vq_codebook_size = vq_codebook_size
    self.vq_codebook_embed_dim = vq_codebook_embed_dim
    self.llm_hidden_size = llm_hidden_size
    self.generation_embeddings = weights_utils.make_embedding(
        vq_codebook_size, vq_codebook_embed_dim
    )
    vq_cfg = configuration_janus.JanusVQVAEConfig(
        embed_dim=vq_codebook_embed_dim,
        num_embeddings=vq_codebook_size,
        projection_dim=llm_hidden_size,
        num_hidden_layers=2,
        hidden_act="gelu",
    )
    self.generation_aligner = modeling_janus.JanusVQVAEAlignerMLP(vq_cfg)

  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    return self.generation_aligner(self.generation_embeddings(token_ids))


def load_vq_image_embedder_weights(
    mod: JanusProVqImageEmbedderModule,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    strict: bool = True,
) -> set[str]:
  """Loads `gen_embed.*` and `gen_aligner.*` weights into `mod`."""
  populated: set[int] = set()
  consumed: set[str] = set()
  copy_weight(
      mod.generation_embeddings.weight,
      state_dict,
      weight_map,
      "gen_embed.weight",
      strict=strict,
      populated=populated,
      consumed_keys=consumed,
  )
  _copy_module_weights(
      mod.generation_aligner.fc1,
      "gen_aligner.layers.0",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )
  _copy_module_weights(
      mod.generation_aligner.hidden_layers[0],
      "gen_aligner.layers.2",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )
  if strict:
    weights_utils.verify_all_parameters_populated(
        mod, populated, "vq_image_embedder"
    )
    verify_checkpoint_keys_consumed(
        state_dict,
        weight_map,
        consumed,
        "vq_image_embedder",
        prefixes=("gen_embed.", "gen_aligner."),
    )
  return consumed


def build_gen_head(
    llm_hidden_size: int = 2048,
    image_token_embed: int = 2048,
    vq_codebook_size: int = 16384,
) -> modeling_janus.JanusVQVAEHead:
  """Builds a `transformers` `JanusVQVAEHead` for autoregressive VQ logits."""
  vq_cfg = configuration_janus.JanusVQVAEConfig(
      num_embeddings=vq_codebook_size,
      image_token_embed_dim=llm_hidden_size,
      projection_dim=image_token_embed,
      hidden_act="gelu",
  )
  return modeling_janus.JanusVQVAEHead(vq_cfg)


def load_gen_head_weights(
    mod: modeling_janus.JanusVQVAEHead,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    strict: bool = True,
) -> set[str]:
  """Loads `gen_head.*` weights into `mod`."""
  populated: set[int] = set()
  consumed: set[str] = set()
  _copy_module_weights(
      mod.proj_out,
      "gen_head.output_mlp_projector",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )
  _copy_module_weights(
      mod.vision_head,
      "gen_head.vision_head",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )
  if strict:
    weights_utils.verify_all_parameters_populated(mod, populated, "gen_head")
    verify_checkpoint_keys_consumed(
        state_dict,
        weight_map,
        consumed,
        "gen_head",
        prefixes=("gen_head.",),
    )
  return consumed


class JanusProVqImageDecoderModule(nn.Module):
  """VQ-16 image decoder wrapping `JanusVQVAEVectorQuantizer` + `JanusVQVAEDecoder`."""

  def __init__(
      self,
      vq_codebook_size: int = 16384,
      vq_codebook_embed_dim: int = 8,
      z_channels: int = 256,
      base_channels: int = 128,
      ch_mult: Sequence[int] = (1, 1, 2, 2, 4),
      num_res_blocks: int = 2,
      image_size: int = 384,
      patch_size: int = 16,
  ):
    super().__init__()
    self.vq_codebook_size = vq_codebook_size
    self.vq_codebook_embed_dim = vq_codebook_embed_dim
    self.image_size = image_size
    self.patch_size = patch_size
    self.grid_size = image_size // patch_size
    self.num_image_tokens = self.grid_size * self.grid_size
    self.ch_mult = tuple(ch_mult)
    self.num_resolutions = len(self.ch_mult)
    self.num_res_blocks = num_res_blocks

    vq_cfg = configuration_janus.JanusVQVAEConfig(
        embed_dim=vq_codebook_embed_dim,
        num_embeddings=vq_codebook_size,
        latent_channels=z_channels,
        base_channels=base_channels,
        channel_multiplier=list(self.ch_mult),
        num_res_blocks=num_res_blocks,
        num_patches=self.grid_size,
        out_channels=3,
        dropout=0.0,
    )
    self.quantize = modeling_janus.JanusVQVAEVectorQuantizer(vq_cfg)
    self.post_quant_conv = nn.Conv2d(
        vq_codebook_embed_dim, z_channels, kernel_size=1, stride=1, padding=0
    )
    self.decoder = modeling_janus.JanusVQVAEDecoder(vq_cfg)

  def forward(self, vq_codes: torch.Tensor) -> torch.Tensor:
    codebook_entry = self.quantize.get_codebook_entry(vq_codes)
    return self.decoder(self.post_quant_conv(codebook_entry))


def load_vq_image_decoder_weights(
    mod: JanusProVqImageDecoderModule,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    strict: bool = True,
) -> set[str]:
  """Loads `gen_vision_model.{quantize,post_quant_conv,decoder}.*` into `mod`."""
  populated: set[int] = set()
  consumed: set[str] = set()
  gv = "gen_vision_model"
  copy_weight(
      mod.quantize.embedding.weight,
      state_dict,
      weight_map,
      f"{gv}.quantize.embedding.weight",
      strict=strict,
      populated=populated,
      consumed_keys=consumed,
  )
  _copy_module_weights(
      mod.post_quant_conv,
      f"{gv}.post_quant_conv",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )
  _copy_module_weights(
      mod.decoder.conv_in,
      f"{gv}.decoder.conv_in",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )
  for mid_idx, mid_block in enumerate((
      mod.decoder.mid.block_1,
      mod.decoder.mid.attn_1,
      mod.decoder.mid.block_2,
  )):
    _copy_module_weights(
        mid_block,
        f"{gv}.decoder.mid.{mid_idx}",
        state_dict,
        weight_map,
        strict,
        populated,
        consumed,
    )

  for level_idx, level in enumerate(mod.decoder.up):
    lp = f"{gv}.decoder.conv_blocks.{level_idx}"
    for r_idx, res_block in enumerate(level.block):
      _copy_module_weights(
          res_block,
          f"{lp}.res.{r_idx}",
          state_dict,
          weight_map,
          strict,
          populated,
          consumed,
      )
    for a_idx, attn_block in enumerate(level.attn):
      _copy_module_weights(
          attn_block,
          f"{lp}.attn.{a_idx}",
          state_dict,
          weight_map,
          strict,
          populated,
          consumed,
      )
    upsample = getattr(level, "upsample", None)
    if upsample is not None:
      _copy_module_weights(
          upsample,
          f"{lp}.upsample",
          state_dict,
          weight_map,
          strict,
          populated,
          consumed,
      )

  _copy_module_weights(
      mod.decoder.norm_out,
      f"{gv}.decoder.norm_out",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )
  _copy_module_weights(
      mod.decoder.conv_out,
      f"{gv}.decoder.conv_out",
      state_dict,
      weight_map,
      strict,
      populated,
      consumed,
  )
  if strict:
    weights_utils.verify_all_parameters_populated(
        mod, populated, "gen_vision_model.decoder"
    )
    verify_checkpoint_keys_consumed(
        state_dict,
        weight_map,
        consumed,
        "gen_vision_model.decoder",
        prefixes=(
            f"{gv}.quantize.",
            f"{gv}.post_quant_conv.",
            f"{gv}.decoder.",
        ),
    )
  return consumed
