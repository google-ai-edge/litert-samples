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
"""Janus-Pro (deepseek-ai/Janus-Pro-1B) Any2Any LiteRT exporter."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import dataclasses
import gc
import json
import os
import shutil
from typing import Any

import huggingface_hub
import litert_torch
from litert_torch._convert import interface as converter_utils
from litert_torch.generative.export_hf.core import attention as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import cache as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import patches as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import weights_loader as weights_loader_lib
from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
import safetensors
import torch
import transformers

from models.deepseek.janus_pro.export import modeling
from models.deepseek.janus_pro.runner import janus_pro_pb2
from litert_lm_builder.runtime.proto import any2any_metadata_pb2

JanusProVisionEncoderModule = modeling.JanusProVisionEncoderModule
JanusProVqImageEmbedderModule = modeling.JanusProVqImageEmbedderModule
JanusProVqImageDecoderModule = modeling.JanusProVqImageDecoderModule

JANUS_PRO_EXTRA_KWARGS: tuple[str, ...] = (
    "cfg_weight",
    "generate_image",
    "image_temperature",
    "include_decode_hidden_states",
    "llm_intermediate_size",
    "llm_num_layers",
    "targets",
    "vision_num_layers",
    "vq_decoder_base_channels",
    "vq_decoder_num_res_blocks",
    "vq_decoder_quantization_recipe",
    "vq_decoder_z_channels",
)
exportable_module_config.register_extra_kwargs_keys(JANUS_PRO_EXTRA_KWARGS)

_ALLOWED_CHECKPOINT_PREFIXES: tuple[str, ...] = (
    "vision_model.",
    "aligner.",
    "gen_embed.",
    "gen_aligner.",
    "gen_head.",
    "gen_vision_model.",
    "language_model.",
    "model.",
    "lm_head.",
)

_SIGLIP_PRESETS: dict[str, dict[str, int]] = {
    "siglip_large_patch16_384": {
        "image_size": 384,
        "patch_size": 16,
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "mlp_ratio": 4,
    },
    "siglip_so400m_patch14_384": {
        "image_size": 384,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 4,
    },
}

_DEFAULT_JANUS_PRO_CONFIG: dict[str, Any] = {
    "model_type": "multi_modality",
    "architectures": ["MultiModalityCausalLM"],
    "vision_config": {
        "cls": "CLIPVisionTower",
        "params": {
            "image_size": 384,
            "patch_size": 16,
            "width": 1024,
            "layers": 24,
            "heads": 16,
            "mlp_ratio": 4,
            "model_name": "siglip_large_patch16_384",
        },
    },
    "aligner_config": {
        "cls": "MlpProjector",
        "params": {
            "depth": 2,
            "input_dim": 1024,
            "n_embed": 2048,
            "projector_type": "mlp_gelu",
        },
    },
    "gen_vision_config": {
        "cls": "VQ-16",
        "params": {
            "image_token_size": 16384,
            "n_embed": 8,
            "z_channels": 256,
            "base_channels": 128,
            "num_res_blocks": 2,
        },
    },
    "gen_aligner_config": {
        "cls": "MlpProjector",
        "params": {
            "depth": 2,
            "input_dim": 8,
            "n_embed": 2048,
            "projector_type": "mlp_gelu",
        },
    },
    "gen_head_config": {
        "cls": "vision_head",
        "params": {
            "image_token_embed": 2048,
            "image_token_size": 16384,
            "n_embed": 2048,
        },
    },
    "language_config": {
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 102400,
        "max_position_embeddings": 16384,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "bos_token_id": 100000,
        "eos_token_id": 100001,
        "tie_word_embeddings": False,
    },
}


def resolve_targets(
    export_config: exportable_module_config.ExportableModuleConfig,
) -> set[str]:
  """Normalizes `extra_kwargs['targets']` into a set of target names."""
  raw_targets = export_config.extra_kwargs.get("targets", ["all"])
  if isinstance(raw_targets, str):
    return {t.strip() for t in raw_targets.split(",") if t.strip()}
  return set(raw_targets)


def resolve_generate_image(
    export_config: exportable_module_config.ExportableModuleConfig,
) -> bool:
  """Resolves and validates `generate_image` against targets and flags."""
  targets = resolve_targets(export_config)
  has_gen_targets = "all" in targets or {
      "vq_image_embedder",
      "gen_head",
      "vq_image_decoder",
  }.issubset(targets)
  include_hidden = bool(
      export_config.extra_kwargs.get("include_decode_hidden_states", True)
  )
  raw_gen_img = export_config.extra_kwargs.get("generate_image")
  if raw_gen_img is None:
    return False
  generate_image = bool(raw_gen_img)
  if generate_image:
    if not include_hidden:
      raise ValueError(
          "generate_image=True requires include_decode_hidden_states=True."
      )
    if not has_gen_targets:
      raise ValueError(
          "generate_image=True requires 'vq_image_embedder', 'gen_head', and"
          " 'vq_image_decoder' in targets."
      )
  return generate_image


_BaseGenerateModule = (
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder
)
_BasePrefillModule = (
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder
)


class _JanusProLlmGenerateModule(_BaseGenerateModule):
  """Generate module concatenating `[logits, last_hidden_state]` in `logits`.

  `JanusProLlmStage` inspects the length of the `decode` output buffer
  (`logits`): when its size equals `vocab_size + llm_hidden_size`, the runtime
  splits the trailing `llm_hidden_size` floats into `last_hidden_state` (used by
  `gen_head` during autoregressive visual generation) and keeps the leading
  `vocab_size` floats for text token sampling.
  """

  # pylint: disable=arguments-renamed
  def forward(
      self,
      embeddings: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: Any,
      mask: torch.Tensor,
      local_mask: torch.Tensor | None = None,
      **kwargs: Any,
  ) -> dict[str, Any]:
    include_hidden = bool(
        self.export_config.extra_kwargs.get(
            "include_decode_hidden_states", True
        )
    )
    if not include_hidden:
      return super().forward(
          embeddings, input_pos, kv_cache, mask, local_mask=local_mask, **kwargs
      )
    orig_verifier = self.export_verifier
    self.export_verifier = True
    try:
      result = super().forward(
          embeddings, input_pos, kv_cache, mask, local_mask=local_mask, **kwargs
      )
    finally:
      self.export_verifier = orig_verifier
    last_hidden = result["activations"].to(torch.float32)
    result["logits"] = torch.cat([result["logits"], last_hidden], dim=-1)
    if not orig_verifier:
      result.pop("activations", None)
    return result


def _export_tflite(
    mod: torch.nn.Module,
    sample_kwargs: dict[str, torch.Tensor],
    out_path: str,
    quant_recipe: str | None,
) -> str:
  """Converts `mod` to FP32 `.tflite`, releases memory, and applies quantization."""
  litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
  del mod
  gc.collect()
  return modeling.maybe_quantize(out_path, quant_recipe)


def _resolve_llm_checkpoint_keys(
    param_names: Sequence[str],
    available_keys: set[str],
    tie_word_embeddings: bool = False,
    strict: bool = True,
) -> set[str]:
  """Resolves LLaMA parameter names to checkpoint keys in `available_keys`."""
  matched: set[str] = set()
  missing: list[str] = []
  for param_name in param_names:
    candidates = [f"language_model.{param_name}", param_name]
    if tie_word_embeddings and param_name == "lm_head.weight":
      candidates.extend((
          "language_model.model.embed_tokens.weight",
          "model.embed_tokens.weight",
      ))
    found = next((c for c in candidates if c in available_keys), None)
    if found is not None:
      matched.add(found)
    elif strict:
      missing.append(param_name)
  if missing:
    raise KeyError(
        f"Missing checkpoint weights for llm_backbone parameters: {missing}"
    )
  return matched


class JanusPro:
  """deepseek-ai/Janus-Pro-1B unified multimodal understanding & generation exporter."""

  def __init__(
      self,
      model_path: str,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ):
    self.model_path = model_path
    self.model_dir = (
        model_path
        if os.path.exists(model_path)
        else huggingface_hub.snapshot_download(model_path)
    )
    self.export_config = export_config
    self._config: dict[str, Any] = copy.deepcopy(_DEFAULT_JANUS_PRO_CONFIG)
    self._load_config()
    self._state_dict: dict[str, torch.Tensor] = {}
    self._weight_map: dict[str, str] = {}
    self._consumed_keys: set[str] = set()
    if export_config is None or not export_config.use_random_weights:
      self._load_weights_if_available()

  def _load_config(self) -> None:
    """Loads and deep-merges `config.json` from `self.model_dir` if present."""
    cfg_path = os.path.join(self.model_dir, "config.json")
    if not os.path.exists(cfg_path):
      return
    with open(cfg_path, "r") as f:
      loaded = json.load(f)
    for k, v in loaded.items():
      if isinstance(v, dict) and isinstance(self._config.get(k), dict):
        if "params" in v and isinstance(self._config[k].get("params"), dict):
          merged_sub = dict(self._config[k])
          merged_params = dict(self._config[k]["params"])
          merged_params.update(v["params"])
          merged_sub.update(v)
          merged_sub["params"] = merged_params
          self._config[k] = merged_sub
        else:
          self._config[k].update(v)
      else:
        self._config[k] = v

  def _load_weights_if_available(self) -> None:
    """Indexes safetensors or loads PyTorch `.bin` weights from `self.model_dir`."""
    single_st = os.path.join(self.model_dir, "model.safetensors")
    index_st = os.path.join(self.model_dir, "model.safetensors.index.json")
    single_bin = os.path.join(self.model_dir, "pytorch_model.bin")
    index_bin = os.path.join(self.model_dir, "pytorch_model.bin.index.json")
    if os.path.exists(single_st):
      with safetensors.safe_open(single_st, framework="pt", device="cpu") as f:
        for k in f.keys():
          self._weight_map[k] = single_st
    elif os.path.exists(index_st):
      with open(index_st, "r") as f:
        index_data = json.load(f)
      for k, shard in index_data.get("weight_map", {}).items():
        shard_path = os.path.join(self.model_dir, shard)
        if os.path.exists(shard_path):
          self._weight_map[k] = shard_path
    elif os.path.exists(single_bin):
      loaded = torch.load(single_bin, map_location="cpu", weights_only=True)
      if isinstance(loaded, dict):
        self._state_dict.update(loaded)
    elif os.path.exists(index_bin):
      with open(index_bin, "r") as f:
        index_data = json.load(f)
      shards = sorted(set(index_data.get("weight_map", {}).values()))
      for shard in shards:
        shard_path = os.path.join(self.model_dir, shard)
        if os.path.exists(shard_path):
          loaded = torch.load(shard_path, map_location="cpu", weights_only=True)
          if isinstance(loaded, dict):
            self._state_dict.update(loaded)

  def _require_weights_or_random(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> None:
    if (
        not export_config.use_random_weights
        and not self._state_dict
        and not self._weight_map
    ):
      raise FileNotFoundError(
          f"No checkpoint weights found in {self.model_dir} and"
          " use_random_weights=False."
      )

  def verify_checkpoint_keys(self) -> None:
    """Verifies every checkpoint tensor key belongs to a known Janus-Pro module."""
    all_keys = set(self._state_dict) | set(self._weight_map)
    if not all_keys:
      raise FileNotFoundError(
          f"No checkpoint weights found in {self.model_dir}."
      )
    unexpected = [
        key
        for key in sorted(all_keys)
        if not key.startswith(modeling.IGNORED_CHECKPOINT_KEY_PREFIXES)
        and not key.startswith(_ALLOWED_CHECKPOINT_PREFIXES)
    ]
    if unexpected:
      raise ValueError(
          f"Unconsumed checkpoint keys in {self.model_dir}: {unexpected}"
      )

  def _vision_params(self) -> dict[str, Any]:
    v_cfg = self._config.get("vision_config", {})
    params = dict(v_cfg.get("params", v_cfg))
    model_name = str(params.get("model_name", "siglip_large_patch16_384"))
    preset = _SIGLIP_PRESETS.get(
        model_name, _SIGLIP_PRESETS["siglip_large_patch16_384"]
    )
    for k, val in preset.items():
      params.setdefault(k, val)
    return params

  def _language_params(self) -> dict[str, Any]:
    return dict(self._config.get("language_config", {}))

  def _gen_vision_params(self) -> dict[str, Any]:
    gv_cfg = self._config.get("gen_vision_config", {})
    return dict(gv_cfg.get("params", gv_cfg))

  def _gen_head_params(self) -> dict[str, Any]:
    gh_cfg = self._config.get("gen_head_config", {})
    return dict(gh_cfg.get("params", gh_cfg))

  def export_vision_encoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports SigLIP-Large-Patch16-384 + Aligner (`tf_lite_vision_encoder`)."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    v_params = self._vision_params()
    l_params = self._language_params()

    vision_hidden = int(
        v_params.get("width", v_params.get("hidden_size", 1024))
    )
    mlp_ratio = int(v_params.get("mlp_ratio", 4))
    vision_inter = int(
        v_params.get("intermediate_size", vision_hidden * mlp_ratio)
    )
    vision_heads = int(
        v_params.get("heads", v_params.get("num_attention_heads", 16))
    )
    default_vision_layers = (
        min(int(v_params.get("layers", 1)), 1)
        if export_config.use_random_weights
        else int(v_params.get("layers", 24))
    )
    vision_layers = int(
        export_config.extra_kwargs.get(
            "vision_num_layers", default_vision_layers
        )
    )
    image_size = int(v_params.get("image_size", 384))
    patch_size = int(v_params.get("patch_size", 16))
    llm_hidden = int(l_params.get("hidden_size", 2048))

    mod = modeling.JanusProVisionEncoderModule(
        vision_hidden_size=vision_hidden,
        vision_intermediate_size=vision_inter,
        vision_num_layers=vision_layers,
        vision_num_heads=vision_heads,
        image_size=image_size,
        patch_size=patch_size,
        llm_hidden_size=llm_hidden,
    ).eval()
    self._consumed_keys.update(
        modeling.load_vision_encoder_weights(
            mod, self._state_dict, self._weight_map, strict=strict
        )
    )

    sample_kwargs = {
        "pixel_values": torch.zeros(
            1, 3, image_size, image_size, dtype=torch.float32
        ),
    }
    quant_recipe = (
        export_config.vision_encoder_quantization_recipe
        if export_config.vision_encoder_quantization_recipe is not None
        else export_config.quantization_recipe
    )
    return _export_tflite(
        mod,
        sample_kwargs,
        os.path.join(output_dir, "vision_encoder_fp32.tflite"),
        quant_recipe,
    )

  def export_text_embedder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports DeepSeek-LLM text token embedder (`tf_lite_text_embedder`)."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    l_params = self._language_params()
    vocab_size = int(l_params.get("vocab_size", 102400))
    llm_hidden = int(l_params.get("hidden_size", 2048))

    mod = modeling.build_text_embedder(
        vocab_size=vocab_size, llm_hidden_size=llm_hidden
    ).eval()
    self._consumed_keys.update(
        modeling.load_text_embedder_weights(
            mod, self._state_dict, self._weight_map, strict=strict
        )
    )

    return _export_tflite(
        mod,
        {"token_ids": torch.zeros(1, 1, dtype=torch.int32)},
        os.path.join(output_dir, "text_embedder_fp32.tflite"),
        export_config.quantization_recipe,
    )

  def export_vq_image_embedder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports `gen_embed` + `gen_aligner` (`tf_lite_vq_image_embedder`)."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    gv_params = self._gen_vision_params()
    l_params = self._language_params()
    ga_params = self._config.get("gen_aligner_config", {}).get("params", {})

    vq_codebook_size = int(gv_params.get("image_token_size", 16384))
    vq_embed_dim = int(gv_params.get("n_embed", ga_params.get("input_dim", 8)))
    llm_hidden = int(l_params.get("hidden_size", 2048))

    mod = modeling.JanusProVqImageEmbedderModule(
        vq_codebook_size=vq_codebook_size,
        vq_codebook_embed_dim=vq_embed_dim,
        llm_hidden_size=llm_hidden,
    ).eval()
    self._consumed_keys.update(
        modeling.load_vq_image_embedder_weights(
            mod, self._state_dict, self._weight_map, strict=strict
        )
    )

    return _export_tflite(
        mod,
        {"token_ids": torch.zeros(1, 1, dtype=torch.int32)},
        os.path.join(output_dir, "vq_image_embedder_fp32.tflite"),
        export_config.quantization_recipe,
    )

  def export_gen_head(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports `gen_head` visual token predictor (`tf_lite_gen_head`)."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    gh_params = self._gen_head_params()
    gv_params = self._gen_vision_params()
    l_params = self._language_params()

    llm_hidden = int(
        l_params.get("hidden_size", gh_params.get("n_embed", 2048))
    )
    image_token_embed = int(gh_params.get("image_token_embed", llm_hidden))
    vq_codebook_size = int(
        gh_params.get(
            "image_token_size", gv_params.get("image_token_size", 16384)
        )
    )

    mod = modeling.build_gen_head(
        llm_hidden_size=llm_hidden,
        image_token_embed=image_token_embed,
        vq_codebook_size=vq_codebook_size,
    ).eval()
    self._consumed_keys.update(
        modeling.load_gen_head_weights(
            mod, self._state_dict, self._weight_map, strict=strict
        )
    )

    return _export_tflite(
        mod,
        {"hidden_states": torch.zeros(1, 1, llm_hidden, dtype=torch.float32)},
        os.path.join(output_dir, "gen_head_fp32.tflite"),
        export_config.quantization_recipe,
    )

  def export_vq_image_decoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports VQ-16 `decode_code` image decoder (`tf_lite_vq_image_decoder`)."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    gv_params = self._gen_vision_params()
    v_params = self._vision_params()

    vq_codebook_size = int(gv_params.get("image_token_size", 16384))
    vq_embed_dim = int(gv_params.get("n_embed", 8))
    conv_in_w = modeling.get_weight(
        self._state_dict,
        self._weight_map,
        "gen_vision_model.decoder.conv_in.weight",
    )
    norm_out_w = modeling.get_weight(
        self._state_dict,
        self._weight_map,
        "gen_vision_model.decoder.norm_out.weight",
    )
    if conv_in_w is not None:
      default_z_ch = int(conv_in_w.shape[1])
    elif export_config.use_random_weights:
      default_z_ch = 8
    else:
      default_z_ch = int(gv_params.get("z_channels", 256))

    if norm_out_w is not None:
      default_base_ch = int(norm_out_w.shape[0])
    elif export_config.use_random_weights:
      # `transformers.models.janus.modeling_janus.JanusVQVAEDecoder` uses
      # 32-group GroupNorm, so `base_channels` must be divisible by 32.
      default_base_ch = 32
    else:
      default_base_ch = int(gv_params.get("base_channels", 128))
    z_channels = int(
        export_config.extra_kwargs.get("vq_decoder_z_channels", default_z_ch)
    )
    base_channels = int(
        export_config.extra_kwargs.get(
            "vq_decoder_base_channels", default_base_ch
        )
    )
    default_res_blocks = (
        1
        if (export_config.use_random_weights and conv_in_w is None)
        else int(gv_params.get("num_res_blocks", 2))
    )
    num_res_blocks = int(
        export_config.extra_kwargs.get(
            "vq_decoder_num_res_blocks", default_res_blocks
        )
    )
    image_size = int(v_params.get("image_size", 384))
    patch_size = int(v_params.get("patch_size", 16))
    num_image_tokens = (image_size // patch_size) ** 2

    mod = modeling.JanusProVqImageDecoderModule(
        vq_codebook_size=vq_codebook_size,
        vq_codebook_embed_dim=vq_embed_dim,
        z_channels=z_channels,
        base_channels=base_channels,
        num_res_blocks=num_res_blocks,
        image_size=image_size,
        patch_size=patch_size,
    ).eval()
    self._consumed_keys.update(
        modeling.load_vq_image_decoder_weights(
            mod, self._state_dict, self._weight_map, strict=strict
        )
    )

    quant_recipe = export_config.extra_kwargs.get(
        "vq_decoder_quantization_recipe", export_config.quantization_recipe
    )
    return _export_tflite(
        mod,
        {"vq_codes": torch.zeros(1, num_image_tokens, dtype=torch.int32)},
        os.path.join(output_dir, "vq_image_decoder_fp32.tflite"),
        quant_recipe,
    )

  def export_llm_backbone(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports DeepSeek-LLM LLaMA backbone (`prefill_*` + `decode`) to TFLite."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    l_params = self._language_params()

    llm_hidden = int(l_params.get("hidden_size", 2048))
    default_inter = (
        min(int(l_params.get("intermediate_size", 5632)), 256)
        if export_config.use_random_weights
        else int(l_params.get("intermediate_size", 5632))
    )
    llm_inter = int(
        export_config.extra_kwargs.get("llm_intermediate_size", default_inter)
    )
    default_layers = (
        min(int(l_params.get("num_hidden_layers", 1)), 1)
        if export_config.use_random_weights
        else int(l_params.get("num_hidden_layers", 24))
    )
    llm_layers = int(
        export_config.extra_kwargs.get("llm_num_layers", default_layers)
    )
    llm_heads = int(l_params.get("num_attention_heads", 16))
    llm_kv_heads = int(l_params.get("num_key_value_heads", llm_heads))
    vocab_size = int(l_params.get("vocab_size", 102400))
    rms_norm_eps = float(l_params.get("rms_norm_eps", 1e-6))
    rope_theta = float(l_params.get("rope_theta", 10000.0))
    max_pos = int(l_params.get("max_position_embeddings", 16384))
    tie_word_embeddings = bool(l_params.get("tie_word_embeddings", False))

    llm_config = transformers.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=llm_hidden,
        intermediate_size=llm_inter,
        num_hidden_layers=llm_layers,
        num_attention_heads=llm_heads,
        num_key_value_heads=llm_kv_heads,
        hidden_act=str(l_params.get("hidden_act", "silu")),
        max_position_embeddings=max_pos,
        rms_norm_eps=rms_norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
        tie_word_embeddings=tie_word_embeddings,
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=int(l_params.get("bos_token_id", 100000)),
        eos_token_id=int(l_params.get("eos_token_id", 100001)),
    )
    # pylint: disable=protected-access
    llm_config._attn_implementation = "lrt_transposed_attention"
    # pylint: enable=protected-access

    has_st_weights = (
        "language_model.lm_head.weight" in self._weight_map
        or "lm_head.weight" in self._weight_map
        or (
            tie_word_embeddings
            and (
                "language_model.model.embed_tokens.weight" in self._weight_map
                or "model.embed_tokens.weight" in self._weight_map
            )
        )
    )
    use_v2 = strict and bool(export_config.use_v2) and has_st_weights
    base_loader = None
    if use_v2:
      with weights_loader_lib.init_params_on_meta():
        llama_model = transformers.LlamaForCausalLM(llm_config)
      llama_model.eval()
      base_loader = weights_loader_lib.HFCheckpointWeightsLoader(
          self.model_dir,
          tie_word_embeddings=tie_word_embeddings,
          extra_prefixes=("language_model.model.",),
      )
      weights_loader_lib.load_persistent_buffers_(llama_model, base_loader)
      param_names = [name for name, _ in llama_model.named_parameters()]
      consumed = _resolve_llm_checkpoint_keys(
          param_names,
          set(self._weight_map),
          tie_word_embeddings=tie_word_embeddings,
          strict=strict,
      )
      modeling.verify_checkpoint_keys_consumed(
          self._state_dict,
          self._weight_map,
          consumed,
          "llm_backbone",
          prefixes=("language_model.", "model.", "lm_head."),
      )
      self._consumed_keys.update(consumed)
    else:
      llama_model = (
          transformers.LlamaForCausalLM(llm_config).eval().to(torch.float32)
      )
      populated: set[int] = set()
      consumed = set()
      for name, param in llama_model.model.named_parameters():
        modeling.copy_weight(
            param,
            self._state_dict,
            self._weight_map,
            f"language_model.model.{name}",
            f"model.{name}",
            strict=strict,
            populated=populated,
            consumed_keys=consumed,
        )
      if tie_word_embeddings:
        lm_head_keys = (
            "language_model.lm_head.weight",
            "language_model.model.embed_tokens.weight",
            "lm_head.weight",
            "model.embed_tokens.weight",
        )
      else:
        lm_head_keys = ("language_model.lm_head.weight", "lm_head.weight")
      modeling.copy_weight(
          llama_model.lm_head.weight,
          self._state_dict,
          self._weight_map,
          *lm_head_keys,
          strict=strict,
          populated=populated,
          consumed_keys=consumed,
      )
      if strict:
        modeling.verify_all_parameters_populated(
            llama_model, populated, "llm_backbone"
        )
        modeling.verify_checkpoint_keys_consumed(
            self._state_dict,
            self._weight_map,
            consumed,
            "llm_backbone",
            prefixes=("language_model.", "model.", "lm_head."),
        )
      self._consumed_keys.update(consumed)

    cache_lengths = export_config.cache_lengths or [export_config.cache_length]
    prefill_lengths = export_config.prefill_lengths or [128]

    out_path = os.path.join(output_dir, "llm_backbone_fp32.tflite")
    intermediates_dir = os.path.join(output_dir, "llm_backbone_intermediates")

    llama_model.set_attn_implementation("lrt_transposed_attention")
    signatures = []
    for cache_len in cache_lengths:
      iter_config = dataclasses.replace(
          export_config,
          cache_length=cache_len,
          prefill_lengths=prefill_lengths,
          externalize_embedder=True,
      )
      prefill_module = _BasePrefillModule(llama_model, iter_config)
      decode_module = _JanusProLlmGenerateModule(llama_model, iter_config)
      for module in (prefill_module, decode_module):
        for sig_name, (inputs, dynamic_shapes) in module.get_sample_inputs(
            llm_config
        ).items():
          if len(cache_lengths) > 1:
            sig_name = f"{sig_name}_cache_{cache_len}"
          signatures.append((sig_name, module, inputs, dynamic_shapes))

    signature_modules = list({id(s[1]): s[1] for s in signatures}.values())
    lightweight = export_config.experimental_lightweight_conversion
    with weights_loader_lib.prepare_for_export(
        base_loader, *signature_modules
    ) as weights_loader:
      converter = converter_utils.Converter()
      for sig_name, module, inputs, _ in signatures:
        converter.add_signature(
            sig_name,
            module.eval(),
            sample_kwargs=inputs,
        )
      if use_v2:
        converter.convert(
            lightweight_conversion=lightweight,
            strict_export=False,
            use_v2=True,
            delete_in_memory_params=True,
            allow_reuse_intermediates=False,
            weights_loader=weights_loader,
            export_dir=intermediates_dir,
            output_file_path=out_path,
            _litert_converter_flags={
                "fold_fp16_casts_into_fully_connected": True,
            },
        )
      else:
        lrt_model = converter.convert(
            lightweight_conversion=lightweight,
            strict_export=False,
        )
        lrt_model.export(out_path)
        del lrt_model

    if os.path.exists(intermediates_dir):
      shutil.rmtree(intermediates_dir, ignore_errors=True)
    del llama_model
    gc.collect()
    return modeling.maybe_quantize(out_path, export_config.quantization_recipe)

  def export(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Exports Janus-Pro vision, text, LLM, and VQ generation submodels."""
    self.export_config = export_config
    resolve_generate_image(export_config)
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
      raise ValueError("Either output_dir or work_dir must be specified.")
    os.makedirs(output_dir, exist_ok=True)
    if not export_config.use_random_weights:
      self.verify_checkpoint_keys()

    artifacts: dict[str, str] = {}
    targets = resolve_targets(export_config)
    export_all = "all" in targets

    if (
        export_all and export_config.export_vision_encoder
    ) or "vision_encoder" in targets:
      artifacts["vision_encoder"] = self.export_vision_encoder(
          output_dir, export_config
      )
    if (
        export_all and export_config.externalize_embedder
    ) or "text_embedder" in targets:
      artifacts["text_embedder"] = self.export_text_embedder(
          output_dir, export_config
      )
    if export_all or "vq_image_embedder" in targets:
      artifacts["vq_image_embedder"] = self.export_vq_image_embedder(
          output_dir, export_config
      )
    if export_all or "llm_backbone" in targets:
      artifacts["llm_backbone"] = self.export_llm_backbone(
          output_dir, export_config
      )
    if export_all or "gen_head" in targets:
      artifacts["gen_head"] = self.export_gen_head(output_dir, export_config)
    if export_all or "vq_image_decoder" in targets:
      artifacts["vq_image_decoder"] = self.export_vq_image_decoder(
          output_dir, export_config
      )

    if (
        export_all
        and export_config.export_vision_encoder
        and not export_config.use_random_weights
    ):
      modeling.verify_checkpoint_keys_consumed(
          self._state_dict,
          self._weight_map,
          self._consumed_keys,
          "janus_pro",
          prefixes=None,
      )

    return artifacts

  def export_submodels(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Alias for `export`."""
    return self.export(export_config)

  def get_any2any_metadata(
      self,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> any2any_metadata_pb2.Any2AnyMetadata:
    """Builds `Any2AnyMetadata` packing `janus_pro_pb2.JanusPro` in `custom_model`."""
    cfg = export_config if export_config is not None else self.export_config
    v_params = self._vision_params()
    gv_params = self._gen_vision_params()
    l_params = self._language_params()
    generate_image = resolve_generate_image(cfg) if cfg is not None else False

    image_size = int(v_params.get("image_size", 384))
    patch_size = int(v_params.get("patch_size", 16))
    num_image_tokens = (image_size // max(1, patch_size)) ** 2
    cfg_weight = float(
        cfg.extra_kwargs.get("cfg_weight", 5.0) if cfg is not None else 5.0
    )
    image_temperature = float(
        cfg.extra_kwargs.get("image_temperature", 1.0)
        if cfg is not None
        else 1.0
    )

    janus_proto = janus_pro_pb2.JanusPro(
        llm_hidden_size=int(l_params.get("hidden_size", 2048)),
        llm_vocab_size=int(l_params.get("vocab_size", 102400)),
        vq_codebook_size=int(gv_params.get("image_token_size", 16384)),
        vq_codebook_embed_dim=int(gv_params.get("n_embed", 8)),
        image_size=image_size,
        patch_size=patch_size,
        num_image_tokens=num_image_tokens,
        cfg_weight=cfg_weight,
        image_temperature=image_temperature,
        generate_image=generate_image,
        max_new_tokens=512,
    )
    metadata = any2any_metadata_pb2.Any2AnyMetadata(
        system_prompt=(
            "You are a helpful language and vision assistant. You are able to"
            " understand the visual content that the user provides, and assist"
            " the user with a variety of tasks using natural language."
        ),
        supported_languages=["en-US", "zh-CN"],
    )
    metadata.any2any_model_type.custom_model.Pack(janus_proto)
    return metadata
