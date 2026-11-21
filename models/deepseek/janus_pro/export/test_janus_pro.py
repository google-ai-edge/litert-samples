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
"""Unit tests for Janus-Pro LiteRT exporter, modules, and .litertlm packaging."""

from __future__ import annotations

import io
import json
import os
import tempfile

from absl.testing import absltest
from litert_torch.generative.export_hf import export as export_script
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F

from models.deepseek.janus_pro import janus_pro_pb2
from models.deepseek.janus_pro.export import export_main as _  # pylint: disable=unused-import
from models.deepseek.janus_pro.export import extension as janus_ext
from models.deepseek.janus_pro.export import janus_pro
from models.deepseek.janus_pro.export import modeling
from ai_edge_litert.compiled_model import CompiledModel
from litert_lm_builder import litertlm_peek


def _write_tiny_janus_pro_config(model_dir: str) -> None:
  """Writes a tiny Janus-Pro config.json and tokenizer.json to `model_dir`."""
  os.makedirs(model_dir, exist_ok=True)
  with open(os.path.join(model_dir, "config.json"), "w") as f:
    json.dump(
        {
            "model_type": "multi_modality",
            "architectures": ["MultiModalityCausalLM"],
            "vision_config": {
                "cls": "CLIPVisionTower",
                "params": {
                    "image_size": 32,
                    "patch_size": 16,
                    "width": 32,
                    "layers": 1,
                    "heads": 2,
                    "mlp_ratio": 2,
                },
            },
            "aligner_config": {
                "cls": "MlpProjector",
                "params": {
                    "depth": 2,
                    "input_dim": 32,
                    "n_embed": 32,
                    "projector_type": "mlp_gelu",
                },
            },
            "gen_vision_config": {
                "cls": "VQ-16",
                "params": {
                    "image_token_size": 64,
                    "n_embed": 8,
                    "z_channels": 8,
                    "base_channels": 32,
                    "num_res_blocks": 1,
                },
            },
            "gen_aligner_config": {
                "cls": "MlpProjector",
                "params": {
                    "depth": 2,
                    "input_dim": 8,
                    "n_embed": 32,
                    "projector_type": "mlp_gelu",
                },
            },
            "gen_head_config": {
                "cls": "vision_head",
                "params": {
                    "image_token_embed": 32,
                    "image_token_size": 64,
                    "n_embed": 32,
                },
            },
            "language_config": {
                "model_type": "llama",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 64,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
        },
        f,
    )
  with open(os.path.join(model_dir, "tokenizer.json"), "w") as f:
    json.dump(
        {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "<bos>": 1, "<eos>": 2, "hello": 3},
                "unk_token": "<unk>",
            },
        },
        f,
    )


def _build_tiny_checkpoint_tensors() -> dict[str, torch.Tensor]:
  """Builds a complete synthetic checkpoint tensor dict for tiny Janus-Pro."""
  torch.manual_seed(42)
  vt = "vision_model.vision_tower"
  tensors: dict[str, torch.Tensor] = {
      # Vision encoder + aligner:
      f"{vt}.patch_embed.proj.weight": torch.randn(32, 3, 16, 16) * 0.1,
      f"{vt}.patch_embed.proj.bias": torch.randn(32) * 0.1,
      f"{vt}.pos_embed": torch.randn(1, 4, 32) * 0.1,
      f"{vt}.blocks.0.norm1.weight": torch.ones(32),
      f"{vt}.blocks.0.norm1.bias": torch.zeros(32),
      f"{vt}.blocks.0.attn.qkv.weight": torch.randn(96, 32) * 0.1,
      f"{vt}.blocks.0.attn.qkv.bias": torch.randn(96) * 0.1,
      f"{vt}.blocks.0.attn.proj.weight": torch.randn(32, 32) * 0.1,
      f"{vt}.blocks.0.attn.proj.bias": torch.randn(32) * 0.1,
      f"{vt}.blocks.0.norm2.weight": torch.ones(32),
      f"{vt}.blocks.0.norm2.bias": torch.zeros(32),
      f"{vt}.blocks.0.mlp.fc1.weight": torch.randn(64, 32) * 0.1,
      f"{vt}.blocks.0.mlp.fc1.bias": torch.randn(64) * 0.1,
      f"{vt}.blocks.0.mlp.fc2.weight": torch.randn(32, 64) * 0.1,
      f"{vt}.blocks.0.mlp.fc2.bias": torch.randn(32) * 0.1,
      f"{vt}.norm.weight": torch.ones(32),
      f"{vt}.norm.bias": torch.zeros(32),
      "aligner.layers.0.weight": torch.randn(32, 32) * 0.1,
      "aligner.layers.0.bias": torch.randn(32) * 0.1,
      "aligner.layers.2.weight": torch.randn(32, 32) * 0.1,
      "aligner.layers.2.bias": torch.randn(32) * 0.1,
      # VQ image embedder:
      "gen_embed.weight": torch.randn(64, 8) * 0.1,
      "gen_aligner.layers.0.weight": torch.randn(32, 8) * 0.1,
      "gen_aligner.layers.0.bias": torch.randn(32) * 0.1,
      "gen_aligner.layers.2.weight": torch.randn(32, 32) * 0.1,
      "gen_aligner.layers.2.bias": torch.randn(32) * 0.1,
      # Gen head:
      "gen_head.output_mlp_projector.weight": torch.randn(32, 32) * 0.1,
      "gen_head.output_mlp_projector.bias": torch.randn(32) * 0.1,
      "gen_head.vision_head.weight": torch.randn(64, 32) * 0.1,
      "gen_head.vision_head.bias": torch.randn(64) * 0.1,
      # LLaMA LLM backbone + text embedder:
      "language_model.model.embed_tokens.weight": torch.randn(64, 32) * 0.1,
      "language_model.model.layers.0.input_layernorm.weight": torch.ones(32),
      "language_model.model.layers.0.self_attn.q_proj.weight": (
          torch.randn(32, 32) * 0.1
      ),
      "language_model.model.layers.0.self_attn.k_proj.weight": (
          torch.randn(32, 32) * 0.1
      ),
      "language_model.model.layers.0.self_attn.v_proj.weight": (
          torch.randn(32, 32) * 0.1
      ),
      "language_model.model.layers.0.self_attn.o_proj.weight": (
          torch.randn(32, 32) * 0.1
      ),
      "language_model.model.layers.0.post_attention_layernorm.weight": (
          torch.ones(32)
      ),
      "language_model.model.layers.0.mlp.gate_proj.weight": (
          torch.randn(64, 32) * 0.1
      ),
      "language_model.model.layers.0.mlp.up_proj.weight": (
          torch.randn(64, 32) * 0.1
      ),
      "language_model.model.layers.0.mlp.down_proj.weight": (
          torch.randn(32, 64) * 0.1
      ),
      "language_model.model.norm.weight": torch.ones(32),
      "language_model.lm_head.weight": torch.randn(64, 32) * 0.1,
  }

  # Populate VQ-16 decoder weights explicitly by checkpoint key schema:
  gv = "gen_vision_model"
  base_ch = 32
  top_ch = base_ch * 4
  tensors[f"{gv}.quantize.embedding.weight"] = torch.randn(64, 8) * 0.1
  tensors[f"{gv}.post_quant_conv.weight"] = torch.randn(8, 8, 1, 1) * 0.1
  tensors[f"{gv}.post_quant_conv.bias"] = torch.zeros(8)
  tensors[f"{gv}.decoder.conv_in.weight"] = torch.randn(top_ch, 8, 3, 3) * 0.1
  tensors[f"{gv}.decoder.conv_in.bias"] = torch.zeros(top_ch)

  def add_res(prefix: str, in_ch: int, out_ch: int) -> None:
    tensors[f"{prefix}.norm1.weight"] = torch.ones(in_ch)
    tensors[f"{prefix}.norm1.bias"] = torch.zeros(in_ch)
    tensors[f"{prefix}.conv1.weight"] = torch.randn(out_ch, in_ch, 3, 3) * 0.1
    tensors[f"{prefix}.conv1.bias"] = torch.zeros(out_ch)
    tensors[f"{prefix}.norm2.weight"] = torch.ones(out_ch)
    tensors[f"{prefix}.norm2.bias"] = torch.zeros(out_ch)
    tensors[f"{prefix}.conv2.weight"] = torch.randn(out_ch, out_ch, 3, 3) * 0.1
    tensors[f"{prefix}.conv2.bias"] = torch.zeros(out_ch)
    if in_ch != out_ch:
      tensors[f"{prefix}.nin_shortcut.weight"] = (
          torch.randn(out_ch, in_ch, 1, 1) * 0.1
      )
      tensors[f"{prefix}.nin_shortcut.bias"] = torch.zeros(out_ch)

  def add_attn(prefix: str, ch: int) -> None:
    tensors[f"{prefix}.norm.weight"] = torch.ones(ch)
    tensors[f"{prefix}.norm.bias"] = torch.zeros(ch)
    for proj in ("q", "k", "v", "proj_out"):
      tensors[f"{prefix}.{proj}.weight"] = torch.randn(ch, ch, 1, 1) * 0.1
      tensors[f"{prefix}.{proj}.bias"] = torch.zeros(ch)

  add_res(f"{gv}.decoder.mid.0", top_ch, top_ch)
  add_attn(f"{gv}.decoder.mid.1", top_ch)
  add_res(f"{gv}.decoder.mid.2", top_ch, top_ch)

  ch_mult = (1, 1, 2, 2, 4)
  block_in = top_ch
  for level_idx, mult in enumerate(reversed(ch_mult)):
    block_out = base_ch * mult
    lp = f"{gv}.decoder.conv_blocks.{level_idx}"
    for r_idx in range(2):
      add_res(
          f"{lp}.res.{r_idx}",
          block_in if r_idx == 0 else block_out,
          block_out,
      )
    if level_idx == 0:
      for a_idx in range(2):
        add_attn(f"{lp}.attn.{a_idx}", block_out)
    if level_idx < len(ch_mult) - 1:
      tensors[f"{lp}.upsample.conv.weight"] = (
          torch.randn(block_out, block_out, 3, 3) * 0.1
      )
      tensors[f"{lp}.upsample.conv.bias"] = torch.zeros(block_out)
    block_in = block_out

  tensors[f"{gv}.decoder.norm_out.weight"] = torch.ones(base_ch)
  tensors[f"{gv}.decoder.norm_out.bias"] = torch.zeros(base_ch)
  tensors[f"{gv}.decoder.conv_out.weight"] = torch.randn(3, base_ch, 3, 3) * 0.1
  tensors[f"{gv}.decoder.conv_out.bias"] = torch.zeros(3)
  return tensors


class JanusProTest(absltest.TestCase):
  """Tests for Janus-Pro modules, weight loading, export, and .litertlm bundle."""

  def test_extension_auto_discovery_matching_section_types_and_dry_run(self):
    """Verifies extension registration, matching, section types, and dry_run."""
    ext = extension_lib.get_model_extension("janus_pro")
    self.assertIsInstance(ext, janus_ext.JanusProExtension)
    assert ext is not None

    self.assertEqual(
        ext.get_litert_lm_section_types(),
        {
            "vq_image_embedder": "tf_lite_vq_image_embedder",
            "gen_head": "tf_lite_gen_head",
            "vq_image_decoder": "tf_lite_vq_image_decoder",
        },
    )
    self.assertTrue(
        ext.match_checkpoint(
            "/tmp/some_dir",
            config_dict={"architectures": ["MultiModalityCausalLM"]},
        )
    )
    self.assertTrue(
        ext.match_checkpoint(
            "/tmp/some_dir",
            config_dict={
                "model_type": "multi_modality",
                "gen_vision_config": {},
                "gen_head_config": {},
            },
        )
    )
    self.assertTrue(
        ext.match_checkpoint("deepseek-ai/Janus-Pro-1B", config_dict={})
    )
    self.assertFalse(
        ext.match_checkpoint(
            "deepseek-ai/Janus-Pro-1B",
            task="text_generation",
            config_dict={},
        )
    )
    self.assertFalse(ext.match_checkpoint("/tmp/other", config_dict={}))
    defaults = ext.get_export_defaults({})
    self.assertEqual(
        defaults.task, exportable_module_config.ExportTask.ANY_TO_ANY
    )
    self.assertIs(ext.get_any2any_model_cls("janus_pro"), janus_pro.JanusPro)

    with tempfile.TemporaryDirectory() as tmp_dir:
      _write_tiny_janus_pro_config(tmp_dir)
      plan = export_script.export(model=tmp_dir, dry_run=True)
      self.assertIsNotNone(plan)
      self.assertEqual(plan.task, "any_to_any")
      self.assertEqual(plan.model_type, "multi_modality")

  def test_real_janus_pro_1b_schema_and_strict_checkpoint_verification(self):
    """Verifies 1B parameter counts, strict loaders, and verify_checkpoint_keys."""
    with torch.device("meta"):
      vis_1b = modeling.JanusProVisionEncoderModule(
          vision_hidden_size=1024,
          vision_intermediate_size=4096,
          vision_num_layers=24,
          vision_num_heads=16,
          image_size=384,
          patch_size=16,
          llm_hidden_size=2048,
      )
      vq_dec_1b = modeling.JanusProVqImageDecoderModule(
          vq_codebook_size=16384,
          vq_codebook_embed_dim=8,
          z_channels=8,
          base_channels=128,
          num_res_blocks=2,
          image_size=384,
          patch_size=16,
      )
      vq_emb_1b = modeling.JanusProVqImageEmbedderModule(
          vq_codebook_size=16384, vq_codebook_embed_dim=8, llm_hidden_size=2048
      )
      gen_head_1b = modeling.build_gen_head(
          llm_hidden_size=2048, image_token_embed=2048, vq_codebook_size=16384
      )
    # 2 (patch_embed) + 1 (pos_embed) + 24 * 16 (unfused Q/K/V ViT layers)
    # + 2 (post_layernorm) + 4 (aligner) = 393 parameters.
    self.assertLen(list(vis_1b.named_parameters()), 393)
    self.assertLen(list(vq_dec_1b.named_parameters()), 197)
    self.assertLen(list(vq_emb_1b.named_parameters()), 5)
    self.assertLen(list(gen_head_1b.named_parameters()), 4)

    tensors = _build_tiny_checkpoint_tensors()
    gen_head = modeling.build_gen_head(
        llm_hidden_size=32, image_token_embed=32, vq_codebook_size=64
    )
    missing_tensors = dict(tensors)
    del missing_tensors["gen_head.vision_head.weight"]
    with self.assertRaises(KeyError):
      modeling.load_gen_head_weights(gen_head, missing_tensors, {}, strict=True)

    bad_shape_tensors = dict(tensors)
    bad_shape_tensors["gen_head.vision_head.weight"] = torch.randn(16, 32)
    with self.assertRaises(ValueError):
      modeling.load_gen_head_weights(
          gen_head, bad_shape_tensors, {}, strict=True
      )

    extra_key_tensors = dict(tensors)
    extra_key_tensors["gen_head.unexpected_layer.weight"] = torch.randn(32, 32)
    with self.assertRaises(ValueError):
      modeling.load_gen_head_weights(
          gen_head, extra_key_tensors, {}, strict=True
      )

    ignored_prefix_tensors = dict(tensors)
    ignored_prefix_tensors["vision_model.vision_tower.attn_pool.q.weight"] = (
        torch.randn(32, 32)
    )
    ignored_prefix_tensors["gen_vision_model.encoder.conv_in.weight"] = (
        torch.randn(8, 3, 3, 3)
    )
    ignored_prefix_tensors["gen_vision_model.quant_conv.weight"] = torch.randn(
        8, 8, 1, 1
    )
    ignored_prefix_tensors["gen_vision_model.quantize.codebook_used"] = (
        torch.zeros(64, dtype=torch.int32)
    )
    consumed_gh = modeling.load_gen_head_weights(
        gen_head, ignored_prefix_tensors, {}, strict=True
    )
    self.assertLen(consumed_gh, 4)

    with tempfile.TemporaryDirectory() as tmp_dir:
      # Single safetensors with ignored prefixes passes verify_checkpoint_keys.
      st_dir = os.path.join(tmp_dir, "single_st")
      _write_tiny_janus_pro_config(st_dir)
      safetensors.torch.save_file(
          ignored_prefix_tensors, os.path.join(st_dir, "model.safetensors")
      )
      janus_pro.JanusPro(st_dir).verify_checkpoint_keys()

      # Sharded safetensors with unexpected top-level key fails verification.
      sharded_dir = os.path.join(tmp_dir, "sharded_st")
      _write_tiny_janus_pro_config(sharded_dir)
      keys = sorted(tensors)
      mid = len(keys) // 2
      shard1 = {k: tensors[k] for k in keys[:mid]}
      shard2 = {k: tensors[k] for k in keys[mid:]}
      shard2["unknown_top_level.weight"] = torch.randn(4)
      safetensors.torch.save_file(
          shard1, os.path.join(sharded_dir, "model-00001-of-00002.safetensors")
      )
      safetensors.torch.save_file(
          shard2, os.path.join(sharded_dir, "model-00002-of-00002.safetensors")
      )
      weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
      weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2})
      with open(
          os.path.join(sharded_dir, "model.safetensors.index.json"), "w"
      ) as f:
        json.dump({"weight_map": weight_map}, f)
      with self.assertRaises(ValueError):
        janus_pro.JanusPro(sharded_dir).verify_checkpoint_keys()

  def test_target_and_generate_image_validation(self):
    """Verifies target resolution and generate_image validation errors."""
    cfg_default = exportable_module_config.ExportableModuleConfig(
        model="dummy", output_dir="dummy"
    )
    self.assertEqual(janus_pro.resolve_targets(cfg_default), {"all"})
    self.assertFalse(janus_pro.resolve_generate_image(cfg_default))

    cfg_gen = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        output_dir="dummy",
        extra_kwargs={"generate_image": True},
    )
    self.assertTrue(janus_pro.resolve_generate_image(cfg_gen))

    cfg_conflict_hidden = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        output_dir="dummy",
        extra_kwargs={
            "generate_image": True,
            "include_decode_hidden_states": False,
        },
    )
    with self.assertRaises(ValueError):
      janus_pro.resolve_generate_image(cfg_conflict_hidden)

    cfg_conflict_targets = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        output_dir="dummy",
        extra_kwargs={
            "generate_image": True,
            "targets": "vision_encoder,llm_backbone",
        },
    )
    with self.assertRaises(ValueError):
      janus_pro.resolve_generate_image(cfg_conflict_targets)

  def test_submodel_litert_numerical_parity_against_pytorch(self):
    """Verifies exported LiteRT submodels match PyTorch reference numerically."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      _write_tiny_janus_pro_config(tmp_dir)
      tensors = _build_tiny_checkpoint_tensors()
      safetensors.torch.save_file(
          tensors, os.path.join(tmp_dir, "model.safetensors")
      )
      out_dir = os.path.join(tmp_dir, "out")
      cfg = exportable_module_config.ExportableModuleConfig(
          model=tmp_dir,
          output_dir=out_dir,
          work_dir=out_dir,
          task=exportable_module_config.ExportTask.ANY_TO_ANY,
          quantization_recipe="none",
          vision_encoder_quantization_recipe="none",
          use_random_weights=False,
          extra_kwargs={
              "targets": (
                  "vision_encoder,text_embedder,vq_image_embedder,"
                  "gen_head,vq_image_decoder"
              )
          },
      )
      model_obj = janus_pro.JanusPro(tmp_dir, export_config=cfg)
      artifacts = model_obj.export(cfg)

      # 1. Vision encoder parity
      vis_pt = modeling.JanusProVisionEncoderModule(
          vision_hidden_size=32,
          vision_intermediate_size=64,
          vision_num_layers=1,
          vision_num_heads=2,
          image_size=32,
          patch_size=16,
          llm_hidden_size=32,
      ).eval()
      modeling.load_vision_encoder_weights(vis_pt, tensors, {}, strict=True)
      pixel_in = torch.randn(1, 3, 32, 32, dtype=torch.float32)
      with torch.no_grad():
        vis_ref = vis_pt(pixel_in).numpy().reshape(-1)
      cm_vis = CompiledModel.from_file(artifacts["vision_encoder"])
      in_b, out_b = cm_vis.create_input_buffers(
          0
      ), cm_vis.create_output_buffers(0)
      in_b[0].write(pixel_in.numpy())
      cm_vis.run_by_index(0, in_b, out_b)
      np.testing.assert_allclose(
          out_b[0].read(vis_ref.size, np.float32), vis_ref, atol=1e-4, rtol=1e-3
      )

      # 2. Text embedder parity (`LiteRTExportableModuleForEmbedder`)
      txt_pt = modeling.build_text_embedder(
          vocab_size=64, llm_hidden_size=32
      ).eval()
      modeling.load_text_embedder_weights(txt_pt, tensors, {}, strict=True)
      tok_in = torch.tensor([[7]], dtype=torch.int32)
      with torch.no_grad():
        txt_ref = txt_pt(tok_in)["embeddings"].numpy().reshape(-1)
      cm_txt = CompiledModel.from_file(artifacts["text_embedder"])
      in_b, out_b = cm_txt.create_input_buffers(
          0
      ), cm_txt.create_output_buffers(0)
      in_b[0].write(tok_in.numpy())
      cm_txt.run_by_index(0, in_b, out_b)
      np.testing.assert_allclose(
          out_b[0].read(txt_ref.size, np.float32), txt_ref, atol=1e-5, rtol=1e-4
      )

      # 3. VQ image embedder parity
      vq_emb_pt = modeling.JanusProVqImageEmbedderModule(
          vq_codebook_size=64, vq_codebook_embed_dim=8, llm_hidden_size=32
      ).eval()
      modeling.load_vq_image_embedder_weights(
          vq_emb_pt, tensors, {}, strict=True
      )
      vq_tok_in = torch.tensor([[5]], dtype=torch.int32)
      with torch.no_grad():
        vq_emb_ref = vq_emb_pt(vq_tok_in).numpy().reshape(-1)
      cm_vqe = CompiledModel.from_file(artifacts["vq_image_embedder"])
      in_b, out_b = cm_vqe.create_input_buffers(
          0
      ), cm_vqe.create_output_buffers(0)
      in_b[0].write(vq_tok_in.numpy())
      cm_vqe.run_by_index(0, in_b, out_b)
      np.testing.assert_allclose(
          out_b[0].read(vq_emb_ref.size, np.float32),
          vq_emb_ref,
          atol=1e-4,
          rtol=1e-3,
      )

      # 4. Gen head parity (`transformers.models.janus.JanusVQVAEHead`)
      gh_pt = modeling.build_gen_head(
          llm_hidden_size=32, image_token_embed=32, vq_codebook_size=64
      ).eval()
      modeling.load_gen_head_weights(gh_pt, tensors, {}, strict=True)
      hid_in = torch.randn(1, 1, 32, dtype=torch.float32)
      with torch.no_grad():
        gh_ref = gh_pt(hid_in).numpy().reshape(-1)
      cm_gh = CompiledModel.from_file(artifacts["gen_head"])
      in_b, out_b = cm_gh.create_input_buffers(0), cm_gh.create_output_buffers(
          0
      )
      in_b[0].write(hid_in.numpy())
      cm_gh.run_by_index(0, in_b, out_b)
      np.testing.assert_allclose(
          out_b[0].read(gh_ref.size, np.float32), gh_ref, atol=1e-4, rtol=1e-3
      )

      # 5. VQ image decoder parity (`JanusVQVAE` quantizer + decoder)
      vq_dec_pt = modeling.JanusProVqImageDecoderModule(
          vq_codebook_size=64,
          vq_codebook_embed_dim=8,
          z_channels=8,
          base_channels=32,
          num_res_blocks=1,
          image_size=32,
          patch_size=16,
      ).eval()
      modeling.load_vq_image_decoder_weights(
          vq_dec_pt, tensors, {}, strict=True
      )
      vq_codes_in = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
      with torch.no_grad():
        vq_dec_ref = vq_dec_pt(vq_codes_in).numpy().reshape(-1)
      cm_vqd = CompiledModel.from_file(artifacts["vq_image_decoder"])
      in_b, out_b = cm_vqd.create_input_buffers(
          0
      ), cm_vqd.create_output_buffers(0)
      in_b[0].write(vq_codes_in.numpy())
      cm_vqd.run_by_index(0, in_b, out_b)
      np.testing.assert_allclose(
          out_b[0].read(vq_dec_ref.size, np.float32),
          vq_dec_ref,
          atol=1e-4,
          rtol=1e-3,
      )

  def test_full_litertlm_bundle_custom_model_metadata_and_safetensors_v2(self):
    """Verifies custom_model metadata packing, .litertlm bundling, and v2 loader."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      # 1. Safetensors full export (exercises use_v2=True streaming weights
      #    loader + full checkpoint key consumption + .litertlm packaging):
      st_dir = os.path.join(tmp_dir, "st_ckpt")
      _write_tiny_janus_pro_config(st_dir)
      tensors = _build_tiny_checkpoint_tensors()
      safetensors.torch.save_file(
          tensors, os.path.join(st_dir, "model.safetensors")
      )

      bundle_dir = os.path.join(tmp_dir, "st_bundle")
      os.makedirs(bundle_dir, exist_ok=True)
      export_script.export(
          model=st_dir,
          output_dir=bundle_dir,
          task="any_to_any",
          quantization_recipe="none",
          vision_encoder_quantization_recipe="none",
          use_random_weights=False,
          use_v2=True,
          cache_length=16,
          prefill_lengths=[8],
          bundle_litert_lm=True,
      )
      bundle_path = os.path.join(bundle_dir, "model.litertlm")
      self.assertTrue(os.path.exists(bundle_path))

      buf = io.StringIO()
      litertlm_peek.peek_litertlm_file(bundle_path, None, buf)
      dump_text = buf.getvalue()
      for expected_section in (
          "Any2AnyMetadataProto",
          "HF_Tokenizer_Zlib",
          "tf_lite_vision_encoder",
          "tf_lite_text_embedder",
          "tf_lite_vq_image_embedder",
          "tf_lite_llm_backbone",
          "tf_lite_gen_head",
          "tf_lite_vq_image_decoder",
      ):
        self.assertIn(expected_section, dump_text)

      # Verify custom_model packs prototypes.litert_models.janus_pro.JanusPro:
      meta_cfg = exportable_module_config.ExportableModuleConfig(
          model=st_dir,
          output_dir=bundle_dir,
          task=exportable_module_config.ExportTask.ANY_TO_ANY,
          use_random_weights=True,
      )
      model_obj = janus_pro.JanusPro(st_dir, export_config=meta_cfg)
      meta = model_obj.get_any2any_metadata(meta_cfg)
      self.assertTrue(meta.any2any_model_type.HasField("custom_model"))
      unpacked = janus_pro_pb2.JanusPro()
      self.assertTrue(meta.any2any_model_type.custom_model.Unpack(unpacked))
      self.assertEqual(unpacked.llm_hidden_size, 32)
      self.assertEqual(unpacked.llm_vocab_size, 64)
      self.assertEqual(unpacked.vq_codebook_size, 64)
      self.assertEqual(unpacked.vq_codebook_embed_dim, 8)
      self.assertEqual(unpacked.image_size, 32)
      self.assertEqual(unpacked.patch_size, 16)
      self.assertEqual(unpacked.num_image_tokens, 4)

      # 2. Verify decode signature concatenates [logits(64), hidden(32)] = 96
      #    and 2D pos_embed bilinear interpolation works on safetensors:
      interp_dir = os.path.join(tmp_dir, "interp_ckpt")
      _write_tiny_janus_pro_config(interp_dir)
      interp_tensors = dict(tensors)
      interp_tensors["vision_model.vision_tower.pos_embed"] = torch.randn(
          1, 16, 32
      )
      safetensors.torch.save_file(
          interp_tensors, os.path.join(interp_dir, "model.safetensors")
      )
      interp_out = os.path.join(tmp_dir, "interp_out")
      interp_cfg = exportable_module_config.ExportableModuleConfig(
          model=interp_dir,
          output_dir=interp_out,
          work_dir=interp_out,
          task=exportable_module_config.ExportTask.ANY_TO_ANY,
          quantization_recipe="none",
          use_random_weights=False,
          cache_length=16,
          prefill_lengths=[8],
          extra_kwargs={"targets": "vision_encoder,llm_backbone"},
      )
      interp_model = janus_pro.JanusPro(interp_dir, export_config=interp_cfg)
      interp_artifacts = interp_model.export(interp_cfg)
      self.assertTrue(os.path.exists(interp_artifacts["vision_encoder"]))
      cm_llm = CompiledModel.from_file(interp_artifacts["llm_backbone"])
      out_details = cm_llm.get_output_tensor_details("decode")
      self.assertEqual(list(out_details["logits"]["shape"]), [1, 1, 96])

  def test_transformers_janus_equivalence_and_int8_export(self):
    """Proves 1e-5 parity vs DeepSeek equations and dynamic_wi8_afp32 export."""
    tensors = _build_tiny_checkpoint_tensors()

    # 1. Exact numerical equivalence of `transformers.models.janus` SigLIP +
    #    Aligner against DeepSeek's fused-QKV timm ViT equations.
    vis_mod = modeling.JanusProVisionEncoderModule(
        vision_hidden_size=32,
        vision_intermediate_size=64,
        vision_num_layers=1,
        vision_num_heads=2,
        image_size=32,
        patch_size=16,
        llm_hidden_size=32,
    ).eval()
    modeling.load_vision_encoder_weights(vis_mod, tensors, {}, strict=True)
    pixel_in = torch.randn(1, 3, 32, 32, dtype=torch.float32)

    vt = "vision_model.vision_tower"
    with torch.no_grad():
      mod_vis_out = vis_mod(pixel_in).numpy()
      x = F.conv2d(
          pixel_in,
          tensors[f"{vt}.patch_embed.proj.weight"],
          tensors[f"{vt}.patch_embed.proj.bias"],
          stride=16,
      )
      x = x.flatten(2).transpose(1, 2) + tensors[f"{vt}.pos_embed"]
      h = F.layer_norm(
          x,
          (32,),
          tensors[f"{vt}.blocks.0.norm1.weight"],
          tensors[f"{vt}.blocks.0.norm1.bias"],
          eps=1e-6,
      )
      qkv = (
          F.linear(
              h,
              tensors[f"{vt}.blocks.0.attn.qkv.weight"],
              tensors[f"{vt}.blocks.0.attn.qkv.bias"],
          )
          .reshape(1, 4, 3, 2, 16)
          .permute(2, 0, 3, 1, 4)
      )
      q, k, v = qkv[0], qkv[1], qkv[2]
      attn = F.softmax((q * (16**-0.5)) @ k.transpose(-2, -1), dim=-1)
      attn_out = (attn @ v).transpose(1, 2).reshape(1, 4, 32)
      x = x + F.linear(
          attn_out,
          tensors[f"{vt}.blocks.0.attn.proj.weight"],
          tensors[f"{vt}.blocks.0.attn.proj.bias"],
      )
      h2 = F.layer_norm(
          x,
          (32,),
          tensors[f"{vt}.blocks.0.norm2.weight"],
          tensors[f"{vt}.blocks.0.norm2.bias"],
          eps=1e-6,
      )
      mlp_h = F.gelu(
          F.linear(
              h2,
              tensors[f"{vt}.blocks.0.mlp.fc1.weight"],
              tensors[f"{vt}.blocks.0.mlp.fc1.bias"],
          )
      )
      x = x + F.linear(
          mlp_h,
          tensors[f"{vt}.blocks.0.mlp.fc2.weight"],
          tensors[f"{vt}.blocks.0.mlp.fc2.bias"],
      )
      x = F.layer_norm(
          x,
          (32,),
          tensors[f"{vt}.norm.weight"],
          tensors[f"{vt}.norm.bias"],
          eps=1e-6,
      )
      ref_vis_out = F.linear(
          F.gelu(
              F.linear(
                  x,
                  tensors["aligner.layers.0.weight"],
                  tensors["aligner.layers.0.bias"],
              )
          ),
          tensors["aligner.layers.2.weight"],
          tensors["aligner.layers.2.bias"],
      ).numpy()
    np.testing.assert_allclose(mod_vis_out, ref_vis_out, atol=1e-5, rtol=1e-5)

    # 2. Exact numerical equivalence of `JanusProVqImageDecoderModule` against
    #    DeepSeek VQ-16 equations (`F.normalize`, `GroupNorm(32)+SiLU`, attn).
    vq_mod = modeling.JanusProVqImageDecoderModule(
        vq_codebook_size=64,
        vq_codebook_embed_dim=8,
        z_channels=8,
        base_channels=32,
        num_res_blocks=1,
        image_size=32,
        patch_size=16,
    ).eval()
    modeling.load_vq_image_decoder_weights(vq_mod, tensors, {}, strict=True)
    vq_codes = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    gv = "gen_vision_model"

    def run_res(t: torch.Tensor, prefix: str) -> torch.Tensor:
      h_res = F.silu(
          F.group_norm(
              t,
              32,
              tensors[f"{prefix}.norm1.weight"],
              tensors[f"{prefix}.norm1.bias"],
              eps=1e-6,
          )
      )
      h_res = F.conv2d(
          h_res,
          tensors[f"{prefix}.conv1.weight"],
          tensors[f"{prefix}.conv1.bias"],
          padding=1,
      )
      h_res = F.silu(
          F.group_norm(
              h_res,
              32,
              tensors[f"{prefix}.norm2.weight"],
              tensors[f"{prefix}.norm2.bias"],
              eps=1e-6,
          )
      )
      h_res = F.conv2d(
          h_res,
          tensors[f"{prefix}.conv2.weight"],
          tensors[f"{prefix}.conv2.bias"],
          padding=1,
      )
      if f"{prefix}.nin_shortcut.weight" in tensors:
        t = F.conv2d(
            t,
            tensors[f"{prefix}.nin_shortcut.weight"],
            tensors[f"{prefix}.nin_shortcut.bias"],
        )
      return t + h_res

    def run_attn(t: torch.Tensor, prefix: str) -> torch.Tensor:
      h_att = F.group_norm(
          t,
          32,
          tensors[f"{prefix}.norm.weight"],
          tensors[f"{prefix}.norm.bias"],
          eps=1e-6,
      )
      q_a = F.conv2d(
          h_att, tensors[f"{prefix}.q.weight"], tensors[f"{prefix}.q.bias"]
      )
      k_a = F.conv2d(
          h_att, tensors[f"{prefix}.k.weight"], tensors[f"{prefix}.k.bias"]
      )
      v_a = F.conv2d(
          h_att, tensors[f"{prefix}.v.weight"], tensors[f"{prefix}.v.bias"]
      )
      b, c, hh, ww = q_a.shape
      w_a = torch.bmm(
          q_a.reshape(b, c, hh * ww).permute(0, 2, 1),
          k_a.reshape(b, c, hh * ww),
      ) * (c**-0.5)
      w_a = F.softmax(w_a, dim=2).permute(0, 2, 1)
      out_a = torch.bmm(v_a.reshape(b, c, hh * ww), w_a).reshape(b, c, hh, ww)
      return t + F.conv2d(
          out_a,
          tensors[f"{prefix}.proj_out.weight"],
          tensors[f"{prefix}.proj_out.bias"],
      )

    with torch.no_grad():
      mod_vq_out = vq_mod(vq_codes).numpy()
      z_q = F.normalize(
          F.embedding(vq_codes, tensors[f"{gv}.quantize.embedding.weight"]),
          p=2,
          dim=-1,
      )
      z_q = z_q.view(1, 2, 2, 8).permute(0, 3, 1, 2).contiguous()
      h_vq = F.conv2d(
          z_q,
          tensors[f"{gv}.post_quant_conv.weight"],
          tensors[f"{gv}.post_quant_conv.bias"],
      )
      h_vq = F.conv2d(
          h_vq,
          tensors[f"{gv}.decoder.conv_in.weight"],
          tensors[f"{gv}.decoder.conv_in.bias"],
          padding=1,
      )
      h_vq = run_res(h_vq, f"{gv}.decoder.mid.0")
      h_vq = run_attn(h_vq, f"{gv}.decoder.mid.1")
      h_vq = run_res(h_vq, f"{gv}.decoder.mid.2")
      for level_idx in range(5):
        lp = f"{gv}.decoder.conv_blocks.{level_idx}"
        for r_idx in range(2):
          h_vq = run_res(h_vq, f"{lp}.res.{r_idx}")
          if level_idx == 0:
            h_vq = run_attn(h_vq, f"{lp}.attn.{r_idx}")
        if level_idx < 4:
          h_vq = F.interpolate(h_vq, scale_factor=2.0, mode="nearest")
          h_vq = F.conv2d(
              h_vq,
              tensors[f"{lp}.upsample.conv.weight"],
              tensors[f"{lp}.upsample.conv.bias"],
              padding=1,
          )
      h_vq = F.silu(
          F.group_norm(
              h_vq,
              32,
              tensors[f"{gv}.decoder.norm_out.weight"],
              tensors[f"{gv}.decoder.norm_out.bias"],
              eps=1e-6,
          )
      )
      ref_vq_out = F.conv2d(
          h_vq,
          tensors[f"{gv}.decoder.conv_out.weight"],
          tensors[f"{gv}.decoder.conv_out.bias"],
          padding=1,
      ).numpy()
    np.testing.assert_allclose(mod_vq_out, ref_vq_out, atol=1e-5, rtol=1e-5)

    # 3. Verify `dynamic_wi8_afp32` quantization and LiteRT CompiledModel
    #    execution succeed across all 5 non-LLM submodels.
    with tempfile.TemporaryDirectory() as tmp_dir:
      _write_tiny_janus_pro_config(tmp_dir)
      safetensors.torch.save_file(
          tensors, os.path.join(tmp_dir, "model.safetensors")
      )
      out_dir = os.path.join(tmp_dir, "quant_out")
      q_cfg = exportable_module_config.ExportableModuleConfig(
          model=tmp_dir,
          output_dir=out_dir,
          work_dir=out_dir,
          task=exportable_module_config.ExportTask.ANY_TO_ANY,
          quantization_recipe="dynamic_wi8_afp32",
          vision_encoder_quantization_recipe="dynamic_wi8_afp32",
          use_random_weights=False,
          extra_kwargs={
              "targets": (
                  "vision_encoder,text_embedder,vq_image_embedder,"
                  "gen_head,vq_image_decoder"
              )
          },
      )
      q_artifacts = janus_pro.JanusPro(tmp_dir, export_config=q_cfg).export(
          q_cfg
      )
      sample_inputs = {
          "vision_encoder": pixel_in.numpy(),
          "text_embedder": np.array([[7]], dtype=np.int32),
          "vq_image_embedder": np.array([[5]], dtype=np.int32),
          "gen_head": np.zeros((1, 1, 32), dtype=np.float32),
          "vq_image_decoder": vq_codes.numpy(),
      }
      for name, arr in sample_inputs.items():
        self.assertTrue(q_artifacts[name].endswith("_quantized.tflite"))
        cm = CompiledModel.from_file(q_artifacts[name])
        in_b = cm.create_input_buffers(0)
        out_b = cm.create_output_buffers(0)
        in_b[0].write(arr)
        cm.run_by_index(0, in_b, out_b)
        out_details = cm.get_output_tensor_details("serving_default")
        total = int(np.prod(list(out_details.values())[0]["shape"]))
        out_vals = out_b[0].read(total, np.float32)
        self.assertTrue(np.all(np.isfinite(out_vals)))


if __name__ == "__main__":
  absltest.main()
