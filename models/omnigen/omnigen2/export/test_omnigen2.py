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
"""Unit tests for OmniGen2 (`OmniGen2/OmniGen2`) LiteRT-LM export."""

import json
import os
import tempfile
from typing import Any

from absl.testing import absltest
from google.protobuf import text_format
import litert_torch
from litert_torch.generative.export_hf import export as export_script
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import extension as ext_reg
from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import modeling_flux2
import numpy as np
from safetensors import torch as safetensors_torch
import torch
import transformers

from models.omnigen.omnigen2.export import extension as omnigen2_ext
from models.omnigen.omnigen2.export import modeling
from models.omnigen.omnigen2.export import omnigen2
from models.omnigen.omnigen2.runner import omnigen2_pb2
from ai_edge_litert.compiled_model import CompiledModel
from litert_lm_builder import litertlm_peek
from litert_lm_builder.runtime.proto import text2image_metadata_pb2
from ai_edge_litert import schema_py_generated as schema_fb


def _write_minimal_tokenizer_files(tok_dir: str) -> None:
  """Writes minimal HuggingFace tokenizer files into `tok_dir`."""
  os.makedirs(tok_dir, exist_ok=True)
  with open(os.path.join(tok_dir, "tokenizer_config.json"), "w") as f:
    json.dump({"tokenizer_class": "PreTrainedTokenizerFast"}, f)
  with open(os.path.join(tok_dir, "tokenizer.json"), "w") as f:
    json.dump(
        {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1},
                "merges": [],
            },
        },
        f,
    )


def _write_compact_checkpoint_configs(
    model_dir: str,
    tokenizer_layout: str = "processor",
    write_weights: bool = False,
) -> None:
  """Writes a compact OmniGen2 checkpoint directory with configs and optional weights."""
  os.makedirs(os.path.join(model_dir, "mllm"), exist_ok=True)
  os.makedirs(os.path.join(model_dir, "transformer"), exist_ok=True)
  os.makedirs(os.path.join(model_dir, "vae"), exist_ok=True)
  os.makedirs(os.path.join(model_dir, "scheduler"), exist_ok=True)

  if tokenizer_layout == "processor":
    _write_minimal_tokenizer_files(os.path.join(model_dir, "processor"))
  elif tokenizer_layout == "tokenizer":
    _write_minimal_tokenizer_files(os.path.join(model_dir, "tokenizer"))
  elif tokenizer_layout == "root":
    _write_minimal_tokenizer_files(model_dir)
  else:
    raise ValueError(f"Unsupported tokenizer_layout: {tokenizer_layout}")

  with open(os.path.join(model_dir, "model_index.json"), "w") as f:
    json.dump({"_class_name": "OmniGen2Pipeline"}, f)
  with open(os.path.join(model_dir, "mllm", "config.json"), "w") as f:
    json.dump(
        {
            "model_type": "qwen2_5_vl",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
        },
        f,
    )
  with open(os.path.join(model_dir, "transformer", "config.json"), "w") as f:
    json.dump(
        {
            "_class_name": "OmniGen2Transformer2DModel",
            "patch_size": 2,
            "in_channels": 16,
            "out_channels": None,
            "hidden_size": 48,
            "num_layers": 6,
            "num_refiner_layers": 1,
            "num_attention_heads": 3,
            "num_kv_heads": 1,
            "multiple_of": 16,
            "norm_eps": 1e-5,
            "axes_dim_rope": [4, 6, 6],
            "axes_lens": [64, 64, 64],
            "text_feat_dim": 32,
            "timestep_scale": 1000.0,
        },
        f,
    )
  with open(os.path.join(model_dir, "vae", "config.json"), "w") as f:
    json.dump(
        {
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            "block_out_channels": [8, 8, 8, 8],
            "layers_per_block": 1,
            "latent_channels": 16,
            "norm_num_groups": 4,
            "scaling_factor": 0.3611,
            "shift_factor": 0.1159,
            "use_quant_conv": False,
            "use_post_quant_conv": False,
            "mid_block_add_attention": True,
        },
        f,
    )
  with open(
      os.path.join(model_dir, "scheduler", "scheduler_config.json"), "w"
  ) as f:
    json.dump(
        {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "dynamic_time_shift": True,
            "num_train_timesteps": 1000,
        },
        f,
    )

  if write_weights:
    torch.manual_seed(0)
    te = modeling.OmniGen2TextEncoder(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    te_sd = dict(te.state_dict())
    te_sd["visual.blocks.0.attn.proj.weight"] = torch.zeros(4, 4)
    safetensors_torch.save_file(
        te_sd, os.path.join(model_dir, "mllm", "model.safetensors")
    )
    dit = modeling.OmniGen2Transformer2DModel(
        patch_size=2,
        in_channels=16,
        hidden_size=48,
        num_layers=6,
        num_refiner_layers=1,
        num_attention_heads=3,
        num_kv_heads=1,
        multiple_of=16,
        axes_dim_rope=(4, 6, 6),
        axes_lens=(64, 64, 64),
        text_feat_dim=32,
    )
    safetensors_torch.save_file(
        dit.state_dict(),
        os.path.join(
            model_dir, "transformer", "diffusion_pytorch_model.safetensors"
        ),
    )
    vae = modeling.OmniGen2AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=4,
    )
    safetensors_torch.save_file(
        vae.state_dict(),
        os.path.join(model_dir, "vae", "diffusion_pytorch_model.safetensors"),
    )


class OmniGen2Test(absltest.TestCase):
  """Tests for OmniGen2 architecture, strict weight loading, and TFLite export."""

  def _assert_gpu_compatible_tflite(self, tflite_path: str) -> None:
    """Asserts that a TFLite model has no BROADCAST_TO, GATHER_ND, or >=5D tensors."""
    with open(tflite_path, "rb") as f:
      buf = f.read()
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    forbidden_ops = {
        schema_fb.BuiltinOperator.BROADCAST_TO: "BROADCAST_TO",
        schema_fb.BuiltinOperator.GATHER_ND: "GATHER_ND",
    }
    for sg_idx in range(model.SubgraphsLength()):
      sg = model.Subgraphs(sg_idx)
      for t_idx in range(sg.TensorsLength()):
        t = sg.Tensors(t_idx)
        self.assertLess(
            t.ShapeLength(),
            5,
            f"Tensor {t.Name()} in sg[{sg_idx}] has rank"
            f" {t.ShapeLength()} >= 5",
        )
      for op_idx in range(sg.OperatorsLength()):
        op = sg.Operators(op_idx)
        oc = model.OperatorCodes(op.OpcodeIndex())
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        self.assertNotIn(
            code,
            forbidden_ops,
            f"Forbidden GPU op {forbidden_ops.get(code)} in sg[{sg_idx}]",
        )

  def _run_tflite(self, mod, inputs, out_shapes, path):
    litert_torch.convert(
        mod, tuple(torch.from_numpy(x) for x in inputs)
    ).export(path)
    self._assert_gpu_compatible_tflite(path)
    cm = CompiledModel.from_file(path)
    in_bufs = cm.create_input_buffers(0)
    out_bufs = cm.create_output_buffers(0)
    for i, arr in enumerate(inputs):
      in_bufs[i].write(arr)
    cm.run_by_index(0, in_bufs, out_bufs)
    return [
        out_bufs[i].read(int(np.prod(shape)), np.float32).reshape(shape)
        for i, shape in enumerate(out_shapes)
    ]

  def test_omnigen2_transformer_real_582_keys_and_strict_safetensors_loading(
      self,
  ):
    """Verifies all 582 real OmniGen2/OmniGen2 transformer keys & shapes and strict loading."""
    with torch.device("meta"):
      real_dit = modeling.OmniGen2Transformer2DModel(
          patch_size=2,
          in_channels=16,
          out_channels=None,
          hidden_size=2520,
          num_layers=32,
          num_refiner_layers=2,
          num_attention_heads=21,
          num_kv_heads=7,
          multiple_of=256,
          ffn_dim_multiplier=None,
          norm_eps=1e-5,
          axes_dim_rope=(40, 40, 40),
          axes_lens=(10240, 10240, 10240),
          text_feat_dim=2048,
          timestep_scale=1000.0,
      )
    sd = {k: list(v.shape) for k, v in real_dit.state_dict().items()}
    self.assertLen(sd, 582)
    self.assertEqual(sd["image_index_embedding"], [5, 2520])
    self.assertEqual(sd["x_embedder.weight"], [2520, 64])
    self.assertEqual(sd["x_embedder.bias"], [2520])
    self.assertEqual(sd["ref_image_patch_embedder.weight"], [2520, 64])
    self.assertEqual(sd["ref_image_patch_embedder.bias"], [2520])
    self.assertEqual(sd["time_caption_embed.caption_embedder.0.weight"], [2048])
    self.assertEqual(
        sd["time_caption_embed.caption_embedder.1.weight"], [2520, 2048]
    )
    self.assertEqual(sd["norm_out.linear_1.weight"], [2520, 1024])
    self.assertEqual(sd["norm_out.linear_2.weight"], [64, 2520])
    for i in range(32):
      self.assertEqual(sd[f"layers.{i}.attn.to_q.weight"], [2520, 2520])
      self.assertEqual(sd[f"layers.{i}.attn.to_k.weight"], [840, 2520])
      self.assertEqual(
          sd[f"layers.{i}.feed_forward.linear_1.weight"], [10240, 2520]
      )

    # Test strict sharded safetensors round-trip.
    cpu_dit = modeling.OmniGen2Transformer2DModel(
        patch_size=2,
        in_channels=16,
        hidden_size=48,
        num_layers=6,
        num_refiner_layers=2,
        num_attention_heads=3,
        num_kv_heads=1,
        multiple_of=16,
        axes_dim_rope=(4, 6, 6),
        text_feat_dim=32,
    ).eval()
    with tempfile.TemporaryDirectory() as tmp_dir:
      all_tensors = cpu_dit.state_dict()
      keys = list(all_tensors.keys())
      mid = len(keys) // 2
      shard1 = {k: all_tensors[k] for k in keys[:mid]}
      shard2 = {k: all_tensors[k] for k in keys[mid:]}
      safetensors_torch.save_file(
          shard1, os.path.join(tmp_dir, "shard-00001-of-00002.safetensors")
      )
      safetensors_torch.save_file(
          shard2, os.path.join(tmp_dir, "shard-00002-of-00002.safetensors")
      )
      weight_map = {
          k: (
              "shard-00001-of-00002.safetensors"
              if k in shard1
              else "shard-00002-of-00002.safetensors"
          )
          for k in keys
      }
      with open(
          os.path.join(
              tmp_dir, "diffusion_pytorch_model.safetensors.index.json"
          ),
          "w",
      ) as f:
        json.dump({"weight_map": weight_map}, f)

      loaded = modeling.load_safetensors_state_dict(tmp_dir)
      self.assertEqual(set(loaded.keys()), set(all_tensors.keys()))
      fresh_dit = modeling.OmniGen2Transformer2DModel(
          patch_size=2,
          in_channels=16,
          hidden_size=48,
          num_layers=6,
          num_refiner_layers=2,
          num_attention_heads=3,
          num_kv_heads=1,
          multiple_of=16,
          axes_dim_rope=(4, 6, 6),
          text_feat_dim=32,
      ).eval()
      fresh_dit.load_state_dict(loaded, strict=True)

  def test_verify_checkpoint_keys_accepts_valid_and_rejects_unconsumed_keys(
      self,
  ):
    """Verifies `verify_checkpoint_keys` catches missing and unrecognized keys."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      _write_compact_checkpoint_configs(tmp_dir, write_weights=True)
      counts = modeling.verify_checkpoint_keys(tmp_dir)
      self.assertGreater(counts["mllm"], 0)
      self.assertGreater(counts["transformer"], 0)
      self.assertGreater(counts["vae"], 0)

      # Inject an unexpected key into transformer safetensors -> ValueError.
      dit_path = os.path.join(
          tmp_dir, "transformer", "diffusion_pytorch_model.safetensors"
      )
      dit_sd = safetensors_torch.load_file(dit_path)
      dit_sd["unexpected_extra.weight"] = torch.zeros(1)
      safetensors_torch.save_file(dit_sd, dit_path)
      with self.assertRaisesRegex(ValueError, "unexpected"):
        modeling.verify_checkpoint_keys(tmp_dir)

  def test_omnigen2_mllm_real_824_keys_normalization_and_text_encoder_parity(
      self,
  ):
    """Verifies real 824-key Qwen2.5-VL-3B mllm normalization and TFLite parity."""
    with torch.device("meta"):
      real_te = modeling.OmniGen2TextEncoder(
          vocab_size=151936,
          hidden_size=2048,
          intermediate_size=11008,
          num_hidden_layers=36,
          num_attention_heads=16,
          num_key_value_heads=2,
          rms_norm_eps=1e-6,
          rope_theta=1000000.0,
      )
    expected_te_sd = real_te.state_dict()
    self.assertLen(expected_te_sd, 434)

    raw_mllm_sd = dict(expected_te_sd)
    for i in range(390):
      raw_mllm_sd[f"visual.blocks.{i}.attn.proj.weight"] = torch.empty(
          (1, 1), device="meta"
      )
    self.assertLen(raw_mllm_sd, 824)
    normalized_sd = modeling.OmniGen2TextEncoder.normalize_mllm_state_dict(
        raw_mllm_sd
    )
    self.assertLen(normalized_sd, 434)
    self.assertEqual(set(normalized_sd.keys()), set(expected_te_sd.keys()))

    with self.assertRaises(ValueError):
      modeling.OmniGen2TextEncoder.normalize_mllm_state_dict(
          {"unexpected_module.weight": torch.zeros(1)}
      )

    torch.manual_seed(42)
    qwen2_cfg_cls: Any = getattr(transformers, "Qwen2Config")
    qwen2_model_cls: Any = getattr(transformers, "Qwen2Model")
    hf_qwen2_cfg = qwen2_cfg_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=64,
    )
    hf_qwen2 = qwen2_model_cls(hf_qwen2_cfg).eval()
    te = modeling.OmniGen2TextEncoder(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    ).eval()
    norm_hf_sd = modeling.OmniGen2TextEncoder.normalize_mllm_state_dict(
        hf_qwen2.state_dict()
    )
    te.load_state_dict(norm_hf_sd, strict=True)

    # Also verify exact FP32 parity against `transformers.Qwen2_5_VLTextModel`
    # loaded with the same weights.
    qwen25_vl_cfg_cls: Any = getattr(transformers, "Qwen2_5_VLTextConfig")
    qwen25_vl_model_cls: Any = getattr(transformers, "Qwen2_5_VLTextModel")
    hf_qwen25_vl_cfg = qwen25_vl_cfg_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        max_position_embeddings=64,
        use_sliding_window=False,
    )
    hf_qwen25_vl = qwen25_vl_model_cls(hf_qwen25_vl_cfg).eval()
    hf_qwen25_vl.load_state_dict(hf_qwen2.state_dict(), strict=True)

    seq_len = 8
    active_len = 5
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]], dtype=torch.int32)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.int32)
    with torch.no_grad():
      hf_out = (
          hf_qwen2(
              input_ids=input_ids.to(torch.int64),
              attention_mask=attention_mask.to(torch.int64),
          ).last_hidden_state
          * attention_mask.to(torch.float32).unsqueeze(-1)
      ).numpy()
      hf_qwen25_vl_out = (
          hf_qwen25_vl(
              input_ids=input_ids.to(torch.int64),
              attention_mask=attention_mask.to(torch.int64),
          ).last_hidden_state
          * attention_mask.to(torch.float32).unsqueeze(-1)
      ).numpy()
      pt_out = te(input_ids, attention_mask).numpy()
      self.assertLess(np.max(np.abs(hf_out - pt_out)), 1e-5)
      self.assertLess(np.max(np.abs(hf_qwen25_vl_out - pt_out)), 1e-5)
      np.testing.assert_allclose(pt_out[:, active_len:, :], 0.0, atol=1e-7)

    with tempfile.TemporaryDirectory() as tmp_dir:
      (tflite_out,) = self._run_tflite(
          te,
          [input_ids.numpy(), attention_mask.numpy()],
          [(1, seq_len, 32)],
          os.path.join(tmp_dir, "text_encoder.tflite"),
      )
    self.assertLess(np.max(np.abs(hf_out - tflite_out)), 1e-4)

  def test_omnigen2_transformer_parity_sharded_and_tflite(self):
    """Tests monolithic, sharded, and TFLite OmniGen2 DiT parity."""
    torch.manual_seed(123)
    our_dit = modeling.OmniGen2Transformer2DModel(
        patch_size=2,
        in_channels=16,
        hidden_size=48,
        num_layers=6,
        num_refiner_layers=2,
        num_attention_heads=3,
        num_kv_heads=1,
        multiple_of=16,
        norm_eps=1e-5,
        axes_dim_rope=(4, 6, 6),
        axes_lens=(64, 64, 64),
        text_feat_dim=32,
        timestep_scale=1000.0,
    ).eval()
    with torch.no_grad():
      our_dit.image_index_embedding.copy_(
          torch.randn_like(our_dit.image_index_embedding) + 0.5
      )

    seq_len = 8
    active_cap_len = 5
    latents = torch.randn(1, 16, 4, 4, dtype=torch.float32)
    timestep = torch.tensor([0.35], dtype=torch.float32)
    text_hidden_states = torch.randn(1, seq_len, 32, dtype=torch.float32)
    text_hidden_states[:, active_cap_len:, :] = 0.0
    text_attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.int64
    )
    ref_image = torch.randn(16, 4, 4, dtype=torch.float32)

    with torch.no_grad():
      our_t2i = our_dit(
          latents,
          timestep,
          text_hidden_states,
          text_attention_mask,
          ref_image_hidden_states=None,
      ).numpy()
      our_edit = our_dit(
          latents,
          timestep,
          text_hidden_states,
          text_attention_mask,
          ref_image_hidden_states=[[ref_image]],
      ).numpy()

    dummy_te = modeling.OmniGen2TextEncoder(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    ).eval()
    dummy_vae = modeling.OmniGen2AutoencoderKL(
        in_channels=3,
        out_channels=3,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=4,
    ).eval()
    sharded = omnigen2.OmniGen2Model(
        dummy_te, our_dit, dummy_vae, seq_len=seq_len, img_size=32
    ).eval()

    cap_cos, cap_sin, img_cos_t2i, img_sin_t2i = omnigen2.build_omnigen2_rope(
        seq_len=seq_len,
        grid_h=2,
        grid_w=2,
        axes_dim_rope=(4, 6, 6),
        active_cap_len=active_cap_len,
    )
    cap_mask, joint_mask_t2i = omnigen2.build_omnigen2_masks(
        seq_len=seq_len,
        img_tokens=4,
        active_cap_len=active_cap_len,
    )
    joint_cos_t2i = torch.cat([cap_cos, img_cos_t2i], dim=1)
    joint_sin_t2i = torch.cat([cap_sin, img_sin_t2i], dim=1)
    packed_x = omnigen2.pack_omnigen2_latents(latents)

    with torch.no_grad():
      cap_emb = sharded.dit_cap_embedder(text_hidden_states)
      cap_ref = sharded.dit_context_refiner(cap_emb, cap_cos, cap_sin, cap_mask)
      x_emb, temb = sharded.dit_x_embedder(packed_x, timestep)
      x_ref = sharded.dit_noise_refiner(x_emb, img_cos_t2i, img_sin_t2i, temb)
      joint = torch.cat([cap_ref, x_ref], dim=1)
      for main_shard in sharded.dit_main_shards:
        joint = main_shard(
            joint, joint_cos_t2i, joint_sin_t2i, joint_mask_t2i, temb
        )
      out_patches = sharded.dit_final(joint, temb)
    sharded_t2i = omnigen2.unpack_omnigen2_latents(
        out_patches, latent_h=4, latent_w=4
    ).numpy()
    self.assertLess(np.max(np.abs(our_t2i - sharded_t2i)), 1e-5)

    # Check sharded reference-image path against `our_edit`.
    packed_ref = omnigen2.pack_omnigen2_latents(ref_image.unsqueeze(0))
    ref_ids = omnigen2.build_omnigen2_img_ids(
        grid_h=2, grid_w=2, pe_shift=active_cap_len
    )
    ref_cos_2d, ref_sin_2d = modeling.build_omnigen2_3d_rope_cos_sin(
        ref_ids, axes_dim_rope=(4, 6, 6)
    )
    ref_cos = ref_cos_2d.unsqueeze(0).unsqueeze(2)
    ref_sin = ref_sin_2d.unsqueeze(0).unsqueeze(2)

    img_ids_edit = omnigen2.build_omnigen2_img_ids(
        grid_h=2, grid_w=2, pe_shift=active_cap_len + 2
    )
    img_cos_edit_2d, img_sin_edit_2d = modeling.build_omnigen2_3d_rope_cos_sin(
        img_ids_edit, axes_dim_rope=(4, 6, 6)
    )
    img_cos_edit = img_cos_edit_2d.unsqueeze(0).unsqueeze(2)
    img_sin_edit = img_sin_edit_2d.unsqueeze(0).unsqueeze(2)

    _, joint_mask_edit = omnigen2.build_omnigen2_masks(
        seq_len=seq_len,
        img_tokens=8,
        active_cap_len=active_cap_len,
    )
    joint_cos_edit = torch.cat([cap_cos, ref_cos, img_cos_edit], dim=1)
    joint_sin_edit = torch.cat([cap_sin, ref_sin, img_sin_edit], dim=1)
    with torch.no_grad():
      ref_emb = sharded.dit_ref_image_embedder(packed_ref)
      ref_refined = sharded.dit_ref_image_refiner(
          ref_emb,
          ref_cos,
          ref_sin,
          temb,
          torch.tensor([0], dtype=torch.int64),
      )
      x_ref_edit = sharded.dit_noise_refiner(
          x_emb, img_cos_edit, img_sin_edit, temb
      )
      joint_edit = torch.cat([cap_ref, ref_refined, x_ref_edit], dim=1)
      for main_shard in sharded.dit_main_shards:
        joint_edit = main_shard(
            joint_edit,
            joint_cos_edit,
            joint_sin_edit,
            joint_mask_edit,
            temb,
        )
      out_patches_edit = sharded.dit_final(joint_edit, temb)
    sharded_edit = omnigen2.unpack_omnigen2_latents(
        out_patches_edit, latent_h=4, latent_w=4
    ).numpy()
    self.assertLess(np.max(np.abs(our_edit - sharded_edit)), 1e-5)

    # Export all TFLite DiT shards and verify CompiledModel numerical parity.
    with tempfile.TemporaryDirectory() as tmp_dir:
      (cap_emb_np,) = self._run_tflite(
          sharded.dit_cap_embedder,
          [text_hidden_states.numpy()],
          [(1, seq_len, 48)],
          os.path.join(tmp_dir, "og2_embc.tflite"),
      )
      (cap_ref_np,) = self._run_tflite(
          sharded.dit_context_refiner,
          [
              cap_emb_np,
              cap_cos.numpy(),
              cap_sin.numpy(),
              cap_mask.numpy(),
          ],
          [(1, seq_len, 48)],
          os.path.join(tmp_dir, "og2_refc.tflite"),
      )
      x_emb_np, temb_np = self._run_tflite(
          sharded.dit_x_embedder,
          [packed_x.numpy(), timestep.numpy()],
          [(1, 4, 48), (1, 48)],
          os.path.join(tmp_dir, "og2_embx.tflite"),
      )
      (x_ref_np,) = self._run_tflite(
          sharded.dit_noise_refiner,
          [
              x_emb_np,
              img_cos_t2i.numpy(),
              img_sin_t2i.numpy(),
              temb_np,
          ],
          [(1, 4, 48)],
          os.path.join(tmp_dir, "og2_refx.tflite"),
      )
      joint_np = np.concatenate([cap_ref_np, x_ref_np], axis=1)
      for idx, main_shard in enumerate(sharded.dit_main_shards):
        (joint_np,) = self._run_tflite(
            main_shard,
            [
                joint_np,
                joint_cos_t2i.numpy(),
                joint_sin_t2i.numpy(),
                joint_mask_t2i.numpy(),
                temb_np,
            ],
            [(1, seq_len + 4, 48)],
            os.path.join(tmp_dir, f"og2_main{idx}.tflite"),
        )
      (vel_patches_np,) = self._run_tflite(
          sharded.dit_final,
          [joint_np, temb_np],
          [(1, 4, 64)],
          os.path.join(tmp_dir, "og2_final.tflite"),
      )
    tflite_t2i = omnigen2.unpack_omnigen2_latents(
        torch.from_numpy(vel_patches_np), latent_h=4, latent_w=4
    ).numpy()
    self.assertLess(np.max(np.abs(our_t2i - tflite_t2i)), 1e-4)

  def test_omnigen2_vae_real_244_keys_and_tflite_decode_parity(self):
    """Verifies all 244 real VAE keys and TFLite decode parity."""
    with torch.device("meta"):
      real_vae = modeling.OmniGen2AutoencoderKL(
          in_channels=3,
          out_channels=3,
          down_block_types=("DownEncoderBlock2D",) * 4,
          up_block_types=("UpDecoderBlock2D",) * 4,
          block_out_channels=(128, 256, 512, 512),
          layers_per_block=2,
          latent_channels=16,
          norm_num_groups=32,
          use_quant_conv=False,
          use_post_quant_conv=False,
          mid_block_add_attention=True,
      )
    our_vae_sd = {k: list(v.shape) for k, v in real_vae.state_dict().items()}
    self.assertLen(our_vae_sd, 244)
    self.assertEqual(our_vae_sd["encoder.conv_in.weight"], [128, 3, 3, 3])
    self.assertEqual(our_vae_sd["encoder.conv_out.weight"], [32, 512, 3, 3])
    self.assertEqual(our_vae_sd["decoder.conv_in.weight"], [512, 16, 3, 3])
    self.assertEqual(our_vae_sd["decoder.conv_out.weight"], [3, 128, 3, 3])
    self.assertNotIn("quant_conv.weight", our_vae_sd)
    self.assertNotIn("post_quant_conv.weight", our_vae_sd)

    torch.manual_seed(77)
    our_vae = modeling.OmniGen2AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=4,
        use_quant_conv=False,
        use_post_quant_conv=False,
        mid_block_add_attention=True,
    ).eval()

    ref_flux2_vae = modeling_flux2.AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=4,
        use_quant_conv=False,
        use_post_quant_conv=False,
        mid_block_add_attention=True,
    ).eval()
    ref_flux2_vae.load_state_dict(our_vae.state_dict(), strict=False)

    latents = torch.randn(1, 16, 4, 4, dtype=torch.float32)
    with torch.no_grad():
      our_rgb = our_vae.decode(latents, return_dict=False)[0].numpy()
      ref_rgb = ref_flux2_vae.decode(latents, return_dict=False)[0].numpy()
    self.assertEqual(our_rgb.shape, (1, 3, 32, 32))
    self.assertLess(np.max(np.abs(our_rgb - ref_rgb)), 1e-6)

    with tempfile.TemporaryDirectory() as tmp_dir:
      vae_shard = omnigen2.OmniGen2VaeDecoderShard(our_vae).eval()
      (tflite_rgb,) = self._run_tflite(
          vae_shard,
          [latents.numpy()],
          [our_rgb.shape],
          os.path.join(tmp_dir, "og2_vae.tflite"),
      )
    self.assertLess(np.max(np.abs(our_rgb - tflite_rgb)), 1e-4)

  def test_extension_registration_match_checkpoint_and_dry_run(self):
    """Tests FlowMatch schedule, ModelExtension discovery, and `--dry_run`."""
    timesteps = omnigen2.compute_omnigen2_timesteps(
        steps=50, latent_tokens=128 * 128, dynamic_time_shift=True
    )
    self.assertLen(timesteps, 51)
    self.assertAlmostEqual(float(timesteps[0]), 0.0, places=7)
    self.assertAlmostEqual(float(timesteps[-1]), 1.0, places=7)
    raw_t = np.linspace(0.0, 1.0, 51, dtype=np.float32)[:-1]
    expected_t = raw_t / (3.2 - 3.2 * raw_t + raw_t)
    np.testing.assert_allclose(
        np.asarray(timesteps[:-1], dtype=np.float32),
        expected_t,
        rtol=1e-6,
        atol=1e-6,
    )

    exp_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy", task="text_to_image"
    )
    for arch_name in (
        "OmniGen2Pipeline",
        "omnigen2",
        "omnigen-2",
        "omnigen_2",
        "OmniGen2/OmniGen2",
    ):
      ext = ext_reg.resolve_model_extension(arch_name, exp_cfg)
      self.assertIsNotNone(ext)
      assert ext is not None
      self.assertIs(ext.get_image_gen_model_cls(arch_name), omnigen2.OmniGen2)
      self.assertIsInstance(ext, omnigen2_ext.OmniGen2Extension)

    for task_arg in (None, "auto", "text_to_image"):
      self.assertTrue(
          omnigen2_ext.OmniGen2Extension.match_checkpoint(
              "/tmp/ckpt",
              task=task_arg,
              config_dict={"_class_name": "OmniGen2Pipeline"},
          )
      )
    self.assertFalse(
        omnigen2_ext.OmniGen2Extension.match_checkpoint(
            "/tmp/ckpt",
            task="text_generation",
            config_dict={"_class_name": "OmniGen2Pipeline"},
        )
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      _write_compact_checkpoint_configs(tmp_dir, tokenizer_layout="processor")
      export_script.export(model=tmp_dir, dry_run=True)

  def test_tokenizer_directory_resolution_for_processor_tokenizer_and_root(
      self,
  ):
    """Verifies `processor/`, `tokenizer/`, and root tokenizer layouts without export_lib edits."""
    for layout in ("processor", "tokenizer", "root"):
      with self.subTest(layout=layout):
        with tempfile.TemporaryDirectory() as tmp_dir:
          _write_compact_checkpoint_configs(tmp_dir, tokenizer_layout=layout)
          model = omnigen2.OmniGen2(tmp_dir)
          if layout == "processor":
            self.assertEqual(
                model.model_dir, os.path.join(tmp_dir, "processor")
            )
          else:
            self.assertEqual(model.model_dir, tmp_dir)

          # Verify `export_lib.py`'s exact tokenizer lookup logic succeeds.
          tok_dir = os.path.join(model.model_dir, "tokenizer")
          if not os.path.exists(tok_dir):
            tok_dir = model.model_dir
          tok = transformers.AutoTokenizer.from_pretrained(tok_dir)
          self.assertIsNotNone(tok)

          # Verify configs were still loaded from the checkpoint root.
          meta = model.get_image_gen_metadata(
              exportable_module_config.ExportableModuleConfig(
                  model=tmp_dir, task="text_to_image"
              )
          )
          unpacked = omnigen2_pb2.OmniGen2()
          self.assertTrue(
              meta.image_gen_model_type.custom_model.Unpack(unpacked)
          )
          self.assertEqual(unpacked.hidden_size, 48)
          self.assertEqual(unpacked.prompt_dim, 32)
          self.assertEqual(list(unpacked.axes_dim_rope), [4, 6, 6])

  def test_omnigen2_export_and_litertlm_packaging(self):
    """Tests that export.export() builds a complete sharded OmniGen2 .litertlm file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      model_dir = os.path.join(tmp_dir, "checkpoint")
      out_dir = os.path.join(tmp_dir, "output")
      _write_compact_checkpoint_configs(model_dir, tokenizer_layout="processor")

      export_script.export(
          model=model_dir,
          output_dir=out_dir,
          task="text_to_image",
          quantization_recipe="none",
          use_random_weights=True,
          t2i_output_image_size=32,
          seq_len=32,
          steps=2,
          text_guidance_scale=3.0,
          bundle_litert_lm=True,
      )

      litertlm_path = os.path.join(out_dir, "model.litertlm")
      self.assertTrue(os.path.exists(litertlm_path))

      dump_dir = os.path.join(tmp_dir, "unpacked")
      os.makedirs(dump_dir, exist_ok=True)
      with open(os.path.join(tmp_dir, "peek.txt"), "w") as peek_out:
        litertlm_peek.peek_litertlm_file(litertlm_path, dump_dir, peek_out)
      with open(os.path.join(tmp_dir, "peek.txt"), "r") as peek_in:
        peek_text = peek_in.read()

      self.assertIn("Data Type:    Text2ImageMetadataProto", peek_text)
      self.assertIn("custom_model", peek_text)
      self.assertIn("type.googleapis.com/litert.lm.proto.OmniGen2", peek_text)

      # Verify dumped metadata unpacks cleanly into `omnigen2_pb2.OmniGen2`.
      meta_files = [
          f for f in os.listdir(dump_dir) if "Text2ImageMetadataProto" in f
      ]
      self.assertNotEmpty(meta_files)
      dumped_meta = text2image_metadata_pb2.Text2ImageMetadata()
      with open(os.path.join(dump_dir, meta_files[0]), "r") as f:
        text_format.Parse(f.read(), dumped_meta)
      unpacked = omnigen2_pb2.OmniGen2()
      self.assertTrue(
          dumped_meta.image_gen_model_type.custom_model.Unpack(unpacked)
      )
      self.assertEqual(unpacked.seq_len, 32)
      self.assertEqual(unpacked.img_size, 32)
      self.assertEqual(unpacked.default_steps, 2)
      self.assertAlmostEqual(unpacked.text_guidance_scale, 3.0, places=5)
      self.assertTrue(unpacked.dynamic_time_shift)

      # Verify all 7 standard TFLiteModel sections are present.
      for expected_type in (
          "tf_lite_text_encoder",
          "tf_lite_diffusion_transformer_single_block_0",
          "tf_lite_diffusion_transformer_single_block_1",
          "tf_lite_diffusion_transformer_single_block_2",
          "tf_lite_diffusion_transformer_single_block_3",
          "tf_lite_diffusion_transformer_final",
          "tf_lite_image_decoder",
      ):
        self.assertIn(
            f"Key: model_type, Value (String): {expected_type}", peek_text
        )

      # Verify all 6 OmniGen2-specific GenericBinaryData shards are present.
      for expected_generic in (
          "dit_cap_embedder",
          "dit_context_refiner",
          "dit_x_embedder",
          "dit_noise_refiner",
          "dit_single_block_4",
          "dit_single_block_5",
      ):
        self.assertIn(
            f"Key: name, Value (String): {expected_generic}", peek_text
        )


if __name__ == "__main__":
  absltest.main()
