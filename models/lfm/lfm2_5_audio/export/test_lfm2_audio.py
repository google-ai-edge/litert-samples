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
"""Unit tests for LFM2.5-Audio-1.5B (`LiquidAI/LFM2.5-Audio-1.5B`) exporter."""

import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest
from google.protobuf import text_format
from litert_torch.generative.export_hf import export as export_script
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import exportables
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches
import numpy as np
import safetensors.torch
import torch
import transformers

from models.lfm.lfm2_5_audio.export import export_main
from models.lfm.lfm2_5_audio.export import extension as lfm2_audio_ext
from models.lfm.lfm2_5_audio.export import lfm2_audio
from models.lfm.lfm2_5_audio.export import modules
from models.lfm.lfm2_5_audio.runner import lfm2_audio_pb2
from ai_edge_litert.compiled_model import CompiledModel
from litert_lm_builder import litertlm_peek
from litert_lm_builder.runtime.proto import any2any_metadata_pb2


def _reference_vocos_istft(
    log_mag: torch.Tensor,
    phase: torch.Tensor,
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
  """Reference Vocos `padding='same'` ISTFT using `torch.fft.irfft`."""
  mag = torch.exp(log_mag)
  spec = mag * (torch.cos(phase) + 1j * torch.sin(phase))
  ifft = torch.fft.irfft(spec, n=n_fft, dim=1, norm="backward")
  ifft = ifft * window[None, :, None]
  t_frames = ifft.shape[-1]
  output_size = (t_frames - 1) * hop_length + n_fft
  pad = (n_fft - hop_length) // 2
  y = torch.zeros((spec.shape[0], output_size), dtype=log_mag.dtype)
  env = torch.zeros((output_size,), dtype=log_mag.dtype)
  win_sq = window**2
  for t in range(t_frames):
    start = t * hop_length
    y[:, start : start + n_fft] += ifft[:, :, t]
    env[start : start + n_fft] += win_sq
  y = y[:, pad : output_size - pad]
  env = env[pad : output_size - pad]
  return y / torch.clamp(env[None, :], min=1e-11)


def _build_expected_checkpoint_schema(
    *,
    conformer_layers: int,
    conformer_d_model: int,
    conformer_conv_channels: int,
    conformer_heads: int,
    adapter_hidden: int,
    llm_vocab_size: int,
    llm_hidden_size: int,
    llm_ff_dim: int,
    llm_kv_dim: int,
    llm_layer_types: list[str],
    codebooks: int,
    audio_vocab_size: int,
    df_layers: int,
    df_dim: int,
    df_ff_dim: int,
    df_qkv_dim: int,
    df_head_dim: int,
    detok_vocab_size: int,
    detok_hidden_size: int,
    detok_ff_dim: int,
    detok_kv_dim: int,
    detok_head_dim: int,
    detok_n_fft: int,
    detok_layer_types: list[str],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
  """Builds the independent HF safetensors key->shape schema."""
  c_ff = conformer_d_model * (4 if conformer_d_model == 512 else 2)
  c_head_dim = conformer_d_model // conformer_heads
  main: dict[str, tuple[int, ...]] = {
      "conformer.pre_encode.conv.0.weight": (conformer_conv_channels, 1, 3, 3),
      "conformer.pre_encode.conv.0.bias": (conformer_conv_channels,),
      "conformer.pre_encode.conv.2.weight": (conformer_conv_channels, 1, 3, 3),
      "conformer.pre_encode.conv.2.bias": (conformer_conv_channels,),
      "conformer.pre_encode.conv.3.weight": (
          conformer_conv_channels,
          conformer_conv_channels,
          1,
          1,
      ),
      "conformer.pre_encode.conv.3.bias": (conformer_conv_channels,),
      "conformer.pre_encode.conv.5.weight": (conformer_conv_channels, 1, 3, 3),
      "conformer.pre_encode.conv.5.bias": (conformer_conv_channels,),
      "conformer.pre_encode.conv.6.weight": (
          conformer_conv_channels,
          conformer_conv_channels,
          1,
          1,
      ),
      "conformer.pre_encode.conv.6.bias": (conformer_conv_channels,),
      "conformer.pre_encode.out.weight": (
          conformer_d_model,
          conformer_conv_channels * 16,
      ),
      "conformer.pre_encode.out.bias": (conformer_d_model,),
      "audio_adapter.model.0.weight": (conformer_d_model,),
      "audio_adapter.model.0.bias": (conformer_d_model,),
      "audio_adapter.model.1.weight": (adapter_hidden, conformer_d_model),
      "audio_adapter.model.1.bias": (adapter_hidden,),
      "audio_adapter.model.3.weight": (llm_hidden_size, adapter_hidden),
      "audio_adapter.model.3.bias": (llm_hidden_size,),
      "lfm.embed_tokens.weight": (llm_vocab_size, llm_hidden_size),
      "lfm.embedding_norm.weight": (llm_hidden_size,),
      "depth_linear.weight": (codebooks * df_dim, llm_hidden_size),
      "depth_linear.bias": (codebooks * df_dim,),
      "audio_embedding.embedding.weight": (
          codebooks * audio_vocab_size,
          llm_hidden_size,
      ),
      "audio_embedding.embedding_norm.weight": (llm_hidden_size,),
      "audio_embedding.to_logits.weight": (
          codebooks * audio_vocab_size,
          llm_hidden_size,
      ),
      "codebook_offsets": (codebooks,),
      "audio_loss_weights": (codebooks,),
  }
  for i in range(conformer_layers):
    p = f"conformer.layers.{i}"
    main.update({
        f"{p}.norm_feed_forward1.weight": (conformer_d_model,),
        f"{p}.norm_feed_forward1.bias": (conformer_d_model,),
        f"{p}.feed_forward1.linear1.weight": (c_ff, conformer_d_model),
        f"{p}.feed_forward1.linear1.bias": (c_ff,),
        f"{p}.feed_forward1.linear2.weight": (conformer_d_model, c_ff),
        f"{p}.feed_forward1.linear2.bias": (conformer_d_model,),
        f"{p}.norm_self_att.weight": (conformer_d_model,),
        f"{p}.norm_self_att.bias": (conformer_d_model,),
        f"{p}.self_attn.linear_q.weight": (
            conformer_d_model,
            conformer_d_model,
        ),
        f"{p}.self_attn.linear_q.bias": (conformer_d_model,),
        f"{p}.self_attn.linear_k.weight": (
            conformer_d_model,
            conformer_d_model,
        ),
        f"{p}.self_attn.linear_k.bias": (conformer_d_model,),
        f"{p}.self_attn.linear_v.weight": (
            conformer_d_model,
            conformer_d_model,
        ),
        f"{p}.self_attn.linear_v.bias": (conformer_d_model,),
        f"{p}.self_attn.linear_out.weight": (
            conformer_d_model,
            conformer_d_model,
        ),
        f"{p}.self_attn.linear_out.bias": (conformer_d_model,),
        f"{p}.self_attn.linear_pos.weight": (
            conformer_d_model,
            conformer_d_model,
        ),
        f"{p}.self_attn.pos_bias_u": (conformer_heads, c_head_dim),
        f"{p}.self_attn.pos_bias_v": (conformer_heads, c_head_dim),
        f"{p}.norm_conv.weight": (conformer_d_model,),
        f"{p}.norm_conv.bias": (conformer_d_model,),
        f"{p}.conv.pointwise_conv1.weight": (
            2 * conformer_d_model,
            conformer_d_model,
            1,
        ),
        f"{p}.conv.pointwise_conv1.bias": (2 * conformer_d_model,),
        f"{p}.conv.depthwise_conv.weight": (conformer_d_model, 1, 9),
        f"{p}.conv.depthwise_conv.bias": (conformer_d_model,),
        f"{p}.conv.batch_norm.weight": (conformer_d_model,),
        f"{p}.conv.batch_norm.bias": (conformer_d_model,),
        f"{p}.conv.batch_norm.running_mean": (conformer_d_model,),
        f"{p}.conv.batch_norm.running_var": (conformer_d_model,),
        f"{p}.conv.batch_norm.num_batches_tracked": (),
        f"{p}.conv.pointwise_conv2.weight": (
            conformer_d_model,
            conformer_d_model,
            1,
        ),
        f"{p}.conv.pointwise_conv2.bias": (conformer_d_model,),
        f"{p}.norm_feed_forward2.weight": (conformer_d_model,),
        f"{p}.norm_feed_forward2.bias": (conformer_d_model,),
        f"{p}.feed_forward2.linear1.weight": (c_ff, conformer_d_model),
        f"{p}.feed_forward2.linear1.bias": (c_ff,),
        f"{p}.feed_forward2.linear2.weight": (conformer_d_model, c_ff),
        f"{p}.feed_forward2.linear2.bias": (conformer_d_model,),
        f"{p}.norm_out.weight": (conformer_d_model,),
        f"{p}.norm_out.bias": (conformer_d_model,),
    })
  llm_head_dim = 64 if llm_hidden_size == 2048 else 8
  for i, lt in enumerate(llm_layer_types):
    p = f"lfm.layers.{i}"
    main.update({
        f"{p}.operator_norm.weight": (llm_hidden_size,),
        f"{p}.ffn_norm.weight": (llm_hidden_size,),
        f"{p}.feed_forward.w1.weight": (llm_ff_dim, llm_hidden_size),
        f"{p}.feed_forward.w2.weight": (llm_hidden_size, llm_ff_dim),
        f"{p}.feed_forward.w3.weight": (llm_ff_dim, llm_hidden_size),
    })
    if lt == "conv":
      main.update({
          f"{p}.conv.conv.weight": (llm_hidden_size, 1, 3),
          f"{p}.conv.in_proj.weight": (3 * llm_hidden_size, llm_hidden_size),
          f"{p}.conv.out_proj.weight": (llm_hidden_size, llm_hidden_size),
      })
    else:
      main.update({
          f"{p}.self_attn.q_proj.weight": (llm_hidden_size, llm_hidden_size),
          f"{p}.self_attn.k_proj.weight": (llm_kv_dim, llm_hidden_size),
          f"{p}.self_attn.v_proj.weight": (llm_kv_dim, llm_hidden_size),
          f"{p}.self_attn.out_proj.weight": (llm_hidden_size, llm_hidden_size),
          f"{p}.self_attn.q_layernorm.weight": (llm_head_dim,),
          f"{p}.self_attn.k_layernorm.weight": (llm_head_dim,),
      })
  for i in range(df_layers):
    p = f"depthformer.layers.{i}"
    main.update({
        f"{p}.operator_norm.weight": (df_dim,),
        f"{p}.ffn_norm.weight": (df_dim,),
        f"{p}.operator.qkv_proj.weight": (df_qkv_dim, df_dim),
        f"{p}.operator.out_proj.weight": (df_dim, df_dim),
        f"{p}.operator.bounded_attention.q_layernorm.weight": (df_head_dim,),
        f"{p}.operator.bounded_attention.k_layernorm.weight": (df_head_dim,),
        f"{p}.feed_forward.w1.weight": (df_ff_dim, df_dim),
        f"{p}.feed_forward.w2.weight": (df_dim, df_ff_dim),
        f"{p}.feed_forward.w3.weight": (df_ff_dim, df_dim),
    })
  for k in range(codebooks):
    main.update({
        f"depth_embeddings.{k}.embedding.weight": (audio_vocab_size, df_dim),
        f"depth_embeddings.{k}.embedding_norm.weight": (df_dim,),
        f"depth_embeddings.{k}.to_logits.weight": (audio_vocab_size, df_dim),
    })

  detok: dict[str, tuple[int, ...]] = {
      "emb.emb.weight": (codebooks * detok_vocab_size, detok_hidden_size),
      "lfm.embed_tokens.weight": (llm_vocab_size, detok_hidden_size),
      "lfm.embedding_norm.weight": (detok_hidden_size,),
      "lin.weight": (detok_n_fft + 2, detok_hidden_size),
      "lin.bias": (detok_n_fft + 2,),
      "istft.window": (detok_n_fft,),
  }
  for i, lt in enumerate(detok_layer_types):
    p = f"lfm.layers.{i}"
    detok.update({
        f"{p}.operator_norm.weight": (detok_hidden_size,),
        f"{p}.ffn_norm.weight": (detok_hidden_size,),
        f"{p}.feed_forward.w1.weight": (detok_ff_dim, detok_hidden_size),
        f"{p}.feed_forward.w2.weight": (detok_hidden_size, detok_ff_dim),
        f"{p}.feed_forward.w3.weight": (detok_ff_dim, detok_hidden_size),
    })
    if lt == "conv":
      detok.update({
          f"{p}.conv.conv.weight": (detok_hidden_size, 1, 3),
          f"{p}.conv.in_proj.weight": (
              3 * detok_hidden_size,
              detok_hidden_size,
          ),
          f"{p}.conv.out_proj.weight": (detok_hidden_size, detok_hidden_size),
      })
    else:
      detok.update({
          f"{p}.self_attn.q_proj.weight": (
              detok_hidden_size,
              detok_hidden_size,
          ),
          f"{p}.self_attn.k_proj.weight": (detok_kv_dim, detok_hidden_size),
          f"{p}.self_attn.v_proj.weight": (detok_kv_dim, detok_hidden_size),
          f"{p}.self_attn.out_proj.weight": (
              detok_hidden_size,
              detok_hidden_size,
          ),
          f"{p}.self_attn.q_layernorm.weight": (detok_head_dim,),
          f"{p}.self_attn.k_layernorm.weight": (detok_head_dim,),
      })
  return main, detok


class Lfm2AudioTest(absltest.TestCase):
  """Tests for LFM2.5-Audio-1.5B modules, strict weight loading, and export."""

  def _write_tiny_lfm2_audio_checkpoint(
      self, model_dir: str, *, single_file_detokenizer: bool = False
  ) -> None:
    """Writes a tiny LFM2.5-Audio checkpoint matching real safetensors keys."""
    os.makedirs(model_dir, exist_ok=True)
    detok_dir = os.path.join(model_dir, "audio_detokenizer")
    if not single_file_detokenizer:
      os.makedirs(detok_dir, exist_ok=True)

    # Match real LiquidAI/LFM2.5-Audio-1.5B config.json structure (no top-level
    # model_type, architectures=["Lfm2AudioForConditionalGeneration"]).
    config_dict = {
        "architectures": ["Lfm2AudioForConditionalGeneration"],
        "codebooks": 8,
        "audio_vocab_size": 33,
        "tie_audio_embeddings": False,
        "semantic_codebook_factor": 100.0,
        "codebook_weight": "log",
        "interleaved_n_text": 6,
        "interleaved_n_audio": 12,
        "audio_adapter_hidden_size": 32,
        "encoder": {
            "feat_in": 128,
            "n_layers": 2,
            "d_model": 32,
            "feat_out": -1,
            "subsampling": "dw_striding",
            "subsampling_factor": 8,
            "subsampling_conv_channels": 16,
            "ff_expansion_factor": 2,
            "self_attention_model": "rel_pos",
            "n_heads": 4,
            "xscaling": False,
            "untie_biases": True,
            "pos_emb_max_len": 128,
            "conv_kernel_size": 9,
        },
        "lfm": {
            "architectures": ["Lfm2ForCausalLM"],
            "model_type": "lfm2",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "conv_L_cache": 3,
            "conv_bias": False,
            "block_auto_adjust_ff_dim": True,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 16,
            "norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": True,
            "layer_types": ["conv", "full_attention"],
        },
        "depthformer": {
            "layers": 2,
            "dim": 32,
            "ff_dim": 64,
            "num_heads": 4,
            "gqa_dim": 2,
            "tie": True,
        },
    }
    detok_config_dict = {
        "codebooks": 8,
        "vocab_size": 32,
        "upsample_factor": 6,
        "n_fft": 64,
        "hop_length": 16,
        "lfm": {
            "model_type": "lfm2",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "conv_L_cache": 3,
            "conv_bias": False,
            "block_auto_adjust_ff_dim": True,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 16,
            "norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "sliding_window": 8,
            "layer_types": ["conv", "sliding_attention"],
        },
    }
    if single_file_detokenizer:
      config_dict["audio_detokenizer"] = detok_config_dict
    else:
      with open(os.path.join(detok_dir, "config.json"), "w") as f:
        json.dump(detok_config_dict, f)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
      json.dump(config_dict, f)

    # Construct tensors independently from the HF checkpoint key->shape schema
    # (not from `Lfm2Audio` modules' own `state_dict()`).
    torch.manual_seed(1234)
    tiny_main_schema, tiny_detok_schema = _build_expected_checkpoint_schema(
        conformer_layers=2,
        conformer_d_model=32,
        conformer_conv_channels=16,
        conformer_heads=4,
        adapter_hidden=32,
        llm_vocab_size=64,
        llm_hidden_size=32,
        llm_ff_dim=48,
        llm_kv_dim=16,
        llm_layer_types=["conv", "full_attention"],
        codebooks=8,
        audio_vocab_size=33,
        df_layers=2,
        df_dim=32,
        df_ff_dim=48,
        df_qkv_dim=64,
        df_head_dim=8,
        detok_vocab_size=32,
        detok_hidden_size=32,
        detok_ff_dim=48,
        detok_kv_dim=16,
        detok_head_dim=8,
        detok_n_fft=64,
        detok_layer_types=["conv", "sliding_attention"],
    )

    def _make_tensor(key: str, shape: tuple[int, ...]) -> torch.Tensor:
      if key.endswith("num_batches_tracked") or key == "codebook_offsets":
        return torch.zeros(shape, dtype=torch.int64)
      if key.endswith("running_var") or key.endswith("norm.weight"):
        return torch.ones(shape, dtype=torch.float32)
      if key.endswith("running_mean") or key.endswith(".bias"):
        return torch.zeros(shape, dtype=torch.float32)
      if key == "istft.window":
        return torch.hann_window(shape[0], dtype=torch.float32)
      return torch.randn(shape, dtype=torch.float32) * 0.02

    main_tensors = {
        k: _make_tensor(k, s) for k, s in sorted(tiny_main_schema.items())
    }
    for k in range(8):
      main_tensors[f"depth_embeddings.{k}.to_logits.weight"] = (
          main_tensors[f"depth_embeddings.{k}.embedding.weight"]
          .detach()
          .clone()
      )
    detok_tensors = {
        k: _make_tensor(k, s) for k, s in sorted(tiny_detok_schema.items())
    }
    if single_file_detokenizer:
      for k, v in detok_tensors.items():
        main_tensors[f"audio_detokenizer.{k}"] = v
      safetensors.torch.save_file(
          main_tensors, os.path.join(model_dir, "model.safetensors")
      )
    else:
      safetensors.torch.save_file(
          main_tensors, os.path.join(model_dir, "model.safetensors")
      )
      safetensors.torch.save_file(
          detok_tensors, os.path.join(detok_dir, "model.safetensors")
      )

  def test_extension_registry_and_metadata(self):
    """Tests LFM2-Audio extension registration and Any2AnyMetadata custom_model."""
    for model_type in lfm2_audio_ext.Lfm2AudioExtension.model_types:
      ext = extension_lib.get_model_extension(model_type)
      self.assertIsInstance(ext, lfm2_audio_ext.Lfm2AudioExtension)
      cls = exportables.get_any2any_model_cls(model_type)
      self.assertIs(cls, lfm2_audio.Lfm2Audio)

    registered_kwargs = exportable_module_config.get_registered_extra_kwargs()
    for key in lfm2_audio.LFM2_AUDIO_EXTRA_KWARGS:
      self.assertIn(key, registered_kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
      model_dir = os.path.join(tmpdir, "ckpt")
      self._write_tiny_lfm2_audio_checkpoint(model_dir)
      with open(os.path.join(model_dir, "config.json"), "r") as f:
        raw_cfg = json.load(f)

      self.assertNotIn("model_type", raw_cfg)
      self.assertTrue(
          lfm2_audio_ext.Lfm2AudioExtension.match_checkpoint(
              model_dir, task=None, config_dict=raw_cfg
          )
      )

      exporter = lfm2_audio.Lfm2Audio(model_dir)
      meta = exporter.get_any2any_metadata()
      self.assertEqual(meta.input_sample_rate_hz, 16000)
      self.assertEqual(meta.output_sample_rate_hz, 24000)
      self.assertEqual(list(meta.supported_languages), ["en-US"])
      self.assertTrue(meta.any2any_model_type.HasField("custom_model"))
      self.assertTrue(
          meta.any2any_model_type.custom_model.Is(
              lfm2_audio_pb2.Lfm2Audio.DESCRIPTOR
          )
      )
      lfm2_meta = lfm2_audio_pb2.Lfm2Audio()
      self.assertTrue(meta.any2any_model_type.custom_model.Unpack(lfm2_meta))
      self.assertEqual(lfm2_meta.llm_hidden_size, 32)
      self.assertEqual(lfm2_meta.llm_vocab_size, 64)
      self.assertEqual(lfm2_meta.audio_mel_bins, 128)
      self.assertEqual(lfm2_meta.audio_subsampling_factor, 8)
      self.assertEqual(lfm2_meta.tts_hidden_size, 32)
      self.assertEqual(lfm2_meta.tts_num_codebooks, 8)
      self.assertEqual(lfm2_meta.tts_vocab_size, 33)
      self.assertTrue(lfm2_meta.generate_audio)
      self.assertEqual(lfm2_meta.max_new_tokens, 512)
      self.assertEqual(lfm2_meta.max_tts_tokens, 1024)
      self.assertEqual(lfm2_meta.voice, "default")
      self.assertEqual(lfm2_meta.language, "en-US")

  def test_resolve_plan_and_export_main_discovery(self):
    """Tests export plan auto-discovery and export_main entrypoint wiring."""
    with tempfile.TemporaryDirectory() as tmpdir:
      model_dir = os.path.join(tmpdir, "ckpt")
      self._write_tiny_lfm2_audio_checkpoint(model_dir)

      dry_plan = export_script.export(model=model_dir, dry_run=True)
      self.assertIsNotNone(dry_plan)
      self.assertEqual(dry_plan.model_type, "lfm2_audio")
      self.assertEqual(dry_plan.task, "any_to_any")
      self.assertEqual(dry_plan.task_source, "model")

    with mock.patch.object(export_main.fire, "Fire") as mock_fire:
      export_main.main()
      mock_fire.assert_called_once_with(export_script.export)

  def test_export_toggles_and_generate_audio_metadata(self):
    """Tests export_audio_encoder / CLI string bool toggles and generate_audio."""
    self.assertNotIn("export_audio_encoder", lfm2_audio.LFM2_AUDIO_EXTRA_KWARGS)

    with tempfile.TemporaryDirectory() as tmpdir:
      model_dir = os.path.join(tmpdir, "ckpt")
      out_dir = os.path.join(tmpdir, "out")
      self._write_tiny_lfm2_audio_checkpoint(model_dir)

      dry_plan = export_script.export(
          model=model_dir,
          export_audio_encoder=False,
          export_tts="false",
          dry_run=True,
      )
      self.assertIsNotNone(dry_plan)
      self.assertEqual(dry_plan.model_type, "lfm2_audio")

      exporter = lfm2_audio.Lfm2Audio(model_dir)
      with (
          mock.patch.object(
              exporter, "export_audio_encoder", return_value="enc.tflite"
          ) as mock_enc,
          mock.patch.object(
              exporter, "export_text_embedder", return_value="emb.tflite"
          ) as mock_emb,
          mock.patch.object(
              exporter, "export_llm_backbone", return_value="llm.tflite"
          ) as mock_llm,
          mock.patch.object(
              exporter, "export_tts_projector", return_value="proj.tflite"
          ) as mock_proj,
          mock.patch.object(
              exporter, "export_tts_decoder", return_value="df.tflite"
          ) as mock_df,
          mock.patch.object(
              exporter, "export_audio_vocoder", return_value="voc.tflite"
          ) as mock_voc,
      ):
        # Default ANY_TO_ANY config (`export_audio_encoder=None` resolves to
        # True in `ExportableModuleConfig.__post_init__`): exports all 6.
        cfg_all = exportable_module_config.ExportableModuleConfig(
            model=model_dir,
            output_dir=out_dir,
            task=exportable_module_config.ExportTask.ANY_TO_ANY,
            bundle_litert_lm=False,
        )
        artifacts_all = exporter.export(cfg_all)
        self.assertCountEqual(
            artifacts_all.keys(),
            [
                "audio_encoder",
                "text_embedder",
                "llm_backbone",
                "tts_projector",
                "tts_decoder",
                "audio_vocoder",
            ],
        )
        meta_all = exporter.get_any2any_metadata()
        lfm2_all = lfm2_audio_pb2.Lfm2Audio()
        self.assertTrue(
            meta_all.any2any_model_type.custom_model.Unpack(lfm2_all)
        )
        self.assertTrue(lfm2_all.generate_audio)

        for m in (mock_enc, mock_emb, mock_llm, mock_proj, mock_df, mock_voc):
          m.reset_mock()

        # export_audio_encoder=False + lowercase Fire string 'false'/'0'/'no'.
        cfg_disabled = exportable_module_config.ExportableModuleConfig(
            model=model_dir,
            output_dir=out_dir,
            task=exportable_module_config.ExportTask.ANY_TO_ANY,
            bundle_litert_lm=False,
            export_audio_encoder=False,
            extra_kwargs={
                "export_llm_backbone": "no",
                "export_tts": "false",
                "export_vocoder": "0",
            },
        )
        artifacts_none = exporter.export(cfg_disabled)
        self.assertEmpty(artifacts_none)
        mock_enc.assert_not_called()
        mock_emb.assert_not_called()
        mock_llm.assert_not_called()
        mock_proj.assert_not_called()
        mock_df.assert_not_called()
        mock_voc.assert_not_called()

        # Text-only bundle: export_tts='false' sets generate_audio=False.
        cfg_no_tts = exportable_module_config.ExportableModuleConfig(
            model=model_dir,
            output_dir=out_dir,
            task=exportable_module_config.ExportTask.ANY_TO_ANY,
            bundle_litert_lm=False,
            export_audio_encoder=False,
            extra_kwargs={"export_tts": "false", "export_vocoder": "true"},
        )
        artifacts_no_tts = exporter.export(cfg_no_tts)
        self.assertCountEqual(
            artifacts_no_tts.keys(),
            ["text_embedder", "llm_backbone", "audio_vocoder"],
        )
        meta_no_tts = exporter.get_any2any_metadata(cfg_no_tts)
        lfm2_no_tts = lfm2_audio_pb2.Lfm2Audio()
        self.assertTrue(
            meta_no_tts.any2any_model_type.custom_model.Unpack(lfm2_no_tts)
        )
        self.assertFalse(lfm2_no_tts.generate_audio)

        # Missing vocoder also disables generate_audio in metadata.
        cfg_no_voc = exportable_module_config.ExportableModuleConfig(
            model=model_dir,
            output_dir=out_dir,
            task=exportable_module_config.ExportTask.ANY_TO_ANY,
            bundle_litert_lm=False,
            extra_kwargs={"export_tts": True, "export_vocoder": "false"},
        )
        meta_no_voc = exporter.get_any2any_metadata(cfg_no_voc)
        lfm2_no_voc = lfm2_audio_pb2.Lfm2Audio()
        self.assertTrue(
            meta_no_voc.any2any_model_type.custom_model.Unpack(lfm2_no_voc)
        )
        self.assertFalse(lfm2_no_voc.generate_audio)

        # Invalid CLI boolean string raises ValueError.
        cfg_invalid = exportable_module_config.ExportableModuleConfig(
            model=model_dir,
            output_dir=out_dir,
            task=exportable_module_config.ExportTask.ANY_TO_ANY,
            bundle_litert_lm=False,
            extra_kwargs={"export_tts": "maybe"},
        )
        with self.assertRaisesRegex(ValueError, "Invalid boolean value"):
          exporter.export(cfg_invalid)

  def test_llm_generate_module_concatenates_logits_and_last_hidden_state(self):
    """Tests `_Lfm2LlmGenerateModule` concatenates `[logits, last_hidden]`."""
    with tempfile.TemporaryDirectory() as tmpdir:
      model_dir = os.path.join(tmpdir, "ckpt")
      self._write_tiny_lfm2_audio_checkpoint(model_dir)
      exporter = lfm2_audio.Lfm2Audio(model_dir)

      cfg_default = exportable_module_config.ExportableModuleConfig(
          model=model_dir,
          output_dir=tmpdir,
          cache_length=16,
          prefill_lengths=[8],
          externalize_embedder=True,
          single_token_embedder=True,
          split_cache=False,
      )
      with model_ext_patches.get_patch_context("lfm2", cfg_default):
        llm_backbone = (
            transformers.Lfm2ForCausalLM(exporter.lfm_config)
            .eval()
            .to(torch.float32)
        )
        sd = lfm2_audio.load_safetensors_subset(
            model_dir,
            prefixes=("lfm.",),
            strip_prefix=True,
            dtype=torch.float32,
        )
        llm_backbone.model.load_state_dict(sd, strict=True)
        with torch.no_grad():
          llm_backbone.lm_head.weight.copy_(sd["embed_tokens.weight"])

      # pylint: disable=protected-access
      with model_ext_patches.patch_model(llm_backbone, "lfm2", cfg_default):
        llm_backbone.set_attn_implementation("lrt_transposed_attention")
        mod_default = lfm2_audio._Lfm2LlmGenerateModule(
            llm_backbone, cfg_default
        ).eval()
        sample_inputs_d, _ = mod_default.get_sample_inputs(exporter.lfm_config)[
            "decode"
        ]
        sample_inputs_d["embeddings"] = torch.randn_like(
            sample_inputs_d["embeddings"]
        )
        with torch.no_grad():
          out_default = mod_default(**sample_inputs_d)
        self.assertFalse(mod_default.export_verifier)
        self.assertEqual(out_default["logits"].shape, (1, 1, 64 + 32))
        self.assertNotIn("activations", out_default)

      cfg_verifier = exportable_module_config.ExportableModuleConfig(
          model=model_dir,
          output_dir=tmpdir,
          cache_length=16,
          prefill_lengths=[8],
          externalize_embedder=True,
          single_token_embedder=True,
          split_cache=False,
          assistant_model="dummy",
      )
      with model_ext_patches.patch_model(llm_backbone, "lfm2", cfg_verifier):
        llm_backbone.set_attn_implementation("lrt_transposed_attention")
        mod_verifier = lfm2_audio._Lfm2LlmGenerateModule(
            llm_backbone, cfg_verifier
        ).eval()
        sample_inputs_v, _ = mod_verifier.get_sample_inputs(
            exporter.lfm_config
        )["decode"]
        sample_inputs_v["embeddings"] = sample_inputs_d["embeddings"]
        with torch.no_grad():
          out_verifier = mod_verifier(**sample_inputs_v)
        self.assertTrue(mod_verifier.export_verifier)
        self.assertEqual(out_verifier["logits"].shape, (1, 1, 64 + 32))
        self.assertIn("activations", out_verifier)
        torch.testing.assert_close(
            out_verifier["logits"][..., 64:],
            out_verifier["activations"].to(torch.float32),
        )
        torch.testing.assert_close(
            out_default["logits"], out_verifier["logits"]
        )
        self.assertFalse(
            torch.allclose(
                out_verifier["logits"][..., :32],
                out_verifier["logits"][..., 64:],
            )
        )
      # pylint: enable=protected-access

  def test_strict_weight_loading_and_numerical_parity(self):
    """Verifies strict=True weight loading and numerical parity vs reference ops."""
    with tempfile.TemporaryDirectory() as tmpdir:
      model_dir = os.path.join(tmpdir, "ckpt")
      self._write_tiny_lfm2_audio_checkpoint(model_dir)
      exporter = lfm2_audio.Lfm2Audio(model_dir)

      # 1. Verify strict=True loading on all 5 custom modules.
      audio_enc = exporter.build_audio_encoder_module(use_random_weights=False)
      text_emb = exporter.build_text_embedder_module(use_random_weights=False)
      tts_proj = exporter.build_tts_projector_module(use_random_weights=False)
      depthformer = exporter.build_depthformer_module(use_random_weights=False)
      detok = exporter.build_detokenizer_module(use_random_weights=False)

      mel = torch.randn(1, 128, 24)
      mel_len = torch.tensor([24], dtype=torch.int32)
      enc_out = audio_enc(mel, mel_len)
      self.assertEqual(tuple(enc_out.shape), (1, 3, 32))

      # Verify FastConformer padding mask invariance and reference parity:
      # right-padding mel from 19 frames (3 subsampled frames) to 40 frames (5
      # subsampled frames) with arbitrary noise while passing mel_length=19 must
      # produce identical first 3 subsampled frames as unpadded 19-frame input.
      mel_19 = mel[:, :, :19]
      mel_len_19 = torch.tensor([19], dtype=torch.int32)
      mel_padded = torch.cat([mel_19, torch.randn(1, 128, 21) * 5.0], dim=-1)
      with torch.no_grad():
        enc_out_19 = audio_enc(mel_19, mel_len_19)
        enc_out_padded = audio_enc(mel_padded, mel_len_19)
      self.assertEqual(tuple(enc_out_19.shape), (1, 3, 32))
      self.assertEqual(tuple(enc_out_padded.shape), (1, 5, 32))
      np.testing.assert_allclose(
          enc_out_padded[:, :3, :].detach().numpy(),
          enc_out_19.detach().numpy(),
          rtol=1e-5,
          atol=1e-5,
      )

      tok_out = text_emb(torch.tensor([[1, 6, 7]], dtype=torch.int32))
      self.assertEqual(tuple(tok_out.shape), (1, 3, 32))

      df_in, aud_emb = tts_proj(
          torch.randn(1, 1, 32),
          torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32),
      )
      self.assertEqual(tuple(df_in.shape), (1, 8, 32))
      self.assertEqual(tuple(aud_emb.shape), (1, 1, 32))

      # 2. Verify ISTFT real-IDFT + conv_transpose1d parity vs torch.fft.irfft.
      istft_mod = modules.ISTFT(n_fft=64, hop_length=16, win_length=64).eval()
      log_mag = torch.randn(1, 33, 12) * 0.2
      phase = torch.randn(1, 33, 12)
      actual_wav = istft_mod(log_mag, phase)
      ref_wav = _reference_vocos_istft(
          log_mag, phase, istft_mod.window, n_fft=64, hop_length=16
      )
      self.assertEqual(tuple(actual_wav.shape), (1, 12 * 16))
      np.testing.assert_allclose(
          actual_wav.detach().numpy(),
          ref_wav.detach().numpy(),
          rtol=1e-5,
          atol=1e-5,
      )

      # Verify detokenizer -1 right-padding invariance and overlap-discard
      # chunked detokenization parity vs one-shot detokenization:
      codes_short = torch.randint(0, 32, (1, 8, 3), dtype=torch.int32)
      codes_padded = torch.cat(
          [codes_short, torch.full((1, 8, 3), -1, dtype=torch.int32)], dim=-1
      )
      with torch.no_grad():
        wav_short = detok(codes_short)
        wav_padded = detok(codes_padded)
      np.testing.assert_allclose(
          wav_padded[:, : wav_short.shape[1]].numpy(),
          wav_short.numpy(),
          rtol=1e-5,
          atol=1e-5,
      )

      # Verify overlap-discard chunked detokenization (chunk=10, left=4, right=1
      # for tiny 2-layer detok with sliding_window=8):
      codes_long = torch.randint(0, 32, (1, 8, 16), dtype=torch.int32)
      spf = 6 * 16  # 96 samples per code frame in tiny detok
      with torch.no_grad():
        wav_oneshot = detok(codes_long)
        chunk0 = detok(codes_long[:, :, :10])[:, : 9 * spf]
        chunk1 = detok(codes_long[:, :, 6:16])[:, 3 * spf :]
        wav_chunked = torch.cat([chunk0, chunk1], dim=1)
      np.testing.assert_allclose(
          wav_chunked.numpy(),
          wav_oneshot.numpy(),
          rtol=1e-5,
          atol=1e-5,
      )

      # 3. Verify DetokLfm2Model parity against transformers.Lfm2Model.
      lfm_cfg_dict = dict(exporter.detok_config["lfm"])
      sliding_win = int(lfm_cfg_dict.pop("sliding_window", 8))
      lfm_cfg_dict["layer_types"] = [
          "full_attention" if lt == "sliding_attention" else lt
          for lt in lfm_cfg_dict["layer_types"]
      ]
      lfm_cfg = transformers.Lfm2Config.from_dict(lfm_cfg_dict)
      lfm_cfg._attn_implementation = "eager"  # pylint: disable=protected-access
      hf_lfm = transformers.Lfm2Model(lfm_cfg).eval()  # pyrefly: ignore[bad-argument-type]
      hf_lfm.load_state_dict(detok.lfm.state_dict(), strict=True)
      test_embeds = torch.randn(1, 12, 32)
      seqlen = test_embeds.shape[1]
      q_idx = torch.arange(seqlen).unsqueeze(1)
      kv_idx = torch.arange(seqlen).unsqueeze(0)
      valid = (kv_idx <= q_idx) & ((q_idx - kv_idx) < sliding_win)
      attn_mask = torch.where(
          valid,
          torch.zeros((seqlen, seqlen), dtype=torch.float32),
          torch.full((seqlen, seqlen), -1e9, dtype=torch.float32),
      ).view(1, 1, seqlen, seqlen)
      with torch.no_grad():
        actual_lfm_out = detok.lfm(test_embeds)
        ref_lfm_out = hf_lfm(
            inputs_embeds=test_embeds,
            attention_mask={"full_attention": attn_mask, "conv": None},
            use_cache=False,
        ).last_hidden_state
      np.testing.assert_allclose(
          actual_lfm_out.numpy(), ref_lfm_out.numpy(), rtol=1e-5, atol=1e-5
      )

      # 4. Verify Depthformer 8-step KV-cached execution vs complex-RoPE ref.
      kv_cache = torch.zeros(
          (
              1,
              2 * depthformer.layers,
              depthformer.codebooks,
              depthformer.kv_dim,
          ),
          dtype=torch.float32,
      )
      codes = [3, 5, 2, 7, 1, 4, 6, 0]
      step_logits_list = []
      prev_code = torch.zeros((1, 1), dtype=torch.int32)
      for step in range(8):
        cond_step = df_in[:, step : step + 1, :]
        step_tensor = torch.tensor([step], dtype=torch.int32)
        logits_step, kv_cache = depthformer(
            cond_step, prev_code, step_tensor, kv_cache
        )
        step_logits_list.append(logits_step)
        prev_code = torch.tensor([[codes[step]]], dtype=torch.int32)
      stacked_logits = torch.cat(step_logits_list, dim=1)
      self.assertEqual(tuple(stacked_logits.shape), (1, 8, 33))

      # Full sequence reference pass through the same Depthformer weights.
      seq_tokens = torch.tensor(codes, dtype=torch.long)
      token_embs = torch.stack(
          [depthformer.depth_embeddings[i](seq_tokens[i]) for i in range(8)],
          dim=0,
      )
      shifted_embs = torch.zeros_like(token_embs)
      shifted_embs[1:] = token_embs[:-1]
      seq_in = df_in + shifted_embs.unsqueeze(0)  # [1, 8, 32]
      # Verify step 0 and step 7 match complex RoPE + SDPA causal attention.
      h_seq = seq_in
      head_dim = depthformer.head_dim
      freqs = 1.0 / (
          1000000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
      )
      freqs_cis = torch.polar(
          torch.ones((8, head_dim // 2)),
          torch.outer(torch.arange(8).float(), freqs),
      )
      for block in depthformer.depthformer.layers:
        normed = block.operator_norm(h_seq)
        qkv = block.operator.qkv_proj(normed)
        q, k, v = torch.split(
            qkv, [depthformer.dim, depthformer.kv_dim, depthformer.kv_dim], -1
        )
        q = block.operator.bounded_attention.q_layernorm(
            q.view(1, 8, depthformer.num_heads, head_dim)
        )
        k = block.operator.bounded_attention.k_layernorm(
            k.view(1, 8, depthformer.gqa_dim, head_dim)
        )
        v = v.view(1, 8, depthformer.gqa_dim, head_dim)
        q_c = torch.view_as_complex(
            q.float().reshape(1, 8, -1, head_dim // 2, 2)
        )
        k_c = torch.view_as_complex(
            k.float().reshape(1, 8, -1, head_dim // 2, 2)
        )
        q = torch.view_as_real(q_c * freqs_cis[None, :, None, :]).flatten(3)
        k = torch.view_as_real(k_c * freqs_cis[None, :, None, :]).flatten(3)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        )
        h_seq = h_seq + block.operator.out_proj(
            attn_out.transpose(1, 2).reshape(1, 8, depthformer.dim)
        )
        h_seq = h_seq + block.feed_forward(block.ffn_norm(h_seq))
      ref_logits = torch.stack(
          [
              depthformer.depth_embeddings[i].get_logits(h_seq[:, i])
              for i in range(8)
          ],
          dim=1,
      )
      np.testing.assert_allclose(
          stacked_logits.detach().numpy(),
          ref_logits.detach().numpy(),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_export_all_submodels_and_verify_compiled_models(self):
    """Tests exporting all 6 submodels with strict weight loading and CompiledModel."""
    with tempfile.TemporaryDirectory() as tmpdir:
      model_dir = os.path.join(tmpdir, "ckpt")
      out_dir = os.path.join(tmpdir, "out")
      self._write_tiny_lfm2_audio_checkpoint(model_dir)
      cfg = exportable_module_config.ExportableModuleConfig(
          model=model_dir,
          output_dir=out_dir,
          work_dir=out_dir,
          task=exportable_module_config.ExportTask.ANY_TO_ANY,
          quantization_recipe=None,
          audio_encoder_quantization_recipe=None,
          use_random_weights=False,
          cache_length=16,
          prefill_lengths=[8],
          extra_kwargs={
              "audio_mel_frames": 24,
              "detok_chunk_frames": 4,
          },
      )
      exporter = lfm2_audio.Lfm2Audio(model_dir, cfg)
      artifacts = exporter.export_submodels(cfg)
      self.assertCountEqual(
          artifacts.keys(),
          [
              "audio_encoder",
              "text_embedder",
              "llm_backbone",
              "tts_projector",
              "tts_decoder",
              "audio_vocoder",
          ],
      )
      for name, path in artifacts.items():
        self.assertTrue(
            os.path.exists(path), f"Missing exported {name}: {path}"
        )

      # 1. Verify audio_encoder (128-bin mel -> 8x dw_striding -> 3 * 32):
      audio_cm = CompiledModel.from_file(artifacts["audio_encoder"])
      audio_in_bufs = audio_cm.create_input_buffers(0)
      audio_out_bufs = audio_cm.create_output_buffers(0)
      mel_np = np.random.RandomState(0).randn(1, 128, 24).astype(np.float32)
      mel_len_np = np.array([16], dtype=np.int32)
      audio_in_bufs[0].write(mel_np)
      audio_in_bufs[1].write(mel_len_np)
      audio_cm.run_by_index(0, audio_in_bufs, audio_out_bufs)
      audio_out = audio_out_bufs[0].read(3 * 32, np.float32).reshape(1, 3, 32)
      py_audio_enc = exporter.build_audio_encoder_module(
          use_random_weights=False
      )
      with torch.no_grad():
        ref_audio_out = py_audio_enc(
            torch.from_numpy(mel_np[:, :, :16]),
            torch.tensor([16], dtype=torch.int32),
        ).numpy()
      np.testing.assert_allclose(
          audio_out[:, :2, :], ref_audio_out, rtol=1e-4, atol=1e-4
      )

      # 2. Verify text_embedder numerical parity against PyTorch module:
      text_cm = CompiledModel.from_file(artifacts["text_embedder"])
      text_in_bufs = text_cm.create_input_buffers(0)
      text_out_bufs = text_cm.create_output_buffers(0)
      token_ids_np = np.array([[7]], dtype=np.int32)
      text_in_bufs[0].write(token_ids_np)
      text_cm.run_by_index(0, text_in_bufs, text_out_bufs)
      text_out = text_out_bufs[0].read(32, np.float32).reshape(1, 1, 32)
      py_text_emb = exporter.build_text_embedder_module(
          use_random_weights=False
      )
      with torch.no_grad():
        ref_text_out = py_text_emb(torch.from_numpy(token_ids_np)).numpy()
      np.testing.assert_allclose(text_out, ref_text_out, rtol=1e-4, atol=1e-4)

      # 3. Verify llm_backbone has hybrid conv + attention KV cache signatures
      # and decode `logits` carries concatenated `[vocab (64), hidden (32)]`:
      llm_cm = CompiledModel.from_file(artifacts["llm_backbone"])
      sig_list = llm_cm.get_signature_list()
      self.assertIn("prefill_8", sig_list)
      self.assertIn("decode", sig_list)
      decode_in_names = sig_list["decode"]["inputs"]
      decode_out_names = sig_list["decode"]["outputs"]
      self.assertIn("kv_cache_c_0", decode_in_names)
      self.assertIn("kv_cache_k_1", decode_in_names)
      self.assertIn("kv_cache_v_1", decode_in_names)
      self.assertIn("logits", decode_out_names)
      decode_out_details = llm_cm.get_output_tensor_details("decode")
      self.assertEqual(list(decode_out_details["logits"]["shape"]), [1, 1, 96])

      # 4. Verify tts_projector both outputs (`depthformer_in`, `audio_emb`)
      # by index against PyTorch module on non-zero inputs:
      rng = np.random.RandomState(42)
      proj_cm = CompiledModel.from_file(artifacts["tts_projector"])
      proj_in_bufs = proj_cm.create_input_buffers(0)
      proj_out_bufs = proj_cm.create_output_buffers(0)
      llm_hidden_np = rng.randn(1, 1, 32).astype(np.float32)
      audio_codes_np = np.array([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=np.int32)
      proj_in_bufs[0].write(llm_hidden_np)
      proj_in_bufs[1].write(audio_codes_np)
      proj_cm.run_by_index(0, proj_in_bufs, proj_out_bufs)
      self.assertLen(proj_out_bufs, 2)
      df_in_out = proj_out_bufs[0].read(8 * 32, np.float32).reshape(1, 8, 32)
      audio_emb_out = proj_out_bufs[1].read(32, np.float32).reshape(1, 1, 32)
      py_tts_proj = exporter.build_tts_projector_module(
          use_random_weights=False
      )
      with torch.no_grad():
        ref_df_in, ref_audio_emb = py_tts_proj(
            torch.from_numpy(llm_hidden_np), torch.from_numpy(audio_codes_np)
        )
      np.testing.assert_allclose(
          df_in_out, ref_df_in.numpy(), rtol=1e-4, atol=1e-4
      )
      np.testing.assert_allclose(
          audio_emb_out, ref_audio_emb.numpy(), rtol=1e-4, atol=1e-4
      )

      # 5. Verify tts_decoder (Depthformer) across all 8 autoregressive steps
      # comparing both `logits` and `new_kv_cache` by index against PyTorch:
      df_cm = CompiledModel.from_file(artifacts["tts_decoder"])
      df_in_bufs = df_cm.create_input_buffers(0)
      df_out_bufs = df_cm.create_output_buffers(0)
      py_df = exporter.build_depthformer_module(use_random_weights=False)
      kv_np = np.zeros((1, 4, 8, 16), dtype=np.float32)
      kv_pt = torch.zeros((1, 4, 8, 16), dtype=torch.float32)
      prev_code = 0
      for step in range(8):
        cond_np = df_in_out[:, step : step + 1, :].astype(np.float32)
        prev_code_np = np.array([[prev_code]], dtype=np.int32)
        step_idx_np = np.array([step], dtype=np.int32)
        df_in_bufs[0].write(cond_np)
        df_in_bufs[1].write(prev_code_np)
        df_in_bufs[2].write(step_idx_np)
        df_in_bufs[3].write(kv_np)
        df_cm.run_by_index(0, df_in_bufs, df_out_bufs)
        self.assertLen(df_out_bufs, 2)
        df_logits = df_out_bufs[0].read(33, np.float32).reshape(1, 1, 33)
        kv_np = df_out_bufs[1].read(4 * 8 * 16, np.float32).reshape(1, 4, 8, 16)
        with torch.no_grad():
          ref_logits, kv_pt = py_df(
              torch.from_numpy(cond_np),
              torch.from_numpy(prev_code_np),
              torch.from_numpy(step_idx_np),
              kv_pt,
          )
        np.testing.assert_allclose(
            df_logits, ref_logits.numpy(), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(kv_np, kv_pt.numpy(), rtol=1e-4, atol=1e-4)
        prev_code = int(np.argmax(df_logits[0, 0]))

      # 6. Verify audio_vocoder (LFM2AudioDetokenizer) numerical parity:
      voc_cm = CompiledModel.from_file(artifacts["audio_vocoder"])
      voc_in_bufs = voc_cm.create_input_buffers(0)
      voc_out_bufs = voc_cm.create_output_buffers(0)
      voc_codes_np = rng.randint(0, 32, size=(1, 8, 4), dtype=np.int32)
      voc_in_bufs[0].write(voc_codes_np)
      voc_cm.run_by_index(0, voc_in_bufs, voc_out_bufs)
      wav_out = voc_out_bufs[0].read(4 * 6 * 16, np.float32).reshape(1, 384)
      py_detok = exporter.build_detokenizer_module(use_random_weights=False)
      with torch.no_grad():
        ref_wav = py_detok(torch.from_numpy(voc_codes_np)).numpy()
      np.testing.assert_allclose(wav_out, ref_wav, rtol=1e-4, atol=1e-4)

  def test_end_to_end_export_bundles_litertlm(self):
    """Tests end-to-end export_script.export bundling a .litertlm file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      model_dir = os.path.join(tmpdir, "ckpt")
      out_dir = os.path.join(tmpdir, "out")
      self._write_tiny_lfm2_audio_checkpoint(model_dir)
      with open(os.path.join(model_dir, "tokenizer.json"), "w") as f:
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
      export_script.export(
          model=model_dir,
          output_dir=out_dir,
          task="auto",
          quantization_recipe="none",
          use_random_weights=False,
          bundle_litert_lm=True,
          cache_length=16,
          prefill_lengths=[8],
          extra_kwargs={
              "audio_mel_frames": 16,
              "detok_chunk_frames": 4,
          },
      )
      litertlm_path = os.path.join(out_dir, "model.litertlm")
      self.assertTrue(os.path.exists(litertlm_path))

      dump_dir = os.path.join(tmpdir, "unpacked")
      os.makedirs(dump_dir, exist_ok=True)
      with open(os.path.join(tmpdir, "peek.txt"), "w") as peek_out:
        litertlm_peek.peek_litertlm_file(litertlm_path, dump_dir, peek_out)
      with open(os.path.join(tmpdir, "peek.txt"), "r") as peek_in:
        peek_text = peek_in.read()

      self.assertIn("Data Type:    Any2AnyMetadataProto", peek_text)
      self.assertIn("custom_model {", peek_text)
      self.assertIn(
          "type.googleapis.com/litert.lm.proto.Lfm2Audio",
          peek_text,
      )
      pbtext_path = os.path.join(dump_dir, "Any2AnyMetadataProto.pbtext")
      self.assertTrue(os.path.exists(pbtext_path))
      with open(pbtext_path, "r") as pbtext_file:
        parsed_meta = text_format.Parse(
            pbtext_file.read(), any2any_metadata_pb2.Any2AnyMetadata()
        )
      self.assertEqual(parsed_meta.input_sample_rate_hz, 16000)
      self.assertEqual(parsed_meta.output_sample_rate_hz, 24000)
      self.assertTrue(parsed_meta.any2any_model_type.HasField("custom_model"))
      self.assertTrue(
          parsed_meta.any2any_model_type.custom_model.Is(
              lfm2_audio_pb2.Lfm2Audio.DESCRIPTOR
          )
      )
      unpacked = lfm2_audio_pb2.Lfm2Audio()
      self.assertTrue(
          parsed_meta.any2any_model_type.custom_model.Unpack(unpacked)
      )
      self.assertEqual(unpacked.llm_hidden_size, 32)
      self.assertEqual(unpacked.llm_vocab_size, 64)
      self.assertEqual(unpacked.tts_hidden_size, 32)
      self.assertEqual(unpacked.tts_vocab_size, 33)
      self.assertEqual(unpacked.tts_num_codebooks, 8)
      self.assertEqual(unpacked.audio_mel_bins, 128)
      self.assertEqual(unpacked.audio_subsampling_factor, 8)
      self.assertTrue(unpacked.generate_audio)
      self.assertIn("Data Type:    HF_Tokenizer_Zlib", peek_text)
      self.assertIn(
          "Key: model_type, Value (String): tf_lite_audio_encoder", peek_text
      )
      self.assertIn(
          "Key: model_type, Value (String): tf_lite_text_embedder", peek_text
      )
      self.assertIn(
          "Key: model_type, Value (String): tf_lite_llm_backbone", peek_text
      )
      self.assertIn(
          "Key: model_type, Value (String): tf_lite_tts_projector", peek_text
      )
      self.assertIn(
          "Key: model_type, Value (String): tf_lite_tts_decoder", peek_text
      )
      self.assertIn(
          "Key: model_type, Value (String): tf_lite_audio_vocoder", peek_text
      )

  def test_full_scale_lfm2_5_audio_1_5b_schema_and_unconsumed_key_rejection(
      self,
  ):
    """Verifies exact full-scale LiquidAI/LFM2.5-Audio-1.5B 931+79 key/shape schema."""
    real_config = {
        "architectures": ["Lfm2AudioForConditionalGeneration"],
        "audio_adapter_hidden_size": 2048,
        "audio_vocab_size": 2049,
        "codebook_weight": "log",
        "codebooks": 8,
        "depthformer": {
            "dim": 1024,
            "ff_dim": 4096,
            "gqa_dim": 8,
            "layers": 6,
            "num_heads": 32,
            "tie": True,
        },
        "encoder": {
            "conv_kernel_size": 9,
            "d_model": 512,
            "feat_in": 128,
            "feat_out": -1,
            "ff_expansion_factor": 4,
            "n_heads": 8,
            "n_layers": 17,
            "pos_emb_max_len": 5000,
            "self_attention_model": "rel_pos",
            "subsampling": "dw_striding",
            "subsampling_conv_channels": 256,
            "subsampling_factor": 8,
            "untie_biases": True,
            "xscaling": False,
        },
        "lfm": {
            "architectures": ["Lfm2ForCausalLM"],
            "block_auto_adjust_ff_dim": True,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 256,
            "conv_L_cache": 3,
            "conv_bias": False,
            "hidden_size": 2048,
            "intermediate_size": 12288,
            "layer_types": [
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
            ],
            "model_type": "lfm2",
            "norm_eps": 1e-05,
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": True,
            "vocab_size": 65536,
        },
        "semantic_codebook_factor": 100.0,
        "tie_audio_embeddings": False,
    }
    real_detok_config = {
        "architectures": ["Lfm25AudioTokenizer"],
        "block_auto_adjust_ff_dim": True,
        "block_dim": 512,
        "block_ff_dim": 3328,
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 256,
        "block_norm_eps": 1e-05,
        "block_use_swiglu": True,
        "bos_token_id": 1,
        "conv_L_cache": 3,
        "conv_bias": False,
        "conv_dim": 512,
        "conv_dim_out": 512,
        "eos_token_id": 7,
        "hidden_size": 512,
        "intermediate_size": 3328,
        "layer_types": [
            "conv",
            "conv",
            "sliding_attention",
            "conv",
            "sliding_attention",
            "conv",
            "sliding_attention",
            "conv",
        ],
        "max_position_embeddings": 128000,
        "model_type": "lfm2",
        "norm_eps": 1e-05,
        "num_attention_heads": 16,
        "num_heads": 16,
        "num_hidden_layers": 8,
        "num_key_value_heads": 8,
        "output_size": 1282,
        "pad_token_id": 0,
        "rope_theta": 1000000.0,
        "sliding_window": 30,
        "use_cache": True,
        "use_pos_enc": True,
        "vocab_size": 65536,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
      model_dir = os.path.join(tmpdir, "full_cfg")
      detok_dir = os.path.join(model_dir, "audio_detokenizer")
      os.makedirs(detok_dir, exist_ok=True)
      with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(real_config, f)
      with open(os.path.join(detok_dir, "config.json"), "w") as f:
        json.dump(real_detok_config, f)

      exporter = lfm2_audio.Lfm2Audio(model_dir)
      with torch.device("meta"):
        audio_enc = exporter.build_audio_encoder_module(use_random_weights=True)
        tts_proj = exporter.build_tts_projector_module(use_random_weights=True)
        depthformer = exporter.build_depthformer_module(use_random_weights=True)
        lfm_causal = transformers.Lfm2ForCausalLM(exporter.lfm_config)
        detok = exporter.build_detokenizer_module(use_random_weights=True)

      main_shapes: dict[str, tuple[int, ...]] = {}
      for mod in (audio_enc, tts_proj, depthformer):
        for k, v in mod.state_dict().items():
          main_shapes[k] = tuple(v.shape)
      for k, v in lfm_causal.model.state_dict().items():
        main_shapes[f"lfm.{k}"] = tuple(v.shape)

      expected_main_shapes, expected_detok_shapes = (
          _build_expected_checkpoint_schema(
              conformer_layers=17,
              conformer_d_model=512,
              conformer_conv_channels=256,
              conformer_heads=8,
              adapter_hidden=2048,
              llm_vocab_size=65536,
              llm_hidden_size=2048,
              llm_ff_dim=8192,
              llm_kv_dim=512,
              llm_layer_types=[
                  "conv",
                  "conv",
                  "full_attention",
                  "conv",
                  "conv",
                  "full_attention",
                  "conv",
                  "conv",
                  "full_attention",
                  "conv",
                  "full_attention",
                  "conv",
                  "full_attention",
                  "conv",
                  "full_attention",
                  "conv",
              ],
              codebooks=8,
              audio_vocab_size=2049,
              df_layers=6,
              df_dim=1024,
              df_ff_dim=2816,
              df_qkv_dim=1536,
              df_head_dim=32,
              detok_vocab_size=2048,
              detok_hidden_size=512,
              detok_ff_dim=2304,
              detok_kv_dim=256,
              detok_head_dim=32,
              detok_n_fft=1280,
              detok_layer_types=[
                  "conv",
                  "conv",
                  "sliding_attention",
                  "conv",
                  "sliding_attention",
                  "conv",
                  "sliding_attention",
                  "conv",
              ],
          )
      )

      # Exact 931 tensors in LiquidAI/LFM2.5-Audio-1.5B/model.safetensors:
      self.assertLen(main_shapes, 931)
      self.assertEqual(main_shapes, expected_main_shapes)

      # Exact 79 tensors in audio_detokenizer/model.safetensors:
      detok_shapes = {k: tuple(v.shape) for k, v in detok.state_dict().items()}
      self.assertLen(detok_shapes, 79)
      self.assertEqual(detok_shapes, expected_detok_shapes)

      # Verify that verify_checkpoint_keys rejects unconsumed checkpoint keys.
      tiny_dir = os.path.join(tmpdir, "tiny_ckpt")
      self._write_tiny_lfm2_audio_checkpoint(tiny_dir)
      tiny_exporter = lfm2_audio.Lfm2Audio(tiny_dir)
      tiny_exporter.verify_checkpoint_keys()

      bad_tensors = safetensors.torch.load_file(
          os.path.join(tiny_dir, "model.safetensors")
      )
      bad_tensors["unexpected_unconsumed_layer.weight"] = torch.zeros((4, 4))
      safetensors.torch.save_file(
          bad_tensors, os.path.join(tiny_dir, "model.safetensors")
      )
      with self.assertRaisesRegex(
          ValueError, "Unconsumed checkpoint keys.*unexpected_unconsumed_layer"
      ):
        tiny_exporter.verify_checkpoint_keys()

  def test_single_file_checkpoint_with_audio_detokenizer_keys(self):
    """Verifies single-file checkpoints with inline `audio_detokenizer.` keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
      split_dir = os.path.join(tmpdir, "split_ckpt")
      single_dir = os.path.join(tmpdir, "single_ckpt")
      self._write_tiny_lfm2_audio_checkpoint(
          split_dir, single_file_detokenizer=False
      )
      self._write_tiny_lfm2_audio_checkpoint(
          single_dir, single_file_detokenizer=True
      )
      self.assertFalse(
          os.path.exists(os.path.join(single_dir, "audio_detokenizer"))
      )

      single_exporter = lfm2_audio.Lfm2Audio(single_dir)
      single_exporter.verify_checkpoint_keys()
      single_detok = single_exporter.build_detokenizer_module(
          use_random_weights=False
      )

      split_exporter = lfm2_audio.Lfm2Audio(split_dir)
      split_detok = split_exporter.build_detokenizer_module(
          use_random_weights=False
      )
      codes = torch.randint(0, 32, (1, 8, 4), dtype=torch.int32)
      with torch.no_grad():
        single_wav = single_detok(codes).numpy()
        split_wav = split_detok(codes).numpy()
      np.testing.assert_allclose(single_wav, split_wav, rtol=0.0, atol=0.0)


if __name__ == "__main__":
  absltest.main()
