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
"""Cross-language CLI and exporter-to-C++-runner E2E tests for LFM2.5-Audio."""

import json
import math
import os
import shutil
import struct
import subprocess
import wave

from absl.testing import absltest
import safetensors.torch
import torch


def _runfiles_path(rel_path: str) -> str:
  """Resolves a workspace-relative data dependency path inside TEST_SRCDIR."""
  srcdir = os.environ.get("TEST_SRCDIR") or absltest.get_default_test_srcdir()
  return os.path.join(srcdir, os.environ.get("TEST_WORKSPACE", "litert"), rel_path)


def _write_test_wav(path: str, sample_rate_hz: int, num_samples: int) -> None:
  """Writes a valid 16-bit mono PCM WAV file with a 440 Hz sine wave."""
  frames = bytearray()
  for i in range(num_samples):
    sample = 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate_hz)
    pcm16 = max(-32768, min(32767, int(round(sample * 32767.0))))
    frames.extend(struct.pack("<h", pcm16))
  with wave.open(path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate_hz)
    wf.writeframes(bytes(frames))


def _write_tiny_e2e_checkpoint(model_dir: str) -> None:
  """Writes a tiny deterministic LFM2.5-Audio HF checkpoint with tokenizer."""
  os.makedirs(model_dir, exist_ok=True)
  detok_dir = os.path.join(model_dir, "audio_detokenizer")
  os.makedirs(detok_dir, exist_ok=True)

  llm_vocab_size = 144
  llm_hidden_size = 32
  llm_ff_dim = 48
  llm_kv_dim = 16
  llm_head_dim = 8
  llm_layer_types = ["conv", "full_attention"]

  conformer_layers = 2
  conformer_d_model = 32
  conformer_conv_channels = 16
  conformer_heads = 4
  conformer_ff = conformer_d_model * 2
  conformer_head_dim = conformer_d_model // conformer_heads
  adapter_hidden = 32

  codebooks = 8
  audio_vocab_size = 33
  df_layers = 2
  df_dim = 32
  df_ff_dim = 48
  df_qkv_dim = 64
  df_head_dim = 8

  detok_vocab_size = 32
  detok_hidden_size = 32
  detok_ff_dim = 48
  detok_kv_dim = 16
  detok_head_dim = 8
  detok_n_fft = 64
  detok_layer_types = ["conv", "sliding_attention"]

  config_dict = {
      "architectures": ["Lfm2AudioForConditionalGeneration"],
      "codebooks": codebooks,
      "audio_vocab_size": audio_vocab_size,
      "tie_audio_embeddings": False,
      "semantic_codebook_factor": 100.0,
      "codebook_weight": "log",
      "interleaved_n_text": 6,
      "interleaved_n_audio": 12,
      "audio_adapter_hidden_size": adapter_hidden,
      "encoder": {
          "feat_in": 128,
          "n_layers": conformer_layers,
          "d_model": conformer_d_model,
          "feat_out": -1,
          "subsampling": "dw_striding",
          "subsampling_factor": 8,
          "subsampling_conv_channels": conformer_conv_channels,
          "ff_expansion_factor": 2,
          "self_attention_model": "rel_pos",
          "n_heads": conformer_heads,
          "xscaling": False,
          "untie_biases": True,
          "pos_emb_max_len": 128,
          "conv_kernel_size": 9,
      },
      "lfm": {
          "architectures": ["Lfm2ForCausalLM"],
          "model_type": "lfm2",
          "vocab_size": llm_vocab_size,
          "hidden_size": llm_hidden_size,
          "intermediate_size": 64,
          "num_hidden_layers": len(llm_layer_types),
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
          "layer_types": llm_layer_types,
      },
      "depthformer": {
          "layers": df_layers,
          "dim": df_dim,
          "ff_dim": 64,
          "num_heads": 4,
          "gqa_dim": 2,
          "tie": True,
      },
  }
  with open(os.path.join(model_dir, "config.json"), "w") as f:
    json.dump(config_dict, f)

  detok_config_dict = {
      "codebooks": codebooks,
      "vocab_size": detok_vocab_size,
      "upsample_factor": 6,
      "n_fft": detok_n_fft,
      "hop_length": 16,
      "lfm": {
          "model_type": "lfm2",
          "vocab_size": llm_vocab_size,
          "hidden_size": detok_hidden_size,
          "intermediate_size": 64,
          "num_hidden_layers": len(detok_layer_types),
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
          "layer_types": detok_layer_types,
      },
  }
  with open(os.path.join(detok_dir, "config.json"), "w") as f:
    json.dump(detok_config_dict, f)

  # Build a WordLevel tokenizer.json containing all LFM2 chat/audio special
  # token IDs (0, 1, 5, 6, 7, 128, 130) plus prompt words so prompts stay
  # compact within `cache_length=64`.
  vocab = {
      "<|pad|>": 0,
      "<|startoftext|>": 1,
      "t2": 2,
      "Hello": 3,
      "world": 4,
      "\n": 5,
      "<|im_start|>": 6,
      "<|im_end|>": 7,
      "Respond": 8,
      "with": 9,
      "interleaved": 10,
      "text": 11,
      "and": 12,
      "audio.": 13,
      "system": 14,
      "user": 15,
      "assistant": 16,
      "<|audio_start|>": 128,
      "<|text_end|>": 130,
  }
  used_ids = set(vocab.values())
  for token_id in range(llm_vocab_size):
    if token_id not in used_ids:
      vocab[f"tok_{token_id}"] = token_id

  tokenizer_json = {
      "version": "1.0",
      "truncation": None,
      "padding": None,
      "added_tokens": [
          {
              "id": tid,
              "content": tok,
              "single_word": False,
              "lstrip": False,
              "rstrip": False,
              "normalized": False,
              "special": True,
          }
          for tok, tid in [
              ("<|pad|>", 0),
              ("<|startoftext|>", 1),
              ("\n", 5),
              ("<|im_start|>", 6),
              ("<|im_end|>", 7),
              ("<|audio_start|>", 128),
              ("<|text_end|>", 130),
          ]
      ],
      "normalizer": None,
      "pre_tokenizer": {"type": "Whitespace"},
      "post_processor": None,
      "decoder": None,
      "model": {
          "type": "WordLevel",
          "vocab": vocab,
          "unk_token": "<|pad|>",
      },
  }
  with open(os.path.join(model_dir, "tokenizer.json"), "w") as f:
    json.dump(tokenizer_json, f)

  main_schema: dict[str, tuple[int, ...]] = {
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
    main_schema.update({
        f"{p}.norm_feed_forward1.weight": (conformer_d_model,),
        f"{p}.norm_feed_forward1.bias": (conformer_d_model,),
        f"{p}.feed_forward1.linear1.weight": (conformer_ff, conformer_d_model),
        f"{p}.feed_forward1.linear1.bias": (conformer_ff,),
        f"{p}.feed_forward1.linear2.weight": (conformer_d_model, conformer_ff),
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
        f"{p}.self_attn.pos_bias_u": (conformer_heads, conformer_head_dim),
        f"{p}.self_attn.pos_bias_v": (conformer_heads, conformer_head_dim),
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
        f"{p}.feed_forward2.linear1.weight": (conformer_ff, conformer_d_model),
        f"{p}.feed_forward2.linear1.bias": (conformer_ff,),
        f"{p}.feed_forward2.linear2.weight": (conformer_d_model, conformer_ff),
        f"{p}.feed_forward2.linear2.bias": (conformer_d_model,),
        f"{p}.norm_out.weight": (conformer_d_model,),
        f"{p}.norm_out.bias": (conformer_d_model,),
    })
  for i, lt in enumerate(llm_layer_types):
    p = f"lfm.layers.{i}"
    main_schema.update({
        f"{p}.operator_norm.weight": (llm_hidden_size,),
        f"{p}.ffn_norm.weight": (llm_hidden_size,),
        f"{p}.feed_forward.w1.weight": (llm_ff_dim, llm_hidden_size),
        f"{p}.feed_forward.w2.weight": (llm_hidden_size, llm_ff_dim),
        f"{p}.feed_forward.w3.weight": (llm_ff_dim, llm_hidden_size),
    })
    if lt == "conv":
      main_schema.update({
          f"{p}.conv.conv.weight": (llm_hidden_size, 1, 3),
          f"{p}.conv.in_proj.weight": (3 * llm_hidden_size, llm_hidden_size),
          f"{p}.conv.out_proj.weight": (llm_hidden_size, llm_hidden_size),
      })
    else:
      main_schema.update({
          f"{p}.self_attn.q_proj.weight": (llm_hidden_size, llm_hidden_size),
          f"{p}.self_attn.k_proj.weight": (llm_kv_dim, llm_hidden_size),
          f"{p}.self_attn.v_proj.weight": (llm_kv_dim, llm_hidden_size),
          f"{p}.self_attn.out_proj.weight": (llm_hidden_size, llm_hidden_size),
          f"{p}.self_attn.q_layernorm.weight": (llm_head_dim,),
          f"{p}.self_attn.k_layernorm.weight": (llm_head_dim,),
      })
  for i in range(df_layers):
    p = f"depthformer.layers.{i}"
    main_schema.update({
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
    main_schema.update({
        f"depth_embeddings.{k}.embedding.weight": (audio_vocab_size, df_dim),
        f"depth_embeddings.{k}.embedding_norm.weight": (df_dim,),
        f"depth_embeddings.{k}.to_logits.weight": (audio_vocab_size, df_dim),
    })

  detok_schema: dict[str, tuple[int, ...]] = {
      "emb.emb.weight": (codebooks * detok_vocab_size, detok_hidden_size),
      "lfm.embed_tokens.weight": (llm_vocab_size, detok_hidden_size),
      "lfm.embedding_norm.weight": (detok_hidden_size,),
      "lin.weight": (detok_n_fft + 2, detok_hidden_size),
      "lin.bias": (detok_n_fft + 2,),
      "istft.window": (detok_n_fft,),
  }
  for i, lt in enumerate(detok_layer_types):
    p = f"lfm.layers.{i}"
    detok_schema.update({
        f"{p}.operator_norm.weight": (detok_hidden_size,),
        f"{p}.ffn_norm.weight": (detok_hidden_size,),
        f"{p}.feed_forward.w1.weight": (detok_ff_dim, detok_hidden_size),
        f"{p}.feed_forward.w2.weight": (detok_hidden_size, detok_ff_dim),
        f"{p}.feed_forward.w3.weight": (detok_ff_dim, detok_hidden_size),
    })
    if lt == "conv":
      detok_schema.update({
          f"{p}.conv.conv.weight": (detok_hidden_size, 1, 3),
          f"{p}.conv.in_proj.weight": (
              3 * detok_hidden_size,
              detok_hidden_size,
          ),
          f"{p}.conv.out_proj.weight": (detok_hidden_size, detok_hidden_size),
      })
    else:
      detok_schema.update({
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

  torch.manual_seed(42)

  def _make_tensor(key: str, shape: tuple[int, ...]) -> torch.Tensor:
    if key.endswith("num_batches_tracked") or key == "codebook_offsets":
      return torch.zeros(shape, dtype=torch.int64)
    if key.endswith("running_var") or key.endswith("norm.weight"):
      return torch.ones(shape, dtype=torch.float32)
    if key.endswith("running_mean") or key.endswith(".bias"):
      return torch.zeros(shape, dtype=torch.float32)
    if key == "istft.window":
      return torch.hann_window(shape[0], dtype=torch.float32)
    return torch.randn(shape, dtype=torch.float32) * 0.005

  main_tensors = {k: _make_tensor(k, s) for k, s in sorted(main_schema.items())}
  # Bias tied LFM2 text embeddings so decode deterministically favors `"Hello"`
  # (token 3) over `<|im_end|>` (7) and `<|text_end|>` (130), guaranteeing both
  # text generation and interleaved transition into speech generation.
  emb = main_tensors["lfm.embed_tokens.weight"]
  emb[:, 0] = 0.25
  emb[3, 0] = 2.5
  emb[7, 0] = -2.5
  emb[130, 0] = -2.5

  # Bias Depthformer conditioning and output heads so codebook sampling
  # strongly prefers code 0 over EOAudio (`audio_vocab_size - 1 = 32`), even
  # with default `tts_temperature > 0`.
  depth_bias = main_tensors["depth_linear.bias"].view(codebooks, df_dim)
  depth_bias[:, 0] = 5.0
  for k in range(codebooks):
    df_w = main_tensors[f"depth_embeddings.{k}.embedding.weight"]
    df_w.zero_()
    df_w[0, 0] = 10.0
    df_w[audio_vocab_size - 1, 0] = -10.0
    main_tensors[f"depth_embeddings.{k}.to_logits.weight"] = (
        df_w.detach().clone()
    )

  safetensors.torch.save_file(
      main_tensors, os.path.join(model_dir, "model.safetensors")
  )

  detok_tensors = {
      k: _make_tensor(k, s) for k, s in sorted(detok_schema.items())
  }
  safetensors.torch.save_file(
      detok_tensors, os.path.join(detok_dir, "model.safetensors")
  )


class ExportAndRunEndToEndTest(absltest.TestCase):
  """Real subprocess tests for `:export_main` and `:lfm2_audio_runner`."""

  def setUp(self):
    super().setUp()
    self.export_main_bin = _runfiles_path(
        "models/lfm/lfm2_5_audio/export/export_main"
    )
    self.runner_bin = _runfiles_path(
        "models/lfm/lfm2_5_audio/lfm2_audio_runner"
    )
    self.assertTrue(
        os.path.exists(self.export_main_bin),
        f"Missing export_main binary at {self.export_main_bin}",
    )
    self.assertTrue(
        os.path.exists(self.runner_bin),
        f"Missing lfm2_audio_runner binary at {self.runner_bin}",
    )

  def test_export_main_cli_dry_run(self):
    """Verifies real `:export_main` CLI entrypoint detects LFM2-Audio."""
    tmpdir = self.create_tempdir().full_path
    ckpt_dir = os.path.join(tmpdir, "ckpt")
    _write_tiny_e2e_checkpoint(ckpt_dir)

    proc = subprocess.run(
        [
            self.export_main_bin,
            f"--model={ckpt_dir}",
            "--dry_run=True",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = f"{proc.stdout}\n{proc.stderr}"
    self.assertEqual(
        proc.returncode,
        0,
        f"export_main --dry_run failed (rc={proc.returncode}):\n{combined}",
    )
    self.assertIn("lfm2_audio", combined)
    self.assertIn("any_to_any", combined)

  def test_export_main_to_cpp_runner_end_to_end(self):
    """Exports real `.litertlm` via `:export_main` and runs `:lfm2_audio_runner`."""
    tmpdir = self.create_tempdir().full_path
    ckpt_dir = os.path.join(tmpdir, "ckpt")
    out_dir = os.path.join(tmpdir, "exported")
    _write_tiny_e2e_checkpoint(ckpt_dir)

    # 1. Run real Python `:export_main` CLI binary to produce `model.litertlm`.
    export_proc = subprocess.run(
        [
            self.export_main_bin,
            f"--model={ckpt_dir}",
            f"--output_dir={out_dir}",
            "--task=auto",
            "--quantization_recipe=none",
            "--use_random_weights=False",
            "--bundle_litert_lm=True",
            "--cache_length=64",
            "--prefill_lengths=[16]",
            "--audio_mel_frames=16",
            "--detok_chunk_frames=4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        export_proc.returncode,
        0,
        f"export_main failed (rc={export_proc.returncode}):\n"
        f"STDOUT:\n{export_proc.stdout}\nSTDERR:\n{export_proc.stderr}",
    )
    litertlm_path = os.path.join(out_dir, "model.litertlm")
    self.assertTrue(os.path.exists(litertlm_path))

    # 2. Text -> Text + Speech via real C++ `:lfm2_audio_runner` binary.
    speech_wav_path = os.path.join(tmpdir, "out_speech.wav")
    tts_proc = subprocess.run(
        [
            self.runner_bin,
            "--model_name=lfm2.5-audio-1.5b",
            f"--model_path={litertlm_path}",
            "--text=Hello",
            "--generate_audio=true",
            "--max_new_tokens=8",
            "--temperature=0.0",
            "--num_threads=1",
            "--cache_dir=:nocache",
            f"--output_path={speech_wav_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        tts_proc.returncode,
        0,
        "lfm2_audio_runner (text->text+speech) failed "
        f"(rc={tts_proc.returncode}):\nSTDOUT:\n{tts_proc.stdout}\n"
        f"STDERR:\n{tts_proc.stderr}",
    )
    self.assertIn("Hello", tts_proc.stdout)
    self.assertIn("@ 24000 Hz", tts_proc.stdout)
    self.assertTrue(os.path.exists(speech_wav_path))
    with wave.open(speech_wav_path, "rb") as wf:
      self.assertEqual(wf.getnchannels(), 1)
      self.assertEqual(wf.getframerate(), 24000)
      self.assertEqual(wf.getsampwidth(), 2)
      self.assertGreater(wf.getnframes(), 0)
      pcm_bytes = wf.readframes(wf.getnframes())
      self.assertTrue(any(b != 0 for b in pcm_bytes))

    # 3. Speech -> Text via real C++ `:lfm2_audio_runner` binary with a real
    #    16 kHz mono 16-bit PCM WAV input.
    input_wav_path = os.path.join(tmpdir, "input_16k.wav")
    _write_test_wav(input_wav_path, sample_rate_hz=16000, num_samples=3200)
    transcript_path = os.path.join(tmpdir, "transcript.txt")
    asr_proc = subprocess.run(
        [
            self.runner_bin,
            "--model_name=lfm2.5-audio-1.5b",
            f"--model_path={litertlm_path}",
            f"--audio_path={input_wav_path}",
            "--generate_audio=false",
            "--max_new_tokens=2",
            "--temperature=0.0",
            "--num_threads=1",
            "--cache_dir=:nocache",
            f"--output_path={transcript_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        asr_proc.returncode,
        0,
        f"lfm2_audio_runner (speech->text) failed (rc={asr_proc.returncode}):\n"
        f"STDOUT:\n{asr_proc.stdout}\nSTDERR:\n{asr_proc.stderr}",
    )
    self.assertTrue(os.path.exists(transcript_path))
    with open(transcript_path, "r") as f:
      self.assertNotEmpty(f.read().strip())

    # 4. Filename-inferred model name (without `--model_name`).
    inferred_path = os.path.join(tmpdir, "lfm2_audio_exported.litertlm")
    shutil.copyfile(litertlm_path, inferred_path)
    infer_proc = subprocess.run(
        [
            self.runner_bin,
            f"--model_path={inferred_path}",
            "--text=Hello",
            "--generate_audio=false",
            "--max_new_tokens=2",
            "--temperature=0.0",
            "--num_threads=1",
            "--cache_dir=:nocache",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        infer_proc.returncode,
        0,
        "lfm2_audio_runner (filename inference) failed "
        f"(rc={infer_proc.returncode}):\nSTDOUT:\n{infer_proc.stdout}\n"
        f"STDERR:\n{infer_proc.stderr}",
    )
    self.assertIn("Hello", infer_proc.stdout)


if __name__ == "__main__":
  absltest.main()
