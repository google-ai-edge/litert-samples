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
"""Cross-language CLI and exporter-to-C++-runner E2E tests for Janus-Pro."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import zlib

from absl.testing import absltest
import safetensors.torch
import torch

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _runfiles_path(rel_path: str) -> str:
  """Resolves a workspace-relative data dependency path inside TEST_SRCDIR."""
  srcdir = os.environ.get("TEST_SRCDIR") or absltest.get_default_test_srcdir()
  return os.path.join(srcdir, os.environ.get("TEST_WORKSPACE", "litert"), rel_path)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
  crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
  return (
      struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)
  )


def _write_test_png(
    path: str, width: int, height: int, rgb: tuple[int, int, int]
) -> None:
  """Writes a valid 8-bit RGB PNG file filled with `rgb`."""
  ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
  row = bytes(rgb) * width
  raw = b"".join(b"\x00" + row for _ in range(height))
  png_bytes = (
      _PNG_SIGNATURE
      + _png_chunk(b"IHDR", ihdr)
      + _png_chunk(b"IDAT", zlib.compress(raw))
      + _png_chunk(b"IEND", b"")
  )
  with open(path, "wb") as f:
    f.write(png_bytes)


def _parse_rgb_png(path: str) -> tuple[int, int, bytes]:
  """Validates a PNG file and returns `(width, height, raw_rgb_bytes)`."""
  with open(path, "rb") as f:
    data = f.read()
  if not data.startswith(_PNG_SIGNATURE):
    raise ValueError(f"Invalid PNG signature in {path}")
  offset = len(_PNG_SIGNATURE)
  width = 0
  height = 0
  bit_depth = 0
  color_type = 0
  idat_parts: list[bytes] = []
  saw_iend = False
  while offset + 12 <= len(data):
    length = struct.unpack(">I", data[offset : offset + 4])[0]
    chunk_type = data[offset + 4 : offset + 8]
    payload_start = offset + 8
    payload_end = payload_start + length
    if payload_end + 4 > len(data):
      raise ValueError("Truncated PNG chunk.")
    payload = data[payload_start:payload_end]
    offset = payload_end + 4
    if chunk_type == b"IHDR":
      width, height, bit_depth, color_type = struct.unpack(
          ">IIBB", payload[:10]
      )
    elif chunk_type == b"IDAT":
      idat_parts.append(payload)
    elif chunk_type == b"IEND":
      saw_iend = True
      break
  if not saw_iend or width <= 0 or height <= 0:
    raise ValueError(f"Incomplete PNG file: {path}")
  if bit_depth != 8 or color_type != 2:
    raise ValueError(
        "Expected 8-bit RGB PNG (bit_depth=8, color_type=2), got"
        f" bit_depth={bit_depth}, color_type={color_type}"
    )
  decompressed = zlib.decompress(b"".join(idat_parts))
  stride = 1 + width * 3
  if len(decompressed) != height * stride:
    raise ValueError(
        f"Unexpected PNG IDAT size: got {len(decompressed)}, expected"
        f" {height * stride}"
    )
  return width, height, decompressed


def _write_tiny_e2e_checkpoint(model_dir: str) -> None:
  """Writes a tiny deterministic Janus-Pro safetensors checkpoint & tokenizer."""
  os.makedirs(model_dir, exist_ok=True)

  vocab_size = 64
  llm_hidden = 32
  image_size = 32
  patch_size = 16
  vq_codebook_size = 64
  vq_embed_dim = 8

  config_dict = {
      "model_type": "multi_modality",
      "architectures": ["MultiModalityCausalLM"],
      "vision_config": {
          "cls": "CLIPVisionTower",
          "params": {
              "image_size": image_size,
              "patch_size": patch_size,
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
              "n_embed": llm_hidden,
              "projector_type": "mlp_gelu",
          },
      },
      "gen_vision_config": {
          "cls": "VQ-16",
          "params": {
              "image_token_size": vq_codebook_size,
              "n_embed": vq_embed_dim,
              "z_channels": 8,
              "base_channels": 32,
              "num_res_blocks": 1,
          },
      },
      "gen_aligner_config": {
          "cls": "MlpProjector",
          "params": {
              "depth": 2,
              "input_dim": vq_embed_dim,
              "n_embed": llm_hidden,
              "projector_type": "mlp_gelu",
          },
      },
      "gen_head_config": {
          "cls": "vision_head",
          "params": {
              "image_token_embed": llm_hidden,
              "image_token_size": vq_codebook_size,
              "n_embed": llm_hidden,
          },
      },
      "language_config": {
          "model_type": "llama",
          "hidden_size": llm_hidden,
          "intermediate_size": 64,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "vocab_size": vocab_size,
          "max_position_embeddings": 128,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000.0,
          "bos_token_id": 1,
          "eos_token_id": 2,
          "tie_word_embeddings": False,
      },
  }
  with open(os.path.join(model_dir, "config.json"), "w") as f:
    json.dump(config_dict, f)

  # Include all 8 Janus-Pro control tokens at IDs < vocab_size (64) so real
  # TFLite Gather ops never receive out-of-bounds fallback IDs.
  special_tokens = [
      ("<｜▁pad▁｜>", 0),
      ("<｜begin▁of▁sentence｜>", 1),
      ("<｜end▁of▁sentence｜>", 2),
      ("<begin_of_image>", 5),
      ("<end_of_image>", 6),
      ("<image_placeholder>", 7),
      ("<|User|>", 8),
      ("<|Assistant|>", 9),
  ]
  vocab: dict[str, int] = {
      "<｜▁pad▁｜>": 0,
      "<｜begin▁of▁sentence｜>": 1,
      "<｜end▁of▁sentence｜>": 2,
      "Hello": 3,
      "world": 4,
      "<begin_of_image>": 5,
      "<end_of_image>": 6,
      "<image_placeholder>": 7,
      "<|User|>": 8,
      "<|Assistant|>": 9,
      "\n": 10,
      ":": 11,
      "Describe": 12,
      "fox": 13,
  }
  used_ids = set(vocab.values())
  for token_id in range(vocab_size):
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
          for tok, tid in special_tokens
      ],
      "normalizer": None,
      "pre_tokenizer": {"type": "Whitespace"},
      "post_processor": None,
      "decoder": None,
      "model": {
          "type": "WordLevel",
          "vocab": vocab,
          "unk_token": "<｜▁pad▁｜>",
      },
  }
  with open(os.path.join(model_dir, "tokenizer.json"), "w") as f:
    json.dump(tokenizer_json, f)

  torch.manual_seed(42)
  vt = "vision_model.vision_tower"
  tensors: dict[str, torch.Tensor] = {
      f"{vt}.patch_embed.proj.weight": torch.randn(32, 3, 16, 16) * 0.01,
      f"{vt}.patch_embed.proj.bias": torch.zeros(32),
      f"{vt}.pos_embed": torch.randn(1, 4, 32) * 0.01,
      f"{vt}.blocks.0.norm1.weight": torch.ones(32),
      f"{vt}.blocks.0.norm1.bias": torch.zeros(32),
      f"{vt}.blocks.0.attn.qkv.weight": torch.randn(96, 32) * 0.01,
      f"{vt}.blocks.0.attn.qkv.bias": torch.zeros(96),
      f"{vt}.blocks.0.attn.proj.weight": torch.randn(32, 32) * 0.01,
      f"{vt}.blocks.0.attn.proj.bias": torch.zeros(32),
      f"{vt}.blocks.0.norm2.weight": torch.ones(32),
      f"{vt}.blocks.0.norm2.bias": torch.zeros(32),
      f"{vt}.blocks.0.mlp.fc1.weight": torch.randn(64, 32) * 0.01,
      f"{vt}.blocks.0.mlp.fc1.bias": torch.zeros(64),
      f"{vt}.blocks.0.mlp.fc2.weight": torch.randn(32, 64) * 0.01,
      f"{vt}.blocks.0.mlp.fc2.bias": torch.zeros(32),
      f"{vt}.norm.weight": torch.ones(32),
      f"{vt}.norm.bias": torch.zeros(32),
      "aligner.layers.0.weight": torch.randn(32, 32) * 0.01,
      "aligner.layers.0.bias": torch.zeros(32),
      "aligner.layers.2.weight": torch.randn(32, 32) * 0.01,
      "aligner.layers.2.bias": torch.zeros(32),
      "gen_embed.weight": torch.randn(vq_codebook_size, vq_embed_dim) * 0.01,
      "gen_aligner.layers.0.weight": torch.randn(32, vq_embed_dim) * 0.01,
      "gen_aligner.layers.0.bias": torch.zeros(32),
      "gen_aligner.layers.2.weight": torch.randn(32, 32) * 0.01,
      "gen_aligner.layers.2.bias": torch.zeros(32),
      "gen_head.output_mlp_projector.weight": torch.randn(32, 32) * 0.01,
      "gen_head.output_mlp_projector.bias": torch.zeros(32),
      "gen_head.vision_head.weight": torch.randn(vq_codebook_size, 32) * 0.01,
      "gen_head.vision_head.bias": torch.full((vq_codebook_size,), -5.0),
      "language_model.model.embed_tokens.weight": (
          torch.randn(vocab_size, llm_hidden) * 0.01
      ),
      "language_model.model.layers.0.input_layernorm.weight": torch.ones(
          llm_hidden
      ),
      "language_model.model.layers.0.self_attn.q_proj.weight": (
          torch.randn(llm_hidden, llm_hidden) * 0.005
      ),
      "language_model.model.layers.0.self_attn.k_proj.weight": (
          torch.randn(llm_hidden, llm_hidden) * 0.005
      ),
      "language_model.model.layers.0.self_attn.v_proj.weight": (
          torch.randn(llm_hidden, llm_hidden) * 0.005
      ),
      "language_model.model.layers.0.self_attn.o_proj.weight": (
          torch.randn(llm_hidden, llm_hidden) * 0.005
      ),
      "language_model.model.layers.0.post_attention_layernorm.weight": (
          torch.ones(llm_hidden)
      ),
      "language_model.model.layers.0.mlp.gate_proj.weight": (
          torch.randn(64, llm_hidden) * 0.005
      ),
      "language_model.model.layers.0.mlp.up_proj.weight": (
          torch.randn(64, llm_hidden) * 0.005
      ),
      "language_model.model.layers.0.mlp.down_proj.weight": (
          torch.randn(llm_hidden, 64) * 0.005
      ),
      "language_model.model.norm.weight": torch.ones(llm_hidden),
      "language_model.lm_head.weight": (
          torch.randn(vocab_size, llm_hidden) * 0.01
      ),
  }

  # Bias text embeddings & lm_head so greedy decode deterministically predicts
  # token 3 ("Hello") rather than EOS (2).
  tensors["language_model.model.embed_tokens.weight"][:, 0] = 1.0
  tensors["language_model.lm_head.weight"][3, 0] = 5.0
  tensors["language_model.lm_head.weight"][2, 0] = -5.0

  # Bias gen_head so VQ codebook index 7 is deterministically selected.
  tensors["gen_head.vision_head.bias"][7] = 10.0

  gv = "gen_vision_model"
  base_ch = 32
  top_ch = base_ch * 4
  tensors[f"{gv}.quantize.embedding.weight"] = torch.randn(64, 8) * 0.01
  tensors[f"{gv}.post_quant_conv.weight"] = torch.randn(8, 8, 1, 1) * 0.01
  tensors[f"{gv}.post_quant_conv.bias"] = torch.zeros(8)
  tensors[f"{gv}.decoder.conv_in.weight"] = torch.randn(top_ch, 8, 3, 3) * 0.01
  tensors[f"{gv}.decoder.conv_in.bias"] = torch.zeros(top_ch)

  def add_res(prefix: str, in_ch: int, out_ch: int) -> None:
    tensors[f"{prefix}.norm1.weight"] = torch.ones(in_ch)
    tensors[f"{prefix}.norm1.bias"] = torch.zeros(in_ch)
    tensors[f"{prefix}.conv1.weight"] = torch.randn(out_ch, in_ch, 3, 3) * 0.005
    tensors[f"{prefix}.conv1.bias"] = torch.zeros(out_ch)
    tensors[f"{prefix}.norm2.weight"] = torch.ones(out_ch)
    tensors[f"{prefix}.norm2.bias"] = torch.zeros(out_ch)
    tensors[f"{prefix}.conv2.weight"] = (
        torch.randn(out_ch, out_ch, 3, 3) * 0.005
    )
    tensors[f"{prefix}.conv2.bias"] = torch.zeros(out_ch)
    if in_ch != out_ch:
      tensors[f"{prefix}.nin_shortcut.weight"] = (
          torch.randn(out_ch, in_ch, 1, 1) * 0.005
      )
      tensors[f"{prefix}.nin_shortcut.bias"] = torch.zeros(out_ch)

  def add_attn(prefix: str, ch: int) -> None:
    tensors[f"{prefix}.norm.weight"] = torch.ones(ch)
    tensors[f"{prefix}.norm.bias"] = torch.zeros(ch)
    for proj in ("q", "k", "v", "proj_out"):
      tensors[f"{prefix}.{proj}.weight"] = torch.randn(ch, ch, 1, 1) * 0.005
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
          torch.randn(block_out, block_out, 3, 3) * 0.005
      )
      tensors[f"{lp}.upsample.conv.bias"] = torch.zeros(block_out)
    block_in = block_out

  tensors[f"{gv}.decoder.norm_out.weight"] = torch.ones(base_ch)
  tensors[f"{gv}.decoder.norm_out.bias"] = torch.zeros(base_ch)
  tensors[f"{gv}.decoder.conv_out.weight"] = (
      torch.randn(3, base_ch, 3, 3) * 0.001
  )
  # Emit strong non-black RGB bias: R ~ 1.0 (255), G ~ 0.0 (127), B ~ -1.0 (0).
  tensors[f"{gv}.decoder.conv_out.bias"] = torch.tensor(
      [1.0, 0.0, -1.0], dtype=torch.float32
  )

  safetensors.torch.save_file(
      tensors, os.path.join(model_dir, "model.safetensors")
  )


class ExportAndRunEndToEndTest(absltest.TestCase):
  """Real subprocess tests for `:export_main` and `:janus_pro_runner`."""

  def setUp(self):
    super().setUp()
    self.export_main_bin = _runfiles_path(
        "models/deepseek/janus_pro/export/export_main"
    )
    self.runner_bin = _runfiles_path(
        "models/deepseek/janus_pro/janus_pro_runner"
    )
    self.assertTrue(
        os.path.exists(self.export_main_bin),
        f"Missing export_main binary at {self.export_main_bin}",
    )
    self.assertTrue(
        os.path.exists(self.runner_bin),
        f"Missing janus_pro_runner binary at {self.runner_bin}",
    )

  def test_export_main_cli_dry_run(self):
    """Verifies real `:export_main` CLI entrypoint detects Janus-Pro."""
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
    self.assertIn("multi_modality", combined)
    self.assertIn("any_to_any", combined)

  def test_export_main_to_cpp_runner_end_to_end(self):
    """Exports real `.litertlm` via `:export_main` and runs `:janus_pro_runner`."""
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
            "--cache_length=128",
            "--prefill_lengths=[64]",
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

    # 2. Image + Text -> Text understanding via real C++ `:janus_pro_runner`.
    input_png_path = os.path.join(tmpdir, "input.png")
    _write_test_png(input_png_path, width=16, height=16, rgb=(180, 120, 60))
    transcript_path = os.path.join(tmpdir, "understanding_out.txt")
    und_proc = subprocess.run(
        [
            self.runner_bin,
            "--model_name=janus-pro-1b",
            f"--model_path={litertlm_path}",
            f"--image_path={input_png_path}",
            "--text=Describe",
            "--generate_image=false",
            "--max_new_tokens=4",
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
        und_proc.returncode,
        0,
        "janus_pro_runner (understanding) failed "
        f"(rc={und_proc.returncode}):\nSTDOUT:\n{und_proc.stdout}\n"
        f"STDERR:\n{und_proc.stderr}",
    )
    self.assertIn("Hello", und_proc.stdout)
    self.assertTrue(os.path.exists(transcript_path))
    with open(transcript_path, "r") as f:
      self.assertIn("Hello", f.read())

    # 3. Text -> Image generation (`--generate_image=true`) writing a valid PNG.
    generated_png_path = os.path.join(tmpdir, "generated.png")
    gen_proc = subprocess.run(
        [
            self.runner_bin,
            "--model_name=janus-pro-1b",
            f"--model_path={litertlm_path}",
            "--text=fox",
            "--generate_image=true",
            "--cfg_weight=5.0",
            "--image_temperature=0.0",
            "--seed=42",
            "--num_threads=1",
            "--cache_dir=:nocache",
            f"--output_path={generated_png_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        gen_proc.returncode,
        0,
        "janus_pro_runner (visual generation) failed "
        f"(rc={gen_proc.returncode}):\nSTDOUT:\n{gen_proc.stdout}\n"
        f"STDERR:\n{gen_proc.stderr}",
    )
    self.assertIn("Saved 32x32 image to", gen_proc.stdout)
    self.assertTrue(os.path.exists(generated_png_path))
    width, height, raw_scanlines = _parse_rgb_png(generated_png_path)
    self.assertEqual(width, 32)
    self.assertEqual(height, 32)
    self.assertTrue(any(b != 0 for b in raw_scanlines))

    # 4. Filename-inferred model name (omitting `--model_name`).
    inferred_path = os.path.join(tmpdir, "janus_pro_exported.litertlm")
    shutil.copyfile(litertlm_path, inferred_path)
    infer_proc = subprocess.run(
        [
            self.runner_bin,
            f"--model_path={inferred_path}",
            "--text=Describe",
            "--generate_image=false",
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
        "janus_pro_runner (filename inference) failed "
        f"(rc={infer_proc.returncode}):\nSTDOUT:\n{infer_proc.stdout}\n"
        f"STDERR:\n{infer_proc.stderr}",
    )
    self.assertIn("Hello", infer_proc.stdout)


if __name__ == "__main__":
  absltest.main()
