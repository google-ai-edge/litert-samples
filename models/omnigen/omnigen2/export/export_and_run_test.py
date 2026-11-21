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
"""Cross-language CLI and exporter-to-C++-runner E2E tests for OmniGen2."""

import json
import os
import shutil
import struct
import subprocess

from absl.testing import absltest
import safetensors.torch
import torch

from models.omnigen.omnigen2.export import modeling


def _runfiles_path(rel_path: str) -> str:
  """Resolves a workspace-relative data dependency path inside TEST_SRCDIR."""
  srcdir = os.environ.get("TEST_SRCDIR") or absltest.get_default_test_srcdir()
  return os.path.join(srcdir, os.environ.get("TEST_WORKSPACE", "litert"), rel_path)


def _parse_png_header(png_path: str) -> tuple[int, int, int, int]:
  """Validates PNG signature + IHDR chunk and returns (width, height, bit_depth, color_type)."""
  with open(png_path, "rb") as f:
    data = f.read()
  if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
    raise AssertionError(f"Invalid PNG signature in {png_path}")
  ihdr_len = struct.unpack(">I", data[8:12])[0]
  if ihdr_len < 13 or data[12:16] != b"IHDR":
    raise AssertionError(f"Missing IHDR chunk in {png_path}")
  width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
  return width, height, bit_depth, color_type


def _write_tiny_e2e_checkpoint(model_dir: str) -> None:
  """Writes a tiny deterministic OmniGen2 checkpoint using OmniGen2's `processor/` layout."""
  mllm_dir = os.path.join(model_dir, "mllm")
  dit_dir = os.path.join(model_dir, "transformer")
  vae_dir = os.path.join(model_dir, "vae")
  sched_dir = os.path.join(model_dir, "scheduler")
  proc_dir = os.path.join(model_dir, "processor")
  for d in (mllm_dir, dit_dir, vae_dir, sched_dir, proc_dir):
    os.makedirs(d, exist_ok=True)

  with open(os.path.join(model_dir, "model_index.json"), "w") as f:
    json.dump({"_class_name": "OmniGen2Pipeline"}, f)

  # Build a compact WordLevel tokenizer inside `processor/` containing all
  # OmniGen2 ChatML control tokens (`<|endoftext|>`, `<|im_start|>`,
  # `<|im_end|>`, `\n`) plus system/user prompt words so C++ dynamic framing
  # keeps every token ID in `[0, vocab_size)`.
  vocab_size = 64
  vocab: dict[str, int] = {
      "<|endoftext|>": 0,
      "<|im_start|>": 1,
      "<|im_end|>": 2,
      "\n": 3,
      "system": 4,
      "user": 5,
      "You": 6,
      "are": 7,
      "a": 8,
      "helpful": 9,
      "assistant": 10,
      "that": 11,
      "generates": 12,
      "high-quality": 13,
      "images": 14,
      "based": 15,
      "on": 16,
      "instructions.": 17,
      "robot": 18,
      "painting": 19,
      "canvas": 20,
      "cat": 21,
  }
  for idx in range(len(vocab), vocab_size):
    vocab[f"tok_{idx}"] = idx

  with open(os.path.join(proc_dir, "tokenizer_config.json"), "w") as f:
    json.dump({"tokenizer_class": "PreTrainedTokenizerFast"}, f)
  with open(os.path.join(proc_dir, "tokenizer.json"), "w") as f:
    json.dump(
        {
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
                for tok, tid in (
                    ("<|endoftext|>", 0),
                    ("<|im_start|>", 1),
                    ("<|im_end|>", 2),
                    ("\n", 3),
                )
            ],
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "<|endoftext|>",
            },
        },
        f,
    )

  mllm_cfg = {
      "model_type": "qwen2_5_vl",
      "vocab_size": vocab_size,
      "hidden_size": 32,
      "intermediate_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "rms_norm_eps": 1e-6,
      "rope_theta": 1000000.0,
  }
  with open(os.path.join(mllm_dir, "config.json"), "w") as f:
    json.dump(mllm_cfg, f)

  dit_cfg = {
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
  }
  with open(os.path.join(dit_dir, "config.json"), "w") as f:
    json.dump(dit_cfg, f)

  vae_cfg = {
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
  }
  with open(os.path.join(vae_dir, "config.json"), "w") as f:
    json.dump(vae_cfg, f)

  with open(os.path.join(sched_dir, "scheduler_config.json"), "w") as f:
    json.dump(
        {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "dynamic_time_shift": True,
            "num_train_timesteps": 1000,
        },
        f,
    )

  torch.manual_seed(42)

  def _init_small(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for k, v in module.state_dict().items():
      if k.endswith("norm.weight") or "norm" in k and k.endswith(".weight"):
        sd[k] = torch.ones_like(v, dtype=torch.float32)
      elif k.endswith(".bias"):
        sd[k] = torch.zeros_like(v, dtype=torch.float32)
      else:
        sd[k] = torch.randn_like(v, dtype=torch.float32) * 0.02
    return sd

  te = modeling.OmniGen2TextEncoder(
      vocab_size=vocab_size,
      hidden_size=32,
      intermediate_size=64,
      num_hidden_layers=2,
      num_attention_heads=2,
      num_key_value_heads=1,
  )
  te_sd = _init_small(te)
  # Include a frozen Qwen2.5-VL visual key to verify real checkpoint filtering.
  te_sd["visual.blocks.0.attn.proj.weight"] = torch.zeros(4, 4)
  safetensors.torch.save_file(
      te_sd, os.path.join(mllm_dir, "model.safetensors")
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
  safetensors.torch.save_file(
      _init_small(dit),
      os.path.join(dit_dir, "diffusion_pytorch_model.safetensors"),
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
  safetensors.torch.save_file(
      _init_small(vae),
      os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"),
  )


class ExportAndRunEndToEndTest(absltest.TestCase):
  """Real subprocess tests for `:export_main` and `:omnigen2_runner`."""

  def setUp(self):
    super().setUp()
    self.export_main_bin = _runfiles_path(
        "models/omnigen/omnigen2/export/export_main"
    )
    self.runner_bin = _runfiles_path(
        "models/omnigen/omnigen2/omnigen2_runner"
    )
    self.assertTrue(
        os.path.exists(self.export_main_bin),
        f"Missing export_main binary at {self.export_main_bin}",
    )
    self.assertTrue(
        os.path.exists(self.runner_bin),
        f"Missing omnigen2_runner binary at {self.runner_bin}",
    )

  def test_export_main_cli_dry_run(self):
    """Verifies real `:export_main` CLI entrypoint auto-detects OmniGen2."""
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
    self.assertIn("OmniGen2Pipeline", combined)
    self.assertIn("text_to_image", combined)

  def test_export_main_to_cpp_runner_end_to_end(self):
    """Exports real `.litertlm` via `:export_main` and runs `:omnigen2_runner`."""
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
            "--t2i_output_image_size=32",
            "--seq_len=32",
            "--steps=2",
            "--text_guidance_scale=2.0",
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

    # 2. Run real C++ `:omnigen2_runner` binary with explicit `--model_name`.
    explicit_png = os.path.join(tmpdir, "out_explicit.png")
    explicit_proc = subprocess.run(
        [
            self.runner_bin,
            "--model_name=omnigen2",
            f"--model_path={litertlm_path}",
            "--text=a robot painting a canvas",
            "--steps=2",
            "--seed=42",
            "--num_threads=1",
            "--cache_dir=:nocache",
            f"--output_path={explicit_png}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        explicit_proc.returncode,
        0,
        f"omnigen2_runner (explicit) failed (rc={explicit_proc.returncode}):\n"
        f"STDOUT:\n{explicit_proc.stdout}\nSTDERR:\n{explicit_proc.stderr}",
    )
    self.assertIn("Saved 32x32 image", explicit_proc.stdout)
    self.assertTrue(os.path.exists(explicit_png))
    self.assertGreater(os.path.getsize(explicit_png), 0)
    self.assertEqual(_parse_png_header(explicit_png), (32, 32, 8, 2))

    # 3. Filename-inferred model name without `--model_name`.
    inferred_bundle = os.path.join(tmpdir, "omnigen2_exported.litertlm")
    shutil.copyfile(litertlm_path, inferred_bundle)
    inferred_png = os.path.join(tmpdir, "out_inferred.png")
    infer_proc = subprocess.run(
        [
            self.runner_bin,
            f"--model_path={inferred_bundle}",
            "--text=a cat",
            "--steps=2",
            "--num_threads=1",
            "--cache_dir=:nocache",
            f"--output_path={inferred_png}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        infer_proc.returncode,
        0,
        f"omnigen2_runner (inferred) failed (rc={infer_proc.returncode}):\n"
        f"STDOUT:\n{infer_proc.stdout}\nSTDERR:\n{infer_proc.stderr}",
    )
    self.assertIn("Saved 32x32 image", infer_proc.stdout)
    self.assertTrue(os.path.exists(inferred_png))
    self.assertEqual(_parse_png_header(inferred_png), (32, 32, 8, 2))


if __name__ == "__main__":
  absltest.main()
