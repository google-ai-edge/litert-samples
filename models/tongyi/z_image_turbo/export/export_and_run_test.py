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
"""Cross-language CLI and exporter-to-C++-runner E2E tests for Z-Image-Turbo."""

import json
import os
import shutil
import struct
import subprocess
from typing import Any

import safetensors.torch
import torch

from absl.testing import absltest as googletest


def _parse_png_ihdr(path: str) -> tuple[int, int, int, int, bytes]:
  """Parses a PNG file and returns `(width, height, bit_depth, color_type, raw)`."""
  with open(path, 'rb') as f:
    raw = f.read()
  if len(raw) < 33 or raw[:8] != b'\x89PNG\r\n\x1a\n':
    raise ValueError(f'Invalid PNG signature in {path}')
  ihdr_len = struct.unpack('>I', raw[8:12])[0]
  if ihdr_len != 13 or raw[12:16] != b'IHDR':
    raise ValueError(f'Missing valid IHDR chunk in {path}')
  width, height, bit_depth, color_type = struct.unpack('>IIBB', raw[16:26])
  return width, height, bit_depth, color_type, raw


def _make_tensor(key: str, shape: tuple[int, ...]) -> torch.Tensor:
  if (
      key.endswith('norm.weight')
      or key.endswith('norm1.weight')
      or (key.endswith('norm2.weight'))
      or key.endswith('norm_q.weight')
      or key.endswith('norm_k.weight')
      or (key.endswith('norm_out.weight'))
      or key.endswith('cap_embedder.0.weight')
  ):
    return torch.ones(shape, dtype=torch.float32)
  if key.endswith('.bias'):
    return torch.zeros(shape, dtype=torch.float32)
  return torch.randn(shape, dtype=torch.float32) * 0.02


def _add_dit_block_schema(
    schema: dict[str, tuple[int, ...]],
    prefix: str,
    dim: int,
    head_dim: int,
    ff_hidden: int,
    adaln_dim: int,
    modulation: bool,
) -> None:
  schema.update({
      f'{prefix}.attention.to_q.weight': (dim, dim),
      f'{prefix}.attention.to_k.weight': (dim, dim),
      f'{prefix}.attention.to_v.weight': (dim, dim),
      f'{prefix}.attention.to_out.0.weight': (dim, dim),
      f'{prefix}.attention.norm_q.weight': (head_dim,),
      f'{prefix}.attention.norm_k.weight': (head_dim,),
      f'{prefix}.feed_forward.w1.weight': (ff_hidden, dim),
      f'{prefix}.feed_forward.w2.weight': (dim, ff_hidden),
      f'{prefix}.feed_forward.w3.weight': (ff_hidden, dim),
      f'{prefix}.attention_norm1.weight': (dim,),
      f'{prefix}.attention_norm2.weight': (dim,),
      f'{prefix}.ffn_norm1.weight': (dim,),
      f'{prefix}.ffn_norm2.weight': (dim,),
  })
  if modulation:
    schema.update({
        f'{prefix}.adaLN_modulation.0.weight': (4 * dim, adaln_dim),
        f'{prefix}.adaLN_modulation.0.bias': (4 * dim,),
    })


def _add_vae_resnet_schema(
    schema: dict[str, tuple[int, ...]],
    prefix: str,
    in_ch: int,
    out_ch: int,
) -> None:
  schema.update({
      f'{prefix}.norm1.weight': (in_ch,),
      f'{prefix}.norm1.bias': (in_ch,),
      f'{prefix}.conv1.weight': (out_ch, in_ch, 3, 3),
      f'{prefix}.conv1.bias': (out_ch,),
      f'{prefix}.norm2.weight': (out_ch,),
      f'{prefix}.norm2.bias': (out_ch,),
      f'{prefix}.conv2.weight': (out_ch, out_ch, 3, 3),
      f'{prefix}.conv2.bias': (out_ch,),
  })
  if in_ch != out_ch:
    schema.update({
        f'{prefix}.conv_shortcut.weight': (out_ch, in_ch, 1, 1),
        f'{prefix}.conv_shortcut.bias': (out_ch,),
    })


def _add_vae_mid_block_schema(
    schema: dict[str, tuple[int, ...]], prefix: str, ch: int
) -> None:
  _add_vae_resnet_schema(schema, f'{prefix}.resnets.0', ch, ch)
  _add_vae_resnet_schema(schema, f'{prefix}.resnets.1', ch, ch)
  schema.update({
      f'{prefix}.attentions.0.group_norm.weight': (ch,),
      f'{prefix}.attentions.0.group_norm.bias': (ch,),
      f'{prefix}.attentions.0.to_q.weight': (ch, ch),
      f'{prefix}.attentions.0.to_q.bias': (ch,),
      f'{prefix}.attentions.0.to_k.weight': (ch, ch),
      f'{prefix}.attentions.0.to_k.bias': (ch,),
      f'{prefix}.attentions.0.to_v.weight': (ch, ch),
      f'{prefix}.attentions.0.to_v.bias': (ch,),
      f'{prefix}.attentions.0.to_out.0.weight': (ch, ch),
      f'{prefix}.attentions.0.to_out.0.bias': (ch,),
  })


def _write_tiny_e2e_checkpoint(model_dir: str) -> None:
  """Writes a tiny deterministic Z-Image-Turbo safetensors checkpoint."""
  for sub in ('tokenizer', 'text_encoder', 'transformer', 'vae', 'scheduler'):
    os.makedirs(os.path.join(model_dir, sub), exist_ok=True)

  with open(os.path.join(model_dir, 'model_index.json'), 'w') as f:
    json.dump({'_class_name': 'ZImagePipeline'}, f)
  with open(
      os.path.join(model_dir, 'scheduler', 'scheduler_config.json'), 'w'
  ) as f:
    json.dump({'shift': 3.0, 'use_dynamic_shifting': False}, f)

  vocab_size = 64
  vocab: dict[str, int] = {
      '<|endoftext|>': 0,
      '<|im_start|>': 1,
      '<|im_end|>': 2,
      '\n': 3,
      'assistant': 4,
      'user': 5,
      'a': 6,
      'red': 7,
      'fox': 8,
      'in': 9,
      'snow': 10,
      'cat': 11,
  }
  for tid in range(vocab_size):
    if tid not in vocab.values():
      vocab[f'tok_{tid}'] = tid

  with open(
      os.path.join(model_dir, 'tokenizer', 'tokenizer_config.json'), 'w'
  ) as f:
    json.dump({'tokenizer_class': 'PreTrainedTokenizerFast'}, f)
  with open(os.path.join(model_dir, 'tokenizer', 'tokenizer.json'), 'w') as f:
    json.dump(
        {
            'version': '1.0',
            'truncation': None,
            'padding': None,
            'added_tokens': [
                {
                    'id': tid,
                    'content': tok,
                    'single_word': False,
                    'lstrip': False,
                    'rstrip': False,
                    'normalized': False,
                    'special': True,
                }
                for tok, tid in (
                    ('<|endoftext|>', 0),
                    ('<|im_start|>', 1),
                    ('<|im_end|>', 2),
                    ('\n', 3),
                )
            ],
            'normalizer': None,
            'pre_tokenizer': {'type': 'Whitespace'},
            'post_processor': None,
            'decoder': None,
            'model': {
                'type': 'WordLevel',
                'vocab': vocab,
                'unk_token': '<|endoftext|>',
            },
        },
        f,
    )

  torch.manual_seed(42)

  # 1. Tiny Qwen3 text encoder
  te_hidden = 32
  te_inter = 64
  te_layers = 3
  te_heads = 4
  te_kv_heads = 2
  te_head_dim = 8
  qwen_cfg_dict: dict[str, Any] = {
      'model_type': 'qwen3',
      'vocab_size': vocab_size,
      'hidden_size': te_hidden,
      'intermediate_size': te_inter,
      'num_hidden_layers': te_layers,
      'num_attention_heads': te_heads,
      'num_key_value_heads': te_kv_heads,
      'head_dim': te_head_dim,
      'tie_word_embeddings': False,
  }
  with open(os.path.join(model_dir, 'text_encoder', 'config.json'), 'w') as f:
    json.dump(qwen_cfg_dict, f)
  te_schema: dict[str, tuple[int, ...]] = {
      'model.embed_tokens.weight': (vocab_size, te_hidden),
      'model.norm.weight': (te_hidden,),
      'lm_head.weight': (vocab_size, te_hidden),
  }
  for i in range(te_layers):
    p = f'model.layers.{i}'
    te_schema.update({
        f'{p}.input_layernorm.weight': (te_hidden,),
        f'{p}.post_attention_layernorm.weight': (te_hidden,),
        f'{p}.self_attn.q_proj.weight': (te_heads * te_head_dim, te_hidden),
        f'{p}.self_attn.k_proj.weight': (te_kv_heads * te_head_dim, te_hidden),
        f'{p}.self_attn.v_proj.weight': (te_kv_heads * te_head_dim, te_hidden),
        f'{p}.self_attn.o_proj.weight': (te_hidden, te_heads * te_head_dim),
        f'{p}.self_attn.q_norm.weight': (te_head_dim,),
        f'{p}.self_attn.k_norm.weight': (te_head_dim,),
        f'{p}.mlp.gate_proj.weight': (te_inter, te_hidden),
        f'{p}.mlp.up_proj.weight': (te_inter, te_hidden),
        f'{p}.mlp.down_proj.weight': (te_hidden, te_inter),
    })
  safetensors.torch.save_file(
      {k: _make_tensor(k, s) for k, s in sorted(te_schema.items())},
      os.path.join(model_dir, 'text_encoder', 'model.safetensors'),
  )

  # 2. Tiny Z-Image S3-DiT transformer
  dit_dim = 24
  dit_heads = 2
  dit_head_dim = dit_dim // dit_heads
  dit_ff_hidden = int(dit_dim / 3 * 8)
  dit_in_ch = 4
  dit_packed_ch = 2 * 2 * dit_in_ch
  dit_adaln_dim = min(dit_dim, 256)
  dit_layers = 6
  dit_cfg_dict: dict[str, Any] = {
      'all_patch_size': [2],
      'all_f_patch_size': [1],
      'in_channels': dit_in_ch,
      'dim': dit_dim,
      'n_layers': dit_layers,
      'n_refiner_layers': 1,
      'n_heads': dit_heads,
      'n_kv_heads': dit_heads,
      'norm_eps': 1e-5,
      'qk_norm': True,
      'cap_feat_dim': te_hidden,
      'rope_theta': 256.0,
      't_scale': 1000.0,
      'axes_dims': [4, 4, 4],
      'axes_lens': [64, 32, 32],
  }
  with open(os.path.join(model_dir, 'transformer', 'config.json'), 'w') as f:
    json.dump(dit_cfg_dict, f)
  dit_schema: dict[str, tuple[int, ...]] = {
      't_embedder.mlp.0.weight': (1024, 256),
      't_embedder.mlp.0.bias': (1024,),
      't_embedder.mlp.2.weight': (dit_adaln_dim, 1024),
      't_embedder.mlp.2.bias': (dit_adaln_dim,),
      'all_x_embedder.2-1.weight': (dit_dim, dit_packed_ch),
      'all_x_embedder.2-1.bias': (dit_dim,),
      'all_final_layer.2-1.linear.weight': (dit_packed_ch, dit_dim),
      'all_final_layer.2-1.linear.bias': (dit_packed_ch,),
      'all_final_layer.2-1.adaLN_modulation.1.weight': (dit_dim, dit_adaln_dim),
      'all_final_layer.2-1.adaLN_modulation.1.bias': (dit_dim,),
      'cap_embedder.0.weight': (te_hidden,),
      'cap_embedder.1.weight': (dit_dim, te_hidden),
      'cap_embedder.1.bias': (dit_dim,),
      'x_pad_token': (1, dit_dim),
      'cap_pad_token': (1, dit_dim),
  }
  _add_dit_block_schema(
      dit_schema,
      'context_refiner.0',
      dit_dim,
      dit_head_dim,
      dit_ff_hidden,
      dit_adaln_dim,
      modulation=False,
  )
  _add_dit_block_schema(
      dit_schema,
      'noise_refiner.0',
      dit_dim,
      dit_head_dim,
      dit_ff_hidden,
      dit_adaln_dim,
      modulation=True,
  )
  for i in range(dit_layers):
    _add_dit_block_schema(
        dit_schema,
        f'layers.{i}',
        dit_dim,
        dit_head_dim,
        dit_ff_hidden,
        dit_adaln_dim,
        modulation=True,
    )
  safetensors.torch.save_file(
      {k: _make_tensor(k, s) for k, s in sorted(dit_schema.items())},
      os.path.join(
          model_dir, 'transformer', 'diffusion_pytorch_model.safetensors'
      ),
  )

  # 3. Tiny AutoencoderKL VAE
  block_ch = (8, 16, 32, 32)
  vae_cfg_dict: dict[str, Any] = {
      'in_channels': 3,
      'out_channels': 3,
      'down_block_types': [
          'DownEncoderBlock2D',
          'DownEncoderBlock2D',
          'DownEncoderBlock2D',
          'DownEncoderBlock2D',
      ],
      'up_block_types': [
          'UpDecoderBlock2D',
          'UpDecoderBlock2D',
          'UpDecoderBlock2D',
          'UpDecoderBlock2D',
      ],
      'block_out_channels': list(block_ch),
      'layers_per_block': 1,
      'latent_channels': dit_in_ch,
      'norm_num_groups': 4,
      'scaling_factor': 0.3611,
      'shift_factor': 0.1159,
      'use_quant_conv': False,
      'use_post_quant_conv': False,
  }
  with open(os.path.join(model_dir, 'vae', 'config.json'), 'w') as f:
    json.dump(vae_cfg_dict, f)
  vae_schema: dict[str, tuple[int, ...]] = {
      'encoder.conv_in.weight': (block_ch[0], 3, 3, 3),
      'encoder.conv_in.bias': (block_ch[0],),
      'encoder.conv_norm_out.weight': (block_ch[-1],),
      'encoder.conv_norm_out.bias': (block_ch[-1],),
      'encoder.conv_out.weight': (2 * dit_in_ch, block_ch[-1], 3, 3),
      'encoder.conv_out.bias': (2 * dit_in_ch,),
      'decoder.conv_in.weight': (block_ch[-1], dit_in_ch, 3, 3),
      'decoder.conv_in.bias': (block_ch[-1],),
      'decoder.conv_norm_out.weight': (block_ch[0],),
      'decoder.conv_norm_out.bias': (block_ch[0],),
      'decoder.conv_out.weight': (3, block_ch[0], 3, 3),
      'decoder.conv_out.bias': (3,),
  }
  prev_ch = block_ch[0]
  for i, ch in enumerate(block_ch):
    _add_vae_resnet_schema(
        vae_schema, f'encoder.down_blocks.{i}.resnets.0', prev_ch, ch
    )
    if i < len(block_ch) - 1:
      vae_schema[f'encoder.down_blocks.{i}.downsamplers.0.conv.weight'] = (
          ch,
          ch,
          3,
          3,
      )
      vae_schema[f'encoder.down_blocks.{i}.downsamplers.0.conv.bias'] = (ch,)
    prev_ch = ch
  _add_vae_mid_block_schema(vae_schema, 'encoder.mid_block', block_ch[-1])

  rev_ch = list(reversed(block_ch))
  prev_ch = rev_ch[0]
  for i, ch in enumerate(rev_ch):
    _add_vae_resnet_schema(
        vae_schema, f'decoder.up_blocks.{i}.resnets.0', prev_ch, ch
    )
    _add_vae_resnet_schema(
        vae_schema, f'decoder.up_blocks.{i}.resnets.1', ch, ch
    )
    if i < len(rev_ch) - 1:
      vae_schema[f'decoder.up_blocks.{i}.upsamplers.0.conv.weight'] = (
          ch,
          ch,
          3,
          3,
      )
      vae_schema[f'decoder.up_blocks.{i}.upsamplers.0.conv.bias'] = (ch,)
    prev_ch = ch
  _add_vae_mid_block_schema(vae_schema, 'decoder.mid_block', block_ch[-1])

  vae_tensors = {k: _make_tensor(k, s) for k, s in sorted(vae_schema.items())}
  # Bias decoder output so generated PNG pixels are non-zero.
  vae_tensors['decoder.conv_out.bias'] = torch.tensor(
      [0.25, -0.1, 0.5], dtype=torch.float32
  )
  safetensors.torch.save_file(
      vae_tensors,
      os.path.join(model_dir, 'vae', 'diffusion_pytorch_model.safetensors'),
  )


class ExportAndRunEndToEndTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    srcdir = os.environ.get('TEST_SRCDIR') or googletest.get_default_test_srcdir()
    workspace = os.environ.get('TEST_WORKSPACE', 'litert')
    self.export_main_bin = os.path.join(
        srcdir, workspace, 'models/tongyi/z_image_turbo/export/export_main'
    )
    self.runner_bin = os.path.join(
        srcdir, workspace, 'models/tongyi/z_image_turbo/z_image_turbo_runner'
    )
    self.assertTrue(
        os.path.exists(self.export_main_bin),
        f'Missing export_main binary at {self.export_main_bin}',
    )
    self.assertTrue(
        os.path.exists(self.runner_bin),
        f'Missing z_image_turbo_runner binary at {self.runner_bin}',
    )

  def test_export_main_cli_dry_run(self):
    tmpdir = self.create_tempdir().full_path
    ckpt_dir = os.path.join(tmpdir, 'ckpt')
    _write_tiny_e2e_checkpoint(ckpt_dir)

    proc = subprocess.run(
        [
            self.export_main_bin,
            f'--model={ckpt_dir}',
            '--dry_run=True',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = f'{proc.stdout}\n{proc.stderr}'
    self.assertEqual(
        proc.returncode,
        0,
        f'export_main --dry_run failed (rc={proc.returncode}):\n{combined}',
    )
    self.assertIn('ZImagePipeline', combined)
    self.assertIn('text_to_image', combined)

  def test_export_main_to_cpp_runner_end_to_end(self):
    tmpdir = self.create_tempdir().full_path
    ckpt_dir = os.path.join(tmpdir, 'ckpt')
    out_dir = os.path.join(tmpdir, 'exported')
    _write_tiny_e2e_checkpoint(ckpt_dir)

    # 1. Run Python `:export_main` CLI binary to emit `model.litertlm`.
    export_proc = subprocess.run(
        [
            self.export_main_bin,
            f'--model={ckpt_dir}',
            f'--output_dir={out_dir}',
            '--task=auto',
            '--quantization_recipe=none',
            '--use_random_weights=False',
            '--bundle_litert_lm=True',
            '--t2i_output_image_size=32',
            '--max_seq_len=16',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        export_proc.returncode,
        0,
        f'export_main failed (rc={export_proc.returncode}):\n'
        f'STDOUT:\n{export_proc.stdout}\nSTDERR:\n{export_proc.stderr}',
    )
    litertlm_path = os.path.join(out_dir, 'model.litertlm')
    self.assertTrue(os.path.exists(litertlm_path))

    # 2. Run C++ `:z_image_turbo_runner` with explicit `--model_name`.
    out_png = os.path.join(tmpdir, 'out_explicit.png')
    run_proc = subprocess.run(
        [
            self.runner_bin,
            '--model_name=z-image-turbo',
            f'--model_path={litertlm_path}',
            '--text=a red fox in snow',
            '--steps=2',
            '--seed=42',
            '--num_threads=1',
            '--cache_dir=:nocache',
            f'--output_path={out_png}',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        run_proc.returncode,
        0,
        f'z_image_turbo_runner failed (rc={run_proc.returncode}):\n'
        f'STDOUT:\n{run_proc.stdout}\nSTDERR:\n{run_proc.stderr}',
    )
    self.assertIn('Saved 32x32 image to', run_proc.stdout)
    self.assertTrue(os.path.exists(out_png))
    width, height, bit_depth, color_type, raw = _parse_png_ihdr(out_png)
    self.assertEqual((width, height, bit_depth, color_type), (32, 32, 8, 2))
    self.assertIn(b'IDAT', raw)
    self.assertGreater(len(raw), 64)

    # 3. Run C++ `:z_image_turbo_runner` with filename-inferred model name.
    inferred_path = os.path.join(tmpdir, 'z_image_turbo_exported.litertlm')
    shutil.copyfile(litertlm_path, inferred_path)
    inferred_png = os.path.join(tmpdir, 'out_inferred.png')
    infer_proc = subprocess.run(
        [
            self.runner_bin,
            f'--model_path={inferred_path}',
            '--text=cat',
            '--steps=2',
            '--seed=7',
            '--num_threads=1',
            '--cache_dir=:nocache',
            f'--output_path={inferred_png}',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        infer_proc.returncode,
        0,
        'z_image_turbo_runner (filename inference) failed '
        f'(rc={infer_proc.returncode}):\nSTDOUT:\n{infer_proc.stdout}\n'
        f'STDERR:\n{infer_proc.stderr}',
    )
    width2, height2, bit_depth2, color_type2, raw2 = _parse_png_ihdr(
        inferred_png
    )
    self.assertEqual((width2, height2, bit_depth2, color_type2), (32, 32, 8, 2))
    self.assertIn(b'IDAT', raw2)


if __name__ == '__main__':
  googletest.main()
