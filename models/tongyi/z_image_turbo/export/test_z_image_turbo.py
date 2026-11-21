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
"""Unit tests for Z-Image-Turbo PyTorch modules, LiteRT export, and bundling."""

import json
import os
import shutil
from typing import Any
from unittest import mock

from google.protobuf import text_format
import litert_torch
from litert_torch.generative.export_hf import export as export_script
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import quantization
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import bonsai_flux2
from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import modeling_flux2
import numpy as np
import safetensors.torch
import torch
import transformers

from models.tongyi.z_image_turbo import z_image_turbo_pb2
from models.tongyi.z_image_turbo.export import export_main as _  # pylint: disable=unused-import
from models.tongyi.z_image_turbo.export import extension as z_image_ext
from models.tongyi.z_image_turbo.export import modeling
from models.tongyi.z_image_turbo.export import z_image_turbo
from absl.testing import absltest as googletest
from absl.testing import parameterized
from ai_edge_litert import compiled_model
from litert_lm_builder import litertlm_peek
from litert_lm_builder.runtime.proto import text2image_metadata_pb2
from ai_edge_litert import schema_py_generated as schema_fb

_FORBIDDEN_GPU_OPS = {
    schema_fb.BuiltinOperator.BROADCAST_TO: 'BROADCAST_TO',
    schema_fb.BuiltinOperator.GATHER_ND: 'GATHER_ND',
    schema_fb.BuiltinOperator.COMPLEX_ABS: 'COMPLEX_ABS',
    schema_fb.BuiltinOperator.REAL: 'REAL',
    schema_fb.BuiltinOperator.IMAG: 'IMAG',
}


def _tiny_qwen_cfg_dict() -> dict[str, Any]:
  return {
      'model_type': 'qwen3',
      'vocab_size': 64,
      'hidden_size': 32,
      'intermediate_size': 64,
      'num_hidden_layers': 3,
      'num_attention_heads': 4,
      'num_key_value_heads': 2,
      'head_dim': 8,
      'tie_word_embeddings': False,
  }


def _tiny_dit_cfg_dict() -> dict[str, Any]:
  return {
      'all_patch_size': [2],
      'all_f_patch_size': [1],
      'in_channels': 4,
      'dim': 24,
      'n_layers': 6,
      'n_refiner_layers': 1,
      'n_heads': 2,
      'n_kv_heads': 2,
      'norm_eps': 1e-5,
      'qk_norm': True,
      'cap_feat_dim': 32,
      'rope_theta': 256.0,
      't_scale': 1000.0,
      'axes_dims': [4, 4, 4],
      'axes_lens': [64, 32, 32],
  }


def _tiny_vae_cfg_dict() -> dict[str, Any]:
  return {
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
      'block_out_channels': [8, 16, 32, 32],
      'layers_per_block': 1,
      'latent_channels': 4,
      'norm_num_groups': 4,
      'scaling_factor': 0.3611,
      'shift_factor': 0.1159,
      'use_quant_conv': False,
      'use_post_quant_conv': False,
  }


def _write_tiny_wordlevel_tokenizer(tokenizer_dir: str, vocab_size: int = 64):
  """Writes a minimal WordLevel tokenizer with Qwen3 framing tokens."""
  os.makedirs(tokenizer_dir, exist_ok=True)
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
  }
  for tid in range(vocab_size):
    if tid not in vocab.values():
      vocab[f'tok_{tid}'] = tid
  with open(os.path.join(tokenizer_dir, 'tokenizer_config.json'), 'w') as f:
    json.dump({'tokenizer_class': 'PreTrainedTokenizerFast'}, f)
  with open(os.path.join(tokenizer_dir, 'tokenizer.json'), 'w') as f:
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


def _write_split_checkpoint(model_dir: str) -> tuple[
    transformers.Qwen3ForCausalLM,
    modeling.ZImageTransformer2DModel,
    modeling.ZImageAutoencoderKL,
]:
  """Writes a valid Diffusers split-directory Z-Image-Turbo checkpoint."""
  for sub in ('tokenizer', 'text_encoder', 'transformer', 'vae', 'scheduler'):
    os.makedirs(os.path.join(model_dir, sub), exist_ok=True)
  with open(os.path.join(model_dir, 'model_index.json'), 'w') as f:
    json.dump({'_class_name': 'ZImagePipeline'}, f)
  with open(
      os.path.join(model_dir, 'scheduler', 'scheduler_config.json'), 'w'
  ) as f:
    json.dump({'shift': 3.0, 'use_dynamic_shifting': False}, f)
  _write_tiny_wordlevel_tokenizer(os.path.join(model_dir, 'tokenizer'))

  torch.manual_seed(0)
  qwen_cfg_dict = _tiny_qwen_cfg_dict()
  with open(os.path.join(model_dir, 'text_encoder', 'config.json'), 'w') as f:
    json.dump(qwen_cfg_dict, f)
  qwen_cfg = transformers.Qwen3Config(
      **{k: v for k, v in qwen_cfg_dict.items() if k != 'model_type'}
  )
  qwen_lm = transformers.Qwen3ForCausalLM(qwen_cfg).eval()
  safetensors.torch.save_file(
      {k: v.contiguous() for k, v in qwen_lm.state_dict().items()},
      os.path.join(model_dir, 'text_encoder', 'model.safetensors'),
  )

  dit_cfg_dict = _tiny_dit_cfg_dict()
  with open(os.path.join(model_dir, 'transformer', 'config.json'), 'w') as f:
    json.dump(dit_cfg_dict, f)
  dit = modeling.ZImageTransformer2DModel(**dit_cfg_dict).eval()
  safetensors.torch.save_file(
      {k: v.contiguous() for k, v in dit.state_dict().items()},
      os.path.join(
          model_dir, 'transformer', 'diffusion_pytorch_model.safetensors'
      ),
  )

  vae_cfg_dict = _tiny_vae_cfg_dict()
  with open(os.path.join(model_dir, 'vae', 'config.json'), 'w') as f:
    json.dump(vae_cfg_dict, f)
  vae = modeling.ZImageAutoencoderKL(**vae_cfg_dict).eval()
  safetensors.torch.save_file(
      {k: v.contiguous() for k, v in vae.state_dict().items()},
      os.path.join(model_dir, 'vae', 'diffusion_pytorch_model.safetensors'),
  )
  return qwen_lm, dit, vae


def _build_tiny_z_image_turbo(
    seq_len: int = 64,
    img_size: int = 64,
) -> z_image_turbo.ZImageTurboModel:
  """Builds a tiny in-memory Z-Image-Turbo model for unit testing."""
  torch.manual_seed(0)
  qwen_cfg = transformers.Qwen3Config(
      vocab_size=64,
      hidden_size=32,
      intermediate_size=64,
      num_hidden_layers=3,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=8,
      max_position_embeddings=128,
      use_cache=False,
  )
  setattr(qwen_cfg, '_attn_implementation', 'eager')
  causal_lm = transformers.Qwen3ForCausalLM(qwen_cfg).eval()
  qwen_model = causal_lm.model

  dit = modeling.ZImageTransformer2DModel(**_tiny_dit_cfg_dict()).eval()
  with torch.no_grad():
    dit.cap_pad_token.copy_(torch.randn_like(dit.cap_pad_token))
    dit.x_pad_token.copy_(torch.randn_like(dit.x_pad_token))

  vae = modeling.ZImageAutoencoderKL(**_tiny_vae_cfg_dict()).eval()
  return z_image_turbo.ZImageTurboModel(
      text_encoder=qwen_model,
      dit=dit,
      vae=vae,
      seq_len=seq_len,
      img_size=img_size,
      default_steps=2,
      num_main_shards=6,
  ).eval()


def _run_dynamic_padded_reference(
    dit: modeling.ZImageTransformer2DModel,
    x_in: torch.Tensor,
    t_in: torch.Tensor,
    valid_cap_feats: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
  """Runs dynamic SEQ_MULTI_OF=32 reference forward matching Diffusers."""
  ori_len = valid_cap_feats.shape[0]
  pad_len = (-ori_len) % 32
  cap_len = ori_len + pad_len
  if pad_len > 0:
    padded_cap = torch.cat(
        [valid_cap_feats, valid_cap_feats[-1:].repeat(pad_len, 1)], dim=0
    )
    pad_mask = torch.cat([
        torch.zeros(ori_len, dtype=torch.bool),
        torch.ones(pad_len, dtype=torch.bool),
    ])
  else:
    padded_cap = valid_cap_feats
    pad_mask = torch.zeros(ori_len, dtype=torch.bool)

  cap_emb = dit.cap_embedder(padded_cap)
  cap_emb = torch.where(pad_mask.unsqueeze(-1), dit.cap_pad_token, cap_emb)
  cap_emb = cap_emb.unsqueeze(0)

  cap_cos, cap_sin, img_cos, img_sin = z_image_turbo.build_z_image_rope(
      seq_len=cap_len,
      grid_h=grid_h,
      grid_w=grid_w,
      axes_dims=dit.axes_dims,
      theta=dit.rope_theta,
  )
  for layer in dit.context_refiner:
    cap_emb = layer(cap_emb, cos=cap_cos, sin=cap_sin)

  adaln = dit.t_embedder(t_in * dit.t_scale)
  x_emb = dit.all_x_embedder['2-1'](x_in)
  for layer in dit.noise_refiner:
    x_emb = layer(x_emb, cos=img_cos, sin=img_sin, adaln_input=adaln)

  unified = torch.cat([x_emb, cap_emb], dim=1)
  unified_cos = torch.cat([img_cos, cap_cos], dim=1)
  unified_sin = torch.cat([img_sin, cap_sin], dim=1)
  for layer in dit.layers:
    unified = layer(
        unified, cos=unified_cos, sin=unified_sin, adaln_input=adaln
    )
  out = dit.all_final_layer['2-1'](unified, adaln)
  return out[:, : x_in.shape[1], :]


class ZImageTurboTest(parameterized.TestCase):
  """Unit tests for Z-Image-Turbo PyTorch modeling and LiteRT export."""

  def _assert_gpu_compatible_tflite(self, tflite_path: str) -> None:
    """Asserts TFLite tensors have rank <= 4 and contain no forbidden GPU ops."""
    with open(tflite_path, 'rb') as f:
      buf = f.read()
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    for sg_idx in range(model.SubgraphsLength()):
      sg = model.Subgraphs(sg_idx)
      for t_idx in range(sg.TensorsLength()):
        t = sg.Tensors(t_idx)
        self.assertLess(
            t.ShapeLength(),
            5,
            f'Tensor {t.Name()} in sg[{sg_idx}] has rank'
            f' {t.ShapeLength()} >= 5',
        )
      for op_idx in range(sg.OperatorsLength()):
        op = sg.Operators(op_idx)
        oc = model.OperatorCodes(op.OpcodeIndex())
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        self.assertNotIn(
            code,
            _FORBIDDEN_GPU_OPS,
            f'Forbidden GPU op {_FORBIDDEN_GPU_OPS.get(code)} in sg[{sg_idx}]',
        )

  def _run_tflite(
      self,
      tflite_path: str,
      inputs: list[np.ndarray],
      out_shapes: list[tuple[int, ...]],
  ) -> list[np.ndarray]:
    """Runs a compiled TFLite graph on CPU and returns reshaped outputs."""
    cm = compiled_model.CompiledModel.from_file(tflite_path)
    in_bufs = cm.create_input_buffers(0)
    out_bufs = cm.create_output_buffers(0)
    for idx, arr in enumerate(inputs):
      in_bufs[idx].write(np.ascontiguousarray(arr, dtype=arr.dtype))
    cm.run_by_index(0, in_bufs, out_bufs)
    results = []
    for idx, shape in enumerate(out_shapes):
      num_elems = int(np.prod(shape))
      results.append(out_bufs[idx].read(num_elems, np.float32).reshape(shape))
    return results

  def test_extension_registration_and_section_types(self):
    for model_type in ('ZImagePipeline', 'z_image_turbo', 'z-image-turbo'):
      ext = extension_lib.get_model_extension(model_type)
      self.assertIsInstance(ext, z_image_ext.ZImageTurboExtension)
    ext = z_image_ext.ZImageTurboExtension()
    self.assertEqual(
        ext.get_litert_lm_section_types(),
        z_image_turbo.Z_IMAGE_TURBO_STANDARD_SECTION_TYPES,
    )
    self.assertIs(
        ext.get_image_gen_model_cls('z_image_turbo'), z_image_turbo.ZImageTurbo
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='diffusers_class_name_text_to_image',
          model_path='/tmp/model',
          task=extension_lib.ExportTask.TEXT_TO_IMAGE,
          config_dict={'_class_name': 'ZImagePipeline'},
          expected=True,
      ),
      dict(
          testcase_name='diffusers_class_name_auto',
          model_path='/tmp/model',
          task='auto',
          config_dict={'_class_name': 'ZImagePipeline'},
          expected=True,
      ),
      dict(
          testcase_name='path_fallback_without_conflicting_config',
          model_path='Tongyi-MAI/Z-Image-Turbo',
          task='text_to_image',
          config_dict={},
          expected=True,
      ),
      dict(
          testcase_name='rejects_other_pipeline',
          model_path='Tongyi-MAI/Z-Image-Turbo',
          task='text_to_image',
          config_dict={'_class_name': 'Flux2Pipeline'},
          expected=False,
      ),
      dict(
          testcase_name='rejects_wrong_task',
          model_path='Tongyi-MAI/Z-Image-Turbo',
          task=extension_lib.ExportTask.TEXT_GENERATION,
          config_dict={'_class_name': 'ZImagePipeline'},
          expected=False,
      ),
  )
  def test_match_checkpoint(
      self,
      model_path: str,
      task: Any,
      config_dict: dict[str, Any],
      expected: bool,
  ):
    """Checks `match_checkpoint` across positive and negative configs."""
    self.assertEqual(
        z_image_ext.ZImageTurboExtension.match_checkpoint(
            model_path, task=task, config_dict=config_dict
        ),
        expected,
    )

  def test_dry_run_auto_detects_z_image_turbo(self):
    tmpdir = self.create_tempdir().full_path
    model_dir = os.path.join(tmpdir, 'z_image_ckpt')
    _write_split_checkpoint(model_dir)
    plan = export_script.export(model=model_dir, task='auto', dry_run=True)
    self.assertEqual(plan.task, 'text_to_image')
    self.assertEqual(plan.model_type, 'ZImagePipeline')

  def test_verify_checkpoint_keys_split_dir_and_errors(self):
    """Checks strict key validation for split-directory checkpoints."""
    tmpdir = self.create_tempdir().full_path
    model_dir = os.path.join(tmpdir, 'split_ckpt')
    qwen_lm, dit, vae = _write_split_checkpoint(model_dir)

    z_image_turbo.verify_checkpoint_keys(model_dir, dit, vae, qwen_lm)

    # Extra unconsumed key in transformer raises ValueError.
    dit_path = os.path.join(
        model_dir, 'transformer', 'diffusion_pytorch_model.safetensors'
    )
    extra_sd = {k: v.contiguous() for k, v in dit.state_dict().items()}
    extra_sd['unexpected.bogus.weight'] = torch.zeros(1)
    safetensors.torch.save_file(extra_sd, dit_path)
    with self.assertRaisesRegex(ValueError, 'unexpected'):
      z_image_turbo.verify_checkpoint_keys(model_dir, dit, vae, qwen_lm)

    # Missing key in transformer raises ValueError.
    missing_sd = {k: v.contiguous() for k, v in dit.state_dict().items()}
    missing_sd.pop('x_pad_token')
    safetensors.torch.save_file(missing_sd, dit_path)
    with self.assertRaisesRegex(ValueError, 'missing'):
      z_image_turbo.verify_checkpoint_keys(model_dir, dit, vae, qwen_lm)

  def test_verify_checkpoint_keys_missing_text_encoder_dir_in_split_layout(
      self,
  ):
    """Requires `text_encoder/` unless `text_encoder_lm` is `None`."""
    tmpdir = self.create_tempdir().full_path
    model_dir = os.path.join(tmpdir, 'split_missing_te')
    qwen_lm, dit, vae = _write_split_checkpoint(model_dir)
    shutil.rmtree(os.path.join(model_dir, 'text_encoder'))

    with self.assertRaisesRegex(
        ValueError, 'Missing expected Z-Image-Turbo subdirectory.*text_encoder'
    ):
      z_image_turbo.verify_checkpoint_keys(model_dir, dit, vae, qwen_lm)

    # Skipping text encoder verification remains allowed when `None`.
    z_image_turbo.verify_checkpoint_keys(
        model_dir, dit, vae, text_encoder_lm=None
    )

  def test_split_checkpoint_missing_text_encoder_dir_raises_on_load_and_export(
      self,
  ):
    """Raises instead of falling back to random text encoder weights."""
    tmpdir = self.create_tempdir().full_path
    model_dir = os.path.join(tmpdir, 'split_missing_te')
    out_dir = os.path.join(tmpdir, 'out')
    _write_split_checkpoint(model_dir)
    shutil.rmtree(os.path.join(model_dir, 'text_encoder'))

    with self.assertRaisesRegex(
        ValueError, 'Missing expected Z-Image-Turbo subdirectory.*text_encoder'
    ):
      z_image_turbo.ZImageTurboModel.from_pretrained(
          model_dir, seq_len=8, img_size=32, use_random_weights=False
      )

    exporter = z_image_turbo.ZImageTurbo(model_dir)
    cfg = exportable_module_config.ExportableModuleConfig(
        model=model_dir,
        output_dir=out_dir,
        task=extension_lib.ExportTask.TEXT_TO_IMAGE,
        quantization_recipe='none',
        use_random_weights=False,
        t2i_output_image_size=32,
        extra_kwargs={'max_seq_len': 16},
    )
    with self.assertRaisesRegex(
        ValueError, 'Missing expected Z-Image-Turbo subdirectory.*text_encoder'
    ):
      exporter.export(cfg)

  def test_from_pretrained_single_file_missing_weights_and_random_weights(self):
    """Rejects missing single-file weights unless `use_random_weights=True`."""
    tmpdir = self.create_tempdir().full_path
    model_dir = os.path.join(tmpdir, 'single_no_weights')
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
      json.dump(
          {
              'text_encoder': _tiny_qwen_cfg_dict(),
              'transformer': _tiny_dit_cfg_dict(),
              'vae': _tiny_vae_cfg_dict(),
          },
          f,
      )

    with self.assertRaisesRegex(
        ValueError, 'Missing single-file checkpoint weights for text_encoder'
    ):
      z_image_turbo.ZImageTurboModel.from_pretrained(
          model_dir, seq_len=8, img_size=32, use_random_weights=False
      )

    loaded = z_image_turbo.ZImageTurboModel.from_pretrained(
        model_dir, seq_len=8, img_size=32, use_random_weights=True
    )
    self.assertEqual(loaded.dit.dim, 24)
    self.assertIsNotNone(loaded.text_encoder_lm)

  def test_from_pretrained_split_dir_use_random_weights(self):
    """Allows random initialization on split checkpoints without weights."""
    tmpdir = self.create_tempdir().full_path
    model_dir = os.path.join(tmpdir, 'split_random')
    _write_split_checkpoint(model_dir)
    os.remove(os.path.join(model_dir, 'text_encoder', 'model.safetensors'))
    os.remove(
        os.path.join(
            model_dir, 'transformer', 'diffusion_pytorch_model.safetensors'
        )
    )
    os.remove(
        os.path.join(model_dir, 'vae', 'diffusion_pytorch_model.safetensors')
    )

    loaded = z_image_turbo.ZImageTurboModel.from_pretrained(
        model_dir, seq_len=8, img_size=32, use_random_weights=True
    )
    self.assertEqual(loaded.dit.dim, 24)
    self.assertIsNotNone(loaded.text_encoder_lm)

    shutil.rmtree(os.path.join(model_dir, 'text_encoder'))
    random_lm = z_image_turbo._load_text_encoder_lm(  # pylint: disable=protected-access
        model_dir,
        has_split_dirs=True,
        root_config={'text_encoder': _tiny_qwen_cfg_dict()},
        single_file_sd=None,
        use_random_weights=True,
    )
    self.assertIsInstance(random_lm, torch.nn.Module)
    self.assertFalse(random_lm.training)

  def test_verify_checkpoint_keys_single_file_prefixed(self):
    """Checks key validation and loading for single-file prefixed weights."""
    tmpdir = self.create_tempdir().full_path
    model_dir = os.path.join(tmpdir, 'single_ckpt')
    os.makedirs(model_dir, exist_ok=True)

    torch.manual_seed(1)
    qwen_cfg_dict = _tiny_qwen_cfg_dict()
    dit_cfg_dict = _tiny_dit_cfg_dict()
    vae_cfg_dict = _tiny_vae_cfg_dict()
    qwen_cfg = transformers.Qwen3Config(
        **{k: v for k, v in qwen_cfg_dict.items() if k != 'model_type'}
    )
    qwen_lm = transformers.Qwen3ForCausalLM(qwen_cfg).eval()
    dit = modeling.ZImageTransformer2DModel(**dit_cfg_dict).eval()
    vae = modeling.ZImageAutoencoderKL(**vae_cfg_dict).eval()

    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
      json.dump(
          {
              'text_encoder': qwen_cfg_dict,
              'transformer': dit_cfg_dict,
              'vae': vae_cfg_dict,
          },
          f,
      )
    combined: dict[str, torch.Tensor] = {}
    for k, v in qwen_lm.state_dict().items():
      combined[f'text_encoder.{k}'] = v.contiguous()
    for k, v in dit.state_dict().items():
      combined[f'transformer.{k}'] = v.contiguous()
    for k, v in vae.state_dict().items():
      combined[f'vae.{k}'] = v.contiguous()
    safetensors_path = os.path.join(model_dir, 'model.safetensors')
    safetensors.torch.save_file(combined, safetensors_path)

    z_image_turbo.verify_checkpoint_keys(model_dir, dit, vae, qwen_lm)
    loaded = z_image_turbo.ZImageTurboModel.from_pretrained(
        model_dir, seq_len=8, img_size=32, use_random_weights=False
    )
    self.assertEqual(loaded.dit.dim, 24)

    combined['stray.unconsumed'] = torch.zeros(1)
    safetensors.torch.save_file(combined, safetensors_path)
    with self.assertRaisesRegex(ValueError, 'unexpected'):
      z_image_turbo.verify_checkpoint_keys(model_dir, dit, vae, qwen_lm)

  def test_rope_matches_diffusers_complex_rope_and_sigmas(self):
    """Verifies 3D RoPE parity with complex64 RoPE and static shift=3.0."""
    cap_cos, cap_sin, img_cos, img_sin = z_image_turbo.build_z_image_rope(
        seq_len=8, grid_h=2, grid_w=2, axes_dims=(4, 4, 4), theta=256.0
    )
    sigmas = z_image_turbo.compute_z_image_sigmas(steps=8, image_tokens=256)

    cap_ids = z_image_turbo.build_z_image_cap_ids(8).long()
    freqs_cis_parts = []
    for axis_idx, d in enumerate((4, 4, 4)):
      freqs = 1.0 / (256.0 ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
      t_steps = torch.arange(32, dtype=torch.float64)
      angle = torch.outer(t_steps, freqs).float()
      cis = torch.polar(torch.ones_like(angle), angle).to(torch.complex64)
      freqs_cis_parts.append(cis[cap_ids[:, axis_idx]])
    freqs_cis = torch.cat(freqs_cis_parts, dim=-1).unsqueeze(0).unsqueeze(2)

    x_in = torch.randn(1, 8, 2, 12, dtype=torch.float32)
    real_rot = modeling_flux2.apply_rotary_emb(
        x_in, (cap_cos, cap_sin), sequence_dim=1
    )
    x_complex = torch.view_as_complex(x_in.reshape(1, 8, 2, 6, 2))
    complex_rot = torch.view_as_real(x_complex * freqs_cis).flatten(3)

    np.testing.assert_allclose(
        real_rot.numpy(), complex_rot.numpy(), rtol=1e-5, atol=1e-5
    )
    self.assertEqual(tuple(img_cos.shape), (1, 4, 1, 12))
    self.assertEqual(tuple(img_sin.shape), (1, 4, 1, 12))
    expected_sigmas = [
        1.0,
        21.0 / 22.0,
        9.0 / 10.0,
        5.0 / 6.0,
        0.75,
        9.0 / 14.0,
        0.50,
        0.30,
        0.0,
    ]
    np.testing.assert_allclose(sigmas, expected_sigmas, rtol=1e-6, atol=1e-6)

  def test_pack_and_unpack_latents_round_trip(self):
    z = torch.arange(1 * 4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
    packed = z_image_turbo.pack_z_image_latents(z, patch_size=2)
    unpacked = z_image_turbo.unpack_z_image_latents(
        packed, latent_h=4, latent_w=4, patch_size=2, in_channels=4
    )
    self.assertEqual(tuple(packed.shape), (1, 4, 16))
    np.testing.assert_allclose(z.numpy(), unpacked.numpy(), rtol=0, atol=0)

  def test_state_dict_keys_match_tongyi_z_image_turbo_checkpoint(self):
    """Verifies exact 521 DiT and 244 VAE state_dict keys."""
    dit = modeling.ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=12,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=16,
        axes_dims=(4, 4, 4),
    )
    vae = modeling.ZImageAutoencoderKL(
        in_channels=3,
        out_channels=3,
        block_out_channels=(8, 16, 32, 32),
        layers_per_block=2,
        latent_channels=4,
        norm_num_groups=4,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )

    dit_keys = set(dit.state_dict().keys())
    vae_keys = set(vae.state_dict().keys())

    self.assertLen(dit_keys, 521)
    self.assertIn('all_x_embedder.2-1.weight', dit_keys)
    self.assertIn('all_final_layer.2-1.adaLN_modulation.1.weight', dit_keys)
    self.assertIn('cap_embedder.0.weight', dit_keys)
    self.assertIn('cap_embedder.1.weight', dit_keys)
    self.assertIn('t_embedder.mlp.0.weight', dit_keys)
    self.assertIn('t_embedder.mlp.2.weight', dit_keys)
    self.assertIn('context_refiner.0.attention.norm_q.weight', dit_keys)
    self.assertIn('noise_refiner.1.adaLN_modulation.0.weight', dit_keys)
    self.assertIn('layers.29.feed_forward.w1.weight', dit_keys)
    self.assertIn('x_pad_token', dit_keys)
    self.assertIn('cap_pad_token', dit_keys)

    self.assertLen(vae_keys, 244)
    self.assertFalse(any(k.startswith('bn.') for k in vae_keys))
    self.assertFalse(any(k.startswith('quant_conv.') for k in vae_keys))
    self.assertFalse(any(k.startswith('post_quant_conv.') for k in vae_keys))

  def test_export_all_shards_and_parity(self):
    """Verifies TFLite conversion and numerical parity across all 13 shards."""
    model = _build_tiny_z_image_turbo(seq_len=64, img_size=64)
    valid_tokens = 12
    active_cap_len = z_image_turbo.compute_z_image_padded_cap_len(
        valid_tokens, seq_len=64
    )
    self.assertEqual(active_cap_len, 32)

    cap_cos, cap_sin, img_cos, img_sin = z_image_turbo.build_z_image_rope(
        seq_len=64,
        grid_h=4,
        grid_w=4,
        axes_dims=(4, 4, 4),
        theta=256.0,
        active_cap_len=active_cap_len,
    )
    unified_cos = torch.cat([img_cos, cap_cos], dim=1)
    unified_sin = torch.cat([img_sin, cap_sin], dim=1)

    input_ids = torch.randint(0, 64, (1, 64), dtype=torch.int32)
    mask_list = [1] * valid_tokens + [0] * (64 - valid_tokens)
    attention_mask = torch.tensor([mask_list], dtype=torch.int32)
    x_in = torch.randn(1, 16, 16, dtype=torch.float32)
    t_in = torch.tensor([0.5], dtype=torch.float32)

    with torch.no_grad():
      pt_cap_feats = model.text_encoder(input_ids, attention_mask)
      ref_out = _run_dynamic_padded_reference(
          model.dit,
          x_in=x_in,
          t_in=t_in,
          valid_cap_feats=pt_cap_feats[0, :valid_tokens, :],
          grid_h=4,
          grid_w=4,
      )
      ref_latents = z_image_turbo.unpack_z_image_latents(
          ref_out, latent_h=8, latent_w=8, patch_size=2, in_channels=4
      )

      pt_cap = model.dit_cap_embedder(pt_cap_feats)
      pt_cap_ref = model.dit_context_refiner(pt_cap, cap_cos, cap_sin)
      pt_x_emb, pt_adaln = model.dit_x_embedder(x_in, t_in)
      pt_x_ref = model.dit_noise_refiner(pt_x_emb, img_cos, img_sin, pt_adaln)
      pt_unified = torch.cat([pt_x_ref, pt_cap_ref], dim=1)
      for shard in model.dit_main_shards:
        pt_unified = shard(pt_unified, unified_cos, unified_sin, pt_adaln)
      pt_out = model.dit_final(pt_unified, pt_adaln)
      pt_latents = z_image_turbo.unpack_z_image_latents(
          pt_out, latent_h=8, latent_w=8, patch_size=2, in_channels=4
      )
      pt_rgb = model.vae_decoder(pt_latents)

    np.testing.assert_allclose(
        pt_latents.numpy(), ref_latents.numpy(), rtol=1e-4, atol=1e-4
    )

    tmp_dir = self.create_tempdir().full_path
    te_path = os.path.join(tmp_dir, 'te.tflite')
    litert_torch.convert(
        model.text_encoder, (input_ids, attention_mask)
    ).export(te_path)
    self._assert_gpu_compatible_tflite(te_path)
    lt_cap_feats = self._run_tflite(
        te_path, [input_ids.numpy(), attention_mask.numpy()], [(1, 64, 32)]
    )[0]
    np.testing.assert_allclose(
        pt_cap_feats.numpy(), lt_cap_feats, rtol=1e-4, atol=1e-4
    )

    embc_path = os.path.join(tmp_dir, 'embc.tflite')
    litert_torch.convert(model.dit_cap_embedder, (pt_cap_feats,)).export(
        embc_path
    )
    self._assert_gpu_compatible_tflite(embc_path)
    lt_cap = self._run_tflite(embc_path, [lt_cap_feats], [(1, 64, 24)])[0]
    np.testing.assert_allclose(pt_cap.numpy(), lt_cap, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        lt_cap[0, valid_tokens],
        model.dit.cap_pad_token.detach().numpy()[0],
        rtol=1e-5,
        atol=1e-5,
    )

    refc_path = os.path.join(tmp_dir, 'refc.tflite')
    litert_torch.convert(
        model.dit_context_refiner, (pt_cap, cap_cos, cap_sin)
    ).export(refc_path)
    self._assert_gpu_compatible_tflite(refc_path)
    lt_cap_ref = self._run_tflite(
        refc_path,
        [lt_cap, cap_cos.numpy(), cap_sin.numpy()],
        [(1, 64, 24)],
    )[0]
    np.testing.assert_allclose(
        pt_cap_ref[:, :active_cap_len].numpy(),
        lt_cap_ref[:, :active_cap_len],
        rtol=1e-4,
        atol=1e-4,
    )

    embx_path = os.path.join(tmp_dir, 'embx.tflite')
    litert_torch.convert(model.dit_x_embedder, (x_in, t_in)).export(embx_path)
    self._assert_gpu_compatible_tflite(embx_path)
    embx_outs = self._run_tflite(
        embx_path, [x_in.numpy(), t_in.numpy()], [(1, 16, 24), (1, 24)]
    )
    lt_x_emb, lt_adaln = embx_outs[0], embx_outs[1]
    np.testing.assert_allclose(pt_x_emb.numpy(), lt_x_emb, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(pt_adaln.numpy(), lt_adaln, rtol=1e-4, atol=1e-4)

    refx_path = os.path.join(tmp_dir, 'refx.tflite')
    litert_torch.convert(
        model.dit_noise_refiner, (pt_x_emb, img_cos, img_sin, pt_adaln)
    ).export(refx_path)
    self._assert_gpu_compatible_tflite(refx_path)
    lt_x_ref = self._run_tflite(
        refx_path,
        [lt_x_emb, img_cos.numpy(), img_sin.numpy(), lt_adaln],
        [(1, 16, 24)],
    )[0]
    np.testing.assert_allclose(pt_x_ref.numpy(), lt_x_ref, rtol=1e-4, atol=1e-4)

    lt_unified = np.concatenate([lt_x_ref, lt_cap_ref], axis=1)
    for idx, shard in enumerate(model.dit_main_shards):
      main_path = os.path.join(tmp_dir, f'main_{idx}.tflite')
      litert_torch.convert(
          shard, (pt_unified, unified_cos, unified_sin, pt_adaln)
      ).export(main_path)
      self._assert_gpu_compatible_tflite(main_path)
      lt_unified = self._run_tflite(
          main_path,
          [lt_unified, unified_cos.numpy(), unified_sin.numpy(), lt_adaln],
          [(1, 80, 24)],
      )[0]

    final_path = os.path.join(tmp_dir, 'final.tflite')
    litert_torch.convert(model.dit_final, (pt_unified, pt_adaln)).export(
        final_path
    )
    self._assert_gpu_compatible_tflite(final_path)
    lt_out = self._run_tflite(
        final_path, [lt_unified, lt_adaln], [(1, 16, 16)]
    )[0]
    np.testing.assert_allclose(ref_out.numpy(), lt_out, rtol=1e-4, atol=1e-4)

    vae_path = os.path.join(tmp_dir, 'vae.tflite')
    litert_torch.convert(model.vae_decoder, (pt_latents,)).export(vae_path)
    self._assert_gpu_compatible_tflite(vae_path)
    lt_rgb = self._run_tflite(vae_path, [pt_latents.numpy()], [(1, 3, 64, 64)])[
        0
    ]
    np.testing.assert_allclose(pt_rgb.numpy(), lt_rgb, rtol=1e-4, atol=1e-4)

  def test_end_to_end_export_bundles_litertlm_with_custom_proto_and_sections(
      self,
  ):
    """Verifies `.litertlm` bundling with custom_model and all 13 sections."""
    tmp_dir = self.create_tempdir().full_path
    model_dir = os.path.join(tmp_dir, 'z_image_ckpt')
    out_dir = os.path.join(tmp_dir, 'out')
    _write_split_checkpoint(model_dir)

    export_script.export(
        model=model_dir,
        output_dir=out_dir,
        task='auto',
        quantization_recipe='none',
        use_random_weights=False,
        t2i_output_image_size=32,
        max_seq_len=16,
        bundle_litert_lm=True,
    )

    litertlm_path = os.path.join(out_dir, 'model.litertlm')
    self.assertTrue(os.path.exists(litertlm_path))

    dump_dir = os.path.join(tmp_dir, 'unpacked')
    os.makedirs(dump_dir, exist_ok=True)
    peek_path = os.path.join(tmp_dir, 'peek.txt')
    with open(peek_path, 'w') as peek_out:
      litertlm_peek.peek_litertlm_file(litertlm_path, dump_dir, peek_out)
    with open(peek_path, 'r') as peek_in:
      peek_text = peek_in.read()

    self.assertIn('Data Type:    Text2ImageMetadataProto', peek_text)
    self.assertIn('custom_model {', peek_text)
    self.assertIn(
        'prototypes.litert_models.z_image_turbo.ZImageTurbo', peek_text
    )
    for (
        expected_type
    ) in z_image_turbo.Z_IMAGE_TURBO_STANDARD_SECTION_TYPES.values():
      self.assertIn(
          f'Key: model_type, Value (String): {expected_type}', peek_text
      )
    for expected_bin in z_image_turbo.Z_IMAGE_TURBO_GENERIC_BINARY_SHARDS:
      self.assertIn(f'Key: name, Value (String): {expected_bin}', peek_text)

    # Verify the unpacked `Text2ImageMetadataProto.pbtext` unpacks cleanly into
    # `z_image_turbo_pb2.ZImageTurbo`.
    pbtext_path = os.path.join(dump_dir, 'Text2ImageMetadataProto.pbtext')
    self.assertTrue(os.path.exists(pbtext_path))
    with open(pbtext_path, 'r') as pbtext_file:
      meta = text_format.Parse(
          pbtext_file.read(), text2image_metadata_pb2.Text2ImageMetadata()
      )
    self.assertTrue(meta.image_gen_model_type.HasField('custom_model'))
    self.assertTrue(
        meta.image_gen_model_type.custom_model.Is(
            z_image_turbo_pb2.ZImageTurbo.DESCRIPTOR
        )
    )
    z_proto = z_image_turbo_pb2.ZImageTurbo()
    self.assertTrue(meta.image_gen_model_type.custom_model.Unpack(z_proto))
    self.assertEqual(z_proto.seq_len, 16)
    self.assertEqual(z_proto.img_size, 32)
    self.assertEqual(z_proto.in_channels, 4)
    self.assertEqual(z_proto.patch_size, 2)
    self.assertEqual(z_proto.dim, 24)
    self.assertEqual(z_proto.prompt_dim, 32)
    self.assertEqual(z_proto.default_steps, 8)
    self.assertEqual(list(z_proto.axes_dims), [4, 4, 4])
    self.assertAlmostEqual(z_proto.rope_theta, 256.0, places=5)
    self.assertAlmostEqual(z_proto.shift, 3.0, places=5)
    self.assertEqual(z_proto.pad_token_id, 0)
    self.assertEqual(list(z_proto.prefix_token_ids), [1])
    self.assertEqual(list(z_proto.suffix_token_ids), [2, 3, 1, 4, 3])

  def test_extracts_framing_ids_without_tokenizer_config(self):
    """Falls back to PreTrainedTokenizerFast without tokenizer_config.json."""
    tmpdir = self.create_tempdir().full_path
    model_dir = os.path.join(tmpdir, 'z_image_ckpt')
    _write_split_checkpoint(model_dir)
    os.remove(os.path.join(model_dir, 'tokenizer', 'tokenizer_config.json'))

    exporter = z_image_turbo.ZImageTurbo(model_dir)
    cfg = exportable_module_config.ExportableModuleConfig(
        model=model_dir,
        task=extension_lib.ExportTask.TEXT_TO_IMAGE,
        t2i_output_image_size=32,
        extra_kwargs={'max_seq_len': 16},
    )
    meta = exporter.get_image_gen_metadata(cfg)
    z_proto = z_image_turbo_pb2.ZImageTurbo()
    self.assertTrue(meta.image_gen_model_type.custom_model.Unpack(z_proto))
    self.assertEqual(z_proto.pad_token_id, 0)
    self.assertEqual(list(z_proto.prefix_token_ids), [1])
    self.assertEqual(list(z_proto.suffix_token_ids), [2, 3, 1, 4, 3])

  def test_non_default_quantization_recipe_delegates_to_export_hf(self):
    """Non-default quantization recipes delegate to export_hf quantization."""
    tmpdir = self.create_tempdir().full_path
    model_dir = os.path.join(tmpdir, 'z_image_ckpt')
    _write_split_checkpoint(model_dir)
    exporter = z_image_turbo.ZImageTurbo(model_dir)

    out_dir = os.path.join(tmpdir, 'quant_check')
    os.makedirs(out_dir, exist_ok=True)
    tiny_linear = torch.nn.Linear(4, 4).eval()
    sample = (torch.zeros(1, 4, dtype=torch.float32),)

    def _fake_quant(src_path: str, recipe: str) -> str:
      del recipe
      q_path = src_path.replace('.tflite', '_quantized.tflite')
      with open(q_path, 'wb') as f:
        f.write(b'quantized')
      return q_path

    with mock.patch.object(
        quantization,
        'maybe_quantize_model',
        autospec=True,
        side_effect=_fake_quant,
    ) as mock_maybe_quant:
      result_path = exporter._export_single_tflite(  # pylint: disable=protected-access
          tiny_linear,
          sample,
          out_dir,
          'sample_shard',
          quantize=True,
          quantization_recipe='dynamic_wi8_afp32',
          keep_temporary_files=False,
          default_quant_fn=bonsai_flux2.quantize_dit,
      )
    mock_maybe_quant.assert_called_once()
    self.assertEqual(mock_maybe_quant.call_args.args[1], 'dynamic_wi8_afp32')
    self.assertTrue(result_path.endswith('_quantized.tflite'))


if __name__ == '__main__':
  googletest.main()
