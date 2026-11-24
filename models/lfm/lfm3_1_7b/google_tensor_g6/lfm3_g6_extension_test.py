# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for lfm3_g6_extension.py (CPU, no weights, no download).

  python lfm3_g6_extension_test.py

The classes below stand in for the checkpoint's remote code: the patched
forwards look up `apply_rotary_pos_emb`, `rotate_half` and the attention
functions in the module of their class, which is this one.
"""

import json
import os
import sys
import types
from unittest import mock

from absl.testing import absltest
import torch
import transformers
from transformers import modeling_utils

from litert_torch.generative.export_hf import export
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import weights_loader
from litert_torch.generative.export_hf.core.split_cache import attention as split_cache_attention
from litert_torch.generative.export_hf.core.split_cache import attention_mask
from litert_torch.generative.export_hf.core.split_cache import cache as split_cache
from litert_torch.generative.export_hf.experimental.composites import short_conv as short_conv_composite
from litert_torch.generative.export_hf.model_ext import metadata_builder
from litert_torch.generative.export_hf.model_ext import testing as ext_testing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lfm3_g6_extension as g6  # pylint: disable=g-import-not-at-top

# The handwritten NPU components the extension turns on.
_NPU_SETTINGS = (
    'enable_neon_for_npu_greedy_sampling',
    'use_hw_masking_for_npu',
    'use_hw_cache_update_for_npu',
)


def _rotate_half(x):
  """The HF `rotate_half` the patch replaces."""
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


ALL_ATTENTION_FUNCTIONS = modeling_utils.ALL_ATTENTION_FUNCTIONS
eager_attention_forward = None
rotate_half = _rotate_half


def apply_rotary_pos_emb(q, k, cos, sin):
  cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
  return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class _Attention(torch.nn.Module):
  """What the patched forward reads of `Lfm3Attention`: 4 heads, 2 KV heads."""

  def __init__(self, hidden=64, heads=4, kv_heads=2, head_dim=16):
    super().__init__()
    linear = lambda n, m: torch.nn.Linear(n, m, bias=False)
    self.q_proj = linear(hidden, heads * head_dim)
    self.k_proj = linear(hidden, kv_heads * head_dim)
    self.v_proj = linear(hidden, kv_heads * head_dim)
    self.o_proj = linear(heads * head_dim, hidden)
    self.gate_proj = linear(hidden, heads)
    self.q_norm = torch.nn.RMSNorm(head_dim, eps=1e-5)
    self.k_norm = torch.nn.RMSNorm(head_dim, eps=1e-5)
    self.layer_idx, self.head_dim, self.sliding_window = 0, head_dim, None
    self.num_key_value_groups, self.scaling = heads // kv_heads, head_dim**-0.5
    self.config = types.SimpleNamespace(
        _attn_implementation='lrt_split_cache_attention'
    )


class _ShortConv(torch.nn.Module):
  """What `Lfm3LiteRTShortConv` takes from `Lfm3ShortConv`: kernel size 7."""

  def __init__(self, hidden, bias):
    super().__init__()
    self.layer_idx = 0
    self.in_proj = torch.nn.Linear(hidden, 3 * hidden, bias=bias)
    self.conv = torch.nn.Conv1d(hidden, hidden, 7, groups=hidden, bias=bias)
    self.out_proj = torch.nn.Linear(hidden, hidden, bias=bias)


class Lfm3RMSNorm(torch.nn.Module):

  def __init__(self, size):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.rand(size) + 0.5)
    self.variance_epsilon = 1e-5


class Lfm3MLP(torch.nn.Module):

  def __init__(self, size, inter=24):
    super().__init__()
    self.w1 = torch.nn.Linear(size, inter, bias=False)
    self.w3 = torch.nn.Linear(size, inter, bias=False)
    self.w2 = torch.nn.Linear(inter, size, bias=False)

  def forward(self, x):
    return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class Lfm3DecoderLayer(torch.nn.Module):

  def __init__(self, size):
    super().__init__()
    self.ffn_norm = Lfm3RMSNorm(size)
    self.feed_forward = Lfm3MLP(size)


class Lfm3Model(torch.nn.Module):

  def __init__(self, vocab, size):
    super().__init__()
    self.embed_tokens = torch.nn.Embedding(vocab, size)
    self.layers = torch.nn.ModuleList([Lfm3DecoderLayer(size)])
    self.embedding_norm = Lfm3RMSNorm(size)


class Lfm3ForCausalLM(torch.nn.Module):

  def __init__(self, vocab=32, size=16):
    super().__init__()
    self.model = Lfm3Model(vocab, size)
    self.lm_head = torch.nn.Linear(size, vocab, bias=False)
    self.lm_head.weight = self.model.embed_tokens.weight  # tied, as in LFM3


_SPLIT = types.SimpleNamespace(split_cache=True)


class PatchTest(absltest.TestCase):

  def test_short_conv_matches_composite(self):
    torch.manual_seed(0)
    hidden = 16
    # (tokens, real tokens): decode, a full prefill, prefills with padding.
    cases = ((1, None), (8, None), (8, 8), (8, 5), (8, 1))
    for bias in (False, True):
      original = _ShortConv(hidden, bias)
      conv = g6.Lfm3LiteRTShortConv(original)
      for t, real in cases:
        with self.subTest(bias=bias, t=t, real=real):
          x = torch.randn(1, t, hidden)
          state = torch.randn(1, 6, hidden)
          cache = types.SimpleNamespace(
              layers=[types.SimpleNamespace(conv_states=state)]
          )
          mask, num_valid = None, None
          if real is not None:
            mask = torch.arange(t)[None] < real
            num_valid = torch.tensor([real], dtype=torch.int32)
          with mock.patch.object(
              short_conv_composite,
              'short_conv',
              wraps=short_conv_composite.short_conv,
          ) as composite:
            got = conv(x, past_key_values=cache, attention_mask=mask)
          # Decode computes the conv itself; a prefill calls the composite.
          self.assertEqual(composite.call_count, int(t > 1))
          want, want_state = short_conv_composite.short_conv(
              in_proj_out=original.in_proj(x),
              conv_state=state,
              conv_weight=original.conv.weight,
              conv_bias=original.conv.bias,
              num_valid_tokens=num_valid,
          )
          torch.testing.assert_close(
              got, original.out_proj(want), rtol=1e-5, atol=1e-5
          )
          torch.testing.assert_close(
              cache.layers[0].conv_states, want_state, rtol=0, atol=0
          )

  def test_non_split_cache_export_stops(self):
    model = Lfm3ForCausalLM()
    head = model.lm_head
    with self.assertRaisesRegex(NotImplementedError, 'split_cache'):
      with g6.patch_lfm3_model(model, types.SimpleNamespace(split_cache=False)):
        pass
    self.assertIs(model.lm_head, head)

  def test_moves_final_norm_scale_into_lm_head(self):
    model = Lfm3ForCausalLM()
    table = model.model.embed_tokens.weight.detach().clone()
    norm_weight = model.model.embedding_norm.weight.detach().clone()
    x = torch.randn(2, 3, 16) * 50
    want = model.lm_head(g6.PlainRMSNorm(norm_weight, 1e-5)(x))
    with g6.patch_lfm3_model(model, _SPLIT):
      head, norm = model.lm_head, model.model.embedding_norm
      self.assertTrue(torch.equal(head.weight, table * 8))
      self.assertTrue(torch.equal(norm.weight, norm_weight / 8))
      self.assertTrue(torch.equal(model.model.embed_tokens.weight, table))
      self.assertTrue(torch.equal(head(norm(x)), want))  # exact: powers of 2
    self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

  def test_moves_ffn_norm_scale_into_w1_w3(self):
    model = Lfm3ForCausalLM()
    layer = model.model.layers[0]
    mlp = layer.feed_forward
    w1, w3 = (w.weight.detach().clone() for w in (mlp.w1, mlp.w3))
    norm_weight = layer.ffn_norm.weight.detach().clone()
    x = torch.randn(2, 3, 16) * 50
    want = mlp(g6.PlainRMSNorm(norm_weight, 1e-5)(x))
    with g6.patch_lfm3_model(model, _SPLIT):
      self.assertTrue(torch.equal(mlp.w1.weight, w1 * 4))
      self.assertTrue(torch.equal(mlp.w3.weight, w3 * 4))
      self.assertTrue(torch.equal(layer.ffn_norm.weight, norm_weight / 4))
      self.assertTrue(torch.equal(mlp(layer.ffn_norm(x)), want))  # exact: powers of 2
    self.assertTrue(torch.equal(mlp.w1.weight, w1))
    self.assertIsInstance(layer.ffn_norm, Lfm3RMSNorm)

  def test_refuses_meta_weights(self):
    # How export_lib loads a checkpoint for use_v2.
    with weights_loader.init_params_on_meta():
      model = Lfm3ForCausalLM()
    norm, head = model.model.embedding_norm, model.lm_head
    with self.assertRaisesRegex(ValueError, 'use_v2'):
      with g6.patch_lfm3_model(model, _SPLIT):
        pass
    self.assertIs(model.model.embedding_norm, norm)
    self.assertIs(model.lm_head, head)

  def test_rotate_half_by_matmul(self):
    x = torch.randn(1, 4, 3, 128)
    torch.testing.assert_close(
        g6.rotate_half_by_matmul(x), _rotate_half(x), rtol=0, atol=0
    )

  def test_plain_rms_norm_matches_rms_norm(self):
    weight = torch.nn.Parameter(torch.rand(64) + 0.5)
    norm = g6.PlainRMSNorm(weight, 1e-5)
    for scale in (1e-4, 1e-2, 1.0, 100.0):
      x = torch.randn(2, 3, 64) * scale
      want = weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
      torch.testing.assert_close(norm(x), want, rtol=1e-5, atol=1e-6)

  def test_real_slots_from_split_mask(self):
    builder = attention_mask.SplitAttentionMask(context_size=16)
    valid = torch.tensor([[True] * 5 + [False] * 3])
    mask = builder(torch.zeros((1, 8), dtype=torch.int32), torch.tensor(4), valid)
    self.assertTrue(torch.equal(g6._real_slots_from_split_mask(mask), valid))

  def test_real_slots_from_split_mask_with_litert_lm_masks(self):
    # LiteRT-LM's C++ masks hold -1e9 (float32) or -inf (float16), and their
    # padding rows see the real slots: only the diagonal marks a padding slot.
    valid = torch.tensor([[True] * 5 + [False] * 3])
    causal = torch.ones(8, 8, dtype=torch.bool).tril()
    for masked in (-1e9, float('-inf')):
      with self.subTest(masked=masked):
        block = torch.where(causal & valid, 0.0, masked)
        mask = torch.cat([torch.zeros(8, 4), block], -1)[None, None]
        self.assertTrue(
            torch.equal(g6._real_slots_from_split_mask(mask), valid)
        )

  def test_query_blocks_equal_one_block(self):
    torch.manual_seed(0)
    attention, s, head_dim = _Attention(), 40, 16
    caches = (torch.randn(1, 2, s, head_dim), torch.randn(1, 2, head_dim, s))
    # A full layer, a sliding layer with a cache longer than its window, and
    # one whose cache is its window, which takes all queries in one block.
    for window in (None, s // 2, s):
      for t in (32, 64, 128):
        with self.subTest(window=window, t=t):
          hidden = torch.randn(1, t, 64)
          angles = torch.arange(t)[None, :, None] * torch.rand(1, 1, head_dim)
          # 10 cached tokens visible, the last 5 of this call's slots padding.
          mask = attention_mask.SplitAttentionMask(context_size=s)(
              torch.zeros((1, t), dtype=torch.int32),
              torch.tensor(10),
              torch.arange(t)[None] < t - 5,
          )
          run = lambda: g6._attention_forward(
              attention,
              hidden,
              (angles.cos(), angles.sin()),
              mask,
              split_cache.LiteRTLMSplitCacheLayer(*caches),
              k_ts_idx=2,
              v_ts_idx=3,
          )[0]
          queries = []
          hook = attention.q_proj.register_forward_hook(
              lambda m, x, y: queries.append(x[0].shape[1])
          )
          attention.sliding_window = window
          with mock.patch.object(
              g6,
              '_query_block_attention_forward',
              wraps=g6._query_block_attention_forward,
          ) as blocked:
            blocks = run()
          with mock.patch.object(g6, '_QUERY_BLOCK', t):
            whole = run()  # The exporter's split-cache attention.
          hook.remove()
          size = t if window == s else 26
          self.assertEqual(blocked.call_count, 1)
          self.assertEqual(
              queries, [min(size, t - i) for i in range(0, t, size)] + [t]
          )
          # Bit for bit on Linux x86_64; up to 1.3e-7 apart on macOS arm64.
          torch.testing.assert_close(blocks, whole, rtol=0, atol=1e-6)

  def test_sliding_layer_floors_decode_mask(self):
    torch.manual_seed(0)
    attention, s, head_dim = _Attention(), 40, 16
    caches = (torch.randn(1, 2, s, head_dim), torch.randn(1, 2, head_dim, s))
    hidden, angles = torch.randn(1, 1, 64), torch.rand(1, 1, head_dim)
    # LiteRT-LM's decode mask: 10 cached tokens and the new one visible.
    mask = torch.full((1, 1, 1, s + 1), -1e9)
    mask[..., :10] = 0.0
    mask[..., -1] = 0.0
    seen = []

    def interface(module, query, key, value, attention_mask, **kwargs):
      seen.append(attention_mask)
      return split_cache_attention.split_cache_attention(
          module, query, key, value, attention_mask, **kwargs
      )

    outputs = []
    with mock.patch.dict(
        globals(),
        ALL_ATTENTION_FUNCTIONS=types.SimpleNamespace(
            get_interface=lambda *_: interface
        ),
    ):
      for window in (None, s):
        attention.sliding_window = window
        outputs.append(
            g6._attention_forward(
                attention,
                hidden,
                (angles.cos(), angles.sin()),
                mask,
                split_cache.LiteRTLMSplitCacheLayer(*caches),
                k_ts_idx=2,
                v_ts_idx=3,
            )[0]
        )
    # -3e4 is finite in the TPU's 16-bit floats; the float32 result is the same.
    self.assertTrue(torch.equal(seen[0], mask))
    self.assertTrue(torch.equal(seen[1], mask.clamp(min=-3e4)))
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)


class Lfm3G6ExtensionTest(ext_testing.ModelExtensionTestMixin,
                          absltest.TestCase):
  extension_cls = g6.Lfm3G6Extension

  def get_sample_export_config(self):
    return exportable_module_config.ExportableModuleConfig(
        model='dummy', split_cache=True)

  def get_sample_model(self):
    return Lfm3ForCausalLM()

  def _checkpoint(self):
    ckpt_dir = self.create_tempdir()
    ckpt_dir.create_file(
        'config.json', content=json.dumps({'model_type': 'lfm3'})
    )
    return ckpt_dir.full_path

  def test_split_cache_export_turns_on_handwritten_npu_components(self):
    plan = export.export(self._checkpoint(), split_cache=True, dry_run=True)

    self.assertEqual(plan.target, 'npu')
    for name in _NPU_SETTINGS:
      self.assertIs(plan.export_kwargs[name], True)
      self.assertEqual(plan.sources[name], export.SOURCE_MODEL)
    self.assertNotIn('use_hw_ple_for_npu', plan.export_kwargs)
    self.assertTrue(plan.export_kwargs['use_jinja_template'])

  def test_user_npu_setting_wins(self):
    plan = export.export(
        self._checkpoint(),
        split_cache=True,
        use_hw_masking_for_npu=False,
        dry_run=True,
    )

    self.assertIs(plan.export_kwargs['use_hw_masking_for_npu'], False)
    self.assertEqual(plan.sources['use_hw_masking_for_npu'], export.SOURCE_USER)
    self.assertIs(plan.export_kwargs['use_hw_cache_update_for_npu'], True)

  def test_ring_buffer_export_has_window_length_caches(self):
    config = transformers.PretrainedConfig()
    config.model_type = 'lfm3'
    config.num_hidden_layers = 3
    config.layer_types = ['conv', 'full_attention', 'sliding_attention']
    config.sliding_window = 256
    source_model_artifacts = export_lib.SourceModelArtifacts(
        model=types.SimpleNamespace(config=config),
        model_config=config,
        text_model_config=config,
        tokenizer=types.SimpleNamespace(),
    )
    export_config = exportable_module_config.ExportableModuleConfig(
        model='dummy',
        split_cache=True,
        cache_length=2048,
        sliding_window_ring_buffer_size=256,
        **self.get_extension().get_export_defaults({}).export_kwargs,
    )
    executor_metadata = metadata_builder.build_executor_metadata(
        source_model_artifacts,
        export_config,
        export_lib.ExportedModelArtifacts(),
    )

    llm_metadata = executor_metadata.llm_executor_metadata
    buffers = {b.prefill_input_name: b for b in llm_metadata.state_buffers}
    state_buffer = metadata_builder.executor_metadata_pb2.StateBuffer
    for name, kind in (
        ('kv_cache_k_2', state_buffer.TYPE_LOCAL_KEY_CACHE),
        ('kv_cache_v_2', state_buffer.TYPE_LOCAL_VALUE_CACHE),
    ):
      self.assertEqual(buffers[name].type, kind)
      self.assertEqual(buffers[name].minimum_sequence_length, 256)
      self.assertEqual(buffers[name].maximum_sequence_length, 256)
    self.assertEqual(
        buffers['kv_cache_k_1'].type, state_buffer.TYPE_GLOBAL_KEY_CACHE
    )
    self.assertEqual(buffers['kv_cache_k_1'].maximum_sequence_length, 2048)
    self.assertEqual(
        llm_metadata.attention_mask_settings.sliding_window_size, 256
    )
    npu_metadata = executor_metadata.llm_npu_executor_metadata
    for name in _NPU_SETTINGS:
      self.assertTrue(getattr(npu_metadata, name))
    self.assertFalse(npu_metadata.HasField('use_hw_ple_for_npu'))


if __name__ == '__main__':
  absltest.main()
