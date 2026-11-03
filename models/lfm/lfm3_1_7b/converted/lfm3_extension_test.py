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

"""Contract, parity and template tests for lfm3_extension.py (CPU, no weights).

  python lfm3_extension_test.py [--model=<dir or repo id>] [--revision=<commit>]

The checkpoint's code, config, tokenizer and chat template (about 18 MB) are
the only download. The tiny model is a random 6-layer LFM3 (conv, conv, full,
conv, sliding, conv; window 8) built from the checkpoint's own remote code.
"""

import contextlib
import copy
import functools
import os
import sys
import types

from absl import flags
from absl.testing import absltest
from huggingface_hub import snapshot_download
import numpy as np
import torch
import transformers

from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import sampler_params_pb2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lfm3_extension  # pylint: disable=g-import-not-at-top

_MODEL = flags.DEFINE_string(
    'model', 'LiquidAI/LFM3-1.7B', 'LFM3-1.7B checkpoint dir or repo id.'
)
_REVISION = flags.DEFINE_string(
    'revision', '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd',
    'Checkpoint revision when --model is a repo id.'
)
_SOURCE_FILES = [
    'config.json', 'generation_config.json', 'configuration_lfm3.py',
    'modeling_lfm3.py', 'lfm3_conv.py', 'tokenizer.json',
    'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja',
]
_EOS_ID = 124900
_LAYER_TYPES = [
    'conv', 'conv', 'full_attention', 'conv', 'sliding_attention', 'conv',
]
_WINDOW = 8
_PAD_ID = 3
# The float32 mask fill of LiteRT-LM's executor (v0.18.1 InitializeAttentionMask).
_MASK_FILL = -0.7 * float(np.finfo(np.float32).max)


@functools.cache
def _source_dir():
  if os.path.isdir(_MODEL.value):
    return _MODEL.value
  return snapshot_download(
      _MODEL.value, revision=_REVISION.value, allow_patterns=_SOURCE_FILES)


@functools.cache
def _tokenizer():
  return transformers.AutoTokenizer.from_pretrained(
      _source_dir(), trust_remote_code=True)


def _tiny_config():
  config = transformers.AutoConfig.from_pretrained(
      _source_dir(), trust_remote_code=True,
      vocab_size=512, hidden_size=64, intermediate_size=96,
      num_hidden_layers=len(_LAYER_TYPES), num_attention_heads=4,
      num_key_value_heads=2, head_dim=16, conv_L_cache=7,
      sliding_window=_WINDOW, layer_types=list(_LAYER_TYPES),
      max_position_embeddings=4096, pad_token_id=_PAD_ID, bos_token_id=1,
      eos_token_id=2, dtype=torch.float32)
  config._attn_implementation = 'eager'  # pylint: disable=protected-access
  return config


def _tiny_model(seed=0):
  torch.manual_seed(seed)
  model = transformers.AutoModelForCausalLM.from_config(
      _tiny_config(), trust_remote_code=True, dtype=torch.float32).eval()
  # Non-trivial norm and conv weights (their init is ones / small).
  with torch.no_grad():
    for name, p in model.named_parameters():
      if name.endswith('norm.weight'):
        p.copy_(1.0 + 0.3 * torch.randn_like(p))
      elif name.endswith('conv.conv.weight'):
        p.copy_(0.5 * torch.randn_like(p))
  return model


@contextlib.contextmanager
def _stock_exporter():
  """The exporter with no `lfm3` extension registered (import registers it)."""
  removed = extension.unregister_model_extension('lfm3')
  try:
    yield
  finally:
    if removed is not None:
      extension.register_model_extension()(removed)


def _runtime_mask(start, steps, rows, cache_length):
  """The runtime's float mask for one call: row i sees positions <= start+i."""
  mask = torch.full((1, 1, rows, cache_length), _MASK_FILL, dtype=torch.float32)
  for i in range(steps):
    mask[0, 0, i, : start + i + 1] = 0.0
  return mask


class _Runner:
  """The exporter's prefill / decode modules, fed as LiteRT-LM feeds the graph.

  A prefill call shorter than its signature gets position 0 and the pad token
  in the unused tail (with embeddings input, random noise there).
  """

  def __init__(self, model, export_config):
    self.model = model
    self.config = export_config
    self.embeddings = export_config.externalize_embedder
    if self.embeddings:
      prefill_cls = external_emb_module.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder
      decode_cls = external_emb_module.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder
    else:
      prefill_cls = exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill
      decode_cls = exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate
    source = types.SimpleNamespace(
        model=model, model_config=model.config,
        text_model_config=model.config, tokenizer=None)
    self.prefill_module = prefill_cls(model, export_config, source)
    self.decode_module = decode_cls(model, export_config, source)
    self.reset()

  def reset(self):
    inputs, _ = self.prefill_module.get_sample_kv_cache(self.model.config)
    self.cache = inputs['kv_cache']

  @torch.no_grad()
  def prefill(self, token_ids, start, pad_to=None):
    n = len(token_ids)
    rows = pad_to or n
    tokens = torch.full((1, rows), _PAD_ID, dtype=torch.int32)
    tokens[0, :n] = torch.tensor(token_ids, dtype=torch.int32)
    pos = torch.zeros(rows, dtype=torch.int32)
    pos[:n] = torch.arange(start, start + n, dtype=torch.int32)
    mask = _runtime_mask(start, n, rows, self.config.cache_length)
    module = self.prefill_module
    if self.embeddings:
      emb = self.model.get_input_embeddings()(tokens)
      if rows > n:
        emb[0, n:] = torch.randn_like(emb[0, n:]) * 3.0
      tok_arg, emb_arg = None, emb
    else:
      tok_arg, emb_arg = tokens, None
    inputs = module.adapt_inputs(
        tok_arg, emb_arg, pos, self.cache, mask, use_bool_mask=False,
        **module.composite_flags())
    inputs |= module.attention_kwargs()
    inputs['prefill_logits'] = True
    return module.model(**inputs).logits[0, :n].float()

  @torch.no_grad()
  def decode(self, token_id, pos):
    tokens = torch.tensor([[token_id]], dtype=torch.int32)
    mask = _runtime_mask(pos, 1, 1, self.config.cache_length)
    input_pos = torch.tensor([pos], dtype=torch.int32)
    first = (self.model.get_input_embeddings()(tokens) if self.embeddings
             else tokens)
    out = self.decode_module(first, input_pos, self.cache, mask)
    return out['logits'][0, 0].float()


def _runtime_paths(runner, ids, chunk, split):
  """Logits per path: one prefill call, token-by-token decode, prefill in
  padded `chunk` calls, and that prefill up to `split` then decode."""
  out = {}
  runner.reset()
  out['one_prefill'] = runner.prefill(ids, 0)
  runner.reset()
  out['decode'] = torch.stack([runner.decode(t, p) for p, t in enumerate(ids)])
  runner.reset()
  out['chunked'] = torch.cat([runner.prefill(ids[s:s + chunk], s, pad_to=chunk)
                              for s in range(0, len(ids), chunk)])
  runner.reset()
  parts = [runner.prefill(ids[s:min(s + chunk, split)], s, pad_to=chunk)
           for s in range(0, split, chunk)]
  parts.append(torch.stack([runner.decode(ids[p], p)
                            for p in range(split, len(ids))]))
  out['chunked_then_decode'] = torch.cat(parts)
  return out


class Lfm3ExtensionTest(ext_testing.ModelExtensionTestMixin, absltest.TestCase):
  extension_cls = lfm3_extension.Lfm3Extension

  def get_sample_model_config(self):
    return _tiny_config()

  def get_sample_model(self):
    return _tiny_model()

  def _export_config(self, externalize_embedder=True, cache_length=16):
    return exportable_module_config.ExportableModuleConfig(
        model='dummy',
        prefill_lengths=[8],
        cache_length=cache_length,
        cache_lengths=[cache_length],
        externalize_embedder=externalize_embedder,
    )

  def test_tiny_model_export(self):
    model = _tiny_model()
    exported = self.assert_tiny_prefill_decode_torch_export(
        model, model_config=model.config, export_config=self._export_config())
    self.assertIn('prefill_8', exported)
    self.assertIn('decode', exported)

  def test_patch_restores_the_model(self):
    """The LiteRT short conv needs the exporter's cache, so numerical parity
    is the runtime-path test below; this checks the exit."""
    model = _tiny_model()
    ids = torch.randint(4, 512, (1, 12), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
      before = model(ids).logits
    classes = [type(m) for m in model.modules()]
    with self.get_extension().patch_model(model, self._export_config()):
      self.assertEqual(sum(isinstance(m, lfm3_extension.Lfm3LiteRTShortConv)
                           for m in model.modules()),
                       _LAYER_TYPES.count('conv'))
    self.assertEqual([type(m) for m in model.modules()], classes)
    self.assertFalse(any('forward' in vars(m) for m in model.modules()))
    with torch.no_grad():
      self.assertTrue(torch.equal(model(ids).logits, before))

  def test_runtime_paths_match_transformers(self):
    """41 tokens across the window, the conv state across calls, a padded
    prefill call whose state the decode reads, for token and embeddings input."""
    reference = _tiny_model()
    ids = torch.randint(4, 512, (41,), generator=torch.Generator().manual_seed(1)).tolist()
    with torch.no_grad():
      want = reference(torch.tensor([ids])).logits[0].float()
    top2 = torch.topk(want, 2, dim=-1).values
    clear = (top2[:, 0] - top2[:, 1]) >= 1e-3  # leave out near ties
    for embeddings in (False, True):
      model = copy.deepcopy(reference)
      model.config._attn_implementation = 'lrt_transposed_attention'  # pylint: disable=protected-access
      config = exportable_module_config.ExportableModuleConfig(
          model='dummy', cache_length=64, externalize_embedder=embeddings,
          quantization_recipe='none')
      with self.get_extension().patch_model(model, config):
        paths = _runtime_paths(_Runner(model, config), ids, chunk=16, split=29)
      for name, got in paths.items():
        delta = float((want - got).abs()[clear].max())
        agree = int((want.argmax(-1) == got.argmax(-1))[clear].sum())
        print(f'{"embeddings" if embeddings else "tokens"} {name}: max |dlogit|'
              f' {delta:.2e}, top-1 {agree}/{int(clear.sum())}')
        with self.subTest(embeddings=embeddings, path=name):
          self.assertTrue(bool(torch.isfinite(got).all()))
          self.assertLessEqual(delta, 1e-4)
          self.assertEqual(agree, int(clear.sum()))

  def _source_template(self):
    return _tokenizer().chat_template

  def _exporter_rewrite(self, template):
    """The template as the exporter embeds it, before the extension's hook."""
    with _stock_exporter():
      meta = litert_lm_builder.build_llm_metadata(
          self._builder_source(), self._builder_config(None), template, None)
    return meta.jinja_prompt_template

  def _builder_source(self):
    config = transformers.AutoConfig.from_pretrained(
        _source_dir(), trust_remote_code=True)
    model = types.SimpleNamespace(
        config=config,
        generation_config=transformers.GenerationConfig.from_pretrained(
            _source_dir()))
    return types.SimpleNamespace(model=model, model_config=config,
                                 tokenizer=_tokenizer())

  def _builder_config(self, extension):
    return exportable_module_config.ExportableModuleConfig(
        model=_source_dir(), cache_length=2048, sampler_top_p=1.0,
        use_jinja_template=True, model_extension=extension)

  def test_builder_writes_the_bundle_metadata(self):
    """The hook inside litert-torch's own LlmMetadata builder."""
    meta = litert_lm_builder.build_llm_metadata(
        self._builder_source(), self._builder_config(self.get_extension()),
        self._source_template(), None)
    self.assertEqual([list(t.token_ids.ids) for t in meta.stop_tokens],
                     [[_EOS_ID]])
    self.assertEqual(meta.start_token.token_str, '<|startoftext|>')
    self.assertEqual(meta.pad_token.token_str, '<|pad|>')
    self.assertEqual(meta.sampler_params.type,
                     sampler_params_pb2.SamplerParameters.TOP_P)
    self.assertEqual((meta.sampler_params.k, meta.sampler_params.p),
                     (50, 1.0))
    self.assertAlmostEqual(meta.sampler_params.temperature, 0.1, places=6)
    self.assertTrue(meta.llm_model_type.HasField('generic_model'))
    self.assertEqual([(c.channel_name, c.start, c.end) for c in meta.channels],
                     [('thought', '<think>', '</think>')])
    self.assertEqual(
        meta.jinja_prompt_template,
        lfm3_extension.history_stable_template(
            self._exporter_rewrite(self._source_template())))

  def test_without_the_hook_the_stops_include_strings(self):
    with _stock_exporter():
      meta = litert_lm_builder.build_llm_metadata(
          self._builder_source(), self._builder_config(None),
          self._source_template(), None)
    self.assertTrue(any(t.token_str for t in meta.stop_tokens))
    self.assertIn(lfm3_extension.KEEP_THINKING_SOURCE,
                  meta.jinja_prompt_template)

  def test_template_edit_is_one_line(self):
    embedded = self._exporter_rewrite(self._source_template())
    fixed = lfm3_extension.history_stable_template(embedded)
    self.assertEqual(fixed.count(lfm3_extension.KEEP_THINKING_BUNDLE), 1)
    self.assertNotIn(lfm3_extension.KEEP_THINKING_SOURCE, fixed)
    self.assertEqual(
        fixed.replace(lfm3_extension.KEEP_THINKING_BUNDLE,
                      lfm3_extension.KEEP_THINKING_SOURCE), embedded)
    self.assertEqual(lfm3_extension.history_stable_template(fixed), fixed)

  def test_flags_without_an_lfm3_form_stop(self):
    for flag in ('use_qkv_norm_rope_composite', 'fuse_qkv', 'split_cache'):
      with self.subTest(flag=flag):
        config = self._export_config()
        setattr(config, flag, True)
        with self.assertRaises(NotImplementedError):
          with self.get_extension().patch_model(_tiny_model(), config):
            pass

  def test_rope_composite_is_turned_off(self):
    config = self._export_config()
    self.assertTrue(config.use_rope_composite)  # the exporter's default
    config = self.get_extension().update_export_config(config, _tiny_config())
    self.assertFalse(config.use_rope_composite)

  def test_unknown_template_raises(self):
    with self.assertRaises(ValueError):
      lfm3_extension.history_stable_template('{{ messages[0].content }}')
    embedded = self._exporter_rewrite(self._source_template())
    with self.assertRaises(ValueError):  # the line twice
      lfm3_extension.history_stable_template(
          embedded + lfm3_extension.KEEP_THINKING_SOURCE)
    with self.assertRaises(ValueError):  # the exporter's rewrite not applied
      lfm3_extension.history_stable_template(self._source_template())

  def _render(self, template, messages, think, tools=None):
    return _tokenizer().apply_chat_template(
        messages, chat_template=template, tools=tools, tokenize=False,
        add_generation_prompt=True, enable_thinking=think)

  def test_template_renders_as_the_source(self):
    """Same text as the source template while the last message is the user's;
    in a tool loop only the thought of the turn after it is dropped."""
    fixed = lfm3_extension.history_stable_template(
        self._exporter_rewrite(self._source_template()))
    user = lambda t: {'role': 'user', 'content': t}
    reply = lambda c, r: {'role': 'assistant', 'content': c,
                          'reasoning_content': r}
    three = [user('Remember TANGERINE-7319.'), reply('Noted.', 'A code word.'),
             user('What is 12 + 30?'), reply('42.', 'Addition.'),
             user('What was the code word?')]
    for think in (False, True):
      for messages in ([user('What is 17 + 25?')], three):
        with self.subTest(think=think, turns=len(messages)):
          self.assertEqual(self._render(fixed, messages, think),
                           self._render(self._source_template(), messages, think))
    tool = {'type': 'function', 'function': {
        'name': 'lookup', 'description': 'Reads a value.',
        'parameters': {'type': 'object',
                       'properties': {'key': {'type': 'string'}},
                       'required': ['key']}}}
    call = ('<tool_call>\n<function=lookup>\n<parameter=key>\nalpha\n'
            '</parameter>\n</function>\n</tool_call>')
    loop = [user('Read the value under alpha.'),
            reply(call, 'I should call the tool.'),
            {'role': 'tool', 'content': '48213'}]
    for think in (False, True):
      with self.subTest(think=think, turns='tool'):
        got = self._render(fixed, loop, think, [tool])
        want = self._render(self._source_template(), loop, think, [tool])
        self.assertEqual(got, want.replace('I should call the tool.', ''))

  def test_stops_come_from_the_generation_config(self):
    model = types.SimpleNamespace(
        generation_config=types.SimpleNamespace(eos_token_id=[7, 9]))
    meta = llm_metadata_pb2.LlmMetadata()
    meta.stop_tokens.add().token_str = '<|im_end|>'
    meta = self.get_extension().build_llm_metadata(
        types.SimpleNamespace(model=model, tokenizer=None), None, None, meta)
    self.assertEqual([list(t.token_ids.ids) for t in meta.stop_tokens],
                     [[7], [9]])


if __name__ == '__main__':
  absltest.main()
