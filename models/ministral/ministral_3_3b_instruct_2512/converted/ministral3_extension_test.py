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

"""Contract, loading and metadata tests for ministral3_extension.py (CPU).

  python ministral3_extension_test.py [--model=<dir or repo id>] \
      [--revision=<commit>]

No weights are downloaded: the checkpoint's config, tokenizer and chat
template (about 17 MB) are the only download. The tiny model is a random
2-layer Ministral3 decoder with the checkpoint's rotary settings; the loading
test writes it as a tiny checkpoint laid out as the real one (decoder weights
under `language_model.model.`, no output layer, a vision encoder next to it)
and loads it with the exporter's own `load_model`.
"""

import contextlib
import functools
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types

from absl import flags
from absl.testing import absltest
from huggingface_hub import snapshot_download
import safetensors.torch
import torch
import transformers

from litert_torch.generative.export_hf import export as export_lib
from litert_torch.generative.export_hf.core import export_lib as core_export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext import testing as ext_testing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ministral3_extension  # pylint: disable=g-import-not-at-top

_MODEL = flags.DEFINE_string(
    'model', 'mistralai/Ministral-3-3B-Instruct-2512-BF16',
    'Ministral-3-3B-Instruct-2512 BF16 checkpoint dir or repo id.'
)
_REVISION = flags.DEFINE_string(
    'revision', 'b6d637bef2393152b3da2b2fde72eecdee30557e',
    'Checkpoint revision when --model is a repo id.'
)
_TOKENIZER_FILES = [
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
    'chat_template.jinja',
]
_SOURCE_FILES = ['config.json', 'generation_config.json', *_TOKENIZER_FILES]
_EOS_ID = 2


@functools.cache
def _source_dir():
  """The checkpoint's small files: the --model directory or a download.

  Returns:
    The local directory that holds them.
  """
  if os.path.isdir(_MODEL.value):
    return _MODEL.value
  return snapshot_download(
      _MODEL.value, revision=_REVISION.value, allow_patterns=_SOURCE_FILES)


@functools.cache
def _tokenizer():
  """The checkpoint's tokenizer.

  Returns:
    The transformers tokenizer.
  """
  return transformers.AutoTokenizer.from_pretrained(_source_dir())


def _sha256(text):
  """The sha256 of a text.

  Args:
    text: the text, encoded as UTF-8.

  Returns:
    The hex digest.
  """
  return hashlib.sha256(text.encode('utf-8')).hexdigest()


@contextlib.contextmanager
def _stock_exporter():
  """The exporter with no Ministral3 extension (the import registers it)."""
  removed = [extension.unregister_model_extension(t)
             for t in ministral3_extension.Ministral3Extension.model_types]
  try:
    yield
  finally:
    restored = next((r for r in removed if r is not None), None)
    if restored is not None:
      extension.register_model_extension()(restored)  # Both model types.


def _tiny_text_config():
  """A 2-layer decoder with the checkpoint's own rotary settings."""
  text = transformers.AutoConfig.from_pretrained(_source_dir()).text_config
  config = text.__class__.from_dict({
      **text.to_dict(),
      'vocab_size': 512, 'hidden_size': 64, 'intermediate_size': 96,
      'num_hidden_layers': 2, 'num_attention_heads': 4,
      'num_key_value_heads': 2, 'head_dim': 16,
  })
  config._attn_implementation = 'eager'  # pylint: disable=protected-access
  return config


def _tiny_model(seed=0):
  """A randomly initialized tiny Ministral3 text model.

  Args:
    seed: the torch seed for the weights.

  Returns:
    The model in eval mode, its norm weights away from one.
  """
  torch.manual_seed(seed)
  model = transformers.Ministral3ForCausalLM(_tiny_text_config()).eval()
  with torch.no_grad():  # Non-trivial norm weights (their init is ones).
    for name, p in model.named_parameters():
      if name.endswith('norm.weight'):
        p.copy_(1.0 + 0.3 * torch.randn_like(p))
  return model


def _write_tiny_checkpoint(directory):
  """A tiny checkpoint in the layout of the BF16 repo; returns its decoder.

  Two bfloat16 shards and their index: the decoder under
  `language_model.model.` with no output layer (tied), and vision encoder and
  projector tensors that the export must not need. The config is the mistral3
  wrapper; the tokenizer files are the checkpoint's.

  Args:
    directory: An empty directory.

  Returns:
    The decoder's expected float32 state dict (bfloat16-rounded).
  """
  decoder = _tiny_model()
  text_state = {}
  expected = {}
  for name, tensor in decoder.state_dict().items():
    if name == 'lm_head.weight':
      continue  # Tied to the embedding table; the checkpoint has no copy.
    text_state['language_model.' + name] = tensor.to(torch.bfloat16)
    expected[name] = tensor.to(torch.bfloat16).to(torch.float32)
  other_state = {
      'vision_tower.patch_conv.weight': torch.randn(8, 3, 4, 4),
      'multi_modal_projector.linear_1.weight': torch.randn(64, 8),
  }
  other_state = {k: v.to(torch.bfloat16) for k, v in other_state.items()}
  weight_map = {}
  for i, state in enumerate((text_state, other_state), 1):
    shard = f'model-{i:05d}-of-00002.safetensors'
    safetensors.torch.save_file(state, os.path.join(directory, shard),
                                metadata={'format': 'pt'})
    weight_map.update({k: shard for k in state})
  with open(os.path.join(directory, 'model.safetensors.index.json'), 'w') as f:
    json.dump({'metadata': {}, 'weight_map': weight_map}, f)
  vision = transformers.PixtralVisionConfig(
      hidden_size=8, intermediate_size=16, num_hidden_layers=1,
      num_attention_heads=2, image_size=16, patch_size=4)
  transformers.Mistral3Config(
      text_config=_tiny_text_config().to_dict(),
      vision_config=vision.to_dict()).save_pretrained(directory)
  for name in _TOKENIZER_FILES:
    shutil.copy(os.path.join(_source_dir(), name), directory)
  return expected


class Ministral3ExtensionTest(
    ext_testing.ModelExtensionTestMixin, absltest.TestCase
):
  extension_cls = ministral3_extension.Ministral3Extension

  def get_sample_model_config(self):
    return _tiny_text_config()

  def get_sample_model(self):
    return _tiny_model()

  def test_tiny_model_export(self):
    model = _tiny_model()
    export_config = exportable_module_config.ExportableModuleConfig(
        model='dummy', prefill_lengths=[8], cache_length=16,
        cache_lengths=[16], externalize_embedder=True)
    exported = self.assert_tiny_prefill_decode_torch_export(
        model, model_config=model.config, export_config=export_config)
    self.assertIn('prefill_8', exported)
    self.assertIn('decode', exported)

  def test_load_model_reads_the_decoder_from_the_checkpoint(self):
    """The exporter's load_model on a checkpoint laid out as the real one."""
    with tempfile.TemporaryDirectory() as directory:
      expected = _write_tiny_checkpoint(directory)
      export_config = exportable_module_config.ExportableModuleConfig(
          model=directory, prefill_lengths=[8], cache_length=16)
      source = core_export_lib.load_model(directory, export_config)
    model = source.model
    self.assertIsInstance(model, transformers.Ministral3ForCausalLM)
    self.assertEqual(source.model_config.model_type, 'ministral3')
    self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
    state = model.state_dict()
    self.assertEqual(set(state) - {'lm_head.weight'}, set(expected))
    for name, tensor in expected.items():
      self.assertTrue(torch.equal(state[name], tensor), name)

  def test_export_defaults(self):
    """The published file's arguments win over the CPU/GPU preset."""
    plan = export_lib.export(model=_source_dir(), dry_run=True)
    kwargs = plan.export_kwargs
    self.assertEqual(plan.task, 'text_generation')
    self.assertEqual(list(kwargs['prefill_lengths']), [128])
    self.assertEqual(kwargs['cache_length'], 4096)
    self.assertEqual(kwargs['llm_metadata_max_num_tokens_override'], 4096)
    self.assertEqual(kwargs['quantization_recipe'],
                     ministral3_extension.QUANTIZATION_RECIPE)
    for key, value in (('externalize_embedder', True),
                       ('use_jinja_template', True),
                       ('apply_gpu_composites', False),
                       ('externalize_weights', False),
                       ('use_rope_composite', False)):
      self.assertEqual(kwargs[key], value, key)
    self.assertIsNone(
        self.get_extension().get_export_defaults({}, None, 'npu'))

  def test_tokenizer_section_is_the_checkpoint_file(self):
    ext = self.get_extension()
    config = exportable_module_config.ExportableModuleConfig(
        model=_source_dir())
    config = ext.update_export_config(config, None)
    self.assertEqual(config.tokenizer_path_override,
                     os.path.join(_source_dir(), 'tokenizer.json'))
    config.tokenizer_path_override = '/given/tokenizer.json'
    config = ext.update_export_config(config, None)
    self.assertEqual(config.tokenizer_path_override, '/given/tokenizer.json')

  def test_template_is_the_published_one(self):
    template = ministral3_extension.bundle_template(_tokenizer())
    self.assertEqual(_sha256(template),
                     ministral3_extension.PUBLISHED_TEMPLATE_SHA256)
    ministral3_extension.check_template(_tokenizer(), template)

  def test_content_parts_render_as_strings(self):
    tok = _tokenizer()
    template = ministral3_extension.bundle_template(tok)
    for messages in ministral3_extension.TEMPLATE_CHECKS:
      parts = [{'role': m['role'],
                'content': [{'type': 'text', 'text': m['content']}]}
               for m in messages]
      self.assertEqual(
          tok.apply_chat_template(parts, chat_template=template,
                                  tokenize=False, add_generation_prompt=True),
          tok.apply_chat_template(messages, chat_template=template,
                                  tokenize=False, add_generation_prompt=True))

  def test_changed_default_turn_stops(self):
    tok = transformers.AutoTokenizer.from_pretrained(_source_dir())
    tok.chat_template = '{{- "X" -}}' + tok.chat_template
    template = ministral3_extension.bundle_template(tok)
    with self.assertRaisesRegex(ValueError, 'checkpoint template changed'):
      ministral3_extension.check_template(tok, template)
    source = self._builder_source()
    source.tokenizer = tok
    with self.assertRaisesRegex(ValueError, 'checkpoint template changed'):
      litert_lm_builder.build_llm_metadata(
          source, self._builder_config(self.get_extension()),
          tok.chat_template, None)

  def test_changed_role_marker_stops(self):
    tok = _tokenizer()
    roles = dict(ministral3_extension.ROLES)
    roles['model'] = ('', '')
    template = ministral3_extension.bundle_template(tok, roles=roles)
    with self.assertRaisesRegex(ValueError, 'renders differently'):
      ministral3_extension.check_template(tok, template, _sha256(template))

  def test_changed_template_stops_before_the_conversion(self):
    """update_export_config checks the template right after the model loads."""
    with tempfile.TemporaryDirectory() as directory:
      for name in _SOURCE_FILES:
        if os.path.exists(os.path.join(_source_dir(), name)):
          shutil.copy(os.path.join(_source_dir(), name), directory)
      path = os.path.join(directory, 'chat_template.jinja')
      with open(path) as f:
        template = f.read()
      with open(path, 'w') as f:
        f.write('{{- "X" -}}' + template)
      config = exportable_module_config.ExportableModuleConfig(model=directory)
      with self.assertRaisesRegex(ValueError, 'checkpoint template changed'):
        self.get_extension().update_export_config(config, None)
      self.assertIsNone(config.tokenizer_path_override)

  def _builder_source(self):
    config = transformers.AutoConfig.from_pretrained(_source_dir()).text_config
    generation_config = transformers.GenerationConfig.from_pretrained(
        _source_dir())
    generation_config.do_sample = False  # As the exporter's load_model sets.
    model = types.SimpleNamespace(
        config=config, generation_config=generation_config)
    return types.SimpleNamespace(model=model, model_config=config,
                                 tokenizer=_tokenizer())

  def _builder_config(self, ext):
    return exportable_module_config.ExportableModuleConfig(
        model=_source_dir(), cache_length=4096,
        llm_metadata_max_num_tokens_override=4096, use_jinja_template=True,
        model_extension=ext)

  def test_builder_writes_the_published_metadata(self):
    """The hook inside litert-torch's own LlmMetadata builder."""
    tok = _tokenizer()
    meta = litert_lm_builder.build_llm_metadata(
        self._builder_source(), self._builder_config(self.get_extension()),
        tok.chat_template, None)
    self.assertEqual(meta.start_token.token_str, '<s>')
    self.assertEqual(
        [(list(t.token_ids.ids), t.token_str) for t in meta.stop_tokens],
        [([_EOS_ID], '')])
    self.assertFalse(meta.HasField('pad_token'))
    self.assertFalse(meta.HasField('sampler_params'))
    self.assertEqual(len(meta.channels), 0)
    for role, (prefix, suffix) in ministral3_extension.ROLES.items():
      field = getattr(meta.prompt_templates, role)
      self.assertEqual((field.prefix, field.suffix), (prefix, suffix), role)
    self.assertEqual(meta.max_num_tokens, 4096)
    self.assertTrue(meta.llm_model_type.HasField('generic_model'))
    self.assertEqual(_sha256(meta.jinja_prompt_template),
                     ministral3_extension.PUBLISHED_TEMPLATE_SHA256)

  def test_without_the_hook(self):
    """What the stock builder writes for this checkpoint (README table)."""
    tok = _tokenizer()
    with _stock_exporter():
      meta = litert_lm_builder.build_llm_metadata(
          self._builder_source(), self._builder_config(None),
          tok.chat_template, None)
    print('stock start_token:', meta.start_token, 'stops:',
          [list(t.token_ids.ids) or t.token_str for t in meta.stop_tokens])
    print('stock pad_token:', meta.pad_token, 'channels:',
          [(c.channel_name, c.start, c.end) for c in meta.channels])
    print('stock prompt_templates:', meta.prompt_templates)
    print('stock template sha256:', _sha256(meta.jinja_prompt_template))
    self.assertNotEqual(_sha256(meta.jinja_prompt_template),
                        ministral3_extension.PUBLISHED_TEMPLATE_SHA256)


if __name__ == '__main__':
  absltest.main()
