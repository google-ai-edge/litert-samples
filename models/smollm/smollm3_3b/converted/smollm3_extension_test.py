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

"""Contract and metadata tests for smollm3_extension.py (CPU, no weights).

  python smollm3_extension_test.py [--model=<dir or repo id>] \
      [--revision=<commit>]

The checkpoint's config, tokenizer and chat template (about 17 MB) are the
only download. The tiny model is a random 4-layer SmolLM3 (rotary embeddings
off in the 4th layer, as in every 4th layer of the checkpoint).
"""

import contextlib
import functools
import hashlib
import os
import shutil
import sys
import tempfile
import types

from absl import flags
from absl.testing import absltest
from huggingface_hub import snapshot_download
import torch
import transformers

from litert_torch.generative.export_hf import export as export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext import testing as ext_testing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smollm3_extension  # pylint: disable=g-import-not-at-top

_MODEL = flags.DEFINE_string(
    'model', 'HuggingFaceTB/SmolLM3-3B', 'SmolLM3-3B checkpoint dir or repo id.'
)
_REVISION = flags.DEFINE_string(
    'revision', 'a07cc9a04f16550a088caea529712d1d335b0ac1',
    'Checkpoint revision when --model is a repo id.'
)
_SOURCE_FILES = [
    'config.json', 'generation_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja',
]
_EOS_ID = 128012


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


def _tiny_config():
  """The checkpoint's config, cut down to 4 layers, the last one without RoPE.

  Returns:
    The transformers config.
  """
  config = transformers.AutoConfig.from_pretrained(
      _source_dir(), vocab_size=512, hidden_size=64, intermediate_size=96,
      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
      no_rope_layers=[1, 1, 1, 0], layer_types=['full_attention'] * 4,
      max_position_embeddings=4096, pad_token_id=3, bos_token_id=1,
      eos_token_id=2, dtype=torch.float32)
  config._attn_implementation = 'eager'  # pylint: disable=protected-access
  return config


def _tiny_model(seed=0):
  """A randomly initialized model of the tiny config.

  Args:
    seed: the torch seed for the weights.

  Returns:
    The model in eval mode.
  """
  torch.manual_seed(seed)
  return transformers.AutoModelForCausalLM.from_config(
      _tiny_config(), dtype=torch.float32).eval()


@contextlib.contextmanager
def _stock_exporter():
  """The exporter with no `smollm3` extension (the import registers it)."""
  removed = extension.unregister_model_extension('smollm3')
  try:
    yield
  finally:
    if removed is not None:
      extension.register_model_extension()(removed)


def _sha256(text):
  """The sha256 of a text.

  Args:
    text: the text, encoded as UTF-8.

  Returns:
    The hex digest.
  """
  return hashlib.sha256(text.encode('utf-8')).hexdigest()


class SmolLM3ExtensionTest(
    ext_testing.ModelExtensionTestMixin, absltest.TestCase
):
  extension_cls = smollm3_extension.SmolLM3Extension

  def get_sample_model_config(self):
    return _tiny_config()

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

  def test_export_defaults(self):
    """The published file's arguments win over the CPU/GPU preset."""
    plan = export_lib.export(model=_source_dir(), dry_run=True)
    kwargs = plan.export_kwargs
    self.assertEqual(plan.task, 'text_generation')
    self.assertEqual(list(kwargs['prefill_lengths']), [128])
    self.assertEqual(kwargs['cache_length'], 4096)
    self.assertEqual(kwargs['llm_metadata_max_num_tokens_override'], 4096)
    self.assertEqual(kwargs['quantization_recipe'],
                     smollm3_extension.QUANTIZATION_RECIPE)
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
    # Idempotent, and a path given by the caller stays.
    config.tokenizer_path_override = '/given/tokenizer.json'
    config = ext.update_export_config(config, None)
    self.assertEqual(config.tokenizer_path_override, '/given/tokenizer.json')

  def test_template_is_the_published_one(self):
    template = smollm3_extension.bundle_template(_tokenizer())
    self.assertEqual(_sha256(template),
                     smollm3_extension.PUBLISHED_TEMPLATE_SHA256)
    self.assertEqual(template.count('strftime_now('), 1)  # The date stays live.
    smollm3_extension.check_template(_tokenizer(), template)

  def test_content_parts_render_as_strings(self):
    tok = _tokenizer()
    template = smollm3_extension.bundle_template(tok)
    for messages in smollm3_extension.TEMPLATE_CHECKS:
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
    template = smollm3_extension.bundle_template(tok)
    with self.assertRaisesRegex(ValueError, 'checkpoint template changed'):
      smollm3_extension.check_template(tok, template)
    source = self._builder_source()
    source.tokenizer = tok
    with self.assertRaisesRegex(ValueError, 'checkpoint template changed'):
      litert_lm_builder.build_llm_metadata(
          source, self._builder_config(self.get_extension()),
          tok.chat_template, None)

  def test_changed_role_marker_stops(self):
    tok = _tokenizer()
    roles = dict(smollm3_extension.ROLES)
    roles['model'] = ('<|im_start|>assistant\n', '<|im_end|>')
    template = smollm3_extension.bundle_template(tok, roles=roles)
    with self.assertRaisesRegex(ValueError, 'renders differently'):
      smollm3_extension.check_template(tok, template, _sha256(template))

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
    config = transformers.AutoConfig.from_pretrained(_source_dir())
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
    self.assertEqual(
        [(list(t.token_ids.ids), t.token_str) for t in meta.stop_tokens],
        [([_EOS_ID], ''), ([], '<|im_end|>\n')])
    self.assertFalse(meta.HasField('start_token'))
    self.assertFalse(meta.HasField('pad_token'))
    self.assertFalse(meta.HasField('sampler_params'))
    for role, (prefix, suffix) in smollm3_extension.ROLES.items():
      field = getattr(meta.prompt_templates, role)
      self.assertEqual((field.prefix, field.suffix), (prefix, suffix), role)
    self.assertEqual([(c.channel_name, c.start, c.end) for c in meta.channels],
                     [('thought', '<think>\n', '\n</think>')])
    self.assertEqual(meta.max_num_tokens, 4096)
    self.assertTrue(meta.llm_model_type.HasField('generic_model'))
    self.assertEqual(_sha256(meta.jinja_prompt_template),
                     smollm3_extension.PUBLISHED_TEMPLATE_SHA256)

  def test_without_the_hook(self):
    """What the stock builder writes for this checkpoint (README table)."""
    tok = _tokenizer()
    with _stock_exporter():
      meta = litert_lm_builder.build_llm_metadata(
          self._builder_source(), self._builder_config(None),
          tok.chat_template, None)
    print('stock stops:',
          [list(t.token_ids.ids) or t.token_str for t in meta.stop_tokens])
    print('stock pad_token:', meta.pad_token, 'channels:',
          [(c.channel_name, c.start, c.end) for c in meta.channels])
    print('stock prompt_templates:', meta.prompt_templates)
    print('stock template sha256:', _sha256(meta.jinja_prompt_template))
    self.assertNotEqual(_sha256(meta.jinja_prompt_template),
                        smollm3_extension.PUBLISHED_TEMPLATE_SHA256)


if __name__ == '__main__':
  absltest.main()
