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

"""Contract and metadata tests for hy_mt2_extension.py (CPU, no weights).

  python hy_mt2_extension_test.py [--tokenizer=<dir or repo id>]

The tokenizer (Hy-MT2-1.8B's own, about 10 MB) is the only download; the
tiny model is a random 2-layer `hunyuan_v1_dense` built from a small config.
"""

import os
import sys
import types

from absl import flags
from absl.testing import absltest
import jinja2
import torch
import transformers

from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
from litert_lm_builder.runtime.proto import llm_metadata_pb2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hy_mt2_extension  # pylint: disable=g-import-not-at-top

_TOKENIZER = flags.DEFINE_string(
    'tokenizer', 'tencent/Hy-MT2-1.8B', 'Hy-MT2-1.8B tokenizer dir or repo id.'
)
_BOS_ID = 120000
_EOS_ID = 120020


def _tiny_config():
  return transformers.AutoConfig.for_model(
      'hunyuan_v1_dense',
      vocab_size=64,
      hidden_size=32,
      intermediate_size=64,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=8,
      dtype=torch.float32,
  )


class HyMt2ExtensionTest(ext_testing.ModelExtensionTestMixin, absltest.TestCase):
  extension_cls = hy_mt2_extension.HyMt2Extension

  def _export_config(self):
    return exportable_module_config.ExportableModuleConfig(
        model='dummy',
        prefill_lengths=[8],
        cache_length=16,
        cache_lengths=[16],
        externalize_embedder=True,
    )

  def test_tiny_model_parity_and_export(self):
    cfg = _tiny_config()
    model = transformers.AutoModelForCausalLM.from_config(
        cfg, dtype=torch.float32
    ).eval()
    self.assert_patch_numerical_parity(
        model,
        sample_kwargs={
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'use_cache': False,
        },
        export_config=self._export_config(),
    )
    exported = self.assert_tiny_prefill_decode_torch_export(
        model, model_config=cfg, export_config=self._export_config()
    )
    self.assertIn('prefill_8', exported)
    self.assertIn('decode', exported)

  def _build(self, llm_metadata):
    tokenizer = transformers.AutoTokenizer.from_pretrained(_TOKENIZER.value)
    source = types.SimpleNamespace(tokenizer=tokenizer)
    return self.get_extension().build_llm_metadata(
        source, self._export_config(), None, llm_metadata
    )

  def test_drops_start_token_and_adds_eos(self):
    meta = llm_metadata_pb2.LlmMetadata()
    meta.start_token.token_str = '<｜hy_begin▁of▁sentence｜>'
    meta.stop_tokens.add().token_str = '<｜hy_place▁holder▁no▁8｜>'
    meta = self._build(meta)
    self.assertFalse(meta.HasField('start_token'))
    ids = {i for t in meta.stop_tokens for i in t.token_ids.ids}
    self.assertIn(_EOS_ID, ids)

  def test_keeps_existing_eos_once(self):
    meta = llm_metadata_pb2.LlmMetadata()
    meta.stop_tokens.add().token_ids.ids.append(_EOS_ID)
    meta = self._build(meta)
    ids = [i for t in meta.stop_tokens for i in t.token_ids.ids]
    self.assertEqual(ids.count(_EOS_ID), 1)

  def _render(self, template, messages):
    env = jinja2.Environment(extensions=['jinja2.ext.loopcontrols'])
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=True)

  def test_template_reads_content_parts(self):
    vendor = transformers.AutoTokenizer.from_pretrained(
        _TOKENIZER.value).chat_template
    fixed = hy_mt2_extension.content_parts_template(vendor)
    turns = [('system', 'Be brief.'), ('user', 'Bonjour'),
             ('assistant', 'Hello'), ('user', 'Merci')]
    as_strings = [{'role': r, 'content': c} for r, c in turns]
    as_parts = [{'role': r, 'content': [{'type': 'text', 'text': c}]}
                for r, c in turns]
    expected = self._render(vendor, as_strings)
    self.assertEqual(self._render(fixed, as_strings), expected)
    self.assertEqual(self._render(fixed, as_parts), expected)
    self.assertNotIn('"type"', self._render(fixed, as_parts))

  def test_unknown_template_raises(self):
    with self.assertRaises(ValueError):
      hy_mt2_extension.content_parts_template('{{ messages[0].text }}')

  def test_hook_rewrites_the_embedded_template(self):
    vendor = transformers.AutoTokenizer.from_pretrained(
        _TOKENIZER.value).chat_template
    meta = llm_metadata_pb2.LlmMetadata()
    meta.jinja_prompt_template = vendor
    meta = self._build(meta)
    self.assertEqual(meta.jinja_prompt_template,
                     hy_mt2_extension.content_parts_template(vendor))

  def test_keeps_start_token_when_template_leaves_bos_to_runtime(self):
    tokenizer = transformers.AutoTokenizer.from_pretrained(_TOKENIZER.value)
    tokenizer.chat_template = tokenizer.chat_template.replace(
        '<｜hy_begin▁of▁sentence｜>', '{{ bos_token }}'
    )
    meta = llm_metadata_pb2.LlmMetadata()
    meta.start_token.token_ids.ids.append(_BOS_ID)
    meta = self.get_extension().build_llm_metadata(
        types.SimpleNamespace(tokenizer=tokenizer), self._export_config(),
        None, meta)
    self.assertTrue(meta.HasField('start_token'))


if __name__ == '__main__':
  absltest.main()
