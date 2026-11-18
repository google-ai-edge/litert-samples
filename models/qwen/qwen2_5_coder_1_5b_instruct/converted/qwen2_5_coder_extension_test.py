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

"""Contract, export-plan, metadata and template tests for
qwen2_5_coder_extension.py.

  python qwen2_5_coder_extension_test.py [--model=<dir or repo id>]
      [--revision=<commit>]

CPU, no weights: the checkpoint's config, generation config and tokenizer
(about 11 MB) are the only download. The tiny model is a random 2-layer Qwen2
built from the checkpoint's config.
"""

import contextlib
import copy
import functools
import json
import os
import random
import sys
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
from litert_torch.generative.export_hf.model_ext.qwen3 import (
    extension as builtin_qwen)
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_quantizer import recipe as recipe_lib
from litert_quantizer import recipe_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# pylint: disable-next=g-import-not-at-top
import qwen2_5_coder_extension as coder_ext

_MODEL = flags.DEFINE_string(
    'model', 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
    'Qwen2.5-Coder-1.5B-Instruct checkpoint dir or repo id.')
_REVISION = flags.DEFINE_string(
    'revision', '2e1fd397ee46e1388853d2af2c993145b0f1098a',
    'Checkpoint revision when --model is a repo id.')
_SOURCE_FILES = [
    'config.json', 'generation_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'vocab.json', 'merges.txt',
]
# The 14 text stops litert-torch's builder derives from the template's end of
# turn (alone and after each of ` .,?!:;"')]}*`), as in the published file.
_TEXT_STOPS = sorted(
    [p + '<|im_end|>\n' for p in ' .,?!:;"\')]}*'] + ['<|im_end|>\n'])


@functools.cache
def _source_dir():
  """The checkpoint's files: --model as a dir, or downloaded at --revision."""
  if os.path.isdir(_MODEL.value):
    return _MODEL.value
  return snapshot_download(
      _MODEL.value, revision=_REVISION.value, allow_patterns=_SOURCE_FILES)


@functools.cache
def _tokenizer():
  """The checkpoint's tokenizer."""
  return transformers.AutoTokenizer.from_pretrained(_source_dir())


def _config_dict():
  """The checkpoint's config.json as a dict."""
  with open(os.path.join(_source_dir(), 'config.json')) as f:
    return json.load(f)


def _tiny_config():
  """The checkpoint's config with 2 layers and small sizes, in float32."""
  return transformers.AutoConfig.from_pretrained(
      _source_dir(), vocab_size=512, hidden_size=64, intermediate_size=128,
      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
      max_position_embeddings=256, dtype=torch.float32)


def _tiny_model(seed=0):
  """A random Qwen2 of `_tiny_config()`, in eval mode.

  Args:
    seed: The torch seed for its random weights.
  """
  torch.manual_seed(seed)
  return transformers.AutoModelForCausalLM.from_config(
      _tiny_config(), dtype=torch.float32).eval()


@contextlib.contextmanager
def _registered(qwen2_extension):
  """The exporter with `qwen2_extension` (None: no extension) for qwen2.

  Args:
    qwen2_extension: The `ModelExtension` class to register for `qwen2`, or
      None for none.
  """
  ours = extension.unregister_model_extension('qwen2')
  if qwen2_extension is not None:
    extension.register_model_extension()(qwen2_extension)
  try:
    yield
  finally:
    extension.unregister_model_extension('qwen2')
    if ours is not None:
      extension.register_model_extension()(ours)


def _builder_source(tokenizer=None):
  """What litert-torch's packager hands the metadata builder.

  Args:
    tokenizer: The tokenizer to hand on; None for the checkpoint's.
  """
  config = transformers.AutoConfig.from_pretrained(_source_dir())
  generation = transformers.GenerationConfig.from_pretrained(_source_dir())
  generation.do_sample = False  # export_lib.load_model turns sampling off.
  model = types.SimpleNamespace(config=config, generation_config=generation)
  return types.SimpleNamespace(model=model, model_config=config,
                               tokenizer=tokenizer or _tokenizer())


def _build_metadata(tokenizer=None, jinja=False):
  """litert-torch's own LlmMetadata builder, as the packager calls it.

  Args:
    tokenizer: The tokenizer to build from; None for the checkpoint's.
    jinja: Whether to pass the Jinja template instead of per-role templates.
  """
  tokenizer = tokenizer or _tokenizer()
  config = exportable_module_config.ExportableModuleConfig(
      model=_source_dir(), cache_length=4096, use_jinja_template=jinja)
  templates = (tokenizer.chat_template if jinja
               else litert_lm_builder.parse_chat_template(tokenizer))
  return litert_lm_builder.build_llm_metadata(
      _builder_source(tokenizer), config, templates, None)


def _changed_tokenizer(old, new):
  """A copy of the checkpoint's tokenizer with its chat template changed.

  Args:
    old: The text to replace in the chat template; it must be there.
    new: The text to put in its place.
  """
  tokenizer = copy.deepcopy(_tokenizer())
  if old not in tokenizer.chat_template:
    raise AssertionError(f'{old!r} not in the source template')
  tokenizer.chat_template = tokenizer.chat_template.replace(old, new)
  return tokenizer


def _plain(recipe):
  """A recipe as JSON values (enums as their values).

  Args:
    recipe: The quantization recipe, a list of rule dicts.
  """
  return json.loads(json.dumps(
      recipe, default=lambda o: getattr(o, 'value', str(o))))


class Qwen25CoderExtensionTest(ext_testing.ModelExtensionTestMixin,
                               absltest.TestCase):
  extension_cls = coder_ext.Qwen25CoderExtension

  def get_sample_model_config(self):
    return _tiny_config()

  def get_sample_model(self):
    return _tiny_model()

  def test_tiny_model_export(self):
    """prefill / decode trace with the recipe's own export arguments."""
    kwargs = dict(self.get_extension().get_export_defaults(
        _config_dict()).export_kwargs)
    kwargs.update(prefill_lengths=[8], cache_length=16, cache_lengths=[16])
    kwargs.pop('quantization_recipe')
    config = exportable_module_config.ExportableModuleConfig(
        model='dummy', **kwargs)
    self.assertTrue(config.externalize_embedder)
    self.assertFalse(config.apply_gpu_composites)
    self.assertFalse(config.use_rope_composite)
    model = _tiny_model()
    exported = self.assert_tiny_prefill_decode_torch_export(
        model, model_config=model.config, export_config=config)
    self.assertIn('prefill_8', exported)
    self.assertIn('decode', exported)

  def test_export_plan(self):
    """The export arguments come from the extension, over the target's."""
    plan = export_lib.export(model=_source_dir(), dry_run=True,
                             extension_modules=coder_ext.__file__)
    want = {
        'quantization_recipe': coder_ext.QUANTIZATION_RECIPE,
        'externalize_embedder': True,
        'prefill_lengths': [1024, 256, 64, 16, 4, 1],
        'cache_length': 4096,
        'llm_metadata_max_num_tokens_override': None,
        'use_jinja_template': False,
        'apply_gpu_composites': False,
        'use_rope_composite': False,
        'externalize_weights': False,
    }
    for key, value in want.items():
      with self.subTest(key=key):
        self.assertEqual(plan.export_kwargs[key], value)
        self.assertEqual(plan.sources[key], export_lib.SOURCE_MODEL)
    self.assertEqual(plan.model_type, 'qwen2')
    self.assertEqual(plan.task, 'text_generation')
    self.assertNotIn('litert_lm_llm_metadata_override', plan.export_kwargs)
    self.assertEqual(sorted(set(plan.export_kwargs) - set(want)),
                     ['extension_modules'])

  def test_export_plan_without_the_extension(self):
    """What the exporter does with this checkpoint without the recipe."""
    with _registered(builtin_qwen.Qwen2Extension):
      plan = export_lib.export(model=_source_dir(), dry_run=True)
    self.assertEqual(plan.export_kwargs['litert_lm_llm_metadata_override'],
                     'qwen2_5')
    self.assertTrue(plan.export_kwargs['use_jinja_template'])
    self.assertTrue(plan.export_kwargs['apply_gpu_composites'])
    self.assertTrue(plan.export_kwargs['externalize_weights'])
    self.assertEqual(plan.export_kwargs['cache_length'], 32771)
    self.assertEqual(plan.export_kwargs['prefill_lengths'], [128, 1024])
    self.assertNotIn('quantization_recipe', plan.export_kwargs)
    print(plan)

  def test_builder_writes_the_bundle_metadata(self):
    """The hook inside litert-torch's own LlmMetadata builder."""
    meta = _build_metadata()
    t = meta.prompt_templates
    self.assertEqual(
        (t.user.prefix, t.user.suffix),
        (coder_ext.DEFAULT_SYSTEM_TURN + '<|im_start|>user\n', '<|im_end|>\n'))
    self.assertEqual((t.system.prefix, t.system.suffix),
                     ('<|im_start|>system\n', '<|im_end|>\n'))
    self.assertEqual((t.model.prefix, t.model.suffix),
                     ('<|im_start|>assistant\n', '<|im_end|>\n'))
    self.assertEqual(
        [list(s.token_ids.ids) for s in meta.stop_tokens if s.token_ids.ids],
        [[151643], [151645]])
    self.assertEqual([s.token_str for s in meta.stop_tokens if s.token_str],
                     _TEXT_STOPS)
    self.assertFalse(meta.HasField('start_token'))
    self.assertEqual(meta.pad_token.token_str, '<|endoftext|>')
    self.assertTrue(meta.llm_model_type.HasField('qwen2p5'))
    self.assertEqual(meta.max_num_tokens, 4096)
    self.assertFalse(meta.HasField('sampler_params'))
    self.assertEqual(meta.jinja_prompt_template, '')
    self.assertEqual(len(meta.channels), 0)

  def test_without_the_hook_the_default_system_turn_is_lost(self):
    """The trap the recipe exists for: the parser reads the template with a
    system message present, so the default turn never reaches the file."""
    with _registered(None):
      meta = _build_metadata()
    self.assertEqual(meta.prompt_templates.user.prefix, '<|im_start|>user\n')

  def test_rendered_prompt_equals_the_source_template(self):
    """One user turn through the bundle's prefixes = the source template."""
    t = _build_metadata().prompt_templates
    prompt = 'Write fib(n).'
    bundle = t.user.prefix + prompt + t.user.suffix + t.model.prefix
    source = _tokenizer().apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False,
        add_generation_prompt=True)
    self.assertEqual(bundle, source)

  def test_changed_template_stops_the_export(self):
    for old, new in (
        ('You are Qwen, created by Alibaba Cloud.', 'You are a coder.'),
        ("{%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen",
         "{%- elif false %}\n        {{- '<|im_start|>system\\nYou are Qwen"),
    ):
      with self.subTest(new=new):
        with self.assertRaisesRegex(ValueError, 'source template changed'):
          _build_metadata(_changed_tokenizer(old, new))

  def test_jinja_template_is_not_this_recipe(self):
    with self.assertRaisesRegex(ValueError, 'per-role ChatML'):
      _build_metadata(jinja=True)

  def test_hook_runs_twice_to_the_same_metadata(self):
    """An explicit and a registered copy of the extension both run it."""
    once = _build_metadata()
    twice = self.get_extension().build_llm_metadata(
        _builder_source(), None, None, copy.deepcopy(once))
    self.assertEqual(twice.SerializeToString(), once.SerializeToString())

  def test_stops_take_a_fixed_order(self):
    meta = _build_metadata()
    want = meta.SerializeToString()
    for seed in range(5):
      stops = list(meta.stop_tokens)
      random.Random(seed).shuffle(stops)
      shuffled = copy.deepcopy(meta)
      del shuffled.stop_tokens[:]
      shuffled.stop_tokens.extend(stops)
      coder_ext.stops_in_fixed_order(shuffled)
      self.assertEqual(shuffled.SerializeToString(), want)

  def test_quantization_recipe_is_the_282_recipe(self):
    """recipe_int4.json = the recipe build_qwen2_5_coder_1_5b.py of #282
    registered in Python: OCTAV blockwise-32 int4 on every op, then min-max
    int8 per channel on EMBEDDING_LOOKUP."""
    int4_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
    int4_rule['algorithm_key'] = recipe_lib.AlgorithmName.OCTAV
    int4_rule['op_config']['weight_tensor_config']['granularity'] = (
        'BLOCKWISE_32')
    emb_rule = copy.deepcopy(recipe_lib.dynamic_wi4_afp32()[0])
    emb_rule['operation'] = 'EMBEDDING_LOOKUP'
    emb_rule['op_config']['weight_tensor_config']['num_bits'] = 8
    with open(coder_ext.QUANTIZATION_RECIPE) as f:
      shipped = json.load(f)
    self.assertEqual(shipped, _plain([int4_rule, emb_rule]))
    manager = recipe_manager.RecipeManager()
    manager.load_quantization_recipe(shipped)
    self.assertEqual(_plain(manager.get_quantization_recipe()), shipped)

  def test_export_defaults_refuse_other_checkpoints(self):
    ext = self.get_extension()
    other = dict(_config_dict(), hidden_size=2048, num_hidden_layers=36)
    with self.assertRaisesRegex(ValueError, 'recipe for'):
      ext.get_export_defaults(other)
    with self.assertRaisesRegex(ValueError, 'not this recipe'):
      ext.get_export_defaults(_config_dict(), target='npu')

  def test_builder_without_a_tokenizer_is_left_alone(self):
    meta = llm_metadata_pb2.LlmMetadata()
    meta.prompt_templates.user.prefix = 'x'
    out = self.get_extension().build_llm_metadata(
        types.SimpleNamespace(tokenizer=None), None, None, copy.deepcopy(meta))
    self.assertEqual(out, meta)


if __name__ == '__main__':
  absltest.main()
