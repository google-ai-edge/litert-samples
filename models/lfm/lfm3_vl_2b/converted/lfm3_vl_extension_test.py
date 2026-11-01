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

"""Contract, parity and template tests for lfm3_vl_extension.py (CPU, no weights).

  python lfm3_vl_extension_test.py [--model=<dir or repo id>] [--revision=<commit>]

The checkpoint's code, configs, tokenizer and chat template (about 18 MB) are
the only download. The tiny model is a random LFM3-VL built from the
checkpoint's own remote code: a 4-layer vision tower (conv, attention, conv,
attention; hidden 64, 4 / 2 heads) and projector on a 2-layer LFM3 decoder.
The decoder side (lfm3_extension.py) has its own test in the LFM3-1.7B recipe.
"""

import functools
import os
import sys
import types

from absl import flags
from absl.testing import absltest
from huggingface_hub import snapshot_download
import numpy as np
from PIL import Image
import torch
import transformers

from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import sampler_params_pb2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'lfm3_1_7b', 'converted'))
import lfm3_extension  # pylint: disable=g-import-not-at-top
import lfm3_vl_extension  # pylint: disable=g-import-not-at-top

_MODEL = flags.DEFINE_string(
    'model', 'LiquidAI/LFM3-VL-2B', 'LFM3-VL-2B checkpoint dir or repo id.'
)
_REVISION = flags.DEFINE_string(
    'revision', 'f83d2b0a2296aaa277f394990a9b8b949ddf9691',
    'Checkpoint revision when --model is a repo id.'
)
_SOURCE_FILES = [
    'config.json', 'generation_config.json', 'configuration_lfm3_vl.py',
    'configuration_lfm3.py', 'modeling_lfm3_vl.py', 'modeling_lfm3.py',
    'lfm3_conv.py', 'image_processing_lfm3_vl.py', 'processing_lfm3_vl.py',
    'video_processing_lfm3_vl.py', 'processor_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'chat_template.jinja',
]
_EOS_ID = 124900
_SIZE = lfm3_vl_extension.IMAGE_SIZE


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


@functools.cache
def _image_processor():
  return transformers.AutoImageProcessor.from_pretrained(
      _source_dir(), trust_remote_code=True)


def _tiny_config():
  config = transformers.AutoConfig.from_pretrained(
      _source_dir(), trust_remote_code=True)
  vision = config.vision_config
  vision.hidden_size, vision.intermediate_size = 64, 96
  vision.num_attention_heads, vision.num_key_value_heads = 4, 2
  vision.layer_types = ['conv', 'full_attention', 'conv', 'full_attention']
  vision.num_hidden_layers = len(vision.layer_types)
  text = config.text_config
  text.vocab_size, text.hidden_size, text.intermediate_size = 512, 64, 96
  text.layer_types = ['conv', 'full_attention']
  text.num_hidden_layers = len(text.layer_types)
  text.num_attention_heads, text.num_key_value_heads, text.head_dim = 4, 2, 16
  config.pad_token_id = text.pad_token_id = 0  # Inside the tiny vocabulary.
  config.projector_hidden_size = 80
  for c in (config, vision, text):
    c.dtype = torch.float32
    c._attn_implementation = 'eager'  # pylint: disable=protected-access
  return config


@functools.cache
def _tiny_model(seed=0):
  torch.manual_seed(seed)
  model = transformers.AutoModelForImageTextToText.from_config(
      _tiny_config(), trust_remote_code=True, dtype=torch.float32).eval()
  # Non-trivial norm weights and biases (their init is ones / zeros).
  with torch.no_grad():
    for name, p in model.named_parameters():
      if name.endswith('norm.weight'):
        p.copy_(1.0 + 0.3 * torch.randn_like(p))
      elif name.endswith('bias'):
        p.copy_(0.1 * torch.randn_like(p))
  return model


def _picture(seed=0):
  """A random picture with flat regions (the input step's hard case)."""
  rng = np.random.default_rng(seed)
  u8 = rng.integers(0, 256, size=(_SIZE, _SIZE, 3), dtype=np.uint8)
  u8[: _SIZE // 4] = 128  # flat grey
  u8[_SIZE // 2:, : _SIZE // 3] = (12, 200, 77)  # flat colour
  return u8


def _ours(model, u8):
  """The two exportables on the runtime's input, float(u8) * (1 / 255)."""
  images = torch.from_numpy(u8.astype(np.float32) * np.float32(1 / 255))[None]
  config = exportable_module_config.ExportableModuleConfig(model='dummy')
  encoder = lfm3_vl_extension.Lfm3VlVisionEncoderExportable(model, config)
  adapter = lfm3_vl_extension.Lfm3VlVisionAdapterExportable(model, config, None)
  with torch.no_grad():
    features = encoder(images)['features']
    return features, adapter(features)['mm_embedding']


def _theirs(model, u8):
  """transformers' tower and projector, fed by the checkpoint's processor."""
  out = _image_processor()(
      images=[Image.fromarray(u8)], do_resize=False, return_tensors='pt')
  with torch.no_grad():
    image = model.model.get_image_features(
        out['pixel_values'], out['image_grid_thw'], return_dict=True)
  return out, torch.cat(image.pooler_output, dim=0)


class Lfm3VlExtensionTest(ext_testing.ModelExtensionTestMixin,
                          absltest.TestCase):
  extension_cls = lfm3_vl_extension.Lfm3VlExtension

  def test_tower_matches_transformers(self):
    """The rewritten tower (raster order, fixed grid, u8 input step) against
    transformers' tower (merge-block order, the processor's pixel values)."""
    model = _tiny_model()
    for seed in (0, 1):
      u8 = _picture(seed)
      _, ours = _ours(model, u8)
      processed, want = _theirs(model, u8)
      grid = _SIZE // 16
      self.assertEqual(processed['image_grid_thw'].tolist(), [[1, grid, grid]])
      self.assertEqual(tuple(ours.shape), (1, (grid // 2) ** 2, 64))
      delta = float((ours[0] - want).abs().max())
      scale = float(want.abs().max())
      print(f'tower seed {seed}: max |d| {delta:.2e} of max |x| {scale:.2e}')
      self.assertLessEqual(delta, 1e-5 * max(scale, 1.0))

  def test_input_step_is_exact(self):
    """floor(x * 255 + 0.5) gives back every u8 the runtime can send."""
    u8 = np.arange(256, dtype=np.float32)
    x = torch.from_numpy(u8 * np.float32(1 / 255))
    self.assertTrue(torch.equal(torch.floor(x * 255.0 + 0.5),
                                torch.from_numpy(u8)))

  def test_exportables_trace(self):
    model = _tiny_model()
    config = exportable_module_config.ExportableModuleConfig(model='dummy')
    encoder = lfm3_vl_extension.Lfm3VlVisionEncoderExportable(model, config)
    adapter = lfm3_vl_extension.Lfm3VlVisionAdapterExportable(model, config, None)
    for module in (encoder, adapter):
      for name, (inputs, dynamic) in module.get_sample_inputs(None).items():
        with self.subTest(signature=name):
          exported = torch.export.export(module, (), kwargs=inputs,
                                         dynamic_shapes=dynamic or None)
          self.assertIsNotNone(exported)
    names = (list(encoder.get_sample_inputs(None)),
             list(adapter.get_sample_inputs(None)))
    self.assertEqual(names, (['vision_encoder'], ['vision_adapter']))

  def test_composes_with_the_lfm3_decoder_extension(self):
    """The stock resolver: lfm3_vl (this file) + lfm3 (the text_config's)."""
    resolved = extension.resolve_model_extension(_tiny_config(), None)
    kinds = [type(e) for e in getattr(resolved, 'extensions', [resolved])]
    self.assertEqual(kinds, [lfm3_vl_extension.Lfm3VlExtension,
                             lfm3_extension.Lfm3Extension])

  def _builder_source(self):
    config = transformers.AutoConfig.from_pretrained(
        _source_dir(), trust_remote_code=True)
    model = types.SimpleNamespace(
        config=config,
        generation_config=transformers.GenerationConfig.from_pretrained(
            _source_dir()))
    return types.SimpleNamespace(model=model, model_config=config,
                                 tokenizer=_tokenizer())

  def _builder_config(self, **sampler_flags):
    return exportable_module_config.ExportableModuleConfig(
        model=_source_dir(), task='image_text_to_text', cache_length=2048,
        use_jinja_template=True, **sampler_flags)

  def test_builder_writes_the_bundle_metadata(self):
    """Both extensions inside litert-torch's own LlmMetadata builder.

    The generation config keeps `do_sample` on, as the stock loader leaves it
    for this task, so the builder writes a sampler and the hook removes it.
    """
    source = _tokenizer().chat_template
    meta = litert_lm_builder.build_llm_metadata(
        self._builder_source(), self._builder_config(), source, None)
    self.assertEqual([list(t.token_ids.ids) for t in meta.stop_tokens],
                     [[_EOS_ID]])
    self.assertEqual(meta.start_token.token_str, '<|startoftext|>')
    self.assertTrue(meta.llm_model_type.HasField('fast_vlm'))
    self.assertEqual((meta.llm_model_type.fast_vlm.image_tensor_height,
                      meta.llm_model_type.fast_vlm.image_tensor_width),
                     (_SIZE, _SIZE))
    self.assertFalse(meta.HasField('sampler_params'))
    self.assertEqual([(c.channel_name, c.start, c.end) for c in meta.channels],
                     [('thought', '<think>', '</think>')])
    tpl = meta.jinja_prompt_template
    self.assertEqual(tpl.count(lfm3_vl_extension.IMAGE_PART_BUNDLE), 2)
    self.assertNotIn(lfm3_vl_extension.IMAGE_PART_SOURCE, tpl)
    self.assertEqual(tpl.count(lfm3_extension.KEEP_THINKING_BUNDLE), 1)
    for left in ('.get(', 'generation -%}', '<|image|>'):
      self.assertNotIn(left, tpl)
    self.assertIn('<|video|>', tpl)  # Video parts stay as the source has them.

  def test_template_edit(self):
    source = _tokenizer().chat_template
    fixed = lfm3_vl_extension.image_soft_token_template(source)
    self.assertEqual(
        fixed.replace(lfm3_vl_extension.IMAGE_PART_BUNDLE,
                      lfm3_vl_extension.IMAGE_PART_SOURCE), source)
    self.assertEqual(lfm3_vl_extension.image_soft_token_template(fixed), fixed)
    with self.assertRaises(ValueError):
      lfm3_vl_extension.image_soft_token_template('{{ messages[0].content }}')
    with self.assertRaises(ValueError):  # one image part only
      lfm3_vl_extension.image_soft_token_template(
          source.replace(lfm3_vl_extension.IMAGE_PART_SOURCE, '""', 1))
    meta = llm_metadata_pb2.LlmMetadata()  # No template: nothing to edit.
    self.assertEqual(
        self.get_extension().build_llm_metadata(None, None, None, meta), meta)

  def test_template_renders_images_as_soft_tokens(self):
    """Same text as the source template, with <image_soft_token> where the
    source renders <|image|>, for one and three turns, thinking off and on."""
    source = _tokenizer().chat_template
    fixed = lfm3_vl_extension.image_soft_token_template(source)
    image = {'type': 'image'}
    text = lambda t: {'type': 'text', 'text': t}
    user = lambda *c: {'role': 'user', 'content': list(c)}
    reply = lambda c: {'role': 'assistant', 'content': c}
    one = [user(image, text('What is in this picture?'))]
    three = [user(image, text('Count the circles.')), reply('Five.'),
             user(text('And the squares?')), reply('Two.'),
             user(image, text('Compare it with this one.'))]
    for think in (False, True):
      for messages in (one, three):
        with self.subTest(think=think, turns=len(messages)):
          render = lambda t: _tokenizer().apply_chat_template(
              messages, chat_template=t, tokenize=False,
              add_generation_prompt=True, enable_thinking=think)
          want = render(source)
          self.assertIn('<|image|>', want)
          self.assertEqual(render(fixed),
                           want.replace('<|image|>', '<image_soft_token>'))

  def test_a_sampler_flag_keeps_the_sampler(self):
    meta = litert_lm_builder.build_llm_metadata(
        self._builder_source(), self._builder_config(sampler_top_p=1.0),
        _tokenizer().chat_template, None)
    self.assertEqual(meta.sampler_params.type,
                     sampler_params_pb2.SamplerParameters.TOP_P)
    self.assertEqual((meta.sampler_params.k, meta.sampler_params.p), (50, 1.0))
    self.assertAlmostEqual(meta.sampler_params.temperature, 0.1, places=6)

  def test_unsupported_vision_config_raises(self):
    config = _tiny_config()
    config.vision_config.rope_parameters = dict(
        config.vision_config.rope_parameters, rope_type='default')
    with self.assertRaises(ValueError):
      lfm3_vl_extension.vision_config(config)


if __name__ == '__main__':
  absltest.main()
