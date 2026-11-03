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

"""Contract, parity and template tests for lfm3_audio_extension.py (CPU, no
weights).

  python lfm3_audio_extension_test.py [--model=<dir or repo id>] [--revision=<commit>]

The checkpoint's code, configs, tokenizer, feature extractor and chat
template (about 18 MB) are the only download. The tiny model is a random
LFM3-Audio built from the checkpoint's own remote code: a 4-layer speech
encoder (conv, attention, conv, attention; 16 conv channels, hidden 64,
4 / 2 heads, window 32) and projector on a 4-layer LFM3 decoder (conv,
sliding, conv, full; hidden 64). The decoder hooks (lfm3_extension.py) have
their own test in the LFM3-1.7B recipe; here they run as the export composes
them, under this file's `hidden` output.
"""

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

from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
from litert_lm_builder.runtime.proto import llm_metadata_pb2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'lfm3_1_7b', 'converted'))
import lfm3_audio_extension as audio_ext  # pylint: disable=g-import-not-at-top
import lfm3_extension  # pylint: disable=g-import-not-at-top

_MODEL = flags.DEFINE_string(
    'model', 'LiquidAI/LFM3-Audio-2B', 'LFM3-Audio-2B checkpoint dir or repo id.'
)
_REVISION = flags.DEFINE_string(
    'revision', '9d49701c372d6011d900f6abb40c6b296fb706dd',
    'Checkpoint revision when --model is a repo id.'
)
_SOURCE_FILES = [
    'config.json', 'generation_config.json', 'configuration_lfm3_audio.py',
    'configuration_lfm3.py', 'configuration_liquid_codec.py',
    'modeling_lfm3_audio.py', 'modeling_lfm3.py', 'modeling_liquid_codec.py',
    'lfm3_conv.py', 'feature_extraction_lfm3_audio.py',
    'processing_lfm3_audio.py', 'preprocessor_config.json',
    'processor_config.json', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'chat_template.jinja',
]
_STOP_IDS = [124900, 125014]  # <|im_end|>, <|text_end|>
_ROWS = 400  # A 4 s window keeps the test fast; the export uses 3,000.
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


@functools.cache
def _feature_extractor():
  return transformers.AutoFeatureExtractor.from_pretrained(
      _source_dir(), trust_remote_code=True)


def _tiny_config():
  config = transformers.AutoConfig.from_pretrained(
      _source_dir(), trust_remote_code=True)
  enc = config.audio_encoder_config
  enc.hidden_size, enc.conv_channels = 64, 16
  lfm = enc.text_config
  lfm.hidden_size, lfm.intermediate_size = 64, 96
  lfm.layer_types = ['conv', 'full_attention', 'conv', 'full_attention']
  lfm.num_hidden_layers = len(lfm.layer_types)
  lfm.num_attention_heads, lfm.num_key_value_heads = 4, 2
  text = config.text_config
  text.hidden_size, text.intermediate_size = 64, 96
  text.layer_types = ['conv', 'sliding_attention', 'conv', 'full_attention']
  text.num_hidden_layers = len(text.layer_types)
  text.num_attention_heads, text.num_key_value_heads, text.head_dim = 4, 2, 16
  text.sliding_window = 8
  depth = config.depth_config
  depth.hidden_size, depth.intermediate_size = 32, 48
  depth.num_attention_heads, depth.num_key_value_heads, depth.head_dim = 4, 2, 8
  for c in (config, enc, lfm, text, depth):
    c.dtype = torch.float32
    c._attn_implementation = 'eager'  # pylint: disable=protected-access
  return config


@functools.cache
def _tiny_model(seed=0):
  torch.manual_seed(seed)
  model = transformers.AutoModelForMultimodalLM.from_config(
      _tiny_config(), trust_remote_code=True, dtype=torch.float32).eval()
  # Non-trivial norm, conv and bias weights (their init is ones / small).
  with torch.no_grad():
    for name, p in model.named_parameters():
      if name.endswith('norm.weight') or name.endswith('layernorm.weight'):
        p.copy_(1.0 + 0.3 * torch.randn_like(p))
      elif name.endswith('conv.conv.weight'):
        p.copy_(0.5 * torch.randn_like(p))
      elif name.endswith('bias'):
        p.copy_(0.1 * torch.randn_like(p))
  return model


def _clip(seconds, seed):
  """Speech-like PCM: a few tones under noise, the last sample non-zero."""
  rng = np.random.default_rng(seed)
  t = np.arange(int(seconds * audio_ext.SAMPLE_RATE)) / audio_ext.SAMPLE_RATE
  x = sum(0.2 * np.sin(2 * np.pi * f * t) for f in (180.0, 520.0, 1330.0))
  x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)) + 0.02 * rng.standard_normal(t.size)
  x[: t.size // 10] *= 1e-3  # a quiet start
  x[-1] = 0.05
  return x.astype(np.float32)


def _window(pcm, rows=_ROWS):
  """The runtime's framing: one clip, zero tail, [1, rows, 160]."""
  buf = np.zeros(rows * audio_ext.HOP, np.float32)
  buf[: pcm.size] = pcm
  return torch.from_numpy(buf.reshape(1, rows, audio_ext.HOP))


def _runtime_mask(start, steps, rows, cache_length):
  mask = torch.full((1, 1, rows, cache_length), _MASK_FILL, dtype=torch.float32)
  for i in range(steps):
    mask[0, 0, i, : start + i + 1] = 0.0
  return mask


class Lfm3AudioExtensionTest(ext_testing.ModelExtensionTestMixin,
                             absltest.TestCase):
  extension_cls = audio_ext.Lfm3AudioExtension

  def test_audio_section_matches_transformers(self):
    """PCM through the section against the feature extractor + transformers'
    encoder and projector (the clip length found in the graph)."""
    model = _tiny_model()
    section = audio_ext.audio_section_from_model(model, rows=_ROWS)
    for seconds, seed in ((1.0, 0), (2.37, 1), (3.9, 2)):
      pcm = _clip(seconds, seed)
      fe = _feature_extractor()(pcm, sampling_rate=audio_ext.SAMPLE_RATE,
                                return_tensors='pt')
      with torch.no_grad():
        want = model.model.get_audio_features(
            fe['input_features'], fe['feature_attention_mask'])
        got = section(_window(pcm))
      n = int(got['mask'].sum())
      self.assertEqual(n, want.shape[0], f'{seconds} s: token count')
      self.assertTrue(bool((got['mask'][0, :n] == 1).all()))
      delta = float((got['features'][0, :n] - want).abs().max())
      scale = float(want.abs().max())
      print(f'audio {seconds} s: {n} tokens, max |d| {delta:.2e} of max |x|'
            f' {scale:.2e}')
      self.assertLessEqual(delta, 1e-5 * max(scale, 1.0))

  def test_log_mel_matches_the_feature_extractor(self):
    pcm = _clip(2.37, 1)
    fe = _feature_extractor()(pcm, sampling_rate=audio_ext.SAMPLE_RATE,
                              return_tensors='pt')
    want = fe['input_features'][0]
    frames = int(fe['feature_attention_mask'].sum())
    with torch.no_grad():
      mel, v1 = audio_ext.LogMel(_ROWS)(_window(pcm))
    self.assertEqual(int(v1.sum()), frames)
    delta = float((mel[0, :, :frames] - want[:, :frames]).abs().max())
    print(f'log-mel: {frames} frames, max |d| {delta:.2e}')
    self.assertLessEqual(delta, 1e-4)

  def test_feature_extractor_matches_the_section(self):
    audio_ext.check_feature_extractor(_feature_extractor())
    other = copy.copy(_feature_extractor())
    other.hop_length = 200
    with self.assertRaises(ValueError):
      audio_ext.check_feature_extractor(other)

  def test_decoder_hidden_and_logits_match_transformers(self):
    """Prefill and decode through the exportables (the 1.7B patch active)
    against transformers: `hidden` = the final norm output, `logits` its
    projection."""
    reference = _tiny_model()
    ids = torch.randint(4, 512, (21,),
                        generator=torch.Generator().manual_seed(1)).tolist()
    with torch.no_grad():
      want_hidden = reference.model.language_model(
          input_ids=torch.tensor([ids])).last_hidden_state[0]
      want_logits = reference(input_ids=torch.tensor([ids])).logits[0]
    model = copy.deepcopy(reference)
    model.set_attn_implementation('lrt_transposed_attention')  # as the export
    config = exportable_module_config.ExportableModuleConfig(
        model='dummy', cache_length=32, externalize_embedder=True,
        quantization_recipe='none')
    source = types.SimpleNamespace(model=model, model_config=model.config,
                                   text_model_config=model.config.text_config,
                                   tokenizer=None)
    prefill = audio_ext.Lfm3AudioPrefill(model, config, source)
    decode = audio_ext.Lfm3AudioDecode(model, config, source)
    emb = model.get_input_embeddings()
    with lfm3_extension.patch_lfm3_model(model, config), torch.no_grad():
      cache = prefill.get_sample_kv_cache(model.config.text_config)[0]['kv_cache']
      split = 13
      pos = torch.arange(split, dtype=torch.int32)
      out = prefill(emb(torch.tensor([ids[:split]])), pos, cache,
                    _runtime_mask(0, split, split, 32))
      hidden = [out['hidden'][0]]
      logits = []
      for p in range(split, len(ids)):
        step = decode(emb(torch.tensor([[ids[p]]])),
                      torch.tensor([p], dtype=torch.int32), cache,
                      _runtime_mask(p, 1, 1, 32))
        hidden.append(step['hidden'][0])
        logits.append(step['logits'][0])
    got_hidden = torch.cat(hidden)
    got_logits = torch.cat(logits)
    dh = float((got_hidden - want_hidden).abs().max())
    dl = float((got_logits - want_logits[split:]).abs().max())
    print(f'decoder: max |d hidden| {dh:.2e}, max |d logit| {dl:.2e}')
    self.assertLessEqual(dh, 1e-4)
    self.assertLessEqual(dl, 1e-4)
    self.assertTrue(torch.equal(got_logits.argmax(-1),
                                want_logits[split:].argmax(-1)))

  def test_exportables_trace(self):
    model = _tiny_model()
    config = exportable_module_config.ExportableModuleConfig(model='dummy')
    encoder = audio_ext.Lfm3AudioEncoderExportable(model, config)
    samples = encoder.get_sample_inputs(None,
                                        feature_extractor=_feature_extractor())
    self.assertEqual(list(samples), ['encode'])
    inputs, _ = samples['encode']
    self.assertEqual(tuple(inputs['audio'].shape), (1, 3000, 160))
    exported = torch.export.export(encoder, (), kwargs=inputs)
    outputs = exported.module()(**inputs)
    self.assertEqual(tuple(outputs['features'].shape), (1, 375, 64))
    self.assertEqual(outputs['mask'].dtype, torch.uint8)

  def test_composes_with_the_lfm3_decoder_extension(self):
    """The stock resolver: lfm3_audio (this file) + lfm3 (the text_config's)."""
    resolved = extension.resolve_model_extension(_tiny_config(), None)
    kinds = [type(e) for e in getattr(resolved, 'extensions', [resolved])]
    self.assertEqual(kinds, [audio_ext.Lfm3AudioExtension,
                             lfm3_extension.Lfm3Extension])
    config = exportable_module_config.ExportableModuleConfig(
        model='dummy', externalize_embedder=True)
    self.assertEqual(resolved.get_prefill_decode_exportables(None, config),
                     (audio_ext.Lfm3AudioPrefill, audio_ext.Lfm3AudioDecode))

  def _builder_source(self):
    config = transformers.AutoConfig.from_pretrained(
        _source_dir(), trust_remote_code=True)
    model = types.SimpleNamespace(
        config=config,
        generation_config=transformers.GenerationConfig.from_pretrained(
            _source_dir()))
    return types.SimpleNamespace(model=model, model_config=config,
                                 tokenizer=_tokenizer())

  def _builder_config(self, **kwargs):
    return exportable_module_config.ExportableModuleConfig(
        model=_source_dir(), task='multimodal_lm', cache_length=2048,
        use_jinja_template=True, export_vision_encoder=False, **kwargs)

  def test_builder_writes_the_bundle_metadata(self):
    """Both extensions inside litert-torch's own LlmMetadata builder."""
    source = _tokenizer().chat_template
    meta = litert_lm_builder.build_llm_metadata(
        self._builder_source(), self._builder_config(), source, None)
    self.assertEqual([list(t.token_ids.ids) for t in meta.stop_tokens],
                     [[i] for i in _STOP_IDS])
    self.assertEqual(meta.start_token.token_str, '<|startoftext|>')
    self.assertFalse(meta.HasField('sampler_params'))
    gm = meta.llm_model_type.generic_model
    self.assertTrue(meta.llm_model_type.HasField('generic_model'))
    self.assertEqual(
        (gm.audio_enabled, gm.delimiter_regex, gm.audio_token_regex,
         gm.skip_mel_spectrogram_extraction, gm.audio_sample_rate_hz,
         gm.audio_num_channels, gm.audio_frame_length, gm.audio_hop_length,
         gm.audio_input_scale, gm.add_audio_end),
        (True, r'(<\|audio\|>)', r'<\|audio\|>', True, 16000, 1, 160, 160,
         1.0, False))
    self.assertEqual([(c.channel_name, c.start, c.end) for c in meta.channels],
                     [('thought', '<think>', '</think>')])
    tpl = meta.jinja_prompt_template
    self.assertEqual(tpl.count(audio_ext.AUDIO_PART), 1)
    self.assertEqual(tpl.count(lfm3_extension.KEEP_THINKING_BUNDLE), 1)
    for left in ('.get(', 'generation -%}'):
      self.assertNotIn(left, tpl)

  def test_text_only_export_has_no_audio_fields(self):
    meta = litert_lm_builder.build_llm_metadata(
        self._builder_source(), self._builder_config(export_audio_encoder=False),
        _tokenizer().chat_template, None)
    self.assertTrue(meta.llm_model_type.HasField('generic_model'))
    self.assertFalse(meta.llm_model_type.generic_model.audio_enabled)

  def test_template_guard(self):
    source = _tokenizer().chat_template
    self.assertEqual(audio_ext.check_audio_template(source), source)
    with self.assertRaises(ValueError):
      audio_ext.check_audio_template(
          source.replace(audio_ext.AUDIO_PART, '"<|audio_in|>"'))
    meta = llm_metadata_pb2.LlmMetadata()  # No template: nothing to check.
    self.assertEqual(
        self.get_extension().build_llm_metadata(None, None, None, meta), meta)

  def test_template_renders_audio_as_the_source(self):
    """The bundle's template (both hooks applied) renders an audio question
    as the source does, one turn and three turns."""
    source = _tokenizer().chat_template
    meta = litert_lm_builder.build_llm_metadata(
        self._builder_source(), self._builder_config(), source, None)
    audio = {'type': 'audio'}
    text = lambda t: {'type': 'text', 'text': t}
    user = lambda *c: {'role': 'user', 'content': list(c)}
    reply = lambda c: {'role': 'assistant', 'content': c}
    system = {'role': 'system', 'content': 'Perform ASR in English.'}
    one = [system, user(audio)]
    three = [user(audio, text('What is said?')), reply('Hello.'),
             user(audio, text('And here?'))]
    for messages in (one, three):
      with self.subTest(turns=len(messages)):
        render = lambda t: _tokenizer().apply_chat_template(
            messages, chat_template=t, tokenize=False,
            add_generation_prompt=True)
        want = render(source)
        self.assertIn('<|audio|>', want)
        self.assertEqual(render(meta.jinja_prompt_template), want)


if __name__ == '__main__':
  absltest.main()
