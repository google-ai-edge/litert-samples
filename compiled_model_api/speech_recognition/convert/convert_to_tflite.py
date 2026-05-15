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

"""Converts an open ASR model to tflite model."""

import math

from absl import app
from absl import flags
from absl import logging
import librosa
import litert_torch
from litert_torch.generative.quantize import quant_attrs
from litert_torch.generative.quantize import quant_recipe
from litert_torch.quantize import quant_config

from compiled_model_api.speech_recognition.convert import supported_models

_MODEL = flags.DEFINE_enum(
    'model',
    next(iter(supported_models.SUPPORTED_MODELS)),
    supported_models.SUPPORTED_MODELS.keys(),
    'HF model ID to convert.',
)

_STATEFUL_AFTER = flags.DEFINE_integer(
    'stateful_after',
    -1,
    'If >= 0, the model runs in stateful mode after this many tokens.',
)

_INPUT_SEC = flags.DEFINE_float(
    'input_sec',
    1.0,
    'Input audio length in seconds.',
)

_SAMPLE_AUDIO = flags.DEFINE_string(
    'sample_audio',
    '',
    'Input path for the sample audio file.',
)

_OUTPUT = flags.DEFINE_string(
    'output',
    '/tmp/parakeet_ctc_0.6b.tflite',
    'Output path for the tflite model.',
)

_QUANT = flags.DEFINE_enum(
    'quant',
    'drq',
    ['none', 'drq'],
    'Quantization option.',
)


def num_decode_tokens() -> int:
  """Derives the number of decoder tokens from the input seconds.

  Gets the smallest multiple of 32 of the input seconds multiplied by 8 assuming
  tokens would not be generated more than 8 tokens per second.

  Returns:
    The decoder token size.
  """
  num_tokens = math.ceil(_INPUT_SEC.value * 8 / 32) * 32
  if _STATEFUL_AFTER.value >= 0 and _STATEFUL_AFTER.value < num_tokens:
    return _STATEFUL_AFTER.value if _STATEFUL_AFTER.value > 0 else 1
  return num_tokens


def main(_):
  logging.info('Loading model from %s...', _MODEL.value)
  model = supported_models.SUPPORTED_MODELS[_MODEL.value](
      override_transformers=True  # Override transformers for GPU inference.
  )
  processor = model.get_processor()

  logging.info('Loading sample audio from %s...', _SAMPLE_AUDIO.value)
  if 'parakeet' in _MODEL.value:
    duration = _INPUT_SEC.value - 0.01  # Parakeet processor adds extra 10ms.
  else:
    duration = _INPUT_SEC.value
  sample, rate = librosa.load(
      _SAMPLE_AUDIO.value,
      sr=processor.get_sampling_rate(),
      duration=duration,
  )
  inputs = processor.process(sample)
  encoder_inputs = model.get_encoder_sample_input(inputs)
  logging.info(
      'Rate=%d, Inputs=%s', rate, [(i.size(), i) for i in encoder_inputs]
  )

  logging.info('Converting model to tflite...')
  if _QUANT.value == 'drq':
    recipe = quant_recipe.LayerQuantRecipe(
        activation_dtype=quant_attrs.Dtype.FP32,
        weight_dtype=quant_attrs.Dtype.INT8,
        mode=quant_attrs.Mode.DYNAMIC_RANGE,
        algorithm=quant_attrs.Algorithm.MIN_MAX,
        granularity=quant_attrs.Granularity.CHANNELWISE,
    )
    gen_recipe = quant_recipe.GenerativeQuantRecipe(default=recipe)
    q_config = quant_config.QuantConfig(generative_recipe=gen_recipe)
  elif _QUANT.value == 'none':
    q_config = None
  else:
    raise app.UsageError(f'Unsupported quantization: {_QUANT.value}')

  encoder = model.get_encoder()
  edge_model = litert_torch.signature('encode', encoder, encoder_inputs)

  try:
    encoder_output = encoder(*encoder_inputs)
    decoder = model.get_decoder()
    decoder_inputs = model.get_decoder_sample_input(
        encoder_output, num_decode_tokens()
    )
    edge_model = edge_model.signature('decode', decoder, decoder_inputs)
    # Stateful model like parakeet TDT needs another decoder with one token.
    if _STATEFUL_AFTER.value > 1:
      decoder_inputs_1 = model.get_decoder_sample_input(encoder_output, 1)
      edge_model = edge_model.signature('decode_1', decoder, decoder_inputs_1)
  except NotImplementedError:
    logging.debug('No decoder found. Skipping decode subgraph.')

  edge_model = edge_model.convert(enable_x64=False, quant_config=q_config)

  logging.info('Exporting model to %s...', _OUTPUT.value)
  edge_model.export(_OUTPUT.value)


if __name__ == '__main__':
  app.run(main)
