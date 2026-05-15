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

"""Verifies a re-authored ASR model against the reference HF model."""

from absl import app
from absl import flags
from absl import logging
import librosa
import numpy as np
import torch

from compiled_model_api.speech_recognition.convert import asr_model
from compiled_model_api.speech_recognition.convert import supported_models

_MODEL = flags.DEFINE_enum(
    'model',
    next(iter(supported_models.SUPPORTED_MODELS)),
    supported_models.SUPPORTED_MODELS.keys(),
    'HF model ID to verify.',
)

_STATEFUL_AFTER = flags.DEFINE_integer(
    'stateful_after',
    -1,
    'If >= 0, model runs in stateful mode after this many tokens.',
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

_RTOL = flags.DEFINE_float(
    'rtol',
    1e-3,
    'Relative tolerance for comparing the reference and tflite models.',
)

_ATOL = flags.DEFINE_float(
    'atol',
    1e-3,
    'Absolute tolerance for comparing the reference and tflite models.',
)


def _decode(
    model: asr_model.AsrModel, encoder_outputs: tuple[torch.Tensor, ...]
) -> tuple[np.ndarray, np.ndarray]:
  """Decodes the encoder output with the re-authored model."""
  reauthored_logits = []
  reauthored_tokens = [model.get_decode_start_token_id()]
  while True:
    i = len(reauthored_tokens)
    decoder_inputs = list(model.get_decoder_sample_input(encoder_outputs, i))
    decoder_inputs[-2] = torch.IntTensor([reauthored_tokens])
    decoder_outputs = model.get_decoder()(*decoder_inputs)
    current_logits = decoder_outputs[0][0, i-1, :].detach().numpy()
    token_id = int(np.argmax(current_logits))
    if token_id == model.get_decode_stop_token_id():
      break
    logging.info('token_id: %s', token_id)
    reauthored_logits.append(current_logits)
    reauthored_tokens.append(token_id)

  return (
      # reauthored_logits is a list of np.array of shape [M].
      # Stack them to get shape [N, M], then expand dims to get [1, N, M].
      np.expand_dims(np.stack(reauthored_logits, axis=0), axis=0),
      np.array([reauthored_tokens[1:]]),
  )


def _stateful(num_tokens: int) -> bool:
  """Returns true if the model runs in stateful mode."""
  return _STATEFUL_AFTER.value >= 0 and num_tokens > _STATEFUL_AFTER.value


def _decode_tdt(
    model: asr_model.AsrModel, encoder_outputs: tuple[torch.Tensor, ...]
) -> tuple[np.ndarray, np.ndarray]:
  """Decodes the encoder output with the TDT re-authored model."""
  reauthored_logits = []
  reauthored_tokens = [model.get_decode_start_token_id()]

  max_time_idx = encoder_outputs[0].shape[-1]
  time_idx = 0
  decoder_inputs = list(model.get_decoder_sample_input(encoder_outputs, 1))
  while time_idx < max_time_idx:
    if _stateful(len(reauthored_tokens)):
      idx = 0
      decoder_inputs[1] = torch.IntTensor([[reauthored_tokens[-1]]])
    else:
      idx = len(reauthored_tokens) - 1
      decoder_inputs[1] = torch.IntTensor([reauthored_tokens])
    decoder_outputs = model.get_decoder()(*decoder_inputs)
    current_logits = decoder_outputs[0][0, time_idx, idx, :].detach().numpy()

    token_logits = current_logits[: -model.NUM_DURATIONS]
    token_id = int(np.argmax(token_logits))
    if token_id != model.BLANK_TOKEN_ID:
      logging.info('token_id: %s', token_id)
      reauthored_logits.append(token_logits)
      reauthored_tokens.append(token_id)

    duration_logits = current_logits[-model.NUM_DURATIONS :]
    skip = int(np.argmax(duration_logits))
    time_idx += 1 if skip == 0 and token_id == model.BLANK_TOKEN_ID else skip

    if _stateful(len(reauthored_tokens)):
      decoder_inputs[-2] = decoder_outputs[1].detach()
      decoder_inputs[-1] = decoder_outputs[2].detach()

  # Same shape of logits and tokens as _decode().
  return (
      np.expand_dims(np.stack(reauthored_logits, axis=0), axis=0),
      np.array([reauthored_tokens[1:]]),
  )


def main(_):
  model_id = _MODEL.value

  logging.info('Loading the reference model from %s...', model_id)
  model = supported_models.SUPPORTED_MODELS[model_id]()
  processor = model.get_processor()

  logging.info('Loading sample audio from %s...', _SAMPLE_AUDIO.value)
  if 'parakeet' in model_id:
    duration = _INPUT_SEC.value - 0.01  # Parakeet processor adds extra 10ms.
  else:
    duration = _INPUT_SEC.value
  sample, _ = librosa.load(
      _SAMPLE_AUDIO.value,
      sr=processor.get_sampling_rate(),
      duration=duration,
  )
  inputs = processor.process(sample)

  logging.info('Running the original model...')
  reference_output = model.run_original_model(inputs)
  logging.info(
      'Reference logits (shape=%s) [0, 0, :100]: %s',
      reference_output.logits.shape,
      reference_output.logits[0, 0, :100],
  )
  logging.info('Reference tokens: %s', reference_output.tokens)

  reference_text = processor.decode(reference_output.tokens)
  logging.info('Reference text: %s', reference_text[0])

  logging.info('Running the re-authored model...')
  encoder_inputs = model.get_encoder_sample_input(inputs)
  encoder_outputs = model.get_encoder()(*encoder_inputs)

  try:
    decode_fn = _decode_tdt if 'tdt' in model_id else _decode
    reauthored_logits, reauthored_tokens = decode_fn(model, encoder_outputs)
  except NotImplementedError:
    reauthored_logits = encoder_outputs[0].detach().numpy()
    reauthored_tokens = np.argmax(reauthored_logits, axis=-1)
  logging.info(
      'Re-authored logits (shape=%s) [0, 0, :100]: %s',
      reauthored_logits.shape,
      reauthored_logits[0, 0, :100],
  )
  logging.info('Re-authored tokens: %s', reauthored_tokens)

  reauthored_text = processor.decode(reauthored_tokens)
  logging.info('Re-authored text: %s', reauthored_text[0])

  assert np.allclose(
      reference_output.logits,
      reauthored_logits,
      rtol=_RTOL.value,
      atol=_ATOL.value,
  )
  logging.info('Logits match!')

  reference_tokens = reference_output.tokens[0].numpy()
  try:
    if (
        reference_tokens[0] != reauthored_tokens[0, 0]
        and reference_tokens[0] == model.get_decode_start_token_id()
    ):
      reference_tokens = reference_tokens[1:]
  except NotImplementedError:
    pass  # No decoder is found.
  assert np.equal(reference_tokens, reauthored_tokens[0]).all()
  logging.info('Tokens match!')

  assert reference_text == reauthored_text
  logging.info('Text matches!')


if __name__ == '__main__':
  app.run(main)
