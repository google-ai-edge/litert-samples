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

"""Verifies an open ASR tflite model against the reference HF model."""

import math

from absl import app
from absl import flags
from absl import logging
import librosa
import numpy as np

from ai_edge_torch.compiled_model import CompiledModel
from ai_edge_torch.hardware_accelerator import HardwareAccelerator
from ai_edge_torch.tensor_buffer import TensorBuffer
from compiled_mode_api.speech_recognition.convert import asr_model
from compiled_mode_api.speech_recognition.convert import supported_models

_REFERENCE_MODEL = flags.DEFINE_enum(
    'reference_model',
    next(iter(supported_models.SUPPORTED_MODELS)),
    supported_models.SUPPORTED_MODELS.keys(),
    'HF model ID as the reference for verification.',
)

_TFLITE_MODEL = flags.DEFINE_string(
    'tflite_model',
    '',
    'Path to the tflite model to verify against the reference model.',
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

_BACKEND = flags.DEFINE_string(
    'backend',
    'cpu',
    'Backend to run the tflite model, cpu or gpu.',
)

_NUM_DECODE_STEPS = flags.DEFINE_integer(
    'num_decode_steps',
    -1,
    'Number of decode steps to run. If -1, run until the stop token.',
)

_SKIP_REFERENCE = flags.DEFINE_bool(
    'skip_reference',
    False,
    'If true, skip running the reference model and not compare the results.',
)


def _read_buffer(buffer: TensorBuffer) -> np.ndarray:
  """Reads the tensor buffer into numpy array of the correct shape and dtype."""
  details = buffer.get_tensor_details()
  shape = details['shape']
  num_elements = math.prod(shape)
  return buffer.read(num_elements, details['dtype']).reshape(shape)


def _clear_buffer(buffer: TensorBuffer):
  """Clears the tensor buffer."""
  details = buffer.get_tensor_details()
  num_elements = math.prod(details['shape'])
  buffer.write(np.zeros(num_elements, details['dtype']))


def _get_num_elements(buffer: TensorBuffer) -> int:
  """Returns the number of elements in the tensor buffer."""
  details = buffer.get_tensor_details()
  return math.prod(details['shape'])


def _decode(
    model: asr_model.AsrModel,
    litert_model: CompiledModel,
    encoder_output_buffers: list[TensorBuffer],
) -> tuple[np.ndarray, np.ndarray]:
  """Decodes the encoder output with the tflite model."""
  decoder_index = litert_model.get_signature_index('decode')
  if decoder_index == -1:
    raise NotImplementedError('No decoder signature found.')

  decoder_input_buffers = litert_model.create_input_buffers(decoder_index)
  decoder_output_buffers = litert_model.create_output_buffers(decoder_index)

  # Pass (zero-copy) the encoder output to the decoder as input.
  for i, buffer in enumerate(encoder_output_buffers):
    decoder_input_buffers[i] = buffer

  tflite_logits = []
  tflite_tokens = [model.get_decode_start_token_id()]
  num_tokens = _get_num_elements(decoder_input_buffers[-2])
  # num_masks may not be the same as num_tokens, e.g. Qwen3-ASR.
  num_masks_squared = _get_num_elements(decoder_input_buffers[-1])
  num_masks = int(math.sqrt(num_masks_squared))
  causal_mask = asr_model.get_causal_mask(num_masks).numpy()

  num_decode_steps = num_tokens - 1
  if _NUM_DECODE_STEPS.value > 0 and _NUM_DECODE_STEPS.value < num_decode_steps:
    num_decode_steps = _NUM_DECODE_STEPS.value
  for i in range(num_decode_steps):
    input_tokens = np.zeros((1, num_tokens), dtype=np.int32)
    input_tokens[0, : len(tflite_tokens)] = tflite_tokens
    decoder_input_buffers[-2].write(input_tokens)
    decoder_input_buffers[-1].write(causal_mask)
    litert_model.run_by_index(
        decoder_index, decoder_input_buffers, decoder_output_buffers
    )
    decoder_output = _read_buffer(decoder_output_buffers[0])
    current_logits = decoder_output[0, i, :]
    token_id = int(np.argmax(current_logits))
    if token_id == model.get_decode_stop_token_id():
      break
    logging.info('token_id: %s', token_id)
    tflite_logits.append(current_logits)
    tflite_tokens.append(token_id)

  return (
      # tflite_logits is a list of np.array of shape [M].
      # Stack them to get shape [N, M], then expand dims to get [1, N, M].
      np.expand_dims(np.stack(tflite_logits, axis=0), axis=0),
      np.array([tflite_tokens[1:]]),
  )


def _decode_tdt(
    model: asr_model.AsrModel,
    litert_model: CompiledModel,
    encoder_output_buffers: list[TensorBuffer],
) -> tuple[np.ndarray, np.ndarray]:
  """Decodes the encoder output with the TDT tflite model."""
  decoder_index = litert_model.get_signature_index('decode')
  if decoder_index == -1:
    raise NotImplementedError('No decoder signature found.')

  decoder_input_buffers = litert_model.create_input_buffers(decoder_index)
  decoder_output_buffers = litert_model.create_output_buffers(decoder_index)

  decode_1_index = litert_model.get_signature_index('decode_1')
  if decode_1_index != -1:
    decode_1_input_buffers = litert_model.create_input_buffers(decode_1_index)
    decode_1_output_buffers = litert_model.create_output_buffers(decode_1_index)

  # Pass (zero-copy) the encoder output to the decoder as input.
  for i, buffer in enumerate(encoder_output_buffers):
    decoder_input_buffers[i] = buffer
    if decode_1_index != -1:
      decode_1_input_buffers[i] = buffer

  tflite_logits = []
  tflite_tokens = [model.get_decode_start_token_id()]
  num_tokens = _get_num_elements(decoder_input_buffers[1])
  max_num_tokens = (
      num_tokens if decode_1_index == -1 else int(_INPUT_SEC.value * 8)
  )
  if _NUM_DECODE_STEPS.value > 0 and _NUM_DECODE_STEPS.value < max_num_tokens:
    max_num_tokens = _NUM_DECODE_STEPS.value

  max_time_idx = encoder_output_buffers[0].get_tensor_details()['shape'][-1]
  time_idx = 0
  input_tokens = np.zeros((1, num_tokens), dtype=np.int32)
  _clear_buffer(decoder_input_buffers[-2])
  _clear_buffer(decoder_input_buffers[-1])
  while time_idx < max_time_idx:
    if decode_1_index != -1 and len(tflite_tokens) > num_tokens:
      idx = 0  # inference with decode_1
    else:
      idx = len(tflite_tokens) - 1  # inference with decode
    input_tokens[0, idx] = tflite_tokens[-1]
    decoder_input_buffers[1].write(input_tokens)
    litert_model.run_by_index(
        decoder_index, decoder_input_buffers, decoder_output_buffers
    )
    decoder_output = _read_buffer(decoder_output_buffers[0])
    current_logits = decoder_output[0, time_idx, idx, :]

    token_logits = current_logits[: -model.NUM_DURATIONS]
    token_id = int(np.argmax(token_logits))
    if token_id != model.BLANK_TOKEN_ID:
      logging.info('token_id: %s', token_id)
      tflite_logits.append(token_logits)
      tflite_tokens.append(token_id)
      if len(tflite_tokens) == max_num_tokens:
        break

    duration_logits = current_logits[-model.NUM_DURATIONS :]
    skip = int(np.argmax(duration_logits))
    time_idx += 1 if skip == 0 and token_id == model.BLANK_TOKEN_ID else skip

    if decode_1_index != -1 and len(tflite_tokens) > num_tokens:
      if decoder_index != decode_1_index:  # Switch to decode_1.
        decoder_index = decode_1_index
        decode_1_input_buffers[-2] = decoder_output_buffers[-2]
        decode_1_input_buffers[-1] = decoder_output_buffers[-1]
        decoder_input_buffers = decode_1_input_buffers
        decoder_output_buffers = decode_1_output_buffers
        input_tokens = np.zeros((1, 1), dtype=np.int32)
      else:  # Swap input/output states of decode_1.
        decoder_input_buffers[-2], decoder_output_buffers[-2] = (
            decoder_output_buffers[-2], decoder_input_buffers[-2]
        )
        decoder_input_buffers[-1], decoder_output_buffers[-1] = (
            decoder_output_buffers[-1], decoder_input_buffers[-1]
        )

  # Same shape of logits and tokens as _decode().
  return (
      np.expand_dims(np.stack(tflite_logits, axis=0), axis=0),
      np.array([tflite_tokens[1:]]),
  )


def main(_):
  reference_model_id = _REFERENCE_MODEL.value

  logging.info('Loading the reference model from %s...', reference_model_id)
  model = supported_models.SUPPORTED_MODELS[reference_model_id]()
  processor = model.get_processor()

  logging.info('Loading sample audio from %s...', _SAMPLE_AUDIO.value)
  if 'parakeet' in reference_model_id:
    duration = _INPUT_SEC.value - 0.01  # Parakeet processor adds extra 10ms.
  else:
    duration = _INPUT_SEC.value
  sample, _ = librosa.load(
      _SAMPLE_AUDIO.value,
      sr=processor.get_sampling_rate(),
      duration=duration,
  )
  inputs = processor.process(sample)

  if not _SKIP_REFERENCE.value:
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

  logging.info('Loading the tflite model from %s...', _TFLITE_MODEL.value)
  hardware_accel = HardwareAccelerator.CPU
  if _BACKEND.value == 'gpu':
    hardware_accel |= HardwareAccelerator.GPU
  litert_model = CompiledModel.from_file(_TFLITE_MODEL.value, hardware_accel)
  signatures = litert_model.get_signature_list()
  for k, v in signatures.items():
    logging.info('Signature %s: I=%s, O=%s', k, v['inputs'], v['outputs'])

  logging.info('Running the tflite model...')
  encoder_index = litert_model.get_signature_index('encode')
  encoder_input_buffers = litert_model.create_input_buffers(encoder_index)
  encoder_output_buffers = litert_model.create_output_buffers(encoder_index)

  encoder_inputs = model.get_encoder_sample_input(inputs)
  input_features = encoder_inputs[0].numpy()
  logging.info(
      'TFLite input_features (shape=%s)\n[0, :100]: %s\n[-1, -100:]: %s',
      input_features.shape,
      input_features[0, :100],
      input_features[-1, -100:],
  )
  encoder_input_buffers[0].write(input_features)
  litert_model.run_by_index(
      encoder_index, encoder_input_buffers, encoder_output_buffers
  )

  try:
    decode_fn = _decode_tdt if 'tdt' in reference_model_id else _decode
    tflite_logits, tflite_tokens = decode_fn(
        model, litert_model, encoder_output_buffers
    )
  except NotImplementedError:
    tflite_logits = _read_buffer(encoder_output_buffers[0])
    tflite_tokens = np.argmax(tflite_logits, axis=-1)
  logging.info(
      'TFLite logits (shape=%s) [0, 0, :100]: %s',
      tflite_logits.shape,
      tflite_logits[0, 0, :100],
  )
  logging.info('TFLite tokens: %s', tflite_tokens)

  tflite_text = processor.decode(tflite_tokens)
  logging.info('TFLite text: %s', tflite_text[0])

  if _SKIP_REFERENCE.value:
    return

  assert np.allclose(
      reference_output.logits,
      tflite_logits,
      rtol=_RTOL.value,
      atol=_ATOL.value,
  )
  logging.info('Logits match!')

  reference_tokens = reference_output.tokens[0].numpy()
  try:
    if (
        reference_tokens[0] != tflite_tokens[0, 0]
        and reference_tokens[0] == model.get_decode_start_token_id()
    ):
      reference_tokens = reference_tokens[1:]
  except NotImplementedError:
    pass  # No decoder is found.
  assert np.equal(reference_tokens, tflite_tokens[0]).all()
  logging.info('Tokens match!')

  assert reference_text == tflite_text
  logging.info('Text matches!')


if __name__ == '__main__':
  app.run(main)
