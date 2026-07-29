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

"""Compile a tflite model for NPUs."""

import os

from absl import app
from absl import flags
from absl import logging

from ai_edge_litert.aot import aot_compile as aot
from ai_edge_litert.aot.vendors.mediatek import target as mtk_target
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target

_MODEL = flags.DEFINE_string(
    'tflite_model',
    '',
    'tflite model path to compile for Qualcomm and/or MediaTek NPUs.',
)

_OUTPUT = flags.DEFINE_string(
    'output',
    '/tmp/asr_npu',
    'Output path for the compiled model for NPUs.',
)

_TARGETS = flags.DEFINE_list(
    'targets',
    [],
    'Targets to compile for, e.g. <empty> (default, all), "Qualcomm", "QNN", '
    '"qnn", "MediaTek", "MTK", "mtk", <soc_model> (e.g. "SM8650")',
)


def main(_):
  targets = []
  for target in _TARGETS.value:
    if target == 'Qualcomm' or target == 'QNN' or target == 'qnn':
      targets.append(qnn_target.Target(qnn_target.SocModel.ALL))
    elif target == 'MediaTek' or target == 'MTK' or target == 'mtk':
      targets.append(mtk_target.Target(mtk_target.SocModel.ALL))
    elif target in qnn_target.SocModel.__members__:
      targets.append(qnn_target.Target(qnn_target.SocModel[target]))
    elif target in mtk_target.SocModel.__members__:
      targets.append(mtk_target.Target(mtk_target.SocModel[target]))
    else:
      raise ValueError(f'Unsupported target: {target}')

  logging.info('Compiling model from %s...', _MODEL.value)
  compiled_models = aot.aot_compile(
      _MODEL.value, keep_going=True, target=targets if targets else None
  )

  model_name, _ = os.path.splitext(os.path.basename(_MODEL.value))
  logging.info('Export compiled models to %s/%s...', _OUTPUT.value, model_name)
  compiled_models.export(output_dir=_OUTPUT.value, model_name=model_name)


if __name__ == '__main__':
  app.run(main)
