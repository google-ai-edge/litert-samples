# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Out-of-tree `ModelExtension` for DeepSeek Janus-Pro (`janus_pro`)."""

from collections.abc import Mapping
import re
from typing import Any

from litert_torch.generative.export_hf.model_ext import extension as extension_lib

from models.deepseek.janus_pro.export import janus_pro as janus_pro_lib

_JANUS_PRO_PATTERN = re.compile(r'janus[-_]?pro', re.IGNORECASE)


@extension_lib.register_model_extension()
class JanusProExtension(extension_lib.ModelExtension):
  """Extension for DeepSeek Janus-Pro Any2Any multimodal models."""

  model_types = (
      'janus_pro',
      'janus-pro',
      'janus-pro-1b',
      'janus-pro-7b',
      'janus_pro_1b',
      'janus_pro_7b',
      'Janus-Pro-1B',
      'Janus-Pro-7B',
      'multi_modality',
  )
  extra_kwargs_keys = janus_pro_lib.JANUS_PRO_EXTRA_KWARGS

  @classmethod
  def match_checkpoint(
      cls,
      model_path: str,
      task: Any = None,
      config_dict: Mapping[str, Any] | None = None,
  ) -> bool:
    """Matches Janus-Pro checkpoints by `config.json` or repo/directory path."""
    task_name = getattr(task, 'value', task)
    if task_name not in (
        None,
        'auto',
        extension_lib.ExportTask.ANY_TO_ANY,
        extension_lib.ExportTask.ANY_TO_ANY.value,
    ):
      return False
    cfg = config_dict or {}
    architectures = cfg.get('architectures') or ()
    if 'MultiModalityCausalLM' in architectures:
      return True
    if (
        cfg.get('model_type') == 'multi_modality'
        and 'gen_vision_config' in cfg
        and 'gen_head_config' in cfg
    ):
      return True
    name_or_path = cfg.get('_name_or_path') or ''
    return bool(
        _JANUS_PRO_PATTERN.search(str(model_path or ''))
        or _JANUS_PRO_PATTERN.search(str(name_or_path))
    )

  def get_export_defaults(
      self,
      config_dict: Mapping[str, Any],
      task: str | None = None,
      target: str = 'cpu_gpu',
  ) -> extension_lib.ExportDefaults:
    """See base class."""
    del config_dict, task, target
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.ANY_TO_ANY,
    )

  def get_any2any_model_cls(
      self, model_type: str
  ) -> type[janus_pro_lib.JanusPro]:
    """See base class."""
    del model_type
    return janus_pro_lib.JanusPro

  def get_litert_lm_section_types(self) -> dict[str, str]:
    """Returns extra Any2Any `.litertlm` TFLite section types for Janus-Pro."""
    return {
        'vq_image_embedder': 'tf_lite_vq_image_embedder',
        'gen_head': 'tf_lite_gen_head',
        'vq_image_decoder': 'tf_lite_vq_image_decoder',
    }
