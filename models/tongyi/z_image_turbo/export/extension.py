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
"""Out-of-tree `ModelExtension` for Tongyi-MAI Z-Image-Turbo (`z_image_turbo`)."""

from collections.abc import Mapping
from typing import Any

from litert_torch.generative.export_hf.model_ext import extension as extension_lib

from models.tongyi.z_image_turbo.export import z_image_turbo


@extension_lib.register_model_extension()
class ZImageTurboExtension(extension_lib.ModelExtension):
  """Extension for Z-Image-Turbo S3-DiT text-to-image generation."""

  model_types = (
      'ZImagePipeline',
      'z_image_turbo',
      'z-image-turbo',
      'z_image',
  )
  extra_kwargs_keys = z_image_turbo.Z_IMAGE_TURBO_EXTRA_KWARGS

  @classmethod
  def match_checkpoint(
      cls,
      model_path: str,
      task: Any = None,
      config_dict: Mapping[str, Any] | None = None,
  ) -> bool:
    """Identifies Z-Image-Turbo Diffusers pipelines and checkpoints."""
    task_name = getattr(task, 'value', task)
    if task_name not in (
        None,
        'auto',
        extension_lib.ExportTask.TEXT_TO_IMAGE,
        extension_lib.ExportTask.TEXT_TO_IMAGE.value,
    ):
      return False
    if config_dict:
      if config_dict.get('_class_name') == 'ZImagePipeline':
        return True
      if config_dict.get('_class_name') or config_dict.get('model_type'):
        return False
    lowered_path = (model_path or '').lower()
    return 'z-image-turbo' in lowered_path or 'z_image_turbo' in lowered_path

  def get_export_defaults(
      self,
      config_dict: Mapping[str, Any],
      task: str | None = None,
      target: str = 'cpu_gpu',
  ) -> extension_lib.ExportDefaults:
    """See base class."""
    del config_dict, task, target
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.TEXT_TO_IMAGE,
        export_kwargs={
            't2i_output_image_size': z_image_turbo.DEFAULT_IMG_SIZE,
            'quantization_recipe': z_image_turbo.DEFAULT_QUANTIZATION_RECIPE,
        },
    )

  def get_image_gen_model_cls(
      self, model_type: str
  ) -> type[z_image_turbo.ZImageTurbo]:
    """See base class."""
    del model_type
    return z_image_turbo.ZImageTurbo

  def get_litert_lm_section_types(self) -> dict[str, str]:
    """Returns standard Text2ImageMetadata TFLite section mappings."""
    return dict(z_image_turbo.Z_IMAGE_TURBO_STANDARD_SECTION_TYPES)
