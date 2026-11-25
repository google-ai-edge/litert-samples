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
"""Out-of-tree `ModelExtension` for OmniGen2 (`OmniGen2/OmniGen2`)."""

from collections.abc import Mapping
from typing import Any

from litert_torch.generative.export_hf.model_ext import extension as extension_lib

from models.omnigen.omnigen2.export import omnigen2 as omnigen2_lib

# Standard `Text2ImageMetadata.TfLiteModelType` sections. The 6 additional
# OmniGen2-specific shards (`dit_cap_embedder`, `dit_context_refiner`,
# `dit_x_embedder`, `dit_noise_refiner`, `dit_single_block_4`,
# `dit_single_block_5`) are emitted as `.tflite.bin` artifacts and packed as
# named `GenericBinaryData` sections by `package_image_gen_model`.
OMNIGEN2_SECTION_TYPES = {
    'text_encoder': 'tf_lite_text_encoder',
    'dit_single_block_0': 'tf_lite_diffusion_transformer_single_block_0',
    'dit_single_block_1': 'tf_lite_diffusion_transformer_single_block_1',
    'dit_single_block_2': 'tf_lite_diffusion_transformer_single_block_2',
    'dit_single_block_3': 'tf_lite_diffusion_transformer_single_block_3',
    'dit_final': 'tf_lite_diffusion_transformer_final',
    'vae_decoder': 'tf_lite_image_decoder',
}


@extension_lib.register_model_extension()
class OmniGen2Extension(extension_lib.ModelExtension):
  """ModelExtension for OmniGen2 multimodal diffusion text-to-image generation."""

  model_types = (
      'OmniGen2Pipeline',
      'omnigen2',
      'omnigen-2',
      'omnigen_2',
      'OmniGen2/OmniGen2',
  )
  extra_kwargs_keys = omnigen2_lib.OMNIGEN2_EXTRA_KWARGS

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
            't2i_output_image_size': 256,
            'quantization_recipe': 'dynamic_wi8_afp32',
        },
    )

  def get_image_gen_model_cls(
      self, model_type: str
  ) -> type[omnigen2_lib.OmniGen2]:
    """See base class."""
    del model_type
    return omnigen2_lib.OmniGen2

  def get_litert_lm_section_types(self) -> dict[str, str]:
    """Returns standard TFLiteModel section type mappings for OmniGen2."""
    return dict(OMNIGEN2_SECTION_TYPES)

  @classmethod
  def match_checkpoint(
      cls,
      model_path: str,
      task: Any = None,
      config_dict: Mapping[str, Any] | None = None,
  ) -> bool:
    """Identifies OmniGen2 Diffusers checkpoints (`OmniGen2Pipeline`)."""
    task_name = getattr(task, 'value', task)
    if task_name not in (
        None,
        'auto',
        extension_lib.ExportTask.TEXT_TO_IMAGE,
        extension_lib.ExportTask.TEXT_TO_IMAGE.value,
    ):
      return False
    if config_dict and config_dict.get('_class_name') == 'OmniGen2Pipeline':
      return True
    lowered_path = (model_path or '').lower()
    return (
        'omnigen2' in lowered_path
        or 'omnigen-2' in lowered_path
        or 'omnigen_2' in lowered_path
    )
