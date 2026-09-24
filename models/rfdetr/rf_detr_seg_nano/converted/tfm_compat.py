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

"""transformers 4.57 <-> 5.x compat shims for rfdetr 1.9.x.

rfdetr vendors a windowed-DINOv2 that touches transformers internals which
moved between 4.57 and 5.x. Import this module BEFORE importing rfdetr.
No-op on transformers >= 5.1.
"""

import inspect

import transformers

if not hasattr(transformers, "BackboneConfigMixin"):
  from transformers.utils.backbone_utils import BackboneConfigMixin
  from transformers.utils.backbone_utils import BackboneMixin
  transformers.BackboneConfigMixin = BackboneConfigMixin
  transformers.BackboneMixin = BackboneMixin

from transformers.utils.backbone_utils import BackboneMixin as _BM

if hasattr(_BM, "_init_transformers_backbone"):
  _orig = _BM._init_transformers_backbone
  if "config" in inspect.signature(_orig).parameters:

    def _compat(self, config=None):
      return _orig(self, self.config if config is None else config)

    _BM._init_transformers_backbone = _compat
