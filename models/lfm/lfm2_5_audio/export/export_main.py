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
"""CLI entrypoint for exporting LiquidAI/LFM2.5-Audio-1.5B to LiteRT-LM."""

import fire
from litert_torch.generative.export_hf import export as lib

from models.lfm.lfm2_5_audio.export import extension as _  # pylint: disable=unused-import


def main(_=None):
  fire.Fire(lib.export)


if __name__ == '__main__':
  main()
