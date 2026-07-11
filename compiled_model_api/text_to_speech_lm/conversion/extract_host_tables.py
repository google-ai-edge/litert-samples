# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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
"""Extracts host-side constant tables for the LiteRT pipeline.

codec_embedding [3072, 1024], text_projection (two biased linears), and
text_embedding [151936, 2048]. The 15 MTP embedding tables are extracted by
export_mtp.py. Note the text embedding alone is ~311M parameters, ~44% of
the "0.6B" model.

Usage (convert env):
    python extract_host_tables.py
"""

import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

OUT = 'out/host'


def main() -> None:
    """Saves the codec/text embedding and text projection tables."""
    os.makedirs(OUT, exist_ok=True)
    src = snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base')
    reader = safe_open(f'{src}/model.safetensors', framework='pt')

    def get(key: str) -> np.ndarray:
        return reader.get_tensor(key).to(torch.float32).numpy()

    np.save(f'{OUT}/codec_embedding.npy',
            get('talker.model.codec_embedding.weight'))
    np.savez(f'{OUT}/text_projection.npz',
             w1=get('talker.text_projection.linear_fc1.weight'),
             b1=get('talker.text_projection.linear_fc1.bias'),
             w2=get('talker.text_projection.linear_fc2.weight'),
             b2=get('talker.text_projection.linear_fc2.bias'))
    np.save(f'{OUT}/text_embedding.npy',
            get('talker.model.text_embedding.weight'))
    print('saved host tables to', OUT)


if __name__ == '__main__':
    main()
