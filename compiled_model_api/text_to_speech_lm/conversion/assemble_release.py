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
"""Assembles the release layout used by the sample and the HF model repo.

Collects the exported graphs and host tables under out/release/ with the
published file names. The two large tables are stored as fp16, which was
verified not to change the generated tokens (waveform correlation 1.0 vs
fp32 tables under greedy decoding).

Usage (convert env), after running the export scripts:
    python assemble_release.py
"""

import os
import shutil

import numpy as np

RELEASE = 'out/release'


def main() -> None:
    """Copies the exported artifacts into the release layout."""
    os.makedirs(f'{RELEASE}/tables', exist_ok=True)
    os.makedirs(f'{RELEASE}/voices', exist_ok=True)

    shutil.copy('out/talker-boctav4/model_quantized.tflite',
                f'{RELEASE}/talker_int4.tflite')
    shutil.copy('out/talker-fp32/model.tflite',
                f'{RELEASE}/talker_fp32.tflite')
    shutil.copy('out/mtp/mtp_step.tflite', f'{RELEASE}/mtp_fp32.tflite')
    shutil.copy('out/codec/codec_decode_t64.tflite',
                f'{RELEASE}/codec_decoder_fp32.tflite')
    shutil.copy('out/talker-fp32/tokenizer.json', f'{RELEASE}/tokenizer.json')

    shutil.copy('out/host/codec_embedding.npy',
                f'{RELEASE}/tables/codec_embedding_fp32.npy')
    shutil.copy('out/host/text_projection.npz',
                f'{RELEASE}/tables/text_projection_fp32.npz')
    np.save(f'{RELEASE}/tables/mtp_embeddings_fp16.npy',
            np.load('out/mtp/mtp_embeddings.npy').astype(np.float16))
    np.save(f'{RELEASE}/tables/text_embedding_fp16.npy',
            np.asarray(np.load('out/host/text_embedding.npy', mmap_mode='r'),
                       np.float32).astype(np.float16))

    for root, _, files in os.walk(RELEASE):
        for name in sorted(files):
            path = os.path.join(root, name)
            print(f'{os.path.getsize(path) / 1e6:9.1f} MB  {path}')


if __name__ == '__main__':
    main()
