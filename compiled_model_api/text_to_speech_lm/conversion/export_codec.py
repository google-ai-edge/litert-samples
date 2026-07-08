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
"""Exports the Qwen3-TTS-Tokenizer-12Hz DECODER to a fixed-shape .tflite.

codes [1, 16, T] int32 -> waveform [1, 1, T*1920] fp32 at 24 kHz. The whole
decoder stack is causal, so right-padding the codes and truncating the
waveform is exact; one fixed-T graph serves any utterance via chunking (the
sample uses T=64 with 25 frames of left context).

The decoder modeling code is imported from qtok12/, a two-file copy of the
qwen-tts reference implementation with three small shims so it loads under
transformers 5.x (see the notes at the top of qtok12/modeling_*.py; in
particular, 5.x meta-device loading silently zeroes the rotary inv_freq
buffer, which this script recomputes after load).

Usage (convert env):
    T=64 python export_codec.py
"""

import os
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qtok12.configuration_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2Config)
from qtok12.modeling_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2Model)

T = int(os.environ.get('T', '64'))
OUT = 'out/codec'


def corr(a, b) -> float:
    """Computes the Pearson correlation of two arrays.

    Args:
        a: First array-like; flattened before comparison.
        b: Second array-like; flattened before comparison.

    Returns:
        The correlation coefficient as a Python float.
    """
    return float(np.corrcoef(np.asarray(a).ravel(),
                             np.asarray(b).ravel())[0, 1])


def main() -> None:
    """Exports the codec decoder to .tflite and checks reference parity."""
    os.makedirs(OUT, exist_ok=True)
    src = snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base',
                            allow_patterns=['speech_tokenizer/*'])
    src = f'{src}/speech_tokenizer'
    config = Qwen3TTSTokenizerV2Config.from_pretrained(src)
    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        src, config=config, torch_dtype=torch.float32).eval()
    decoder = model.decoder
    decoder.pre_transformer.config.use_cache = False

    # transformers 5.x meta-device from_pretrained discards buffers computed
    # in __init__ -> the rotary inv_freq loads as zeros (silently: cos=1,
    # sin=0 everywhere). Recompute it.
    rotary = decoder.pre_transformer.rotary_emb
    dim = decoder.pre_transformer.config.head_dim
    theta = decoder.pre_transformer.config.rope_theta
    with torch.no_grad():
        rotary.inv_freq.copy_(1.0 / (theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32) / dim)))
    rotary.attention_scaling = 1.0

    ref = None
    if os.path.exists('ref/codec_equiv_ref.npz'):
        ref = np.load('ref/codec_equiv_ref.npz')
        with torch.no_grad():
            wav = decoder(torch.tensor(ref['codes'])).numpy()
        print(f'torch vs reference: corr {corr(wav, ref["wav"]):.8f} '
              f'max|d| {np.abs(wav - ref["wav"]).max():.3e}')

    class CodecDecode(torch.nn.Module):

        def __init__(self, dec):
            super().__init__()
            self.dec = dec

        def forward(self, codes):
            return self.dec(codes)

    import litert_torch

    path = f'{OUT}/codec_decode_t{T}.tflite'
    sample = (torch.zeros(1, 16, T, dtype=torch.int32),)
    litert_torch.convert(CodecDecode(decoder).eval(), sample).export(path)
    print('converted ->', path)

    if ref is not None:
        from ai_edge_litert.compiled_model import CompiledModel

        tfl = CompiledModel.from_file(path)
        inputs = tfl.create_input_buffers(0)
        outputs = tfl.create_output_buffers(0)
        t_ref = ref['codes'].shape[-1]
        padded = np.concatenate(
            [ref['codes'],
             np.zeros((1, 16, T - t_ref), ref['codes'].dtype)],
            -1).astype(np.int32)
        inputs[0].write(np.ascontiguousarray(padded))
        tfl.run_by_index(0, inputs, outputs)
        wav = outputs[0].read(T * 1920, np.float32).reshape(
            1, 1, T * 1920)[..., :t_ref * 1920]
        print(f'tflite vs reference: corr {corr(wav, ref["wav"]):.8f} '
              f'max|d| {np.abs(wav - ref["wav"]).max():.3e}')


if __name__ == '__main__':
    main()
