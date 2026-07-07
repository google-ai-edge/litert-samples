# Copyright 2026 The AI Edge LiteRT Authors.
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
"""Synthesizes a standard Qwen3ForCausalLM checkpoint from the TTS talker.

The talker transformer is exactly Qwen3-0.6B geometry (28 layers, hidden 1024,
16Q/8KV heads, head_dim 128, q/k-norm, rope theta 1e6). Its TTS-specific
multimodal RoPE receives identical position ids on all three streams, so it
reduces to standard 1D RoPE and the stock litert-torch LLM exporter applies
unchanged (verified bit-exact by verify_talker.py).

Two checkpoint tricks:
  * lm_head = [codec_head (3072 rows); identity (1024 rows)] so the exported
    graph's logits also carry the last hidden state (dims 3072:4096), which
    seeds the MTP inner loop.
  * The identity block's off-diagonal is filled with 1e-6: blockwise weight
    quantization rejects all-zero blocks (zero scale), and the epsilon adds
    only ~1e-4 readout error.

Usage (convert env):
    python extract_talker_ckpt.py
"""

import json
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

SRC = snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base')
DST = 'out/talker-llm'

_CONFIG = {
    'architectures': ['Qwen3ForCausalLM'],
    'model_type': 'qwen3',
    'attention_bias': False,
    'attention_dropout': 0.0,
    'bos_token_id': 2149,
    'eos_token_id': 2150,
    'head_dim': 128,
    'hidden_act': 'silu',
    'hidden_size': 1024,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 32768,
    'max_window_layers': 28,
    'num_attention_heads': 16,
    'num_hidden_layers': 28,
    'num_key_value_heads': 8,
    'rms_norm_eps': 1e-06,
    'rope_scaling': None,
    'rope_theta': 1000000,
    'sliding_window': None,
    'tie_word_embeddings': False,
    'torch_dtype': 'float32',
    'use_cache': True,
    'vocab_size': 4096,
}


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    reader = safe_open(f'{SRC}/model.safetensors', framework='pt')
    out = {}
    for key in reader.keys():
        if not key.startswith('talker.model.') and (
                key != 'talker.codec_head.weight'):
            continue
        if key.startswith('talker.model.text_embedding'):
            continue  # The text side is handled host-side.
        tensor = reader.get_tensor(key).to(torch.float32)
        if key == 'talker.model.codec_embedding.weight':
            pad = torch.zeros(4096 - tensor.shape[0], tensor.shape[1])
            out['model.embed_tokens.weight'] = torch.cat([tensor, pad], 0)
        elif key == 'talker.codec_head.weight':
            eye = torch.eye(1024)
            eye = eye + 1e-6 * (1.0 - eye)
            out['lm_head.weight'] = torch.cat([tensor, eye], 0)
        else:
            out[key.replace('talker.model.', 'model.')] = tensor

    save_file(out, f'{DST}/model.safetensors', metadata={'format': 'pt'})
    json.dump(_CONFIG, open(f'{DST}/config.json', 'w'), indent=1)
    json.dump({'bos_token_id': 2149, 'eos_token_id': 2150},
              open(f'{DST}/generation_config.json', 'w'))
    for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json'):
        if os.path.exists(f'{SRC}/{name}'):
            shutil.copy(f'{SRC}/{name}', f'{DST}/{name}')
    print(f'{len(out)} tensors -> {DST}')


if __name__ == '__main__':
    main()
