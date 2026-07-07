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
"""Verifies the exported talker .tflite against the PyTorch reference dump.

1) Torch-level equivalence: loads the synthesized Qwen3ForCausalLM checkpoint
   and compares against ref/talker_equiv_ref.npz (expects bit-exact — this is
   the proof that the TTS multimodal RoPE reduces to standard RoPE).
2) tflite decode sweep: feeds the 24 reference embeddings step by step with
   the KV cache carried and compares codec logits and hidden readout.

Usage (convert env):
    python verify_talker.py [model.tflite path]
"""

import sys

import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from transformers import Qwen3ForCausalLM

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'out/talker-fp32/model.tflite'
CACHE = 1024
NEG = -1e9


def corr(a, b) -> float:
    return float(np.corrcoef(np.asarray(a).ravel(),
                             np.asarray(b).ravel())[0, 1])


def main() -> None:
    ref = np.load('ref/talker_equiv_ref.npz')
    x, ref_logits, ref_hidden = ref['x'], ref['logits'], ref['hidden']

    torch_model = Qwen3ForCausalLM.from_pretrained(
        'out/talker-llm', torch_dtype=torch.float32)
    with torch.no_grad():
        logits = torch_model(inputs_embeds=torch.tensor(x),
                             use_cache=False).logits.numpy()
    print('torch codec corr', corr(logits[..., :3072], ref_logits),
          'max|d|', float(np.abs(logits[..., :3072] - ref_logits).max()))

    interp = Interpreter(model_path=MODEL)
    decode = interp.get_signature_runner('decode')
    kv = {n: np.zeros(decode.get_input_details()[n]['shape'], np.float32)
          for n in decode.get_input_details() if n.startswith('kv_cache')}
    outs = []
    for t in range(x.shape[1]):
        mask = np.full((1, 1, 1, CACHE), NEG, np.float32)
        mask[..., :t + 1] = 0.0
        result = decode(embeddings=x[:, t].reshape(1, 1, 1024),
                        input_pos=np.array([t], np.int32), mask=mask, **kv)
        outs.append(result.pop('logits')[0, 0])
        kv = result
    outs = np.stack(outs)
    top1 = (outs[:, :3072].argmax(-1) == ref_logits[0].argmax(-1)).mean()
    print(f'tflite decode sweep: codec corr {corr(outs[:, :3072], ref_logits[0]):.8f} '
          f'hidden corr {corr(outs[:, 3072:], ref_hidden[0]):.8f} '
          f'top1 {top1:.2%}')


if __name__ == '__main__':
    main()
