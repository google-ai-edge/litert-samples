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
"""Dumps a greedy MTP (code predictor) inner-loop trace for export_mtp.py.

Usage (reference env):
    python dump_mtp_ref.py
"""

import os

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


def main() -> None:
    torch.manual_seed(7)
    model = Qwen3TTSModel.from_pretrained(
        'Qwen/Qwen3-TTS-12Hz-0.6B-Base', device_map='cpu',
        dtype=torch.float32).model
    talker = model.talker

    past_hidden = torch.randn(1, 1, 1024) * 2.0  # talker-like magnitude
    cb0 = torch.tensor([[1234]])
    with torch.no_grad():
        cb0_embed = talker.get_input_embeddings()(cb0)
        result = talker.code_predictor.generate(
            inputs_embeds=torch.cat((past_hidden, cb0_embed), dim=1),
            max_new_tokens=15, do_sample=False, output_scores=True,
            return_dict_in_generate=True)

    os.makedirs('ref', exist_ok=True)
    np.savez('ref/mtp_equiv_ref.npz',
             past_hidden=past_hidden.numpy(),
             last_id_hidden=cb0_embed.numpy(),
             seq=result.sequences.numpy(),
             scores=np.stack([s.numpy() for s in result.scores], axis=0))
    print('greedy codes:', result.sequences.numpy().tolist())


if __name__ == '__main__':
    main()
