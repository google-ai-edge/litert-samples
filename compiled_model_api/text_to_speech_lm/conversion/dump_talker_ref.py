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
"""Dumps talker reference activations for the cross-env equivalence check.

Feeds a fixed random embedding sequence through the raw talker (prefill
style, no cache) and stores logits + hidden states for verify_talker.py.

Usage (reference env):
    python dump_talker_ref.py
"""

import os

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


def main() -> None:
    """Dumps talker hidden states and logits on a fixed input to ref/."""
    torch.manual_seed(123)
    model = Qwen3TTSModel.from_pretrained(
        'Qwen/Qwen3-TTS-12Hz-0.6B-Base', device_map='cpu',
        dtype=torch.float32).model
    talker = model.talker

    x = torch.randn(1, 24, 1024, dtype=torch.float32) * 0.02
    with torch.no_grad():
        hidden = talker.model(inputs_embeds=x, use_cache=False)
        hidden = hidden.last_hidden_state
        logits = talker.codec_head(hidden)

    os.makedirs('ref', exist_ok=True)
    np.savez('ref/talker_equiv_ref.npz', x=x.numpy(), hidden=hidden.numpy(),
             logits=logits.numpy())
    print('dumped', x.shape, hidden.shape, logits.shape)


if __name__ == '__main__':
    main()
