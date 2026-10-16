# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Emit device probe fixture: probe_input.bin (inputs_embeds) +
ref_out.npy (CPU wrapper).

Run: python make_fixture.py
"""
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from build_qwen3emb import Qwen3EmbGPU, MODEL_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
L = 128
SENT = "The capital of France is Paris."


def main():
    """Writes probe_input.bin, ref_out.npy, and meta.txt fixtures."""
    base = AutoModel.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    enc = tok(SENT, return_tensors="pt", padding="max_length",
              truncation=True, max_length=L)
    real_len = int(enc.attention_mask.sum())
    with torch.no_grad():
        embeds = base.embed_tokens(enc.input_ids)       # [1,L,1024] host lookup
        net = Qwen3EmbGPU(base, L, 28).eval()
        out = net(embeds)                                # [1,L,1024]

    embeds[0].numpy().astype("<f4").tofile(
        os.path.join(HERE, "probe_input.bin"))
    np.save(os.path.join(HERE, "ref_out.npy"), out[0].numpy())
    with open(os.path.join(HERE, "meta.txt"), "w") as f:
        f.write(f"real_len={real_len}\nL={L}\nsent={SENT}\n")
    lt = out[0, real_len - 1].numpy()
    lt = lt / np.linalg.norm(lt)
    print(f"real_len={real_len}  input {embeds.numel()} floats  "
          f"ref last-token norm-emb absmax={np.abs(lt).max():.4f}")
    print("wrote probe_input.bin, ref_out.npy, meta.txt")


if __name__ == "__main__":
    main()
