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

"""Reranker device probe fixture: probe_input.bin (inputs_embeds) + ref score for a pair.

Run: python make_fixture_rerank.py
"""
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from build_qwen3rerank import Qwen3EmbGPU, build_ids, MODEL_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
L = 256
QUERY = "What is the capital of China?"
DOC = "The capital of China is Beijing."     # relevant -> expect P(yes) ~0.9995


def main():
    base = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids, rl = build_ids(tok, QUERY, DOC, L)
    with torch.no_grad():
        embeds = base.embed_tokens(ids)              # [1,L,1024] host lookup
        net = Qwen3EmbGPU(base, L, 28).eval()
        out = net(embeds)                            # [1,L,2]
    embeds[0].numpy().astype("<f4").tofile(os.path.join(HERE, "probe_input.bin"))
    np.save(os.path.join(HERE, "ref_out.npy"), out[0].numpy())
    with open(os.path.join(HERE, "meta.txt"), "w") as f:
        f.write(f"real_len={rl}\nL={L}\n")
    score = torch.log_softmax(out[0, rl - 1], -1)[1].exp().item()
    print(f"real_len={rl}  input {embeds.numel()} floats  ref P(yes)={score:.4f}")
    print("wrote probe_input.bin, ref_out.npy, meta.txt")


if __name__ == "__main__":
    main()
