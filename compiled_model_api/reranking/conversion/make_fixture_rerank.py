#!/usr/bin/env python3
"""Reranker device probe fixture: probe_input.bin (inputs_embeds) + ref score for a pair."""
import os, numpy as np, torch
from build_qwen3rerank import Qwen3EmbGPU, build_ids, MODEL_DIR, NO_ID, YES_ID
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
L = 256
QUERY = "What is the capital of China?"
DOC = "The capital of China is Beijing."     # relevant -> expect P(yes) ~0.9995

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
