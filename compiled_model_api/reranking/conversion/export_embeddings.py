#!/usr/bin/env python3
"""Export the tied token-embedding table as fp16 for host-side lookup on device.

  embeddings_fp16.bin : [vocab, 1024] fp16, little-endian, row-major (row = token id)

Usage:
  python export_embeddings.py [MODEL_DIR] [OUT_DIR]
  # defaults: ~/Downloads/meeting/qwen3emb-work/Qwen3-Embedding-0.6B  ->  same work dir
"""
import os, sys, torch
from transformers import AutoModel

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Downloads/meeting/qwen3emb-work/Qwen3-Embedding-0.6B")
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
    "~/Downloads/meeting/qwen3emb-work")
out = os.path.join(OUT_DIR, "embeddings_fp16.bin")

m = AutoModel.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
w = m.embed_tokens.weight.detach().to(torch.float16).cpu().numpy()   # [vocab, 1024]
w.astype("<f2").tofile(out)
print(f"wrote {out}  shape={w.shape}  {os.path.getsize(out)/1e6:.1f} MB")
