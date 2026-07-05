#!/usr/bin/env python3
"""Compare device probe_out_0.bin to ref_out.npy (last-token embedding parity)."""
import os, sys, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = 128
real_len = 8
for line in open(os.path.join(HERE, "meta.txt")):
    if line.startswith("real_len="):
        real_len = int(line.strip().split("=")[1])

ref = np.load(os.path.join(HERE, "ref_out.npy")).reshape(L, -1)      # [L,1024]
dev = np.fromfile(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "probe_out_0.bin"),
                  dtype="<f4").reshape(L, -1)

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

p = real_len - 1
lt_cos = cos(ref[p], dev[p])
print(f"[device parity] last-token(pos{p}) cos={lt_cos:.6f}")
print(f"  ref  absmax={np.abs(ref[p]).max():.4f}  dev absmax={np.abs(dev[p]).max():.4f}  dev mean={dev[p].mean():.5f}")
# per-position cos over real tokens (depth-collapse profile)
cs = [cos(ref[i], dev[i]) for i in range(real_len)]
print("  per-token cos (real tokens):", " ".join(f"{c:.3f}" for c in cs))
print("  VERDICT:", "fp16 OK (>0.99)" if lt_cos > 0.99 else
      ("DEGRADED" if lt_cos > 0.9 else "fp16 WALL / collapse"))
