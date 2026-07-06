#!/usr/bin/env python3
"""Compare device probe_out_0.bin to ref_out.npy — reranker P(yes) score."""
import os, sys, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L, real_len = 256, 0
for line in open(os.path.join(HERE, "meta.txt")):
    if line.startswith("real_len="): real_len = int(line.split("=")[1])
    if line.startswith("L="): L = int(line.split("=")[1])

ref = np.load(os.path.join(HERE, "ref_out.npy")).reshape(L, 2)
dev = np.fromfile(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "probe_out_0.bin"),
                  dtype="<f4").reshape(L, 2)

def pyes(v):
    a = v - v.max(); e = np.exp(a); return float(e[1] / e.sum())

p = real_len - 1
sr, sd = pyes(ref[p]), pyes(dev[p])
print(f"[device] P(yes)  ref={sr:.4f}  dev={sd:.4f}  |Δ|={abs(sr-sd):.5f}")
print(f"  ref logits[no,yes]={ref[p]}  dev={dev[p]}")
print("  VERDICT:", "fp16 OK" if abs(sr - sd) < 0.02 else "DEGRADED")
