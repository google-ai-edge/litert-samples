#!/usr/bin/env python3
"""Extract every initializer from the official basic-pitch ONNX into bp_weights.npz,
plus a readable manifest (name -> shape) for the torch re-implementation."""
import numpy as np
import onnx
from onnx import numpy_helper

m = onnx.load("nmp_official.onnx")
g = m.graph
weights = {}
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    weights[init.name] = arr
np.savez("bp_weights.npz", **{k: v for k, v in weights.items()})

with open("bp_manifest.txt", "w") as f:
    for k, v in sorted(weights.items(), key=lambda kv: -kv[1].size):
        f.write(f"{str(v.shape):>22}  {v.dtype}  {k}\n")
print(f"{len(weights)} initializers -> bp_weights.npz")

# consumers map: which node uses which big weight (for unambiguous mapping)
inits = set(weights)
with open("bp_consumers.txt", "w") as f:
    for n in g.node:
        used = [i for i in n.input if i in inits and weights[i].size >= 36]
        if used:
            attrs = {a.name: list(a.ints) for a in n.attribute if a.name in ("strides", "pads", "kernel_shape")}
            f.write(f"{n.op_type:>10} {n.name[-60:]:62} {attrs} {[ (u[-40:], list(weights[u].shape)) for u in used]}\n")
print("manifest + consumers written")
