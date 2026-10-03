# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

"""Extract every initializer from the official basic-pitch ONNX.

Writes bp_weights.npz plus a readable manifest (name -> shape) for the
torch re-implementation.
"""
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
            attrs = {
                a.name: list(a.ints) for a in n.attribute
                if a.name in ("strides", "pads", "kernel_shape")}
            f.write(
                f"{n.op_type:>10} {n.name[-60:]:62} {attrs} "
                f"{[ (u[-40:], list(weights[u].shape)) for u in used]}\n")
print("manifest + consumers written")
