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

"""Validate dmcount.tflite: GPU op scan + CompiledModel parity vs the fixtures.

Checks two things against the ref_in.npy / ref_out.npy fixtures written by
build_dmcount.py:
  1. Static flatbuffer scan — no GPU-banned ops, no tensors of rank > 4.
  2. LiteRT CompiledModel inference matches the PyTorch reference (corr,
     absolute count difference).

Run: python validate_dmcount.py
"""
import numpy as np
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.compiled_model import CompiledModel

MODEL_PATH = "dmcount.tflite"
GPU_BANNED = {
    "GATHER", "GATHER_ND", "SELECT", "SELECT_V2", "NOT_EQUAL", "EQUAL",
    "GREATER", "LESS", "TOPK_V2", "CAST", "PACK", "SPLIT",
}


def opcheck(path):
    """Returns True when the flatbuffer has no banned ops and no rank>4 tensors."""
    with open(path, "rb") as f:
        model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    clean = True
    for graph in model.subgraphs:
        for op in graph.operators:
            code = model.operatorCodes[op.opcodeIndex]
            name = names.get(max(code.builtinCode, code.deprecatedBuiltinCode), "?")
            if name in GPU_BANNED:
                print("banned op:", name)
                clean = False
        for tensor in graph.tensors:
            if tensor.shape is not None and len(tensor.shape) > 4:
                print("rank>4 tensor:", tensor.name, tensor.shape)
                clean = False
    return clean


def main():
    print("opcheck clean:", opcheck(MODEL_PATH))

    x = np.load("ref_in.npy")
    ref = np.load("ref_out.npy")
    model = CompiledModel.from_file(MODEL_PATH)
    inputs = model.create_input_buffers(0)
    outputs = model.create_output_buffers(0)
    inputs[0].write(np.ascontiguousarray(x, np.float32))
    model.run_by_index(0, inputs, outputs)
    n = model.get_output_buffer_requirements(0, 0)["buffer_size"] // 4
    out = outputs[0].read(n, np.float32)

    corr = np.corrcoef(out.ravel(), ref.ravel())[0, 1]
    print("corr vs PyTorch: %.6f" % corr)
    print("count: tflite %.2f vs torch %.2f" % (out.sum(), ref.sum()))


if __name__ == "__main__":
    main()
