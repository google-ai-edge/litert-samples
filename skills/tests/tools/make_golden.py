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
"""Writes the golden output that the migration tests compare with.

MigrationValidationTest (litert-runtime/references/migration) runs a fixed
pattern bitmap through its placeholder preprocessing and the app's model on the
CPU, and compares the output with golden/output.txt in the test APK's assets.
This script computes that output for the same input on the workstation, with
the CompiledModel API of ai-edge-litert on the CPU, and writes it to
<fixtures>/golden/output.txt. It also writes output_wrong.txt, the same values
with the first one moved by 0.01, for the test that the gate fails on a wrong
golden. The outputs are checked in, so run this only when model.tflite changes.

  pip install ai-edge-litert
  python make_golden.py ../litert-runtime/fixtures
"""

import os
import sys

from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.compiled_model import CompiledModel
import numpy as np

# The input of MigrationValidationTest: fixedBitmap(224), then toInput(224).
SIZE = 224
# Ten times the gate's bound of 1e-3.
SHIFT = 0.01


def pattern(size):
  """The RGB pattern of fixedBitmap in the templates, [1, size, size, 3]."""
  y, x = np.mgrid[0:size, 0:size]
  r = (17 * x + 3 * y) & 0xFF
  g = (5 * x + 11 * y) & 0xFF
  b = (7 * x + 13 * y) & 0xFF
  return np.stack([r, g, b], axis=-1)[np.newaxis]


def first_input_and_output(path):
  """The shape and type of the model's first input and output tensor."""
  with open(path, "rb") as f:
    buf = f.read()
  model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))
  graph = model.subgraphs[0]
  tensors = [graph.tensors[graph.inputs[0]], graph.tensors[graph.outputs[0]]]
  return [([int(d) for d in t.shape], t.type) for t in tensors]


def number(value):
  """The shortest text that reads back as the same float32."""
  return np.format_float_positional(np.float32(value), unique=True, trim="-")


def main():
  fixtures = sys.argv[1]
  path = os.path.join(fixtures, "model.tflite")
  (in_shape, in_type), (out_shape, out_type) = first_input_and_output(path)
  if in_shape != [1, SIZE, SIZE, 3] or in_type != schema.TensorType.FLOAT32:
    sys.exit(f"input {in_shape} type {in_type}: expected"
             f" [1, {SIZE}, {SIZE}, 3] FLOAT32")
  if out_type != schema.TensorType.FLOAT32:
    sys.exit(f"output type {out_type}: expected FLOAT32")
  count = int(np.prod(out_shape))

  # The template's placeholder preprocessing in float32: (channel - 127.5) / 127.5.
  half = np.float32(127.5)
  data = (pattern(SIZE).astype(np.float32) - half) / half

  model = CompiledModel.from_file(path)
  print("signatures:", model.get_signature_list())
  inputs = model.create_input_buffers(0)
  outputs = model.create_output_buffers(0)
  inputs[0].write(data)
  model.run_by_index(0, inputs, outputs)
  values = outputs[0].read(count, np.float32)
  print("output:", " ".join(number(v) for v in values))

  wrong = values.copy()
  wrong[0] = np.float32(wrong[0] + SHIFT)
  golden = os.path.join(fixtures, "golden")
  os.makedirs(golden, exist_ok=True)
  for name, out in (("output.txt", values), ("output_wrong.txt", wrong)):
    with open(os.path.join(golden, name), "w") as f:
      f.write("".join(number(v) + "\n" for v in out))
  print("wrote", os.path.join(golden, "output.txt"), "and output_wrong.txt")


if __name__ == "__main__":
  main()
