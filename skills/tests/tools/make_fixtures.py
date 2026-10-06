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
"""Writes the small models the litert-runtime tests load.

Each one makes a different step of the skill's code throw. The outputs are
checked in as litert-runtime/fixtures/*.bin, so run this only to change one.

  pip install litert-torch
  python make_fixtures.py ../litert-runtime/fixtures
"""

import os
import sys

from ai_edge_litert import schema_py_generated as schema
import flatbuffers
import litert_torch
import torch


class Head(torch.nn.Module):
  """[1, H, W, 3] float32 -> [1, 10] float32."""

  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.fc = torch.nn.Linear(3, 10)

  def forward(self, x):
    return self.fc(x.mean(dim=(1, 2)))


class SixDims(Head):
  """The same head, then a 6-D transpose, which the GPU does not compile."""

  def forward(self, x):
    y = super().forward(x)
    y = y.reshape(1, 1, 2, 1, 5, 1).permute(0, 1, 4, 3, 2, 5)
    return y.reshape(1, 10)


def convert(module, height, path):
  """Converts a module that takes one [1, height, height, 3] input."""
  sample = (torch.zeros(1, height, height, 3),)
  litert_torch.convert(module.eval(), sample).export(path)


def redeclare(source, path, tensor, element_type):
  """Copies a model with its input or output type declared differently."""
  with open(source, "rb") as f:
    buf = bytearray(f.read())
  model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))
  graph = model.subgraphs[0]
  index = graph.inputs[0] if tensor == "input" else graph.outputs[0]
  graph.tensors[index].type = element_type
  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), b"TFL3")
  with open(path, "wb") as f:
    f.write(builder.Output())


def main():
  out = sys.argv[1]
  os.makedirs(out, exist_ok=True)
  ok = os.path.join(out, "model.bin")
  # The model the app ships: the input that the skill's INPUT_SIZE describes.
  convert(Head(), 224, ok)
  # A smaller input: the warm-up write of INPUT_SIZE floats throws.
  convert(Head(), 32, os.path.join(out, "small_input.bin"))
  # Compiles on the CPU, not on the GPU.
  convert(SixDims(), 224, os.path.join(out, "gpu_unsupported.bin"))
  # A string input or output: the model is created, then creating the input
  # or the output buffers throws.
  redeclare(ok, os.path.join(out, "string_input.bin"), "input",
            schema.TensorType.STRING)
  redeclare(ok, os.path.join(out, "string_output.bin"), "output",
            schema.TensorType.STRING)
  # A float16 output: the model and its buffers are created, then run throws.
  redeclare(ok, os.path.join(out, "float16_output.bin"), "output",
            schema.TensorType.FLOAT16)
  # Not a model: create throws.
  with open(os.path.join(out, "not_a_model.bin"), "wb") as f:
    f.write(b"TFL3" + bytes(60))


if __name__ == "__main__":
  main()
