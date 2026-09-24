#!/usr/bin/env python3
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

"""Validates the DINOv2 fp16 tflite with the LiteRT CompiledModel API.

Runs the fp16 model through the CompiledModel Python API on a test image and
checks that its patch tokens match the fp32 PyTorch reference (build_dinov2.py).

Requires test.jpg and dinov2_s_fp16.tflite next to this script.

Run:  python validate_dinov2.py
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_dinov2 import DINOv2, FP16, N_PATCH, C, load, preprocess


def main() -> None:
    """Runs the fp16 tflite and checks patch features vs fp32 torch."""
    from ai_edge_litert.compiled_model import CompiledModel

    model = DINOv2(load()).eval()
    x = preprocess(os.path.join(HERE, "test.jpg"))
    with torch.no_grad():
        ref = model(x)[0].numpy().flatten()

    cm = CompiledModel.from_file(FP16)
    inputs = cm.create_input_buffers(0)
    outputs = cm.create_output_buffers(0)
    inputs[0].write(np.ascontiguousarray(x.numpy(), np.float32))
    cm.run_by_index(0, inputs, outputs)
    feats = outputs[0].read(N_PATCH * C, np.float32)

    corr = np.corrcoef(feats, ref)[0, 1]
    print("tflite patch features vs fp32 torch: corr %.5f" % corr)
    assert corr > 0.99, "fp16 tflite features diverge from fp32 torch"
    print("PASS")


if __name__ == "__main__":
    main()
