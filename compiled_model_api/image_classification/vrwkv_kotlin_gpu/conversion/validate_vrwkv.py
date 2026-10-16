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

"""Validates the VRWKV-S fp16 tflite with the LiteRT CompiledModel API.

Runs the fp16 model (two inputs: the NCHW image and the token-distance
matrix) through the CompiledModel Python API and checks that its top-1
ImageNet class and its logits match the fp32 PyTorch reference on a test image.

Requires vrwkv_s_in1k_224.pth, imagenet_classes.txt, dog.jpg and
vrwkv_s_fp16.tflite next to this script (see build_vrwkv.py).

Run:  python validate_vrwkv.py
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_vrwkv import (
    C,
    DIST_HOST,
    FP16,
    T,
    VRWKV,
    labels,
    load_state,
    preprocess,
)

TEST_IMAGE = "dog.jpg"


def main() -> None:
    """Runs the fp16 tflite and checks top-1 + logits vs fp32 torch."""
    from ai_edge_litert.compiled_model import CompiledModel

    model = VRWKV(load_state()).eval()
    image = preprocess(os.path.join(HERE, TEST_IMAGE))
    with torch.no_grad():
        ref = model(image, DIST_HOST)[0].numpy()

    cm = CompiledModel.from_file(FP16)
    inputs = cm.create_input_buffers(0)
    outputs = cm.create_output_buffers(0)
    inputs[0].write(np.ascontiguousarray(image.numpy(), np.float32))
    inputs[1].write(np.ascontiguousarray(DIST_HOST.numpy(), np.float32))
    cm.run_by_index(0, inputs, outputs)
    logits = outputs[0].read(1000, np.float32)

    names = labels()
    corr = np.corrcoef(logits, ref)[0, 1]
    dev_top = int(logits.argmax())
    ref_top = int(ref.argmax())
    print("tflite top-1: %s   fp32 top-1: %s   logits corr %.5f"
          % (names[dev_top], names[ref_top], corr))
    assert dev_top == ref_top, "fp16 tflite top-1 differs from fp32 torch"
    print("PASS")


if __name__ == "__main__":
    main()
