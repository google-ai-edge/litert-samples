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

"""Convert NIMA (idealo MobileNet) aesthetic + technical to fp16 tflite. Run: python build_nima.py"""
import os
import sys
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "image-quality-assessment", "models", "MobileNet")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

def build(weights):
    base = MobileNet(input_shape=(224, 224, 3), weights=None, include_top=False, pooling="avg")
    x = Dense(10, activation="softmax")(Dropout(0.0)(base.output))
    m = Model(base.inputs, x)
    m.load_weights(weights)
    return m

def run_tflite(path, x):
    """Single inference through the LiteRT CompiledModel API; returns the flat fp32 output."""
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(path)
    inputs = model.create_input_buffers(0)
    outputs = model.create_output_buffers(0)
    inputs[0].write(np.ascontiguousarray(x, dtype=np.float32))
    model.run_by_index(0, inputs, outputs)
    n = model.get_output_buffer_requirements(0, 0)["buffer_size"] // np.dtype(np.float32).itemsize
    return outputs[0].read(n, np.float32)

for kind, tag in [("aesthetic", "07"), ("technical", "11")]:
    m = build(os.path.join(REPO, f"weights_mobilenet_{kind}_0.{tag}.hdf5"))
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    path = os.path.join(OUT, f"nima_{kind}_fp16.tflite")
    open(path, "wb").write(tfl)
    # parity: tflite vs keras on the ref input
    ref = np.load(os.path.join(HERE, f"ref_{kind}.npz"))
    p = run_tflite(path, ref["x"])
    score = float(np.sum((np.arange(10) + 1) * p))
    corr = float(np.corrcoef(p, ref["p"])[0, 1])
    print(f"[{kind}] {os.path.getsize(path)/1e6:.1f}MB  tflite score {score:.3f} vs ref {float(ref['score']):.3f}  corr {corr:.6f}")
    ref["x"].astype("<f4").tofile(os.path.join(OUT, f"nima_input_{kind}.bin"))
