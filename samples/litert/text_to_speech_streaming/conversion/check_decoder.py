# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify the TF decoder stage against the deterministic ONNX golden."""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.insert(0, str(Path(__file__).parent))
import kitten_tf as K

G = dict(np.load(K.OUT / "kitten_golden.npz"))
D = dict(np.load(K.OUT / "kitten_golden_det.npz"))

style = tf.constant(G["style"])
style_ref = style[:, :128]
asr = tf.constant(np.transpose(D["asr"], (0, 2, 1)))       # [1,T,128]
f0 = tf.constant(D["f0_pred"][:, 0])                       # [1,2T]
n = tf.constant(D["n_pred"][:, 0])

har = K.harmonics_graph(f0)
print("har:", har.shape)
wav = K.decoder(asr, f0, n, style_ref, har).numpy()[0]
ref = D["waveform"]
m = min(len(wav), len(ref))
err = np.abs(wav[:m] - ref[:m]).max()
corr = np.corrcoef(wav[:m], ref[:m])[0, 1]
print(f"wav: {wav.shape} vs {ref.shape}  maxerr={err:.4e}  corr={corr:.6f}")
