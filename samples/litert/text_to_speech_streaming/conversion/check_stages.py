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

"""Verify each TF-ported KittenTTS stage against the ONNX goldens."""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
import kitten_tf as K  # noqa: E402

GOLD = dict(np.load(K.OUT / "kitten_golden.npz"))


def cl(a):  # [1,C,T] -> [1,T,C]
    return np.transpose(a, (0, 2, 1))


def report(name, got, ref):
    got, ref = np.asarray(got), np.asarray(ref)
    err = np.abs(got - ref).max()
    scale = max(np.abs(ref).max(), 1e-9)
    flag = "OK " if err / scale < 1e-3 else "BAD"
    print(f"  [{flag}] {name}: maxerr={err:.3e} (ref absmax {scale:.2f}) shapes {got.shape} vs {ref.shape}")


ids = tf.constant(GOLD["input_ids"].astype(np.int32))
style = tf.constant(GOLD["style"])
style_p = style[:, 128:]
speed = tf.constant(GOLD["speed"])

x = K.bert(ids)
d_en = K.bert_encoder(x)
report("d_en", d_en.numpy(), GOLD["d_en"])

t_en = K.text_encoder(ids)
report("t_en", t_en.numpy(), cl(GOLD["t_en"]))

d = K.duration_encoder(tf.constant(GOLD["d_en"]), style_p)
report("d", d.numpy(), cl(GOLD["d"]))

dur = K.duration_head(tf.constant(cl(GOLD["d"])), speed)
ref_dur = GOLD["duration"].astype(np.int32)
print(f"  durations equal: {np.array_equal(dur.numpy(), ref_dur)} "
      f"(sum {dur.numpy().sum()} vs {ref_dur.sum()})")

# expansion equivalence: repeat vs golden aln matmul
en_repeat = np.repeat(cl(GOLD["d"])[0], ref_dur, axis=0)[None]
report("en(repeat)", en_repeat, cl(GOLD["en"]))
asr_repeat = np.repeat(cl(GOLD["t_en"])[0], ref_dur, axis=0)[None]
report("asr(repeat)", asr_repeat, cl(GOLD["asr"]))

f0, n = K.prosody(tf.constant(cl(GOLD["en"])), style_p)
report("f0_pred", f0.numpy(), GOLD["f0_pred"][:, 0])
report("n_pred", n.numpy(), GOLD["n_pred"][:, 0])
