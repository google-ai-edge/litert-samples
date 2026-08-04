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

"""Build the three KittenTTS nano LiteRT graphs (dynamic length, CPU/XNNPACK).

    kitten_predictor.tflite : ids[1,N] i32, style[1,256], speed[1]
                              -> d[1,N,256], t_en[1,N,128], durations[N] i32
    host                    : en = repeat(d, dur), asr = repeat(t_en, dur)
    kitten_prosody.tflite   : en[1,T,256], style[1,256]
                              -> f0[1,2T], n[1,2T], har[1,120T+1,22]
    kitten_vocoder.tflite   : asr[1,T,128], f0, n, har, style -> wav[1,600T]

The predictor/prosody graphs go through tf_keras + from_keras_model so the
five BiLSTMs become fused dynamic-length TFLite LSTM ops; the vocoder is a
pure-op concrete function. The vocoder is fully convolutional, so it can be
run on overlapping chunks for streaming (see verify_kitten_litert.py).
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras

sys.path.insert(0, str(Path(__file__).parent))
import kitten_tf as K  # noqa: E402


def build_predictor():
    ids = keras.Input(shape=(None,), batch_size=1, dtype=tf.int32, name="input_ids")
    style = keras.Input(shape=(256,), batch_size=1, name="style")
    speed = keras.Input(shape=(), batch_size=1, name="speed")
    style_p = style[:, 128:]
    d_en = K.bert_encoder(K.bert(ids))
    d = K.duration_encoder(d_en, style_p)
    durations = K.duration_head(d, speed[0])
    t_en = K.text_encoder(ids)
    return keras.Model([ids, style, speed], [d, t_en, durations])


def build_prosody():
    en = keras.Input(shape=(None, 256), batch_size=1, name="en")
    style = keras.Input(shape=(256,), batch_size=1, name="style")
    style_p = style[:, 128:]
    f0, n = K.prosody(en, style_p)
    har = K.harmonics_graph(f0)
    return keras.Model([en, style], [f0, n, har])


def vocoder_fn(asr, f0, n, har, style):
    return K.decoder(asr, f0, n, style[:, :128], har)


def finish(conv, path, fp16):
    if fp16:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
        path = path.with_name(path.stem + "_fp16" + path.suffix)
    flat = conv.convert()
    Path(path).write_bytes(flat)
    print(f"[convert] {Path(path).name}  {len(flat)/1e6:.1f} MB")


def convert_keras(model, path, fp16):
    finish(tf.lite.TFLiteConverter.from_keras_model(model), path, fp16)


def convert_fn(fn, specs, path, fp16):
    cf = tf.function(fn, input_signature=specs).get_concrete_function()
    finish(tf.lite.TFLiteConverter.from_concrete_functions([cf]), path, fp16)


def main():
    fp16 = "--fp16" in sys.argv[1:]
    convert_keras(build_predictor(), K.OUT / "kitten_predictor.tflite", fp16)
    convert_keras(build_prosody(), K.OUT / "kitten_prosody.tflite", fp16)
    convert_fn(vocoder_fn, [
        tf.TensorSpec([1, None, 128], tf.float32, name="asr"),
        tf.TensorSpec([1, None], tf.float32, name="f0"),
        tf.TensorSpec([1, None], tf.float32, name="n"),
        tf.TensorSpec([1, None, 22], tf.float32, name="har"),
        tf.TensorSpec([1, 256], tf.float32, name="style"),
    ], K.OUT / "kitten_vocoder.tflite", fp16)


if __name__ == "__main__":
    main()
