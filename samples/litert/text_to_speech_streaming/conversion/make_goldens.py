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

"""Golden reference dump for the KittenTTS nano TF port.

Uses the kittentts pip package for text preprocessing and onnxruntime for
inference, with extra graph outputs patched in at the module cut points.
Saves inputs + intermediates + wav to out/kitten_golden.npz and the full
weight set to out/kitten_weights.npz.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "nano-0.8-fp32" / "kitten_tts_nano_v0_8.onnx"
VOICES = ROOT / "models" / "nano-0.8-fp32" / "voices.npz"
OUT = ROOT / "out"

TEXT = "Hello, this is a test of the Kitten text to speech model running on device."
VOICE = "expr-voice-2-m"
SPEED = 1.0

# tensor name -> golden key
CUT_POINTS = {
    "/bert_encoder/Add_output_0": "d_en",
    "/text_encoder_1/Where_3_output_0": "d",
    "/text_encoder/Where_3_output_0": "t_en",
    "/Unsqueeze_11_output_0": "aln",
    "/MatMul_output_0": "en",
    "/MatMul_1_output_0": "asr",
    "/F0_proj/Conv_output_0": "f0_pred",
    "/N_proj/Conv_output_0": "n_pred",
}


def prepare_inputs():
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    from kittentts.onnx_model import TextCleaner, basic_english_tokenize
    import phonemizer

    backend = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True)
    phonemes = " ".join(basic_english_tokenize(backend.phonemize([TEXT])[0]))
    tokens = TextCleaner()(phonemes)
    tokens.insert(0, 0)
    tokens.append(0)
    input_ids = np.array([tokens], dtype=np.int64)

    voices = np.load(VOICES)
    ref_id = min(len(TEXT), voices[VOICE].shape[0] - 1)
    style = voices[VOICE][ref_id:ref_id + 1].astype(np.float32)
    return input_ids, style, np.array([SPEED], dtype=np.float32)


def main():
    OUT.mkdir(exist_ok=True)
    model = onnx.load(str(MODEL))
    existing = {o.name for o in model.graph.output}
    value_infos = {v.name: v for v in model.graph.value_info}
    for tname in CUT_POINTS:
        if tname not in existing:
            vi = value_infos.get(tname)
            model.graph.output.append(
                vi if vi is not None else onnx.helper.make_empty_tensor_value_info(tname))
    patched = OUT / "kitten_patched.onnx"
    onnx.save(model, str(patched))

    input_ids, style, speed = prepare_inputs()
    sess = ort.InferenceSession(str(patched))
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {"input_ids": input_ids, "style": style, "speed": speed})
    by_name = dict(zip(names, outs))

    golden = {
        "input_ids": input_ids,
        "style": style,
        "speed": speed,
        "waveform": by_name["waveform"],
        "duration": by_name["duration"],
    }
    for tname, key in CUT_POINTS.items():
        golden[key] = by_name[tname]
    np.savez(OUT / "kitten_golden.npz", **golden)
    for k, v in golden.items():
        print(f"  {k}: {np.asarray(v).shape} {np.asarray(v).dtype}")

    weights = {}
    graph = onnx.load(str(MODEL)).graph
    for t in graph.initializer:
        weights[t.name] = onnx.numpy_helper.to_array(t)
    np.savez(OUT / "kitten_weights.npz", **weights)
    print(f"weights: {len(weights)} tensors")

    import soundfile as sf
    sf.write(OUT / "ref_onnx.wav", by_name["waveform"][..., :-5000].ravel(), 24000)
    print("wrote out/ref_onnx.wav")


if __name__ == "__main__":
    main()
