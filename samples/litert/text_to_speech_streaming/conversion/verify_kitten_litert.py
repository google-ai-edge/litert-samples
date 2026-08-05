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

"""Verify the KittenTTS LiteRT graphs: accuracy, dynamic lengths, streaming, RTF.

Runs in the torch venv (ai_edge_litert + espeak frontend + kittentts pip).
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
VOICES = ROOT / "models" / "nano-0.8-fp32" / "voices.npz"

SR = 24000
SAMPLES_PER_FRAME = 600
HAR_PER_FRAME = 120
CHUNK = 40       # vocoder frames per streaming chunk (~1 s of audio)
OVERLAP = 20     # context frames each side, discarded after decode

from ai_edge_litert.interpreter import Interpreter  # noqa: E402


def canonical(name):
    name = name.split(":")[0]
    if name.startswith("serving_default_"):
        name = name[len("serving_default_"):]
    return name


def run_graph(path, feeds):
    it = Interpreter(model_path=str(path), num_threads=4)
    details = it.get_input_details()
    for d in details:
        it.resize_tensor_input(d["index"], list(feeds[canonical(d["name"])].shape))
    it.allocate_tensors()
    for d in details:
        it.set_tensor(d["index"], feeds[canonical(d["name"])])
    t0 = time.perf_counter()
    it.invoke()
    dt = time.perf_counter() - t0
    outs = [it.get_tensor(o["index"]) for o in it.get_output_details()]
    return outs, dt


def name_map(path):
    it = Interpreter(model_path=str(path))
    return ([d["name"] for d in it.get_input_details()],
            [d["name"] for d in it.get_output_details()])


def tokens_for(text):
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    import phonemizer
    from kittentts.onnx_model import TextCleaner, basic_english_tokenize
    backend = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True)
    phonemes = " ".join(basic_english_tokenize(backend.phonemize([text])[0]))
    toks = TextCleaner()(phonemes)
    return np.array([[0] + toks + [0]], dtype=np.int32)


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    m = min(len(a), len(b))
    return float(np.corrcoef(a[:m], b[:m])[0, 1])


def pick(outs, shape_tail):
    return [o for o in outs if list(o.shape[-len(shape_tail):]) == shape_tail]


def synthesize(ids, style, speed=1.0):
    p_out, t_pred = run_graph(OUT / "kitten_predictor.tflite", {
        "input_ids": ids, "style": style,
        "speed": np.array([speed], dtype=np.float32)})
    d = [o for o in p_out if o.ndim == 3 and o.shape[-1] == 256][0]
    t_en = [o for o in p_out if o.ndim == 3 and o.shape[-1] == 128][0]
    dur = [o for o in p_out if o.dtype == np.int32][0]

    en = np.repeat(d[0], dur, axis=0)[None]
    asr = np.repeat(t_en[0], dur, axis=0)[None]

    pr_out, t_pro = run_graph(OUT / "kitten_prosody.tflite",
                              {"en": en.astype(np.float32), "style": style})
    har = [o for o in pr_out if o.ndim == 3][0]
    f0, n = [o for o in pr_out if o.ndim == 2]

    v_out, t_voc = run_graph(OUT / "kitten_vocoder.tflite", {
        "asr": asr.astype(np.float32), "f0": f0, "n": n,
        "har": har, "style": style})
    wav = v_out[0][0]
    return wav, dict(d=d, t_en=t_en, dur=dur, en=en, asr=asr, f0=f0, n=n, har=har,
                     times=(t_pred, t_pro, t_voc))


def stream_vocoder(asr, f0, n, har, style):
    t_frames = asr.shape[1]
    pieces, times = [], []
    start = 0
    while start < t_frames:
        end = min(start + CHUNK, t_frames)
        lo, hi = max(0, start - OVERLAP), min(t_frames, end + OVERLAP)
        feeds = {"asr": asr[:, lo:hi].astype(np.float32),
                 "f0": f0[:, 2 * lo:2 * hi], "n": n[:, 2 * lo:2 * hi],
                 "har": har[:, HAR_PER_FRAME * lo:HAR_PER_FRAME * hi + 1],
                 "style": style}
        outs, dt = run_graph(OUT / "kitten_vocoder.tflite", feeds)
        wav = outs[0][0]
        a = (start - lo) * SAMPLES_PER_FRAME
        pieces.append(wav[a:a + (end - start) * SAMPLES_PER_FRAME])
        times.append(dt)
        start = end
    return np.concatenate(pieces), times


def main():
    gold = dict(np.load(OUT / "kitten_golden.npz"))
    det = dict(np.load(OUT / "kitten_golden_det.npz"))
    style = gold["style"]

    ids = gold["input_ids"].astype(np.int32)
    wav, st = synthesize(ids, style)
    ref = det["waveform"]
    print(f"golden: durations equal={np.array_equal(st['dur'], gold['duration'].astype(np.int32))} "
          f"wav corr={corr(wav, ref):.5f}")
    tp, tpr, tv = st["times"]
    dur_s = len(wav) / SR
    print(f"timing: predictor={tp*1e3:.0f}ms prosody={tpr*1e3:.0f}ms vocoder={tv*1e3:.0f}ms "
          f"audio={dur_s:.2f}s RTF={(tp+tpr+tv)/dur_s:.3f}")
    sf.write(OUT / "litert_golden.wav", wav, SR)

    for tag, text in [("short", "Good morning everyone."),
                      ("long", "The quick brown fox jumps over the lazy dog, "
                               "while seventy six trombones led the big parade.")]:
        ids2 = tokens_for(text)
        wav2, st2 = synthesize(ids2, style)
        tp, tpr, tv = st2["times"]
        print(f"[{tag}] N={ids2.shape[1]} T={int(st2['dur'].sum())} "
              f"audio={len(wav2)/SR:.2f}s pred={tp*1e3:.0f}ms pro={tpr*1e3:.0f}ms "
              f"voc={tv*1e3:.0f}ms RTF={(tp+tpr+tv)/(len(wav2)/SR):.3f}")
        sf.write(OUT / f"litert_{tag}.wav", wav2, SR)

        swav, stimes = stream_vocoder(st2["asr"], st2["f0"], st2["n"], st2["har"], style)
        print(f"[{tag}] streaming: chunks={len(stimes)} first={stimes[0]*1e3:.0f}ms "
              f"corr(vs full)={corr(swav, wav2):.6f}")
        sf.write(OUT / f"litert_{tag}_streamed.wav", swav, SR)


if __name__ == "__main__":
    main()
