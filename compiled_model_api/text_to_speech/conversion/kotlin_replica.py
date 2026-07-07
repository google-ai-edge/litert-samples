# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

"""Validates the Kotlin port by replicating its exact host logic in Python.

Replicates MatchaG2P + MatchaSynthesizer — per-word tflite G2P + manual assembly +
the integer length-regulator + Euler loop — and synthesizes through the same fp16
graphs. Confirms the Android pipeline before an on-device build.

Run: ~/clipconv/bin/python kotlin_replica.py
"""

import _stub  # noqa: F401  (must be first: scipy / getsourcefile guards)

import json
import math
import os
import re

import numpy as np

import build_matcha as B
from ai_edge_litert.compiled_model import CompiledModel

ART = os.path.join(B.HERE, "artifacts")
MAX_TEXT = 256
MAX_MEL = 512
N_FEATS = 80
N_CHANNELS = 192
HOP = 256
SAMPLE_RATE = 22050
MEL_MEAN = -5.536622
MEL_STD = 2.116101
LENGTH_SCALE = 0.95
TIME_DIM = 160

with open(os.path.join(ART, "config.json")) as _f:
    _config = json.load(_f)
SYMBOLS = _config["symbols"]
SYM2ID = {s: i for i, s in enumerate(SYMBOLS) if len(s) == 1}
with open(os.path.join(ART, "g2p_meta.json")) as _f:
    _meta = json.load(_f)
C2I = {k: v for k, v in _meta["char2idx"].items() if len(k) == 1}
I2P = {int(k): v for k, v in _meta["idx2ph"].items()}
REP = _meta["char_repeats"]
START = _meta["start"]
END = _meta["end"]
MAXT = _meta["MAXT"]
SPECIAL = set(_meta["special"])
NPH = _meta["n_phonemes"]

emb = np.load(os.path.join(ART, "emb.npy"))
DICT = {}
with open(os.path.join(ART, "g2p_dict.txt"), encoding="utf-8") as _f:
    for _line in _f:
        if "\t" in _line:
            _word, _ipa = _line.rstrip("\n").split("\t", 1)
            DICT[_word] = _ipa

g2p = CompiledModel.from_file(os.path.join(ART, "dp_g2p_matcha_fp16.tflite"))
te = CompiledModel.from_file(os.path.join(ART, "matcha_textenc_fp16.tflite"))
dec = CompiledModel.from_file(os.path.join(ART, "matcha_decoder_fp16.tflite"))
voc = CompiledModel.from_file(os.path.join(ART, "matcha_vocoder_fp16.tflite"))

WORD = re.compile(r"[a-z']+")
TOKEN = re.compile(r"[a-z']+|[.,!?;:—…\"]")


def run(model, *inputs):
    """Runs a LiteRT CompiledModel on numpy inputs and returns all shaped outputs."""
    signatures = model.get_signature_list()
    key = list(signatures)[0]
    in_details = model.get_input_tensor_details(key)
    out_details = model.get_output_tensor_details(key)
    input_buffers = model.create_input_buffers(0)
    output_buffers = model.create_output_buffers(0)
    for name, buffer, x in zip(signatures[key]["inputs"], input_buffers, inputs):
        buffer.write(np.ascontiguousarray(x, dtype=np.dtype(in_details[name]["dtype"])))
    model.run_by_index(0, input_buffers, output_buffers)
    outputs = []
    for name, buffer in zip(signatures[key]["outputs"], output_buffers):
        detail = out_details[name]
        outputs.append(buffer.read(int(np.prod(detail["shape"])),
                                   np.dtype(detail["dtype"])).reshape(detail["shape"]))
    return outputs


def phon_word(word: str) -> str:
    """MatchaG2P.phonemizeWord replica: one word -> espeak-style IPA (neural G2P)."""
    ids = [START]
    for c in word:
        if c in C2I:
            ids += [C2I[c]] * REP
    ids.append(END)
    length = min(len(ids), MAXT)
    inputs = np.array([[ids[i] if i < length else 0 for i in range(MAXT)]], np.float32)
    logits = run(g2p, inputs)[0][0]  # [MAXT, NPH]
    pieces, previous = [], -1
    for t in range(length):
        best = int(logits[t].argmax())
        if best == previous:
            continue
        previous = best
        phoneme = I2P.get(best)
        if phoneme is None or phoneme in SPECIAL or best == 0:
            continue
        pieces.append("".join(ch for ch in phoneme if ch != "-"))
    return "".join(pieces)


def phonemize(text: str):
    """MatchaG2P.phonemize replica: text -> (matcha symbol ids, IPA string)."""
    ipa, first = [], True
    for match in TOKEN.finditer(text.lower()):
        token = match.group(0)
        if WORD.fullmatch(token):
            phonemes = DICT.get(token) or phon_word(token)  # Dict primary, neural OOV.
            if phonemes:
                if not first:
                    ipa.append(" ")
                ipa.append(phonemes)
                first = False
        else:
            ipa.append(token)
    ipa_string = "".join(ipa)
    return [SYM2ID[c] for c in ipa_string if c in SYM2ID], ipa_string


def sin_pos_emb(t: float) -> np.ndarray:
    """MatchaSynthesizer.sinusoidalPositionEmbedding replica."""
    half = TIME_DIM // 2
    out = np.zeros(TIME_DIM, np.float32)
    k = -math.log(10000) / (half - 1)
    for i in range(half):
        angle = 1000.0 * t * math.exp(i * k)
        out[i] = math.sin(angle)
        out[half + i] = math.cos(angle)
    return out[None]


def synth(phoneme_ids, nsteps: int = 10, seed: int = 0):
    """MatchaSynthesizer.synthesize replica through the fp16 tflite graphs."""
    text_length = min(len(phoneme_ids) * 2 + 1, MAX_TEXT)
    ids = np.zeros(MAX_TEXT, np.int64)
    i = 1
    for p in phoneme_ids:
        if i >= MAX_TEXT:
            break
        ids[i] = p
        i += 2
    tmask = np.array([[[1.0 if k < text_length else 0.0 for k in range(MAX_TEXT)]]],
                     np.float32)
    embx = emb[ids].reshape(1, MAX_TEXT, N_CHANNELS).astype(np.float32)
    outputs = run(te, embx, tmask)
    mu, logw = ((outputs[0], outputs[1]) if outputs[0].shape[1] == 80
                else (outputs[1], outputs[0]))
    mu = mu[0]  # [80, MAX_TEXT]
    logw = logw[0, 0]  # [MAX_TEXT]
    wceil = np.ceil(np.exp(logw) * tmask[0, 0]) * LENGTH_SCALE
    cum = np.cumsum(wceil)
    ylen = int(min(max(int(cum[-1]), 1), MAX_MEL))
    mu_y = np.zeros((N_FEATS, MAX_MEL), np.float32)
    p = 0
    for f in range(ylen):
        while p < MAX_TEXT - 1 and cum[p] <= f:
            p += 1
        mu_y[:, f] = mu[:, p]
    ymask = np.array([[[1.0 if f < ylen else 0.0 for f in range(MAX_MEL)]]], np.float32)
    rng = np.random.RandomState(seed)
    x = np.zeros((1, N_FEATS, MAX_MEL), np.float32)
    x[0, :, :ylen] = rng.randn(N_FEATS, ylen).astype(np.float32)
    dt = 1.0 / nsteps
    t = 0.0
    mu_y_batched = mu_y[None]
    for _ in range(nsteps):
        v = run(dec, x, mu_y_batched, sin_pos_emb(t), ymask)[0]
        x = x + dt * v
        t += dt
    mel = np.zeros((1, N_FEATS, MAX_MEL), np.float32)
    mel[0, :, :ylen] = x[0, :, :ylen] * MEL_STD + MEL_MEAN
    wav = run(voc, mel)[0].reshape(-1)[:ylen * HOP]
    return np.clip(wav, -1, 1), ylen


def main():
    import soundfile as sf
    from matcha.text import text_to_sequence

    for i, sentence in enumerate(["Hello, this is Matcha running on the mobile GPU.",
                                  "The quick brown fox jumps over the lazy dog."]):
        ids, ipa = phonemize(sentence)
        espeak_ipa = text_to_sequence(sentence, ["english_cleaners2"])[1]
        print(f"[{i}] {sentence!r}")
        print(f"    kotlin-G2P IPA: {ipa!r}")
        print(f"    espeak     IPA: {espeak_ipa!r}")
        wav, ylen = synth(ids)
        out = os.path.join(ART, f"kotlin_{i}.wav")
        sf.write(out, wav, SAMPLE_RATE)
        print(f"    {len(ids)} phonemes, {ylen} frames, "
              f"{len(wav) / SAMPLE_RATE:.2f}s -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
