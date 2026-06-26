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

#!/usr/bin/env python3
"""Validate the Kotlin port (MatchaG2P + MatchaSynthesizer) by replicating its EXACT
host logic in Python — per-word tflite G2P + manual assembly + the integer length-
regulator + Euler loop — and synthesizing through the same fp16 graphs. Confirms the
Android pipeline before an on-device build. Run: ~/clipconv/bin/python kotlin_replica.py
"""
import _stub
import os, re, json, math, numpy as np, torch
import build_matcha as B
from ai_edge_litert.interpreter import Interpreter

ART = os.path.join(B.HERE, "artifacts")
MAX_TEXT, MAX_MEL, N_FEATS, N_CH, HOP, SR = 256, 512, 80, 192, 256, 22050
MEL_MEAN, MEL_STD, LS = -5.536622, 2.116101, 0.95
TIME_DIM = 160
cfg = json.load(open(os.path.join(ART, "config.json")))
SYMBOLS = cfg["symbols"]; SYM2ID = {s: i for i, s in enumerate(SYMBOLS) if len(s) == 1}
meta = json.load(open(os.path.join(ART, "g2p_meta.json")))
C2I = {k: v for k, v in meta["char2idx"].items() if len(k) == 1}
I2P = {int(k): v for k, v in meta["idx2ph"].items()}
REP = meta["char_repeats"]; START = meta["start"]; END = meta["end"]
MAXT = meta["MAXT"]; SPECIAL = set(meta["special"]); NPH = meta["n_phonemes"]
emb = np.load(os.path.join(ART, "emb.npy"))
DICT = {}
for ln in open(os.path.join(ART, "g2p_dict.txt"), encoding="utf-8"):
    if "\t" in ln:
        w, ipa = ln.rstrip("\n").split("\t", 1); DICT[w] = ipa

g2p = Interpreter(model_path=os.path.join(ART, "dp_g2p_matcha_fp16.tflite")); g2p.allocate_tensors()
te = Interpreter(model_path=os.path.join(ART, "matcha_textenc_fp16.tflite")); te.allocate_tensors()
dec = Interpreter(model_path=os.path.join(ART, "matcha_decoder_fp16.tflite")); dec.allocate_tensors()
voc = Interpreter(model_path=os.path.join(ART, "matcha_vocoder_fp16.tflite")); voc.allocate_tensors()


def run(it, *ins):
    ds = it.get_input_details()
    for d, x in zip(ds, ins): it.set_tensor(d["index"], x.astype(d["dtype"]))
    it.invoke()
    return [it.get_tensor(o["index"]) for o in it.get_output_details()]


# ---- MatchaG2P replica ----
WORD = re.compile(r"[a-z']+"); TOKEN = re.compile(r"[a-z']+|[.,!?;:—…\"]")


def phon_word(w):
    ids = [START]
    for c in w:
        if c in C2I: ids += [C2I[c]] * REP
    ids.append(END); L = min(len(ids), MAXT)
    inb = np.array([[ids[i] if i < L else 0 for i in range(MAXT)]], np.float32)
    logits = run(g2p, inb)[0][0]    # [MAXT, NPH]
    sb, prev = [], -1
    for t in range(L):
        best = int(logits[t].argmax())
        if best == prev: continue
        prev = best
        ph = I2P.get(best)
        if ph is None or ph in SPECIAL or best == 0: continue
        sb.append("".join(ch for ch in ph if ch != "-"))
    return "".join(sb)


def phonemize(text):
    ipa, first = [], True
    for m in TOKEN.finditer(text.lower()):
        tok = m.group(0)
        if WORD.fullmatch(tok):
            p = DICT.get(tok) or phon_word(tok)   # dict primary, neural OOV fallback
            if p:
                if not first: ipa.append(" ")
                ipa.append(p); first = False
        else:
            ipa.append(tok)
    s = "".join(ipa)
    return [SYM2ID[c] for c in s if c in SYM2ID], s


# ---- MatchaSynthesizer replica ----
def sin_pos_emb(t):
    half = TIME_DIM // 2; out = np.zeros(TIME_DIM, np.float32)
    k = -math.log(10000) / (half - 1)
    for i in range(half):
        e = 1000.0 * t * math.exp(i * k); out[i] = math.sin(e); out[half + i] = math.cos(e)
    return out[None]


def synth(pids, nsteps=10, seed=0):
    tx = min(len(pids) * 2 + 1, MAX_TEXT)
    ids = np.zeros(MAX_TEXT, np.int64); i = 1
    for p in pids:
        if i >= MAX_TEXT: break
        ids[i] = p; i += 2
    tmask = np.array([[[1.0 if k < tx else 0.0 for k in range(MAX_TEXT)]]], np.float32)
    embx = emb[ids].reshape(1, MAX_TEXT, N_CH).astype(np.float32)
    mu, logw = None, None
    o = run(te, embx, tmask)
    mu, logw = (o[0], o[1]) if o[0].shape[1] == 80 else (o[1], o[0])
    mu = mu[0]; logw = logw[0, 0]                       # [80,256], [256]
    wceil = np.ceil(np.exp(logw) * tmask[0, 0]) * LS
    cum = np.cumsum(wceil); ylen = int(min(max(int(cum[-1]), 1), MAX_MEL))
    muY = np.zeros((N_FEATS, MAX_MEL), np.float32)
    p = 0
    for f in range(ylen):
        while p < MAX_TEXT - 1 and cum[p] <= f: p += 1
        muY[:, f] = mu[:, p]
    ymask = np.array([[[1.0 if f < ylen else 0.0 for f in range(MAX_MEL)]]], np.float32)
    rng = np.random.RandomState(seed)
    x = np.zeros((1, N_FEATS, MAX_MEL), np.float32)
    x[0, :, :ylen] = rng.randn(N_FEATS, ylen).astype(np.float32)
    dt = 1.0 / nsteps; t = 0.0
    muYb = muY[None]
    for _ in range(nsteps):
        v = run(dec, x, muYb, sin_pos_emb(t), ymask)[0]
        x = x + dt * v; t += dt
    mel = np.zeros((1, N_FEATS, MAX_MEL), np.float32)
    mel[0, :, :ylen] = x[0, :, :ylen] * MEL_STD + MEL_MEAN
    wav = run(voc, mel)[0].reshape(-1)[:ylen * HOP]
    return np.clip(wav, -1, 1), ylen


def main():
    import soundfile as sf
    from matcha.text import text_to_sequence
    for i, s in enumerate(["Hello, this is Matcha running on the mobile GPU.",
                           "The quick brown fox jumps over the lazy dog."]):
        ids, ipa = phonemize(s)
        esp = text_to_sequence(s, ["english_cleaners2"])[1]
        print(f"[{i}] {s!r}")
        print(f"    kotlin-G2P IPA: {ipa!r}")
        print(f"    espeak     IPA: {esp!r}")
        wav, ylen = synth(ids)
        out = os.path.join(ART, f"kotlin_{i}.wav"); sf.write(out, wav, SR)
        print(f"    {len(ids)} phonemes, {ylen} frames, {len(wav)/SR:.2f}s -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
