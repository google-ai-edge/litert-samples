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
"""Convert the OpenPhonemizer DeepPhonemizer (ForwardTransformer, espeak-IPA, MIT/BSD)
to a fixed-shape LiteRT graph for CompiledModel CPU — the clean Matcha-TTS G2P.
Adapts kokoro/scripts/convert_dp_g2p_litert.py (same arch) to the espeak-IPA checkpoint.

Char -> espeak IPA, per word. Fixed [1,96]; in-graph pad mask from id==0; CPU delegate
(EQUAL/SELECT_V2/5D attn that GPU rejects, CPU runs). Output: artifacts/dp_g2p_matcha.tflite
plus g2p_meta.json (char->id table, id->phoneme table, char_repeats, MAXT) for the Kotlin port.
Run: ~/clipconv/bin/python convert_g2p_matcha.py
"""
import _stub
import os, json, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ART = os.path.join(HERE, "artifacts")
CKPT = os.path.join(HERE, "openphonemizer_best_model.pt")
OUT = os.path.join(ART, "dp_g2p_matcha.tflite")
MAXT = 96

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
from dp.phonemizer import Phonemizer

phon = Phonemizer.from_checkpoint(CKPT, device="cpu")
pred = phon.predictor
m = pred.model.eval()
tt, pt = pred.text_tokenizer, pred.phoneme_tokenizer
CHAR2IDX = dict(tt.token_to_idx); REP = tt.char_repeats
IDX2PH = {int(k): v for k, v in pt.idx_to_token.items()}
SPECIAL = {"_", "<en_us>", "<end>", "<start>"}
# DeepPhonemizer text tokenizer start/end ids (start = the <en_us> language token)
START, END = tt.token_to_idx["<en_us>"], tt.end_index


class Wrap(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, text):                 # [N,T] float32, 0.0 = pad
        ids = text.to(torch.int64)
        pad_mask = ids.eq(0)
        x = ids.transpose(0, 1)
        x = self.m.embedding(x)
        x = self.m.pos_encoder(x)
        x = self.m.encoder(x, src_key_padding_mask=pad_mask)
        x = self.m.fc_out(x)
        return x.transpose(0, 1)             # [N,T,n_phonemes]


def tokenize(w):
    ids = [START]
    for c in w.lower():
        if c in CHAR2IDX and c != "_":
            ids += [CHAR2IDX[c]] * REP
    return ids + [END]


def padded(w):
    ids = tokenize(w)[:MAXT]; L = len(ids)
    return np.array([ids + [0] * (MAXT - L)], np.float32), L


def decode(seq):
    d, p = [], None
    for t in seq:
        if t != p: d.append(t); p = t
    return "".join(IDX2PH[t] for t in d if t in IDX2PH and IDX2PH[t] not in SPECIAL and t != 0)


def main():
    import litert_torch
    wrap = Wrap(m).eval()
    dummy, _ = padded("hello")
    litert_torch.convert(wrap, (torch.from_numpy(dummy),)).export(OUT)
    print("wrote", OUT, os.path.getsize(OUT) // 1_000_000, "MB")

    from ai_edge_litert.interpreter import Interpreter
    itp = Interpreter(model_path=OUT); itp.allocate_tensors()
    ti, oi = itp.get_input_details()[0], itp.get_output_details()[0]
    ok = 0
    words = ["hello", "matcha", "depth", "anything", "github", "anthropic", "daisuke", "kubernetes"]
    for w in words:
        t, L = padded(w)
        itp.set_tensor(ti["index"], t); itp.invoke()
        got = decode(itp.get_tensor(oi["index"])[0].argmax(-1).tolist()[:L])
        lib = phon(w, lang="en_us")
        ok += got == lib
        print(f"  {'ok ' if got == lib else 'DIFF'} {w}: tflite={got!r} lib={lib!r}")
    print(f"match {ok}/{len(words)}")

    # meta for the Kotlin port
    meta = dict(char2idx=CHAR2IDX, idx2ph={str(k): v for k, v in IDX2PH.items()},
                char_repeats=REP, start=START, end=END, MAXT=MAXT,
                special=sorted(SPECIAL), n_phonemes=len(IDX2PH))
    json.dump(meta, open(os.path.join(ART, "g2p_meta.json"), "w"), ensure_ascii=False, indent=2)
    print("wrote g2p_meta.json")


if __name__ == "__main__":
    main()
