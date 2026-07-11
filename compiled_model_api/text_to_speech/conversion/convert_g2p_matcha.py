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

"""Converts the OpenPhonemizer DeepPhonemizer G2P to a fixed-shape LiteRT graph.

The ForwardTransformer (espeak-IPA checkpoint, MIT / Clear BSD) becomes the
clean Matcha-TTS G2P, run on the CompiledModel CPU delegate. Adapts
kokoro/scripts/convert_dp_g2p_litert.py (same architecture) to the espeak-IPA
checkpoint.

Char -> espeak IPA, per word. Fixed [1, 96]; in-graph pad mask from id==0; CPU
delegate (EQUAL / SELECT_V2 / 5D attention that the GPU rejects, the CPU runs).

Outputs:
    artifacts/dp_g2p_matcha.tflite
    artifacts/g2p_meta.json  (char->id table, id->phoneme table, char_repeats,
                              MAXT) for the Kotlin port.

Run: python convert_g2p_matcha.py
"""

import _stub  # noqa: F401  (must be first: scipy / getsourcefile guards)

import json
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
CKPT = os.path.join(HERE, "openphonemizer_best_model.pt")
OUT = os.path.join(ART, "dp_g2p_matcha.tflite")
MAXT = 96

_original_torch_load = torch.load


def _torch_load_full(*args, **kwargs):
    """torch.load with weights_only=False (the ckpt pickles custom classes).

    Args:
        *args: Positional torch.load arguments.
        **kwargs: Keyword torch.load arguments; weights_only is overridden.

    Returns:
        The loaded checkpoint object.
    """
    return _original_torch_load(*args, **{**kwargs, "weights_only": False})


torch.load = _torch_load_full

from dp.phonemizer import Phonemizer  # noqa: E402  (needs the torch.load patch)

phonemizer = Phonemizer.from_checkpoint(CKPT, device="cpu")
predictor = phonemizer.predictor
model = predictor.model.eval()
text_tokenizer = predictor.text_tokenizer
phoneme_tokenizer = predictor.phoneme_tokenizer
CHAR2IDX = dict(text_tokenizer.token_to_idx)
REP = text_tokenizer.char_repeats
IDX2PH = {int(k): v for k, v in phoneme_tokenizer.idx_to_token.items()}
SPECIAL = {"_", "<en_us>", "<end>", "<start>"}
# DeepPhonemizer text tokenizer start/end ids (start = the <en_us> language
# token).
START = text_tokenizer.token_to_idx["<en_us>"]
END = text_tokenizer.end_index


class Wrap(nn.Module):
    """Fixed-shape wrapper: float ids in, per-position phoneme logits out."""

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, text):
        # text: [N, T] float32, 0.0 = pad.
        ids = text.to(torch.int64)
        pad_mask = ids.eq(0)
        x = ids.transpose(0, 1)
        x = self.wrapped.embedding(x)
        x = self.wrapped.pos_encoder(x)
        x = self.wrapped.encoder(x, src_key_padding_mask=pad_mask)
        x = self.wrapped.fc_out(x)
        return x.transpose(0, 1)  # [N, T, n_phonemes]


def tokenize(word: str) -> list:
    """Converts a word to DeepPhonemizer char ids.

    Args:
        word: The word to tokenize.

    Returns:
        Char ids with char_repeats applied, wrapped in start/end markers.
    """
    ids = [START]
    for c in word.lower():
        if c in CHAR2IDX and c != "_":
            ids += [CHAR2IDX[c]] * REP
    return ids + [END]


def padded(word: str):
    """Zero-pads a tokenized word to the fixed graph length.

    Args:
        word: The word to tokenize.

    Returns:
        A ([1, MAXT] float32 array, valid length) tuple.
    """
    ids = tokenize(word)[:MAXT]
    length = len(ids)
    return np.array([ids + [0] * (MAXT - length)], np.float32), length


def decode(seq) -> str:
    """Collapses repeats and strips special tokens from argmax ids.

    Args:
        seq: Sequence of argmax phoneme ids.

    Returns:
        The decoded phoneme string.
    """
    collapsed, previous = [], None
    for token in seq:
        if token != previous:
            collapsed.append(token)
            previous = token
    return "".join(
        IDX2PH[t] for t in collapsed
        if t in IDX2PH and IDX2PH[t] not in SPECIAL and t != 0
    )


def main():
    """Converts the G2P graph and spot-checks it against the library."""
    import litert_torch

    wrap = Wrap(model).eval()
    dummy, _ = padded("hello")
    litert_torch.convert(wrap, (torch.from_numpy(dummy),)).export(OUT)
    print("wrote", OUT, os.path.getsize(OUT) // 1_000_000, "MB")

    from ai_edge_litert.compiled_model import CompiledModel

    compiled = CompiledModel.from_file(OUT)
    input_buffers = compiled.create_input_buffers(0)
    output_buffers = compiled.create_output_buffers(0)
    n_out = (compiled.get_output_buffer_requirements(0, 0)["buffer_size"]
             // np.dtype(np.float32).itemsize)
    matches = 0
    words = ["hello", "matcha", "depth", "anything", "github", "anthropic",
             "daisuke", "kubernetes"]
    for word in words:
        tensor, length = padded(word)
        input_buffers[0].write(np.ascontiguousarray(tensor))
        compiled.run_by_index(0, input_buffers, output_buffers)
        logits = output_buffers[0].read(n_out, np.float32).reshape(MAXT, -1)
        got = decode(logits.argmax(-1).tolist()[:length])
        reference = phonemizer(word, lang="en_us")
        matches += got == reference
        print(f"  {'ok ' if got == reference else 'DIFF'} {word}: "
              f"tflite={got!r} lib={reference!r}")
    print(f"match {matches}/{len(words)}")

    # Metadata for the Kotlin port.
    meta = dict(char2idx=CHAR2IDX,
                idx2ph={str(k): v for k, v in IDX2PH.items()},
                char_repeats=REP, start=START, end=END, MAXT=MAXT,
                special=sorted(SPECIAL), n_phonemes=len(IDX2PH))
    with open(os.path.join(ART, "g2p_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("wrote g2p_meta.json")


if __name__ == "__main__":
    main()
