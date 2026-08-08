# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Speech recognition via moonshine-tiny on LiteRT.

The contract was captured on-device, not guessed.

    encode signature
      input  args_0  [1, 80000] float32   raw PCM 16 kHz, exactly 5 seconds
      output output_0 [1, 207, 288]

    decode signature
      input  args_0  [1, 207, 288]  float32  encoder output
      input  args_1  [1, 64]        int32    full token buffer
      input  args_2  [1, 1, 64, 64] float32  attention mask
      output output_0 [1, 64, 32768]         logits over the full vocab

No mel frontend needed: the model consumes raw audio directly. 52 subgraphs total.

On cost. The decoder gets the entire 64-token buffer on every step, meaning
there's no KV cache in this export and each step costs the same regardless of
position — 37 ms. Full window cost = 51 + 37*N; for a typical ~9 tokens that's
~390 ms for 5 seconds of audio, i.e. a ~13x margin.

The decode inputs are distinguished by shape and dtype, not by name order: the
order in the signature's dict is not guaranteed, and args_N-style names carry
no information.

Two properties were found by brute force, since guessing didn't work, and they
only work together:

  1. **The mask is additive**: 0 allows a position, a large negative blocks it.
     A multiplicative 1/0 mask doesn't work at all.
  2. **The buffer starts with BOS = 1** (`<s>`), not zero. Zero is `<unk>`,
     and with it the model outputs `<unk>` forever regardless of the mask.

Neither fix helps alone: with BOS=0, all four mask variants produced nothing;
with BOS=1 and an additive mask, coherent text came out. The bug would have
been invisible without a reference recording, because the cost matched the
budget perfectly either way.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from emulator.litert_runtime import CompiledRunner

logger = logging.getLogger(__name__)

WINDOW_SAMPLES = 80000        # 5 seconds at 16 kHz
MAX_DECODE_TOKENS = 64
BOS_TOKEN = 1                 # `<s>` in the vocab; zero is `<unk>`
MASK_BLOCKED = -1e9           # mask is additive, not multiplicative


def prepare_window(pcm: np.ndarray) -> np.ndarray:
    """Reshape audio to [1, 80000] — the model requires an exact length."""
    if pcm.ndim != 1:
        raise ValueError(f"expected mono, got {pcm.shape}")
    if len(pcm) < WINDOW_SAMPLES:
        pcm = np.pad(pcm, (0, WINDOW_SAMPLES - len(pcm)))
    return pcm[:WINDOW_SAMPLES].astype(np.float32).reshape(1, WINDOW_SAMPLES)


def classify_decode_inputs(details: dict) -> dict[str, str]:
    """Which decode input is which.

    Distinguished by shape and dtype: 3D float is the encoder output,
    integer is the token buffer, the remaining one is the attention mask.
    Order or names can't be relied on, and feeding garbage into the wrong
    slot means getting plausible-looking but wrong text.
    """
    hidden = tokens = mask = None
    for key, detail in details.items():
        shape = [int(x) for x in detail["shape"]]
        dtype = np.dtype(detail["dtype"])
        if np.issubdtype(dtype, np.integer):
            tokens = key
        elif len(shape) == 3:
            hidden = key
        else:
            mask = key
    if hidden is None or tokens is None or mask is None:
        raise ValueError(
            f"decode contract changed: didn't find all three inputs among "
            f"{ {k: [int(x) for x in v['shape']] for k, v in details.items()} }"
        )
    return {"hidden": hidden, "tokens": tokens, "mask": mask}


def causal_mask(step: int, size: int = MAX_DECODE_TOKENS,
                additive: bool = True) -> np.ndarray:
    """Attention mask: at step N, positions zero through N are visible.

    Additive by default — 0 allows, a large negative blocks. This is exactly
    what this export expects: a multiplicative 1/0 mask gives an empty output.
    The additive parameter is kept to test the opposite variant in tests.
    """
    if step >= size:
        raise ValueError(f"step {step} is not less than buffer size {size}")
    rows = np.arange(size).reshape(-1, 1)
    cols = np.arange(size).reshape(1, -1)
    allowed = (cols <= rows) & (rows <= step)
    if additive:
        mask = np.where(allowed, 0.0, MASK_BLOCKED)
    else:
        mask = allowed.astype(np.float32)
    return mask.astype(np.float32).reshape(1, 1, size, size)


class MoonshineTokenizer:
    """Reverse mapping from token ids to text."""

    SPECIAL = frozenset({"<s>", "</s>", "<pad>", "<unk>"})
    END = "</s>"

    def __init__(self, vocab_path: Path) -> None:
        data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        vocab = data.get("model", {}).get("vocab", data.get("vocab", {}))
        if not vocab:
            # A wrong-format tokenizer JSON yields {} here; init would then
            # succeed with an empty table and EVERY transcribe() would return
            # "" — the robot would only ever wake on scene change, never on
            # speech, with no error. Fail loud at construction instead.
            raise ValueError(
                f"tokenizer at {vocab_path} has an empty vocab: expected a "
                "'model.vocab' or top-level 'vocab' mapping")
        self._by_id = {int(v): k for k, v in vocab.items()}

    def is_end(self, token_id: int) -> bool:
        """End of sequence. Public method: the decode loop must be able to
        stop without reaching into the tokenizer's internals."""
        return self._by_id.get(int(token_id)) == self.END

    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        unknown: list[int] = []
        for token_id in ids:
            token = self._by_id.get(int(token_id))
            if token is None:
                unknown.append(int(token_id))
                continue
            if token == self.END:
                break          # what follows is garbage from the unrun buffer
            if token in self.SPECIAL:
                continue
            pieces.append(token)
        if unknown:
            # An unknown id being silently dropped would mask a vocab/model
            # mismatch. Log both the count and the ids themselves so this
            # is visible instead of getting lost in the decoded text.
            logger.warning("decode: %d unknown token id(s) dropped: %s",
                           len(unknown), unknown)
        # The ▁ marker denotes a word start: join pieces and turn it into a
        # space, otherwise words would fall apart into subwords.
        joined = "".join(pieces)
        if "▁" in joined:
            return joined.replace("▁", " ").strip()
        return " ".join(pieces).strip()


class Recognizer:
    def __init__(self, model_path: Path, tokenizer: MoonshineTokenizer,
                 threads: int = 4) -> None:
        # CompiledModel is the current LiteRT API (Interpreter is deprecated).
        # The two signatures — encode, decode — are called just like the old
        # signature runners, so transcribe() below is unchanged.
        runner = CompiledRunner(model_path, threads=threads)
        self._encode = runner.signature("encode")
        self._decode = runner.signature("decode")
        self._tokenizer = tokenizer
        self._keys = classify_decode_inputs(self._decode.get_input_details())
        self._runner = runner

    def transcribe(self, pcm: np.ndarray) -> str:
        window = prepare_window(pcm)
        encode_key = next(iter(self._encode.get_input_details()))
        hidden = next(iter(self._encode(**{encode_key: window}).values()))

        details = self._decode.get_input_details()
        tokens_detail = details[self._keys["tokens"]]
        size = int(tokens_detail["shape"][1])
        tokens = np.zeros(
            (1, size), dtype=np.dtype(tokens_detail["dtype"]).type)
        # The first token must be BOS: with zero (`<unk>`) the model outputs
        # `<unk>` forever regardless of the mask.
        tokens[0, 0] = BOS_TOKEN

        produced: list[int] = []
        for step in range(size - 1):
            out = self._decode(**{
                self._keys["hidden"]: hidden,
                self._keys["tokens"]: tokens,
                self._keys["mask"]: causal_mask(step, size),
            })
            logits = next(iter(out.values()))
            token_id = int(np.argmax(logits[0, step]))
            produced.append(token_id)
            if self._tokenizer.is_end(token_id):
                break
            tokens[0, step + 1] = token_id

        return self._tokenizer.decode(produced)
