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

"""Recognition: the decode contract was captured from hardware, not guessed.

The three decode signature inputs differ by shape and dtype, and must be
told apart exactly that way, not by order: the order in the signature's
dict is not guaranteed.

  args_0  (1, 207, 288) float32  — encoder output, the only 3D one
  args_1  (1, 64)       int32    — token buffer, the only integer one
  args_2  (1, 1, 64, 64) float32 — attention mask, the remaining one
"""

import json
from pathlib import Path

import numpy as np
import pytest

from emulator.asr import (
    BOS_TOKEN,
    MASK_BLOCKED,
    MAX_DECODE_TOKENS,
    WINDOW_SAMPLES,
    MoonshineTokenizer,
    Recognizer,
    causal_mask,
    classify_decode_inputs,
    prepare_window,
)


# --- window preparation ---

def test_window_is_five_seconds_at_16k():
    assert WINDOW_SAMPLES == 80000


def test_prepare_window_pads_short_audio():
    # The model expects exactly 80000 samples; a short input must be padded,
    # not raise or get truncated.
    out = prepare_window(np.ones(1000, dtype=np.float32))
    assert out.shape == (1, WINDOW_SAMPLES)
    assert out[0, :1000].min() == 1.0
    assert out[0, 1000:].max() == 0.0


def test_prepare_window_truncates_long_audio():
    out = prepare_window(np.ones(WINDOW_SAMPLES * 2, dtype=np.float32))
    assert out.shape == (1, WINDOW_SAMPLES)


def test_prepare_window_rejects_stereo():
    with pytest.raises(ValueError, match="mono"):
        prepare_window(np.zeros((1000, 2), dtype=np.float32))


# --- distinguishing decode inputs ---

def test_classify_decode_inputs_by_shape_and_dtype():
    # This is exactly how the inputs are distinguished: by shape and dtype, not name order.
    details = {
        "args_0": {"shape": [1, 207, 288], "dtype": np.float32},
        "args_1": {"shape": [1, 64], "dtype": np.int32},
        "args_2": {"shape": [1, 1, 64, 64], "dtype": np.float32},
    }
    keys = classify_decode_inputs(details)
    assert keys["hidden"] == "args_0"
    assert keys["tokens"] == "args_1"
    assert keys["mask"] == "args_2"


def test_classify_survives_reordered_names():
    # The order in the signature's dict is not guaranteed, and names carry no meaning.
    details = {
        "zzz": {"shape": [1, 1, 64, 64], "dtype": np.float32},
        "aaa": {"shape": [1, 64], "dtype": np.int32},
        "mmm": {"shape": [1, 207, 288], "dtype": np.float32},
    }
    keys = classify_decode_inputs(details)
    assert keys["hidden"] == "mmm"
    assert keys["tokens"] == "aaa"
    assert keys["mask"] == "zzz"


def test_classify_fails_loudly_on_unexpected_contract():
    # If the export changes, better to raise than silently feed garbage into the wrong slot.
    details = {
        "a": {"shape": [1, 64], "dtype": np.int32},
        "b": {"shape": [1, 64], "dtype": np.int32},
    }
    with pytest.raises(ValueError, match="contract"):
        classify_decode_inputs(details)


# --- attention mask ---

def test_mask_is_additive_not_multiplicative():
    # Found by brute force: a multiplicative 1/0 mask gives an EMPTY output,
    # an additive one (0 allowed, large negative blocked) gives coherent text.
    mask = causal_mask(step=3, size=8)
    assert mask[0, 0, 3, :4].max() == 0.0, "allowed is zero"
    assert mask[0, 0, 3, 4:].min() == MASK_BLOCKED, "blocked is a large negative"


def test_causal_mask_allows_only_past_positions():
    # At step 0, one position is visible; at step 3, four are.
    mask = causal_mask(step=3, size=8)
    assert mask.shape == (1, 1, 8, 8)
    assert (mask[0, 0, 3, :4] == 0.0).all()
    assert (mask[0, 0, 3, 4:] == MASK_BLOCKED).all()


def test_causal_mask_is_cumulative_over_steps():
    mask = causal_mask(step=2, size=8)
    for row in range(3):
        assert (mask[0, 0, row, :row + 1] == 0.0).all(), f"row {row}"
        assert (mask[0, 0, row, row + 1:] == MASK_BLOCKED).all(), f"row {row}"


def test_multiplicative_variant_kept_for_comparison():
    # The variant that does NOT work is kept so the difference is visible in the code.
    mask = causal_mask(step=1, size=4, additive=False)
    assert mask[0, 0, 1, :2].min() == 1.0
    assert mask[0, 0, 1, 2:].max() == 0.0


def test_bos_token_is_one_not_zero():
    # Zero in this vocab is `<unk>`, and with it the model outputs `<unk>`
    # forever regardless of the mask. `<s>` = 1 is required.
    assert BOS_TOKEN == 1


def test_causal_mask_rejects_step_beyond_size():
    with pytest.raises(ValueError, match="step"):
        causal_mask(step=64, size=64)


# --- tokenizer ---

def write_vocab(path: Path, vocab: dict[str, int]) -> None:
    path.write_text(json.dumps({"model": {"vocab": vocab}}), encoding="utf-8")


def test_tokenizer_decodes_ids(tmp_path: Path):
    path = tmp_path / "tok.json"
    write_vocab(path, {"hello": 5, "world": 6, "<s>": 1, "</s>": 2})
    assert MoonshineTokenizer(path).decode([5, 6]) == "hello world"


def test_tokenizer_drops_special_tokens(tmp_path: Path):
    path = tmp_path / "tok.json"
    write_vocab(path, {"<s>": 1, "</s>": 2, "yes": 7})
    # Special tokens must not leak into the text, or they'll end up in the prompt.
    assert MoonshineTokenizer(path).decode([1, 7, 2]) == "yes"


def test_tokenizer_exposes_end_check(tmp_path: Path):
    path = tmp_path / "tok.json"
    write_vocab(path, {"</s>": 2, "yes": 7})
    tok = MoonshineTokenizer(path)
    # The decode loop must be able to stop without reaching into the internals.
    assert tok.is_end(2) is True
    assert tok.is_end(7) is False


def test_tokenizer_stops_at_end_token(tmp_path: Path):
    path = tmp_path / "tok.json"
    write_vocab(path, {"</s>": 2, "a": 7, "b": 8})
    # After the end of sequence comes garbage from the unrun buffer.
    assert MoonshineTokenizer(path).decode([7, 2, 8]) == "a"


def test_tokenizer_rejects_empty_vocab(tmp_path: Path):
    # A wrong-format tokenizer JSON yields an empty vocab; construction must
    # fail loud rather than making every transcribe() silently return "" (which
    # would leave the robot deaf, waking only on scene change).
    path = tmp_path / "empty.json"
    write_vocab(path, {})
    with pytest.raises(ValueError, match="empty vocab"):
        MoonshineTokenizer(path)


def test_tokenizer_joins_sentencepiece_marks(tmp_path: Path):
    path = tmp_path / "tok.json"
    write_vocab(path, {"▁ever": 10, "▁tried": 11, "ing": 12})
    # The ▁ marker denotes a word start; without joining it'd come out as "ever tried ing".
    assert MoonshineTokenizer(path).decode([10, 11, 12]) == "ever trieding"


def test_tokenizer_ignores_unknown_ids(tmp_path: Path):
    path = tmp_path / "tok.json"
    write_vocab(path, {"yes": 7})
    assert MoonshineTokenizer(path).decode([7, 99999]) == "yes"


def test_tokenizer_logs_unknown_ids_instead_of_silently_dropping(
        tmp_path: Path, caplog):
    # An unknown id used to be silently dropped — with a mismatched vocab
    # that would be invisible. Now the count and the ids themselves must reach the log.
    path = tmp_path / "tok.json"
    write_vocab(path, {"yes": 7})
    with caplog.at_level("WARNING", logger="emulator.asr"):
        MoonshineTokenizer(path).decode([7, 99999, 12345])
    messages = [r.getMessage() for r in caplog.records]
    assert any("2" in m and "unknown" in m for m in messages), \
        f"unknown ids were not logged: {messages}"


# --- cost ---

def test_decode_budget_matches_measured():
    # encode is 51 ms once, plus 37 ms per token; a typical utterance is ~9 tokens.
    assert MAX_DECODE_TOKENS == 64
    typical = 51.0 + 37.0 * 9
    assert 350 < typical < 430, f"typical cost {typical:.0f} ms"
    # Pin the reconciliation to the pipeline's actual ASR budget, so this test
    # fails if the two drift apart rather than silently guarding stale literals.
    from emulator.timeline import BUDGET_MS
    assert abs(typical - BUDGET_MS["asr_full"]) < 20, \
        f"asr_full budget {BUDGET_MS['asr_full']:.0f} ms drifted from measured {typical:.0f} ms"
    assert typical / 1000 / 5.0 < 0.2, "margin over real time is more than five"


# --- Recognizer constructor: the threads parameter must not be dead ---

def test_recognizer_forwards_threads_to_compiled_runner(monkeypatch,
                                                        tmp_path: Path):
    captured = {}

    class FakeEncodeSig:
        def get_input_details(self):
            return {"args_0": {"shape": [1, 80000], "dtype": np.float32}}

    class FakeDecodeSig:
        def get_input_details(self):
            return {
                "args_0": {"shape": [1, 207, 288], "dtype": np.float32},
                "args_1": {"shape": [1, 64], "dtype": np.int32},
                "args_2": {"shape": [1, 1, 64, 64], "dtype": np.float32},
            }

    class FakeRunner:
        def __init__(self, model_path, threads=4):
            captured["threads"] = threads

        def signature(self, key):
            return FakeEncodeSig() if key == "encode" else FakeDecodeSig()

    monkeypatch.setattr("emulator.asr.CompiledRunner", FakeRunner)
    vocab_path = tmp_path / "vocab.json"
    write_vocab(vocab_path, {"<s>": 1, "</s>": 2})
    Recognizer(Path("fake.tflite"), MoonshineTokenizer(vocab_path), threads=7)
    assert captured["threads"] == 7, \
        "threads must be forwarded to CompiledRunner, not sit unused"
