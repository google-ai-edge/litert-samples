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

"""Prompt and reply chunking against measured constraints.

Two numbers from measurements drive everything else. First: TTFT doesn't
grow up to ~110 prompt tokens, then jumps from 1.17 to 10.5 s at 310. Second:
the synthesis window holds 55 phoneme tokens, roughly 45 characters of text,
and is billed in full regardless of how filled it is.

Hence the rule: keep the prompt short, and chunk the reply to FILL the
window, not by sentence. A short phrase costs the same as a full window, so
splitting finer than needed means paying for silence.
"""

import json

import pytest

from emulator.llm import (
    GESTURE_NAMES,
    PHONEMES_PER_CHAR,
    PROMPT_CHAR_LIMIT,
    TTS_WINDOW_CHARS,
    LlmClient,
    build_prompt,
    split_into_phrases,
)


# --- prompt ---

def test_prompt_mentions_seen_objects():
    prompt = build_prompt(objects=["person", "cup"], heard="what do you see")
    assert "person" in prompt and "cup" in prompt


def test_prompt_puts_question_first():
    # What the person said comes before the vision context, so the model
    # answers the question instead of describing the scene.
    prompt = build_prompt(objects=["person"], heard="what do you see")
    assert prompt.index("what do you see") < prompt.lower().index("currently see:")


def test_prompt_omits_vision_line_when_nothing_seen():
    # Empty scene: the vision line is absent entirely — not "I see nothing",
    # just no vision context at all.
    prompt = build_prompt(objects=[], heard="hello")
    assert "currently see:" not in prompt.lower()
    assert "hello" in prompt


def test_prompt_and_system_are_english():
    # Not a style choice: the af_heart voice and phoneme dictionary are
    # English. With a Russian reply, coverage drops to zero and the pipeline
    # correctly skips the phrase.
    from emulator.llm import SYSTEM
    prompt = build_prompt(objects=["person"], heard="hello")
    assert all(ord(c) < 128 for c in prompt), f"non-ASCII in prompt: {prompt!r}"
    assert "English" in SYSTEM


def test_prompt_includes_heard_speech():
    prompt = build_prompt(objects=["person"], heard="what do you see")
    assert "what do you see" in prompt


def test_prompt_stays_within_prefill_sweet_spot():
    # Measured on the live model: a total prompt (system + user) up to ~490
    # characters holds TTFT at 1.2 s, at ~518 it cliffs to 10.6 s. system is
    # ~280, so the user prompt must stay short (< PROMPT_CHAR_LIMIT).
    prompt = build_prompt(objects=["person", "cup", "laptop"],
                          heard="what do you see here")
    assert len(prompt) < PROMPT_CHAR_LIMIT, \
        f"prompt is {len(prompt)} chars — risk of TTFT cliff"


def test_prompt_truncates_a_long_object_list():
    # In a crowded frame the detector finds dozens of objects; the full list
    # can't go into the prompt, or we'd blow past the prefill cliff.
    many = [f"object{i}" for i in range(60)]
    prompt = build_prompt(objects=many, heard="what do you see around you")
    assert len(prompt) < PROMPT_CHAR_LIMIT


def test_prompt_dedupes_repeated_objects():
    # Repeats are rare after dedup, but not impossible. Count within the
    # "currently see:" part, since the word person also appears in "the
    # person said".
    prompt = build_prompt(objects=["person", "person", "cup"],
                          heard="what do you see here")
    vision_part = prompt.split("currently see:")[1]
    assert vision_part.count("person") == 1


def test_prompt_always_includes_objects_even_when_not_asked_about_vision():
    # The wants_vision gate is removed: objects are ALWAYS in the prompt —
    # not sounding like a robot is enforced by the system prompt
    # ("background sensor data, mention only if asked"), not by withholding
    # data from the prompt. A live E2B measurement of this combination
    # recorded 0/10 false read-outs and 0/5 misses.
    prompt = build_prompt(objects=["person", "cup"], heard="tell me a joke")
    assert "cup" in prompt
    assert "currently see:" in prompt.lower()
    assert "tell me a joke" in prompt


def test_prompt_includes_vision_on_a_vision_question_too():
    prompt = build_prompt(objects=["person", "cup"], heard="what can you see right now")
    assert "cup" in prompt


def test_prompt_caps_long_heard_to_protect_prefill_budget():
    # A long utterance is truncated in the prompt: otherwise it plus system
    # would push the total prompt past the prefill cliff. The full heard
    # text lives elsewhere (log/reply).
    from emulator.llm import HEARD_CHAR_LIMIT
    long_heard = "please " * 40  # ~280 characters
    prompt = build_prompt(objects=["person"], heard=long_heard.strip())
    # The prompt's copy of the utterance is no longer than the cap...
    assert prompt.count("please") <= HEARD_CHAR_LIMIT // len("please ") + 1
    # ...and the reply instruction isn't cut off by the tail limit.
    assert prompt.endswith("English sentence.")


def test_build_prompt_no_longer_takes_a_gesture_argument():
    # The gesture is no longer a prompt parameter: the model itself decides
    # via function calling (move_head, demo/serve.py) — build_prompt doesn't
    # know about it.
    with pytest.raises(TypeError):
        build_prompt(objects=["person"], heard="shake your head",
                    gesture="shake")


def test_gesture_names_cover_the_expected_moves():
    # Gesture names remain the shared dictionary — the enum for move_head and
    # the check on the name returned by the model (demo/serve.py).
    assert set(GESTURE_NAMES) == {
        "nod", "shake", "look_left", "look_right", "look_up", "look_down"}


def test_build_prompt_with_max_objects_stays_within_budget():
    # Worst case: 4 objects (the MAX_OBJECTS_IN_PROMPT cap) + a long question
    # — the user part stays under the limit, the reply instruction isn't cut.
    prompt = build_prompt(objects=["person", "cup", "laptop", "chair"],
                          heard="please look to your left and tell me what you see")
    assert len(prompt) < PROMPT_CHAR_LIMIT
    assert prompt.endswith("English sentence.")


def test_system_plus_prompt_under_cliff():
    # HARD BUDGET (prefill cliff): the SUM of system+user stays under
    # ~490 characters. The worst-case USER prompt is: max heard length
    # (HEARD_CHAR_LIMIT) + the MAX_OBJECTS_IN_PROMPT *longest* object labels.
    # Every COCO class is named, but with SHORT labels — the longest is
    # "motorbike" (9 chars, label_name() in emulator/pipeline.py). That cap is
    # the whole point of the budget: the full COCO names ("baseball glove", 14)
    # would push the worst case over the cliff, so the long ones are abbreviated
    # (glove, stoplight, table, ...). This test guards that the abbreviations
    # keep the worst case under the limit.
    #
    # We guard BOTH system prompts: STREAM_SYSTEM (live streaming path, 277 →
    # 468 combined) AND the offline SYSTEM (the non-streaming reply() path, 281
    # → 472 combined). The cliff is a prefill-length limit that applies to any
    # prompt, and the offline SYSTEM is the longer of the two — so it is the
    # tighter bound, and a regression there must fail this test too.
    from demo.serve import STREAM_SYSTEM
    from emulator.llm import SYSTEM, HEARD_CHAR_LIMIT, MAX_OBJECTS_IN_PROMPT
    from emulator.pipeline import label_name

    longest_objects = sorted((label_name(i) for i in range(80)), key=len,
                             reverse=True)[:MAX_OBJECTS_IN_PROMPT]
    worst_prompt = build_prompt(objects=longest_objects,
                                heard="x" * HEARD_CHAR_LIMIT)
    for label, system in (("STREAM_SYSTEM", STREAM_SYSTEM), ("SYSTEM", SYSTEM)):
        combined = len(system) + len(worst_prompt)
        assert combined < 490, \
            f"{label}: {combined} chars — risk of TTFT cliff"


# --- reply chunking ---

def test_window_chars_derived_from_phoneme_budget():
    # 55 phoneme tokens at ~1.2 tokens per character is roughly 45 characters.
    assert 40 <= TTS_WINDOW_CHARS <= 50
    assert TTS_WINDOW_CHARS == int(55 / PHONEMES_PER_CHAR)


def test_split_respects_max_length():
    text = "Hello there. I can see you clearly now. How are you doing today?"
    phrases = split_into_phrases(text, max_chars=30)
    assert all(len(p) <= 30 for p in phrases)
    assert "".join(phrases).replace(" ", "") == text.replace(" ", "")


def test_split_fills_the_window_instead_of_splitting_per_sentence():
    # Short sentences need to be GLUED TOGETHER up to window size: the window
    # is billed in full, and three one-word calls cost three times as much
    # as one.
    phrases = split_into_phrases("Yes. No. Maybe. Sure. Okay.", max_chars=45)
    assert len(phrases) == 1, f"should have glued into one chunk, got {phrases}"


def test_split_does_not_exceed_window_when_gluing():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    phrases = split_into_phrases(text, max_chars=30)
    assert all(len(p) <= 30 for p in phrases)
    assert len(phrases) >= 2


def test_split_breaks_long_sentence_at_word_boundary():
    text = "I can see absolutely everything that is happening in this room now"
    phrases = split_into_phrases(text, max_chars=25)
    assert all(len(p) <= 25 for p in phrases)
    # A break mid-word sounds like a glitch, a pause between words doesn't.
    for phrase in phrases:
        assert not phrase.startswith(" ")
        assert phrase == phrase.strip()


def test_split_handles_word_longer_than_window():
    phrases = split_into_phrases("a" * 100, max_chars=30)
    assert len(phrases) == 4
    assert all(len(p) <= 30 for p in phrases)
    assert "".join(phrases) == "a" * 100, "the remainder of a word must not be dropped"


def test_split_of_empty_text_is_empty():
    assert split_into_phrases("", max_chars=45) == []
    assert split_into_phrases("   ", max_chars=45) == []


def test_split_keeps_sentence_punctuation():
    # Punctuation is kept because it shapes intonation in synthesis.
    phrases = split_into_phrases("Hello there!", max_chars=45)
    assert phrases[0].endswith("!")


# --- streaming client (reply_stream) ---

class _FakeSSE:
    """A stand-in for the urlopen response: a context manager that yields the
    given raw SSE lines (bytes) when iterated. Lets reply_stream be tested
    without a live serve process."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> "_FakeSSE":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def __iter__(self):
        return iter(self._lines)


def _sse(obj: dict) -> bytes:
    """One SSE `data:` line carrying a chat-completion chunk."""
    return ("data: " + json.dumps(obj) + "\n").encode()


def _client_over(monkeypatch, lines: list[bytes]) -> LlmClient:
    client = LlmClient(endpoint="http://x", model="m")
    monkeypatch.setattr(client, "_post", lambda *a, **k: _FakeSSE(lines))
    return client


def test_reply_stream_yields_content_chunks_in_order(monkeypatch):
    # The happy path: content deltas come out as {"type": "content"} events,
    # in arrival order, and [DONE] ends the stream.
    client = _client_over(monkeypatch, [
        _sse({"choices": [{"delta": {"content": "Hello"}}]}),
        _sse({"choices": [{"delta": {"content": " there"}}]}),
        b"data: [DONE]\n",
    ])
    assert list(client.reply_stream("hi")) == [
        {"type": "content", "text": "Hello"},
        {"type": "content", "text": " there"},
    ]


def test_reply_stream_stops_at_done_and_ignores_trailing_chunks(monkeypatch, caplog):
    # [DONE] TERMINATES the stream: a chunk after it must never be yielded,
    # and [DONE] must NOT be mistaken for a malformed chunk. Without this,
    # deleting the `if data == "[DONE]": break` branch would still pass the
    # other streaming tests (the terminator would silently fall through to the
    # malformed-chunk handler) — this test makes the break load-bearing.
    client = _client_over(monkeypatch, [
        _sse({"choices": [{"delta": {"content": "before"}}]}),
        b"data: [DONE]\n",
        _sse({"choices": [{"delta": {"content": "after"}}]}),
    ])
    with caplog.at_level("WARNING"):
        events = list(client.reply_stream("hi"))
    assert events == [{"type": "content", "text": "before"}]
    assert "malformed SSE chunk" not in caplog.text


def test_reply_stream_accumulates_tool_call_arguments_split_across_chunks(monkeypatch):
    # The defensive invariant: even if the runtime splits function.arguments
    # across several chunks (it currently doesn't — see the reply_stream
    # docstring), reply_stream accumulates them by index and emits ONE
    # tool_call, with arguments parsed, only after the stream ends.
    client = _client_over(monkeypatch, [
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "move_head",
                                      "arguments": '{"gesture":'}}]}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ' "nod"}'}}]}}]}),
        b"data: [DONE]\n",
    ])
    assert list(client.reply_stream("nod please", tools=[{}])) == [
        {"type": "tool_call", "name": "move_head", "arguments": {"gesture": "nod"}},
    ]


def test_reply_stream_skips_and_logs_a_malformed_chunk(monkeypatch, caplog):
    # A malformed SSE chunk is skipped, not fatal — and it's logged (used to
    # be dropped silently, which made live debugging guesswork).
    client = _client_over(monkeypatch, [
        b"data: not-json\n",
        _sse({"choices": [{"delta": {"content": "ok"}}]}),
        b"data: [DONE]\n",
    ])
    with caplog.at_level("WARNING"):
        events = list(client.reply_stream("hi"))
    assert events == [{"type": "content", "text": "ok"}]
    assert "malformed SSE chunk" in caplog.text


def test_reply_stream_skips_empty_choices_keepalive(monkeypatch):
    # A usage-only / keepalive chunk (choices == []) must be skipped, not crash
    # with IndexError — some serve versions emit these mid-stream.
    client = _client_over(monkeypatch, [
        _sse({"choices": []}),
        _sse({"choices": [{"delta": {"content": "ok"}}]}),
        b"data: [DONE]\n",
    ])
    assert list(client.reply_stream("hi")) == [{"type": "content", "text": "ok"}]


def test_reply_stream_drops_and_logs_a_tool_call_with_no_name(monkeypatch, caplog):
    # A tool_call that accumulated arguments but never a name is dropped (and
    # logged, not silently) — a gesture shouldn't vanish without a trace.
    client = _client_over(monkeypatch, [
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{}"}}]}}]}),
        b"data: [DONE]\n",
    ])
    with caplog.at_level("WARNING"):
        events = list(client.reply_stream("hi", tools=[{}]))
    assert events == []
    assert "no name" in caplog.text


def test_reply_stream_bad_tool_arguments_default_to_empty(monkeypatch, caplog):
    # Unparseable function arguments default to {} (and are logged) rather than
    # crashing the stream — the gesture name still comes through.
    client = _client_over(monkeypatch, [
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "move_head",
                                      "arguments": "{not json"}}]}}]}),
        b"data: [DONE]\n",
    ])
    with caplog.at_level("WARNING"):
        events = list(client.reply_stream("hi", tools=[{}]))
    assert events == [{"type": "tool_call", "name": "move_head", "arguments": {}}]
    assert "unparseable" in caplog.text


def test_reply_coerces_null_content_to_empty_string(monkeypatch):
    # A completion ending on tool_calls can carry content: null; reply() must
    # coerce it to "" instead of AttributeError on .strip().
    client = LlmClient(endpoint="http://x", model="m")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": None}}]}).encode()

    monkeypatch.setattr(client, "_post", lambda *a, **k: _Resp())
    assert client.reply("hi") == ""
