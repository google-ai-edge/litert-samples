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

"""LLM client and prompt preparation.

The model runs as a separate process via `litert-lm serve` — the Reachy demo
requires this, and it isolates the most resource-hungry component (2161 MB).

Two measured numbers drive everything else.

The whole prompt is kept under a measured prefill cliff, not out of tidiness:
a TOTAL prompt (system + user) up to ~490 characters holds TTFT at ~1.2 s,
while at ~518 characters it jumps to ~10.6 s and stays flat beyond that.

On the streaming path the reply is emitted sentence by sentence (see
demo/serve.py) so synthesis can start on the first finished sentence instead of
waiting for the whole reply. On the offline path the reply is chopped into short
phrases (split_into_phrases below) so each is synthesized as a natural unit.
Inflect-Nano-v2 streams within a sentence at ~210 ms per phrase.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from collections.abc import Iterator

logger = logging.getLogger(__name__)

# The prefill cliff was measured precisely on live Gemma E2B: a TOTAL prompt
# (system + user) up to ~490 characters holds TTFT at 1.2 s, while at ~518 it
# jumps to 10.6 s and stays flat beyond that (the litert-lm model bills a
# fixed large prefill window).
PREFILL_SAFE_TOTAL_CHARS = 490   # system + user must stay at or under this

# The person's utterance goes into the prompt truncated: a long question would
# otherwise push the total prompt past the prefill cliff. The full text is
# still visible in the log and in the reply's heard field — only the prompt
# copy gets cut. With this cap the real worst-case user prompt is ~198 chars
# (80 heard + the four longest object labels + the reply instruction), so
# SYSTEM(281) + user stays ~479, under PREFILL_SAFE_TOTAL_CHARS. The guard is
# test_system_plus_prompt_under_cliff, which fails if the HEARD/OBJECT caps or
# either system prompt drift enough to bust the budget.
HEARD_CHAR_LIMIT = 80

# Phoneme tokens per character of text: "hello world" (11 characters) yields
# 13 IPA tokens, i.e. roughly 1.2.
PHONEMES_PER_CHAR = 1.2

# Target phrase length for the offline path's split_into_phrases: about 55
# phoneme tokens' worth of text — a comfortable synthesis chunk.
TTS_WINDOW_CHARS = int(55 / PHONEMES_PER_CHAR)

# Replies must be IN ENGLISH, and that's not a stylistic choice: the af_heart
# voice and phoneme dictionary are English. With a Russian reply, phoneme
# coverage drops to zero and the pipeline correctly skips the phrase — the
# robot stays silent. Verified on a live run: with a Russian system prompt
# the model replied "Attempts and failures are part of the journey", and all
# seven words fell outside the dictionary.
#
# A persona, not an answering machine: the small model latches onto anything
# in the prompt, so "speak naturally" and "don't read out labels" are stated
# directly in the system prompt.
#
# The object list used to only enter the prompt behind the wants_vision gate
# — the small model held the negative instruction poorly and read out
# "person, cup" without the gate. Live E2B measurement showed that with the
# objects ALWAYS in the prompt E2B doesn't do that — no false read-outs, no
# misses — provided the prompt frames the object line as background sensor
# data rather than part of the conversation (the "V2" wording below).
# DECISION: the gate is removed, objects are always in the prompt, and the
# model itself decides whether the question is about vision based on the
# system prompt.
SYSTEM = (
    'You are Reachy, a small friendly desk robot. Talk naturally, in one '
    'short English sentence, not like a machine. "(You currently see: ...)" '
    'is sensor data: mention it only if asked what you see or about your '
    'surroundings, else ignore it. If you miss their words, ask them to '
    "repeat."
)

# A hard backstop on the user prompt: system + user must not exceed the safe
# prefill total even if build_prompt's inputs grow. Derived from the budget
# minus the (longer, offline) SYSTEM persona — NOT a magic number — so it can
# never silently drift looser than the cliff allows. Real prompts are ~198
# chars and never reach this; STREAM_SYSTEM (demo/serve.py, shorter) has even
# more headroom.
PROMPT_CHAR_LIMIT = PREFILL_SAFE_TOTAL_CHARS - len(SYSTEM)   # 490 - 281 = 209

# Head gesture names. A gesture used to be recognized by a substring
# heuristic (detect_gesture/GESTURE_HINTS on words like "shake your head")
# — replaced by native function calling: the model itself decides via the
# move_head tool (see MOVE_HEAD_TOOL in demo/serve.py). The list stays here
# as the shared name dictionary — the enum for the tool and the check on the
# returned name.
GESTURE_NAMES = ("nod", "shake", "look_left", "look_right", "look_up", "look_down")


# How many objects from the frame to mention: in a crowded frame the detector
# finds dozens, and the full list would push the prompt past the prefill
# cliff. Keep it short — the top three by detector confidence (detections
# arrive score-sorted, see non_max_suppression) are enough for "what do you
# see", and the character budget under the cliff is tight.
MAX_OBJECTS_IN_PROMPT = 3


def _unique(items: list[str], limit: int) -> list[str]:
    """Unique values, order preserved, capped at limit."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
        if len(out) >= limit:
            break
    return out


def build_prompt(objects: list[str], heard: str) -> str:
    """Build the prompt: the person's question comes first, the object list always follows.

    The object list used to enter the prompt only behind the `wants_vision`
    gate — the gate is removed (see the DECISION note by SYSTEM above): the
    list is always included, and not sounding like a robot is enforced by
    the system prompt ("background sensor data, mention only if asked"),
    not by withholding data. The gesture is no longer a prompt parameter —
    the model itself decides via function calling (move_head, demo/serve.py).
    """
    heard = heard.strip()[:HEARD_CHAR_LIMIT]
    parts = []
    if heard:
        parts.append(f'The person said: "{heard}".')
    else:
        parts.append("The person did not say anything clear.")
    seen = _unique(objects, MAX_OBJECTS_IN_PROMPT)
    if seen:
        parts.append(f"(You currently see: {', '.join(seen)}.)")
    parts.append("Reply in one short English sentence.")
    prompt = " ".join(parts)
    return prompt[:PROMPT_CHAR_LIMIT]


def split_into_phrases(text: str,
                       max_chars: int = TTS_WINDOW_CHARS) -> list[str]:
    """Chop the reply into chunks, each about one synthesis phrase wide.

    Short sentences are GLUED TOGETHER rather than spoken separately, so the
    synthesizer gets natural-length phrases instead of a stream of one-word
    calls.

    A long sentence is cut at word boundaries: a pause between words sounds
    natural, a break mid-word sounds like a glitch. A word longer than the
    window is chopped into pieces — the remainder can't be dropped, or part
    of the reply would go unspoken.
    """
    if not text.strip():
        return []

    phrases: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            phrases.append(current)
            current = ""

    for sentence in re.findall(r"[^.!?]+[.!?]*", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        # The whole sentence fits in the rest of the current chunk — append it.
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        flush()
        if len(sentence) <= max_chars:
            current = sentence
            continue
        # The sentence itself is longer than the window: cut by words.
        for word in sentence.split():
            while len(word) > max_chars:
                flush()
                phrases.append(word[:max_chars])
                word = word[max_chars:]
            candidate = f"{current} {word}".strip()
            if len(candidate) > max_chars:
                flush()
                current = word
            else:
                current = candidate
    flush()
    return phrases


class LlmClient:
    def __init__(self, endpoint: str, model: str,
                 max_chars: int = 120, timeout: float = 120.0) -> None:
        self.endpoint = endpoint
        self.model = model
        self.max_chars = max_chars
        self.timeout = timeout

    def _post(self, prompt: str, stream: bool, system: str | None = None,
             tools: list[dict] | None = None):
        body = {
            "model": self.model,
            "stream": stream,
            "max_tokens": 64,
            "messages": [
                {"role": "system", "content": system or SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }
        if tools:
            body["tools"] = tools
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=self.timeout)

    def reply(self, prompt: str, system: str | None = None) -> str:
        with self._post(prompt, stream=False, system=system) as response:
            payload = json.loads(response.read())
        # The truncation here applies to the whole reply (max_chars): the
        # offline path, where the text is synthesized in one call. The
        # reply_stream below does NOT truncate by max_chars — chunks go to
        # TTS as they arrive, and reply length is kept in check by the short
        # system prompt, not by truncating after the fact.
        # content is a required key (its absence is contract drift → KeyError,
        # fail loud), but its VALUE can legitimately be null when a completion
        # ends on tool_calls — coerce that to "" so .strip() can't AttributeError.
        message = payload["choices"][0]["message"]
        return (message["content"] or "").strip()[:self.max_chars]

    def reply_stream(self, prompt: str, system: str | None = None,
                     tools: list[dict] | None = None) -> Iterator[dict]:
        """Events as they're generated: reply text or a tool call.

        Streaming matters: the full E2B reply takes 6.09 s, and waiting for
        it in full makes the robot look dead. With streaming, the first
        sound comes out after TTFT plus synthesis of one window.

        Each event is a dict: `{"type": "content", "text": ...}` — a chunk of
        the reply, to be synthesized as before; `{"type": "tool_call",
        "name": ..., "arguments": {...}}` — a function call (see
        MOVE_HEAD_TOOL in demo/serve.py). Verified live on serve 0.15.0: the
        runtime emits tool_calls AND content mutually exclusively (gesture — silent,
        speech — no tool_calls), in one chunk with no splitting — but the
        code still accumulates `function.arguments` by index in case the
        runtime ever starts splitting it across several chunks, and only
        yields the tool_call once the stream has ended.
        """
        calls: dict[int, dict] = {}
        with self._post(prompt, stream=True, system=system,
                        tools=tools) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    # Used to be dropped silently — during live debugging it
                    # was unclear whether the runtime sent nothing at all or
                    # sent garbage that got quietly discarded. Log the chunk
                    # itself.
                    logger.warning("reply_stream: dropped malformed SSE chunk: %r",
                                   data)
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    # A usage-only / keepalive chunk (choices == []) is valid in
                    # OpenAI-compatible streams from serve versions other than
                    # the one this was built against — skip it, don't IndexError
                    # and crash the reply generator mid-stream.
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield {"type": "content", "text": content}
                for call in delta.get("tool_calls") or []:
                    idx = call.get("index", 0)
                    entry = calls.setdefault(idx, {"name": "", "arguments": ""})
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        entry["name"] = fn["name"]
                    if fn.get("arguments"):
                        entry["arguments"] += fn["arguments"]
        for entry in calls.values():
            if not entry["name"]:
                # A tool_call accumulated arguments but never a name. The
                # runtime always sends the name in the first tool_call chunk
                # (see docstring), so this is practically unreachable — but log
                # it rather than drop a gesture with no trace, consistent with
                # the malformed-chunk and unparseable-args branches.
                logger.warning("reply_stream: dropping a tool_call with no "
                               "name (arguments=%r)", entry["arguments"])
                continue
            try:
                arguments = json.loads(entry["arguments"]) if entry["arguments"] else {}
            except json.JSONDecodeError:
                # Match the malformed-SSE-chunk branch above: don't drop a bad
                # tool call silently, or a gesture just vanishes with no trace.
                logger.warning("reply_stream: tool call %r has unparseable "
                               "arguments, dropping them: %r",
                               entry["name"], entry["arguments"])
                arguments = {}
            yield {"type": "tool_call", "name": entry["name"], "arguments": arguments}
