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

"""Incrementally parse the model's token stream into complete sentences.

Streaming the response into the TTS pays off: the first sentence is ready long
before the full reply finishes (~6.09s), so the robot starts talking several
seconds earlier instead of waiting for the whole thing. To do this the token
stream needs to be cut into sentences as it arrives: as soon as a complete one
has accumulated (ends in . ! ?), hand it off to synthesis, and keep buffering
the tail.

Pure function, tested without a network or a model.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?]+")


def extract_sentences(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences out of the buffer, return them plus the leftover tail.

    A complete sentence ends in . ! ? A tail without a closing mark stays in
    the buffer until the next tokens arrive. Empty pieces (whitespace only)
    are dropped.
    """
    sentences: list[str] = []
    last = 0
    for match in _SENTENCE_END.finditer(buffer):
        piece = buffer[last:match.end()].strip()
        if piece:
            sentences.append(piece)
        last = match.end()
    tail = buffer[last:]
    return sentences, tail


def flush_tail(tail: str) -> str | None:
    """Leftover after the stream ends: a sentence without a closing mark.

    The model may have cut off mid-word (token limit) — if there's text left
    in the tail, it still needs to be spoken, otherwise the end of the
    response is lost.
    """
    cleaned = tail.strip()
    return cleaned or None
