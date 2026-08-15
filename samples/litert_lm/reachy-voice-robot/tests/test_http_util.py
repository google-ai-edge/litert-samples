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

"""The shared HTTP reason-phrase sanitizer.

http.server writes the reason phrase verbatim into the status line, so a
dynamic error message must be made safe: no non-ASCII (would crash the latin-1
encoder) and no control chars (CR/LF could split the response).
"""

from demo.http_util import ascii_reason


def test_replaces_each_non_ascii_codepoint_with_one_question_mark():
    assert ascii_reason("é☃") == "??"      # é, snowman
    assert ascii_reason("ab☃cd") == "ab?cd"


def test_neutralizes_crlf_so_it_cannot_split_the_response():
    out = ascii_reason("bad\r\nX-Injected: 1")
    assert "\r" not in out and "\n" not in out


def test_neutralizes_all_control_chars():
    assert ascii_reason("a\tb\x00c\x7f") == "a b c "


def test_result_is_always_pure_printable_ascii():
    out = ascii_reason("\U0001f525\r\n\x07 é" * 50)
    assert all(0x20 <= ord(c) < 0x7f for c in out)


def test_caps_length():
    assert len(ascii_reason("x" * 500, limit=200)) == 200
