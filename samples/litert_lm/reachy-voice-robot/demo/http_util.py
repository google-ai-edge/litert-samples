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

"""Small shared helpers for the demo's http.server-based services.

serve.py, gpu_detect.py and display/web.py all expose an HTTP endpoint and all
hit the same sharp edge: http.server encodes the status line's reason phrase as
latin-1, so a non-ASCII codepoint in a *dynamic* reason (an exception message,
a malformed field) makes send_error itself raise UnicodeEncodeError, escaping
the handler. `ascii_reason` is the one place that guards it.
"""

from __future__ import annotations


def ascii_reason(text: str, limit: int = 200) -> str:
    """A send_error reason phrase safe for http.server's latin-1 encoder.

    Replaces any non-ASCII codepoint with '?' and caps the length, so a
    dynamic error message can never make send_error raise. Also neutralizes
    ASCII control chars (CR/LF in particular): http.server writes the reason
    phrase verbatim into the status line, so a raw newline could inject
    headers/body (HTTP response splitting). The current callers all repr-escape
    user input, but this is the shared helper for three servers — mapping every
    control char to a space closes the gap permanently.
    """
    ascii_text = text[:limit].encode("ascii", "replace").decode("ascii")
    return "".join(c if 0x20 <= ord(c) < 0x7f else " " for c in ascii_text)
