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

"""InflectSynthesizer.speak on empty text.

An empty phrase (e.g. everything got cut by split_into_phrases) must return
an empty array via the short-circuit path, WITHOUT touching the model: the
`__init__` constructor loads the espeak frontend and LiteRT weights from
disk, which is heavy and not what needs checking here. The instance is
built without `__init__` (object.__new__), so any access to self._tts would
raise AttributeError — if speak() failed to short-circuit on empty text, the
test would catch that as an error rather than passing by accident.
"""

import numpy as np

from emulator.inflect_tts import SAMPLE_RATE, InflectSynthesizer


def _bare_synth() -> InflectSynthesizer:
    """Instance without loading the model — only speak() should ever run here."""
    return object.__new__(InflectSynthesizer)


def test_speak_empty_text_returns_empty_float32_array():
    synth = _bare_synth()
    out = synth.speak("")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.size == 0


def test_speak_whitespace_only_text_is_also_short_circuited():
    synth = _bare_synth()
    out = synth.speak("   \n\t  ")
    assert out.size == 0


def test_sample_rate_matches_measured_inflect_output():
    assert SAMPLE_RATE == 24000
