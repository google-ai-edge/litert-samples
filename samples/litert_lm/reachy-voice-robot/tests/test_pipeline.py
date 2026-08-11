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

"""Assembling the pipeline: stage order and when the heavy model wakes up.

The real models are verified by live runs in their own test suites, so here
the stages are fakes — logic matters, not inference. The main property under
test: the language model costs ~1.2 s and must not wake on every frame.
"""

import numpy as np

from emulator.detector import Detection
from emulator.pipeline import COCO_NAMES, Pipeline, gaze_target, label_name
from emulator.robot import RobotStub
from emulator.timeline import Timeline

PERSON = [Detection(label=0, score=0.9, box=(0.0, 0.0, 1.0, 1.0))]
PERSON_CUP = PERSON + [Detection(label=41, score=0.8, box=(0.0, 0.0, 1.0, 1.0))]
FRAME = np.zeros((640, 640, 3), dtype=np.float32)
SILENCE = np.zeros(80000, dtype=np.float32)


class FakeDetector:
    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.calls = 0

    def detect(self, frame):
        result = self._sequence[min(self.calls, len(self._sequence) - 1)]
        self.calls += 1
        return result

    def input_shape(self):
        return (640, 640, 3)


class FakeRecognizer:
    def __init__(self, text=""):
        self.text = text

    def transcribe(self, pcm):
        return self.text


class FakeLlm:
    def __init__(self, reply="I can see you."):
        self.prompts = []
        self._reply = reply

    def reply(self, prompt):
        self.prompts.append(prompt)
        return self._reply


class FakeSynth:
    sample_rate = 16000

    def __init__(self, produce=True):
        self.inputs = []
        self._produce = produce

    def speak(self, text):
        self.inputs.append(text)
        # Empty array = "couldn't voice it", non-empty = voiced.
        return np.zeros(16000, dtype=np.float32) if self._produce \
            else np.zeros(0, dtype=np.float32)


def make(detections, heard="", reply="I can see you.", produce=True):
    return Pipeline(
        detector=FakeDetector(detections),
        recognizer=FakeRecognizer(heard),
        llm=FakeLlm(reply),
        synthesizer=FakeSynth(produce=produce),
        robot=RobotStub(),
    )


# --- when the language model wakes up ---

def test_llm_not_woken_when_scene_unchanged():
    pipe = make([PERSON, PERSON])
    pipe.step(FRAME, SILENCE, Timeline())
    pipe.step(FRAME, SILENCE, Timeline())
    # The language model costs ~1.2 s TTFT — waking it every frame is out.
    assert len(pipe.llm.prompts) == 1


def test_llm_woken_when_new_object_appears():
    pipe = make([PERSON, PERSON_CUP])
    pipe.step(FRAME, SILENCE, Timeline())
    pipe.step(FRAME, SILENCE, Timeline())
    assert len(pipe.llm.prompts) == 2


def test_speech_wakes_llm_even_without_scene_change():
    pipe = make([PERSON, PERSON], heard="hello robot")
    pipe.step(FRAME, SILENCE, Timeline())
    pipe.step(FRAME, SILENCE, Timeline())
    assert len(pipe.llm.prompts) == 2
    assert "hello robot" in pipe.llm.prompts[-1]


def test_first_step_always_wakes_llm():
    # On the first step there's nothing to compare against, and the robot
    # must not stay silent.
    pipe = make([PERSON])
    result = pipe.step(FRAME, SILENCE, Timeline())
    assert result["woke_llm"] is True


# --- speech output ---

def test_synthesizer_receives_phrase_text():
    # The pipeline is TTS-agnostic: it hands the synthesizer the phrase text,
    # and phonemization (if needed) is the synthesizer's own business.
    pipe = make([PERSON], reply="I can see you.")
    pipe.step(FRAME, SILENCE, Timeline())
    assert pipe.synthesizer.inputs == ["I can see you."]


def test_phrase_the_synthesizer_cannot_say_is_skipped():
    # An empty array from speak = "couldn't voice it". The phrase goes
    # into skipped, the robot stays silent about it — silence beats a
    # mangled utterance.
    pipe = make([PERSON], reply="Zorblax frotzed.", produce=False)
    result = pipe.step(FRAME, SILENCE, Timeline())
    assert result["skipped"]
    assert not result["spoken"]
    kinds = {name for name, _ in pipe.robot.events()}
    assert "say" not in kinds


def test_reply_is_split_into_window_sized_chunks():
    long_reply = ("I can see you. " * 6).strip()
    pipe = make([PERSON], reply=long_reply)
    result = pipe.step(FRAME, SILENCE, Timeline())
    assert len(result["phrases"]) > 1
    # Each chunk is at most one full synthesis window.
    assert all(len(p) <= 45 for p in result["phrases"])


# --- timeline and robot ---

def test_timeline_records_every_stage():
    pipe = make([PERSON])
    timeline = Timeline()
    pipe.step(FRAME, SILENCE, timeline)
    names = {s.name for s in timeline.spans()}
    assert {"detector", "asr_full", "llm_reply", "tts_call"} <= names


def test_robot_gets_speech_and_gaze():
    pipe = make([[Detection(label=0, score=0.9, box=(0.2, 0.2, 0.4, 0.4))]])
    pipe.step(FRAME, SILENCE, Timeline())
    kinds = {name for name, _ in pipe.robot.events()}
    assert "say" in kinds
    assert "look_at" in kinds


def test_gaze_centres_on_most_confident_box():
    strong = Detection(label=0, score=0.95, box=(0.6, 0.6, 0.8, 0.8))
    weak = Detection(label=41, score=0.4, box=(0.0, 0.0, 0.2, 0.2))
    x, y = gaze_target([weak, strong])
    assert x > 0 and y > 0, "gaze should head toward the confident detection"


def test_gaze_is_straight_ahead_on_empty_frame():
    assert gaze_target([]) == (0.0, 0.0)


def test_label_name_maps_coco_indices():
    # A dropped/inserted COCO_NAMES entry would silently shift every later label
    # (dog->horse); pin the count and a spread of indices to the canonical
    # COCO-80 order.
    assert len(COCO_NAMES) == 80
    assert label_name(0) == "person"
    assert label_name(16) == "dog"
    assert label_name(41) == "cup"
    assert label_name(67) == "phone"
    assert label_name(79) == "brush"


def test_unknown_class_is_numbered_not_renamed():
    # Lying with a name is worse than labeling with a number — out of range in
    # either direction must fall back to objectN, not wrap around.
    assert label_name(80) == "object80"       # first index past the end
    assert label_name(777) == "object777"
    assert label_name(-1) == "object-1"        # negative must not read brush
