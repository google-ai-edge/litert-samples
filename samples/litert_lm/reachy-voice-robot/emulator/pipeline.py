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

"""Assembling the pipeline: when each stage fires.

The key decision is event-driven triggering. The language model costs ~1.2 s
to the first token, so waking it on every frame is out of the question. The
detector runs continuously (a few hundred ms per frame), but the heavy model only wakes
up when the set of objects in frame changed or the person said something.

The reply is spoken phrase by phrase. The synthesizer is hidden behind a single
interface `speak(text) -> samples` with a `sample_rate` attribute, so the
pipeline doesn't depend on which synthesizer is underneath. An empty array from
`speak` means the phrase couldn't be spoken; the pipeline counts it as skipped —
silence beats a mangled utterance. Inflect-Nano-v2's espeak frontend can voice
any text, so in practice it never returns empty, but the pipeline handles that
case defensively all the same.
"""

from __future__ import annotations

import numpy as np

from emulator.detector import Detection, scene_changed
from emulator.llm import build_prompt, split_into_phrases
from emulator.timeline import Timeline

# All 80 COCO classes the YOLO detector can emit, named for the robot. Names are kept
# SHORT on purpose: the whole (system + user) prompt is held under a prefill-
# cliff budget (~490 chars, see tests/test_llm.py), and the full COCO names
# ("baseball glove", "traffic light", "dining table") would push the worst case
# over it — so the long ones are abbreviated (glove, stoplight, table, ...). The
# "objectN" fallback then only fires for an out-of-range index, which a valid
# detection never has.
COCO_NAMES = (
    "person", "bicycle", "car", "motorbike", "airplane", "bus", "train",
    "truck", "boat", "stoplight", "hydrant", "stop sign", "meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "ball", "kite", "bat", "glove", "skate",
    "surfboard", "racket", "bottle", "glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "plant", "bed",
    "table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "phone",
    "microwave", "oven", "toaster", "sink", "fridge", "book", "clock", "vase",
    "scissors", "plushy", "drier", "brush",
)


def label_name(label: int) -> str:
    return COCO_NAMES[label] if 0 <= label < len(COCO_NAMES) else f"object{label}"


def gaze_target(detections: list[Detection]) -> tuple[float, float]:
    """Where to look: center of the most confident box, in [-1, 1] coords.

    An empty frame means looking straight ahead, not at a random point.
    """
    if not detections:
        return (0.0, 0.0)
    best = max(detections, key=lambda d: d.score)
    x1, y1, x2, y2 = best.box
    return ((x1 + x2) - 1.0, (y1 + y2) - 1.0)


class Pipeline:
    def __init__(self, detector, recognizer, llm, synthesizer, robot) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.llm = llm
        self.synthesizer = synthesizer
        self.robot = robot
        self._previous: list[Detection] = []
        self._first_step = True

    def step(self, frame: np.ndarray, pcm: np.ndarray,
             timeline: Timeline) -> dict:
        with timeline.stage("detector"):
            detections = self.detector.detect(frame)

        with timeline.stage("asr_full"):
            heard = self.recognizer.transcribe(pcm)

        changed = self._first_step or scene_changed(self._previous, detections)
        self._first_step = False
        wake = changed or bool(heard.strip())

        result = {
            "objects": [label_name(d.label) for d in detections],
            "heard": heard,
            "scene_changed": changed,
            "woke_llm": wake,
            "reply": "",
            "phrases": [],
            "spoken": [],
            "skipped": [],
            "speech_s": 0.0,
        }

        if not wake:
            self._previous = detections
            return result

        prompt = build_prompt(objects=result["objects"], heard=heard)
        result["prompt_chars"] = len(prompt)
        with timeline.stage("llm_reply"):
            reply = self.llm.reply(prompt)
        result["reply"] = reply

        self.robot.look_at(*gaze_target(detections))

        phrases = split_into_phrases(reply)
        result["phrases"] = phrases
        sample_rate = getattr(self.synthesizer, "sample_rate", 24000)
        for phrase in phrases:
            with timeline.stage("tts_call"):
                audio = self.synthesizer.speak(phrase)
            if len(audio) == 0:
                # The synthesizer couldn't voice the phrase. The skip shows up
                # in the record; silence beats a mangled utterance.
                result["skipped"].append(phrase)
                continue
            self.robot.say(audio, sample_rate=sample_rate)
            result["spoken"].append(phrase)
            result["speech_s"] += len(audio) / sample_rate

        self._previous = detections
        return result
