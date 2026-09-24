# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bake a fixed two-speaker voice prefix into an on-device asset.

Dia2 samples the speaker when no prefix is given, so the app's voice
changes on every run. The official fix is a voice prefix, but building one
needs Whisper word timings and a Mimi encode -- both host-only. Everything is
therefore precomputed here and shipped as a small JSON: the device only has to
replay `warmup_with_prefix`, which primes the temporal KV cache and needs no
Mimi encoder.

Emits assets/dia2_prefix.json:
    frames          number of aligned prefix frames
    aligned         [32][frames] undelayed Mimi codes
    new_word_steps  frames at which the state machine must force a new word
    entries         the prefix words: {tokens, padding}
"""
import json
import os
import sys
import wave

import numpy as np
import torch

sys.path.insert(0, "dia2")
from dia2.engine import Dia2
from dia2.generation import PrefixConfig
from dia2.runtime.voice_clone import WhisperWord, build_prefix_plan

ASSETS = os.environ.get("DIA2_ASSETS", "app/src/main/assets")
P1_TEXT = "[S1] Hey there, this is a quick voice sample."
P2_TEXT = "[S2] And this one is the second speaker talking."
P1_SEED = 0     # pyin F0 ~247 Hz
P2_SEED = 15    # pyin F0 ~88 Hz

WORD_TIMINGS: dict[str, list] = {}


def capture(model, text, path, seed):
    """Generates a prefix clip and keeps Dia2's own word timings for it.

    Args:
      model: A loaded Dia2 engine.
      text: The prompt script for this speaker.
      path: Where the prompt wav is written.
      seed: Torch manual seed selecting the sampled voice.
    """
    torch.manual_seed(seed)
    res = model.generate(text, cfg_scale=2.0, output_wav=path)
    duration = res.waveform.shape[-1] / res.sample_rate
    words = [
        WhisperWord(
            text=word, start=float(start),
            end=float(res.timestamps[i + 1][1])
            if i + 1 < len(res.timestamps) else duration)
        for i, (word, start) in enumerate(res.timestamps)
    ]
    WORD_TIMINGS[path] = words
    print(f"  {path}: {duration:.2f}s, {len(words)} words (seed {seed})")


def main():
    """Bakes the two-speaker voice prompt into the app assets."""
    model = Dia2.from_repo("nari-labs/Dia2-1B")
    model.set_device("cpu")
    # Build before seeding: lazy weight loading would otherwise consume the RNG.
    runtime = model._ensure_runtime()

    print("generating prefix clips")
    capture(model, P1_TEXT, "prefix_speaker1.wav", P1_SEED)
    capture(model, P2_TEXT, "prefix_speaker2.wav", P2_SEED)

    plan = build_prefix_plan(
        runtime,
        PrefixConfig(speaker_1="prefix_speaker1.wav",
                     speaker_2="prefix_speaker2.wav",
                     include_audio=False),
        transcribe_fn=lambda path, device: WORD_TIMINGS[str(path)],
    )
    aligned = plan.aligned_tokens.cpu().numpy().astype(int)
    print(f"aligned tokens {aligned.shape}  frames={plan.aligned_frames}  "
          f"entries={len(plan.entries)}  new_word_steps={plan.new_word_steps}")

    payload = {
        "frames": int(plan.aligned_frames),
        "aligned": aligned.tolist(),
        "new_word_steps": [int(s) for s in plan.new_word_steps],
        "entries": [
            {"tokens": [int(t) for t in e.tokens], "padding": int(e.padding)}
            for e in plan.entries],
    }
    os.makedirs(ASSETS, exist_ok=True)
    out = os.path.join(ASSETS, "dia2_prefix.json")
    with open(out, "w") as handle:
        json.dump(payload, handle)
    print(f"wrote {out} ({os.path.getsize(out) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
