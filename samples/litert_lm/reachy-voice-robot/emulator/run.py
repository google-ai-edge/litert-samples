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

"""Emulator entry point.

By default everything is deterministic: a synthetic frame and silence. Live
sources are wired in via flags — that way two runs can be compared against
each other.

The language model, meanwhile, is expected to run as a separate process. Import
the model into LiteRT-LM once (registering it under the id "e2b", the name
emulator/models.py requests), then serve it:

    litert-lm import --from-huggingface-repo litert-community/gemma-4-E2B-it-litert-lm \
                     gemma-4-E2B-it.litertlm e2b
    litert-lm serve --host 127.0.0.1 --port 9379
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from emulator import models
from emulator.asr import WINDOW_SAMPLES, MoonshineTokenizer, Recognizer
from emulator.detector import Detector
from emulator.llm import LlmClient
from emulator.pipeline import Pipeline
from emulator.robot import RobotStub
from emulator.sources import (
    SilenceAudioSource,
    StillFrameSource,
    WavAudioSource,
)
from emulator.timeline import Timeline


def _download(m: models.Model) -> Path:
    """Local path to a model file: a bundled asset if it ships in the repo
    (``dir`` + ``file``), otherwise a Hugging Face download (``repo`` +
    ``file``)."""
    if m.dir is not None:
        if m.file is None:
            raise ValueError(
                f"{m!r} is a bundled directory — load it via its own loader, "
                "not _download")
        path = m.dir / m.file
        if not path.exists():
            raise SystemExit(
                f"bundled model not found: {path} — convert it per "
                f"{m.dir}/README.md before running")
        return path
    return Path(hf_hub_download(repo_id=m.repo, filename=m.file))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice robot emulator")
    parser.add_argument("--wav", type=Path, default=None,
                        help="audio file instead of silence")
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("assets/moonshine_tokenizer.json"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--tts", choices=("inflect",), default="inflect",
                        help="speech synthesizer (Inflect-Nano-v2 on LiteRT)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="no language model: check the other stages")
    return parser.parse_args(argv)


class OfflineLlm:
    """Stand-in language model for runs without a server."""

    def reply(self, prompt: str) -> str:
        return "I can see a person here."


def build_synthesizer(kind: str = "inflect"):
    """Inflect-Nano-v2 on LiteRT — the sample's speech synthesizer.

    Imported inside the function (not at module top) so its espeak-ng frontend
    is only pulled in when a run actually synthesizes. The model is resolved by
    name from the catalog (emulator/models.py).
    """
    from emulator.inflect_tts import load_inflect
    return load_inflect(models_dir=models.get(models.TTS["inflect"]).dir)


def build_pipeline(tokenizer_path: Path, skip_llm: bool,
                   tts: str) -> tuple[Pipeline, tuple[int, int, int]]:
    # Each stage's model is resolved by name from the catalog
    # (emulator/models.py). Threshold 0.3 — the same value the GPU detect
    # service (demo/gpu_detect.py) uses, so the CPU fallback and the live GPU
    # path agree on what counts as a detection.
    detector = Detector(_download(models.get(models.DETECTOR)),
                        score_threshold=0.3)

    recognizer = Recognizer(
        _download(models.get(models.ASR)),
        MoonshineTokenizer(tokenizer_path),
    )

    synthesizer = build_synthesizer(tts)

    if skip_llm:
        llm = OfflineLlm()
    else:
        srv = models.get(models.LLM)
        llm = LlmClient(srv.endpoint, srv.model)

    pipeline = Pipeline(detector=detector, recognizer=recognizer, llm=llm,
                        synthesizer=synthesizer, robot=RobotStub())
    return pipeline, detector.input_shape()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline, frame_shape = build_pipeline(args.tokenizer, args.skip_llm,
                                           args.tts)
    frames = StillFrameSource(shape=frame_shape)
    audio = WavAudioSource(args.wav) if args.wav else SilenceAudioSource()

    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "emulator.jsonl"

    for index in range(args.steps):
        timeline = Timeline()
        result = pipeline.step(frames.read(), audio.read(WINDOW_SAMPLES),
                               timeline)
        record = timeline.to_record() | {"step": index} | result
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        print(f"\n=== step {index + 1} ===")
        print(f"  objects: {result['objects'] or 'none'}")
        print(f"  heard: {result['heard']!r}")
        print(f"  scene changed: {result['scene_changed']}, "
              f"woke model: {result['woke_llm']}")
        if result["reply"]:
            print(f"  reply: {result['reply']!r}")
            print(f"  chunks: {len(result['phrases'])}, "
                  f"spoken {len(result['spoken'])}, "
                  f"skipped {len(result['skipped'])}")
            if result["skipped"]:
                print(f"  synthesizer failed on: {result['skipped']}")
            print(f"  speech produced: {result['speech_s']:.2f} s")
        for span in timeline.spans():
            mark = " FAILED" if span.failed else ""
            print(f"    {span.name:<12} {span.duration_ms:8.0f} ms{mark}")
        print(f"  total wall time: {timeline.total_ms():.0f} ms")
        problems = timeline.check_budget()
        print(f"  over budget: {problems if problems else 'none'}")
        if result["speech_s"] > 0:
            compute = sum(s.duration_ms for s in timeline.spans()
                          if s.name == "tts_call") / 1000
            print(f"  synthesis RTF relative to speech: "
                  f"{compute / result['speech_s']:.2f}")

    print(f"\nevent log: {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
