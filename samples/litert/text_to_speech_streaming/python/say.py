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
"""KittenTTS nano streaming demo CLI (Raspberry Pi friendly).

Synthesizes text sentence by sentence — the same streaming granularity as the
Android sample — printing time-to-first-audio and per-sentence RTF, and writes
the result as a 24 kHz WAV.

    python say.py "Hello there! This runs fully offline." -o hello.wav
    python say.py --voice expr-voice-3-f --speed 1.1

Runtime dependencies: numpy + ai-edge-litert (pip install ai-edge-litert).
The model files are built by ../conversion/ (see README.md).
"""

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

import kitten_tts

_HERE = Path(__file__).resolve().parent
_DEFAULT_MODELS_DIR = _HERE.parent / "out"
_DEFAULT_ASSETS_DIR = (
    _HERE.parent / "kotlin_cpu" / "android" / "app" / "src" / "main" / "assets"
)
_DEFAULT_TEXT = (
    "Hi there! I am a tiny voice with fifteen million parameters. "
    "I live entirely on this device, so nothing you type ever leaves it. "
    "I am speaking to you in a stream, sentence by sentence, while the rest "
    "is still being synthesized."
)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    """Writes float32 PCM in [-1, 1] as a 16-bit mono WAV."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(kitten_tts.SAMPLE_RATE)
        wav_file.writeframes(
            (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())


def main() -> int:
    """Runs the streaming synthesis demo."""
    parser = argparse.ArgumentParser(
        description="KittenTTS nano on LiteRT — streaming TTS demo")
    parser.add_argument("text", nargs="?", default=_DEFAULT_TEXT,
                        help="text to speak (default: the app's demo text)")
    parser.add_argument("-o", "--output", type=Path, default=Path("say.wav"),
                        help="output WAV path (default: say.wav)")
    parser.add_argument("--models-dir", type=Path,
                        default=_DEFAULT_MODELS_DIR,
                        help="directory with the model files "
                             "(default: ../out, the conversion output)")
    parser.add_argument("--assets-dir", type=Path,
                        default=_DEFAULT_ASSETS_DIR,
                        help="directory with the G2P/host tables "
                             "(default: the Android app's assets)")
    parser.add_argument("--voice", default="expr-voice-5-m",
                        choices=kitten_tts.VOICES,
                        help="voice (default: expr-voice-5-m, as the app)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="speech speed, 1.0 = normal")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads for the LSTM graphs (default: 4)")
    args = parser.parse_args()

    try:
        engine = kitten_tts.KittenTTS(
            args.models_dir, args.assets_dir, args.voice, args.threads)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    engine.warm_up()
    print(f"voice={args.voice} speed={args.speed} threads={args.threads}")

    start = time.perf_counter()
    first_audio_seconds = None
    pieces = []
    for result in engine.stream(args.text, args.speed):
        if first_audio_seconds is None:
            first_audio_seconds = time.perf_counter() - start
        audio_seconds = result.audio.size / kitten_tts.SAMPLE_RATE
        rtf = result.synthesis_seconds / audio_seconds
        print(f"  [{time.perf_counter() - start:6.2f}s] "
              f"{audio_seconds:5.2f}s audio  RTF={rtf:.3f}  "
              f"{result.sentence[:60]}")
        pieces.append(result.audio)

    if not pieces:
        print("error: no sentences found in the input text", file=sys.stderr)
        return 1

    audio = np.concatenate(pieces)
    total_seconds = time.perf_counter() - start
    audio_seconds = audio.size / kitten_tts.SAMPLE_RATE
    _write_wav(args.output, audio)
    print(f"wrote {args.output}: {audio_seconds:.2f}s audio in "
          f"{total_seconds:.2f}s (TTFA {first_audio_seconds:.2f}s, "
          f"overall RTF {total_seconds / audio_seconds:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
