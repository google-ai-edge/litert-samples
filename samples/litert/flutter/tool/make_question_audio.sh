#!/bin/sh
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

# Regenerates the spoken questions the integration tests play into the microphone (test_assets/q_*.wav,
# test_assets/france_16k.pcm and test_assets/showcase/q_*.wav) with the app's own speech synthesizer,
# Inflect-nano-v2 (Apache-2.0, built into the app), on a Mac:
#
#   tool/make_question_audio.sh
#
# Runs integration_test/tools/make_question_audio_test.dart on macOS: the app's TtsService loads the
# built-in Inflect bundle exactly as setup does and speaks each question of the list at the top of that
# test (the source of truth for the wording); the 24 kHz speech is resampled to 16 kHz, padded with
# 150 ms of silence at each end and written as 16 kHz mono PCM16 (WAV, or raw samples for the .pcm)
# into the app's sandbox container. This script then copies the files into test_assets/ and prints
# each one's size and duration.
#
# Needs the built-in models, which are not in git: run tool/fetch_models.sh once per checkout (this
# script verifies them first with --check). Keep the test app's window visible and the screen unlocked
# while it runs. Afterwards, update the tests that pin a clip's exact size (e.g.
# integration_test/voice_loop_test.dart for france_16k.pcm).
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# The clips must come from the verified built-in Inflect files.
tool/fetch_models.sh --check

FLUTTER=flutter
if command -v fvm >/dev/null 2>&1; then
  FLUTTER="fvm flutter"
fi

# Where the test writes (getApplicationDocumentsDirectory in the macOS sandbox). The test empties it
# first; the stamp proves every file there was written by this run.
BUNDLE_ID=com.google.ai.edge.examples.litertEdgeDemos
OUT=$HOME/Library/Containers/$BUNDLE_ID/Data/Documents/question_audio
STAMP=$(mktemp "${TMPDIR:-/tmp}/make_question_audio.XXXXXX")
trap 'rm -f "$STAMP"' EXIT

# Word splitting of $FLUTTER is intended ("fvm flutter").
$FLUTTER test integration_test/tools/make_question_audio_test.dart -d macos

if [ ! -d "$OUT" ]; then
  echo "error: the test wrote no $OUT" >&2
  exit 1
fi
STALE=$(find "$OUT" -type f ! -newer "$STAMP" | wc -l | tr -d ' ')
if [ "$STALE" -ne 0 ]; then
  echo "error: $OUT holds $STALE file(s) older than this run" >&2
  exit 1
fi
COUNT=$( (cd "$OUT" && find . -type f \( -name '*.wav' -o -name '*.pcm' \)) | wc -l | tr -d ' ')
if [ "$COUNT" -eq 0 ]; then
  echo "error: no .wav or .pcm file in $OUT" >&2
  exit 1
fi

# Every file the test wrote, at the same path under test_assets/. Duration from the sample bytes:
# 16 kHz mono PCM16 is 32000 bytes per second; the WAVs carry the canonical 44-byte header.
(cd "$OUT" && find . -type f \( -name '*.wav' -o -name '*.pcm' \)) | sort |
  while IFS= read -r f; do
    f=${f#./}
    mkdir -p "test_assets/$(dirname "$f")"
    cp "$OUT/$f" "test_assets/$f"
    bytes=$(wc -c < "test_assets/$f" | tr -d ' ')
    case $f in
      *.wav) data=$((bytes - 44)) ;;
      *) data=$bytes ;;
    esac
    secs=$(awk -v b="$data" 'BEGIN { printf "%.3f", b / 32000 }')
    printf '%-38s %7s bytes  %s s\n' "test_assets/$f" "$bytes" "$secs"
  done
echo "Copied $COUNT clips. Update the tests that pin a clip's size or duration" \
  "(integration_test/voice_loop_test.dart: france_16k.pcm's byte length)."
