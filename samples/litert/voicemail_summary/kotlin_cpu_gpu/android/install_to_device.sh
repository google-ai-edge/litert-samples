#!/usr/bin/env bash
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

# Pushes the model files into the app's external files dir (the app reads them there; no storage permission needed).
#   ./install_to_device.sh <dir with the model files>
# The dir holds the five files of README.md (the .litertlm bundle and four .tflite graphs) and, for the NPU build on a
# Galaxy S26 (SM8850), the optional sixth (lfm3_audio_depth_pass_r5b_aot.tflite). Install the app first
# (./gradlew :app:installDebug) so that the app's dir exists; `adb -s <serial>` via ANDROID_SERIAL.
set -euo pipefail
SRC=${1:?usage: $0 <dir with the model files>}
PKG=com.google.ai.edge.examples.voicemail_summary
DST=/sdcard/Android/data/$PKG/files
FILES=(
  lfm3-audio-2b_int4_b128_c2048_h30.litertlm
  lfm3_audio_depth_proj_r4b.tflite
  lfm3_audio_depth_pass_r4b_int8.tflite
  lfm3_audio_frame_embed_f32.tflite
  lfm3_audio_codec_decoder_f12_r4b2.tflite
)
OPTIONAL=(lfm3_audio_depth_pass_r5b_aot.tflite)
adb shell pm list packages "$PKG" | grep -q "$PKG" || { echo "install the app first: ./gradlew :app:installDebug"; exit 1; }
adb shell mkdir -p "$DST"
push() {  # file: pushed unless the phone has it at the same size; the size read back after the push
  local f=$1 want got
  want=$(stat -L -f %z "$SRC/$f" 2>/dev/null || stat -L -c %s "$SRC/$f")
  got=$(adb shell stat -c %s "$DST/$f" 2>/dev/null | tr -d '\r' || true)
  if [ "$got" = "$want" ]; then echo "already there: $f"; return; fi
  echo "pushing $f ($want bytes)"
  adb push "$SRC/$f" "$DST/$f"
  got=$(adb shell stat -c %s "$DST/$f" 2>/dev/null | tr -d '\r' || true)
  [ "$got" = "$want" ] || { echo "$f: $got bytes on the phone, $want expected; run the script again"; exit 1; }
}
for f in "${FILES[@]}"; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f"; exit 1; }
  push "$f"
done
for f in "${OPTIONAL[@]}"; do [ -f "$SRC/$f" ] && push "$f"; done
echo "done; open the app (the first start copies three sections out of the bundle once, then says Ready)"
