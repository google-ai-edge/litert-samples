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
# The dir is the `hf download` dir of README.md: the .litertlm bundle at its top and the three graphs under speech/
# (or all four files side by side). Install the app first (./gradlew :app:installDebug) so that the app's dir exists;
# `adb -s <serial>` via ANDROID_SERIAL.
set -euo pipefail
SRC=${1:?usage: $0 <dir with the model files>}
PKG=com.google.ai.edge.examples.photo_voice_chat
DST=/sdcard/Android/data/$PKG/files
FILES=(
  LFM3-Omni-2B_int8.litertlm
  LFM3-Omni-2B_depth_step_f32.tflite
  LFM3-Omni-2B_frame_embed_f32.tflite
  LFM3-Omni-2B_codec_decoder_f12_f32.tflite
)
adb shell pm list packages "$PKG" | grep -q "$PKG" || { echo "install the app first: ./gradlew :app:installDebug"; exit 1; }
adb shell mkdir -p "$DST"
push() {  # file: pushed unless the phone has it at the same size; the size read back after the push
  local path=$1 f want got
  f=$(basename "$path")
  want=$(stat -L -c %s "$path" 2>/dev/null || stat -L -f %z "$path")  # GNU stat first: its -f reads the file system
  got=$(adb shell stat -c %s "$DST/$f" 2>/dev/null | tr -d '\r' || true)
  if [ "$got" = "$want" ]; then echo "already there: $f"; return; fi
  echo "pushing $f ($want bytes)"
  adb push "$path" "$DST/$f"
  got=$(adb shell stat -c %s "$DST/$f" 2>/dev/null | tr -d '\r' || true)
  [ "$got" = "$want" ] || { echo "$f: $got bytes on the phone, $want expected; run the script again"; exit 1; }
}
for f in "${FILES[@]}"; do
  if [ -f "$SRC/$f" ]; then push "$SRC/$f"
  elif [ -f "$SRC/speech/$f" ]; then push "$SRC/speech/$f"
  else echo "missing $f in $SRC (or $SRC/speech)"; exit 1
  fi
done
echo "done; open the app (the first start copies five sections out of the bundle once, then says Ready)"
