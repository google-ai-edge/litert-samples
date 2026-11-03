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

# Copies the model file into the app's external files dir, where the app reads it (no storage permission needed), and
# restarts the app so that it loads the model.
#   ./install_to_device.sh <dir with LFM3-1.7B_mixed_int4_ekv2048.litertlm>
# Install the app first (./gradlew :app:installDebug). The app must create that dir itself, so the script opens the
# app once when the dir is not there yet. `adb -s <serial>` via ANDROID_SERIAL.
set -euo pipefail
SRC="${1:?usage: $0 <dir with the .litertlm file>}"
FILE="LFM3-1.7B_mixed_int4_ekv2048.litertlm"
PKG="com.google.ai.edge.examples.offline_planner"
DEST="/sdcard/Android/data/$PKG/files"
[ -f "$SRC/$FILE" ] || { echo "missing: $SRC/$FILE"; exit 1; }
adb shell pm list packages "$PKG" | grep -q "$PKG" || { echo "install the app first: ./gradlew :app:installDebug"; exit 1; }
if ! adb shell ls -d "$DEST" > /dev/null 2>&1; then
  adb shell am start -n "$PKG/.MainActivity" > /dev/null
  for _ in $(seq 20); do adb shell ls -d "$DEST" > /dev/null 2>&1 && break; sleep 0.5; done
  adb shell ls -d "$DEST" > /dev/null 2>&1 || { echo "the app did not create $DEST; open it once and run the script again"; exit 1; }
fi
# GNU stat (Linux) first: there `-f` means the file system and does not fail. BSD stat (macOS) has no `-c`.
want=$(stat -L -c %s "$SRC/$FILE" 2>/dev/null || stat -L -f %z "$SRC/$FILE")
got=$(adb shell stat -c %s "$DEST/$FILE" 2>/dev/null | tr -d '\r' || true)
if [ "$got" = "$want" ]; then echo "already there: $FILE ($got bytes)"; exit 0; fi
adb push "$SRC/$FILE" "$DEST/$FILE"
got=$(adb shell stat -c %s "$DEST/$FILE" | tr -d '\r')
[ "$got" = "$want" ] || { echo "$FILE: $got bytes on the phone, $want expected; run the script again"; exit 1; }
# An app that is already running looked for the file when it started: start it again so that it loads the model.
adb shell am start -S -n "$PKG/.MainActivity" > /dev/null
echo "copied $FILE ($got bytes) to $DEST and restarted the app: it loads the model on the GPU, then says Ready"
