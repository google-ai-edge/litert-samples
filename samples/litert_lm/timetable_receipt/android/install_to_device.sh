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

# Pushes the model file into the app's external files dir (the app reads it there; no storage permission needed).
#   ./install_to_device.sh <dir with LFM3-VL-2B_f16.litertlm>
# Install the app and open it once first: the app creates its files dir (a dir made by adb shell before the app's first
# start can be unreadable to the app). `adb -s <serial>` via ANDROID_SERIAL.
set -euo pipefail
SRC=${1:?usage: $0 <dir with LFM3-VL-2B_f16.litertlm>}
PKG=com.google.ai.edge.examples.timetable_receipt
DST=/sdcard/Android/data/$PKG/files
F=LFM3-VL-2B_f16.litertlm
[ -f "$SRC/$F" ] || { echo "missing $SRC/$F"; exit 1; }
[ -n "$(adb shell pm list packages "$PKG" | tr -d '\r')" ] || { echo "install the app first: ./gradlew :app:installDebug"; exit 1; }
[ "$(adb shell "[ -d $DST ] && echo yes" | tr -d '\r')" = yes ] || { echo "open the app once first: it creates $DST"; exit 1; }
# GNU stat first (Linux); BSD stat (macOS) rejects -c and takes -f.
want=$(stat -L -c %s "$SRC/$F" 2>/dev/null || stat -L -f %z "$SRC/$F")
got=$(adb shell stat -c %s "$DST/$F" 2>/dev/null | tr -d '\r' || true)
if [ "$got" = "$want" ]; then echo "already there: $F"; exit 0; fi
echo "pushing $F ($want bytes)"
adb push "$SRC/$F" "$DST/$F"
got=$(adb shell stat -c %s "$DST/$F" 2>/dev/null | tr -d '\r' || true)
[ "$got" = "$want" ] || { echo "$F: $got bytes on the phone, $want expected; run the script again"; exit 1; }
echo "done; open the app again: it loads the model on the GPU, then says ready"
