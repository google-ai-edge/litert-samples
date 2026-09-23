#!/bin/bash
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

set -euo pipefail
cd "$(dirname "$0")"
PRINT_BUILD_COMMAND=0
if [ "${1:-}" = "--print-build-command" ]; then
  PRINT_BUILD_COMMAND=1
  shift
fi
TEXT="${1:-The rain is soft today.}"
SEED="${2:-7}"
STEPS="${3:-10}"
PLACEMENT="${4:-te=cpu,dec=cpu,voc=gpu}"
RUNS="${5:-10}"
BUNDLE_ID="${BUNDLE_ID:-com.google.ai.edge.TextToSpeech}"
SIGNING=()
if [ -n "${SIGNING_TEAM:-}" ]; then
  SIGNING+=("DEVELOPMENT_TEAM=$SIGNING_TEAM")
fi
BUILD=(-project TextToSpeech.xcodeproj -scheme TextToSpeech -configuration Release
  -destination 'generic/platform=iOS' -allowProvisioningUpdates
  "PRODUCT_BUNDLE_IDENTIFIER=$BUNDLE_ID" ${SIGNING[@]+"${SIGNING[@]}"})
if [ "$PRINT_BUILD_COMMAND" = 1 ]; then
  printf '%q ' xcodebuild "${BUILD[@]}" build
  printf '\n'
  exit 0
fi
for tool in xcrun xcodebuild awk; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool" >&2
    exit 1
  fi
done
DEVICE="${DEVICE_ID:?Set DEVICE_ID to the target device identifier.}"
bash generate_project.sh
xcodebuild "${BUILD[@]}" build
SETTINGS="$(xcodebuild "${BUILD[@]}" -showBuildSettings)"
PRODUCTS="$(printf '%s\n' "$SETTINGS" | awk -F ' = ' '$1 ~ / TARGET_BUILD_DIR$/ { print $2; exit }')"
APP="$PRODUCTS/TextToSpeech.app"
if [ -z "$PRODUCTS" ] || [ ! -d "$APP" ]; then
  echo "Unable to locate the built app in TARGET_BUILD_DIR." >&2
  exit 1
fi
xcrun devicectl device install app --device "$DEVICE" "$APP"
xcrun devicectl device process launch --device "$DEVICE" --console --terminate-existing \
  "$BUNDLE_ID" -- -speak "$TEXT" -seed "$SEED" -steps "$STEPS" -placement "$PLACEMENT" -runs "$RUNS"
mkdir -p output
for file in result.json out.wav; do
  xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" --source "Documents/$file" --destination "output/$file"
done
