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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LITERT_CHECKOUT="${LITERT_CHECKOUT:-$SCRIPT_DIR/../../../../../LiteRT}"
if [[ "$LITERT_CHECKOUT" != /* ]]; then
  LITERT_CHECKOUT="$SCRIPT_DIR/$LITERT_CHECKOUT"
fi
TESTED_REVISION=ccff78483e972a975b5300242084a6e2147c9776
for tool in git bazelisk xcodebuild unzip zip shasum; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool" >&2
    exit 1
  fi
done
if [ ! -f "$LITERT_CHECKOUT/Package.swift" ]; then
  echo "LiteRT checkout not found. Set LITERT_CHECKOUT or use the default sibling location." >&2
  exit 1
fi
LITERT_CHECKOUT="$(cd "$LITERT_CHECKOUT" && pwd)"
REVISION="$(git -C "$LITERT_CHECKOUT" rev-parse HEAD)"
if [ "$REVISION" != "$TESTED_REVISION" ]; then
  echo "WARNING: tested LiteRT revision $TESTED_REVISION; found $REVISION. Continuing." >&2
fi
(
  cd "$LITERT_CHECKOUT"
  bazelisk build -c opt //litert/swift:CLiteRT //litert/swift:LiteRtMetalAccelerator //litert/swift:CLiteRT_mac
)
BIN="$LITERT_CHECKOUT/bazel-bin/litert/swift"
mkdir -p "$LITERT_CHECKOUT/prebuilt"
for name in CLiteRT LiteRtMetalAccelerator; do
  # Bazel outputs are read-only; replace staged archives on reruns.
  staged="$LITERT_CHECKOUT/prebuilt/$name.xcframework.zip"
  cp -f "$BIN/$name.xcframework.zip" "$staged.new"
  chmod u+w "$staged.new"
  mv -f "$staged.new" "$staged"
done

# The macOS target emits a plain zip, so package its library and C module map.
WORK="$SCRIPT_DIR/build/litert_mac"
mkdir -p "$WORK"
SOURCE_HASH="$(shasum -a 256 "$BIN/CLiteRT_mac.zip" | cut -d ' ' -f 1)"
MAC_ZIP="$LITERT_CHECKOUT/prebuilt/CLiteRT_mac.xcframework.zip"
if [ ! -f "$MAC_ZIP" ] || [ ! -f "$WORK/source.sha256" ] || [ "$(cat "$WORK/source.sha256")" != "$SOURCE_HASH" ]; then
  rm -rf "$WORK/CLiteRT_mac" "$WORK/CLiteRT_mac.xcframework"
  unzip -q -o "$BIN/CLiteRT_mac.zip" -d "$WORK"
  xcodebuild -create-xcframework \
    -library "$WORK/CLiteRT_mac/libCLiteRT_mac.dylib" \
    -headers "$WORK/CLiteRT_mac/Headers" \
    -output "$WORK/CLiteRT_mac.xcframework"
  rm -f "$MAC_ZIP"
  (cd "$WORK" && zip -q -r "$MAC_ZIP" CLiteRT_mac.xcframework)
  printf '%s\n' "$SOURCE_HASH" > "$WORK/source.sha256"
fi
for name in CLiteRT LiteRtMetalAccelerator CLiteRT_mac; do
  unzip -tq "$LITERT_CHECKOUT/prebuilt/$name.xcframework.zip"
done
echo "Prepared LiteRT Swift package archives from revision $REVISION."
