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
export LITERT_CHECKOUT="${LITERT_CHECKOUT:-../../../../../LiteRT}"
for tool in git xcodegen; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool" >&2
    exit 1
  fi
done
if [ ! -f "$LITERT_CHECKOUT/Package.swift" ]; then
  echo "LiteRT checkout not found. Set LITERT_CHECKOUT or use the default sibling location." >&2
  exit 1
fi
TESTED_REVISION=ccff78483e972a975b5300242084a6e2147c9776
LITERT_COMMIT="$(git -C "$LITERT_CHECKOUT" rev-parse HEAD)"
export LITERT_COMMIT
if [ "$LITERT_COMMIT" != "$TESTED_REVISION" ]; then
  echo "WARNING: tested LiteRT revision $TESTED_REVISION; found $LITERT_COMMIT. Continuing." >&2
fi
xcodegen generate --spec project.yml
