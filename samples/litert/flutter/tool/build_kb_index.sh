#!/bin/bash
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

# Rebuilds assets/kb_index/ — the prebuilt knowledge-base index the app
# installs on first launch instead of embedding every chunk on the device —
# on a Mac:
#
#   tool/build_kb_index.sh
#
# Runs integration_test/build_kb_index_test.dart on macOS (the built-in
# EmbeddingGemma 2 740m .litertlm on the CPU, loaded as the app loads it;
# documents embedded as text; the app's own repository and sqlite-vec store),
# then copies kb.db and manifest.json out of the app's sandbox container. The
# manifest records the index key and the RAG profile id kb.db is bound to.
# Rebuild after changing assets/kb, the chunker, the embedder file, the RAG
# profile id (kKbEmbeddingProfileId) or flutter_edge_ai_sqlite:
# test/data/services/knowledge/kb_prebuilt_asset_test.dart compares the
# manifest with each of them (the key, the profile id, the store's version)
# and fails until you do. Keep the test app's window visible while it runs.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# The index must come from the verified built-in EmbeddingGemma 2 (not in git;
# public, so tool/fetch_models.sh needs no token).
tool/fetch_models.sh --check

FLUTTER="flutter"
command -v fvm >/dev/null 2>&1 && FLUTTER="fvm flutter"

# The store writes kb.db in its own vec0 layout: record which release.
STORE_VERSION=$(awk '
  /^  flutter_edge_ai_sqlite:$/ { found = 1; next }
  found && /^    version:/ { gsub(/"/, "", $2); print $2; exit }
' pubspec.lock)
if [ -z "$STORE_VERSION" ]; then
  echo "error: flutter_edge_ai_sqlite is not in pubspec.lock" >&2
  exit 1
fi

LOG=$(mktemp -t build_kb_index.XXXXXX)
trap 'rm -f "$LOG"' EXIT
$FLUTTER test integration_test/build_kb_index_test.dart -d macos \
  --dart-define=KB_INDEX_STORE="flutter_edge_ai_sqlite $STORE_VERSION" \
  2>&1 | tee "$LOG"

OUT=$(sed -n 's/.*KB_INDEX_OUT=//p' "$LOG" | tail -n 1)
if [ -z "$OUT" ] || [ ! -f "$OUT/kb.db" ] || [ ! -f "$OUT/manifest.json" ]; then
  echo "error: the test printed no KB_INDEX_OUT with kb.db and manifest.json" >&2
  exit 1
fi
mkdir -p assets/kb_index
cp "$OUT/kb.db" "$OUT/manifest.json" assets/kb_index/
echo "assets/kb_index/kb.db: $(wc -c < assets/kb_index/kb.db | tr -d ' ') bytes"
grep -E '"(profileId|chunks|store|builtOn)"' assets/kb_index/manifest.json
