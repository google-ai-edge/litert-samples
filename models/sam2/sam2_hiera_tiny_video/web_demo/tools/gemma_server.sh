#!/usr/bin/env bash
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
# ==============================================================================

# Starts Gemma 4 on LiteRT-LM for the demo's "Ask Gemma" (text -> boxes):
#   1. installs the LiteRT-LM CLI into its own venv (if missing),
#   2. imports Gemma 4 E4B and E2B from Hugging Face (if missing; 3.7 / 2.6 GB,
#      the full .litertlm files: the web-only builds are text-only),
#   3. runs tools/gemma_server.py on port 9379 (OpenAI-compatible, allowing
#      requests from the demo's origin): the LiteRT-LM Python API with warm
#      engines, the language model AND the vision encoder on the GPU, streamed
#      replies. `litert-lm serve --config tools/litert_lm_config.json` also
#      works (same API), but sets up constrained decoding on every request
#      (~1 s slower) and the demo then gets its boxes only at the end.
#
#   tools/gemma_server.sh [origin ...]      default origin: http://localhost:5175
#   VENV=<dir> to use another venv; MODELS="e4b e2b" to choose which to import.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$(dirname "$HERE")/.venv-litertlm}"
MODELS="${MODELS:-e4b e2b}"
ORIGINS=("$@")
[ ${#ORIGINS[@]} -gt 0 ] || ORIGINS=(http://localhost:5175)

if [ ! -x "$VENV/bin/litert-lm" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q litert-lm==0.17.1
fi
LM="$VENV/bin/litert-lm"
for m in $MODELS; do
  id="gemma-4-$m"
  if ! "$LM" list 2>/dev/null | grep -q "^$id "; then
    M=$(echo "$m" | tr a-z A-Z)
    "$LM" import --from-huggingface-repo "litert-community/gemma-4-${M}-it-litert-lm" "gemma-4-${M}-it.litertlm" "$id"
  fi
done
"$LM" list
args=()
for o in "${ORIGINS[@]}"; do args+=(--cors-origin "$o"); done
exec "$VENV/bin/python" "$HERE/gemma_server.py" --port 9379 "${args[@]}"
