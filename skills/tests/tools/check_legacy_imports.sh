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
#
# The source search of stage 1 of the litert-runtime migration (step 6 of
# litert-runtime/references/migration.md): lists every reference to the legacy
# TensorFlow Lite runtime, its delegates, the Support Model class and any
# org.tensorflow.lite class left in the Kotlin, Java, Gradle and R8 / ProGuard
# files under a directory. The litert 2.3.0 artifacts still contain
# org.tensorflow.lite.Interpreter, InterpreterApi and DataType, so leftover
# Interpreter code compiles and only a search finds it.
#
#   check_legacy_imports.sh <directory>
#
# Prints each reference and FAIL, and exits 1, when there is one; prints PASS
# and exits 0 when there is none.

set -u

if [ $# -ne 1 ] || [ ! -d "$1" ]; then
  echo "usage: $0 <directory>" >&2
  exit 2
fi

# The pattern of the search in references/migration.md.
pattern='org\.tensorflow\.lite\.|GpuDelegate|NnApiDelegate|FlexDelegate'
pattern+='|support\.model\.Model|org\.tensorflow:tensorflow-lite'

hits=$(grep -rnE "$pattern" \
  --include='*.kt' --include='*.java' --include='*.kts' --include='*.gradle' \
  --include='*.toml' --include='*.pro' \
  --exclude-dir=build --exclude-dir=.gradle "$1")
status=$?
if [ "$status" -gt 1 ]; then
  echo "ERROR: grep exited with $status" >&2
  exit 2
fi

if [ -n "$hits" ]; then
  echo "$hits"
  echo "FAIL: $(echo "$hits" | wc -l | tr -d ' ') legacy reference(s) under $1"
  exit 1
fi
echo "PASS: no legacy reference under $1"
