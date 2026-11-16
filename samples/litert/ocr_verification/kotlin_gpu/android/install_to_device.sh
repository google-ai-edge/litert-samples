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

# Copies the files of litert-community/MiniCPM-OCR-Decision-2B-LiteRT that the app reads into its
# private files/models/.
#
#   ./install_to_device.sh [dir-with-the-model-files]
#
# The directory is the README's hf download folder (default
# ~/Downloads/MiniCPM-OCR-Decision-2B-LiteRT): tokenizer.json, tables/tables.safetensors and the
# graph files side by side. The default set is the tokenizer, the tables, the 400- and 576-patch
# towers and the 384-token decision file (5,110,221,205 bytes): it holds a line of printed text.
# MORE="decide_L512_f16sfc.tflite tower_N784_fp16fc.tflite ..." adds other single files of the
# repository (download them too); a request then takes the smallest file that holds it.
# Every file is checked by size before anything is copied and by SHA-256 on the phone after the
# copy; the app checks the SHA-256 again at its first start.
# Install the debug APK first: run-as needs a debuggable package. ANDROID_SERIAL selects the device.
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [dir-with-the-model-files]\n' "$0" >&2
    exit 2
fi
SOURCE_DIR="${1:-$HOME/Downloads/MiniCPM-OCR-Decision-2B-LiteRT}"
PACKAGE=com.google.ai.edge.examples.ocr_verification
TEMP_DIR=/data/local/tmp/ocr_verification_install

# name bytes sha256 (the values the app checks: ModelFiles.kt)
KNOWN=(
    "tokenizer.json 19989325 06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    "tables.safetensors 1021845928 8fa765826c3676521fa3cc6af61f1cc2c783b129c9564051a0abc28d51678986"
    "decide_L256_f16sfc.tflite 2750394480 526daaa7071c1ffb924844856259c9987cef8e6df88cbc2766f78c03286f2999"
    "decide_L384_f16sfc.tflite 2750868256 c93ddd7239325ca92aa84f8fa681e610f07709228ae792afb20a16e12794e832"
    "decide_L512_f16sfc.tflite 2751472944 83a715086ee914c12771eb986556ea6d8f246794aeb4ebb2949c0456df767a91"
    "decide_L768_f16sfc.tflite 2753075504 a35df5ccaff271bc6b7b100cab04df3e81db6043739fb9d9d27cdafadb7a0163"
    "decide_L1024_f16sfc.tflite 2755202352 170842c13d4684fb8220eb92fd22b829e0a593a96171a457abe0e8930e59eb11"
    "decide_L1536_f16sfc.tflite 2761028912 ceeabf92b4c6bbce2087eb8dfb0aee4c64871b5c2d66eb12285b6bcab52c667a"
    "decide_L2048_f16sfc.tflite 2768952624 0f9b8f0384a6e158d7a24062f25f18fe088e5fb3401cf8de5cad9f90bf44d2ea"
    "decide_L3072_f16sfc.tflite 2791091504 088772270e2fb447e2cc292464b30e32f3983c3b34c70c2570f2b54626b9a419"
    "tower_N400_fp16fc.tflite 658758848 4e66c14e26780c61b7b18f62469cfe41054268912973881e862e8ec2b194f244"
    "tower_N576_fp16fc.tflite 658758848 9742c0c2bd791f501e65eb752f86e18be75b42c7e5f3efd3e0af5f7155f49dcd"
    "tower_N784_fp16fc.tflite 658758848 4d6cdc7130641db90a9036b8ec18187703de2218e631ca95db1a43ce080e6642"
    "tower_N1024_fp16fc.tflite 658758848 1857499d81652664bd26585087f59b8eb09253e78cf1a0e43bcbb48078871424"
    "tower_N1296_fp16fc.tflite 658758848 56a6a8d71ef2c67fc73303a4080af802a675961c05031d19bca4407407223b66"
    "tower_N1600_fp16fc.tflite 658758848 347b77b7a3b034f90b9ac2c8fad2bd632555bfb2e9804367fbbe0aa9f89b3c37"
    "tower_N1936_fp16fc.tflite 658758848 6bf103d1e4e2efeb9c42b9f420f690b30949ab2929790c2647c678df0ec8eaad"
    "tower_N2304_fp16fc.tflite 658758848 e6d27ae15aacef61c638506e6a5cb030acd91d1a23238cd7ac2a4528c771675e"
    "tower_N3136_fp16fc.tflite 658758848 0fe5b485bfb7c6e8b9d42a089ce0f159c096a0559a45c6c7c97ae3241339382c"
)
NAMES=(tokenizer.json tables.safetensors tower_N400_fp16fc.tflite tower_N576_fp16fc.tflite decide_L384_f16sfc.tflite)
# shellcheck disable=SC2206
NAMES+=(${MORE:-})

entry_of() {
    local e
    for e in "${KNOWN[@]}"; do
        if [[ "${e%% *}" == "$1" ]]; then
            printf '%s\n' "$e"
            return 0
        fi
    done
    return 1
}

source_of() {
    if [[ "$1" == tables.safetensors && -f "$SOURCE_DIR/tables/$1" ]]; then
        printf '%s\n' "$SOURCE_DIR/tables/$1"
    else
        printf '%s\n' "$SOURCE_DIR/$1"
    fi
}

# Check every source before any device command to avoid a partial installation.
for name in "${NAMES[@]}"; do
    if ! entry="$(entry_of "$name")"; then
        printf 'Not a file this app reads: %s\n' "$name" >&2
        exit 1
    fi
    read -r _ bytes _ <<< "$entry"
    source="$(source_of "$name")"
    if [[ ! -f "$source" ]]; then
        printf 'Missing source file: %s\n' "$source" >&2
        exit 1
    fi
    actual="$(wc -c < "$source" | tr -d ' ')"
    if [[ "$actual" != "$bytes" ]]; then
        printf '%s is %s bytes, expected %s\n' "$source" "$actual" "$bytes" >&2
        exit 1
    fi
done
if [[ -z "$(adb shell pm list packages "$PACKAGE" < /dev/null | tr -d '\r')" ]]; then
    printf 'Install the app first: ./gradlew :app:installDebug\n' >&2
    exit 1
fi

pending=''
cleanup() {
    if [[ -n "$pending" ]]; then
        adb shell rm -f "$pending" < /dev/null
    fi
}
trap cleanup EXIT

adb shell mkdir -p "$TEMP_DIR" < /dev/null
adb shell run-as "$PACKAGE" mkdir -p files/models < /dev/null
for name in "${NAMES[@]}"; do
    read -r _ bytes sha <<< "$(entry_of "$name")"
    got="$(adb shell run-as "$PACKAGE" stat -c %s "files/models/$name" < /dev/null 2> /dev/null | tr -d '\r' || true)"
    if [[ "$got" == "$bytes" ]]; then
        printf 'Already on the phone: %s\n' "$name"
        continue
    fi
    printf 'Installing %s (%s bytes)\n' "$name" "$bytes"
    pending="$TEMP_DIR/$$-$name"
    adb push "$(source_of "$name")" "$pending" < /dev/null
    adb shell run-as "$PACKAGE" cp "$pending" "files/models/$name" < /dev/null
    adb shell rm "$pending" < /dev/null
    pending=''
    got="$(adb shell run-as "$PACKAGE" sha256sum "files/models/$name" < /dev/null | cut -d' ' -f1 | tr -d '\r')"
    if [[ "$got" != "$sha" ]]; then
        printf 'files/models/%s on the phone has SHA-256 %s, expected %s\n' "$name" "$got" "$sha" >&2
        exit 1
    fi
done
adb shell rmdir "$TEMP_DIR" < /dev/null 2> /dev/null || true
adb shell run-as "$PACKAGE" ls -l files/models < /dev/null
printf 'Files installed. Open OCR Check: the first start checks the files and builds the graphs on the GPU.\n'
