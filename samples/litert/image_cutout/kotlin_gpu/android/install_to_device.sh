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

# Copies the WeVisOne-VL-2B files into the app's private files/models/ and the example photos into
# files/examples/.
#
#   ./install_to_device.sh [dir-with-the-model-files]
#
# The directory holds the files below side by side: the cutout/ folder of
# litert-community/WeVisOne-VL-2B on Hugging Face, which the README's hf download command puts at
# ~/Downloads/WeVisOne-VL-2B-LiteRT/cutout (the default). FORM selects the language model file:
# "int8" (wevisone_decoder_b_int8.tflite, the app's default GPU form, default), "fp16"
# (wevisone_decoder_b_fp16.tflite, the GPU fp16 and CPU forms) or "both". EXAMPLES=0 skips the
# NASA example photos (downloaded from images.nasa.gov with their SHA-256 checked; public domain,
# credit NASA). Every file is checked by size before anything is copied and by SHA-256 on the phone
# after the copy; the app checks the SHA-256 again at its first run.
# Install the debug APK first: run-as needs a debuggable package. ANDROID_SERIAL selects the device.
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [dir-with-the-model-files]\n' "$0" >&2
    exit 2
fi
SOURCE_DIR="${1:-$HOME/Downloads/WeVisOne-VL-2B-LiteRT/cutout}"
PACKAGE=com.google.ai.edge.examples.image_cutout
TEMP_DIR=/data/local/tmp/wevisone_install

# name bytes sha256 (the values the app checks: ModelFiles.kt)
COMMON=(
    "tokenizer.json 11607393 3bb172429b952469b51c64783f52fa01c26d1b30f1f31a4693e385a43f9cfc5e"
    "wevisone_embed_tokens_fp16.bin 625377280 5ff0fcb29afce21c5efc2c717fe723e4329443373513ba53b7c33d18508050fe"
    "wevisone_vision_encoder_512_fp16.tflite 611814624 47d4089213cfdb6da006263ac92cfee92e8e2ff36c4e59027e4fe16dce2e837a"
    "wevisone_vision_adapter_512_fp16.tflite 201460224 d0c8dc73b2a5a358a3bb06438d643cadbc454245f011d89853d184d9d646c530"
    "wevisone_mask_decoder_gs16.tflite 21878000 ac0765dc75ce81708564421bfe930a7714b589bae43edac2ddc810c57fa5f7e1"
)
FP16="wevisone_decoder_b_fp16.tflite 3458517312 56dc97e27115ab8be102c8e8eb543895735a0eb2d15367fd7a44a736b3ee9864"
INT8="wevisone_decoder_b_int8.tflite 1785462176 f93bf881dccd203a1e55d8fa43cf51798739a5aa5fe441ad7cb3b53808579a6f"

FILES=("${COMMON[@]}")
case "${FORM:-int8}" in
    fp16) FILES+=("$FP16") ;;
    int8) FILES+=("$INT8") ;;
    both) FILES+=("$FP16" "$INT8") ;;
    *)
        printf 'FORM is %s; use fp16, int8 or both\n' "$FORM" >&2
        exit 2
        ;;
esac

# Check every source before any device command to avoid a partial installation.
for entry in "${FILES[@]}"; do
    read -r name bytes sha <<< "$entry"
    source="$SOURCE_DIR/$name"
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

pending=''
cleanup() {
    if [[ -n "$pending" ]]; then
        adb shell rm -f "$pending" < /dev/null
    fi
}
trap cleanup EXIT

# Copies $1 (a Mac file) to files/$2 of the app and checks the SHA-256 $3 on the phone.
install_file() {
    local src="$1" dst="$2" sha="$3"
    pending="$TEMP_DIR/$$-$(basename "$dst")"
    adb push "$src" "$pending" < /dev/null
    adb shell run-as "$PACKAGE" cp "$pending" "files/$dst" < /dev/null
    adb shell rm "$pending" < /dev/null
    pending=''
    local got
    got="$(adb shell run-as "$PACKAGE" sha256sum "files/$dst" < /dev/null | cut -d' ' -f1 | tr -d '\r')"
    if [[ "$got" != "$sha" ]]; then
        printf 'files/%s on the device has SHA-256 %s, expected %s\n' "$dst" "$got" "$sha" >&2
        exit 1
    fi
}

adb shell mkdir -p "$TEMP_DIR" < /dev/null
adb shell run-as "$PACKAGE" mkdir -p files/models files/examples < /dev/null
for entry in "${FILES[@]}"; do
    read -r name bytes sha <<< "$entry"
    printf 'Installing %s (%s bytes)\n' "$name" "$bytes"
    install_file "$SOURCE_DIR/$name" "models/$name" "$sha"
done

if [[ "${EXAMPLES:-1}" != 0 ]]; then
    examples="$(mktemp -d)"
    while read -r name sha url; do
        curl -sS --fail --max-time 600 -o "$examples/$name" "$url"
        got="$(shasum -a 256 "$examples/$name" | cut -d' ' -f1)"
        if [[ "$got" != "$sha" ]]; then
            printf '%s downloaded with SHA-256 %s, expected %s\n' "$name" "$got" "$sha" >&2
            exit 1
        fi
        printf 'Installing example %s\n' "$name"
        install_file "$examples/$name" "examples/$name" "$sha"
    done <<'LIST'
nasa_S67-34267.jpg 7e748be3b43b91a25652329f4026b1c19c96ae60decc2ea99cd71a538754db21 https://images-assets.nasa.gov/image/S67-34267/S67-34267~orig.jpg
nasa_KSC-20170424-PH_SWW01_0007.jpg 497419455401d41fd0042b2af1d07944d4013a8362d549f6787f4cc6e66110a2 https://images-assets.nasa.gov/image/KSC-20170424-PH_SWW01_0007/KSC-20170424-PH_SWW01_0007~orig.JPG
nasa_KSC-08pd3809.jpg 4b187db43aaea1720286e4fc35989e020a170510d3d11fa3d21dfefb07fabfd4 https://images-assets.nasa.gov/image/KSC-08pd3809/KSC-08pd3809~orig.jpg
LIST
    rm -rf "$examples"
fi
adb shell rmdir "$TEMP_DIR" < /dev/null 2> /dev/null || true
adb shell run-as "$PACKAGE" ls -la files/models files/examples < /dev/null
printf 'Files installed. Open WeVisOne Cut-out.\n'
