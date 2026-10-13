#!/bin/sh
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

# Packages a Linux release bundle:
#   litert_edge_demos-v<version>-linux-<arch>/
#     litert_edge_demos, lib/, data/   the Flutter bundle
#     run.sh                          pre-flight launcher (tool/linux/run.sh)
#     README.md                       how to start it
#     licenses/                       app licence (Apache-2.0), model notices, AGPL-3.0 text + the YOLO26n
#                                     derivation script (the "source" of the bundled AGPL model), and
#                                     libjpeg-turbo's licences when lib/libturbojpeg.so.0 is shipped
#   + a .tar.gz of that folder and its SHA-256 line.
#
# Usage (on the Linux machine that built it): tool/linux/package.sh [out_dir]
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$ROOT/build/dist}
case $(uname -m) in x86_64) ARCH=x64 ;; aarch64) ARCH=arm64 ;; *) echo "unsupported arch" >&2; exit 1 ;; esac
BUNDLE="$ROOT/build/linux/$ARCH/release/bundle"
if command -v fvm >/dev/null 2>&1; then FLUTTER='fvm flutter'; else FLUTTER=flutter; fi
[ -x "$BUNDLE/litert_edge_demos" ] || { echo "error: build first: $FLUTTER build linux --release" >&2; exit 1; }
# The built-in models are not in git (tool/fetch_models.sh): package only a bundle built with the
# verified ones.
"$ROOT/tool/fetch_models.sh" --check --dir "$BUNDLE/data/flutter_assets/assets/models" || {
  echo "error: $BUNDLE was built without the verified models: run tool/fetch_models.sh, then rebuild" >&2
  exit 1
}
VERSION=$(sed -n 's/^version: *\([0-9.]*\).*/\1/p' "$ROOT/pubspec.yaml")
NAME="litert_edge_demos-v$VERSION-linux-$ARCH"
DEST="$OUT/$NAME"
[ -e "$DEST" ] && { echo "error: $DEST exists; move it away first" >&2; exit 1; }

mkdir -p "$DEST/licenses"
cp -R "$BUNDLE/." "$DEST/"
cp "$ROOT/tool/linux/run.sh" "$DEST/run.sh" && chmod +x "$DEST/run.sh"
cat > "$DEST/README.md" <<'EOF'
# LiteRT Demos for Linux

    ./run.sh               # checks sound, GPU (Vulkan) and camera, then starts the app
    ./run.sh --selftest    # headless self-test (hardware, detector, chat model, audio), then a report

`run.sh` only warns: the app reports what is missing itself. Everything also goes to `run.log` next
to `run.sh`. The chat model is a `.litertlm` file you choose in the app (for example
`litert-community/gemma-4-E2B-it-litert-lm` on Hugging Face); the speech, embedding and detector
models are built in. Licences: `licenses/README.md`.
EOF
cp "$ROOT/LICENSE" "$DEST/licenses/LICENSE-app-Apache-2.0.txt"
cp "$ROOT/assets/models/NOTICE.md" "$DEST/licenses/MODELS-NOTICE.md"
cp "$ROOT/licenses/AGPL-3.0.txt" "$DEST/licenses/AGPL-3.0.txt"
cp "$ROOT/tool/prune_yolo26n_head.py" "$DEST/licenses/prune_yolo26n_head.py"
turbojpeg_note=''
if [ -e "$DEST/lib/libturbojpeg.so.0" ]; then
  # The Debian copyright file of the copy that was shipped (full licence texts).
  # The repository's notice (IJG statement + BSD-3-Clause text) always; the
  # build machine's Debian copyright file too when the system has it.
  cp "$ROOT/licenses/libjpeg-turbo.txt" "$DEST/licenses/libjpeg-turbo-copyright.txt"
  copyright=$(ls /usr/share/doc/libturbojpeg*/copyright 2>/dev/null | head -n 1 || true)
  if [ -n "$copyright" ]; then
    cp "$copyright" "$DEST/licenses/libjpeg-turbo-debian-copyright.txt"
  fi
  turbojpeg_note='- `lib/libturbojpeg.so.0` (the network camera'"'"'s JPEG decoder) is libjpeg-turbo, under the IJG License and
  the Modified (3-clause) BSD License (`libjpeg-turbo-copyright.txt`). This software is based in part on the work of
  the Independent JPEG Group.'
fi
cat > "$DEST/licenses/README.md" <<'EOF'
# Licences

- The app's own code: Apache-2.0 (`LICENSE-app-Apache-2.0.txt`).
- Bundled models and their terms: `MODELS-NOTICE.md`.
- YOLO26n (object detector) is AGPL-3.0 (`AGPL-3.0.txt`). The bundled file `yolo26n_fp16_rawhead.tflite` is derived
  from Arm's `yolo26n_conv2d_f16_weights.tflite` (https://huggingface.co/Arm/yolo26n-fp16-litert, itself derived from
  Ultralytics YOLO26) by `prune_yolo26n_head.py` in this folder: it cuts the in-graph selection head and keeps the
  raw [1, 8400, 84] output. Weights are unchanged.
- The app's full list of package licences is in the app: home screen ⋮ (More) → Licences.
EOF
if [ -n "$turbojpeg_note" ]; then
  printf '%s\n' "$turbojpeg_note" >>"$DEST/licenses/README.md"
fi

tar -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
(cd "$OUT" && sha256sum "$NAME.tar.gz" | tee "$NAME.tar.gz.sha256")
du -sh "$OUT/$NAME.tar.gz"
