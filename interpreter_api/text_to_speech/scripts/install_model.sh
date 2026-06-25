#!/usr/bin/env bash
#
# Stage the Kokoro-82M LiteRT model into the app's internal files dir.
#
# The model (~338 MB) is intentionally NOT bundled in the APK. It is downloaded
# from the published litert-community repo and pushed onto the connected device,
# then copied into the app's private files dir via run-as.
#
# Prerequisite: install the app first so its data dir exists:
#     cd android && ./gradlew :app:installDebug
#
# Usage:
#     ./scripts/install_model.sh           # download from Hugging Face, then stage
#     ./scripts/install_model.sh <path>    # stage an existing local .tflite
#
set -euo pipefail

PKG="com.google.aiedge.examples.texttospeech"
DEST="kokoro_specout.tflite"   # filename the app loads from filesDir (KokoroTtsHelper.MODEL_FILE)
URL="https://huggingface.co/litert-community/Kokoro-82M/resolve/main/kokoro_82m_fixedlen_fp32.tflite"
SRC="${1:-kokoro_82m_fixedlen_fp32.tflite}"

if [ ! -f "$SRC" ]; then
  echo "Downloading model from ${URL} ..."
  curl -L --fail -o "$SRC" "$URL"
fi

echo "Pushing $(du -h "$SRC" | cut -f1) to /data/local/tmp/${DEST} ..."
adb push "$SRC" "/data/local/tmp/${DEST}"

echo "Copying into ${PKG} files dir ..."
# If the app sandbox cannot read /data/local/tmp on your device, pipe instead:
#   adb shell "cat /data/local/tmp/${DEST}" | adb shell "run-as ${PKG} sh -c 'cat > files/${DEST}'"
adb shell "run-as ${PKG} sh -c 'cp /data/local/tmp/${DEST} files/${DEST} && ls -l files/${DEST}'"

echo "Done. Launch the app and tap Synthesize."
