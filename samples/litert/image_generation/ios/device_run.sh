#!/bin/bash
# Install + CLI-drive the Bonsai app on the connected iPhone (USB, unlocked).
# First run on a device (or if the app shows "Model files missing"): copy the
# model set into the app container first —
#   M=~/models/bonsai-image-4b-tflite/hf_upload
#   xcrun devicectl device copy to --device <UDID> \
#     --source $M/dit_int4b32.tflite --source $M/textenc_int4.tflite \
#     --source $M/vae_dec_fp32.tflite --source $M/pipeline_meta.json \
#     --destination Documents --domain-type appDataContainer \
#     --domain-identifier com.google.ai.edge.ImageGeneration
# Documents persist across reinstalls of the same bundle id, so this is a
# one-time transfer per device (unless the app is deleted).
# Usage: ./device_run.sh ["prompt"] [seed] [steps]
set -euo pipefail
UDID="${BONSAI_UDID:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}"   # DaisukeのiPhone 17 Pro
APP="$HOME/Library/Developer/Xcode/DerivedData/ImageGeneration-*/Build/Products/Release-iphoneos/ImageGeneration.app"
PROMPT="${1:-a red panda drinking tea in a bamboo forest, watercolor}"
SEED="${2:-7}"
STEPS="${3:-4}"

xcrun devicectl device install app --device "$UDID" "$APP"
ENV=$(printf '{"BONSAI_AUTORUN":"1","BONSAI_PROMPT":"%s","BONSAI_SEED":"%s","BONSAI_STEPS":"%s"}' \
      "$PROMPT" "$SEED" "$STEPS")
xcrun devicectl device process launch --console --terminate-existing \
    --environment-variables "$ENV" --device "$UDID" com.google.ai.edge.ImageGeneration \
    || echo "CLI launch failed — the app is installed; launch it from the home screen instead."
