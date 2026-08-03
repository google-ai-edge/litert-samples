#!/bin/bash
# Install + CLI-drive the Bonsai app on a connected iPhone (USB, unlocked).
# First run on a device (or if the app shows "Model files missing"): copy the
# model set into the app container first —
#   M=<dir with the model download>
#   xcrun devicectl device copy to --device <UDID> \
#     --source $M/dit_int4b32.tflite --source $M/textenc_int4.tflite \
#     --source $M/vae_dec_fp32.tflite --source $M/pipeline_meta.json \
#     --destination Documents --domain-type appDataContainer \
#     --domain-identifier com.google.ai.edge.Bonsai
# Documents persist across reinstalls of the same bundle id, so this is a
# one-time transfer per device (unless the app is deleted).
# Usage: BONSAI_UDID=<udid> ./device_run.sh ["prompt"] [seed] [steps]
#        (find the UDID with: xcrun devicectl list devices)
set -euo pipefail
cd "$(dirname "$0")"
UDID="${BONSAI_UDID:?set BONSAI_UDID — xcrun devicectl list devices}"
PROMPT="${1:-a red panda drinking tea in a bamboo forest, watercolor}"
SEED="${2:-7}"
STEPS="${3:-4}"

APP="${BONSAI_APP:-build/Build/Products/Release-iphoneos/BonsaiApp.app}"
if [ ! -d "$APP" ]; then
  echo "App not found at $APP — build it first:"
  echo "  xcodebuild -project BonsaiApp.xcodeproj -scheme BonsaiApp \\"
  echo "    -configuration Release -destination generic/platform=iOS \\"
  echo "    -derivedDataPath build build"
  exit 1
fi

xcrun devicectl device install app --device "$UDID" "$APP"
ENV=$(printf '{"BONSAI_AUTORUN":"1","BONSAI_PROMPT":"%s","BONSAI_SEED":"%s","BONSAI_STEPS":"%s"}' \
      "$PROMPT" "$SEED" "$STEPS")
xcrun devicectl device process launch --console --terminate-existing \
    --environment-variables "$ENV" --device "$UDID" com.google.ai.edge.Bonsai \
    || echo "CLI launch failed — the app is installed; launch it from the home screen instead."
