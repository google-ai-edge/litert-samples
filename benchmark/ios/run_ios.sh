#!/bin/bash
# Runs one benchmark on a connected iPhone: pushes the model, launches the app with benchmark_model flags,
# waits for it to finish, and pulls benchmark.log and results.pb into OUT.
#
# Usage: BUNDLE_ID=<app bundle id> ./run_ios.sh <device-name-or-udid> <model.tflite> [benchmark_model flags...]
# Example: BUNDLE_ID=com.example.LiteRTBenchmark ./run_ios.sh "My iPhone" mobilenet_v2.tflite --num_runs=50
# The app must already be installed (open LiteRTBenchmark.xcodeproj once, set your team and bundle id,
# run on the phone). The script sets --graph and --result_file_path; pass the other flags after the model.
# Environment: BUNDLE_ID (required), OUT (output directory), WAIT_SECS (how long to wait for the run, default 600).
set -euo pipefail
DEVICE="$1"; MODEL="$2"; shift 2
: "${BUNDLE_ID:?set BUNDLE_ID to the bundle identifier of the app}"
OUT="${OUT:-out/$(date +%Y%m%d-%H%M%S)}"
WAIT_SECS="${WAIT_SECS:-600}"
mkdir -p "$OUT"
xcrun devicectl device copy to --device "$DEVICE" --source "$MODEL" --destination "Documents/$(basename "$MODEL")" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console "$BUNDLE_ID" \
  "--graph=$(basename "$MODEL")" --result_file_path=results.pb "$@" | tee "$OUT/console.log" || true
# The app writes Documents/benchmark.done (exit status) when the run ends; poll for it, then pull the files.
done=0
for _ in $(seq 1 $((WAIT_SECS / 5))); do
  if xcrun devicectl device copy from --device "$DEVICE" --source Documents/benchmark.done --destination "$OUT/benchmark.done" \
       --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" >/dev/null 2>&1; then done=1; break; fi
  sleep 5
done
if [ "$done" -ne 1 ]; then echo "no benchmark.done after ${WAIT_SECS}s; the run may still be going" >&2; exit 2; fi
for f in benchmark.log results.pb; do
  xcrun devicectl device copy from --device "$DEVICE" --source "Documents/$f" --destination "$OUT/$f" \
    --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" || true
done
echo "status: $(cat "$OUT/benchmark.done" 2>/dev/null || echo unknown)"; grep -E 'Inference timings|Memory footprint delta|count=' "$OUT/benchmark.log" || true
echo "files in $OUT"
