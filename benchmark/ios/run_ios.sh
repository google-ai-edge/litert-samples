#!/bin/bash
# Runs one benchmark on a connected iPhone and writes the outputs into a session directory in the layout the
# leaderboard driver reads: SESSION_DIR/<accelerator>-<device id>/{results.pb, stdout.txt, console.log} next to
# SESSION_DIR/session.json. Run it once per accelerator with the same SESSION_DIR to get one session.
#
# Usage: BUNDLE_ID=<app bundle id> ./run_ios.sh <device name or UDID> <model.tflite> [benchmark_model flags...]
# The app must already be installed (open LiteRTBenchmark.xcodeproj once, set your team and bundle id, run on
# the phone). The script sets --graph and --result_file_path; --use_gpu=true makes the job a gpu job, else cpu.
# Environment: BUNDLE_ID (required), SESSION_DIR (default ~/.cache/litert-samples-benchmark/app/<UTC time>),
#   DEVICE_ID (default: the phone's marketing name lowercased without spaces, e.g. iphone17pro),
#   WAIT_SECS (how long to wait for the run, default 600).
set -euo pipefail
DEVICE="$1"; MODEL="$2"; shift 2
: "${BUNDLE_ID:?set BUNDLE_ID to the bundle identifier of the app}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WAIT_SECS="${WAIT_SECS:-600}"
case " $* " in *" --use_gpu=true "*) ACCEL=gpu;; *) ACCEL=cpu;; esac

# Device facts for session.json come from devicectl.
DETAILS="$(mktemp)"
xcrun devicectl device info details --device "$DEVICE" --json-output "$DETAILS" >/dev/null
NAME=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["hardwareProperties"]["marketingName"])' "$DETAILS")
OS=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["result"]["deviceProperties"]; print("iOS " + d["osVersionNumber"] + " (" + d["osBuildUpdate"] + ")")' "$DETAILS")
rm -f "$DETAILS"
DEVICE_ID="${DEVICE_ID:-$(printf '%s' "$NAME" | tr '[:upper:]' '[:lower:]' | tr -d ' ')}"
SESSION_DIR="${SESSION_DIR:-$HOME/.cache/litert-samples-benchmark/app/$(date -u +%Y%m%dT%H%M%SZ)}"
JOB="$SESSION_DIR/$ACCEL-$DEVICE_ID"
mkdir -p "$JOB"
BUILD_INFO="$(cat "$HERE/Frameworks/BUILD_INFO" 2>/dev/null || echo "source (see build_framework.sh)")"
if [ ! -f "$SESSION_DIR/session.json" ]; then
  python3 - "$SESSION_DIR/session.json" "$DEVICE_ID" "$NAME" "$OS" "$BUILD_INFO" <<'PY'
import datetime, json, re, sys
path, device_id, name, os_, build = sys.argv[1:6]
m = re.search(r"v(\d+(?:\.\d+)+)", build)
json.dump({
    "runner": "app", "platform": "ios", "device_id": device_id, "device": name, "os": os_,
    "runtime_version": m.group(1) if m else "unknown", "binary": build + ", app in benchmark/ios",
    "created": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
}, open(path, "w"), indent=1)
PY
fi

xcrun devicectl device copy to --device "$DEVICE" --source "$MODEL" --destination "Documents/$(basename "$MODEL")" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console "$BUNDLE_ID" \
  "--graph=$(basename "$MODEL")" --result_file_path=results.pb "$@" | tee "$JOB/console.log" || true
# The app writes Documents/benchmark.done (exit status) when the run ends; poll for it, then pull the files.
done=0
for _ in $(seq 1 $((WAIT_SECS / 5))); do
  if xcrun devicectl device copy from --device "$DEVICE" --source Documents/benchmark.done --destination "$JOB/benchmark.done" \
       --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" >/dev/null 2>&1; then done=1; break; fi
  sleep 5
done
if [ "$done" -ne 1 ]; then echo "no benchmark.done after ${WAIT_SECS}s; the run may still be going" >&2; exit 2; fi
xcrun devicectl device copy from --device "$DEVICE" --source Documents/benchmark.log --destination "$JOB/stdout.txt" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" >/dev/null || true
xcrun devicectl device copy from --device "$DEVICE" --source Documents/results.pb --destination "$JOB/results.pb" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" >/dev/null || true
echo "status: $(cat "$JOB/benchmark.done" 2>/dev/null || echo unknown)"
grep -E 'Inference \(avg\)|Model initialization|Overall footprint' "$JOB/stdout.txt" || true
echo "session: $SESSION_DIR"
echo "job: $JOB"
