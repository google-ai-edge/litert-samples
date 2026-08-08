#!/usr/bin/env bash
# Reachy Mini robot — MuJoCo simulation daemon.
#
# Run this in its OWN terminal: it opens a GUI window and must stay in the
# foreground (macOS only opens the MuJoCo window under mjpython on the main
# thread). Wait for the window, then start the rest in another terminal:
#
#   Terminal 1:  ./demo/robot.sh
#   Terminal 2:  ./demo/launch.sh
#
# The demo's robot client connects to this already-running daemon.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

[ -x ./.venv/bin/mjpython ] || {
  echo "mjpython not found at .venv/bin/mjpython — is reachy_mini installed?" >&2
  exit 1
}

echo "[robot] starting Reachy Mini (MuJoCo sim) — a window will open; keep this terminal open."
exec ./.venv/bin/mjpython -m reachy_mini.daemon.app.main --sim --no-media
