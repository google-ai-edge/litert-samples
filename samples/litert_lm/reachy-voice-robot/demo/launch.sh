#!/usr/bin/env bash
# One-command launch for the Reachy voice + vision demo (split Mac + Pi mode).
#
# Brings up the whole stack:
#   Pi  : Gemma LLM (litert-lm :9379) -> pipeline server (serve.py :9500)
#         -> GPU object detector (gpu_detect.py :9600)
#   Mac : Reachy MuJoCo sim daemon  +  the demo (voice loop + live web dashboard)
#
# The robot (MuJoCo sim) runs in its OWN terminal — start it FIRST:
#   Terminal 1:  ./demo/robot.sh          # opens the MuJoCo window, keep it open
#   Terminal 2:  ./demo/launch.sh         # everything else + the demo
#
# Point it at your Pi with environment variables (defaults shown):
#   PI=raspberrypi.local     # hostname or IP of the Pi
#   REMOTE_USER=pi           # ssh user on the Pi
#   SSH_PORT=22              # ssh port
#   SSH_KEY=                 # path to an ssh key (empty = your default identity)
#   PI_REPO=/home/pi/rpi     # where this repo is checked out on the Pi
#   V3D_ICD=                 # path to the Mesa v3dv ICD json for the GPU detector
#                            # (leave empty to run the detector without the GPU env)
#
# Examples:
#   PI=raspberrypi.local ./demo/launch.sh
#   ./demo/launch.sh --no-robot          # console robot, no sim window, single terminal
#
# The Pi services are systemd --user units, reused across runs (a second launch
# is fast). Stop them with:
#   ssh $REMOTE_USER@$PI 'systemctl --user stop gemma pipeserve gpud'

set -uo pipefail

PI="${PI:-raspberrypi.local}"
REMOTE_USER="${REMOTE_USER:-pi}"
SSH_PORT="${SSH_PORT:-22}"
PI_REPO="${PI_REPO:-/home/pi/rpi}"
V3D_ICD="${V3D_ICD:-}"
SSH="ssh -p $SSH_PORT ${SSH_KEY:+-i $SSH_KEY} -o ConnectTimeout=15 -o BatchMode=yes"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DASHBOARD="http://127.0.0.1:8080"

NO_ROBOT=""
[ "${1:-}" = "--no-robot" ] && NO_ROBOT="--no-robot"

say()  { printf '\033[36m[launch]\033[0m %s\n' "$1"; }
die()  { printf '\033[31m[launch] %s\033[0m\n' "$1" >&2; exit 1; }

pi() { $SSH "$REMOTE_USER@$PI" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); $1"; }

# Start a Pi systemd --user unit only if it is not already active (idempotent).
#   pi_unit <name> <logfile> "<extra systemd-run props>" "<command>"
pi_unit() {
  local name="$1" log="$2" props="$3" cmd="$4"
  pi "systemctl --user reset-failed $name 2>/dev/null || true
      if systemctl --user is-active --quiet $name; then echo '$name already running';
      else rm -f $log
           systemd-run --user --unit=$name -p StandardOutput=file:$log -p StandardError=file:$log $props $cmd >/dev/null 2>&1
           echo '$name started'
      fi"
}

# ---------------------------------------------------------------- Pi
say "Pi ($PI): Gemma LLM on :9379…"
pi_unit gemma /tmp/gemma.log \
  "--working-directory=$PI_REPO" \
  "$PI_REPO/.venv/bin/litert-lm serve --host 127.0.0.1 --port 9379" \
  || die "cannot reach the Pi over SSH (set PI / REMOTE_USER / SSH_PORT / SSH_KEY)"

say "waiting for the LLM to load (Gemma E2B ~2.5 GB, up to ~a minute on a cold start)…"
pi "until curl -s --max-time 2 http://127.0.0.1:9379/v1/models | grep -q e2b; do sleep 3; done" \
  || die "LLM did not become ready"
say "LLM ready (model e2b)."

say "Pi: pipeline server (serve.py) on :9500…"
pi_unit pipeserve /tmp/pipeserve.log \
  "--working-directory=$PI_REPO" \
  "$PI_REPO/.venv/bin/python -m demo.serve --host 127.0.0.1 --port 9500"

say "Pi: GPU object detector (gpu_detect.py) on :9600…"
GPU_ENV=""
[ -n "$V3D_ICD" ] && GPU_ENV="--setenv=VK_ICD_FILENAMES=$V3D_ICD --setenv=V3D_WEBGPU_OVERRIDE=1"
pi_unit gpud /tmp/gpud.log \
  "--working-directory=$PI_REPO $GPU_ENV" \
  "$PI_REPO/.venv/bin/python -u -m demo.gpu_detect --host 127.0.0.1 --port 9600"

say "waiting for server + detector to listen…"
pi "for p in 9500 9600; do until ss -ltn | grep -q \":\$p\"; do sleep 1; done; done"
say "Pi stack up: :9379 LLM  ·  :9500 server  ·  :9600 detector."

# ---------------------------------------------------------------- SSH tunnel
# The Pi services bind 127.0.0.1 only — they are NEVER exposed on the network.
# The Mac reaches them over an encrypted SSH tunnel: localhost:9500/9600 on the
# Mac are forwarded to the Pi's own loopback. (Binding 0.0.0.0 instead would put
# an unauthenticated inference/detection endpoint on the LAN — don't.)
say "opening SSH tunnel (Mac localhost:9500/9600 -> Pi loopback)…"
$SSH -N -L 9500:127.0.0.1:9500 -L 9600:127.0.0.1:9600 "$REMOTE_USER@$PI" &
TUNNEL_PID=$!
trap 'kill "$TUNNEL_PID" 2>/dev/null' EXIT
for p in 9500 9600; do
  ok=""
  for _ in $(seq 1 20); do
    (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null && { exec 3>&- 3<&-; ok=1; break; }
    sleep 0.5
  done
  [ -n "$ok" ] || die "SSH tunnel port $p never came up — is the Pi reachable and are its services running?"
done
say "tunnel up."

# ---------------------------------------------------------------- Mac: robot check
if [ -z "$NO_ROBOT" ]; then
  if ! pgrep -f "reachy_mini.daemon" >/dev/null 2>&1; then
    die "the robot is not running. Start it FIRST in a separate terminal:
         ./demo/robot.sh
       wait for the MuJoCo window to open, then re-run this command.
       (Or pass --no-robot to run without the robot.)"
  fi
  say "robot daemon detected."
fi

# ---------------------------------------------------------------- Demo
say "live dashboard: $DASHBOARD  (open it in a browser)"
say "starting the demo — just talk to the robot. Ctrl-C stops everything on the Mac."
cd "$REPO" || die "cannot cd to repo root: $REPO"
# --pi 127.0.0.1: reach serve.py/gpu_detect through the local end of the SSH
# tunnel, not the Pi's LAN address.
uv run python -m demo.run_demo --pi 127.0.0.1 --port 9500 --gpu-detect-port 9600 $NO_ROBOT
