#!/usr/bin/env bash
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

# Launcher shipped next to the Linux bundle (tool/linux/package.sh copies it).
# Checks what the app needs from the system, prints what to install or start,
# then replaces itself with the app, passing every argument through:
#
#   ./run.sh                      # the app
#   ./run.sh --selftest           # the headless self-test, then a report
#
# The checks only warn: the app itself fails with a clear message when
# something it needs is missing.
#   audio   a running sound server (pactl info), parecord, a microphone and an
#           output (pactl list short sources|sinks)
#   gpu     vulkaninfo --summary (a software device such as llvmpipe is the CPU)
#   camera  GStreamer's v4l2src plugin and /dev/video*
#   jpeg    libturbojpeg.so.0 (the network camera's fast JPEG decoder): the
#           bundle's lib/ copy, else the system's (/usr/lib/<arch>, …)
#
# Environment:
#   RUN_LOG=PATH      also write everything to PATH (default: run.log next to
#                     this script, or ~/.local/state/litert_edge_demos/run.log
#                     when the bundle is read-only); RUN_LOG= turns it off
#   RUN_SH_DRY_RUN=1  run the checks, print the app's command line instead of
#                     starting it
#
# Bash 3.2 compatible (the dry run is tested on macOS too).
set -u

here=$(cd "$(dirname "$0")" && pwd)
app="$here/litert_edge_demos"

if [ "${RUN_LOG+set}" = set ]; then
  log=$RUN_LOG
elif [ -w "$here" ]; then
  log="$here/run.log"
else
  log="${XDG_STATE_HOME:-$HOME/.local/state}/litert_edge_demos/run.log"
fi
if [ -n "$log" ] && mkdir -p "$(dirname "$log")" 2>/dev/null &&
  : >>"$log" 2>/dev/null; then
  exec > >(tee -a "$log") 2>&1
else
  log=""
fi

warnings=0
ok() { printf '  ok    %s\n' "$*"; }
warn() {
  printf '  WARN  %s\n' "$*"
  warnings=$((warnings + 1))
}
hint() { printf '        -> %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
# A hung sound server blocks pactl (and a broken driver vulkaninfo): give
# each probe 5 s when coreutils' timeout is there.
t5() {
  if have timeout; then
    timeout 5 "$@"
  else
    "$@"
  fi
}

printf '=== litert_edge_demos pre-flight (%s) ===\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

echo 'audio'
if ! have pactl; then
  warn 'pactl not found: the sound server cannot be checked'
  hint 'sudo apt install pulseaudio-utils'
elif info=$(LC_ALL=C t5 pactl info 2>&1); then
  server=$(printf '%s\n' "$info" | sed -n 's/^Server Name: //p')
  ok "sound server: ${server:-unnamed}"
  sources=$(LC_ALL=C t5 pactl list short sources 2>/dev/null |
    awk '$2 !~ /\.monitor$/' | wc -l | tr -d ' ')
  sinks=$(LC_ALL=C t5 pactl list short sinks 2>/dev/null |
    awk '$2 != "auto_null"' | wc -l | tr -d ' ')
  if [ "$sources" -gt 0 ]; then
    ok "microphones: $sources"
  else
    warn 'no recording device (only monitors of outputs)'
    hint 'plug in a microphone or enable it in the sound settings (pavucontrol)'
  fi
  if [ "$sinks" -gt 0 ]; then
    ok "outputs: $sinks"
  else
    warn 'no audio output (no sink, or only the dummy auto_null)'
    hint 'connect speakers or headphones; in a VM add a sound device or a null sink'
  fi
else
  warn "no sound server: $(printf '%s\n' "${info:-pactl info did not answer within 5 s}" | head -n 1)"
  hint 'systemctl --user start pipewire pipewire-pulse   (or: pulseaudio --start)'
fi
if have parecord; then
  ok 'parecord'
else
  warn 'parecord not found: the microphone cannot record'
  hint 'sudo apt install pulseaudio-utils'
fi

echo 'gpu'
shopt -s nocasematch
if ! have vulkaninfo; then
  warn 'vulkaninfo not found: the GPU cannot be checked'
  hint 'sudo apt install vulkan-tools libvulkan1 mesa-vulkan-drivers'
else
  devices=$(t5 vulkaninfo --summary 2>/dev/null |
    sed -n 's/^[[:space:]]*deviceName[[:space:]]*=[[:space:]]*//p')
  if [ -z "$devices" ]; then
    warn 'vulkaninfo lists no Vulkan device: the GPU mode will fail'
    hint 'sudo apt install libvulkan1 mesa-vulkan-drivers (or the NVIDIA driver)'
  else
    hardware=0
    while IFS= read -r device; do
      case "$device" in
      *llvmpipe* | *lavapipe* | *swiftshader* | *softpipe*)
        warn "software Vulkan device: $device (the CPU, not a GPU)"
        ;;
      *)
        ok "Vulkan device: $device"
        hardware=$((hardware + 1))
        ;;
      esac
    done <<EOF
$devices
EOF
    if [ "$hardware" -eq 0 ]; then
      hint 'no hardware GPU: install its Vulkan driver (Mesa: mesa-vulkan-drivers; NVIDIA: the proprietary driver), or use the CPU'
    fi
  fi
fi
shopt -u nocasematch

echo 'camera'
if have gst-inspect-1.0; then
  if t5 gst-inspect-1.0 v4l2src >/dev/null 2>&1; then
    ok 'GStreamer v4l2src'
  else
    warn 'GStreamer v4l2src is missing: no camera'
    hint 'sudo apt install gstreamer1.0-plugins-good'
  fi
else
  plugin=''
  for f in /usr/lib/*/gstreamer-1.0/libgstvideo4linux2.so \
    /usr/lib/gstreamer-1.0/libgstvideo4linux2.so \
    /usr/lib64/gstreamer-1.0/libgstvideo4linux2.so; do
    if [ -e "$f" ]; then
      plugin=$f
      break
    fi
  done
  if [ -n "$plugin" ]; then
    ok "GStreamer v4l2src ($plugin)"
  else
    warn 'GStreamer v4l2src not found (and no gst-inspect-1.0 to ask)'
    hint 'sudo apt install gstreamer1.0-plugins-good gstreamer1.0-tools'
  fi
fi
cameras=0
for video in /dev/video*; do
  [ -e "$video" ] || continue
  cameras=$((cameras + 1))
  if [ -r "$video" ] && [ -w "$video" ]; then
    ok "$video"
  else
    warn "$video is not accessible"
    hint "sudo usermod -aG video $USER   (then log out and in)"
  fi
done
if [ "$cameras" -eq 0 ]; then
  warn 'no camera (/dev/video*)'
  hint 'plug in a webcam (in a VM, pass it through)'
fi

echo 'network camera'
system_turbojpeg=''
for f in /usr/lib/*/libturbojpeg.so.0 /usr/lib/libturbojpeg.so.0 \
  /usr/lib64/libturbojpeg.so.0 /usr/local/lib/libturbojpeg.so.0; do
  if [ -e "$f" ]; then
    system_turbojpeg=$f
    break
  fi
done
if [ -e "$here/lib/libturbojpeg.so.0" ]; then
  ok 'libturbojpeg.so.0 (shipped in lib/)'
elif [ -n "$system_turbojpeg" ]; then
  ok "libturbojpeg.so.0 ($system_turbojpeg)"
else
  warn 'libturbojpeg.so.0 not found: the network camera falls back to a slow JPEG decoder (a few fps)'
  hint 'sudo apt install libturbojpeg   (Raspberry Pi OS / Debian: libturbojpeg0)'
fi

printf '=== %s warning(s)%s ===\n' "$warnings" "${log:+; log: $log}"

if [ ! -x "$app" ]; then
  echo "ERROR: $app is missing or not executable (run.sh belongs next to the bundle's binary)"
  exit 127
fi
if [ "${RUN_SH_DRY_RUN:-0}" = 1 ]; then
  printf 'would run:'
  printf ' %q' "$app" "$@"
  printf '\n'
  exit 0
fi
exec "$app" "$@"
