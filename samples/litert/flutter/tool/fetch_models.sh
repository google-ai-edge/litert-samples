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

# Fetches the models built into the app into assets/models/ and checks each file's size and SHA-256
# against tool/models.lock. The files are not in git: run this once per checkout, before
# `flutter build`, `run` or `test`. Files that are already there and verified are skipped, so a
# second run downloads nothing.
#
#   tool/fetch_models.sh               fetch what is missing or different, then verify every file
#   tool/fetch_models.sh --check       verify only, no network; exit 1 if a file is missing or different
#   tool/fetch_models.sh --dir <dir>   use <dir> instead of assets/models (with --check: e.g. a built
#                                      bundle's data/flutter_assets/assets/models)
#
# HF_TOKEN: litert-community/embeddinggemma-300m is gated. Accept the Gemma terms on
# https://huggingface.co/litert-community/embeddinggemma-300m, create a read token, and set HF_TOKEN in
# the environment or as a line `HF_TOKEN=hf_…` in .env (gitignored). The token is sent only with the
# gated files, and never printed.
#
# YOLO26n: Arm's original is downloaded, and tool/prune_yolo26n_head.py derives the raw-head file from
# it in a Python venv with the `pip` line of the lock. Needs python3 3.10-3.14 with venv (Debian/Ubuntu:
# python3-venv). ai-edge-litert has wheels for macOS arm64 and Linux x86_64/aarch64. The output must
# match the lock byte for byte.
#
# Work files (partial downloads, Arm's original, the venv) go to build/fetch_models/; set
# FETCH_MODELS_CACHE to move them. Exit status: 0 every file verified, 1 a file is missing or failed,
# 2 usage.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOCK=$ROOT/tool/models.lock
DIR=$ROOT/assets/models
CACHE=${FETCH_MODELS_CACHE:-$ROOT/build/fetch_models}
MODE=fetch

while [ $# -gt 0 ]; do
  case $1 in
    --check) MODE=check ;;
    --dir)
      [ $# -ge 2 ] || { echo "error: --dir needs a directory" >&2; exit 2; }
      DIR=$2
      shift
      ;;
    --dir=*) DIR=${1#--dir=} ;;
    -h | --help)
      sed -n '2,/^set -eu$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "error: unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[ -f "$LOCK" ] || { echo "error: $LOCK not found" >&2; exit 1; }

if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  echo "error: neither sha256sum nor shasum found" >&2
  exit 1
fi

size_of() { wc -c <"$1" | tr -d ' '; }

# check_file <file> <size> <sha256>: 0 when the file has that size and SHA-256; else sets REASON.
check_file() {
  if [ ! -f "$1" ]; then
    REASON=missing
    return 1
  fi
  got_size=$(size_of "$1")
  if [ "$got_size" != "$2" ]; then
    REASON="$got_size bytes, want $2"
    return 1
  fi
  if [ "$(sha256_of "$1")" != "$3" ]; then
    REASON="SHA-256 differs from the lock"
    return 1
  fi
}

# HF_TOKEN from the environment, else the last `HF_TOKEN=` line of .env (never `source`d): a
# ` #` comment and trailing blanks are cut, then the quotes around the value.
load_token() {
  [ -n "${HF_TOKEN:-}" ] && return 0
  HF_TOKEN=
  [ -f "$ROOT/.env" ] || return 0
  HF_TOKEN=$(sed -n 's/^[[:space:]]*\(export[[:space:]][[:space:]]*\)\{0,1\}HF_TOKEN[[:space:]]*=[[:space:]]*//p' \
    "$ROOT/.env" | tail -n 1 | tr -d '\r' | sed 's/[[:space:]][[:space:]]*#.*$//; s/[[:space:]]*$//')
  case $HF_TOKEN in
    \"*\") HF_TOKEN=${HF_TOKEN#\"} && HF_TOKEN=${HF_TOKEN%\"} ;;
    \'*\') HF_TOKEN=${HF_TOKEN#\'} && HF_TOKEN=${HF_TOKEN%\'} ;;
  esac
}

gated_hint() {
  echo "  $1 is gated: accept the Gemma terms on https://huggingface.co/$1 with your Hugging Face"
  echo "  account, create a read token (https://huggingface.co/settings/tokens) and set HF_TOKEN in the"
  echo "  environment or as a line HF_TOKEN=hf_... in $ROOT/.env (gitignored)."
}

# download <repo> <revision> <path> <access> <size> <sha256> <out>: fetches into the cache with
# resume, verifies, then moves the file to <out>. On failure sets REASON and returns 1.
download() {
  url="https://huggingface.co/$1/resolve/$2/$3"
  part="$CACHE/downloads/$6.part"
  mkdir -p "$CACHE/downloads" "$(dirname "$7")"
  have=0
  [ -f "$part" ] && have=$(size_of "$part")
  if [ "$have" -gt "$5" ]; then
    rm -f "$part"
    have=0
  fi
  if [ "$have" -lt "$5" ]; then
    progress=-sS
    [ -t 2 ] && progress=--progress-bar
    rc=0
    # -q first: ~/.curlrc is not read, so a `verbose` there cannot print the Authorization header.
    if [ "$4" = gated ]; then
      # The header goes in through stdin so the token is not on the command line. curl does not
      # resend it to the CDN host the redirect points at.
      code=$(printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" |
        curl -q -K - -fL "$progress" --retry 3 -C - -o "$part" -w '%{http_code}' "$url") || rc=$?
    else
      code=$(curl -q -fL "$progress" --retry 3 -C - -o "$part" -w '%{http_code}' "$url" </dev/null) ||
        rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
      case $code in
        401 | 403) REASON="HTTP $code from $1 (gated: accept its terms with the account of HF_TOKEN)" ;;
        404) REASON="HTTP 404: $3 is not at revision $2 of $1" ;;
        *) REASON="curl exit $rc (HTTP $code) for $url; rerun to resume" ;;
      esac
      return 1
    fi
  fi
  if ! check_file "$part" "$5" "$6"; then
    rm -f "$part"
    REASON="downloaded $1/$3 at $2: $REASON"
    return 1
  fi
  mv -f "$part" "$7"
}

# derive_yolo <dest> <size> <sha256> <repo> <revision> <path> <source size> <source sha256>
derive_yolo() {
  src="$CACHE/yolo26n/$6"
  if ! check_file "$src" "$7" "$8"; then
    download "$4" "$5" "$6" public "$7" "$8" "$src" || return 1
  fi
  venv="$CACHE/venv"
  # A venv whose setup failed at ensurepip (Debian/Ubuntu without python3-venv) still has
  # bin/python, but no pip: it is made again (--clear empties that directory only).
  clear=
  if [ -x "$venv/bin/python" ] &&
    ! "$venv/bin/python" -m pip --version </dev/null >/dev/null 2>&1; then
    echo "  $venv has no pip (an earlier setup failed): making it again"
    clear=--clear
  fi
  if [ ! -x "$venv/bin/python" ] || [ -n "$clear" ]; then
    python=${PYTHON:-python3}
    if ! command -v "$python" >/dev/null 2>&1; then
      REASON="python3 not found (needed to derive YOLO26n with tool/prune_yolo26n_head.py)"
      return 1
    fi
    if ! "$python" -m venv ${clear:+"$clear"} "$venv" </dev/null; then
      REASON="$python -m venv failed (Debian/Ubuntu: apt install python3-venv)"
      return 1
    fi
  fi
  if [ "$(cat "$venv/.requirements" 2>/dev/null || true)" != "$PIP" ]; then
    echo "  installing $PIP into $venv"
    # shellcheck disable=SC2086 # PIP is a list of requirements
    if ! "$venv/bin/python" -m pip install --disable-pip-version-check --quiet --only-binary=:all: \
      $PIP </dev/null; then
      REASON="pip install $PIP failed (wheels: macOS arm64, Linux x86_64/aarch64; Python 3.10-3.14)"
      return 1
    fi
    printf '%s\n' "$PIP" >"$venv/.requirements"
  fi
  out="$CACHE/yolo26n/$(basename "$1").part"
  rm -f "$out"
  if ! "$venv/bin/python" -I "$ROOT/tool/prune_yolo26n_head.py" "$src" "$out" \
    </dev/null >"$CACHE/yolo26n/prune.log" 2>&1; then
    REASON="tool/prune_yolo26n_head.py failed: see $CACHE/yolo26n/prune.log"
    return 1
  fi
  if ! check_file "$out" "$2" "$3"; then
    REASON="the derived file does not reproduce the lock ($REASON) with $PIP"
    return 1
  fi
  mkdir -p "$(dirname "$DIR/$1")"
  mv -f "$out" "$DIR/$1"
}

# Pass 1: what is already there and verified.
PIP=
TOTAL=0
PRESENT=0
PENDING=
FAILED=0
while read -r kind dest size sha rest <&3; do
  case $kind in
    '' | \#*) continue ;;
    pip) PIP="$dest${size:+ $size}${sha:+ $sha}${rest:+ $rest}" && continue ;;
    hf | yolo26n-rawhead) ;;
    *) echo "error: $LOCK: unknown kind '$kind'" >&2 && exit 1 ;;
  esac
  TOTAL=$((TOTAL + 1))
  if check_file "$DIR/$dest" "$size" "$sha"; then
    PRESENT=$((PRESENT + 1))
    [ "$MODE" = fetch ] && echo "  ok          $dest"
  elif [ "$MODE" = check ]; then
    if [ "$REASON" = missing ]; then
      echo "  MISSING     $dest" >&2
    else
      echo "  DIFFERENT   $dest ($REASON)" >&2
    fi
    FAILED=$((FAILED + 1))
  else
    PENDING="$PENDING $dest"
  fi
done 3<"$LOCK"

if [ "$MODE" = check ]; then
  if [ "$FAILED" -gt 0 ]; then
    echo "error: $FAILED of $TOTAL model files missing or different in $DIR." >&2
    echo "  The models are not in git: run tool/fetch_models.sh (needs HF_TOKEN), then rebuild." >&2
    exit 1
  fi
  echo "models verified: $TOTAL files in $DIR"
  exit 0
fi

# A gated file to fetch and no token: say so before downloading anything.
load_token
if [ -z "$HF_TOKEN" ]; then
  gated_missing=
  while read -r kind dest size sha access repo rest <&3; do
    case $kind in hf | yolo26n-rawhead) ;; *) continue ;; esac
    case " $PENDING " in *" $dest "*) ;; *) continue ;; esac
    [ "$access" = gated ] && gated_missing="$gated_missing $dest" && gated_repo=$repo
  done 3<"$LOCK"
  if [ -n "$gated_missing" ]; then
    echo "error: HF_TOKEN is not set, and these files come from a gated repo:$gated_missing" >&2
    gated_hint "$gated_repo" >&2
    exit 1
  fi
fi

# Pass 2: fetch (or derive) what is missing.
FETCHED=0
FAILURES=
while read -r kind dest size sha access repo revision path src_size src_sha <&3; do
  case $kind in hf | yolo26n-rawhead) ;; *) continue ;; esac
  case " $PENDING " in *" $dest "*) ;; *) continue ;; esac
  echo "  fetching    $dest ($size bytes, $repo@$(printf '%.8s' "$revision"))"
  REASON=
  if [ "$kind" = yolo26n-rawhead ]; then
    derive_yolo "$dest" "$size" "$sha" "$repo" "$revision" "$path" "$src_size" "$src_sha" && ok=1 || ok=0
  else
    download "$repo" "$revision" "$path" "$access" "$size" "$sha" "$DIR/$dest" && ok=1 || ok=0
  fi
  if [ "$ok" = 1 ]; then
    FETCHED=$((FETCHED + 1))
    echo "  ok          $dest"
  else
    FAILED=$((FAILED + 1))
    FAILURES="$FAILURES
  FAILED      $dest: $REASON"
    echo "  FAILED      $dest: $REASON" >&2
  fi
done 3<"$LOCK"

echo "fetch_models: $TOTAL files in $DIR: $PRESENT already verified, $FETCHED fetched, $FAILED failed"
if [ "$FAILED" -gt 0 ]; then
  printf '%s\n' "$FAILURES" | sed '/^$/d' >&2
  exit 1
fi
