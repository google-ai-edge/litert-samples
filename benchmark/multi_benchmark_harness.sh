#!/system/bin/sh
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
# ==============================================================================
#
# multi_benchmark_harness.sh
#
# A lightweight multi-run benchmark harness for Android.
# Executes a target binary (e.g. LiteRT benchmark_model) multiple times
# with different backend configurations (CPU, GPU, NPU) on the same device.
#
# Compatible with Android /system/bin/sh (toybox / mksh).

set -e

# Default settings
BINARY="/data/local/tmp/benchmark_model"
RUNS=""
COOLDOWN=10
OUTPUT_DIR="/data/local/tmp/output"
COMMON_ARGS=""
KEEP_SCREEN_ON=false

# 1. Parse global harness flags first
for arg in "$@"; do
  case "$arg" in
    --binary=*)
      BINARY="${arg#*=}"
      ;;
    --runs=*|--runspec_ids=*)
      RUNS="${arg#*=}"
      ;;
    --cooldown=*|--sleep_seconds=*)
      COOLDOWN="${arg#*=}"
      ;;
    --output_dir=*)
      OUTPUT_DIR="${arg#*=}"
      ;;
    --common_args=*)
      COMMON_ARGS="$COMMON_ARGS ${arg#*=}"
      ;;
    --keep_screen_on=*|--keep-screen-on=*)
      KEEP_SCREEN_ON="${arg#*=}"
      ;;
    --keep_screen_on|--keep-screen-on)
      KEEP_SCREEN_ON=true
      ;;
  esac
done

# Validate inputs
if [ -z "$RUNS" ]; then
  echo "Error: No runs specified. Specify via --runs=cpu,gpu,npu" >&2
  exit 1
fi

if [ ! -f "$BINARY" ]; then
  echo "Error: Benchmark binary not found at $BINARY" >&2
  exit 1
fi
chmod +x "$BINARY"

mkdir -p "$OUTPUT_DIR"

# Convert comma-separated list into space-separated
RUN_LIST=$(echo "$RUNS" | tr ',' ' ')

# Helper to check if an argument is targeted to a specific run (e.g. --cpu_... or --gpu_... or --npu_...)
is_run_specific() {
  _test_arg="$1"
  for _r in $RUN_LIST; do
    case "$_test_arg" in
      --${_r}_*)
        return 0
        ;;
    esac
  done
  return 1
}

# Collect remaining un-prefixed flags into COMMON_ARGS
for arg in "$@"; do
  case "$arg" in
    --binary=*|--runs=*|--runspec_ids=*|--cooldown=*|--sleep_seconds=*|--output_dir=*|--keep_screen_on*|--common_args=*)
      ;;
    *)
      if ! is_run_specific "$arg"; then
        COMMON_ARGS="$COMMON_ARGS $arg"
      fi
      ;;
  esac
done

# Optional: Keep screen awake during execution
if [ "$KEEP_SCREEN_ON" = "true" ]; then
  echo "Keeping screen awake..."
  svc power stayon true 2>/dev/null || true
  trap 'svc power stayon false 2>/dev/null || true' EXIT INT TERM
fi

echo "================================================="
echo "Starting Multi-Configuration Benchmark"
echo "Binary:      $BINARY"
echo "Runs:        $RUNS"
echo "Common Args: $COMMON_ARGS"
echo "Output dir:  $OUTPUT_DIR"
echo "Cooldown:    ${COOLDOWN}s"
echo "================================================="

FIRST_RUN=true
for RUN in $RUN_LIST; do
  # Thermal cooldown between runs (skip before first run)
  if [ "$FIRST_RUN" = "true" ]; then
    FIRST_RUN=false
  else
    if [ "$COOLDOWN" -gt 0 ]; then
      echo ""
      echo "Cooling down for ${COOLDOWN}s to mitigate thermal throttling..."
      sleep "$COOLDOWN"
    fi
  fi

  echo ""
  echo "-------------------------------------------------"
  echo ">>> Starting run: [$RUN]"
  echo "-------------------------------------------------"

  RUN_ARGS=""
  RUN_ENV=""

  # Extract run-specific configuration:
  # 1. Bundled args: --<run>_args="..."
  # 2. Bundled env:  --<run>_env="..."
  # 3. Dedicated library paths: --<run>_ld_library_path=... --<run>_adsp_library_path=...
  # 4. Prefixed args: --<run>_<flag>=<val> (e.g. --cpu_num_threads=4 -> --num_threads=4)
  for arg in "$@"; do
    case "$arg" in
      --${RUN}_args=*)
        RUN_ARGS="$RUN_ARGS ${arg#*=}"
        ;;
      --${RUN}_env=*)
        RUN_ENV="$RUN_ENV ${arg#*=}"
        ;;
      --${RUN}_ld_library_path=*)
        RUN_ENV="$RUN_ENV LD_LIBRARY_PATH=${arg#*=}"
        ;;
      --${RUN}_adsp_library_path=*)
        RUN_ENV="$RUN_ENV ADSP_LIBRARY_PATH=${arg#*=}"
        ;;
      --${RUN}_*)
        flag_and_val="${arg#--${RUN}_}"
        RUN_ARGS="$RUN_ARGS --$flag_and_val"
        ;;
    esac
  done

  # Default isolated output files so subsequent runs don't overwrite each other
  DEFAULT_OUT_ARGS="--result_file_path=${OUTPUT_DIR}/results_${RUN}.pb --model_runtime_info_output_file=${OUTPUT_DIR}/runtime_info_${RUN}.pb"

  LOG_FILE="${OUTPUT_DIR}/${RUN}.log"
  LOGCAT_FILE="${OUTPUT_DIR}/${RUN}_logcat.txt"

  # Clear logcat to capture fresh logs for this run
  logcat -c 2>/dev/null || true

  CMD="$BINARY $COMMON_ARGS $DEFAULT_OUT_ARGS $RUN_ARGS"
  echo "Executing: $RUN_ENV $CMD"
  echo "Log file:  $LOG_FILE"

  # Execute with environment variables and redirect output
  STATUS=0
  eval "$RUN_ENV $CMD" > "$LOG_FILE" 2>&1 || STATUS=$?

  # Capture logcat dump
  logcat -d > "$LOGCAT_FILE" 2>/dev/null || true

  if [ $STATUS -eq 0 ]; then
    echo ">>> [$RUN] Completed successfully!"
  else
    echo ">>> [$RUN] Failed with exit code $STATUS (check $LOG_FILE)" >&2
  fi
done

echo ""
echo "================================================="
echo "All benchmark runs completed."
echo "Artifacts saved to: $OUTPUT_DIR"
echo "================================================="
