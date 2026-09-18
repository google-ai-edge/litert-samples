#!/system/bin/sh
# litert_lm_harness.sh
#
# Runs LiteRT-LM's litert_lm_advanced_main --benchmark on one Android device,
# once per entry in --runs (a backend name; repeat it for a second run), and
# keeps everything a run produces under one output directory so a Device Run
# session can pull it:
#   <run>.log          the binary's stdout + stderr (BenchmarkInfo is here)
#   <run>.exit         its exit code
#   metrics_<run>.pb   litert.lm.proto.LitertLmMetricsList (--metric_proto_file_path)
#   <run>_logcat.txt   logcat for that run
#   provenance.txt     device props, sha256 of the binary, the .so set and the model
# <run> is the backend name, with _2, _3 ... appended for repeats (cpu, gpu, cpu_2, gpu_2).
# Any other --flag is passed to the binary unchanged.
#
# Shape follows litert-samples benchmark/multi_benchmark_harness.sh.
# Compatible with Android /system/bin/sh (toybox / mksh).

set -e

DIR="/data/local/tmp"
BINARY="litert_lm_advanced_main"
MODEL=""
RUNS=""
OUTPUT_DIR=""
PREFILL_TOKENS=128
DECODE_TOKENS=256
MAX_NUM_TOKENS=1024
COOLDOWN=30
BINARY_ARGS=""

for arg in "$@"; do
  case "$arg" in
    --dir=*) DIR="${arg#*=}" ;;
    --binary=*) BINARY="${arg#*=}" ;;
    --model=*) MODEL="${arg#*=}" ;;
    --runs=*) RUNS="${arg#*=}" ;;
    --output_dir=*) OUTPUT_DIR="${arg#*=}" ;;
    --prefill_tokens=*) PREFILL_TOKENS="${arg#*=}" ;;
    --decode_tokens=*) DECODE_TOKENS="${arg#*=}" ;;
    --max_num_tokens=*) MAX_NUM_TOKENS="${arg#*=}" ;;
    --cooldown=*) COOLDOWN="${arg#*=}" ;;
    --*) BINARY_ARGS="$BINARY_ARGS $arg" ;;
    *)
      echo "Error: unexpected argument $arg (flags start with --)" >&2
      exit 1
      ;;
  esac
done

if [ -z "$RUNS" ]; then
  echo "Error: no runs specified. Use --runs=cpu,gpu" >&2
  exit 1
fi
if [ -z "$MODEL" ]; then
  echo "Error: no model specified. Use --model=<file under --dir>" >&2
  exit 1
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$DIR/output"
fi
if [ ! -f "$DIR/$BINARY" ]; then
  echo "Error: binary not found at $DIR/$BINARY" >&2
  exit 1
fi
if [ ! -f "$DIR/$MODEL" ]; then
  echo "Error: model not found at $DIR/$MODEL" >&2
  exit 1
fi

# A pushed binary may arrive without the execute bit; the .so files are dlopen'ed and need none.
chmod 755 "$DIR/$BINARY"

mkdir -p "$OUTPUT_DIR"
RUN_LIST=$(echo "$RUNS" | tr ',' ' ')

PROV="$OUTPUT_DIR/provenance.txt"
{
  echo "date: $(date)"
  echo "product: $(getprop ro.product.model) ($(getprop ro.product.device))"
  echo "build: $(getprop ro.build.fingerprint)"
  echo "kernel: $(uname -a)"
  echo "dir: $DIR"
  echo "runs: $RUNS"
  echo "prefill_tokens: $PREFILL_TOKENS decode_tokens: $DECODE_TOKENS max_num_tokens: $MAX_NUM_TOKENS"
  echo "binary_args:$BINARY_ARGS"
  echo "sha256:"
  (cd "$DIR" && sha256sum "$BINARY" "$MODEL" *.so 2>/dev/null) || true
} > "$PROV"
cat "$PROV"

echo "================================================="
echo "LiteRT-LM benchmark harness"
echo "Binary:     $DIR/$BINARY"
echo "Model:      $DIR/$MODEL"
echo "Runs:       $RUNS"
echo "Tokens:     prefill $PREFILL_TOKENS, decode $DECODE_TOKENS, max_num_tokens $MAX_NUM_TOKENS"
echo "Extra args:$BINARY_ARGS"
echo "Output dir: $OUTPUT_DIR"
echo "Cooldown:   ${COOLDOWN}s"
echo "================================================="

FIRST_RUN=true
FAILED=0
SEEN=""
for BACKEND in $RUN_LIST; do
  # Name repeats of a backend cpu, cpu_2, cpu_3 ...
  N=1
  for S in $SEEN; do
    [ "$S" = "$BACKEND" ] && N=$((N + 1))
  done
  SEEN="$SEEN $BACKEND"
  if [ "$N" -eq 1 ]; then RUN="$BACKEND"; else RUN="${BACKEND}_$N"; fi

  if [ "$FIRST_RUN" = "true" ]; then
    FIRST_RUN=false
  elif [ "$COOLDOWN" -gt 0 ]; then
    echo ""
    echo "Cooling down for ${COOLDOWN}s ..."
    sleep "$COOLDOWN"
  fi

  echo ""
  echo ">>> Starting run: [$RUN] (backend $BACKEND)"
  LOG_FILE="$OUTPUT_DIR/$RUN.log"
  EXIT_FILE="$OUTPUT_DIR/$RUN.exit"
  LOGCAT_FILE="$OUTPUT_DIR/${RUN}_logcat.txt"
  METRICS_FILE="$OUTPUT_DIR/metrics_$RUN.pb"

  echo "thermal before [$RUN]: $(dumpsys thermalservice 2>/dev/null | grep -m1 'Thermal Status' || echo unavailable)" >> "$PROV"
  logcat -c 2>/dev/null || true

  CMD="./$BINARY --backend=$BACKEND --model_path=$DIR/$MODEL --benchmark=true --benchmark_prefill_tokens=$PREFILL_TOKENS --benchmark_decode_tokens=$DECODE_TOKENS --max_num_tokens=$MAX_NUM_TOKENS --metric_proto_file_path=$METRICS_FILE$BINARY_ARGS"
  echo "Executing: cd $DIR && LD_LIBRARY_PATH=$DIR $CMD"
  echo "Log file:  $LOG_FILE"

  STATUS=0
  (cd "$DIR" && eval "LD_LIBRARY_PATH=$DIR $CMD") > "$LOG_FILE" 2>&1 < /dev/null || STATUS=$?
  echo "$STATUS" > "$EXIT_FILE"
  logcat -d > "$LOGCAT_FILE" 2>/dev/null || true
  echo "thermal after  [$RUN]: $(dumpsys thermalservice 2>/dev/null | grep -m1 'Thermal Status' || echo unavailable)" >> "$PROV"

  if [ "$STATUS" -eq 0 ]; then
    echo ">>> [$RUN] completed (exit 0)"
  else
    echo ">>> [$RUN] failed with exit code $STATUS (see $LOG_FILE)" >&2
    FAILED=$((FAILED + 1))
  fi
  grep -E "Init Total|Time to first token|Prefill Turn|Prefill Speed|Decode Turn|Decode Speed|[Pp]eak memory" "$LOG_FILE" || true
done

echo ""
echo "================================================="
echo "All runs finished; $FAILED failed (see the .exit files). Artifacts in: $OUTPUT_DIR"
ls -l "$OUTPUT_DIR"
echo "================================================="
# Always exit 0 so the session's pull step runs; each run's own exit code is in <run>.exit.
exit 0
