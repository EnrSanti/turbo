#!/usr/bin/env bash
# Runs Turbo (barebones) on several problems with several timeouts, one run at a time,
# and saves each run's full output to its own log file.
#
# Usage (from the turbo-1.2.8 directory):
#   ./run_timeout_series.sh                 # default problems and timeouts
#   ./run_timeout_series.sh --dry-run       # only print the commands
#   PROBLEMS="benchmarks/trains15.fzn" TIMEOUTS="100 200" ./run_timeout_series.sh
#
# Logs: logs/<problem>/<problem>_<T>s_<date>.log  (one file per run, never overwritten)

set -u

cd "$(dirname "$0")"

TURBO=./build/gpu-release/turbo
PROBLEMS=${PROBLEMS:-"benchmarks/accap_a3.fzn benchmarks/example_wordpress7_500.fzn"}
TIMEOUTS=${TIMEOUTS:-"100 200 500 1000 2000"}   # seconds
# Same options as your previous runs (they are the defaults, written explicitly for reproducibility).
OPTS=${OPTS:-" -s"}
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

if [ ! -x "$TURBO" ]; then
  echo "error: $TURBO not found or not executable (build it first)" >&2
  exit 1
fi
for p in $PROBLEMS; do
  [ -f "$p" ] || { echo "error: problem file $p not found" >&2; exit 1; }
done

# Two runs on the same GPU would share the SMs and the memory, and distort every statistic.
if [ $DRY_RUN -eq 0 ] && pgrep -x turbo > /dev/null; then
  echo "error: another turbo process is running on the GPU:" >&2
  pgrep -a -x turbo >&2
  echo "wait for it to finish (or kill it) and start again." >&2
  exit 1
fi

total=0
for p in $PROBLEMS; do for t in $TIMEOUTS; do total=$((total + t)); done; done
echo "runs: $(echo $PROBLEMS | wc -w) problems x $(echo $TIMEOUTS | wc -w) timeouts, total solving time ~ $((total / 60)) min"

for p in $PROBLEMS; do
  name=$(basename "$p" .fzn)
  mkdir -p "logs/$name"
  for t in $TIMEOUTS; do
    log="logs/$name/${name}_${t}s_$(date +%Y%m%d_%H%M%S).log"
    cmd="$TURBO -t $((t * 1000)) $OPTS $p"
    if [ $DRY_RUN -eq 1 ]; then
      echo "$cmd  > $log"
      continue
    fi
    echo "[$(date +%H:%M:%S)] $name, timeout ${t}s -> $log"
    {
      echo "% run_timeout_series: $(date --iso-8601=seconds)"
      echo "% command: $cmd"
      echo "% gpu: $(nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null)"
    } > "$log"
    start=$(date +%s)
    # Safety net: kill the run if it hangs 5 minutes past its own timeout.
    timeout --signal=KILL $((t + 300)) $cmd >> "$log" 2>&1
    rc=$?
    end=$(date +%s)
    echo "% exit_code: $rc   wall_time_s: $((end - start))" >> "$log"
    [ $rc -ne 0 ] && echo "  warning: exit code $rc (see $log)"
  done
done
echo "done. logs in logs/"
