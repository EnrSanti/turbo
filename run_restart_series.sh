#!/usr/bin/env bash
# One script for the whole restart experiment on THIS machine:
#   1. compiles Turbo for this machine's GPU (fresh build directory),
#   2. a smoke test of a few seconds (cooperative launch, number of blocks),
#   3. every instance x every timeout x every restart mode, with ONE run per GPU at the same time
#      (default: the MiniZinc Challenge 2024, 1200 s, no restart and -restart 10),
#   4. the statistics comparison (compare_restart_stats.py) on the results.
#
# Parallel runs: one worker per GPU, each worker sees only its own GPU (CUDA_VISIBLE_DEVICES), and
# takes the next run from a shared queue as soon as its previous run ends. The queue is ordered
# instance by instance (norestart, restart20 of the same instance one after the other), so the two
# modes of an instance usually run at the same time on two different GPUs of the same node.
#
# Results (one folder per series):
#   logs/restart_series_<host>_<date>/          (or $OUT_DIR)
#       build_info.txt, smoke.log, runs.txt     (runs.txt: start/end of every run, with its GPU)
#       <instance>/T1200_norestart.log  T1200_restart20.log
#       comparison.csv                          (written by compare_restart_stats.py)
#   A run whose log already ends with "% exit_code:" is skipped, so the same OUT_DIR can be given
#   again to finish an interrupted series (e.g. a job killed by its time limit).
#
# Expected layout (copy both directories side by side; the build/ directory is not needed):
#   <somewhere>/turbo-1.2.8/   (this repo)
#   <somewhere>/lala-core/     (local clone used by CMakeLists.txt: LOCAL_LALA_CORE=ON)
#
# Usage:
#   ./run_restart_series.sh               # build + smoke test + all runs + comparison
#   ./run_restart_series.sh --dry-run     # print what would be done, run and write nothing
#   ./run_restart_series.sh --build-only  # build + smoke test only
#   ./run_restart_series.sh --no-build    # reuse the binary already in $BUILD_DIR
#   ./run_restart_series.sh --restart-only    # only the -restart runs (drops NONE from RESTARTS)
#   ./run_restart_series.sh --norestart-only  # only the runs without restart
#   (options can be combined, e.g. --no-build --restart-only)
#
# Environment variables (all optional):
#   PROBLEMS   default: every benchmarks/mznc2024/fzn/*.fzn   (benchmarks/mznc2024/flatten_mznc2024.sh)
#   TIMEOUTS   default "1200"           (seconds; several values allowed, e.g. "1000 2000")
#   RESTARTS   default "NONE 10"        (NONE = no -restart option; N = -restart N, i.e. N % idle blocks)
#   OPTS       default "-s"             (extra Turbo options, the same for every run)
#   GPUS       default: every visible GPU (CUDA_VISIBLE_DEVICES, else nvidia-smi), e.g. "0 1 2 3";
#              GPUS="0" = one run at a time
#   SHARD      "K/N": only the instances number K, K+N, K+2N, ... (0-based, sorted by name), to split
#              the series over N jobs (the SLURM job array of meluxina_restart_job.sh sets it)
#   OUT_DIR    default "logs/restart_series_<host>_<date>"
#   CUDA_ARCH  default "native"         (80 = A100, 90 = H100, 89 = RTX 40xx; "native" needs a visible GPU)
#   CUDA_HOME  default: auto (/usr/local/cuda, else nvcc on PATH)
#   BUILD_DIR  default "build/gpu-release-<hostname>"   (never the copied build/gpu-release)
#   JOBS       default: number of cores
#   DEPS_SRC   for machines WITHOUT internet: a copy of this machine's build/gpu-release/_deps
#              (DEPS_SRC/<name>-src for lala_pc, lala_parsing, lala_power, cuda_battery,
#              lala_interval, cpp_peglib, xcsp3parser).
#
# Solving time with the defaults: 100 instances x 2 modes x 1200 s = 66.7 GPU-hours,
# i.e. ~16.7 h on one node with 4 GPUs (less for instances that finish before the timeout).

set -u
cd "$(dirname "$0")"

DRY_RUN=0; DO_BUILD=1; DO_RUNS=1; ONLY=""
for a in "$@"; do
  case "$a" in
    --dry-run)    DRY_RUN=1 ;;
    --build-only) DO_RUNS=0 ;;
    --no-build)   DO_BUILD=0 ;;
    --restart-only)   ONLY=restart ;;
    --norestart-only) ONLY=norestart ;;
    -h|--help)    sed -n '2,57p' "$0"; exit 0 ;;
    *) echo "unknown option: $a (see --help)" >&2; exit 1 ;;
  esac
done

HOST=$(hostname -s 2>/dev/null || hostname)
STAMP=$(date +%Y%m%d_%H%M%S)
PROBLEMS=${PROBLEMS:-$(ls benchmarks/mznc2024/fzn/*.fzn 2>/dev/null | sort | tr '\n' ' ')}
TIMEOUTS=${TIMEOUTS:-"1200"}
RESTARTS=${RESTARTS:-"NONE 10"}
if [ "$ONLY" = "restart" ]; then RESTARTS=$(echo $RESTARTS | tr ' ' '\n' | grep -vx NONE | tr '\n' ' ')
elif [ "$ONLY" = "norestart" ]; then RESTARTS="NONE"; fi
[ -n "${RESTARTS// /}" ] || { echo "error: no restart mode left to run (RESTARTS=\"$RESTARTS\")" >&2; exit 1; }
OPTS=${OPTS:-"-s"}
CUDA_ARCH=${CUDA_ARCH:-native}
BUILD_DIR=${BUILD_DIR:-"build/gpu-release-$HOST"}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
TURBO="$BUILD_DIR/turbo"
OUT=${OUT_DIR:-"logs/restart_series_${HOST}_${STAMP}"}

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "error: $*" >&2; exit 1; }

[ -n "${PROBLEMS// /}" ] || die "no instance: flatten them first (benchmarks/mznc2024/flatten_mznc2024.sh) or set PROBLEMS."

# Shard: keep instances K, K+N, ... of the sorted list.
if [ -n "${SHARD:-}" ]; then
  K=${SHARD%/*}; N=${SHARD#*/}
  [[ "$K" =~ ^[0-9]+$ && "$N" =~ ^[0-9]+$ && $K -lt $N ]] || die "SHARD must be K/N with 0 <= K < N (got $SHARD)"
  PROBLEMS=$(echo $PROBLEMS | tr ' ' '\n' | sort | awk -v k=$K -v n=$N '(NR-1) % n == k' | tr '\n' ' ')
fi

# GPUs: one worker per entry.
if [ -z "${GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ')
  else GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' '); fi
fi
GPUS=$(echo ${GPUS:-0})
NGPU=$(echo $GPUS | wc -w)
FIRST_GPU=${GPUS%% *}

if [ $DRY_RUN -eq 1 ]; then INFO=/dev/null; else mkdir -p "$OUT"; INFO="$OUT/build_info.txt"; fi

# ---------------------------------------------------------------- 1. checks
[ -d ../lala-core/include ] || die "../lala-core not found next to this repo (needed by LOCAL_LALA_CORE=ON)."
for p in $PROBLEMS; do [ -f "$p" ] || die "problem file $p not found"; done

if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then NVCC="$CUDA_HOME/bin/nvcc"
elif [ -x /usr/local/cuda/bin/nvcc ]; then NVCC=/usr/local/cuda/bin/nvcc   # before PATH: /usr/bin/nvcc is often an old distro CUDA
elif command -v nvcc >/dev/null 2>&1; then NVCC=$(command -v nvcc)
else NVCC=""; fi
NVCC_VER=$([ -n "$NVCC" ] && "$NVCC" --version | grep -oE 'release [0-9]+\.[0-9]+' | cut -d' ' -f2)
CUDA_BIN=$(dirname "${NVCC:-/usr/local/cuda/bin/nvcc}")
CUOBJDUMP="$CUDA_BIN/cuobjdump"; [ -x "$CUOBJDUMP" ] || CUOBJDUMP=$(command -v cuobjdump || true)

GPU_LINE=$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
NPROB=$(echo $PROBLEMS | wc -w)
{
  echo "% host: $HOST   date: $(date --iso-8601=seconds)"
  echo "% gpu: ${GPU_LINE:-none visible}   (x$(nvidia-smi -L 2>/dev/null | wc -l) visible)   workers on GPUs: $GPUS"
  echo "% nvcc: ${NVCC:-not found} (release ${NVCC_VER:-?})"
  echo "% cmake: $(cmake --version 2>/dev/null | head -1)"
  echo "% CUDA_ARCH=$CUDA_ARCH  BUILD_DIR=$BUILD_DIR  DEPS_SRC=${DEPS_SRC:-<download>}"
  echo "% lala-core: $(git -C ../lala-core describe --tags --always --dirty 2>/dev/null || echo 'no git')"
  echo "% turbo: $(git describe --tags --always --dirty 2>/dev/null || echo 'no git')"
  echo "% instances: $NPROB${SHARD:+ (shard $SHARD)}   timeouts: $TIMEOUTS   restarts: $RESTARTS   opts: $OPTS"
} | tee -a "$INFO"

if [ $DO_BUILD -eq 1 ]; then
  [ -n "$NVCC" ] || die "nvcc not found: set CUDA_HOME (this code was built with CUDA 12.6)."
  command -v cmake >/dev/null || die "cmake not found (>= 3.24 needed)."
  case "$NVCC_VER" in 12.[6-9]|12.1[0-9]|1[3-9].*) ;; *) echo "warning: nvcc $NVCC_VER; this code was built and tested with CUDA 12.6 (set CUDA_HOME to choose another)." ;; esac
  if [ "$CUDA_ARCH" = "native" ] && [ -z "$GPU_CC" ]; then
    die "CUDA_ARCH=native but no GPU is visible here: set CUDA_ARCH (80 = A100, 90 = H100) or build on a GPU node."
  fi
fi

if [ $DO_RUNS -eq 1 ] && [ $DRY_RUN -eq 0 ] && pgrep -x turbo >/dev/null; then
  pgrep -a -x turbo >&2
  die "another turbo process is using the GPU: it would distort every statistic."
fi

# ---------------------------------------------------------------- 2. build
if [ $DO_BUILD -eq 1 ]; then
  # Fresh build directory per machine: a build/gpu-release copied from elsewhere has absolute paths
  # in its CMakeCache.txt and code for that machine's GPU only.
  CMAKE_ARGS=(-S . -B "$BUILD_DIR" -DGPU=ON -DCMAKE_BUILD_TYPE=Release -DREDUCE_PTX_SIZE=OFF
              -DLOCAL_LALA_CORE=ON -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
  if [ -n "${DEPS_SRC:-}" ]; then
    for d in lala_pc lala_parsing lala_power cuda_battery lala_interval cpp_peglib xcsp3parser; do
      [ -d "$DEPS_SRC/$d-src" ] || die "DEPS_SRC: $DEPS_SRC/$d-src missing"
      CMAKE_ARGS+=("-DFETCHCONTENT_SOURCE_DIR_$(echo $d | tr a-z A-Z)=$(cd "$DEPS_SRC/$d-src" && pwd)")
    done
  fi
  if [ $DRY_RUN -eq 1 ]; then
    echo "  (dry-run) cmake ${CMAKE_ARGS[*]}"
    echo "  (dry-run) cmake --build $BUILD_DIR -j $JOBS --target turbo"
  else
    mkdir -p "$BUILD_DIR"
    say "configure -> $BUILD_DIR (log: $BUILD_DIR/configure.log)"
    cmake "${CMAKE_ARGS[@]}" > "$BUILD_DIR/configure.log" 2>&1 || { tail -30 "$BUILD_DIR/configure.log"; die "configure failed"; }
    say "build with $JOBS jobs (log: $BUILD_DIR/build.log), this takes a while"
    cmake --build "$BUILD_DIR" -j "$JOBS" --target turbo > "$BUILD_DIR/build.log" 2>&1 || { tail -40 "$BUILD_DIR/build.log"; die "build failed"; }
    say "build done"
  fi
fi

if [ $DRY_RUN -eq 0 ]; then
  [ -x "$TURBO" ] || die "$TURBO not found (build first, or drop --no-build)."
  if [ -n "$CUOBJDUMP" ]; then
    ELF=$("$CUOBJDUMP" --list-elf "$TURBO" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')
    RES=$("$CUOBJDUMP" --dump-resource-usage "$TURBO" 2>/dev/null | grep -A1 gpu_barebones_solve \
          | grep -oE 'REG:[0-9]+|STACK:[0-9]+|SHARED:[0-9]+|LOCAL:[0-9]+' | sort -u | tr '\n' ' ')
    { echo "% binary archs: $ELF"; echo "% gpu_barebones_solve resources: $RES"; } | tee -a "$INFO"
    if [ -n "$GPU_CC" ] && ! echo "$ELF" | grep -q "sm_$GPU_CC"; then
      echo "warning: the binary has no sm_$GPU_CC code for this GPU: the kernel may fail to launch." | tee -a "$INFO"
    fi
  fi

  # ---------------------------------------------------------------- 3. smoke test (a few seconds, first GPU)
  # With -restart, Turbo prints a WARNING and disables the restarts if the GPU has no cooperative
  # launch, or reduces the number of blocks if they cannot all be resident at the same time.
  SMOKE="benchmarks/enr_small/t03_queens5.fzn"
  if [ -f "$SMOKE" ]; then
    CUDA_VISIBLE_DEVICES=$FIRST_GPU timeout 120 "$TURBO" -t 20000 -s -restart 20 "$SMOKE" > "$OUT/smoke.log" 2>&1; rc=$?
    nb=$(grep 'mzn-stat: num_blocks=' "$OUT/smoke.log" | tail -1 | cut -d= -f2)
    say "smoke test (GPU $FIRST_GPU): exit $rc, num_blocks=$nb, $(grep -c WARNING "$OUT/smoke.log") warning(s)"
    echo "% smoke: exit=$rc num_blocks=$nb" >> "$INFO"
    grep WARNING "$OUT/smoke.log" | tee -a "$INFO"
    [ $rc -eq 0 ] || die "smoke test failed (exit $rc), see $OUT/smoke.log"
  fi
fi

[ $DO_RUNS -eq 1 ] || { say "--build-only: done. info: $INFO"; exit 0; }

# ---------------------------------------------------------------- 4. the series
# Queue: one line per run "<timeout> <restart> <problem>". Shortest timeout first; inside a timeout,
# instance by instance, so the modes of one instance run side by side on different GPUs.
QUEUE=$(for t in $TIMEOUTS; do for p in $PROBLEMS; do for r in $RESTARTS; do echo "$t $r $p"; done; done; done)
NRUNS=$(echo "$QUEUE" | wc -l)
total=0; for t in $(echo "$QUEUE" | cut -d' ' -f1); do total=$((total + t)); done
say "$NRUNS runs on $NGPU GPU(s) ($GPUS): $((total / 3600)) GPU-h $(((total % 3600) / 60)) min," \
    "wall ~ $((total / NGPU / 3600)) h $(((total / NGPU % 3600) / 60)) min at most; results in $OUT"

run_one() {   # $1 = gpu, $2 = timeout, $3 = restart, $4 = problem
  local g=$1 t=$2 r=$3 p=$4 name ropt cfg log cmd start end rc
  name=$(basename "$p" .fzn)
  if [ "$r" = "NONE" ]; then ropt=""; cfg="T${t}_norestart"
  elif [ "$r" = "EACH" ]; then ropt="-restart_each_block"; cfg="T${t}_restart_each"   # restart when the first block is idle
  else ropt="-restart $r"; cfg="T${t}_restart${r}"; fi
  log="$OUT/$name/$cfg.log"
  cmd="$TURBO -t $((t * 1000)) $OPTS $ropt $p"
  if [ $DRY_RUN -eq 1 ]; then echo "  (dry-run) GPU $g: $cmd > $log"; return; fi
  if [ -f "$log" ] && grep -q '^% exit_code:' "$log"; then say "GPU $g  skip $name $cfg (already done)"; return; fi
  mkdir -p "$OUT/$name"
  say "GPU $g  $name  $cfg"
  {
    echo "% run_restart_series: $(date --iso-8601=seconds)   host: $HOST   gpu_index: $g"
    echo "% command: CUDA_VISIBLE_DEVICES=$g $cmd"
    echo "% gpu: $(nvidia-smi -i "$g" --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | head -1)"
  } > "$log"
  start=$(date +%s)
  echo "$(date --iso-8601=seconds) start GPU $g $name $cfg" >> "$OUT/runs.txt"
  # Safety net: kill the run if it hangs 5 minutes past its own timeout.
  CUDA_VISIBLE_DEVICES=$g timeout --signal=KILL $((t + 300)) $cmd >> "$log" 2>&1
  rc=$?
  end=$(date +%s)
  echo "% exit_code: $rc   wall_time_s: $((end - start))" >> "$log"
  echo "$(date --iso-8601=seconds) end   GPU $g $name $cfg exit=$rc wall=$((end - start))s" >> "$OUT/runs.txt"
  [ $rc -ne 0 ] && echo "  warning: exit code $rc (see $log)"
}

if [ $DRY_RUN -eq 1 ]; then
  i=0; gl=($GPUS)
  echo "$QUEUE" | while read -r t r p; do run_one "${gl[$((i % NGPU))]}" "$t" "$r" "$p"; i=$((i + 1)); done
  exit 0
fi

# Shared queue: each worker takes the next line number under a lock (flock), runs it, repeats.
QFILE="$OUT/.queue"; NEXT="$OUT/.queue_next"; LOCK="$OUT/.queue_lock"
echo "$QUEUE" > "$QFILE"; echo 1 > "$NEXT"
worker() {
  local g=$1 i line
  while true; do
    i=$(flock "$LOCK" bash -c 'i=$(cat "$1"); echo $((i + 1)) > "$1"; echo $i' _ "$NEXT")
    [ "$i" -le "$NRUNS" ] || break
    line=$(sed -n "${i}p" "$QFILE")
    run_one "$g" $line
  done
  say "GPU $g: no more runs"
}
pids=()
for g in $GPUS; do worker "$g" & pids+=($!); sleep 2; done
trap 'echo "interrupted: stopping the workers"; kill ${pids[*]} 2>/dev/null; pkill -P $$ 2>/dev/null; exit 130' INT TERM
wait "${pids[@]}"
rm -f "$QFILE" "$NEXT" "$LOCK"

# ---------------------------------------------------------------- 5. comparison
say "all runs done"
if command -v python3 >/dev/null; then
  python3 compare_restart_stats.py "$OUT"
else
  echo "python3 not found: run  python3 compare_restart_stats.py $OUT  on another machine."
fi
