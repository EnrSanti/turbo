#!/bin/bash -l
## SLURM job for MeluXina: the restart series (run_restart_series.sh) on a WHOLE GPU node,
## 4 runs at the same time (one per A100). Default series: MiniZinc Challenge 2024
## (benchmarks/mznc2024/fzn, 100 instances), 1200 s, no restart and -restart 10
## = 200 runs = 66.7 GPU-h = ~17 h on one node.
##
## Submit from the turbo-1.2.8 directory, on a LOGIN node:
##   1 node, everything (build + all runs, ~17 h):
##       sbatch meluxina_restart_job.sh
##   N nodes in parallel (job array, one shard of the instances per node, ~17/N h each); the binary
##   must already exist (it is NOT built by the array, N builds in the same directory would clash):
##       sbatch meluxina_restart_job.sh --build-only                       # once, ~5 min (or --qos=test)
##       sbatch --array=0-3 --time=05:30:00 meluxina_restart_job.sh        # 4 nodes, ~4.2 h each
##   Extra arguments go to run_restart_series.sh (e.g. --dry-run, --norestart-only).
## Follow it:   squeue --me      tail -f slurm-restart-<jobid>[_<task>].out      cancel: scancel <jobid>
##
## Resuming: every job of the same submission writes into logs/mznc2024_<jobid>/; a run whose log
## is complete is skipped. To finish a series killed by the time limit, resubmit with the same folder:
##       sbatch --export=ALL,OUT_DIR=logs/mznc2024_<jobid> meluxina_restart_job.sh --no-build
##   (for an array, add the same --array=0-N; each task redoes only its own unfinished runs)
##
## MeluXina gives whole nodes in exclusive mode and bills the whole node, so this job asks for all
## 4 GPUs and uses them all.
##
## BEFORE THE FIRST SUBMISSION:
##   - flatten the instances on the laptop (benchmarks/mznc2024/flatten_mznc2024.sh) and copy
##     benchmarks/mznc2024/fzn/ to MeluXina
##   - check the module names on a compute node (module avail CUDA ; module avail CMake)
#SBATCH --job-name=turbo-restart
#SBATCH --account=p201230
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=4
#SBATCH --time=19:00:00
#SBATCH --output=slurm-restart-%A_%a.out
#SBATCH --error=slurm-restart-%A_%a.out

set -u

## Software environment (CUDA 12.6 is what this code was built with).
ENV_MODULE=${ENV_MODULE:-env/release/2024.1}
CUDA_MODULE=${CUDA_MODULE:-CUDA}
CMAKE_MODULE=${CMAKE_MODULE:-CMake}
module load "$ENV_MODULE" || echo "warning: module $ENV_MODULE not found"
module load "$CUDA_MODULE" || { echo "error: module $CUDA_MODULE not found (module avail CUDA)"; exit 1; }
module load "$CMAKE_MODULE" || { echo "error: module $CMAKE_MODULE not found (module avail CMake)"; exit 1; }
module load "${XML_MODULE:-libxml2}" || { echo "error: module libxml2 not found (module avail libxml2)"; exit 1; }
module list 2>&1

cd "$SLURM_SUBMIT_DIR"
echo "job $SLURM_JOB_ID (array ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}) on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

## Compute nodes usually have no internet access, so the dependencies cannot be downloaded during
## the build. Copy build/gpu-release/_deps from the laptop to deps_src/ and keep this line.
export DEPS_SRC="$SLURM_SUBMIT_DIR/deps_src"

export CUDA_ARCH=80          # A100
export BUILD_DIR=${BUILD_DIR:-build/gpu-release-a100}   # override per job to avoid sharing a binary; default: fixed name: every job lands on a different node (mel2xxx), keep one build
export JOBS=$SLURM_CPUS_PER_TASK
export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")

## The series: MZC 2024, 1200 s, no restart and -restart 10 (override with --export=ALL,TIMEOUTS=...,RESTARTS=...).
export TIMEOUTS=${TIMEOUTS:-1200}
export RESTARTS=${RESTARTS:-"NONE 10"}
export OUT_DIR=${OUT_DIR:-logs/mznc2024_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}

EXTRA=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  export SHARD="$((SLURM_ARRAY_TASK_ID - ${SLURM_ARRAY_TASK_MIN:-0}))/${SLURM_ARRAY_TASK_COUNT}"
  EXTRA=(--no-build)          # array tasks never build (see above)
  echo "array task: shard $SHARD of the instances, --no-build"
fi

srun --cpus-per-task=$SLURM_CPUS_PER_TASK --gpus-per-task=4 ./run_restart_series.sh "${EXTRA[@]}" "$@"
