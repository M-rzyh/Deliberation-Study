#!/bin/bash
#SBATCH --job-name=pebble
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail


# 1. Clean environment
module --force purge
module load StdEnv/2023

# 2. Activate conda properly
export BPR_ENV="/scratch/marzii/envs/bpref654"
CONDA_SH="/scratch/marzii/miniconda3/etc/profile.d/conda.sh"

if [[ -f "$CONDA_SH" ]]; then
    # shellcheck source=/dev/null
    source "$CONDA_SH"
    conda activate "$BPR_ENV" || {
        echo "WARN: conda activate failed, falling back to PATH prepend"
        export PATH="$BPR_ENV/bin:$PATH"
        export CONDA_PREFIX="$BPR_ENV"
    }
else
    echo "WARN: conda.sh not found, falling back to PATH prepend"
    export PATH="$BPR_ENV/bin:$PATH"
    export CONDA_PREFIX="$BPR_ENV"
fi

hash -r
echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "python=$(which python)"
python --version
python - <<'PY'
import os, sys
print('sys.executable=', sys.executable)
print('sys.prefix=', sys.prefix)
print('CONDA_PREFIX=', os.environ.get('CONDA_PREFIX'))
print('CONDA_DEFAULT_ENV=', os.environ.get('CONDA_DEFAULT_ENV'))
PY

# Fail fast on required deps
python - <<'PY'
import gym
import imageio
import imageio_ffmpeg
print('deps OK:', gym.__version__)
PY

# export COMPARE_RUN_DIR="$SCRATCH/compare_runs/pebble/${SLURM_JOB_ID}"
# mkdir -p "$COMPARE_RUN_DIR"
# echo "COMPARE_RUN_DIR=$COMPARE_RUN_DIR"

# 4. Headless MuJoCo
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset DISPLAY
export PYTHONWARNINGS="ignore::DeprecationWarning"
export PYTHONUNBUFFERED=1

python - <<'PY'
import torch
print("torch", torch.__version__, "compiled_cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
PY

echo "starting training..."
echo "Starting task ${SLURM_ARRAY_TASK_ID:-single}"
echo "hostname=$(hostname)"
echo "python=$(which python)"
echo "jobid=${SLURM_JOB_ID:-unknown}"

start_time=`date +%s`

# 5. Run
cd ~/cmput654/PEBBLE || exit 1
#python train_PEBBLE.py env=walker_walk seed=12345 device=cpu
./scripts/walker_walk/500/oracle/run_PEBBLE.sh

end_time=`date +%s`
echo "run time $((end_time-start_time)) sec"
