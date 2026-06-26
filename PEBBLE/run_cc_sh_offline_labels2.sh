#!/bin/bash
#SBATCH --job-name=Gary_s1_500
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:30:00
#SBATCH --output=logs/offline/%x_%j.out
#SBATCH --error=logs/offline/%x_%j.err

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

# Always use env python explicitly to avoid cluster default python bleed-through
if [[ -x "$BPR_ENV/bin/python" ]]; then
    PYTHON_BIN="$BPR_ENV/bin/python"
else
    PYTHON_BIN="$(command -v python)"
fi

echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "python=$(which python)"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PY'
import os, sys
print('sys.executable=', sys.executable)
print('sys.prefix=', sys.prefix)
print('CONDA_PREFIX=', os.environ.get('CONDA_PREFIX'))
print('CONDA_DEFAULT_ENV=', os.environ.get('CONDA_DEFAULT_ENV'))
PY

# Fail fast on required deps
"$PYTHON_BIN" - <<'PY'
import gym
import imageio
import imageio_ffmpeg
print('deps OK:', gym.__version__)
PY

# 3. Offline-label training configuration (override with env vars if needed)
OFFLINE_QUERY_DIR=${OFFLINE_QUERY_DIR:-/home/marzii/cmput654/PEBBLE/logs/human_queries}
OFFLINE_JOB_ID=${OFFLINE_JOB_ID:-4484717}
JOB_NAME=${JOB_NAME:-Gary_s1}
OFFLINE_LABELS_CSV=${OFFLINE_LABELS_CSV:-human_labels_${JOB_NAME}.csv}
SEED=${SEED:-12345}
OFFLINE_HUMAN_MAX_LABELS=${OFFLINE_HUMAN_MAX_LABELS:-200}
OFFLINE_CONTINUE_WITH_SYNTHETIC=${OFFLINE_CONTINUE_WITH_SYNTHETIC:-true}
MAX_FEEDBACK=${MAX_FEEDBACK:-500}
SAVE_LAST_TRAIN_EPISODE_VIDEOS=${SAVE_LAST_TRAIN_EPISODE_VIDEOS:-true}
LAST_TRAIN_VIDEO_COUNT=${LAST_TRAIN_VIDEO_COUNT:-5}
LAST_TRAIN_VIDEO_FPS=${LAST_TRAIN_VIDEO_FPS:-30}

LABEL_PATH="${OFFLINE_QUERY_DIR}/${OFFLINE_JOB_ID}/${OFFLINE_LABELS_CSV}"
if [[ ! -f "$LABEL_PATH" ]]; then
    echo "ERROR: labels file not found: $LABEL_PATH"
    exit 1
fi

# export COMPARE_RUN_DIR="$SCRATCH/compare_runs/pebble/${SLURM_JOB_ID}"
# mkdir -p "$COMPARE_RUN_DIR"
# echo "COMPARE_RUN_DIR=$COMPARE_RUN_DIR"

# 4. Headless MuJoCo
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset DISPLAY
export PYTHONWARNINGS="ignore::DeprecationWarning"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__, "compiled_cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
PY

# Save outputs under scratch
export COMPARE_RUN_DIR="${SCRATCH}/compare_runs/pebble/cmput654/offline/${JOB_NAME}_500_${OFFLINE_JOB_ID}"
mkdir -p "$COMPARE_RUN_DIR"
echo "COMPARE_RUN_DIR=$COMPARE_RUN_DIR"

echo "starting offline-label PEBBLE training..."
echo "hostname=$(hostname)"
echo "python=$(which python)"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "jobid=${SLURM_JOB_ID:-unknown}"
echo "offline_query_dir=$OFFLINE_QUERY_DIR"
echo "offline_job_id=$OFFLINE_JOB_ID"
echo "offline_labels_csv=$OFFLINE_LABELS_CSV"
echo "offline_human_max_labels=$OFFLINE_HUMAN_MAX_LABELS"
echo "offline_continue_with_synthetic=$OFFLINE_CONTINUE_WITH_SYNTHETIC"
echo "max_feedback=$MAX_FEEDBACK"
echo "save_last_train_episode_videos=$SAVE_LAST_TRAIN_EPISODE_VIDEOS"
echo "last_train_video_count=$LAST_TRAIN_VIDEO_COUNT"
echo "last_train_video_fps=$LAST_TRAIN_VIDEO_FPS"

start_time=$(date +%s)

cd ~/cmput654/PEBBLE || exit 1

"$PYTHON_BIN" train_PEBBLE.py \
  env=walker_walk seed=$SEED \
  device=cuda \
  use_offline_human_labels=true \
  collect_human_queries=false \
  offline_human_query_dir=$OFFLINE_QUERY_DIR \
  offline_human_job_id=$OFFLINE_JOB_ID \
  offline_human_labels_csv=$OFFLINE_LABELS_CSV \
    offline_human_max_labels=$OFFLINE_HUMAN_MAX_LABELS \
    offline_human_continue_with_synthetic=$OFFLINE_CONTINUE_WITH_SYNTHETIC \
    save_last_train_episode_videos=$SAVE_LAST_TRAIN_EPISODE_VIDEOS \
    last_train_video_count=$LAST_TRAIN_VIDEO_COUNT \
    last_train_video_fps=$LAST_TRAIN_VIDEO_FPS \
  feed_type=0 \
  num_unsup_steps=9000 num_train_steps=1000000 \
    num_interact=20000 max_feedback=$MAX_FEEDBACK \
    reward_batch=20 reward_update=20 \
  agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005

end_time=$(date +%s)
echo "run time $((end_time-start_time)) sec"
