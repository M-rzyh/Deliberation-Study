#!/bin/bash
#SBATCH --job-name=tw_p3
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:30:00
#SBATCH --output=logs/offline/%x_%j.out
#SBATCH --error=logs/offline/%x_%j.err

# Time-weighted PEBBLE training: Person 3 (Kiarash) S1 on Walker Walk
# Compares against unweighted baseline: kiarash_s1_4521211

set -euo pipefail

module --force purge
module load StdEnv/2023

export BPR_ENV="/scratch/marzii/envs/bpref654"
CONDA_SH="/scratch/marzii/miniconda3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    conda activate "$BPR_ENV" || {
        export PATH="$BPR_ENV/bin:$PATH"
        export CONDA_PREFIX="$BPR_ENV"
    }
else
    export PATH="$BPR_ENV/bin:$PATH"
    export CONDA_PREFIX="$BPR_ENV"
fi
hash -r

PYTHON_BIN="${BPR_ENV}/bin/python"
echo "python=$PYTHON_BIN"
"$PYTHON_BIN" --version

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset DISPLAY
export PYTHONWARNINGS="ignore::DeprecationWarning"
export PYTHONUNBUFFERED=1

# --- Config ---
STRATEGY=${STRATEGY:-linear}  # linear, sqrt, or log
OFFLINE_QUERY_DIR="/home/marzii/cmput654/PEBBLE/logs/human_queries"
OFFLINE_JOB_ID="4484717"
OFFLINE_LABELS_CSV="human_labels_kiarash_s1.csv"
SEED=${SEED:-12345}

export COMPARE_RUN_DIR="${SCRATCH}/compare_runs/pebble/cmput654/offline/kiarash_s1_tw_${STRATEGY}_${SLURM_JOB_ID}"
mkdir -p "$COMPARE_RUN_DIR"
echo "COMPARE_RUN_DIR=$COMPARE_RUN_DIR"
echo "time_weight_strategy=$STRATEGY"

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
  offline_human_max_labels=200 \
  offline_human_continue_with_synthetic=true \
  time_weight_strategy=$STRATEGY \
  feed_type=0 \
  num_unsup_steps=9000 num_train_steps=1000000 \
  num_interact=20000 max_feedback=1000 \
  reward_batch=50 reward_update=50 \
  label_margin=0.0 teacher_eps_equal=0 \
  agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005

end_time=$(date +%s)
echo "run time $((end_time-start_time)) sec"
