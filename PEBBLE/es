#!/bin/bash
#SBATCH --job-name=bpref_pebble
#SBATCH --account=def-mtaylor3   # ← change this
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --array=0
# ------------------------
# 1. Modules
# ------------------------
module purge
module load python/3.9

# ------------------------
# 2. Activate environment
# ------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate bpref39

start_time=`date +%s`
echo "starting training..."
echo "Starting task $SLURM_ARRAY_TASK_ID"

# ------------------------
# 3. MuJoCo headless setup
# ------------------------
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset DISPLAY

# Optional: silence warnings
export PYTHONWARNINGS="ignore::DeprecationWarning"

# ------------------------
# 4. Go to project
# ------------------------
cd ~/BPref2 || exit 1

# ------------------------
# 5. Run training
# ------------------------
python train_PEBBLE.py \
  env=walker_walk \
  seed=12345 \
  device=cpu

end_time=`date +%s`
runtime=$((end_time-start_time))

echo "run time"
echo $runtime
