#!/bin/bash
# Evaluate one checkpoint with scripts/evaluate.py.
# Submit with:  sbatch --export=ALL,CKPT=/path/to/model_4999.pt,LABEL=baseline slurm/eval.slurm
#SBATCH --job-name=hc-eval
#SBATCH --account=seita_1951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-task=a40:1
#SBATCH --time=01:00:00
#SBATCH --output=/project2/seita_1951/jrajan/slurm-logs/%x-%j.out

: "${CKPT:?set CKPT=/path/to/model.pt}"
: "${LABEL:?set LABEL=name}"

cd /scratch1/$USER/645/csci645-in-hand-rotation
export UV_CACHE_DIR=/scratch1/$USER/.cache/uv
export WARP_CACHE_PATH=/scratch1/$USER/.cache/warp
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

echo "host: $(hostname)  commit: $(git rev-parse --short HEAD)  ckpt: $CKPT  label: $LABEL  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

uv run python scripts/evaluate.py \
  --checkpoint "$CKPT" \
  --label "$LABEL" \
  --out-dir /project2/seita_1951/jrajan/eval
STATUS=$?
echo "end: $(date)  exit: $STATUS"
exit $STATUS
