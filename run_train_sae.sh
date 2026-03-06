#!/bin/bash
#SBATCH --job-name=token_sae
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=token_sae_%A_%a.out
#SBATCH --error=token_sae_%A_%a.err
#SBATCH --array=0-3

# ============ PARALLEL CONFIG ============
# Array job: each task runs a different D_LATENT in parallel on separate GPUs.
# Task 0 -> 1024, Task 1 -> 4096, Task 2 -> 8192, Task 3 -> 16384
# Edit LATENT_DIMS to change which configs run. Use --array=0 to run single job.
# ========================================

LATENT_DIMS=(1024 4096 8192 16384)
export BASE=/path/to/your/CompleteProteins   # EDIT: Your protein dir on PACE scratch

D_LATENT=${LATENT_DIMS[$SLURM_ARRAY_TASK_ID]}
EXPANSION=$((D_LATENT / 128))
OUTPUT_DIR="token_sae_output_E${EXPANSION}"

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

module load anaconda3  # or: conda activate your_env

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: D_LATENT=${D_LATENT} (${EXPANSION}x) on GPU"
python train_token_sae.py \
    --protein_dir "$BASE" \
    --d_latent "$D_LATENT" \
    --tau 0.90 \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 4 \
    --output_dir "$OUTPUT_DIR"
