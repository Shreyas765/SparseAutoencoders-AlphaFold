#!/bin/bash
# Single-job SLURM script (one GPU, one D_LATENT).
# For parallel sweep across multiple D_LATENT, use run_train_sae.sh (array job).

#SBATCH --job-name=token_sae
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=token_sae_%j.out
#SBATCH --error=token_sae_%j.err

# ============ EDIT THESE ============
export BASE=/path/to/your/CompleteProteins
D_LATENT=3000
# ====================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

module load anaconda3

python train_token_sae.py \
    --protein_dir "$BASE" \
    --d_latent "$D_LATENT" \
    --tau 0.90 \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 4 \
    --output_dir "token_sae_output_d${D_LATENT}"
