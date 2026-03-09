#!/bin/bash
# Ablation Sweep: Token SAE Latent Space Capacity
# Tests Expansion Factors: 8x, 32x, 64x, 128x

set -e

# Run from script directory so Python finds sparse_autoencoder.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Override with env PROTEIN_DIR or BASE; default for cluster
PROTEIN_DIR="${PROTEIN_DIR:-${BASE:-../proteins_layer47}}"
TAU=0.90
EPOCHS=100
BATCH_SIZE=16

LATENT_DIMS=(1024 4096 8192 16384)

echo "Starting SAE Latent Capacity Ablation Study..."
echo "Targeting Protein Directory: $PROTEIN_DIR"
echo "---------------------------------------------------"

for D_LATENT in "${LATENT_DIMS[@]}"; do
    EXPANSION=$((D_LATENT / 128))
    OUTPUT_DIR="token_sae_output_E${EXPANSION}"

    echo ">>> Launching Job: Expansion Factor ${EXPANSION}x (Latent Size: ${D_LATENT})"
    echo ">>> Output Directory: ${OUTPUT_DIR}"

    python train_token_sae.py \
        --protein_dir "$PROTEIN_DIR" \
        --d_latent "$D_LATENT" \
        --tau "$TAU" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR"

    echo ">>> Completed Job: E${EXPANSION}"
    echo "---------------------------------------------------"
done

echo "All ablation runs completed successfully."
