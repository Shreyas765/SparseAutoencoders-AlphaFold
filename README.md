# Token-Based Sparse Autoencoders for Protein Pair Representations

This repository implements a mathematically rigorous, token-based Sparse Autoencoder (SAE) pipeline designed to extract highly interpretable, monosemantic features from the pairwise representations of protein language models (e.g., MIT Boltz).

Unlike standard convolutional approaches that rely on spatial interpolation (which distorts biological distances), this architecture treats the $L \times L \times 128$ distance matrix as a packed sequence of variable-length tokens. It utilizes a fully vectorized, CDF-based Adaptive Top-$k$ Softmax mechanism to enforce dynamic sparsity, preventing feature superposition while maintaining perfect spatial fidelity.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [End-to-End Pipeline](#end-to-end-pipeline)
  - [Training & Ablation](#1-training--ablation)
  - [Inference & Reconstruction](#2-inference--reconstruction)
  - [Structure Generation](#3-structure-generation)
  - [Evaluation (TM-Score)](#4-evaluation-tm-score)
- [Architecture Details](#architecture-details)

## Installation

Ensure you have a CUDA-compatible GPU and PyTorch 2.0+ installed.

```bash
git clone https://github.com/Shreyas765/SparseAutoencoders-AlphaFold.git
cd SparseAutoencoders-AlphaFold
pip install -r requirements.txt
```

**External Dependencies:**

- Structure generation requires the `run_structure_module.py` script and its associated dependencies (OpenFold).
- TM-Score evaluation requires the **TMalign** binary to be compiled and accessible in your system's PATH.

## Data Preparation

The pipeline expects un-interpolated pairwise representation `.npy` files extracted from the target language model (default layer: 47). These should be organized in a base directory, with each protein containing its respective block.

```
CompleteProteins/
├── 6tf4_A/
│   └── 6tf4_A_pair_block_47.npy  # Shape: (L, L, 128)
├── 7xyz_B/
│   └── 7xyz_B_pair_block_47.npy
...
```

## End-to-End Pipeline

### 1. Training & Ablation

The token SAE flattens spatial dimensions to process $(\sum L_i^2, 128)$ packed tokens in parallel. To conduct an ablation study across different expansion factors (e.g., testing latent capacities of 1024, 4096, 8192), specify the `--d_latent` parameter and assign isolated output directories.

```bash
python train_token_sae.py \
    --protein_dir /path/to/CompleteProteins \
    --d_latent 4096 \
    --tau 0.90 \
    --lambda_entropy 0.01 \
    --epochs 100 \
    --batch_size 16 \
    --output_dir token_sae_output_E32
```

Or run the full ablation sweep (E = 8×, 32×, 64×, 128×):

```bash
bash sweep.sh
```

### 2. Inference & Reconstruction

Once trained, unpack the optimized latent representations back into the physical biological space. This script loads the learned parameters, reconstructs the pairwise interactions, and reshapes them back to the original $(L, L, 128)$ geometries.

```bash
python reconstruct_token_sae.py \
    --protein_dir /path/to/CompleteProteins \
    --checkpoint token_sae_output_E32/token_sae_best.pt \
    --d_latent 4096 \
    --output_dir token_sae_reconstructions
```

### 3. Structure Generation

Convert the reconstructed mathematical distance matrices into 3D Cartesian coordinates. This batch script wraps the underlying structure module, generating physical `.pdb` files for each reconstructed `.npy` tensor.

```bash
python generate_pdbs_from_reconstructions.py \
    --reconst_dir token_sae_reconstructions \
    --output_dir token_sae_pdbs \
    --script_path run_structure_module.py
```

### 4. Evaluation (TM-Score)

Quantify the structural fidelity of the SAE reconstructions. This script aligns the autoencoder-generated structures against the ground-truth reference structures using TMalign, yielding an aggregate measure of topological preservation.

```bash
python compute_tm_scores_token_sae.py \
    --pred_dir token_sae_pdbs \
    --ref_dir pair_block_47_pdbs \
    --output_file tm_scores_E32.json
```

Reference PDBs (`pair_block_47_pdbs/`) must be generated separately from the original pair representations (e.g., using your structure module on the raw `*_pair_block_47.npy` files).

## Architecture Details

### Packing vs. Padding

To process varying sequence lengths ($L$) efficiently without generating quadratic memory bottlenecks ($O(L^2)$), this repository implements **exact tensor packing**. Batches of spatial matrices are flattened and concatenated along the sequence dimension prior to the forward pass, and split by their original lengths during reconstruction.

### Adaptive Top-$k$ Softmax

Traditional SAEs enforce a hardcoded top-$k$ sparsity, which forces complex and simple tokens to utilize the same representational capacity. We implement a **dynamic threshold** mechanism:

1. Latent pre-activations are converted to a probability distribution via **Softmax**.
2. The elements are sorted and evaluated against a **Cumulative Distribution Function (CDF)**.
3. The model adaptively selects the minimum number of features $k$ required to reach the threshold $\tau$ (default: 0.90).
4. A differentiable **entropy penalty** is applied to the loss function to organically sharpen the probability distribution over epochs, minimizing $k$ without degrading reconstruction quality.
