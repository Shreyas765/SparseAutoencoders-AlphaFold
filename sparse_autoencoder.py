"""
Token-based Sparse Autoencoder for pair representations (research-grade pipeline).

- No spatial interpolation: preserves true L×L sequence length per protein.
- Single SAE over 128-d tokens (shared weights across L² positions).
- Adaptive top-k via CDF threshold (vectorized); optional entropy penalty in training.
- Packed batching: (sum L_i², 128) for variable-length proteins.

Legacy Lanczos/interpolation and batch-coupled BatchTopK code lives in
sparse_autoencoder_legacy.py (deprecated; use train_token_sae.py for training).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List


# ============================================================================
# Adaptive Top-K Softmax (vectorized, CDF-based dynamic sparsity)
# ============================================================================

class AdaptiveTopKSoftmax(nn.Module):
    """
    Dynamic sparsity via CDF threshold: find smallest k such that CDF_k >= tau.
    Fully vectorized implementation without Python loops.
    """
    def __init__(self, tau: float = 0.90):
        super().__init__()
        self.tau = tau

    def forward(self, z):
        # 1. Softmax to get probabilities
        p = torch.softmax(z, dim=-1)

        # 2. Sort descending to prepare for CDF
        p_sorted, indices = torch.sort(p, dim=-1, descending=True)

        # 3. Calculate CDF and find dynamic k
        cdf = torch.cumsum(p_sorted, dim=-1)
        k_per_sample = (cdf >= self.tau).int().argmax(dim=-1) + 1
        k_per_sample = k_per_sample.clamp(min=1, max=z.size(-1))

        # 4. Vectorized mask: argsort(indices) gives rank of each original element
        ranks = torch.argsort(indices, dim=-1)
        mask = (ranks < k_per_sample.unsqueeze(-1)).float()
        mask = mask.detach()

        return torch.relu(z) * mask


# ============================================================================
# Token-based SAE (one autoencoder for all channels, no spatial interpolation)
# ============================================================================

class TokenSparseAutoencoder(nn.Module):
    """
    Single SAE over 128-d pairwise tokens. Input (L^2, 128), latent (L^2, d_latent).
    Returns recon, p_softmax (for entropy loss), and latents.
    """
    def __init__(self, d_in: int = 128, d_latent: int = 3000, tau: float = 0.90):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.encoder = nn.Linear(d_in, d_latent)
        self.adaptive_topk = AdaptiveTopKSoftmax(tau=tau)
        self.decoder = nn.Linear(d_latent, d_in)

    def forward(self, x_packed):
        # x_packed shape: (sum L_i^2, 128)
        z = self.encoder(x_packed)
        p_softmax = torch.softmax(z, dim=-1)
        latents = self.adaptive_topk(z)
        recon_packed = torch.tanh(self.decoder(latents))  # [-1, 1] to match normalized input
        return recon_packed, p_softmax, latents


# ============================================================================
# Packing collate and no-interpolation dataset for token pipeline
# ============================================================================

def pack_proteins_collate(batch):
    """
    Packs a list of protein tensors of shape (L_i, L_i, 128) into one tensor
    (sum L_i^2, 128). Returns (packed_batch, original_shapes) for unpacking.
    """
    packed_tokens = []
    original_shapes = []

    for protein in batch:
        # protein shape: (L, L, 128)
        L = protein.shape[0]
        C = protein.shape[2]
        original_shapes.append((L, L, C))
        packed_tokens.append(protein.reshape(-1, C))

    packed_batch = torch.cat(packed_tokens, dim=0)
    return packed_batch, original_shapes


class PairRepresentationDataset(Dataset):
    """
    Loads pair_block_47.npy files without interpolation. Each sample is (L, L, 128).
    Optional per-sample normalization to [-1, 1].
    """
    def __init__(self, data_paths: List[str], normalize: bool = True):
        self.data_paths = data_paths
        self.normalize = normalize

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        arr = np.load(self.data_paths[idx]).astype(np.float32)
        if self.normalize:
            max_val = np.abs(arr).max()
            if max_val > 1e-8:
                arr = arr / max_val
        return torch.from_numpy(arr)


def unpack_reconstructions(recon_packed, original_shapes):
    """
    Split packed (sum L_i^2, 128) back into list of (L_i, L_i, 128) tensors.
    Used by inference to recover per-protein reconstructions.
    """
    lengths = [s[0] * s[1] for s in original_shapes]
    chunks = torch.split(recon_packed, lengths, dim=0)
    return [chunks[i].view(original_shapes[i]) for i in range(len(original_shapes))]
