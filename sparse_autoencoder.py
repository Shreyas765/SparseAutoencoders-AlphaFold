"""
Context-aware token Sparse Autoencoder for protein pair representations.

- No spatial interpolation: preserves true L×L sequence length per protein.
- Context per (i,j) is [pair_ij, pair_ii, pair_jj] (384 dims); decoder targets pair (128 dims).
- Adaptive Top-k via CDF threshold (vectorized); optional entropy penalty in training.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset


# ============================================================================
# Adaptive Top-K Softmax (vectorized, CDF-based dynamic sparsity)
# ============================================================================


class AdaptiveTopKSoftmax(nn.Module):
    """
    Dynamic sparsity via CDF threshold: find smallest k such that CDF_k >= tau.
    Fully vectorized implementation without Python loops.
    """

    def __init__(self, tau: float = 0.90) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Pre-activations, shape (batch, latent_dim).

        Returns:
            Sparse activations, same shape (batch, latent_dim).
        """
        p = torch.softmax(z, dim=-1)
        p_sorted, indices = torch.sort(p, dim=-1, descending=True)
        cdf = torch.cumsum(p_sorted, dim=-1)
        k_per_sample = (cdf >= self.tau).int().argmax(dim=-1) + 1
        k_per_sample = k_per_sample.clamp(min=1, max=z.size(-1))
        ranks = torch.argsort(indices, dim=-1)
        mask = (ranks < k_per_sample.unsqueeze(-1)).float()
        mask = mask.detach()
        return torch.relu(z) * mask


# ============================================================================
# Contextual token SAE (384-d context in, 128-d pair out)
# ============================================================================


class ContextualTokenSAE(nn.Module):
    """
    Encoder maps packed context tokens to latent; decoder reconstructs pair channels only.

    Forward:
        x_context_packed: (S, d_context_in) with S = sum_i L_i^2
        recon_packed: (S, d_recon_out)
        latents: (S, d_latent) after adaptive top-k
        p_softmax: (S, d_latent) softmax of pre-sparsity z (for entropy loss)

    Interventions (research): pass ``latent_hook`` to :meth:`forward` to modify sparse
    codes after adaptive top-k and before decoding, e.g. ablate dimensions or add a
    steering vector (same shape as ``latents``).
    """

    def __init__(
        self,
        d_context_in: int = 384,
        d_latent: int = 4096,
        d_recon_out: int = 128,
        tau: float = 0.90,
    ) -> None:
        super().__init__()
        self.d_context_in = d_context_in
        self.d_latent = d_latent
        self.d_recon_out = d_recon_out
        self.encoder = nn.Linear(d_context_in, d_latent)
        self.adaptive_topk = AdaptiveTopKSoftmax(tau=tau)
        self.decoder = nn.Linear(d_latent, d_recon_out)

    def forward(
        self,
        x_context_packed: Tensor,
        latent_hook: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x_context_packed: (sum L_i^2, d_context_in) stacked context tokens.
            latent_hook: If set, ``f(latents) -> latents'`` applied after sparsity, before
                the decoder. Use for ablations (zero some dims) or steering (add α·v).

        Returns:
            recon_packed: (sum L_i^2, d_recon_out)
            p_softmax: (sum L_i^2, d_latent)
            latents: (sum L_i^2, d_latent) **post-hook** sparse activations fed to decoder
        """
        z: Tensor = self.encoder(x_context_packed)
        p_softmax = torch.softmax(z, dim=-1)
        latents = self.adaptive_topk(z)
        if latent_hook is not None:
            latents = latent_hook(latents)
        recon_packed = torch.tanh(self.decoder(latents))
        return recon_packed, p_softmax, latents


# ============================================================================
# Dataset + collate (context assembly fully vectorized over L)
# ============================================================================


def pack_context_collate(
    batch: List[Tuple[Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, List[Tuple[int, int, int]]]:
    """
    Pack a batch of (x_context, target_pair) into packed tensors.

    Args:
        batch: List of length B; each entry is
            x_context: (L, L, 384)
            target_pair: (L, L, 128)

    Returns:
        packed_context: (sum_i L_i^2, 384)
        packed_targets: (sum_i L_i^2, 128)
        original_shapes: list of (L, L, 128) per sample (for unpacking reconstructions)
    """
    packed_context_list: List[Tensor] = []
    packed_target_list: List[Tensor] = []
    original_shapes: List[Tuple[int, int, int]] = []

    for x_context, target_pair in batch:
        L_i = x_context.shape[0]
        c_ctx = x_context.shape[2]
        c_tgt = target_pair.shape[2]
        original_shapes.append((L_i, L_i, c_tgt))
        packed_context_list.append(x_context.reshape(-1, c_ctx))
        packed_target_list.append(target_pair.reshape(-1, c_tgt))

    packed_context = torch.cat(packed_context_list, dim=0)
    packed_targets = torch.cat(packed_target_list, dim=0)
    return packed_context, packed_targets, original_shapes


class ContextualPairDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Loads pair_block_47.npy as (L, L, 128), builds context (L, L, 384) = [pair, diag_i, diag_j].

    __getitem__ returns (x_context, tensor_target) with tensor_target the normalized pair (L,L,128).
    """

    def __init__(self, data_paths: List[str], normalize: bool = True) -> None:
        self.data_paths = data_paths
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        arr = np.load(self.data_paths[idx]).astype(np.float32)
        if self.normalize:
            max_val = float(np.abs(arr).max())
            if max_val > 1e-8:
                arr = arr / max_val
        tensor: Tensor = torch.from_numpy(arr)
        # tensor: (L, L, 128)
        diag: Tensor = torch.diagonal(tensor, dim1=0, dim2=1).T  # (L, 128)
        L = tensor.shape[0]
        diag_i: Tensor = diag.unsqueeze(1).expand(L, L, 128)
        diag_j: Tensor = diag.unsqueeze(0).expand(L, L, 128)
        x_context: Tensor = torch.cat([tensor, diag_i, diag_j], dim=-1)  # (L, L, 384)
        return x_context, tensor


def unpack_reconstructions(
    recon_packed: Tensor, original_shapes: List[Tuple[int, int, int]]
) -> List[Tensor]:
    """
    Split packed (sum L_i^2, 128) back into list of (L_i, L_i, 128) tensors.

    Args:
        recon_packed: (sum_i L_i^2, C) typically C=128.
        original_shapes: [(L, L, C), ...] per protein.

    Returns:
        List of tensors each (L, L, C).
    """
    lengths: List[int] = [s[0] * s[1] for s in original_shapes]
    chunks: Tuple[Tensor, ...] = torch.split(recon_packed, lengths, dim=0)
    return [chunks[i].view(original_shapes[i]) for i in range(len(original_shapes))]
