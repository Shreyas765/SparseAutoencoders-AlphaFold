"""
Matryoshka Sparse Autoencoder for AlphaFold2 Pair Representations
=================================================================

Adapted from "Scaling SAEs to ESM2" methodology for protein pair representations.

Key Design Principles:
----------------------
1. CRITICAL FIX: Each residue pair (i, j) is treated as a single 'token' of dimension 128.
   - Input is reshaped from [Batch, L, L, 128] → [N_pairs, 128] where N_pairs = B * L * L
   - This learns features of PAIRS (hydrogen bonds, hydrophobic contacts, salt bridges)
     rather than memorizing specific protein layouts.

2. TopK Activation: Uses TopK(k=32) instead of L1 penalty for sparsity.
   - More stable training, easier hyperparameter tuning.
   - Favored in recent scaling studies (Anthropic, EleutherAI).

3. Matryoshka Nested Reconstruction: The latent space has nested structure [128, 1024, 2048, 4096].
   - First 128 latents should be most important, next 384 (up to 512) next important, etc.
   - Enables adaptive compute: use fewer latents for fast inference, more for accuracy.

4. Standard SAE Formulation:
   - Encoder: z = TopK(W_enc @ (x - b_dec) + b_enc)
   - Decoder: x_hat = W_dec @ z + b_dec
   - Decoder weights normalized to unit norm after each step.

Author: Mechanistic Interpretability Research
"""

# Disable PyTorch dynamo so optimizer creation does not trigger sympy/mpmath
# (avoids ModuleNotFoundError: No module named 'mpmath.libmp' on some systems)
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import math
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class MatryoshkaSAEConfig:
    """Configuration for Matryoshka SAE."""
    
    # Core dimensions
    input_dim: int = 128  # Feature dimension per pair (FIXED for AF2 pair rep)
    n_latents: int = 4096  # Total number of latent features (increased for better reconstruction)
    expansion_factor: int = 32  # n_latents = expansion_factor * input_dim
    
    # Matryoshka nested dimensions (must be sorted, last = n_latents)
    matryoshka_dims: Tuple[int, ...] = (128, 1024, 2048, 4096)
    
    # Sparsity via TopK (sparsity budget: more active latents = richer reconstruction)
    topk: int = 64  # Number of active latents per input
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    
    # Normalization
    normalize_decoder: bool = True  # Unit-norm decoder columns after each step
    pre_encoder_bias: bool = True  # Subtract b_dec before encoding (standard SAE)
    
    # Device
    device: str = "auto"
    
    def __post_init__(self):
        # Validate matryoshka dims
        if self.matryoshka_dims[-1] != self.n_latents:
            raise ValueError(
                f"Last matryoshka dim ({self.matryoshka_dims[-1]}) must equal n_latents ({self.n_latents})"
            )
        for i in range(1, len(self.matryoshka_dims)):
            if self.matryoshka_dims[i] <= self.matryoshka_dims[i-1]:
                raise ValueError(f"Matryoshka dims must be strictly increasing: {self.matryoshka_dims}")
        
        # Auto-calculate expansion factor if needed
        if self.n_latents != self.expansion_factor * self.input_dim:
            self.expansion_factor = self.n_latents // self.input_dim


# =============================================================================
# 2. TOPK ACTIVATION FUNCTION
# =============================================================================

class TopKActivation(nn.Module):
    """
    TopK activation: keep only the k largest activations, zero out the rest.
    
    This provides explicit sparsity control without L1 penalty tuning.
    Used in recent SAE scaling work (Anthropic dictionary learning, EleutherAI).
    """
    
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [..., n_latents]
            
        Returns:
            Sparse tensor with only top-k values, same shape as input
        """
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(x, self.k, dim=-1)
        
        # Create sparse output
        output = torch.zeros_like(x)
        output.scatter_(-1, topk_indices, topk_values)
        
        return output
    
    def forward_with_indices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns the indices and values of active latents.
        
        Useful for interpretability analysis.
        
        Returns:
            output: Sparse tensor
            indices: Top-k indices of shape [..., k]
            values: Top-k values of shape [..., k]
        """
        topk_values, topk_indices = torch.topk(x, self.k, dim=-1)
        output = torch.zeros_like(x)
        output.scatter_(-1, topk_indices, topk_values)
        return output, topk_indices, topk_values


# =============================================================================
# 3. MATRYOSHKA SPARSE AUTOENCODER
# =============================================================================

class MatryoshkaSAE(nn.Module):
    """
    Matryoshka Sparse Autoencoder for AlphaFold2 Pair Representations.
    
    Architecture:
    -------------
    - Input: (*, 128) where * is any batch dimensions (flattened pairs)
    - Encoder: z_pre = W_enc @ (x - b_dec) + b_enc
    - Activation: z = TopK(z_pre, k=32)
    - Decoder: x_hat = W_dec @ z + b_dec
    
    Matryoshka Training:
    -------------------
    The loss is computed at multiple nested checkpoints [128, 512, 1024, 2048].
    This encourages the first 128 latents to capture the most important features,
    enabling adaptive compute at inference time.
    
    Critical Reshaping:
    ------------------
    If input is [Batch, L, L, 128], it's automatically reshaped to [B*L*L, 128]
    so each (i,j) pair is treated as an independent sample. This is ESSENTIAL
    for learning generalizable pair features rather than protein-specific layouts.
    """
    
    def __init__(self, config: Optional[MatryoshkaSAEConfig] = None, **kwargs):
        super().__init__()
        
        # Allow config or kwargs
        if config is None:
            config = MatryoshkaSAEConfig(**kwargs)
        self.config = config
        
        self.input_dim = config.input_dim
        self.n_latents = config.n_latents
        self.matryoshka_dims = config.matryoshka_dims
        self.topk = config.topk
        self.normalize_decoder = config.normalize_decoder
        self.pre_encoder_bias = config.pre_encoder_bias
        
        # Encoder: maps input_dim -> n_latents
        self.W_enc = nn.Parameter(torch.empty(self.n_latents, self.input_dim))
        self.b_enc = nn.Parameter(torch.zeros(self.n_latents))
        
        # Decoder: maps n_latents -> input_dim (transposed structure)
        self.W_dec = nn.Parameter(torch.empty(self.input_dim, self.n_latents))
        self.b_dec = nn.Parameter(torch.zeros(self.input_dim))
        
        # TopK activation
        self.activation = TopKActivation(k=self.topk)
        
        # Initialize weights
        self._init_weights()
        
        # Track original shape for unflattening
        self._original_shape = None
    
    def _init_weights(self):
        """
        Initialize weights following best practices for SAEs:
        - Encoder: Xavier uniform
        - Decoder: Xavier uniform, then normalize to unit norm
        """
        # Xavier initialization
        nn.init.xavier_uniform_(self.W_enc)
        nn.init.xavier_uniform_(self.W_dec)
        
        # Initialize biases to zero
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)
        
        # Normalize decoder columns to unit norm
        if self.normalize_decoder:
            with torch.no_grad():
                self._normalize_decoder_weights()
    
    @torch.no_grad()
    def _normalize_decoder_weights(self):
        """Normalize decoder weight columns to unit norm."""
        # W_dec shape: (input_dim, n_latents)
        # Normalize each column (each latent's decoder direction)
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)
    
    def _flatten_pairs(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Flatten input to treat each (i,j) pair as independent sample.
        
        Args:
            x: Input tensor, can be:
               - (N, 128): Already flattened pairs
               - (B, L, L, 128): Batch of pair representations
               - (L, L, 128): Single pair representation
        
        Returns:
            x_flat: Tensor of shape (N_pairs, 128)
            original_shape: Tuple to restore original shape
        """
        original_shape = x.shape
        
        if x.ndim == 2 and x.shape[-1] == self.input_dim:
            # Already flat: (N, 128)
            return x, original_shape
        elif x.ndim == 3 and x.shape[-1] == self.input_dim:
            # Single protein: (L, L, 128) -> (L*L, 128)
            L = x.shape[0]
            return x.reshape(-1, self.input_dim), original_shape
        elif x.ndim == 4 and x.shape[-1] == self.input_dim:
            # Batched: (B, L, L, 128) -> (B*L*L, 128)
            return x.reshape(-1, self.input_dim), original_shape
        else:
            raise ValueError(
                f"Unexpected input shape: {x.shape}. "
                f"Expected (N, {self.input_dim}), (L, L, {self.input_dim}), "
                f"or (B, L, L, {self.input_dim})"
            )
    
    def _unflatten_pairs(self, x_flat: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
        """
        Restore flattened output to original shape.
        
        Args:
            x_flat: Flattened tensor of shape (N_pairs, 128)
            original_shape: Original input shape
            
        Returns:
            Tensor restored to original shape
        """
        return x_flat.reshape(original_shape)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse latent representation.
        
        Formula: z = TopK(W_enc @ (x - b_dec) + b_enc)
        
        Args:
            x: Input tensor of shape (*, 128)
            
        Returns:
            z: Sparse latent tensor of shape (*, n_latents)
        """
        # Flatten if needed
        x_flat, self._original_shape = self._flatten_pairs(x)
        
        # Pre-encoder bias subtraction (standard SAE formulation)
        if self.pre_encoder_bias:
            x_centered = x_flat - self.b_dec
        else:
            x_centered = x_flat
        
        # Linear transformation: (N, 128) @ (128, n_latents) -> (N, n_latents)
        z_pre = F.linear(x_centered, self.W_enc, self.b_enc)
        
        # Apply ReLU before TopK (ensures non-negative activations)
        z_pre = F.relu(z_pre)
        
        # TopK sparsity
        z = self.activation(z_pre)
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse latent back to input space.
        
        Formula: x_hat = z @ W_dec.T + b_dec
        
        Args:
            z: Sparse latent tensor of shape (*, n_latents)
            
        Returns:
            x_hat: Reconstructed tensor of shape (*, 128)
        """
        # F.linear computes: z @ W_dec.T + b_dec
        # W_dec shape: (input_dim, n_latents) = (128, 2048)
        # z shape: (N, n_latents) = (N, 2048)
        # Output: (N, 128)
        x_hat = F.linear(z, self.W_dec, self.b_dec)
        return x_hat
    
    def decode_matryoshka(self, z: torch.Tensor, n_latents: int) -> torch.Tensor:
        """
        Decode using only the first n_latents dimensions (Matryoshka).
        
        Args:
            z: Sparse latent tensor of shape (*, n_latents_full)
            n_latents: Number of latent dimensions to use
            
        Returns:
            x_hat: Reconstructed tensor using first n_latents
        """
        # Use only first n_latents
        z_truncated = z[..., :n_latents]
        W_dec_truncated = self.W_dec[:, :n_latents]
        
        # F.linear computes: z_truncated @ W_dec_truncated.T + b_dec
        x_hat = F.linear(z_truncated, W_dec_truncated, self.b_dec)
        return x_hat
    
    def forward(
        self, 
        x: torch.Tensor,
        return_all_reconstructions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]]:
        """
        Full forward pass: encode -> decode with optional Matryoshka outputs.
        
        Args:
            x: Input tensor of shape (*, 128) or (B, L, L, 128)
            return_all_reconstructions: If True, return reconstructions at all Matryoshka dims
            
        Returns:
            x_hat: Reconstructed tensor (full n_latents), same shape as input
            z: Sparse latent tensor
            matryoshka_recons (optional): Dict mapping dim -> reconstruction
        """
        # Store original shape for unflattening
        original_shape = x.shape
        
        # Flatten pairs
        x_flat, _ = self._flatten_pairs(x)
        
        # Encode to sparse latents
        z = self.encode(x_flat)
        
        # Full reconstruction
        x_hat_flat = self.decode(z)
        
        # Unflatten to original shape
        x_hat = self._unflatten_pairs(x_hat_flat, original_shape)
        
        if return_all_reconstructions:
            matryoshka_recons = {}
            for dim in self.matryoshka_dims:
                recon_flat = self.decode_matryoshka(z, dim)
                matryoshka_recons[dim] = self._unflatten_pairs(recon_flat, original_shape)
            return x_hat, z, matryoshka_recons
        
        return x_hat, z
    
    def forward_with_info(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with detailed information for analysis.
        
        Returns dict with:
            - x_hat: Full reconstruction
            - z: Sparse latent activations
            - active_indices: Indices of active latents
            - active_values: Values of active latents
            - matryoshka_recons: Reconstructions at each Matryoshka dim
            - matryoshka_losses: MSE loss at each Matryoshka dim
        """
        original_shape = x.shape
        x_flat, _ = self._flatten_pairs(x)
        
        # Pre-encoder processing
        if self.pre_encoder_bias:
            x_centered = x_flat - self.b_dec
        else:
            x_centered = x_flat
        
        # Get pre-activation
        z_pre = F.linear(x_centered, self.W_enc, self.b_enc)
        z_pre = F.relu(z_pre)
        
        # TopK with indices
        z, active_indices, active_values = self.activation.forward_with_indices(z_pre)
        
        # Full reconstruction
        x_hat_flat = self.decode(z)
        x_hat = self._unflatten_pairs(x_hat_flat, original_shape)
        
        # Matryoshka reconstructions and losses
        matryoshka_recons = {}
        matryoshka_losses = {}
        for dim in self.matryoshka_dims:
            recon_flat = self.decode_matryoshka(z, dim)
            matryoshka_recons[dim] = self._unflatten_pairs(recon_flat, original_shape)
            matryoshka_losses[dim] = F.mse_loss(recon_flat, x_flat).item()
        
        return {
            'x_hat': x_hat,
            'z': z,
            'active_indices': active_indices,
            'active_values': active_values,
            'matryoshka_recons': matryoshka_recons,
            'matryoshka_losses': matryoshka_losses,
            'sparsity': (z != 0).float().mean().item()
        }
    
    def get_active_latents(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the active latent indices and values for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            indices: Active latent indices of shape (N_pairs, k)
            values: Active latent values of shape (N_pairs, k)
        """
        x_flat, _ = self._flatten_pairs(x)
        
        if self.pre_encoder_bias:
            x_centered = x_flat - self.b_dec
        else:
            x_centered = x_flat
        
        z_pre = F.linear(x_centered, self.W_enc, self.b_enc)
        z_pre = F.relu(z_pre)
        
        _, indices, values = self.activation.forward_with_indices(z_pre)
        return indices, values
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 4. TRAINING UTILITIES
# =============================================================================


class _SimpleAdam:
    """
    Minimal Adam optimizer implemented with plain tensor ops.
    Used to avoid torch.optim.Adam, which on some systems triggers
    torch._dynamo -> sympy -> mpmath and causes ImportError.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.beta1, self.beta2 = betas[0], betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.param_groups = [{'params': self.params, 'lr': lr}]
        self.state = []
        for p in self.params:
            self.state.append({
                'm': torch.zeros_like(p.data, device=p.device),
                'v': torch.zeros_like(p.data, device=p.device),
                't': 0,
            })

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        lr = self.param_groups[0]['lr']
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0:
                grad = grad.add(p, alpha=self.weight_decay)
            st = self.state[i]
            st['t'] += 1
            st['m'].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            st['v'].mul_(self.beta2).add_(grad.pow(2), alpha=1 - self.beta2)
            m_hat = st['m'] / (1 - self.beta1 ** st['t'])
            v_hat = st['v'] / (1 - self.beta2 ** st['t'])
            p.sub_(m_hat / (v_hat.sqrt() + self.eps), alpha=lr)

    def state_dict(self):
        return {
            'state': [
                {'m': st['m'].clone(), 'v': st['v'].clone(), 't': st['t']}
                for st in self.state
            ],
            'param_groups': [{'lr': self.param_groups[0]['lr']}],
        }

    def load_state_dict(self, state_dict):
        for i, st in enumerate(state_dict['state']):
            if i < len(self.state):
                self.state[i]['m'].copy_(st['m'].to(self.state[i]['m'].device))
                self.state[i]['v'].copy_(st['v'].to(self.state[i]['v'].device))
                self.state[i]['t'] = st['t']
        if state_dict.get('param_groups') and len(state_dict['param_groups']) > 0:
            self.param_groups[0]['lr'] = state_dict['param_groups'][0]['lr']


class MatryoshkaSAETrainer:
    """
    Trainer for Matryoshka SAE with proper loss computation and decoder normalization.
    
    Loss Function:
    -------------
    Matryoshka loss = mean of MSE losses at each nested dimension:
        L = (1/M) * sum_{d in matryoshka_dims} MSE(x, decode_d(z))
    
    This encourages the first dimensions to be most informative.
    """
    
    def __init__(
        self,
        model: MatryoshkaSAE,
        config: Optional[MatryoshkaSAEConfig] = None,
        normalizer: Optional[object] = None,  # PairChannelNormalizer
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        matryoshka_weights: Optional[List[float]] = None,
    ):
        """
        Args:
            model: MatryoshkaSAE instance
            config: Optional config (uses model.config if not provided)
            normalizer: Optional PairChannelNormalizer for input normalization
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: L2 regularization
            matryoshka_weights: Optional weights for each Matryoshka dim loss
                                (default: uniform weights)
        """
        self.model = model
        self.config = config or model.config
        self.normalizer = normalizer
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        if self.normalizer is not None:
            self.normalizer.to(self.device)
        
        # Matryoshka weights (uniform by default)
        if matryoshka_weights is None:
            n_dims = len(self.config.matryoshka_dims)
            self.matryoshka_weights = [1.0 / n_dims] * n_dims
        else:
            assert len(matryoshka_weights) == len(self.config.matryoshka_dims)
            # Normalize weights
            total = sum(matryoshka_weights)
            self.matryoshka_weights = [w / total for w in matryoshka_weights]
        
        # Prefer torch.optim.Adam (needed for LR schedulers); fall back to
        # _SimpleAdam only if torch.optim triggers dynamo/sympy import errors.
        try:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
            )
        except Exception:
            self.optimizer = _SimpleAdam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
            )
        
        # Training history
        self.history = {
            'total_loss': [],
            'matryoshka_losses': {dim: [] for dim in self.config.matryoshka_dims},
            'sparsity': [],
            'epoch': [],
        }
        
        print(f"MatryoshkaSAE Trainer on {self.device}")
        print(f"  Input dim: {self.config.input_dim}")
        print(f"  Latents: {self.config.n_latents}")
        print(f"  TopK: {self.config.topk}")
        print(f"  Matryoshka dims: {self.config.matryoshka_dims}")
        print(f"  Parameters: {self.model.count_parameters():,}")
    
    def compute_matryoshka_loss(
        self, 
        x: torch.Tensor, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        Compute Matryoshka loss: weighted average of MSE at each nested dimension.
        
        Args:
            x: Original input (flattened to N, 128)
            z: Sparse latent (N, n_latents)
            
        Returns:
            total_loss: Weighted sum of MSE losses
            per_dim_losses: Dict mapping dim -> loss value
        """
        # Flatten x if needed
        x_flat, _ = self.model._flatten_pairs(x)
        
        total_loss = 0.0
        per_dim_losses = {}
        
        for dim, weight in zip(self.config.matryoshka_dims, self.matryoshka_weights):
            # Reconstruct using only first 'dim' latents
            x_hat = self.model.decode_matryoshka(z, dim)
            
            # MSE loss
            mse = F.mse_loss(x_hat, x_flat)
            per_dim_losses[dim] = mse.item()
            
            # Weighted contribution
            total_loss = total_loss + weight * mse
        
        return total_loss, per_dim_losses
    
    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Single training step on a batch of pair representations.
        
        This handles the critical reshaping: [B, L, L, 128] -> [-1, 128]
        
        Args:
            x: Input tensor of shape (B, L, L, 128) or (L, L, 128) or (N, 128)
            
        Returns:
            Dict with loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        x = x.to(self.device)
        
        # Apply normalization if available
        if self.normalizer is not None:
            # Normalizer expects (B, 128, L, L) format
            if x.ndim == 4:
                x = self.normalizer.transform(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            elif x.ndim == 3:
                x = x.unsqueeze(0)
                x = self.normalizer.transform(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = x.squeeze(0)
        
        # Forward pass (handles flattening internally)
        x_hat, z = self.model(x)
        
        # Compute Matryoshka loss
        total_loss, per_dim_losses = self.compute_matryoshka_loss(x, z)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Normalize decoder weights after each step (CRITICAL for SAE)
        if self.model.normalize_decoder:
            self.model._normalize_decoder_weights()
        
        # Compute sparsity
        sparsity = (z != 0).float().mean().item()
        
        return {
            'total_loss': total_loss.item(),
            'per_dim_losses': per_dim_losses,
            'sparsity': sparsity,
        }
    
    def train_epoch(
        self, 
        pair_tensors: List[torch.Tensor],
        shuffle: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch over a list of pair representation tensors.
        
        Args:
            pair_tensors: List of tensors, each of shape (L_i, L_i, 128)
            shuffle: Whether to shuffle the order of proteins
            
        Returns:
            Average metrics for the epoch
        """
        self.model.train()
        
        # Shuffle if requested
        indices = list(range(len(pair_tensors)))
        if shuffle:
            np.random.shuffle(indices)
        
        epoch_loss = 0.0
        epoch_dim_losses = {dim: 0.0 for dim in self.config.matryoshka_dims}
        epoch_sparsity = 0.0
        total_pairs = 0
        
        for idx in indices:
            x = pair_tensors[idx]
            n_pairs = x.shape[0] * x.shape[1] if x.ndim >= 3 else x.shape[0]
            
            metrics = self.train_step(x)
            
            # Accumulate weighted by number of pairs
            epoch_loss += metrics['total_loss'] * n_pairs
            for dim in self.config.matryoshka_dims:
                epoch_dim_losses[dim] += metrics['per_dim_losses'][dim] * n_pairs
            epoch_sparsity += metrics['sparsity'] * n_pairs
            total_pairs += n_pairs
        
        # Average
        avg_metrics = {
            'total_loss': epoch_loss / total_pairs,
            'per_dim_losses': {dim: loss / total_pairs for dim, loss in epoch_dim_losses.items()},
            'sparsity': epoch_sparsity / total_pairs,
        }
        
        return avg_metrics
    
    def train(
        self,
        pair_tensors: List[torch.Tensor],
        num_epochs: int = 100,
        print_every: int = 10,
        lr_schedule: bool = True,
        early_stopping_patience: int = 50,
        checkpoint_epochs: Optional[List[int]] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict:
        """
        Full training loop with optional mid-training checkpoints.
        
        Args:
            pair_tensors: List of pair representation tensors
            num_epochs: Number of epochs
            print_every: Print frequency
            lr_schedule: Use cosine annealing LR
            early_stopping_patience: Early stopping patience
            checkpoint_epochs: Save model at these epochs (e.g. [50, 100])
            checkpoint_dir: Directory for checkpoints (defaults to cwd)
            
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Training MatryoshkaSAE for {num_epochs} epochs")
        print(f"  {len(pair_tensors)} proteins")
        print(f"  Total pairs: {sum(t.shape[0] * t.shape[1] for t in pair_tensors):,}")
        if checkpoint_epochs:
            print(f"  Checkpoints at epochs: {checkpoint_epochs}")
        print(f"{'='*60}\n")
        
        if checkpoint_dir is None:
            checkpoint_dir = "."
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        scheduler = None
        if lr_schedule:
            try:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=num_epochs, eta_min=1e-6
                )
            except TypeError:
                print("  Warning: LR scheduler not compatible with optimizer, using constant LR")
        
        best_loss = float('inf')
        best_epoch = 0
        patience = 0
        
        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(pair_tensors)
            
            # Record history
            self.history['total_loss'].append(metrics['total_loss'])
            for dim in self.config.matryoshka_dims:
                self.history['matryoshka_losses'][dim].append(metrics['per_dim_losses'][dim])
            self.history['sparsity'].append(metrics['sparsity'])
            self.history['epoch'].append(epoch)
            
            if scheduler is not None:
                scheduler.step()
            
            # Early stopping
            if metrics['total_loss'] < best_loss:
                best_loss = metrics['total_loss']
                best_epoch = epoch
                patience = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
            
            # Print progress
            if epoch % print_every == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]['lr']
                dim_str = " | ".join([f"d{d}:{metrics['per_dim_losses'][d]:.5f}" 
                                     for d in self.config.matryoshka_dims])
                print(f"Epoch {epoch:4d}/{num_epochs} | Loss: {metrics['total_loss']:.5f} | "
                      f"{dim_str} | Sparsity: {metrics['sparsity']:.3f} | LR: {lr:.2e}")
            
            # Mid-training checkpoint
            if checkpoint_epochs and epoch in checkpoint_epochs:
                ckpt_path = os.path.join(checkpoint_dir, f"matryoshka_sae_epoch{epoch}.pt")
                self.save(ckpt_path)
                print(f"  ** Checkpoint saved: {ckpt_path}")
            
            if patience >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
            print(f"\nRestored best model from epoch {best_epoch} (loss: {best_loss:.6f})")
        
        return self.history
    
    @torch.no_grad()
    def evaluate(self, pair_tensors: List[torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate the model on a list of pair tensors.
        
        Returns per-protein and aggregate metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_dim_losses = {dim: 0.0 for dim in self.config.matryoshka_dims}
        total_sparsity = 0.0
        total_pairs = 0
        per_protein_losses = {}
        
        for i, x in enumerate(pair_tensors):
            x = x.to(self.device)
            n_pairs = x.shape[0] * x.shape[1] if x.ndim >= 3 else x.shape[0]
            
            # Apply normalization
            if self.normalizer is not None:
                if x.ndim == 3:
                    x = x.unsqueeze(0)
                    x = self.normalizer.transform(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                    x = x.squeeze(0)
            
            x_hat, z = self.model(x)
            loss, per_dim_losses = self.compute_matryoshka_loss(x, z)
            sparsity = (z != 0).float().mean().item()
            
            per_protein_losses[i] = loss.item()
            total_loss += loss.item() * n_pairs
            for dim in self.config.matryoshka_dims:
                total_dim_losses[dim] += per_dim_losses[dim] * n_pairs
            total_sparsity += sparsity * n_pairs
            total_pairs += n_pairs
        
        return {
            'total_loss': total_loss / total_pairs,
            'per_dim_losses': {dim: loss / total_pairs for dim, loss in total_dim_losses.items()},
            'sparsity': total_sparsity / total_pairs,
            'per_protein_losses': per_protein_losses,
        }
    
    @torch.no_grad()
    def reconstruct(
        self, 
        x: torch.Tensor,
        n_latents: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct pair representation, optionally using fewer latents.
        
        Args:
            x: Input tensor of shape (L, L, 128) or (B, L, L, 128)
            n_latents: Number of latent dimensions to use (None = full)
            
        Returns:
            x_hat: Reconstruction in original shape
            z: Sparse latent activations
        """
        self.model.eval()
        original_shape = x.shape
        x = x.to(self.device)
        
        # Apply normalization
        if self.normalizer is not None:
            if x.ndim == 3:
                x = x.unsqueeze(0)
                x = self.normalizer.transform(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = x.squeeze(0)
            elif x.ndim == 4:
                x = self.normalizer.transform(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # Encode
        z = self.model.encode(x)
        
        # Decode (full or truncated)
        x_flat, _ = self.model._flatten_pairs(x)
        if n_latents is None:
            x_hat_flat = self.model.decode(z)
        else:
            x_hat_flat = self.model.decode_matryoshka(z, n_latents)
        
        # Unflatten
        x_hat = self.model._unflatten_pairs(x_hat_flat, original_shape)
        
        # Inverse normalize
        if self.normalizer is not None:
            if x_hat.ndim == 3:
                x_hat = x_hat.unsqueeze(0)
                x_hat = self.normalizer.inverse_transform(x_hat.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x_hat = x_hat.squeeze(0)
            elif x_hat.ndim == 4:
                x_hat = self.normalizer.inverse_transform(x_hat.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        return x_hat.cpu(), z.cpu()
    
    def save(self, path: str):
        """Save model and training state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, normalizer: Optional[object] = None, device: str = "auto"):
        """Load model from checkpoint."""
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = MatryoshkaSAE(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        trainer = cls(model, config, normalizer, device)
        trainer.history = checkpoint['history']
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded model from {path}")
        return trainer


# =============================================================================
# 5. INFERENCE UTILITIES
# =============================================================================

class PairSAEInference:
    """
    Inference utilities for MatryoshkaSAE, including shape restoration.
    
    Key Methods:
    -----------
    - reconstruct_for_structure_module: Returns [B, L, L, 128] for AlphaFold
    - get_feature_activations: Returns which latents are active for each pair
    - interpret_latent: Get the decoder direction for a specific latent
    """
    
    def __init__(
        self, 
        model: MatryoshkaSAE,
        normalizer: Optional[object] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.normalizer = normalizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        if normalizer is not None:
            self.normalizer.to(self.device)
    
    @torch.no_grad()
    def reconstruct_for_structure_module(
        self,
        pair_rep: torch.Tensor,
        n_latents: Optional[int] = None,
        symmetrize: bool = True
    ) -> torch.Tensor:
        """
        Reconstruct pair representation for feeding back to AlphaFold's Structure Module.
        
        Args:
            pair_rep: Original pair representation of shape (B, L, L, 128) or (L, L, 128)
            n_latents: Number of Matryoshka latents to use (None = all)
            symmetrize: Enforce symmetry (pair_ij = pair_ji)
            
        Returns:
            Reconstructed tensor in original shape, ready for Structure Module
        """
        was_3d = pair_rep.ndim == 3
        if was_3d:
            pair_rep = pair_rep.unsqueeze(0)
        
        original_shape = pair_rep.shape
        pair_rep = pair_rep.to(self.device)
        
        # Normalize
        if self.normalizer is not None:
            pair_rep = self.normalizer.transform(pair_rep.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # Encode
        z = self.model.encode(pair_rep)
        
        # Decode
        if n_latents is None:
            recon_flat = self.model.decode(z)
        else:
            recon_flat = self.model.decode_matryoshka(z, n_latents)
        
        # Unflatten
        recon = self.model._unflatten_pairs(recon_flat, original_shape)
        
        # Denormalize
        if self.normalizer is not None:
            recon = self.normalizer.inverse_transform(recon.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # Symmetrize
        if symmetrize:
            recon = (recon + recon.transpose(-2, -3)) / 2
        
        if was_3d:
            recon = recon.squeeze(0)
        
        return recon.cpu()
    
    @torch.no_grad()
    def get_structure_module_output(
        self,
        pair_rep: torch.Tensor,
        n_latents: Optional[int] = None,
        symmetrize: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        protein_id: Optional[str] = None,
    ) -> Dict[str, Union[torch.Tensor, Tuple[int, ...], str, None]]:
        """
        Get reconstruction ready for AlphaFold Structure Module, with optional save.
        
        Use this when you need to feed the reconstructed pair representation into
        a structure module and then compute TM-score to verify structure quality.
        
        Args:
            pair_rep: Original pair representation (L, L, 128) or (B, L, L, 128)
            n_latents: Matryoshka latents to use (None = all)
            symmetrize: Enforce symmetry
            save_path: If set, save tensor as .npy (e.g. for loading in structure pipeline)
            protein_id: Optional label for metadata
            
        Returns:
            Dict with:
                - "pair_rep": tensor (L, L, 128) or (B, L, L, 128), ready for structure module
                - "shape": tuple shape
                - "n_latents": int or None
                - "protein_id": str or None
                - "save_path": path string if saved, else None
        """
        recon = self.reconstruct_for_structure_module(
            pair_rep, n_latents=n_latents, symmetrize=symmetrize
        )
        out = {
            "pair_rep": recon,
            "shape": tuple(recon.shape),
            "n_latents": n_latents,
            "protein_id": protein_id,
            "save_path": None,
        }
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(save_path), recon.numpy())
            out["save_path"] = str(save_path.resolve())
        return out
    
    @torch.no_grad()
    def get_feature_activations(
        self, 
        pair_rep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get active features for each pair position.
        
        Args:
            pair_rep: Pair representation of shape (L, L, 128)
            
        Returns:
            indices: Active feature indices, shape (L, L, k)
            values: Active feature values, shape (L, L, k)
        """
        pair_rep = pair_rep.to(self.device)
        if pair_rep.ndim == 3:
            pair_rep = pair_rep.unsqueeze(0)
        
        L = pair_rep.shape[1]
        
        # Normalize
        if self.normalizer is not None:
            pair_rep = self.normalizer.transform(pair_rep.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # Get active latents
        indices, values = self.model.get_active_latents(pair_rep)
        
        # Reshape to (L, L, k)
        indices = indices.reshape(L, L, self.model.topk)
        values = values.reshape(L, L, self.model.topk)
        
        return indices.cpu(), values.cpu()
    
    @torch.no_grad()
    def get_latent_decoder_direction(self, latent_idx: int) -> torch.Tensor:
        """
        Get the decoder direction for a specific latent feature.
        
        This is the "dictionary direction" that the latent represents.
        
        Args:
            latent_idx: Index of the latent feature
            
        Returns:
            Decoder direction of shape (128,)
        """
        return self.model.W_dec[:, latent_idx].cpu()
    
    @torch.no_grad()
    def get_all_decoder_directions(self) -> torch.Tensor:
        """
        Get all decoder directions (dictionary).
        
        Returns:
            Decoder matrix of shape (128, n_latents)
        """
        return self.model.W_dec.cpu()
    
    @torch.no_grad()
    def ablate_latent(
        self,
        pair_rep: torch.Tensor,
        latent_idx: int,
        symmetrize: bool = True
    ) -> torch.Tensor:
        """
        Reconstruct with a specific latent ablated (set to zero).
        
        Useful for understanding what a latent feature contributes.
        
        Args:
            pair_rep: Original pair representation
            latent_idx: Which latent to ablate
            symmetrize: Enforce output symmetry
            
        Returns:
            Reconstruction with latent ablated
        """
        was_3d = pair_rep.ndim == 3
        if was_3d:
            pair_rep = pair_rep.unsqueeze(0)
        
        original_shape = pair_rep.shape
        pair_rep = pair_rep.to(self.device)
        
        # Normalize
        if self.normalizer is not None:
            pair_rep = self.normalizer.transform(pair_rep.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # Encode
        z = self.model.encode(pair_rep)
        
        # Ablate
        z[..., latent_idx] = 0.0
        
        # Decode
        recon_flat = self.model.decode(z)
        recon = self.model._unflatten_pairs(recon_flat, original_shape)
        
        # Denormalize
        if self.normalizer is not None:
            recon = self.normalizer.inverse_transform(recon.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        if symmetrize:
            recon = (recon + recon.transpose(-2, -3)) / 2
        
        if was_3d:
            recon = recon.squeeze(0)
        
        return recon.cpu()


# =============================================================================
# 6. DATA LOADING UTILITIES
# =============================================================================

def load_pair_representations(
    data_dir: str,
    file_extension: str = "npy",
    return_file_paths: bool = False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[Path]]]:
    """
    Load pair representation tensors from a directory.
    
    Supports both flat layout (data_dir/*.npy) and nested layout
    (data_dir/<protein_id>/*_pair_block_47.npy).
    
    Args:
        data_dir: Directory containing .npy or .pt files (flat or nested)
        file_extension: File extension to look for
        return_file_paths: If True, also return the list of file paths
        
    Returns:
        List of tensors, each of shape (L, L, 128)
        (optionally) List of Path objects for each loaded file
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} not found")
    
    # Try flat layout first, then recursive for nested subdirectories
    file_paths = sorted(data_dir.glob(f"*.{file_extension}"))
    if not file_paths:
        file_paths = sorted(data_dir.glob(f"**/*.{file_extension}"))
    if not file_paths:
        raise ValueError(f"No .{file_extension} files found in {data_dir} (checked flat and nested layouts)")
    
    tensors = []
    print(f"Loading {len(file_paths)} pair representation files...")
    
    for fp in file_paths:
        if fp.suffix == ".npy":
            arr = np.load(fp)
        else:
            arr = torch.load(fp).numpy()
        
        # Normalize shape to (L, L, 128)
        if arr.shape[-1] == 128:
            tensor = torch.from_numpy(arr).float()
        elif arr.shape[0] == 128:
            tensor = torch.from_numpy(arr).float().permute(1, 2, 0)
        else:
            raise ValueError(f"Unexpected shape {arr.shape} for {fp.name}")
        
        tensors.append(tensor)
        print(f"  Loaded {fp.name}: shape {tensor.shape}")
    
    print(f"Loaded {len(tensors)} proteins, total pairs: {sum(t.shape[0]**2 for t in tensors):,}")
    if return_file_paths:
        return tensors, file_paths
    return tensors


# =============================================================================
# 6.5 TM-SCORE UTILITIES (TMalign)
# =============================================================================

_tmalign_warned: set = set()  # avoid spamming when TMalign not found


def compute_tm_score(
    pdb_ref: str,
    pdb_pred: str,
    tmalign_bin: str = "TMalign",
) -> Optional[float]:
    """
    Run TMalign and parse TM-score (pred vs ref).
    
    Use this to compare predicted structure (e.g. from structure module fed with
    reconstructed pair rep) against ground truth. Check that TM-score from
    reconstructed pair rep is close to TM-score from original pair rep.
    
    Args:
        pdb_ref: Path to reference PDB (e.g. ground truth)
        pdb_pred: Path to predicted PDB (e.g. structure module output)
        tmalign_bin: Path to TMalign binary (or "TMalign" if in PATH)
        
    Returns:
        TM-score in [0, 1] or None if TMalign failed
    """
    # Use absolute path for TMalign so subprocess finds it even when cwd is set to a PDB directory
    tmalign_bin = os.path.abspath(os.path.expanduser(tmalign_bin))
    pdb_ref = os.path.abspath(pdb_ref)
    pdb_pred = os.path.abspath(pdb_pred)
    if not os.path.isfile(pdb_ref):
        print(f"[WARN] Reference PDB not found: {pdb_ref}")
        return None
    if not os.path.isfile(pdb_pred):
        print(f"[WARN] Predicted PDB not found: {pdb_pred}")
        return None

    try:
        result = subprocess.run(
            [tmalign_bin, pdb_pred, pdb_ref],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(pdb_pred) or ".",
        )
    except FileNotFoundError:
        if tmalign_bin not in _tmalign_warned:
            _tmalign_warned.add(tmalign_bin)
            print(f"[WARN] TMalign binary not found: {tmalign_bin!r}. Install TMalign and add to PATH, or set config 'tmalign_bin' to the full path.")
        return None
    except OSError as e:
        print(f"[WARN] Could not run TMalign: {e}")
        return None

    if result.returncode != 0:
        print(f"[WARN] TMalign failed on {pdb_pred} vs {pdb_ref}")
        if result.stderr:
            print(result.stderr.strip())
        return None

    for line in result.stdout.splitlines():
        if "TM-score=" in line:
            parts = line.split()
            try:
                return float(parts[1])
            except (IndexError, ValueError):
                continue

    print(f"[WARN] Could not parse TM-score for {pdb_pred}")
    return None


def compare_tm_scores(
    gt_pdb: str,
    pdb_from_original: str,
    pdb_from_recon: str,
    tmalign_bin: str = "TMalign",
) -> Dict[str, Optional[float]]:
    """
    Compare TM-scores: structure from original pair rep vs structure from
    reconstructed pair rep, both against ground truth.
    
    If the Matryoshka reconstruction preserves information well, the two
    TM-scores should be similar (recon TM-score close to original TM-score).
    
    Args:
        gt_pdb: Ground truth structure PDB
        pdb_from_original: PDB from structure module run on *original* pair rep
        pdb_from_recon: PDB from structure module run on *reconstructed* pair rep
        tmalign_bin: Path to TMalign binary
        
    Returns:
        Dict with keys:
            - "tm_original": TM-score(gt, pdb_from_original)
            - "tm_recon": TM-score(gt, pdb_from_recon)
            - "delta": tm_original - tm_recon (positive = recon slightly worse)
    """
    tm_orig = compute_tm_score(gt_pdb, pdb_from_original, tmalign_bin)
    tm_recon = compute_tm_score(gt_pdb, pdb_from_recon, tmalign_bin)
    delta = None
    if tm_orig is not None and tm_recon is not None:
        delta = tm_orig - tm_recon
    return {
        "tm_original": tm_orig,
        "tm_recon": tm_recon,
        "delta": delta,
    }


# Default list of 10 proteins for batch TM-score
TM_SCORE_PROTEIN_IDS = [
    "7b3a", "7dkk", "7dq9", "7ebt", "7f6e",
    "7kdx", "7mro", "7pbk", "7tbs", "7tkv",
]

# Chain per protein (must match pair rep filenames: 7ebt_B_pair_block_47.npy -> B, etc.)
TM_SCORE_CHAIN_BY_PROTEIN = {
    "7b3a": "A", "7dkk": "A", "7dq9": "A",
    "7ebt": "B", "7f6e": "B", "7kdx": "B",
    "7mro": "A", "7pbk": "A", "7tbs": "A", "7tkv": "A",
}


def _pdb_path(
    base_dir: str,
    pid: str,
    ch: str,
    suffix: Optional[str] = None,
) -> str:
    """Build PDB path: base_dir/{pid}_{ch}[suffix].pdb"""
    if suffix:
        name = f"{pid}_{ch}{suffix}.pdb"
    else:
        name = f"{pid}_{ch}.pdb"
    return os.path.join(base_dir, name)


def compute_tm_scores_batch(
    gt_dir: str,
    original_pdb_dir: str,
    recon_pdb_dir: str,
    protein_ids: Optional[List[str]] = None,
    chain_by_protein: Optional[Dict[str, str]] = None,
    chain: str = "A",
    tmalign_bin: str = "TMalign",
    pred_pdb_suffix: Optional[str] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute TM-scores for multiple proteins: for each protein, compare
    structure-from-original and structure-from-recon against chain-filtered ground truth.
    
    Expects:
        gt_dir: directory with correct-chain GT PDBs (e.g. truth_pdbs/ with 7b3a_A.pdb, 7ebt_B.pdb, ...)
        original_pdb_dir: PDBs from structure module (original pair rep)
        recon_pdb_dir: PDBs from structure module (reconstructed pair rep)
    
    If pred_pdb_suffix is set (e.g. "_pair_block_47_structure"), predicted PDB filenames are
    {pid}_{chain}{suffix}.pdb (e.g. 7b3a_A_pair_block_47_structure.pdb). Otherwise {pid}_{chain}.pdb.
    Ground truth is always {pid}_{chain}.pdb.
    
    If chain_by_protein is set (e.g. TM_SCORE_CHAIN_BY_PROTEIN), filenames are {pid}_{chain}.pdb per protein.
    Otherwise uses single chain for all.
    
    Returns:
        Dict mapping protein_id -> {"tm_original", "tm_recon", "delta"}
    """
    if protein_ids is None:
        protein_ids = TM_SCORE_PROTEIN_IDS
    chain_map = chain_by_protein if chain_by_protein is not None else {p: chain for p in protein_ids}
    
    results = {}
    for pid in protein_ids:
        ch = chain_map.get(pid, "A")
        gt_pdb = _pdb_path(gt_dir, pid, ch, suffix=None)
        pdb_orig = _pdb_path(original_pdb_dir, pid, ch, suffix=pred_pdb_suffix)
        pdb_recon = _pdb_path(recon_pdb_dir, pid, ch, suffix=pred_pdb_suffix)
        results[pid] = compare_tm_scores(gt_pdb, pdb_orig, pdb_recon, tmalign_bin=tmalign_bin)
    return results


def compute_tm_scores_single_pred(
    gt_dir: str,
    pred_pdb_dir: str,
    protein_ids: Optional[List[str]] = None,
    chain_by_protein: Optional[Dict[str, str]] = None,
    chain: str = "A",
    tmalign_bin: str = "TMalign",
    pred_pdb_suffix: Optional[str] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute TM-scores when you have a single set of predicted PDBs (e.g. output_structures/).
    Returns TM(ground_truth, predicted) per protein.
    
    Use this when you only have one structure output dir (e.g. from original or from recon),
    to compare predictions vs ground truth.
    
    Args:
        gt_dir: directory with ground truth PDBs (e.g. truth_pdbs/)
        pred_pdb_dir: directory with predicted PDBs (e.g. output_structures/)
        pred_pdb_suffix: if set, predicted filenames are {pid}_{chain}{suffix}.pdb
    
    Returns:
        Dict mapping protein_id -> {"tm_score", "tm_original", "tm_recon", "delta"}
        (tm_original and tm_recon both set to the same value; delta = None)
    """
    if protein_ids is None:
        protein_ids = TM_SCORE_PROTEIN_IDS
    chain_map = chain_by_protein if chain_by_protein is not None else {p: chain for p in protein_ids}
    
    results = {}
    for pid in protein_ids:
        ch = chain_map.get(pid, "A")
        gt_pdb = _pdb_path(gt_dir, pid, ch, suffix=None)
        pdb_pred = _pdb_path(pred_pdb_dir, pid, ch, suffix=pred_pdb_suffix)
        tm = compute_tm_score(gt_pdb, pdb_pred, tmalign_bin)
        results[pid] = {
            "tm_original": tm,
            "tm_recon": tm,
            "delta": None,
        }
    return results


def print_tm_scores_table(results: Dict[str, Dict[str, Optional[float]]]) -> None:
    """Print a table of TM-scores for batch results (e.g. from compute_tm_scores_batch)."""
    print(f"{'Protein':<10} {'TM(orig)':>10} {'TM(recon)':>10} {'Delta':>8}")
    print("-" * 42)
    for pid, r in results.items():
        to = r["tm_original"]
        tr = r["tm_recon"]
        d = r["delta"]
        to_s = f"{to:.4f}" if to is not None else "  N/A"
        tr_s = f"{tr:.4f}" if tr is not None else "  N/A"
        d_s = f"{d:+.4f}" if d is not None else "  N/A"
        print(f"{pid:<10} {to_s:>10} {tr_s:>10} {d_s:>8}")
    print("-" * 42)
    valid = sum(1 for r in results.values() if r["delta"] is not None)
    print(f"Computed {valid}/{len(results)} TM-score comparisons.")


def print_tm_scores_single_table(results: Dict[str, Dict[str, Optional[float]]]) -> None:
    """Print a table of TM-scores when only one set of predictions is available (pred vs GT)."""
    print(f"{'Protein':<10} {'TM(score)':>10}")
    print("-" * 24)
    for pid, r in results.items():
        tm = r["tm_original"]
        tm_s = f"{tm:.4f}" if tm is not None else "  N/A"
        print(f"{pid:<10} {tm_s:>10}")
    print("-" * 24)
    valid = sum(1 for r in results.values() if r["tm_original"] is not None)
    print(f"Computed {valid}/{len(results)} TM-scores (pred vs ground truth).")


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def main(
    data_dir: str = "Proteins_layer47",
    output_dir: str = "output_matryoshka_sae",
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    n_latents: int = 4096,
    topk: int = 64,
    matryoshka_dims: Tuple[int, ...] = (128, 1024, 2048, 4096),
    use_normalizer: bool = True,
    train_split: float = 0.5,
    seed: int = 42,
):
    """
    Train MatryoshkaSAE on pair representations with a train/test split.
    
    The dataset is split 50/50 (configurable via train_split) into train and test
    sets. The model is trained only on the train set and evaluated on both sets
    separately so we can measure generalisation to unseen proteins.
    
    Reconstructed pair representations are saved to Arisa_reconstructed/ with
    names derived from input files (e.g. 7b3a_A_pair_block_47.npy -> 7b3a_A_pair_block_47_reconstructed.npy).
    
    Args:
        data_dir: Directory with .npy pair representations
        output_dir: Output directory for model and plots
        num_epochs: Training epochs
        learning_rate: Learning rate
        n_latents: Total latent dimensions
        topk: Number of active latents (sparsity budget)
        matryoshka_dims: Nested dimension checkpoints
        use_normalizer: Whether to use channel normalization
        train_split: Fraction of proteins used for training (default 0.5 = 50/50)
        seed: Random seed for reproducible splits
    """
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load data (keep sorted file list for output naming)
    data_path = Path(data_dir)
    npy_files = sorted(data_path.glob("*.npy"))
    if not npy_files:
        npy_files = sorted(data_path.glob("**/*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found in {data_dir} (checked flat and nested layouts)")
    print("\n" + "="*60)
    print("STEP 1: Load Pair Representations")
    print("="*60)
    pair_tensors = load_pair_representations(data_dir)
    
    # 1b. Train/test split (50/50 by default)
    print("\n" + "="*60)
    print("STEP 1b: Train/Test Split")
    print("="*60)
    n_total = len(pair_tensors)
    n_train = int(n_total * train_split)
    
    rng = np.random.RandomState(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    
    train_indices = sorted(indices[:n_train])
    test_indices = sorted(indices[n_train:])
    
    train_tensors = [pair_tensors[i] for i in train_indices]
    test_tensors = [pair_tensors[i] for i in test_indices]
    train_files = [npy_files[i] for i in train_indices]
    test_files = [npy_files[i] for i in test_indices]
    
    train_pairs = sum(t.shape[0] * t.shape[1] for t in train_tensors)
    test_pairs = sum(t.shape[0] * t.shape[1] for t in test_tensors)
    
    print(f"  Total proteins: {n_total}")
    print(f"  Train: {len(train_tensors)} proteins ({train_pairs:,} pairs)")
    print(f"  Test:  {len(test_tensors)} proteins ({test_pairs:,} pairs)")
    print(f"  Split ratio: {train_split:.0%} train / {1-train_split:.0%} test")
    print(f"  Random seed: {seed}")
    
    # Save split info for reproducibility
    split_info_path = os.path.join(output_dir, f"train_test_split_{timestamp}.txt")
    with open(split_info_path, "w") as f:
        f.write(f"seed={seed}\n")
        f.write(f"train_split={train_split}\n")
        f.write(f"n_total={n_total}\n")
        f.write(f"n_train={len(train_tensors)}\n")
        f.write(f"n_test={len(test_tensors)}\n\n")
        f.write("TRAIN FILES:\n")
        for fp in train_files:
            f.write(f"  {fp.name}\n")
        f.write("\nTEST FILES:\n")
        for fp in test_files:
            f.write(f"  {fp.name}\n")
    print(f"  Split info saved to: {split_info_path}")
    
    # 2. Optional: Fit normalizer (on TRAIN set only to avoid data leakage)
    normalizer = None
    if use_normalizer:
        print("\n" + "="*60)
        print("STEP 2: Fit Channel Normalizer (train set only)")
        print("="*60)
        try:
            from af2_autoencoder import PairRepresentationDataset, PairChannelNormalizer
            dataset = PairRepresentationDataset(data_dir)
            normalizer = PairChannelNormalizer()
            normalizer.fit(dataset)
        except ImportError:
            print("Warning: Could not import normalizer, training without normalization")
    
    # 3. Create model
    print("\n" + "="*60)
    print("STEP 3: Create MatryoshkaSAE")
    print("="*60)
    config = MatryoshkaSAEConfig(
        input_dim=128,
        n_latents=n_latents,
        matryoshka_dims=matryoshka_dims,
        topk=topk,
        learning_rate=learning_rate,
    )
    model = MatryoshkaSAE(config)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # 4. Train (on TRAIN set only)
    print("\n" + "="*60)
    print("STEP 4: Train MatryoshkaSAE (train set only)")
    print("="*60)
    trainer = MatryoshkaSAETrainer(
        model=model,
        config=config,
        normalizer=normalizer,
        learning_rate=learning_rate,
    )
    history = trainer.train(
        pair_tensors=train_tensors,
        num_epochs=num_epochs,
        print_every=10,
        checkpoint_epochs=[50, num_epochs],
        checkpoint_dir=output_dir,
    )
    
    # 5. Evaluate on TRAIN set
    print("\n" + "="*60)
    print("STEP 5a: Evaluate on TRAIN set")
    print("="*60)
    train_metrics = trainer.evaluate(train_tensors)
    print(f"Train loss: {train_metrics['total_loss']:.6f}")
    print(f"Train sparsity: {train_metrics['sparsity']:.4f}")
    print("Train per-dimension losses:")
    for dim, loss in train_metrics['per_dim_losses'].items():
        print(f"  {dim} latents: {loss:.6f}")
    
    # 5b. Evaluate on TEST set
    print("\n" + "="*60)
    print("STEP 5b: Evaluate on TEST set")
    print("="*60)
    test_metrics = trainer.evaluate(test_tensors)
    print(f"Test loss: {test_metrics['total_loss']:.6f}")
    print(f"Test sparsity: {test_metrics['sparsity']:.4f}")
    print("Test per-dimension losses:")
    for dim, loss in test_metrics['per_dim_losses'].items():
        print(f"  {dim} latents: {loss:.6f}")
    
    # 5c. Summary comparison
    print("\n" + "-"*40)
    print("Train vs Test Comparison:")
    print(f"  {'Metric':<25} {'Train':>10} {'Test':>10} {'Ratio':>10}")
    print(f"  {'Total loss':<25} {train_metrics['total_loss']:>10.6f} {test_metrics['total_loss']:>10.6f} {test_metrics['total_loss']/max(train_metrics['total_loss'],1e-10):>10.2f}x")
    for dim in config.matryoshka_dims:
        tr = train_metrics['per_dim_losses'][dim]
        te = test_metrics['per_dim_losses'][dim]
        print(f"  {f'd={dim} loss':<25} {tr:>10.6f} {te:>10.6f} {te/max(tr,1e-10):>10.2f}x")
    print("-"*40)
    
    # 6. Save
    print("\n" + "="*60)
    print("STEP 6: Save Model")
    print("="*60)
    model_path = os.path.join(output_dir, f"matryoshka_sae_{timestamp}.pt")
    trainer.save(model_path)
    
    if normalizer is not None:
        normalizer_path = os.path.join(output_dir, f"normalizer_{timestamp}.pt")
        normalizer.save(normalizer_path)
    
    # 7. Save test set: reconstructed .npy + copy originals for comparison
    import shutil
    
    print("\n" + "="*60)
    print("STEP 7: Save Test Set — Reconstructed & Original .npy")
    print("="*60)
    inference = PairSAEInference(model, normalizer)
    
    recon_dir = Path("reconstructed_proteins_layer47")
    orig_dir = Path("original_proteins_layer47")
    recon_dir.mkdir(parents=True, exist_ok=True)
    orig_dir.mkdir(parents=True, exist_ok=True)
    
    for i, tensor in enumerate(test_tensors):
        stem = test_files[i].stem
        recon_t = inference.reconstruct_for_structure_module(tensor)
        np.save(str(recon_dir / f"{stem}.npy"), recon_t.numpy())
        shutil.copy2(str(test_files[i]), str(orig_dir / f"{stem}.npy"))
        print(f"  {stem}  →  reconstructed + original saved")
    
    print(f"\n  reconstructed_proteins_layer47/  ({len(test_tensors)} files)")
    print(f"  original_proteins_layer47/       ({len(test_tensors)} files)")
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print(f"  Train proteins: {len(train_tensors)}, Test proteins: {len(test_tensors)}")
    print(f"  Train loss: {train_metrics['total_loss']:.6f}")
    print(f"  Test loss:  {test_metrics['total_loss']:.6f}")
    print("="*60)
    
    return model, trainer, history, train_metrics, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Matryoshka SAE for AF2 Pair Representations")
    parser.add_argument("--data_dir", type=str, default="Proteins_layer47",
                        help="Directory with pair rep .npy files (e.g. Proteins_layer47 or protein_layer47)")
    parser.add_argument("--output_dir", type=str, default="output_matryoshka_sae")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_latents", type=int, default=4096, help="Total latent dimensions")
    parser.add_argument("--topk", type=int, default=64, help="Sparsity budget: active latents per pair")
    parser.add_argument("--no_normalizer", action="store_true")
    parser.add_argument("--train_split", type=float, default=0.5,
                        help="Fraction of proteins for training (default 0.5 = 50/50 split)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split reproducibility")
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        n_latents=args.n_latents,
        topk=args.topk,
        use_normalizer=not args.no_normalizer,
        train_split=args.train_split,
        seed=args.seed,
    )
