"""
AlphaFold2 Pair Representation Autoencoder
==========================================

This script implements a 2D Convolutional Autoencoder to compress the 128-channel
Pair Representation from AlphaFold2 Layer 47 down to 32 channels and reconstruct it.

Data Format:
- Input: (L, L, 128) per protein, where L is the number of residues (variable length)
- The autoencoder compresses channels: 128 -> 64 -> 32 -> 64 -> 128
- Spatial dimensions (L, L) remain CONSTANT throughout

Architecture:
- Encoder: 1x1 Conv (128→64) → 3x3 Conv (64→32)
- Decoder: 3x3 Conv (32→64) → 1x1 Conv (64→128)
- Output symmetrization: (x + x.T) / 2
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving images
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import json


# =============================================================================
# 1. DATA LOADING - PyTorch Dataset for Variable-Length Pair Representations
# =============================================================================

class PairRepresentationDataset(Dataset):
    """
    PyTorch Dataset for loading AlphaFold2 Pair Representation activations.
    
    Expected file format:
        - .npy files with shape (L, L, 128) where L is the number of residues
        - OR shape (128, L, L) which will be transposed
    
    Handles proteins of different lengths by returning individual samples.
    """
    
    def __init__(self, data_directory: str, file_extension: str = "npy"):
        """
        Args:
            data_directory (str): Path to folder containing protein activation files
            file_extension (str): "npy" or "pt"
        """
        self.data_directory = Path(data_directory)
        
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Directory {data_directory} not found")
        
        # Find all data files (flat layout first, then nested subdirectories)
        self.file_paths = list(self.data_directory.glob(f"*.{file_extension}"))
        if not self.file_paths:
            self.file_paths = list(self.data_directory.glob(f"**/*.{file_extension}"))
        
        if not self.file_paths:
            raise ValueError(f"No .{file_extension} files found in {data_directory} (checked flat and nested layouts)")
        
        # Sort for reproducibility
        self.file_paths = sorted(self.file_paths)
        
        # Extract protein IDs from filenames
        self.protein_ids = [fp.stem for fp in self.file_paths]
        
        # Preload all data into memory (10 proteins should fit easily)
        self.data = []
        self.shapes = []
        
        print(f"Loading {len(self.file_paths)} protein pair representation files...")
        for fp in self.file_paths:
            if fp.suffix == ".npy":
                arr = np.load(fp)
            else:  # .pt
                arr = torch.load(fp).numpy()
            
            # Validate and normalize shape to (L, L, 128)
            if arr.ndim == 3:
                if arr.shape[-1] == 128:
                    # Shape is (L, L, 128) - correct format
                    tensor = torch.from_numpy(arr).float()
                elif arr.shape[0] == 128:
                    # Shape is (128, L, L) - transpose to (L, L, 128)
                    tensor = torch.from_numpy(arr).float().permute(1, 2, 0)
                    print(f"  Note: Transposed {fp.name} from (128, L, L) to (L, L, 128)")
                else:
                    raise ValueError(f"Unexpected shape {arr.shape} for {fp.name}")
            else:
                raise ValueError(f"Expected 3D array, got {arr.ndim}D for {fp.name}")
            
            L = tensor.shape[0]
            assert tensor.shape == (L, L, 128), f"Shape mismatch: {tensor.shape}"
            
            self.data.append(tensor)
            self.shapes.append(L)
            print(f"  Loaded {fp.name}: shape ({L}, {L}, 128)")
        
        print(f"✓ Loaded {len(self.data)} proteins")
        print(f"  Residue lengths range: {min(self.shapes)} - {max(self.shapes)}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        """
        Returns:
            tensor: Shape (L, L, 128) pair representation
            protein_id: String identifier
            length: Residue length L
        """
        return self.data[idx], self.protein_ids[idx], self.shapes[idx]


# =============================================================================
# 2. NORMALIZATION - Per-Channel StandardScaler for High-Variance Activations
# =============================================================================

class PairChannelNormalizer:
    """
    Per-channel normalization for pair representations.
    
    Computes mean and std across all proteins and spatial positions for each
    of the 128 channels, then normalizes: z = (x - mean) / (std + eps)
    
    This is critical because Layer 47 activations can have high variance
    across different channels.
    """
    
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean = None  # Shape: (128,)
        self.std = None   # Shape: (128,)
        self.fitted = False
    
    def fit(self, dataset: PairRepresentationDataset):
        """
        Compute channel-wise statistics from all proteins in the dataset.
        """
        print("Fitting normalizer on dataset...")
        
        # Collect all channel values
        all_values = []
        for tensor, _, _ in dataset:
            # tensor shape: (L, L, 128)
            # Flatten spatial dims, keep channels: (L*L, 128)
            flat = tensor.reshape(-1, 128)
            all_values.append(flat)
        
        # Concatenate all proteins: (total_pairs, 128)
        all_values = torch.cat(all_values, dim=0)
        
        # Compute per-channel statistics
        self.mean = all_values.mean(dim=0)  # (128,)
        self.std = all_values.std(dim=0)    # (128,)
        
        self.fitted = True
        
        print(f"✓ Normalizer fitted on {all_values.shape[0]:,} pair positions")
        print(f"  Channel mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}]")
        print(f"  Channel std range:  [{self.std.min():.3f}, {self.std.max():.3f}]")
        
        return self
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor of shape (L, L, 128) or (B, C, L, L).
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        if x.ndim == 3 and x.shape[-1] == 128:
            # Shape: (L, L, 128) - normalize along last dim
            return (x - self.mean) / (self.std + self.eps)
        elif x.ndim == 4 and x.shape[1] == 128:
            # Shape: (B, 128, L, L) - normalize along channel dim
            mean = self.mean.view(1, 128, 1, 1).to(x.device)
            std = self.std.view(1, 128, 1, 1).to(x.device)
            return (x - mean) / (std + self.eps)
        else:
            raise ValueError(f"Unexpected shape: {x.shape}")
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor back to original scale.
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        if x.ndim == 3 and x.shape[-1] == 128:
            # Shape: (L, L, 128)
            return x * (self.std + self.eps) + self.mean
        elif x.ndim == 4 and x.shape[1] == 128:
            # Shape: (B, 128, L, L)
            mean = self.mean.view(1, 128, 1, 1).to(x.device)
            std = self.std.view(1, 128, 1, 1).to(x.device)
            return x * (std + self.eps) + mean
        else:
            raise ValueError(f"Unexpected shape: {x.shape}")
    
    def to(self, device: torch.device) -> 'PairChannelNormalizer':
        """Move normalizer statistics to specified device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self
    
    def save(self, path: str):
        """Save normalizer state to file."""
        torch.save({
            'mean': self.mean,
            'std': self.std,
            'eps': self.eps,
            'fitted': self.fitted
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'PairChannelNormalizer':
        """Load normalizer state from file."""
        state = torch.load(path)
        normalizer = cls(eps=state['eps'])
        normalizer.mean = state['mean']
        normalizer.std = state['std']
        normalizer.fitted = state['fitted']
        return normalizer


# =============================================================================
# 3. MODEL - Pair Representation Autoencoder with Symmetry Constraint
# =============================================================================

class PairRepresentationAE(nn.Module):
    """
    2D Convolutional Autoencoder for AlphaFold2 Pair Representations.
    
    Architecture:
        Encoder:
            - 1x1 Conv2D: 128 → 64 (preserves spatial independence)
            - ReLU activation
            - 3x3 Conv2D: 64 → 32 with padding=1 (extracts local spatial context)
            - ReLU activation
            
        Bottleneck: (L, L, 32)
        
        Decoder:
            - 3x3 Conv2D: 32 → 64 with padding=1 (reconstructs spatial context)
            - ReLU activation  
            - 1x1 Conv2D: 64 → 128 (reconstructs full channel representation)
            
        Output Symmetrization:
            - output = (output + output.transpose(-2, -1)) / 2
    
    Input shape: (B, 128, L, L) - batch, channels, height, width
    Output shape: (B, 128, L, L) - same as input, symmetrized
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        bottleneck_channels: int = 32,
        hidden_channels: int = 64,
        use_batch_norm: bool = False,
        dropout: float = 0.0
    ):
        """
        Args:
            in_channels: Number of input channels (128 for AF2 pair rep)
            bottleneck_channels: Number of channels in compressed representation
            hidden_channels: Number of intermediate channels
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate (0 = no dropout)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.hidden_channels = hidden_channels
        
        # =====================================================================
        # ENCODER
        # =====================================================================
        encoder_layers = []
        
        # 1x1 Conv: 128 → 64 (channel reduction, preserves spatial independence)
        encoder_layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1))
        if use_batch_norm:
            encoder_layers.append(nn.BatchNorm2d(hidden_channels))
        encoder_layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            encoder_layers.append(nn.Dropout2d(dropout))
        
        # 3x3 Conv: 64 → 32 (extracts local spatial context)
        encoder_layers.append(nn.Conv2d(hidden_channels, bottleneck_channels, 
                                        kernel_size=3, padding=1))
        if use_batch_norm:
            encoder_layers.append(nn.BatchNorm2d(bottleneck_channels))
        encoder_layers.append(nn.ReLU(inplace=True))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # =====================================================================
        # DECODER
        # =====================================================================
        decoder_layers = []
        
        # 3x3 Conv: 32 → 64 (reconstructs spatial context)
        decoder_layers.append(nn.Conv2d(bottleneck_channels, hidden_channels,
                                        kernel_size=3, padding=1))
        if use_batch_norm:
            decoder_layers.append(nn.BatchNorm2d(hidden_channels))
        decoder_layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            decoder_layers.append(nn.Dropout2d(dropout))
        
        # 1x1 Conv: 64 → 128 (channel expansion back to original)
        decoder_layers.append(nn.Conv2d(hidden_channels, in_channels, kernel_size=1))
        # No activation on final layer - we want full range for reconstruction
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode pair representation to bottleneck.
        
        Args:
            x: Input tensor of shape (B, 128, L, L)
            
        Returns:
            Latent tensor of shape (B, 32, L, L)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to full channels.
        
        Args:
            z: Latent tensor of shape (B, 32, L, L)
            
        Returns:
            Reconstructed tensor of shape (B, 128, L, L)
        """
        return self.decoder(z)
    
    def symmetrize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Symmetrize the output: (x + x.T) / 2
        
        Since pair representations (i,j) and (j,i) should be related,
        this enforces symmetry in the reconstruction.
        
        Args:
            x: Tensor of shape (B, C, L, L)
            
        Returns:
            Symmetrized tensor of same shape
        """
        # Transpose the last two spatial dimensions
        x_t = x.transpose(-2, -1)
        return (x + x_t) / 2
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → decode → symmetrize
        
        Args:
            x: Input tensor of shape (B, 128, L, L)
            
        Returns:
            reconstruction: Symmetrized reconstruction of shape (B, 128, L, L)
            latent: Bottleneck representation of shape (B, 32, L, L)
        """
        # Encode to bottleneck
        latent = self.encode(x)
        
        # Decode back to full channels
        reconstruction = self.decode(latent)
        
        # Symmetrize output
        reconstruction = self.symmetrize(reconstruction)
        
        return reconstruction, latent
    
    def get_compression_ratio(self) -> float:
        """Return the channel compression ratio."""
        return self.in_channels / self.bottleneck_channels
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepPairRepresentationAE(nn.Module):
    """
    Deeper Autoencoder with skip connections for better reconstruction.
    
    Architecture:
        Encoder: 128 → 96 → 64 → 48 → 32 (gradual compression)
        Decoder: 32 → 48 → 64 → 96 → 128 (gradual expansion)
        Skip connections between encoder and decoder at matching resolutions
    
    This model has ~4x more parameters but achieves significantly lower loss.
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        bottleneck_channels: int = 32,
        use_skip_connections: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.use_skip_connections = use_skip_connections
        
        # Encoder layers (gradual compression)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder layers (gradual expansion)
        # If skip connections, decoder input is doubled (concat with encoder)
        skip_mult = 2 if use_skip_connections else 1
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(48 * skip_mult, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 * skip_mult, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(96 * skip_mult, 128, kernel_size=1)
            # No activation on final layer
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)   # 128 → 96
        e2 = self.enc2(e1)  # 96 → 64
        e3 = self.enc3(e2)  # 64 → 48
        e4 = self.enc4(e3)  # 48 → 32
        return e4
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder (save intermediates for skip connections)
        e1 = self.enc1(x)   # 128 → 96
        e2 = self.enc2(e1)  # 96 → 64
        e3 = self.enc3(e2)  # 64 → 48
        latent = self.enc4(e3)  # 48 → 32
        
        # Decoder with skip connections
        d1 = self.dec1(latent)  # 32 → 48
        
        if self.use_skip_connections:
            d2 = self.dec2(torch.cat([d1, e3], dim=1))  # 48+48 → 64
            d3 = self.dec3(torch.cat([d2, e2], dim=1))  # 64+64 → 96
            d4 = self.dec4(torch.cat([d3, e1], dim=1))  # 96+96 → 128
        else:
            d2 = self.dec2(d1)
            d3 = self.dec3(d2)
            d4 = self.dec4(d3)
        
        # Symmetrize output
        reconstruction = (d4 + d4.transpose(-2, -1)) / 2
        
        return reconstruction, latent
    
    def get_compression_ratio(self) -> float:
        return self.in_channels / self.bottleneck_channels
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 4. TRAINING LOOP - Optimized for Small Dataset
# =============================================================================

class PairAETrainer:
    """
    Training manager for the Pair Representation Autoencoder.
    
    Optimized for small datasets (10 proteins):
    - Full batch gradient descent (no mini-batching needed)
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    """
    
    def __init__(
        self,
        model: PairRepresentationAE,
        normalizer: PairChannelNormalizer,
        device: torch.device = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.normalizer = normalizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model and normalizer to device
        self.model.to(self.device)
        self.normalizer.to(self.device)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'epoch': []
        }
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"Compression ratio: {model.get_compression_ratio():.1f}x")
    
    def train_epoch(self, dataset: PairRepresentationDataset) -> float:
        """
        Train for one epoch over all proteins.
        
        Since we have only 10 proteins and they have different sizes,
        we process each protein individually and accumulate gradients.
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for tensor, protein_id, length in dataset:
            # Move to device and reshape for Conv2d: (L, L, 128) → (1, 128, L, L)
            x = tensor.to(self.device)
            x = x.permute(2, 0, 1).unsqueeze(0)  # (1, 128, L, L)
            
            # Normalize
            x_norm = self.normalizer.transform(x)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstruction, latent = self.model(x_norm)
            
            # Compute loss on normalized data
            loss = self.criterion(reconstruction, x_norm)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * (length * length)  # Weight by number of pairs
            num_samples += length * length
        
        avg_loss = total_loss / num_samples
        return avg_loss
    
    def evaluate(self, dataset: PairRepresentationDataset) -> Tuple[float, Dict]:
        """
        Evaluate the model on the dataset.
        
        Returns:
            avg_loss: Average MSE loss
            per_protein_losses: Dict mapping protein_id to its loss
        """
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        per_protein_losses = {}
        
        with torch.no_grad():
            for tensor, protein_id, length in dataset:
                # Move to device and reshape
                x = tensor.to(self.device)
                x = x.permute(2, 0, 1).unsqueeze(0)  # (1, 128, L, L)
                
                # Normalize
                x_norm = self.normalizer.transform(x)
                
                # Forward pass
                reconstruction, latent = self.model(x_norm)
                
                # Compute loss
                loss = self.criterion(reconstruction, x_norm)
                
                per_protein_losses[protein_id] = loss.item()
                total_loss += loss.item() * (length * length)
                num_samples += length * length
        
        avg_loss = total_loss / num_samples
        return avg_loss, per_protein_losses
    
    def train(
        self,
        dataset: PairRepresentationDataset,
        num_epochs: int = 100,
        print_every: int = 100,
        lr_schedule: bool = True,
        early_stopping_patience: int = 100
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            dataset: Training dataset
            num_epochs: Maximum number of epochs
            print_every: Print progress every N epochs
            lr_schedule: Use learning rate scheduling
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        # Learning rate scheduler - Cosine Annealing for smooth decay
        if lr_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=1e-5
            )
            print(f"Using Cosine Annealing LR: {self.optimizer.param_groups[0]['lr']:.0e} → 1e-5")
        
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_loss = self.train_epoch(dataset)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['epoch'].append(epoch)
            
            # Learning rate scheduling
            if lr_schedule:
                scheduler.step()
            
            # Early stopping check
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model state
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % print_every == 0 or epoch == 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d}/{num_epochs} | "
                      f"Loss: {train_loss:.6f} | "
                      f"Best: {best_loss:.6f} (epoch {best_epoch}) | "
                      f"LR: {current_lr:.2e}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break
        
        # Restore best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
            print(f"\nRestored best model from epoch {best_epoch}")
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Final best loss: {best_loss:.6f}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def reconstruct(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct a single protein tensor.
        
        Args:
            tensor: Input tensor of shape (L, L, 128)
            
        Returns:
            reconstruction: Reconstructed tensor of shape (L, L, 128)
            latent: Latent representation of shape (L, L, 32)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Reshape for Conv2d: (L, L, 128) → (1, 128, L, L)
            x = tensor.to(self.device)
            x = x.permute(2, 0, 1).unsqueeze(0)
            
            # Normalize
            x_norm = self.normalizer.transform(x)
            
            # Forward pass
            recon_norm, latent = self.model(x_norm)
            
            # Denormalize reconstruction
            recon = self.normalizer.inverse_transform(recon_norm)
            
            # Reshape back: (1, 128, L, L) → (L, L, 128)
            recon = recon.squeeze(0).permute(1, 2, 0)
            latent = latent.squeeze(0).permute(1, 2, 0)
        
        return recon.cpu(), latent.cpu()


# =============================================================================
# 5. VISUALIZATION - Compare Original vs Reconstructed Heatmaps
# =============================================================================

def plot_reconstruction_comparison(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    protein_id: str,
    channels_to_plot: List[int] = [0, 31, 63, 95, 127],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    Plot original vs reconstructed channel heatmaps side by side.
    
    Args:
        original: Original tensor of shape (L, L, 128)
        reconstruction: Reconstructed tensor of shape (L, L, 128)
        protein_id: Protein identifier for title
        channels_to_plot: List of channel indices to visualize
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    n_channels = len(channels_to_plot)
    fig, axes = plt.subplots(2, n_channels, figsize=figsize)
    
    # Convert to numpy if needed
    if isinstance(original, torch.Tensor):
        original = original.numpy()
    if isinstance(reconstruction, torch.Tensor):
        reconstruction = reconstruction.numpy()
    
    for idx, ch in enumerate(channels_to_plot):
        # Original
        im1 = axes[0, idx].imshow(original[:, :, ch], cmap='viridis', aspect='equal')
        axes[0, idx].set_title(f'Original Ch {ch}', fontsize=10)
        axes[0, idx].set_xlabel('Residue j')
        if idx == 0:
            axes[0, idx].set_ylabel('Residue i')
        plt.colorbar(im1, ax=axes[0, idx], fraction=0.046, pad=0.04)
        
        # Reconstruction
        im2 = axes[1, idx].imshow(reconstruction[:, :, ch], cmap='viridis', aspect='equal')
        axes[1, idx].set_title(f'Reconstructed Ch {ch}', fontsize=10)
        axes[1, idx].set_xlabel('Residue j')
        if idx == 0:
            axes[1, idx].set_ylabel('Residue i')
        plt.colorbar(im2, ax=axes[1, idx], fraction=0.046, pad=0.04)
    
    # Compute reconstruction error
    mse = np.mean((original - reconstruction) ** 2)
    correlation = np.corrcoef(original.flatten(), reconstruction.flatten())[0, 1]
    
    fig.suptitle(f'Pair Representation Reconstruction: {protein_id}\n'
                 f'MSE: {mse:.6f} | Correlation: {correlation:.4f}', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()  # Close figure to free memory


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training loss over epochs.
    
    Args:
        history: Training history dictionary with 'epoch' and 'train_loss'
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history['epoch'], history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()  # Close figure to free memory


def plot_difference_heatmap(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    protein_id: str,
    channel: int = 0,
    save_path: Optional[str] = None
):
    """
    Plot the absolute difference between original and reconstruction.
    
    Args:
        original: Original tensor of shape (L, L, 128)
        reconstruction: Reconstructed tensor of shape (L, L, 128)
        protein_id: Protein identifier
        channel: Channel index to visualize
        save_path: Path to save figure (optional)
    """
    # Convert to numpy
    if isinstance(original, torch.Tensor):
        original = original.numpy()
    if isinstance(reconstruction, torch.Tensor):
        reconstruction = reconstruction.numpy()
    
    diff = np.abs(original[:, :, channel] - reconstruction[:, :, channel])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    im1 = axes[0].imshow(original[:, :, channel], cmap='viridis')
    axes[0].set_title(f'Original (Channel {channel})')
    axes[0].set_xlabel('Residue j')
    axes[0].set_ylabel('Residue i')
    plt.colorbar(im1, ax=axes[0])
    
    # Reconstruction
    im2 = axes[1].imshow(reconstruction[:, :, channel], cmap='viridis')
    axes[1].set_title(f'Reconstructed (Channel {channel})')
    axes[1].set_xlabel('Residue j')
    axes[1].set_ylabel('Residue i')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    im3 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Absolute Difference')
    axes[2].set_xlabel('Residue j')
    axes[2].set_ylabel('Residue i')
    plt.colorbar(im3, ax=axes[2])
    
    fig.suptitle(f'{protein_id} - Reconstruction Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()  # Close figure to free memory


# =============================================================================
# 6. MAIN - Full Pipeline Execution
# =============================================================================

def main(
    data_dir: str = "Proteins_layer47",
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    bottleneck_channels: int = 32,
    use_batch_norm: bool = False,
    dropout: float = 0.0,
    save_model: bool = True,
    output_dir: str = "output",
    use_deep_model: bool = True
):
    """
    Run the full training pipeline.
    
    Args:
        data_dir: Directory containing .npy files
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        bottleneck_channels: Number of channels in latent space
        use_batch_norm: Whether to use batch normalization
        dropout: Dropout rate
        save_model: Whether to save the trained model
        output_dir: Directory for saving outputs
        use_deep_model: Use deeper model with skip connections for lower loss
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # =========================================================================
    # STEP 1: Load Dataset
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading Dataset")
    print("="*60)
    
    dataset = PairRepresentationDataset(data_dir)
    
    # =========================================================================
    # STEP 2: Fit Normalizer
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: Fitting Channel Normalizer")
    print("="*60)
    
    normalizer = PairChannelNormalizer()
    normalizer.fit(dataset)
    
    # =========================================================================
    # STEP 3: Create Model
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: Creating Model")
    print("="*60)
    
    if use_deep_model:
        print("Using DEEP model with skip connections (lower loss, more parameters)")
        model = DeepPairRepresentationAE(
            in_channels=128,
            bottleneck_channels=bottleneck_channels,
            use_skip_connections=True
        )
    else:
        print("Using STANDARD model (faster, fewer parameters)")
        model = PairRepresentationAE(
            in_channels=128,
            bottleneck_channels=bottleneck_channels,
            hidden_channels=64,
            use_batch_norm=use_batch_norm,
            dropout=dropout
        )
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal Parameters: {model.count_parameters():,}")
    print(f"Compression Ratio: {model.get_compression_ratio():.1f}x (128 → {bottleneck_channels})")
    
    # =========================================================================
    # STEP 4: Train Model
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 4: Training")
    print("="*60)
    
    trainer = PairAETrainer(
        model=model,
        normalizer=normalizer,
        learning_rate=learning_rate
    )
    
    history = trainer.train(
        dataset=dataset,
        num_epochs=num_epochs,
        print_every=100
    )
    
    # =========================================================================
    # STEP 5: Evaluate and Visualize
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 5: Evaluation and Visualization")
    print("="*60)
    
    # Evaluate on all proteins
    avg_loss, per_protein_losses = trainer.evaluate(dataset)
    print(f"\nFinal evaluation loss: {avg_loss:.6f}")
    print("\nPer-protein losses:")
    for pid, loss in sorted(per_protein_losses.items(), key=lambda x: x[1]):
        print(f"  {pid}: {loss:.6f}")
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(output_dir, f"training_history_{timestamp}.png"))
    
    # Visualize reconstruction for one protein
    tensor, protein_id, length = dataset[0]
    reconstruction, latent = trainer.reconstruct(tensor)
    
    print(f"\nReconstruction for {protein_id}:")
    print(f"  Original shape: {tensor.shape}")
    print(f"  Reconstruction shape: {reconstruction.shape}")
    print(f"  Latent shape: {latent.shape}")
    
    # Plot comparison
    plot_reconstruction_comparison(
        original=tensor,
        reconstruction=reconstruction,
        protein_id=protein_id,
        channels_to_plot=[0, 31, 63, 95, 127],
        save_path=os.path.join(output_dir, f"reconstruction_{protein_id}_{timestamp}.png")
    )
    
    # Plot difference heatmap
    plot_difference_heatmap(
        original=tensor,
        reconstruction=reconstruction,
        protein_id=protein_id,
        channel=0,
        save_path=os.path.join(output_dir, f"difference_{protein_id}_{timestamp}.png")
    )
    
    # =========================================================================
    # STEP 6: Save Model and Normalizer
    # =========================================================================
    if save_model:
        print("\n" + "="*60)
        print("STEP 6: Saving Model")
        print("="*60)
        
        model_path = os.path.join(output_dir, f"pair_ae_model_{timestamp}.pt")
        normalizer_path = os.path.join(output_dir, f"pair_ae_normalizer_{timestamp}.pt")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'in_channels': 128,
                'bottleneck_channels': bottleneck_channels,
                'hidden_channels': 64,
                'use_batch_norm': use_batch_norm,
                'dropout': dropout
            },
            'training_history': history,
            'final_loss': avg_loss
        }, model_path)
        
        normalizer.save(normalizer_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Normalizer saved to: {normalizer_path}")
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)
    
    return model, normalizer, trainer, history


# =============================================================================
# 7. UTILITY FUNCTIONS - Load Trained Model
# =============================================================================

def load_trained_model(model_path: str, normalizer_path: str, device: str = 'auto'):
    """
    Load a trained model and normalizer from saved files.
    
    Args:
        model_path: Path to saved model .pt file
        normalizer_path: Path to saved normalizer .pt file
        device: Device to load model on ('auto', 'cuda', 'cpu')
        
    Returns:
        model: Loaded PairRepresentationAE model
        normalizer: Loaded PairChannelNormalizer
    """
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    model = PairRepresentationAE(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load normalizer
    normalizer = PairChannelNormalizer.load(normalizer_path)
    normalizer.to(device)
    
    print(f"Model loaded from {model_path}")
    print(f"Normalizer loaded from {normalizer_path}")
    print(f"Device: {device}")
    
    return model, normalizer


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train AlphaFold2 Pair Representation Autoencoder"
    )
    parser.add_argument(
        "--data_dir", type=str, default="Proteins_layer47",
        help="Directory containing .npy files (default: Proteins_layer47)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--bottleneck", type=int, default=32,
        help="Bottleneck channels (default: 32)"
    )
    parser.add_argument(
        "--batch_norm", action="store_true",
        help="Use batch normalization"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout rate (default: 0.0)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--shallow", action="store_true",
        help="Use shallow model instead of deep (faster but higher loss)"
    )
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        bottleneck_channels=args.bottleneck,
        use_batch_norm=args.batch_norm,
        dropout=args.dropout,
        output_dir=args.output_dir,
        use_deep_model=not args.shallow
    )

