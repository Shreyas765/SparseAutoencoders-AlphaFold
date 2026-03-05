"""
Sparse Autoencoder (SAE) with Large Latent Space and Batch Top-K Sparsity
- Fully connected feed-forward only (no convolutions, no pooling)
- Latent space is 10x the input size
- Batch top-k sparsity instead of ReLU
- Supports channel-wise processing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import json
import time
from typing import List, Optional, Tuple
from PIL import Image


# ============================================================================
# Batch Top-K Sparsity Activation
# ============================================================================

class BatchTopK(nn.Module):
    """
    Batch Top-K sparsity: Keep only top k activations across entire batch.
    This enforces sparsity by zeroing out all but the top k values.
    
    Args:
        k: Number of top activations to keep (can be absolute or fraction)
        k_fraction: If True, k is interpreted as fraction of total elements
    """
    def __init__(self, k: int, k_fraction: bool = False):
        super().__init__()
        self.k = k
        self.k_fraction = k_fraction
    
    def forward(self, x):
        """
        Args:
            x: (batch, features) - flattened activations
        Returns:
            Sparse activations with only top-k values kept
        """
        batch_size, num_features = x.shape
        total_elements = batch_size * num_features
        
        # Determine actual k
        if self.k_fraction:
            actual_k = max(1, int(total_elements * self.k))
        else:
            actual_k = min(self.k, total_elements)
        
        # Flatten to find top-k across entire batch
        x_flat = x.view(-1)  # (batch * features,)
        
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(x_flat, actual_k, dim=0)
        
        # Create sparse tensor
        sparse_x = torch.zeros_like(x_flat)
        sparse_x[topk_indices] = topk_values
        
        # Reshape back
        return sparse_x.view(batch_size, num_features)


# ============================================================================
# Dataset with Lanczos Interpolation
# ============================================================================

class LanczosProteinDataset(Dataset):
    """
    Loads protein pair representations and resizes them to target_size using Lanczos interpolation.
    """
    def __init__(
        self,
        data_paths: List[str],
        channel_indices: Optional[List[int]] = None,
        target_size: int = 342,
    ):
        self.data_paths = data_paths
        self.channel_indices = channel_indices
        self.target_size = target_size
        self.protein_names = []
        self.original_shapes = []
        self.data = []
        
        for path in data_paths:
            arr = np.load(path)
            self.original_shapes.append(arr.shape)
            if channel_indices is not None:
                arr = arr[:, :, channel_indices]
            self.data.append(arr)
            # Extract protein name from path
            protein_name = os.path.basename(path).replace('_pair_block_47.npy', '')
            if not protein_name:
                protein_name = os.path.basename(os.path.dirname(path))
            self.protein_names.append(protein_name)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        arr = self.data[idx].copy()
        h, w, c = arr.shape
        
        # Normalize per sample to [-1, 1]
        max_val = np.abs(arr).max()
        if max_val > 1e-8:
            arr = arr / max_val
        
        # Interpolate each channel using Lanczos
        resized = np.zeros((self.target_size, self.target_size, c), dtype=np.float32)
        for ch in range(c):
            img = Image.fromarray(arr[:, :, ch].astype(np.float32), mode="F")
            img_resized = img.resize(
                (self.target_size, self.target_size), Image.Resampling.LANCZOS
            )
            resized[:, :, ch] = np.array(img_resized)
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(resized).permute(2, 0, 1)  # (C, H, W)
        return tensor, idx, max_val


# ============================================================================
# Sparse Autoencoder Model
# ============================================================================

class SparseAutoEncoder(nn.Module):
    """
    Sparse Autoencoder with large latent space (10x input size).
    Fully connected feed-forward only - no convolutions or pooling.
    
    Architecture:
    - Encoder: Input → Latent (10x size) with batch top-k
    - Decoder: Latent → Output with batch top-k
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        topk_k: int = 1000,
        topk_fraction: bool = False,
        use_batch_norm: bool = False,
    ):
        """
        Args:
            input_dim: Input dimension (flattened size, e.g., 128*342*342)
            latent_dim: Latent dimension (should be 10x input_dim)
            encoder_hidden_dims: Hidden dimensions for encoder (optional)
            decoder_hidden_dims: Hidden dimensions for decoder (optional)
            topk_k: Number of top activations to keep in batch top-k
            topk_fraction: If True, topk_k is fraction of total elements
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.topk_k = topk_k
        self.topk_fraction = topk_fraction
        
        # Default: no hidden layers (direct projection)
        if encoder_hidden_dims is None:
            encoder_hidden_dims = []
        if decoder_hidden_dims is None:
            decoder_hidden_dims = []
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(BatchTopK(topk_k, topk_fraction))
            prev_dim = hidden_dim
        
        # Final encoder layer to latent
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(BatchTopK(topk_k, topk_fraction))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(BatchTopK(topk_k, topk_fraction))
            prev_dim = hidden_dim
        
        # Final decoder layer to output
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Tanh())  # Output in [-1, 1]
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x, return_latent=False):
        """
        Args:
            x: (batch, input_dim) - flattened input
            return_latent: Whether to return latent representation
        Returns:
            reconstructed: (batch, input_dim)
            latent: (batch, latent_dim) if return_latent=True
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        if return_latent:
            return reconstructed, latent
        return reconstructed


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model, dataloader, criterion, optimizer, scheduler, device, topk_k, epoch
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_data, batch_indices, max_vals in dataloader:
        batch_data = batch_data.to(device)
        batch_size, channels, height, width = batch_data.shape
        
        # Flatten: (batch, C, H, W) -> (batch, C*H*W)
        batch_data_flat = batch_data.view(batch_size, -1)
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed_flat = model(batch_data_flat)
        
        # Compute loss
        loss = criterion(reconstructed_flat, batch_data_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data, batch_indices, max_vals in dataloader:
            batch_data = batch_data.to(device)
            batch_size, channels, height, width = batch_data.shape
            
            # Flatten
            batch_data_flat = batch_data.view(batch_size, -1)
            
            # Forward pass
            reconstructed_flat = model(batch_data_flat)
            
            # Compute loss
            loss = criterion(reconstructed_flat, batch_data_flat)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoder with Large Latent Space and Batch Top-K"
    )
    parser.add_argument(
        "--protein_dir",
        type=str,
        default=".",
        help="Directory containing protein .npy files",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=47,
        help="Layer number (block number)",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=342,
        help="Target size for interpolation",
    )
    parser.add_argument(
        "--channel_indices",
        type=str,
        default=None,
        help="Comma-separated channel indices (e.g., '0,1' or None for all 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--topk_k",
        type=int,
        default=1000,
        help="Number of top activations for batch top-k",
    )
    parser.add_argument(
        "--topk_fraction",
        action="store_true",
        help="Interpret topk_k as fraction of total elements",
    )
    parser.add_argument(
        "--encoder_hidden",
        type=str,
        default="",
        help="Comma-separated encoder hidden dimensions (e.g., '512,256')",
    )
    parser.add_argument(
        "--decoder_hidden",
        type=str,
        default="",
        help="Comma-separated decoder hidden dimensions (e.g., '256,512')",
    )
    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
        help="Use batch normalization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sae_results",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse channel indices
    if args.channel_indices:
        channel_indices = [int(x) for x in args.channel_indices.split(",")]
    else:
        channel_indices = None  # All 128 channels
    
    # Parse hidden dimensions
    encoder_hidden = (
        [int(x) for x in args.encoder_hidden.split(",")]
        if args.encoder_hidden
        else []
    )
    decoder_hidden = (
        [int(x) for x in args.decoder_hidden.split(",")]
        if args.decoder_hidden
        else []
    )
    
    # Find protein files
    if channel_indices is None:
        # Look for files directly in protein_dir
        protein_files = [
            os.path.join(args.protein_dir, f)
            for f in os.listdir(args.protein_dir)
            if f.endswith(f"_pair_block_{args.layer}.npy")
        ]
    else:
        # Look in subdirectories
        protein_dirs = [
            d
            for d in os.listdir(args.protein_dir)
            if os.path.isdir(os.path.join(args.protein_dir, d))
            and not d.startswith(".")
        ]
        protein_files = []
        for pdir in protein_dirs:
            pfile = os.path.join(
                args.protein_dir, pdir, f"{pdir}_pair_block_{args.layer}.npy"
            )
            if os.path.exists(pfile):
                protein_files.append(pfile)
    
    protein_files.sort()
    print(f"Found {len(protein_files)} protein files")
    
    if len(protein_files) == 0:
        raise ValueError(f"No protein files found in {args.protein_dir}")
    
    # Create dataset
    dataset = LanczosProteinDataset(
        protein_files,
        channel_indices=channel_indices,
        target_size=args.target_size,
    )
    
    # Get input dimension
    sample, _, _ = dataset[0]
    channels, height, width = sample.shape
    input_dim = channels * height * width
    print(f"Input shape: ({channels}, {height}, {width})")
    print(f"Input dimension (flattened): {input_dim:,}")
    
    # Latent dimension is 10x input
    latent_dim = input_dim * 10
    print(f"Latent dimension (10x input): {latent_dim:,}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    # Create model
    model = SparseAutoEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden,
        decoder_hidden_dims=decoder_hidden,
        topk_k=args.topk_k,
        topk_fraction=args.topk_fraction,
        use_batch_norm=args.use_batch_norm,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {total_params:,} parameters")
    print(f"Architecture:")
    print(f"  Input: {input_dim:,}")
    if encoder_hidden:
        print(f"  Encoder hidden: {' → '.join(map(str, encoder_hidden))}")
    print(f"  Latent: {latent_dim:,} (10x input)")
    if decoder_hidden:
        print(f"  Decoder hidden: {' → '.join(map(str, decoder_hidden))}")
    print(f"  Output: {input_dim:,}")
    print(f"  Batch Top-K: k={args.topk_k}, fraction={args.topk_fraction}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    
    print("\nStarting training...")
    print("=" * 80)
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, args.topk_k, epoch
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "model_best.pth"),
            )
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"Train: {train_loss:.6f}, Val: {val_loss:.6f} | "
                f"Time: {elapsed:.1f}s"
            )
    
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Save final model
    torch.save(
        model.state_dict(), os.path.join(args.output_dir, "model_final.pth")
    )
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker="o", markersize=3)
    plt.plot(val_losses, label="Val Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.savefig(
        os.path.join(args.output_dir, "losses.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    
    # Save training info
    info = {
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "input_shape": [channels, height, width],
        "encoder_hidden": encoder_hidden,
        "decoder_hidden": decoder_hidden,
        "topk_k": args.topk_k,
        "topk_fraction": args.topk_fraction,
        "total_params": total_params,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "channel_indices": channel_indices,
        "target_size": args.target_size,
        "num_proteins": len(protein_files),
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
