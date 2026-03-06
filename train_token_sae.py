"""
Train the token-based Sparse Autoencoder (no spatial interpolation, packed L² tokens).

Usage:
  python train_token_sae.py --protein_dir /path/to/CompleteProteins [options]

DataLoader uses pack_proteins_collate; loss = MSE + lambda_entropy * entropy.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Import from project root
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sparse_autoencoder import (
    PairRepresentationDataset,
    TokenSparseAutoencoder,
    pack_proteins_collate,
)


def discover_pair_paths(protein_dir: Path, layer: int = 47):
    """Return list of paths to *_pair_block_{layer}.npy under protein_dir."""
    paths = []
    if not protein_dir.is_dir():
        return paths
    for item in sorted(protein_dir.iterdir()):
        if not item.is_dir():
            continue
        name = item.name
        p = item / f"{name}_pair_block_{layer}.npy"
        if p.is_file():
            paths.append(str(p))
    return paths


def train_epoch(model, dataloader, optimizer, device, lambda_entropy: float):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_entropy = 0.0
    num_batches = 0

    for packed_batch, original_shapes in dataloader:
        packed_batch = packed_batch.to(device)
        optimizer.zero_grad()

        recon_packed, p_softmax, _ = model(packed_batch)

        loss_recon = nn.functional.mse_loss(recon_packed, packed_batch)
        entropy = -(p_softmax * (p_softmax + 1e-10).log()).sum(dim=-1).mean()
        loss_total = loss_recon + lambda_entropy * entropy

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_total.item()
        total_recon += loss_recon.item()
        total_entropy += entropy.item()
        num_batches += 1

    n = max(num_batches, 1)
    return total_loss / n, total_recon / n, total_entropy / n


@torch.no_grad()
def evaluate(model, dataloader, device, lambda_entropy: float):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_entropy = 0.0
    num_batches = 0

    for packed_batch, _ in dataloader:
        packed_batch = packed_batch.to(device)
        recon_packed, p_softmax, _ = model(packed_batch)

        loss_recon = nn.functional.mse_loss(recon_packed, packed_batch)
        entropy = -(p_softmax * (p_softmax + 1e-10).log()).sum(dim=-1).mean()
        loss_total = loss_recon + lambda_entropy * entropy

        total_loss += loss_total.item()
        total_recon += loss_recon.item()
        total_entropy += entropy.item()
        num_batches += 1

    n = max(num_batches, 1)
    return total_loss / n, total_recon / n, total_entropy / n


def unpack_reconstructions(recon_packed, original_shapes):
    """Split packed (sum L_i^2, 128) back into list of (L_i, L_i, 128) tensors."""
    lengths = [s[0] * s[1] for s in original_shapes]
    chunks = torch.split(recon_packed, lengths, dim=0)
    return [chunks[i].view(original_shapes[i]) for i in range(len(original_shapes))]


def main():
    parser = argparse.ArgumentParser(
        description="Train token-based SAE (packed L² tokens, no interpolation)"
    )
    parser.add_argument(
        "--protein_dir",
        type=str,
        default=os.environ.get("BASE", "."),
        help="Directory containing protein subdirs with *_pair_block_47.npy",
    )
    parser.add_argument("--layer", type=int, default=47)
    parser.add_argument("--val_frac", type=float, default=0.2, help="Fraction for validation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16, help="Number of proteins per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_latent", type=int, default=3000)
    parser.add_argument("--tau", type=float, default=0.90, help="CDF threshold for adaptive top-k")
    parser.add_argument("--lambda_entropy", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    protein_dir = Path(args.protein_dir)
    paths = discover_pair_paths(protein_dir, args.layer)
    if not paths:
        print(f"No *_pair_block_{args.layer}.npy found under {protein_dir}")
        return 1

    torch.manual_seed(args.seed)
    full_dataset = PairRepresentationDataset(paths, normalize=True)
    n_val = max(1, int(len(full_dataset) * args.val_frac))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pack_proteins_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pack_proteins_collate,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenSparseAutoencoder(d_in=128, d_latent=args.d_latent, tau=args.tau).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir or str(protein_dir / "token_sae_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss, train_recon, train_entropy = train_epoch(
            model, train_loader, optimizer, device, args.lambda_entropy
        )
        val_loss, val_recon, val_entropy = evaluate(
            model, val_loader, device, args.lambda_entropy
        )

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train loss: {train_loss:.6f} (recon: {train_recon:.6f}, entropy: {train_entropy:.6f}) | "
            f"Val loss: {val_loss:.6f} (recon: {val_recon:.6f}, entropy: {val_entropy:.6f})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                output_dir / "token_sae_best.pt",
            )

    torch.save(model.state_dict(), output_dir / "token_sae_final.pt")
    print(f"Saved best and final checkpoints to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
