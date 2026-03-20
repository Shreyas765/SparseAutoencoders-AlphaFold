"""
Train the contextual token SAE (384-d context in, 128-d pair reconstruction; packed L² tokens).

Usage:
  python train_token_sae.py --protein_dir /path/to/CompleteProteins [options]

DataLoader uses pack_context_collate; recon loss = MSE(recon, packed_targets) only;
entropy penalty on p_softmax as before.

GPU smoke test:
  python train_token_sae.py --protein_dir ... --smoke_batches 3 [--smoke_batch_size 4]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Import from same directory (_current_saemodel)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sparse_autoencoder import (
    ContextualPairDataset,
    ContextualTokenSAE,
    pack_context_collate,
)


def discover_pair_paths(protein_dir: Path, layer: int = 47) -> List[str]:
    """Return list of paths to *_pair_block_{layer}.npy under protein_dir."""
    paths: List[str] = []
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_entropy: float,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_entropy = 0.0
    num_batches = 0

    for packed_context, packed_targets, _original_shapes in dataloader:
        packed_context = packed_context.to(device)
        packed_targets = packed_targets.to(device)
        optimizer.zero_grad()

        recon_packed, p_softmax, _ = model(packed_context)

        loss_recon = nn.functional.mse_loss(recon_packed, packed_targets)
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
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    lambda_entropy: float,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_entropy = 0.0
    num_batches = 0

    for packed_context, packed_targets, _original_shapes in dataloader:
        packed_context = packed_context.to(device)
        packed_targets = packed_targets.to(device)
        recon_packed, p_softmax, _ = model(packed_context)

        loss_recon = nn.functional.mse_loss(recon_packed, packed_targets)
        entropy = -(p_softmax * (p_softmax + 1e-10).log()).sum(dim=-1).mean()
        loss_total = loss_recon + lambda_entropy * entropy

        total_loss += loss_total.item()
        total_recon += loss_recon.item()
        total_entropy += entropy.item()
        num_batches += 1

    n = max(num_batches, 1)
    return total_loss / n, total_recon / n, total_entropy / n


def run_smoke_test(
    train_loader: DataLoader,
    model: ContextualTokenSAE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_entropy: float,
    smoke_batches: int,
) -> int:
    """Run a few training steps and report peak CUDA memory, then exit."""
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    ran = 0
    for bi, (packed_context, packed_targets, _shapes) in enumerate(train_loader):
        if bi >= smoke_batches:
            break
        packed_context = packed_context.to(device)
        packed_targets = packed_targets.to(device)
        optimizer.zero_grad()
        recon_packed, p_softmax, _ = model(packed_context)
        loss_recon = nn.functional.mse_loss(recon_packed, packed_targets)
        entropy = -(p_softmax * (p_softmax + 1e-10).log()).sum(dim=-1).mean()
        (loss_recon + lambda_entropy * entropy).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ran += 1

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"Smoke test: ran {ran} batches. Peak CUDA memory: {peak_mb:.2f} MB")
    else:
        print(f"Smoke test: ran {ran} batches (CUDA not used).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train contextual token SAE (packed L², 384-d context → 128-d recon)"
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
    parser.add_argument("--d_latent", type=int, default=4096)
    parser.add_argument("--tau", type=float, default=0.90, help="CDF threshold for adaptive top-k")
    parser.add_argument("--lambda_entropy", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke_batches",
        type=int,
        default=0,
        help="If > 0, run only this many train batches, print peak GPU memory, then exit",
    )
    parser.add_argument(
        "--smoke_batch_size",
        type=int,
        default=None,
        help="Override --batch_size during smoke test only (default: use --batch_size)",
    )
    args = parser.parse_args()

    protein_dir = Path(args.protein_dir)
    paths = discover_pair_paths(protein_dir, args.layer)
    if not paths:
        print(f"No *_pair_block_{args.layer}.npy found under {protein_dir}")
        return 1

    torch.manual_seed(args.seed)
    full_dataset = ContextualPairDataset(paths, normalize=True)
    n_val = max(1, int(len(full_dataset) * args.val_frac))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    train_bs = args.batch_size
    if args.smoke_batches > 0 and args.smoke_batch_size is not None:
        train_bs = args.smoke_batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        collate_fn=pack_context_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pack_context_collate,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextualTokenSAE(
        d_context_in=384,
        d_latent=args.d_latent,
        d_recon_out=128,
        tau=args.tau,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.smoke_batches > 0:
        print(f"Smoke mode: batch_size={train_bs}, steps={args.smoke_batches}")
        return run_smoke_test(
            train_loader,
            model,
            optimizer,
            device,
            args.lambda_entropy,
            args.smoke_batches,
        )

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
