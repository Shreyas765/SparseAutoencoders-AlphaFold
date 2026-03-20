"""
Inference script for Token SAE.
Loads trained checkpoint, processes proteins, and saves reconstructed (L, L, 128) .npy files.
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sparse_autoencoder import (
    ContextualPairDataset,
    ContextualTokenSAE,
    pack_context_collate,
    unpack_reconstructions,
)


def discover_pair_paths(protein_dir: Path, layer: int = 47):
    paths = []
    if not protein_dir.is_dir():
        return paths
    for item in sorted(protein_dir.iterdir()):
        if item.is_dir():
            p = item / f"{item.name}_pair_block_{layer}.npy"
            if p.is_file():
                paths.append(str(p))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate .npy reconstructions from Token SAE")
    parser.add_argument("--protein_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to token_sae_best.pt")
    parser.add_argument("--output_dir", type=str, default="token_sae_reconstructions")
    parser.add_argument("--d_latent", type=int, default=4096)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--layer", type=int, default=47)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_pair_paths(Path(args.protein_dir), args.layer)
    if not paths:
        print(f"No *_pair_block_{args.layer}.npy found under {args.protein_dir}")
        return 1

    dataset = ContextualPairDataset(paths, normalize=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=pack_context_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextualTokenSAE(
        d_context_in=384,
        d_latent=args.d_latent,
        d_recon_out=128,
        tau=args.tau,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print(f"Starting inference on {len(paths)} proteins...")
    with torch.no_grad():
        for i, (packed_context, _packed_targets, original_shapes) in enumerate(loader):
            packed_context = packed_context.to(device)
            recon_packed, _, _ = model(packed_context)

            unpacked_list = unpack_reconstructions(recon_packed, original_shapes)

            original_path = Path(paths[i])
            protein_id = original_path.parent.name

            recon_tensor = unpacked_list[0].cpu().numpy()
            save_path = out_dir / f"{protein_id}_reconstructed_pair.npy"
            np.save(save_path, recon_tensor)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(paths)}...")

    print(f"All reconstructions saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
