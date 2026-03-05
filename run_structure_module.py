#!/usr/bin/env python3
"""
Run OpenFold's structure module from saved pair (and optionally single) representations.

The structure module expects:
  - pair:   [N_res, N_res, 128]  (pair representation from Evoformer block 47)
  - single: [N_res, 384]         (single representation from Evoformer output)
  - aatype: [N_res]              (amino acid type indices 0-20)
  - mask:   [N_res]              (sequence mask)

Your pair_block_47.npy files provide the pair representation. You also need:
  1. single_block_47.npy (or single representation) - from the same pipeline that produced the pair files
  2. A sequence (FASTA) or aatype array - to get residue types

If you only have pair representations, you can try --single-from-pair to approximate
single from pair (experimental; results may be poor).
"""

import argparse
import os
import numpy as np
import torch

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.np import protein, residue_constants
from openfold.utils.import_weights import import_jax_weights_, import_openfold_weights_


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pair_npy",
        help="Path to pair_block_47.npy file (shape: [N_res, N_res, 128])",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write output PDB files",
    )
    parser.add_argument(
        "--single-npy",
        help="Path to single_block_47.npy or single representation (shape: [N_res, 384]). Required unless --single-from-pair.",
    )
    parser.add_argument(
        "--sequence",
        help="Amino acid sequence (one-letter codes) for aatype. Alternative: --fasta or --aatype-npy",
    )
    parser.add_argument(
        "--fasta",
        help="Path to FASTA file (uses first sequence). Use if chain differs from pair file.",
    )
    parser.add_argument(
        "--aatype-npy",
        help="Path to .npy file containing aatype indices [N_res]",
    )
    parser.add_argument(
        "--single-from-pair",
        action="store_true",
        help="(Experimental) Approximate single rep from pair via mean pooling. Use only if you lack single representations.",
    )
    parser.add_argument(
        "--openfold-checkpoint-path",
        default=None,
        help="Path to OpenFold checkpoint (default: downloads model_1_ptm)",
    )
    parser.add_argument(
        "--jax-param-path",
        default=None,
        help="Path to JAX/AlphaFold params (alternative to OpenFold checkpoint)",
    )
    parser.add_argument(
        "--model-device",
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device to run the structure module",
    )
    parser.add_argument(
        "--config-name",
        default="model_1_ptm",
        help="Model config preset (model_1_ptm or model_1_multimer_v3)",
    )
    return parser.parse_args()


def sequence_to_aatype(seq: str) -> np.ndarray:
    """Convert one-letter sequence to aatype indices (0-20)."""
    return np.array(
        [
            residue_constants.restype_order.get(
                c, residue_constants.restype_num
            )
            for c in seq.upper()
        ],
        dtype=np.int64,
    )


def load_sequence_from_fasta(path: str) -> str:
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith(">")]
    return "".join(lines)


def main():
    args = parse_args()

    # Load pair representation
    pair = np.load(args.pair_npy, allow_pickle=True)
    if pair.ndim == 3:
        pass  # [N, N, C_z]
    elif pair.ndim == 4:
        pair = pair[0]  # [1, N, N, C_z] -> [N, N, C_z]
    else:
        raise ValueError(f"pair must be 3D or 4D, got shape {pair.shape}")

    n_res = pair.shape[0]
    c_z = pair.shape[2]
    if c_z != 128:
        # Autoencoder may output different dim; pad or truncate to 128
        if c_z < 128:
            pad = np.zeros((n_res, n_res, 128 - c_z), dtype=pair.dtype)
            pair = np.concatenate([pair, pad], axis=2)
        else:
            pair = pair[:, :, :128].copy()

    # Load or create single representation
    c_s = 384
    if args.single_npy:
        single = np.load(args.single_npy, allow_pickle=True)
        if single.ndim == 2:
            pass
        elif single.ndim == 3:
            single = single[0]
        else:
            raise ValueError(f"single must be 2D or 3D, got shape {single.shape}")
        if single.shape != (n_res, c_s):
            raise ValueError(
                f"single shape {single.shape} incompatible with pair (N={n_res}, C_s={c_s})"
            )
    elif args.single_from_pair:
        # Experimental: mean-pool pair to get a per-residue vector, then pad/truncate to c_s
        single = pair.mean(axis=1)  # [N_res, C_z]
        if single.shape[1] < c_s:
            pad = np.zeros((n_res, c_s - single.shape[1]), dtype=single.dtype)
            single = np.concatenate([single, pad], axis=1)
        else:
            single = single[:, :c_s]
    else:
        raise ValueError(
            "Provide --single-npy or use --single-from-pair (experimental)"
        )

    # Load aatype
    if args.sequence:
        aatype = sequence_to_aatype(args.sequence)
    elif args.fasta:
        seq = load_sequence_from_fasta(args.fasta)
        aatype = sequence_to_aatype(seq)
    elif args.aatype_npy:
        aatype = np.load(args.aatype_npy, allow_pickle=True).astype(np.int64)
    else:
        raise ValueError(
            "Provide --sequence, --fasta, or --aatype-npy for residue types"
        )

    if len(aatype) != n_res:
        raise ValueError(
            f"aatype length {len(aatype)} != pair N_res {n_res}. "
            "Sequence/aatype must match the pair representation length."
        )

    # Add batch dim
    pair_b = np.expand_dims(pair, 0)
    single_b = np.expand_dims(single, 0)
    aatype_b = np.expand_dims(aatype, 0)
    mask = np.ones((1, n_res), dtype=np.float32)

    evoformer_output = {
        "single": torch.from_numpy(single_b).float().to(args.model_device),
        "pair": torch.from_numpy(pair_b).float().to(args.model_device),
    }
    aatype_t = torch.from_numpy(aatype_b).long().to(args.model_device)
    mask_t = torch.from_numpy(mask).float().to(args.model_device)

    # Load model (structure module only)
    config = model_config(args.config_name)
    model = AlphaFold(config)
    if args.openfold_checkpoint_path:
        d = torch.load(args.openfold_checkpoint_path, map_location="cpu")
        if "ema" in d:
            d = d["ema"]["params"]
        elif "state_dict" in d:
            d = d["state_dict"]
        elif "module" in d:
            d = {k.replace("module.", ""): v for k, v in d["module"].items()}
        import_openfold_weights_(model, d)
    elif args.jax_param_path:
        import_jax_weights_(model, args.jax_param_path)
    else:
        # Try default checkpoint
        ckpt = "openfold/resources/openfold_params/finetuning_ptm_1.pt"
        if os.path.exists(ckpt):
            d = torch.load(ckpt, map_location="cpu")
            if "ema" in d:
                d = d["ema"]["params"]
            elif "state_dict" in d:
                d = d["state_dict"]
            import_openfold_weights_(model, d)
        else:
            raise FileNotFoundError(
                "No checkpoint specified. Use --openfold-checkpoint-path or "
                "--jax-param-path, or place finetuning_ptm_1.pt in openfold/resources/openfold_params/"
            )

    model = model.to(args.model_device)
    model.eval()

    # Use inplace_safe=False on CPU (CUDA in-place softmax kernel has no CPU impl)
    inplace_safe = args.model_device != "cpu"

    with torch.no_grad():
        sm_out = model.structure_module(
            evoformer_output,
            aatype_t,
            mask=mask_t,
            inplace_safe=inplace_safe,
        )

    # Convert atom14 positions to atom37 for PDB output
    atom14_pos = sm_out["positions"][-1]
    feats_for_convert = {
        "residx_atom37_to_atom14": torch.from_numpy(
            np.take(residue_constants.RESTYPE_ATOM37_TO_ATOM14, aatype, axis=0)
        ).long().to(args.model_device).unsqueeze(0),
        "atom37_atom_exists": torch.from_numpy(
            np.take(residue_constants.RESTYPE_ATOM37_MASK, aatype, axis=0)
        ).float().to(args.model_device).unsqueeze(0),
    }
    from openfold.utils.feats import atom14_to_atom37
    atom37_pos = atom14_to_atom37(atom14_pos, feats_for_convert)
    atom37_mask = feats_for_convert["atom37_atom_exists"]

    final_atom_positions = atom37_pos[0].cpu().numpy()
    final_atom_mask = atom37_mask[0].cpu().numpy()

    unprocessed_features = {
        "aatype": np.expand_dims(aatype, 0),
        "residue_index": np.expand_dims(np.arange(n_res), 0),
    }
    result = {
        "final_atom_positions": final_atom_positions,
        "final_atom_mask": final_atom_mask,
    }

    pdb_str = protein.to_pdb(
        protein.from_prediction(unprocessed_features, result)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.pair_npy))[0]
    out_path = os.path.join(args.output_dir, f"{basename}_structure.pdb")
    with open(out_path, "w") as f:
        f.write(pdb_str)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
