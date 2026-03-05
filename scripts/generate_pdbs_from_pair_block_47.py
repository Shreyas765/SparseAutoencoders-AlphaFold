#!/usr/bin/env python3
"""
Generate PDB files from pair_block_47.npy for all proteins under the base directory.

Expects layout:
  {base}/{protein_id}/{protein_id}_pair_block_47.npy

Optionally uses (if present in same folder):
  - single_block_47.npy  (else uses --single-from-pair)
  - {protein_id}.fasta, seq.fasta, or any *.fasta  (else aatype_47.npy, else dummy alanine)

Usage:
  python scripts/generate_pdbs_from_pair_block_47.py
  python scripts/generate_pdbs_from_pair_block_47.py --base /path/to/CompleteProteins --output-dir pair_block_47_pdbs

Requires: run_structure_module.py in project root, OpenFold installed.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

LAYER = 47
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def discover_proteins(base: Path) -> List[Tuple[str, Path]]:
    """Return [(protein_id, pair_npy_path), ...] for all proteins with pair_block_47.npy."""
    pairs = []
    if not base.is_dir():
        return pairs
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        protein_id = item.name
        pair_npy = item / f"{protein_id}_pair_block_{LAYER}.npy"
        if pair_npy.is_file():
            pairs.append((protein_id, pair_npy))
    return pairs


def find_fasta(subdir: Path, protein_id: str) -> Optional[Path]:
    for fn in [f"{protein_id}.fasta", "seq.fasta"]:
        p = subdir / fn
        if p.is_file():
            return p
    for f in subdir.iterdir():
        if f.suffix == ".fasta":
            return f
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDBs from pair_block_47.npy for all proteins under base"
    )
    parser.add_argument(
        "--base",
        default=os.environ.get(
            "BASE",
            "/storage/scratch1/5/sshrestha304/Autoencoder/CompleteProteins",
        ),
        help="Base directory containing protein subdirs (e.g. 6tf4_A/)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for PDBs (default: {base}/pair_block_47_pdbs)",
    )
    parser.add_argument(
        "--run-structure-module",
        default=None,
        help="Path to run_structure_module.py (default: project_root/run_structure_module.py)",
    )
    parser.add_argument(
        "--model-device",
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device for structure module",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N proteins (for testing)",
    )
    args = parser.parse_args()

    base = Path(args.base).resolve()
    output_dir = Path(args.output_dir or str(base / "pair_block_47_pdbs")).resolve()
    run_script = Path(args.run_structure_module or str(PROJECT_ROOT / "run_structure_module.py"))

    if not run_script.is_file():
        print(f"Error: run_structure_module.py not found at {run_script}")
        return 1

    pairs = discover_proteins(base)
    if not pairs:
        print(f"No pair_block_{LAYER}.npy found under {base}")
        return 1

    total = len(pairs)
    if args.limit:
        pairs = pairs[: args.limit]
        print(f"Processing first {len(pairs)} of {total} proteins (--limit {args.limit})")
    else:
        print(f"Found {total} proteins with pair_block_{LAYER}.npy under {base}")
    print(f"Output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0
    for i, (protein_id, pair_npy) in enumerate(pairs, 1):
        subdir = pair_npy.parent
        single_npy = subdir / f"single_block_{LAYER}.npy"
        aatype_npy = subdir / f"aatype_{LAYER}.npy"
        fasta_path = find_fasta(subdir, protein_id)

        cmd = [
            sys.executable,
            str(run_script),
            str(pair_npy),
            str(output_dir),
            "--model-device",
            args.model_device,
        ]
        if single_npy.is_file():
            cmd.extend(["--single-npy", str(single_npy)])
        else:
            cmd.append("--single-from-pair")

        temp_aatype = None
        if fasta_path:
            cmd.extend(["--fasta", str(fasta_path)])
        elif aatype_npy.is_file():
            cmd.extend(["--aatype-npy", str(aatype_npy)])
        else:
            # Dummy aatype (all alanine) - need n_res from pair
            import numpy as np
            pair = np.load(pair_npy, allow_pickle=True)
            n_res = pair.shape[0] if pair.ndim >= 2 else pair.shape[1]
            fd, temp_aatype = tempfile.mkstemp(suffix=".npy")
            os.close(fd)
            np.save(temp_aatype, np.zeros(n_res, dtype=np.int64))
            cmd.extend(["--aatype-npy", temp_aatype])

        if args.dry_run:
            print(f"[{i}/{len(pairs)}] Would run: {' '.join(cmd)}")
            ok += 1
            if temp_aatype and os.path.exists(temp_aatype):
                os.unlink(temp_aatype)
            continue

        print(f"[{i}/{len(pairs)}] {protein_id}...", end=" ", flush=True)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                out_pdb = output_dir / f"{pair_npy.stem}_structure.pdb"
                if out_pdb.is_file():
                    print(f"✅ {out_pdb.name}")
                    ok += 1
                else:
                    print(f"⚠️ No PDB written")
                    fail += 1
            else:
                print(f"❌ {result.stderr[:200] if result.stderr else result.stdout[:200]}")
                fail += 1
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
            fail += 1
        except Exception as e:
            print(f"❌ {e}")
            fail += 1
        finally:
            if temp_aatype and os.path.exists(temp_aatype):
                try:
                    os.unlink(temp_aatype)
                except OSError:
                    pass

    print(f"\nDone: {ok} OK, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
