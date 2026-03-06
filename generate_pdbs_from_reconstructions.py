"""
Batch wrapper for the Structure Module.
Converts a directory of Token SAE reconstructed .npy files into .pdb files.
Uses run_structure_module.py (pair_npy, output_dir, --single-from-pair, --fasta or --aatype-npy).
"""
import os
import sys
import glob
import tempfile
import subprocess
import argparse
from pathlib import Path


def find_fasta(subdir: Path, protein_id: str) -> Path:
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
        description="Convert Token SAE reconstructed .npy files to PDB via structure module"
    )
    parser.add_argument(
        "--reconst_dir",
        type=str,
        default="token_sae_reconstructions",
        help="Directory containing *_reconstructed_pair.npy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="token_sae_pdbs",
        help="Output directory for PDB files",
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default=None,
        help="Path to run_structure_module.py (default: same dir as this script)",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="Base dir with protein subdirs (e.g. 6tf4_A/) for FASTA/single lookup",
    )
    parser.add_argument(
        "--model-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device for structure module",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    run_script = Path(args.script_path or str(script_dir / "run_structure_module.py"))
    if not run_script.is_file():
        print(f"Error: run_structure_module.py not found at {run_script}")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(Path(args.reconst_dir) / "*_reconstructed_pair.npy")
    npy_files = sorted(glob.glob(pattern))
    print(f"Found {len(npy_files)} reconstructions. Beginning folding...")

    import numpy as np

    ok = 0
    fail = 0
    for npy_path in npy_files:
        basename = os.path.basename(npy_path)
        protein_id = basename.replace("_reconstructed_pair.npy", "")

        pair = np.load(npy_path, allow_pickle=True)
        n_res = pair.shape[0] if pair.ndim >= 2 else pair.shape[1]

        cmd = [
            sys.executable,
            str(run_script),
            npy_path,
            str(out_dir),
            "--single-from-pair",
            "--model-device",
            args.model_device,
        ]

        fasta_path = None
        if args.base:
            subdir = Path(args.base) / protein_id
            if subdir.is_dir():
                fasta_path = find_fasta(subdir, protein_id)

        temp_aatype = None
        if fasta_path:
            cmd.extend(["--fasta", str(fasta_path)])
        else:
            fd, temp_aatype = tempfile.mkstemp(suffix=".npy")
            os.close(fd)
            np.save(temp_aatype, np.zeros(n_res, dtype=np.int64))
            cmd.extend(["--aatype-npy", temp_aatype])

        if args.dry_run:
            print(f"Would run: {' '.join(cmd)}")
            ok += 1
            if temp_aatype and os.path.exists(temp_aatype):
                os.unlink(temp_aatype)
            continue

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                expected_pdb = out_dir / f"{Path(npy_path).stem}_structure.pdb"
                if expected_pdb.is_file():
                    print(f"  {protein_id} -> {expected_pdb.name}")
                    ok += 1
                else:
                    print(f"  {protein_id} -> no PDB written")
                    fail += 1
            else:
                print(f"  {protein_id} -> failed: {result.stderr[:150] if result.stderr else result.stdout[:150]}")
                fail += 1
        except subprocess.TimeoutExpired:
            print(f"  {protein_id} -> timeout")
            fail += 1
        except Exception as e:
            print(f"  {protein_id} -> {e}")
            fail += 1
        finally:
            if temp_aatype and os.path.exists(temp_aatype):
                try:
                    os.unlink(temp_aatype)
                except OSError:
                    pass

    print(f"\nDone: {ok} OK, {fail} failed. PDBs in {out_dir}/")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
