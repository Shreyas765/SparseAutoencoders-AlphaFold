#!/usr/bin/env python3
"""
Compute TM-scores: NPB epoch 100 PDBs vs original pair_block_47 structure (reference).

PREDICTED: npb_epoch_100_pdbs/{id}_reconstructed_pair_block_47_structure.pdb
REFERENCE: pair_block_47_pdbs/{id}_pair_block_47_structure.pdb

Use this to compare epoch 100 vs epoch 200 (sae_pdbs) — see how much better the autoencoder got with more training.

Usage:
  cd /storage/scratch1/5/sshrestha304/Autoencoder/CompleteProteins
  python compute_tm_scores_epoch_100.py --tmalign /storage/scratch1/5/sshrestha304/bin/TMalign

Output: tm_scores_epoch_100.json and tm_scores_epoch_100.csv in tm_scores_results/
"""
import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from typing import Optional, List, Tuple

TMSCORE_RE = re.compile(r"TM-score\s*=\s*([\d.]+)")
TMALIGN_BIN = os.environ.get("TMALIGN_BIN", "TMalign")
TMSCORE_BIN = os.environ.get("TMSCORE_BIN", "")


def compute_tm_tmalign(pdb_ref: str, pdb_pred: str, tmalign_bin: str) -> Optional[float]:
    """Run TMalign(ref, pred). -ter 0 = do not split on TER/ENDMDL."""
    result = subprocess.run(
        [tmalign_bin, pdb_ref, pdb_pred, "-ter", "0"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return None
    scores = []
    for line in result.stdout.splitlines():
        m = TMSCORE_RE.search(line)
        if m:
            try:
                scores.append(float(m.group(1)))
            except ValueError:
                continue
    return scores[0] if scores else None


def compute_tm_tmscore(pdb_pred: str, pdb_ref: str, tmscore_bin: str, use_seq: bool = False) -> Optional[float]:
    """Run Zhang TMscore (model, native)."""
    cmd = [tmscore_bin, pdb_pred, pdb_ref]
    if use_seq:
        cmd.insert(1, "-seq")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if "TM-score" in line and "=" in line:
            parts = line.split("=")
            if len(parts) >= 2:
                try:
                    return float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    continue
    return None


def get_ref_pdb(protein_id: str, ref_dir: str) -> Optional[str]:
    """Reference: {ref_dir}/{id}_pair_block_47_structure.pdb"""
    path = os.path.join(ref_dir, f"{protein_id}_pair_block_47_structure.pdb")
    return path if os.path.isfile(path) else None


def discover_pairs(pred_dir: str, ref_dir: str) -> List[Tuple[str, str, str]]:
    """Return [(protein_id, pred_path, ref_path)] for proteins with both pred and ref."""
    pred_suffix = "_reconstructed_pair_block_47_structure.pdb"
    pred_ids = set()
    if os.path.isdir(pred_dir):
        for f in os.listdir(pred_dir):
            if f.endswith(pred_suffix):
                pid = f.replace(pred_suffix, "")
                pred_ids.add(pid)
    ref_ids = set()
    if os.path.isdir(ref_dir):
        for f in os.listdir(ref_dir):
            if f.endswith("_pair_block_47_structure.pdb") and "_reconstructed_" not in f:
                pid = f.replace("_pair_block_47_structure.pdb", "")
                ref_ids.add(pid)
    candidates = sorted(pred_ids & ref_ids)
    pairs = []
    for pid in candidates:
        pred_path = os.path.join(pred_dir, f"{pid}_reconstructed_pair_block_47_structure.pdb")
        ref_path = get_ref_pdb(pid, ref_dir)
        if os.path.isfile(pred_path) and ref_path:
            pairs.append((pid, pred_path, ref_path))
    return pairs


def main():
    default_base = os.environ.get(
        "BASE",
        "/storage/scratch1/5/sshrestha304/Autoencoder/CompleteProteins",
    )
    parser = argparse.ArgumentParser(
        description="TM-scores: NPB epoch 100 PDBs vs original structure (reference)"
    )
    parser.add_argument(
        "--base",
        default=default_base,
        help="Project base",
    )
    parser.add_argument(
        "--pred-dir",
        default=None,
        help="Predicted PDBs (default: {base}/npb_epoch_100_pdbs)",
    )
    parser.add_argument(
        "--ref-dir",
        default=None,
        help="Reference PDBs (default: {base}/pair_block_47_pdbs)",
    )
    parser.add_argument(
        "--output-dir",
        default="tm_scores_results",
        help="Where to write tm_scores_epoch_100.json and .csv",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tmalign", default=TMALIGN_BIN)
    parser.add_argument("--tmscore", default=TMSCORE_BIN or None)
    parser.add_argument("--use-seq", action="store_true")
    args = parser.parse_args()

    base = os.path.abspath(args.base)
    pred_dir = os.path.abspath(args.pred_dir or os.path.join(base, "npb_epoch_100_pdbs"))
    ref_dir = os.path.abspath(args.ref_dir or os.path.join(base, "pair_block_47_pdbs"))

    pairs = discover_pairs(pred_dir, ref_dir)
    if not pairs:
        print(
            f"No pairs found. Need pred={pred_dir}/*_reconstructed_pair_block_47_structure.pdb "
            f"and ref={ref_dir}/*_pair_block_47_structure.pdb"
        )
        return 1

    print(f"Comparing {len(pairs)} proteins: NPB epoch 100 vs original (reference)")
    print(f"  Pred: {pred_dir}")
    print(f"  Ref:  {ref_dir}")

    use_tmscore = args.tmscore and os.path.isfile(args.tmscore)
    if not use_tmscore:
        try:
            subprocess.run([args.tmalign, "--help"], capture_output=True, timeout=5)
        except Exception:
            print("Error: TMalign not found. Set --tmalign /path/to/TMalign or TMALIGN_BIN")
            return 1

    tm_scores = {}
    skipped = []
    for protein_id, pred_path, ref_path in pairs:
        if args.verbose:
            print(f"  [DEBUG {protein_id}] pred: {pred_path}")
            print(f"  [DEBUG {protein_id}] ref:  {ref_path}")
        if use_tmscore:
            score = compute_tm_tmscore(pred_path, ref_path, args.tmscore, args.use_seq)
        else:
            score = compute_tm_tmalign(ref_path, pred_path, args.tmalign)
        if score is not None:
            tm_scores[protein_id] = score
            print(f"  {protein_id}: TM-score = {score:.4f}")
        else:
            skipped.append((protein_id, "TMalign failed"))

    for pid, reason in skipped:
        print(f"  Skip {pid}: {reason}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "tm_scores_epoch_100.json")
    avg = float(sum(tm_scores.values()) / len(tm_scores)) if tm_scores else None
    with open(out_json, "w") as f:
        json.dump(
            {
                "tm_scores": tm_scores,
                "num_proteins": len(tm_scores),
                "avg_tm_score": avg,
                "skipped": [{"protein_id": p, "reason": r} for p, r in skipped],
                "pred_dir": pred_dir,
                "ref_dir": ref_dir,
                "description": "NPB epoch 100 vs original pair_block_47 structure",
                "timestamp": datetime.utcnow().isoformat(),
            },
            f,
            indent=2,
        )
    avg_display = avg if avg is not None else 0.0
    print(f"\nSaved {out_json} (avg TM-score: {avg_display:.4f}, {len(tm_scores)} proteins)")

    out_csv = os.path.join(args.output_dir, "tm_scores_epoch_100.csv")
    with open(out_csv, "w") as f:
        f.write("protein_id,tm_score\n")
        for pid, sc in sorted(tm_scores.items()):
            f.write(f"{pid},{sc:.6f}\n")
    print(f"Saved {out_csv}")

    return 0


if __name__ == "__main__":
    exit(main())
