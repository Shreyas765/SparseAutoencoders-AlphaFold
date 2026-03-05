#!/usr/bin/env python3
"""
Run TMalign: Compare autoencoder structure vs original structure.

PREDICTED (autoencoder → structure module): sae_pdbs/{id}_predicted.pdb
REFERENCE (original pair_block_47 → structure module): pair_block_47_pdbs/{id}_pair_block_47_structure.pdb

NO Structures/unrelaxed — only structure-module outputs from pair_block_47.

Without --training-info: uses ALL proteins that have both pred and ref (discovers from dirs).
With --training-info: restricts to test_protein_ids (or derives from protein_files 50-50 split).

Usage:
  python compute_tm_scores_structures_predictions.py --tmalign /path/to/TMalign

Output: tm_scores_77.json and tm_scores_77.csv in --output-dir.
"""
import argparse
import json
import os
import random
import re
import subprocess
from datetime import datetime
from typing import Optional, List, Tuple

# Match TM-score value: "TM-score= 0.79880" or "TM-score=0.79880" (optional space)
TMSCORE_RE = re.compile(r"TM-score\s*=\s*([\d.]+)")

TMALIGN_BIN = os.environ.get("TMALIGN_BIN", "TMalign")
TMSCORE_BIN = os.environ.get("TMSCORE_BIN", "")

def compute_tm_tmalign(
    pdb_ref: str, pdb_pred: str, tmalign_bin: str, dump_stdout: Optional[list] = None
) -> Optional[float]:
    """Run TMalign(ref, pred). -ter 0 = do not split on TER/ENDMDL (treat whole file as one chain), matches older TMalign behavior."""
    result = subprocess.run(
        [tmalign_bin, pdb_ref, pdb_pred, "-ter", "0"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if dump_stdout is not None:
        dump_stdout.append(result.stdout)
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
    # TMalign prints: first line TM2 (norm by Structure_1=ref), second TM1 (norm by Structure_2=pred). Use first.
    return scores[0] if scores else None


def compute_tm_tmscore(pdb_pred: str, pdb_ref: str, tmscore_bin: str, use_seq: bool = False) -> Optional[float]:
    """Run Zhang TMscore (model, native) and parse TM-score normalized by native."""
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
    """Reference PDB: structure from original pair_block_47 → {ref_dir}/{id}_pair_block_47_structure.pdb."""
    path = os.path.join(ref_dir, f"{protein_id}_pair_block_47_structure.pdb")
    return path if os.path.isfile(path) else None


def discover_proteins_from_pair_block_47_pdbs(pred_dir: str) -> List[Tuple[str, str]]:
    """List (protein_id, pred_path) from pair_block_47_pdbs/*_pair_block_47_structure.pdb."""
    if not os.path.isdir(pred_dir):
        return []
    pairs = []
    suffix = "_pair_block_47_structure.pdb"
    for f in os.listdir(pred_dir):
        if f.endswith(suffix):
            protein_id = f.replace(suffix, "")
            pred_path = os.path.join(pred_dir, f)
            if os.path.isfile(pred_path):
                pairs.append((protein_id, pred_path))
    return sorted(pairs)


def discover_proteins_from_sae_pdbs(pred_dir: str) -> List[Tuple[str, str]]:
    """List (protein_id, pred_path) from sae_pdbs/*_predicted.pdb."""
    if not os.path.isdir(pred_dir):
        return []
    pairs = []
    for f in os.listdir(pred_dir):
        if f.endswith("_predicted.pdb"):
            protein_id = f.replace("_predicted.pdb", "")
            pred_path = os.path.join(pred_dir, f)
            if os.path.isfile(pred_path):
                pairs.append((protein_id, pred_path))
    return sorted(pairs)


def discover_proteins(pred_dir: str, pred_format: Optional[str] = None) -> List[Tuple[str, str]]:
    """Discover (protein_id, pred_path). Auto-detect format if pred_format is None."""
    if pred_format == "pair_block_47":
        return discover_proteins_from_pair_block_47_pdbs(pred_dir)
    if pred_format == "sae_pdbs":
        return discover_proteins_from_sae_pdbs(pred_dir)
    # Auto-detect: prefer pair_block_47, then sae_pdbs
    pairs = discover_proteins_from_pair_block_47_pdbs(pred_dir)
    if not pairs:
        pairs = discover_proteins_from_sae_pdbs(pred_dir)
    return pairs


def get_test_protein_pairs(
    test_ids: Optional[set],
    pred_dir: str,
    ref_dir: str,
) -> List[Tuple[str, str, str]]:
    """Return [(protein_id, pred_path, ref_path)] for proteins that have both pred and ref."""
    if test_ids:
        candidates = sorted(test_ids)
    else:
        # Discover: all proteins that have BOTH pred and ref (no training_info needed)
        pred_ids = set()
        ref_ids = set()
        if os.path.isdir(pred_dir):
            pred_ids = {f.replace("_predicted.pdb", "") for f in os.listdir(pred_dir) if f.endswith("_predicted.pdb")}
        if os.path.isdir(ref_dir):
            ref_ids = {f.replace("_pair_block_47_structure.pdb", "") for f in os.listdir(ref_dir) if f.endswith("_pair_block_47_structure.pdb")}
        candidates = sorted(pred_ids & ref_ids)
    pairs = []
    for pid in candidates:
        pred_path = os.path.join(pred_dir, f"{pid}_predicted.pdb")
        ref_path = get_ref_pdb(pid, ref_dir)
        if os.path.isfile(pred_path) and ref_path:
            pairs.append((pid, pred_path, ref_path))
    return pairs


def load_test_protein_ids(training_info_path: str) -> Optional[set]:
    """Load test_protein_ids from training_info.json. Falls back to same 50-50 split (seed 42) on protein_files if missing."""
    if not os.path.isfile(training_info_path):
        return None
    try:
        with open(training_info_path) as f:
            cfg = json.load(f)
        ids = set(cfg.get("test_protein_ids") or [])
        if not ids and cfg.get("test_proteins"):
            ids = {
                p.replace("_pair_block_47.npy", "").replace(".npy", "")
                for p in cfg["test_proteins"]
            }
        if not ids and cfg.get("protein_files"):
            # Same split as training notebook: 50-50 train/test, split_seed 42, second half = test
            seed = int(cfg.get("split_seed", 42))
            files = list(cfg["protein_files"])
            order = list(range(len(files)))
            random.seed(seed)
            random.shuffle(order)
            n_test = len(files) // 2
            test_indices = order[-n_test:]
            ids = {
                files[i].replace("_pair_block_47.npy", "").replace(".npy", "")
                for i in test_indices
            }
        return ids if ids else None
    except (json.JSONDecodeError, KeyError):
        return None


def main():
    default_base = os.environ.get(
        "BASE",
        "/storage/scratch1/5/sshrestha304/Autoencoder/CompleteProteins",
    )
    parser = argparse.ArgumentParser(
        description="TM-scores: autoencoder+structure PDBs vs original structure (77 test proteins only)"
    )
    parser.add_argument(
        "--base",
        default=default_base,
        help="Project base (used for default pred-dir and ref-dir)",
    )
    parser.add_argument(
        "--pred-dir",
        default=None,
        help="Predicted PDBs: sae_pdbs/{id}_predicted.pdb (structure from autoencoder) (default: {base}/sae_pdbs)",
    )
    parser.add_argument(
        "--ref-dir",
        default=None,
        help="Reference PDBs: pair_block_47_pdbs/{id}_pair_block_47_structure.pdb (structure from original) (default: {base}/pair_block_47_pdbs)",
    )
    parser.add_argument(
        "--training-info",
        default=None,
        help="Path to training_info.json (optional). If missing, uses all proteins with both pred and ref.",
    )
    parser.add_argument("--output-dir", default="tm_scores_results", help="Where to write tm_scores_77.json and .csv")
    parser.add_argument("--verbose", action="store_true", help="Print exact pred and ref path for each protein")
    parser.add_argument("--tmalign", default=TMALIGN_BIN, help="Path to TMalign binary")
    parser.add_argument("--tmscore", default=TMSCORE_BIN or None, help="Path to TMscore (overrides TMalign if set)")
    parser.add_argument("--use-seq", action="store_true", help="Pass -seq to TMscore if residue numbering differs")
    args = parser.parse_args()

    base = os.path.abspath(args.base)
    pred_dir = os.path.abspath(args.pred_dir or os.path.join(base, "sae_pdbs"))
    ref_dir = os.path.abspath(args.ref_dir or os.path.join(base, "pair_block_47_pdbs"))

    test_ids = load_test_protein_ids(args.training_info) if args.training_info else None

    pairs = get_test_protein_pairs(test_ids, pred_dir, ref_dir)
    if not pairs:
        print(f"No protein pairs found. Need pred={pred_dir}/*_predicted.pdb and ref={ref_dir}/*_pair_block_47_structure.pdb")
        return 1

    missing = (test_ids - {p[0] for p in pairs}) if test_ids else set()
    if missing:
        print(f"Missing pred or ref for {len(missing)} test proteins: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    print(f"Comparing {len(pairs)} proteins: sae_pdbs (autoencoder) vs pair_block_47_pdbs (original)")
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
    first = True
    for protein_id, pred_path, ref_path in pairs:
        if first or args.verbose:
            print(f"  [DEBUG {protein_id}] pred: {pred_path}")
            print(f"  [DEBUG {protein_id}] ref:  {ref_path}")
        tmalign_stdout = [] if first else None
        if use_tmscore:
            score = compute_tm_tmscore(pred_path, ref_path, args.tmscore, args.use_seq)
        else:
            score = compute_tm_tmalign(ref_path, pred_path, args.tmalign, dump_stdout=tmalign_stdout)
        if first and tmalign_stdout:
            print("  [DEBUG] TMalign raw stdout (first protein):")
            for line in tmalign_stdout[0].splitlines():
                print("    |", line)
            first = False
        if score is not None:
            tm_scores[protein_id] = score
            print(f"  {protein_id}: TM-score = {score:.4f}")
        else:
            skipped.append((protein_id, "TMalign failed"))

    for pid in missing:
        skipped.append((pid, "missing pred or ref"))
    for pid, reason in skipped:
        print(f"  Skip {pid}: {reason}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "tm_scores_77.json")
    avg = float(sum(tm_scores.values()) / len(tm_scores)) if tm_scores else None
    with open(out_json, "w") as f:
        json.dump({
            "tm_scores": tm_scores,
            "num_proteins": len(tm_scores),
            "avg_tm_score": avg,
            "skipped": [{"protein_id": p, "reason": r} for p, r in skipped],
            "pred_dir": pred_dir,
            "ref_dir": ref_dir,
            "training_info": args.training_info,
            "timestamp": datetime.utcnow().isoformat(),
        }, f, indent=2)
    avg_display = avg if avg is not None else 0.0
    print(f"\nSaved {out_json} (avg TM-score: {avg_display:.4f}, {len(tm_scores)} proteins)")

    out_csv = os.path.join(args.output_dir, "tm_scores_77.csv")
    with open(out_csv, "w") as f:
        f.write("protein_id,tm_score\n")
        for pid, sc in sorted(tm_scores.items()):
            f.write(f"{pid},{sc:.6f}\n")
    print(f"Saved {out_csv}")

    return 0


if __name__ == "__main__":
    exit(main())
