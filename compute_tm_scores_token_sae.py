"""
Calculates TM-Scores comparing Token SAE reconstructed PDBs to reference PDBs.
Pred: token_sae_pdbs/{id}_reconstructed_pair_structure.pdb
Ref:  pair_block_47_pdbs/{id}_pair_block_47_structure.pdb
Outputs JSON and CSV for research metrics.
"""
import os
import re
import glob
import json
import csv
import subprocess
import argparse
from pathlib import Path

TMSCORE_RE = re.compile(r"TM-score\s*=\s*([\d.]+)")


def run_tmalign(pred_pdb: str, ref_pdb: str, tmalign_bin: str) -> float:
    """Run TMalign(ref, pred) with -ter 0; return TM-score normalized by ref (first line)."""
    try:
        result = subprocess.run(
            [tmalign_bin, ref_pdb, pred_pdb, "-ter", "0"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            m = TMSCORE_RE.search(line)
            if m:
                return float(m.group(1))
    except Exception as e:
        print(f"Error running TMalign on {pred_pdb}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="TM-scores: Token SAE PDBs vs reference pair_block_47 structure PDBs"
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="token_sae_pdbs",
        help="Predicted PDBs: *_reconstructed_pair_structure.pdb",
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        default="pair_block_47_pdbs",
        help="Reference PDBs: *_pair_block_47_structure.pdb",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="tm_scores_token_sae.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--tmalign",
        type=str,
        default=os.environ.get("TMALIGN_BIN", "TMalign"),
        help="Path to TMalign binary",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)
    if not pred_dir.is_dir():
        print(f"Pred dir not found: {pred_dir}")
        return 1

    pred_files = sorted(glob.glob(str(pred_dir / "*_reconstructed_pair_structure.pdb")))
    scores = {}

    print(f"Calculating TM-scores for {len(pred_files)} structures...")
    print(f"  Pred: {pred_dir}")
    print(f"  Ref:  {ref_dir}")

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        protein_id = filename.replace("_reconstructed_pair_structure.pdb", "")

        ref_path = ref_dir / f"{protein_id}_pair_block_47_structure.pdb"

        if ref_path.is_file():
            score = run_tmalign(pred_path, str(ref_path), args.tmalign)
            if score is not None:
                scores[protein_id] = score
        else:
            print(f"Warning: Missing reference for {protein_id} at {ref_path}")

    valid_scores = list(scores.values())
    if not valid_scores:
        print("No valid TM-scores calculated.")
        return 1

    avg_score = sum(valid_scores) / len(valid_scores)
    print(f"\n--- Results ---")
    print(f"Total aligned: {len(valid_scores)}")
    print(f"Average TM-Score: {avg_score:.4f}")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "average_tm_score": avg_score,
        "total_evaluated": len(valid_scores),
        "individual_scores": scores,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Scores saved to {out_path}")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protein_id", "tm_score"])
        for pid, s in sorted(scores.items()):
            w.writerow([pid, f"{s:.6f}"])
    print(f"CSV saved to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
