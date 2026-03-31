"""
Calculates TM-Scores comparing Token SAE reconstructed PDBs to reference PDBs.
Pred: token_sae_pdbs/{id}_reconstructed_pair_structure.pdb
Ref:  pair_block_47_pdbs/{id}_pair_block_47_structure.pdb
Outputs JSON and CSV for research metrics.
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

TMSCORE_RE = re.compile(r"TM-score\s*=\s*([\d.]+)")
TMALIGN_CPP_URL = "https://zhanggroup.org/TM-align/TMalign.cpp"

_SCRIPT_DIR = Path(__file__).resolve().parent

# Zhang lab (and similar) often return 403 for urllib's default User-Agent.
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def _download_url_to_path(url: str, dest: Path, *, timeout: int = 120) -> None:
    req = urllib.request.Request(url, headers=_HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        dest.write_bytes(resp.read())


def _find_local_tmalign_cpp(
    search_roots: Sequence[Path],
    install_under: Path,
) -> Optional[Path]:
    """First existing ``TMalign.cpp`` under search roots or install_under (e.g. next to the notebook)."""
    seen: set[Path] = set()
    for base in (*search_roots, Path(install_under)):
        p = Path(base).resolve() / "TMalign.cpp"
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            return p
    return None


def resolve_tmalign_bin(
    user_value: str,
    *,
    search_roots: Optional[Sequence[Path]] = None,
) -> Optional[str]:
    """
    Return an executable path for TM-align.

    Accepts a full path to the binary, or a name looked up on PATH (TMalign / tmalign).
    If ``search_roots`` is set, also tries ``<root>/bin/TMalign`` and ``<root>/bin/tmalign``
    (useful on clusters where the binary is shipped next to the project, not on PATH).
    """
    if not user_value:
        return None

    path_like = Path(user_value).expanduser()
    if path_like.is_absolute() or os.path.sep in user_value:
        p = path_like.resolve()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        return None

    if search_roots:
        for root in search_roots:
            r = Path(root).resolve()
            for name in ("TMalign", "tmalign"):
                cand = r / "bin" / name
                if cand.is_file() and os.access(cand, os.X_OK):
                    return str(cand)

    for candidate in (user_value, "TMalign", "tmalign"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def ensure_tmalign_or_build(
    user_value: str,
    *,
    search_roots: Sequence[Path],
    install_under: Path,
    allow_build: bool = True,
) -> str:
    """
    Return a usable TM-align executable path.

    Uses :func:`resolve_tmalign_bin` first. If nothing is found and ``allow_build`` is
    true, compiles ``TMalign.cpp`` and installs to ``<install_under>/bin/TMalign``.
    Source file is taken, in order, from: ``TMalign.cpp`` next to the notebook /
    under ``search_roots`` / ``install_under``, else downloaded from the Zhang lab URL
    (needs network). Building always needs ``g++``.
    """
    found = resolve_tmalign_bin(user_value, search_roots=search_roots)
    if found:
        return found

    bin_dir = Path(install_under).resolve() / "bin"
    out_bin = bin_dir / "TMalign"
    if out_bin.is_file() and os.access(out_bin, os.X_OK):
        return str(out_bin)

    if not allow_build:
        raise FileNotFoundError(
            "TM-align not found and auto-build disabled "
            "(set allow_build=True or install TM-align manually)."
        )

    bin_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tmalign_build_") as tmp:
        tmp_path = Path(tmp)
        local_cpp = _find_local_tmalign_cpp(search_roots, install_under)
        if local_cpp is not None:
            text = local_cpp.read_text(encoding="utf-8", errors="replace")
            if "#include <malloc.h>" in text:
                src = tmp_path / "TMalign.cpp"
                src.write_text(
                    text.replace("#include <malloc.h>", "#include <cstdlib>"),
                    encoding="utf-8",
                )
            else:
                src = local_cpp
        else:
            src = tmp_path / "TMalign.cpp"
            try:
                _download_url_to_path(TMALIGN_CPP_URL, src)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                raise RuntimeError(
                    f"Could not download TMalign.cpp ({e}). "
                    "Place TMalign.cpp in the same folder as this notebook (original_SAE) "
                    "or under AlphaFold_Autoencoder/, then re-run; or use bash build_tmalign_linux.sh; "
                    f"or set TMALIGN_BIN={out_bin}."
                ) from e
            text = src.read_text(encoding="utf-8", errors="replace")
            if "#include <malloc.h>" in text:
                src.write_text(
                    text.replace("#include <malloc.h>", "#include <cstdlib>"),
                    encoding="utf-8",
                )

        exe = tmp_path / "TMalign"
        last_err = ""

        for static in (True, False):
            cmd = ["g++", "-O3", "-ffast-math", "-lm", "-o", str(exe), str(src)]
            if static:
                cmd.insert(1, "-static")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and exe.is_file():
                shutil.copy2(exe, out_bin)
                os.chmod(out_bin, 0o755)
                return str(out_bin)
            last_err = (r.stderr or r.stdout or "").strip()

        hint = (
            "Could not compile TM-align (needs g++, often: module load gcc on a cluster).\n"
            "Last compiler output:\n"
            f"{last_err[:4000] or '(empty)'}\n"
            "Or run: bash build_tmalign_linux.sh in original_SAE/"
        )
        raise RuntimeError(hint)


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

    tmalign_resolved = resolve_tmalign_bin(
        args.tmalign,
        search_roots=(_SCRIPT_DIR, _SCRIPT_DIR.parent),
    )
    if tmalign_resolved is None:
        print(
            "ERROR: TM-align executable not found.\n"
            f"  Tried: {args.tmalign!r}, PATH (TMalign/tmalign), and "
            f"{_SCRIPT_DIR / 'bin' / 'TMalign'}, {_SCRIPT_DIR.parent / 'bin' / 'TMalign'}\n"
            "  Fix on a cluster: in original_SAE run bash build_tmalign_linux.sh "
            "(needs g++; try: module load gcc). Or install from "
            "https://zhanggroup.org/TM-align/ and export TMALIGN_BIN=/full/path/to/TMalign."
        )
        return 1
    args.tmalign = tmalign_resolved
    print(f"Using TM-align: {args.tmalign}")

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
