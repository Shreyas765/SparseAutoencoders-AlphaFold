# Matryoshka SAE on GPU Cluster — Setup & Run

Run the Matryoshka autoencoder notebook on a GPU super cluster with everything in one directory.

## 1. Folder layout (same directory as the notebook)

Put these in **the same directory** as `matryoshka_sae.ipynb` and `matryoshka_sae.py`:

| Item | Description |
|------|-------------|
| `Proteins_layer47/` or `protein_layer47/` | Pair representation `.npy` files (input). Name either way; the notebook auto-detects. |
| `output_structures/` | Predicted structure PDBs for TM-score (e.g. `7b3a_A_pair_block_47_structure.pdb`). Add this if you already have structure module outputs. |
| `truth_pdbs/` | Ground truth PDBs per chain (e.g. `7b3a_A.pdb`). Use `download_ground_truth_pdbs.py` if needed. |
| `bin/TMalign` | TMalign binary for TM-score (build on Mac with `build_tmalign_mac.sh`; on cluster use a Linux-built binary). |
| `af2_autoencoder.py` | Required by the notebook (same directory). |

Generated outputs (created when you run the notebook):

- `output_matryoshka_sae/` — model checkpoints, plots, `original_pair_reps/`, `recon_pair_reps/`
- `original_pair_reps/`, `recon_pair_reps/` — if you use the notebook’s “save pair reps” cell

## 2. Environment (GPU cluster)

```bash
# Example: conda
conda create -n matryoshka python=3.10
conda activate matryoshka
pip install -r requirements_matryoshka.txt

# Or venv
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
pip install -r requirements_matryoshka.txt
```

Request a GPU when starting your job (e.g. `--gres=gpu:1` for Slurm).

## 3. Run the notebook

### Option A: Jupyter on the cluster

1. Start Jupyter (or JupyterLab) on a GPU node and open `matryoshka_sae.ipynb`.
2. Run the **first setup cell** (sets working directory).
3. Run all cells (e.g. “Run All”). Training and TM-score will use the GPU and the paths above.

If your session starts in a different directory, set `PROJECT_DIR` in the setup cell, e.g.:

```python
PROJECT_DIR = Path("/path/to/your/VizFoldAutoencoder")
```

### Option B: Execute notebook in batch (no browser)

From the directory that contains `matryoshka_sae.ipynb`:

```bash
# From VizFoldAutoencoder/
jupyter nbconvert --to notebook --execute matryoshka_sae.ipynb --output matryoshka_sae_executed.ipynb
```

Or use the provided script (see below).

## 4. Run script (batch job)

`run_matryoshka_notebook.sh` runs the notebook with the correct working directory. Use it in your job script:

```bash
#!/bin/bash
#SBATCH --job-name=matryoshka
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=matryoshka_%j.out

cd /path/to/VizFoldAutoencoder    # or $SLURM_SUBMIT_DIR
source .venv/bin/activate         # or conda activate matryoshka
bash run_matryoshka_notebook.sh
```

## 5. TM-score

- If `output_structures/` is present and contains PDBs named like `7b3a_A_pair_block_47_structure.pdb`, the TM-score cell will run and compare them to `truth_pdbs/`.
- Config uses `pred_pdb_suffix": "_pair_block_47_structure"` by default; change it in the config cell if your PDB names differ.

## 6. TMalign (for TM-score) — use `bin/` in project folder

The notebook uses **TMalign** from **`bin/TMalign`** inside the project directory (path is set automatically; works on your Mac and on the supercluster).

- **On your Mac:** Build with `build_tmalign_mac.sh`; the binary is placed in `VizFoldAutoencoder/bin/TMalign`.
- **On the supercluster:** Add a **`bin/`** folder to your project copy and put the TMalign binary there as **`bin/TMalign`**.
  - Either copy a **Linux-built** TMalign (the Mac binary will not run on Linux). Build on the cluster:
    ```bash
    # On the cluster (Linux), in a folder containing TMalign.cpp:
    g++ -static -O3 -ffast-math -lm -o TMalign TMalign.cpp
    # If malloc.h error: sed -i 's|#include <malloc.h>|#include <cstdlib>|g' TMalign.cpp then re-run g++
    mkdir -p /path/to/VizFoldAutoencoder/bin
    mv TMalign /path/to/VizFoldAutoencoder/bin/
    ```
  - Or download the Linux prebuilt binary from [Zhang Lab](https://zhanggroup.org/TM-align/) and place it at `bin/TMalign`.

Upload your project folder to the cluster **including** `bin/TMalign` so the TM-score cell runs without changes.

## 7. Quick checklist

- [ ] `Proteins_layer47/` or `protein_layer47/` in same dir as notebook  
- [ ] `truth_pdbs/` in same dir (and optionally `output_structures/`)  
- [ ] **`bin/TMalign`** in same dir (Linux binary on cluster, Mac binary on Mac)  
- [ ] `af2_autoencoder.py` and `matryoshka_sae.py` in same dir (local modules)  
- [ ] Python env with dependencies (`pip install -r requirements_matryoshka.txt`)  
- [ ] GPU requested for your job  
- [ ] Run setup cell first, then Run All
