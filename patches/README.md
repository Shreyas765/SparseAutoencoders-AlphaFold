# Patches for OpenFold clone issues

## `FileNotFoundError: No checkpoint` / `finetuning_ptm_1.pt`

`run_structure_module.py` needs OpenFold **weights**. Either:

1. Place **`finetuning_ptm_1.pt`** under  
   `OPENFOLD_REPO/openfold/resources/openfold_params/`  
   (download from OpenFold’s [docs](https://openfold.readthedocs.io/) / install script), or  
2. Set **`OPENFOLD_CHECKPOINT_PATH=/full/path/to/finetuning_ptm_1.pt`** before running the PDB notebook.

The notebook passes **`--openfold-checkpoint-path`** automatically when that file exists.

---


## `attention_core.py` — `IndentationError` after `try:`

Your OpenFold checkout may have a **corrupted** `openfold/utils/kernel/attention_core.py` (often a stray empty `try:` block).

**Fix on the cluster** (replace `OPENFOLD_REPO` with your path, e.g. `/storage/ice1/1/0/varisa3/AlphaFold_Autoencoder/openfold`):

```bash
cp patches/attention_core.py "$OPENFOLD_REPO/openfold/utils/kernel/attention_core.py"
```

Or from git:

```bash
cd "$OPENFOLD_REPO"
git checkout -- openfold/utils/kernel/attention_core.py
```

Or:

```bash
curl -L -o "$OPENFOLD_REPO/openfold/utils/kernel/attention_core.py" \
  https://raw.githubusercontent.com/aqlaboratory/openfold/main/openfold/utils/kernel/attention_core.py
```

## `ModuleNotFoundError: No module named 'attn_core_inplace_cuda'`

That module is a **compiled CUDA extension** built when you install OpenFold. Copying only `.py` files is not enough.

**On the cluster (GPU node, CUDA module loaded if your site requires it):**

```bash
cd /storage/ice1/1/0/varisa3/AlphaFold_Autoencoder/openfold   # your OPENFOLD_REPO
# optional: module load cuda/12.x   # if your center uses modules
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"   # adjust to where nvcc lives
# Use the SAME python as Jupyter; --no-build-isolation so setup.py can import torch
python -m pip install -e . --no-build-isolation
```

Then verify:

```bash
python -c "import attn_core_inplace_cuda; print('ok')"
```

If `pip install -e .` fails, read the compiler output (missing `nvcc`, wrong `CUDA_HOME`, PyTorch CUDA mismatch). See [OpenFold docs](https://openfold.readthedocs.io/).

**Important:** `pip install` must run in the **same environment** as the notebook kernel (same `which python` / `conda env`).
