#!/usr/bin/env bash
# Run the Matryoshka SAE notebook in batch (e.g. on a GPU cluster).
# Usage: from the directory containing this script, run: bash run_matryoshka_notebook.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! [[ -f matryoshka_sae.ipynb ]]; then
    echo "Error: matryoshka_sae.ipynb not found in $SCRIPT_DIR"
    exit 1
fi

echo "Working directory: $SCRIPT_DIR"
echo "Executing notebook..."
jupyter nbconvert --to notebook --execute matryoshka_sae.ipynb \
    --output "matryoshka_sae_executed.ipynb" \
    --ExecutePreprocessor.timeout=86400

echo "Done. Output notebook: matryoshka_sae_executed.ipynb"
