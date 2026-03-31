#!/usr/bin/env bash
# Build TM-align for Linux (ICE/HPC): installs to <AlphaFold_Autoencoder>/bin/TMalign
# Prereq: g++ (e.g. module load gcc on your cluster)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$ROOT/bin"
mkdir -p "$BIN_DIR"

TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT
cd "$TMP"

SRC_URL="https://zhanggroup.org/TM-align/TMalign.cpp"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
if command -v curl >/dev/null 2>&1; then
  curl -fsSL -A "$UA" -o TMalign.cpp "$SRC_URL"
else
  wget -q -U "$UA" -O TMalign.cpp "$SRC_URL"
fi

if grep -q '#include <malloc.h>' TMalign.cpp; then
  sed -i 's|#include <malloc.h>|#include <cstdlib>|g' TMalign.cpp
fi

compile() {
  g++ "$@" -O3 -ffast-math -lm -o TMalign TMalign.cpp
}

if compile -static 2>/dev/null; then
  :
elif compile; then
  echo "Note: built without -static (still fine for this node)." >&2
else
  echo "g++ failed. Try: module load gcc && $0" >&2
  exit 1
fi

cp TMalign "$BIN_DIR/TMalign"
chmod 755 "$BIN_DIR/TMalign"
echo "TM-align installed: $BIN_DIR/TMalign"
echo "Re-run the notebook or: export TMALIGN_BIN=$BIN_DIR/TMalign"
