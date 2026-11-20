#!/usr/bin/env bash
# Rebuild goligol/data/otca_db.npz from scratch (Linux / macOS / WSL; needs gcc).
# Takes ~20 minutes, almost all of it in the brute-force variant derivation.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-build}
mkdir -p "$W"
echo "== 1. build the OTCA metapixel ON/OFF tiles from Golly's metafier recipe"
# metafier.py is part of Golly (GPL) and is therefore not included in this repo;
# it is downloaded here and only read as data. Any copy of Golly's
# Scripts/Python/metafier.py works; pass its path as 2nd argument to skip the download.
MF=${2:-"$W/metafier.py"}
if [ ! -f "$MF" ]; then
  curl -fsSL -o "$MF" https://raw.githubusercontent.com/AlephAlpha/golly/master/Scripts/Python/metafier.py
fi
python3 tools/build_metacell.py "$MF" "$W"
echo "== 2. derive the 8 phase-0 variants by brute-force simulation (8x8 torus, 4 periods)"
gcc -O3 -march=native -fopenmp -o "$W/torus_sim" tools/torus_sim.c
python3 tools/make_variants.py "$W/otca_on.npy" "$W/otca_off.npy" "$W/torus_sim" "$W/variants_work" "$W/variants.bin"
echo "== 3. torus layout containing all 2048 contexts"
python3 tools/make_layout.py 64 64 7 "$W/lay64.bin"
echo "== 4. HashLife: one full period of a 64x64 metapixel torus, checkpoints every 12 generations"
gcc -O3 -march=native -o "$W/hl_db" tools/hl_db.c
"$W/hl_db" "$W/lay64.bin" "$W/variants.bin" 12 "$W/db12"
echo "== 5. verify and pack"
python3 tools/pack_db.py "$W/lay64.bin" 12 "$W/db12" goligol/data/otca_db.npz
