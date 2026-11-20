"""Cross-check the HashLife generator (hl_db) against brute force (gen_db).

usage: crosscheck.py variants.bin workdir
Builds a 4x4 torus with random variants, runs gen_db (dense bit-parallel
simulation) on it and hl_db on the same 4x4 pattern tiled 16x16 times (the
64x64 torus hl_db needs), and compares cells and birth parity of every tile
at every checkpoint of one full period.  Needs gen_db and hl_db compiled
into workdir.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from goligol.tiledb import TileDB  # noqa: E402

variants, wd = os.path.abspath(sys.argv[1]), sys.argv[2]
K = 92
rng = np.random.default_rng(3)
v4 = rng.integers(0, 8, (4, 4)).astype(np.uint8)
v4.tofile(os.path.join(wd, "v4.bin"))
np.tile(v4, (16, 16)).tofile(os.path.join(wd, "v64.bin"))
subprocess.run(["./gen_db", "4", "4", "v4.bin", variants, str(K), "bf"], cwd=wd, check=True)
subprocess.run(["./hl_db", "v64.bin", variants, str(K), "hl"], cwd=wd, check=True)


class Raw:
    expand = TileDB.expand

    def __init__(self, prefix, ntiles):
        self.blocks = np.fromfile(prefix + ".blocks", "<u8").reshape(-1, 64)
        self.nodes = [None] + [np.fromfile(f"{prefix}.n{L}", "<u4").reshape(-1, 4) for L in range(1, 6)]
        self.roots = np.fromfile(prefix + ".roots", "<u4").reshape(-1, ntiles, 2)


bf = Raw(os.path.join(wd, "bf"), 16)
hl = Raw(os.path.join(wd, "hl"), 4096)
bad = 0
for c in range(len(bf.roots)):
    for i in range(16):
        ty, tx = divmod(i, 4)
        for w in (0, 1):
            a = bf.expand(int(bf.roots[c, i, w]))
            b = hl.expand(int(hl.roots[c, (ty + 20) * 64 + tx + 36, w]))
            bad += int(not (a == b).all())
print(f"compared {len(bf.roots)} checkpoints x 16 tiles x (cells, parity): {bad} mismatches")
sys.exit(1 if bad else 0)
