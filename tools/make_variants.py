"""Derive the 8 phase-0 variants of the OTCA metapixel by brute force.

A metapixel at the start of its 35328-generation period is fully described
by three bits:
  prev - the state it is *displaying* (HWSS-filled interior = on),
  cur  - the state it has just computed (stored in its state bit; the display
         is redrawn to show it ~5000-9000 generations into the period),
  hb   - a "honeybit": a small object at tile rows 105-112, cols 1925-1932
         whose presence toggles whenever the cell is born
         (hb' = hb xor (cur and not prev)).

Procedure: run an 8x8 torus of freshly built metapixels (random states) for
four periods with torus_sim, then read the tiles at the end of periods 3 and
4 (steady state), label them with the meta-level Life history and check that
equal labels give identical tiles.

usage: make_variants.py otca_on.npy otca_off.npy torus_sim_binary workdir out_variants.bin
"""
import os
import subprocess
import sys

import numpy as np

on, off, sim, wd, out = sys.argv[1:6]
os.makedirs(wd, exist_ok=True)
N, P, PER = 8, 4, 35328
np.load(on).astype("<i4").tofile(os.path.join(wd, "on.i32"))
np.load(off).astype("<i4").tofile(os.path.join(wd, "off.i32"))
rng = np.random.default_rng(7)
s0 = (rng.random((N, N)) < 0.4).astype(np.uint8)
s0.tofile(os.path.join(wd, "s8.bin"))
ks = [str(768 * p) for p in range(P + 1)]
subprocess.run([sim, str(N), "s8.bin", "on.i32", "off.i32", str(PER * P), "46", "h.bin", "t.bin", *ks],
               cwd=wd, check=True)
t = np.fromfile(os.path.join(wd, "t.bin"), "<u8").reshape(P + 1, N * N, 2048, 32)


def life_t(b):
    n = sum(np.roll(np.roll(b, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0))
    return ((n == 3) | ((n == 2) & (b == 1))).astype(np.uint8)


S = [s0]
for _ in range(P + 1):
    S.append(life_t(S[-1]))

variants = {}
for p in (3, 4):
    for i in range(N * N):
        tile = t[p, i]
        hbwin = np.unpackbits(np.ascontiguousarray(tile[105:113, 30]).view(np.uint8), bitorder="little")
        hbwin = hbwin.reshape(8, 64)[:, 5:13]
        hb = int(hbwin.sum() > 0)
        v = int(S[p - 1].ravel()[i]) + 2 * int(S[p].ravel()[i]) + 4 * hb
        if v in variants:
            assert (variants[v] == tile).all(), "label does not determine the tile"
        else:
            variants[v] = tile.copy()
missing = [v for v in range(8) if v not in variants]
assert not missing, f"variants not seen: {missing} (try another seed)"
np.stack([variants[v] for v in range(8)]).astype("<u8").tofile(out)
print("wrote", out)
