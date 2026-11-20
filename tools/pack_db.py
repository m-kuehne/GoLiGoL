"""Pack hl_db output into goligol/data/otca_db.npz and verify it.

usage: pack_db.py layout.bin K prefix out.npz

Checks:
  * functional dependence: all tiles with the same context have identical
    cells and birth parity at every checkpoint (the 64x64 torus contains each
    of the 2048 contexts about twice, with different surroundings);
  * at generation 35328 every tile equals the phase-0 variant predicted by
    applying the Life rule to the meta-cells
    (prev' = cur, cur' = life(cur), hb' = hb xor (cur and not prev)).
Also searches for the self-similar chain used by tower mode.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from goligol import life  # noqa: E402
from goligol.tiledb import TileDB  # noqa: E402

NT = 64
lay = np.fromfile(sys.argv[1], np.uint8).reshape(NT, NT)
K = int(sys.argv[2]); prefix = sys.argv[3]; out = sys.argv[4]
M = 35328 // K

prev = (lay & 1).astype(np.int64); cur = ((lay >> 1) & 1).astype(np.int64); hb = (lay >> 2).astype(np.int64)
nb = np.zeros_like(cur)
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        nb = nb * 2 + np.roll(np.roll(cur, -dy, 0), -dx, 1)
ctx = ((nb << 2) | prev | (hb << 1)).ravel()

roots = np.fromfile(prefix + ".roots", "<u4").reshape(M, NT * NT, 2)
first = np.full(2048, -1)
for i, c in enumerate(ctx):
    if first[c] < 0:
        first[c] = i
have = first >= 0
table = np.zeros((2048, M, 2), np.uint32)
table[have] = roots[:, first[have]].transpose(1, 0, 2)
dup = np.array([i for i, c in enumerate(ctx) if first[c] != i])
bad = int((roots[:, dup] != roots[:, first[ctx[dup]]]).any(axis=(0, 2)).sum()) if len(dup) else 0
print(f"contexts covered {int(have.sum())}/2048, repeated samples {len(dup)}, FD violations {bad}")
assert bad == 0 and have.all()

n = sum(np.roll(np.roll(cur, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0))
nxt = ((n == 3) | ((n == 2) & (cur == 1))).astype(np.int64)
v2 = cur + 2 * nxt + 4 * (hb ^ ((1 - prev) & cur))
fin = np.fromfile(prefix + ".final", "<u4").reshape(NT, NT)
vroots = np.fromfile(prefix + ".vroots", "<u4")
ok = bool((fin == vroots[v2]).all())
print("generation-35328 tiles equal the Life-predicted variants:", ok)
assert ok

blocks = np.fromfile(prefix + ".blocks", "<u8").reshape(-1, 64)
nodes = {f"n{L}": np.fromfile(f"{prefix}.n{L}", "<u4").reshape(-1, 4) for L in range(1, 6)}
vkey = (ctx & 1) + 2 * ((ctx >> 6) & 1) + 4 * ((ctx >> 1) & 1)
vctx = np.array([ctx[np.nonzero(vkey == v)[0][0]] for v in range(8)])


class _Tmp(TileDB):
    def __init__(self):
        self.K, self.M, self.blocks = K, M, blocks
        self.nodes = [None] + [nodes[f"n{L}"] for L in range(1, 6)]
        self.roots = table


db = _Tmp()


def child_ctx(ctx_id, c):
    """Context of every cell of tile ctx_id at generation c*K + 1."""
    t0p = db.tile(ctx_id, c, 0)
    t0 = life.unpack(t0p).astype(np.int32)
    p0 = life.unpack(db.tile(ctx_id, c, 1)).astype(np.int32)
    t1 = life.unpack(life.step(t0p)).astype(np.int32)
    pad = np.pad(t1, 1)
    k = np.zeros((2048, 2048), np.int32)
    for dy in range(3):
        for dx in range(3):
            k = (k << 1) | pad[dy:dy + 2048, dx:dx + 2048]
    return (k << 2) | t0 | (p0 << 1)


chain = None
rng = np.random.default_rng(0)
for c in range(M // 3, M, max(1, M // 24)):
    samples = [child_ctx(int(x), c) for x in rng.choice(2048, 6, replace=False)]
    same = np.all([s == samples[0] for s in samples[1:]], axis=0)
    cand = same & ((samples[0] >> 6) & 1).astype(bool)          # live, context-independent
    cand[:48] = cand[-48:] = False
    cand[:, :48] = cand[:, -48:] = False
    ys, xs = np.nonzero(cand)
    order = np.argsort((ys - 1024) ** 2 + (xs - 1024) ** 2)
    tried = {}
    for j in order[:20000]:
        k = int(samples[0][ys[j], xs[j]])
        if k not in tried:
            tried[k] = child_ctx(k, c)
        if tried[k][ys[j], xs[j]] == k:
            chain = np.array([k, c, xs[j], ys[j]], np.int64)
            break
        if len(tried) > 12:
            break
    if chain is not None:
        break
print("self-similar chain (context, checkpoint, x, y):", chain)

extra = {} if chain is None else {"chain": chain}
np.savez_compressed(out, K=np.int64(K), blocks=blocks, roots=table, have=have,
                    variant_ctx=vctx, **nodes, **extra)
print("wrote", out, os.path.getsize(out) // 1024, "KiB; blocks", len(blocks),
      "nodes", [len(nodes[f'n{L}']) for L in range(1, 6)])
