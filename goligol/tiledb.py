"""Hash-consed quadtree database of OTCA metapixel tiles.

Leaves are 64x64 blocks (64 uint64 rows, bit i of a row = column i).
Level-L nodes (L = 1..5) have four children (NW, NE, SW, SE) of level L-1;
a level-5 node is a full 2048x2048 tile.  Id 0 is the empty block/node at
every level.

For each metapixel *context* (see ``context_id``) and each checkpoint
``c`` (generation ``c * K`` of the 35328-generation metapixel period) the
database stores two roots: the tile's cells and the tile's *birth parity*
(parity of births of each cell since generation 0 of the period).
"""
from __future__ import annotations

import os

import numpy as np

PERIOD = 35328
TILE = 2048
LEAF = 64


def context_id(nbhd: int, prev: int, hb: int) -> int:
    """nbhd: 9-bit 3x3 neighbourhood, row-major from the NW corner, MSB first
    (the cell itself is bit 4, value 16)."""
    return (nbhd << 2) | prev | (hb << 1)


class TileDB:
    def __init__(self, path: str):
        z = np.load(path)
        self.K = int(z["K"])
        self.M = PERIOD // self.K
        self.blocks = z["blocks"]                      # (nb, 64) uint64
        self.nodes = [None] + [z[f"n{L}"] for L in range(1, 6)]   # (n, 4) uint32
        self.roots = z["roots"]                        # (2048, M, 2) uint32
        self.variant_ctx = z["variant_ctx"]            # (8,) context of each phase-0 variant
        self.have = z["have"] if "have" in z else np.ones(2048, bool)
        self.extra = {k: z[k] for k in z.files if k in ("chain",)}
        self._pop = None
        self._img_cache = {}

    # ------------------------------------------------------------------
    def expand(self, root: int, level: int = 5) -> np.ndarray:
        """Return the (64*2**level)^2 bitmap of a node as packed uint64 rows
        (shape (side, side//64))."""
        ids = np.array([[root]], dtype=np.int64)
        for L in range(level, 0, -1):
            ch = self.nodes[L][ids]                    # (h, w, 4)
            h, w = ids.shape
            nxt = np.empty((2 * h, 2 * w), np.int64)
            nxt[0::2, 0::2] = ch[..., 0]; nxt[0::2, 1::2] = ch[..., 1]
            nxt[1::2, 0::2] = ch[..., 2]; nxt[1::2, 1::2] = ch[..., 3]
            ids = nxt
        b = self.blocks[ids]                           # (h, w, 64)
        h, w = ids.shape
        return np.ascontiguousarray(b.transpose(0, 2, 1).reshape(h * 64, w))

    def tile(self, ctx: int, c: int, which: int = 0) -> np.ndarray:
        """Packed 2048x2048 bitmap (2048, 32) of context ``ctx`` at checkpoint ``c``.
        which = 0: cells, 1: birth parity."""
        return self.expand(int(self.roots[ctx, c, which]))

    # ------------------------------------------------------------------
    def _populations(self):
        if self._pop is None:
            nb = len(self.blocks)
            pop0 = np.zeros(nb, np.int64)
            bp16 = np.zeros((nb, 16, 16), np.uint8)
            for i in range(0, nb, 4096):
                bits = np.unpackbits(self.blocks[i:i + 4096].view(np.uint8), axis=1, bitorder="little")
                bits = bits.reshape(-1, 64, 64)
                pop0[i:i + 4096] = bits.sum((1, 2))
                bp16[i:i + 4096] = bits.reshape(-1, 16, 4, 16, 4).sum((2, 4))
            pop = [pop0]
            for L in range(1, 6):
                pop.append(pop[L - 1][self.nodes[L]].sum(1))
            self._pop = pop
            self._bp16 = bp16
        return self._pop

    def ids(self, roots: np.ndarray, level: int) -> np.ndarray:
        """Expand level-5 roots (n,) to their level-``level`` descendants:
        (n, 2**(5-level), 2**(5-level)) node ids (level 0 = block ids)."""
        ids = np.asarray(roots, np.int64).reshape(-1, 1, 1)
        for L in range(5, level, -1):
            ch = self.nodes[L][ids]
            n, h, w = ids.shape
            nxt = np.empty((n, 2 * h, 2 * w), np.int64)
            nxt[:, 0::2, 0::2] = ch[..., 0]; nxt[:, 0::2, 1::2] = ch[..., 1]
            nxt[:, 1::2, 0::2] = ch[..., 2]; nxt[:, 1::2, 1::2] = ch[..., 3]
            ids = nxt
        return ids

    def density_many(self, roots, res: int) -> np.ndarray:
        """(n, res, res) float32 live-cell fraction images of tiles."""
        roots = np.asarray(roots, np.int64).ravel()
        pop = self._populations()
        if res <= 32:
            level = 5 - int(np.log2(res))
            side = 64 << level
            return (pop[level][self.ids(roots, level)] / float(side * side)).astype(np.float32)
        if res <= 512:
            ids = self.ids(roots, 0)                         # (n, 32, 32)
            img = self._bp16[ids]                            # (n, 32, 32, 16, 16)
            n = len(roots)
            img = img.transpose(0, 1, 3, 2, 4).reshape(n, 512, 512).astype(np.float32) / 16.0
            f = 512 // res
            if f > 1:
                img = img.reshape(n, res, f, res, f).mean((2, 4))
            return img
        out = np.empty((len(roots), res, res), np.float32)
        f = TILE // res
        for i, r in enumerate(roots):
            bits = np.unpackbits(self.expand(int(r)).view(np.uint8), axis=1, bitorder="little")
            out[i] = bits.reshape(res, f, res, f).mean((1, 3)) if f > 1 else bits
        return out


def default_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "otca_db.npz")
