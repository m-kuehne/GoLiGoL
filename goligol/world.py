"""The infinite tower of Life-in-Life.

Level L+1 cells are OTCA metapixels made of 2048x2048 level-L cells; every
level is an exact Game of Life.  A level-L cell at relative coordinate ``x``
lies in parent cell ``floor((x + off) / 2048)`` at tile column
``(x + off) mod 2048`` (``off`` is 0 in pattern mode and the chain position
in tower mode).

Time is a mixed-radix number: ``digits[L]`` is level L's generation inside
the current generation (metapixel period) of level L+1, 0 <= digit < 35328.

A metapixel's 2048^2 cells during one period are a function of its *context*
(own previous state, own honeybit, 3x3 neighbourhood of current states); the
context database stores them every K generations, and the exact cells in
between are obtained by running Life from the nearest checkpoint.

Pattern mode: level 0 is a user-editable toroidal Life universe (the top).
Tower mode:   there is no top.  A self-reproducing chain of metapixels
              (context C at period phase g* contains, at tile position p, a
              cell whose own context is again C) is used for every level
              above the ones whose digits have been touched, so one can zoom
              out forever.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import life
from .tiledb import PERIOD, TILE, TileDB

WPT = TILE // 64          # words per tile row


@dataclass
class Region:
    """Packed state of a rectangle of one level.  Bit b of word w in row j is
    the cell (ox + 64*w + b, oy + j)."""
    ox: int
    oy: int
    cur: np.ndarray
    prev: np.ndarray
    par: np.ndarray      # birth parity since the start of the parent's period
    hb: np.ndarray       # honeybit (= birth parity one generation earlier)

    @property
    def width(self):
        return self.cur.shape[1] * 64

    @property
    def height(self):
        return self.cur.shape[0]

    def crop(self, x0, y0, x1, y1) -> "Region":
        """Crop (word-aligned in x) to cover [x0,x1) x [y0,y1)."""
        w0 = (x0 - self.ox) // 64
        w1 = -((self.ox - x1) // 64)
        r0 = y0 - self.oy
        r1 = y1 - self.oy
        assert w0 >= 0 and r0 >= 0 and w1 <= self.cur.shape[1] and r1 <= self.cur.shape[0], \
            ("crop outside region", (x0, y0, x1, y1), (self.ox, self.oy, self.width, self.height))
        s = (slice(r0, r1), slice(w0, w1))
        return Region(self.ox + 64 * w0, y0, self.cur[s].copy(), self.prev[s].copy(),
                      self.par[s].copy(), self.hb[s].copy())

    def step(self, n: int = 1):
        for _ in range(n):
            nxt = life.step(self.cur)
            self.hb = self.par
            self.par = self.par ^ (nxt & ~self.cur)
            self.prev = self.cur
            self.cur = nxt

    def bits(self, arr, x0, y0, x1, y1) -> np.ndarray:
        """uint8 0/1 array of ``arr`` (cur/prev/hb) over [x0,x1) x [y0,y1)."""
        u = life.unpack(arr[y0 - self.oy:y1 - self.oy])
        return u[:, x0 - self.ox:x1 - self.ox]


def neighbourhood_keys(cur_pad: np.ndarray) -> np.ndarray:
    """cur_pad: (h+2, w+2) uint8 -> (h, w) 9-bit neighbourhood codes."""
    h, w = cur_pad.shape[0] - 2, cur_pad.shape[1] - 2
    k = np.zeros((h, w), np.int32)
    for dy in range(3):
        for dx in range(3):
            k = (k << 1) | cur_pad[dy:dy + h, dx:dx + w]
    return k


def contexts(reg: Region, x0, y0, x1, y1) -> np.ndarray:
    """Context ids of the cells [x0,x1) x [y0,y1) of a region."""
    c = reg.bits(reg.cur, x0 - 1, y0 - 1, x1 + 1, y1 + 1).astype(np.int32)
    p = reg.bits(reg.prev, x0, y0, x1, y1).astype(np.int32)
    h = reg.bits(reg.hb, x0, y0, x1, y1).astype(np.int32)
    return (neighbourhood_keys(c) << 2) | p | (h << 1)


class World:
    def __init__(self, db: TileDB, top_size: int = 256):
        self.db = db
        self.K = db.K
        self.top_size = top_size
        self._tile_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.chain = None
        if "chain" in db.extra:
            self._init_chain(db.extra["chain"])
        self.set_pattern(np.zeros((top_size, top_size), np.uint8))

    # ------------------------------------------------------------ modes
    def set_pattern(self, cells: np.ndarray):
        """Pattern mode with a fresh top-level universe (torus)."""
        self.mode = "pattern"
        n = self.top_size
        top = np.zeros((n, n), np.uint8)
        h, w = min(n, cells.shape[0]), min(n, cells.shape[1])
        top[:h, :w] = cells[:h, :w]
        self.top_hist = [(top, top.copy(), np.zeros_like(top))]
        self.G = 0
        self.off = 0
        self.digits = {}           # level (< 0) -> digit

    def set_tower(self):
        if self.chain is None:
            raise RuntimeError("database has no self-similar chain")
        self.mode = "tower"
        self.off = self.chain_off
        self.digits = {}

    @property
    def top_level(self):
        return 0 if self.mode == "pattern" else None

    def digit(self, L):
        if L in self.digits:
            return self.digits[L]
        return self.gstar if self.mode == "tower" else 0

    # ------------------------------------------------------------ time
    def advance(self, L: int, n: int):
        """Advance level L by n generations (carrying into the parents)."""
        while n:
            if self.mode == "pattern" and L >= 0:
                for _ in range(n):
                    self._top_step()
                return
            d = self.digit(L) + n
            self.digits[L] = d % PERIOD
            n = d // PERIOD
            L += 1

    @property
    def top_cur(self):
        return self.top_hist[-1][0]

    def _top_step(self):
        c, p, h = self.top_hist[-1]
        nb = sum(np.roll(np.roll(c, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy)
        nxt = ((nb == 3) | ((nb == 2) & (c == 1))).astype(np.uint8)
        self.top_hist = [self.top_hist[-1], (nxt, c, h ^ ((1 - p) & c))]
        self.G += 1

    def edit_top(self, x, y, value):
        """Edit the top universe; time restarts so that every level stays exact."""
        n = self.top_size
        c = self.top_cur.copy()
        c[y % n, x % n] = value
        self.set_pattern(c)

    # ------------------------------------------------------------ tiles
    def _tile(self, root: int) -> np.ndarray:
        t = self._tile_cache.get(root)
        if t is None:
            t = self.db.expand(root)
            self._tile_cache[root] = t
            if len(self._tile_cache) > 384:
                self._tile_cache.popitem(last=False)
        else:
            self._tile_cache.move_to_end(root)
        return t

    def roots(self, ctx: np.ndarray, c: int, which: int = 0) -> np.ndarray:
        return self.db.roots[ctx, c, which]

    # ------------------------------------------------------------ tower chain
    def _init_chain(self, chain):
        ctx, c, px, py = (int(v) for v in chain)
        self.chain = (ctx, c, px, py)
        self.chain_ctx, self.chain_c = ctx, c
        self.gstar = c * self.K + 1
        self.chain_off = (px, py)
        t0 = self.db.tile(ctx, c, 0)
        p0 = self.db.tile(ctx, c, 1)
        t1 = life.step(t0)
        R = 32           # base box: cells within +-R of the chain cell
        x0 = (px - R) // 64 * 64
        x1 = -((-(px + R)) // 64) * 64
        s = (slice(py - R, py + R), slice(x0 // 64, x1 // 64))
        self.base = Region(x0 - px, -R, t1[s].copy(), t0[s].copy(),
                           (p0[s] ^ (t1[s] & ~t0[s])), p0[s].copy())
        self.base_box = R - 2
        cc = contexts(self.base, 0, 0, 1, 1)[0, 0]
        assert cc == ctx, "chain is not self-similar"

    # ------------------------------------------------------------ regions
    def region(self, L: int, x0: int, y0: int, x1: int, y1: int, digits=None) -> Region:
        """Exact state of level L over [x0,x1) x [y0,y1) at the current time."""
        dg = self.digits if digits is None else digits
        return self._region(L, x0, y0, x1, y1, dg)

    def _dig(self, dg, L):
        if L in dg:
            return dg[L]
        return self.gstar if self.mode == "tower" else 0

    def _region(self, L, x0, y0, x1, y1, dg) -> Region:
        if self.mode == "pattern" and L >= 0:
            return self._top_region(x0, y0, x1, y1, dg)
        if self.mode == "tower":
            touched = max([k for k, v in dg.items() if v != self.gstar], default=-10**9)
            b = self.base_box
            if L > touched and -b <= x0 and x1 <= b and -b <= y0 and y1 <= b:
                return self.base
        g = self._dig(dg, L)
        if g == 0:
            prev = self._borrow(dg, L)
            if prev is None:            # the very beginning of time: prev = cur
                reg = self._from_tiles(L, x0, y0, x1, y1, dg, 0, 0)
                reg.prev = reg.cur.copy()
                reg.hb = np.zeros_like(reg.cur)
                reg.par = np.zeros_like(reg.cur)
                return reg
            reg = self._region(L, x0 - 1, y0 - 1, x1 + 1, y1 + 1, prev)
            reg = reg.crop(x0 - 1, y0 - 1, x1 + 1, y1 + 1)
            reg.step(1)
            reg.par = np.zeros_like(reg.cur)
            reg.hb = np.zeros_like(reg.cur)
            return reg
        c = (g - 1) // self.K
        r = g - c * self.K
        reg = self._from_tiles(L, x0, y0, x1, y1, dg, c, r)
        reg.step(r)
        return reg

    def _borrow(self, dg, L):
        d = dict(dg)
        k = L
        while True:
            if self.mode == "pattern" and k >= 0:
                if self.G == 0 or d.get("T", 0):
                    return None
                d["T"] = 1
                return d
            v = self._dig(d, k)
            if v > 0:
                d[k] = v - 1
                return d
            d[k] = PERIOD - 1
            k += 1

    def _top_region(self, x0, y0, x1, y1, dg):
        n = self.top_size
        back = dg.get("T", 0) if isinstance(dg, dict) else 0
        cur, prev, hb = self.top_hist[-1 - back]
        xs0 = x0 // 64 * 64
        xs1 = -((-x1) // 64) * 64
        ys = np.arange(y0, y1) % n
        xs = np.arange(xs0, xs1) % n

        def pk(a):
            return life.pack(a[np.ix_(ys, xs)])
        return Region(xs0, y0, pk(cur), pk(prev), np.zeros((y1 - y0, (xs1 - xs0) // 64), np.uint64), pk(hb))

    def _from_tiles(self, L, x0, y0, x1, y1, dg, c, r) -> Region:
        """Assemble checkpoint-c tiles covering [x0-r, x1+r) x [y0-r, y1+r)."""
        ox, oy = self.off if isinstance(self.off, tuple) else (self.off, self.off)
        px0 = (x0 - r + ox) // TILE
        px1 = (x1 - 1 + r + ox) // TILE + 1
        py0 = (y0 - r + oy) // TILE
        py1 = (y1 - 1 + r + oy) // TILE + 1
        par = self._region(L + 1, px0 - 1, py0 - 1, px1 + 1, py1 + 1, dg)
        ctx = contexts(par, px0, py0, px1, py1)
        rc = self.db.roots[ctx, c, 0]
        rp = self.db.roots[ctx, c, 1]
        # crop window inside the assembled tile block (tile-aligned origin)
        bx = px0 * TILE - ox
        by = py0 * TILE - oy
        cx0 = (x0 - r - bx) // 64
        cx1 = -((bx - (x1 + r)) // 64)
        cy0 = y0 - r - by
        cy1 = y1 + r - by
        nr, nw = cy1 - cy0, cx1 - cx0
        cur = np.empty((nr, nw), np.uint64)
        pa = np.empty((nr, nw), np.uint64)
        for j in range(py1 - py0):
            ty0, ty1 = j * TILE, (j + 1) * TILE
            a0, a1 = max(ty0, cy0), min(ty1, cy1)
            if a0 >= a1:
                continue
            for i in range(px1 - px0):
                tx0, tx1 = i * WPT, (i + 1) * WPT
                b0, b1 = max(tx0, cx0), min(tx1, cx1)
                if b0 >= b1:
                    continue
                dst = (slice(a0 - cy0, a1 - cy0), slice(b0 - cx0, b1 - cx0))
                src = (slice(a0 - ty0, a1 - ty0), slice(b0 - tx0, b1 - tx0))
                cur[dst] = self._tile(int(rc[j, i]))[src]
                pa[dst] = self._tile(int(rp[j, i]))[src]
        z = np.zeros_like(cur)
        return Region(bx + 64 * cx0, y0 - r, cur, z.copy(), pa, z)
