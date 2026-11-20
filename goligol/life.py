"""Bit-packed Conway's Game of Life on numpy uint64 arrays.

A grid is a 2-D array of shape (H, W) uint64; bit ``i`` of word ``j`` in a row
is the cell at x = 64*j + i.  ``step`` treats everything outside the array as
dead, so callers that need exact results keep a margin of at least one cell
per generation around the region they care about (information in Life cannot
travel faster than one cell per generation).
"""
from __future__ import annotations

import numpy as np

U64 = np.uint64
_ONE = U64(1)
_S63 = U64(63)


def step(g: np.ndarray, torus: bool = False) -> np.ndarray:
    """Return the next generation of packed grid ``g`` (B3/S23)."""
    w = g.shape[1]
    # horizontal neighbours
    left = g << _ONE            # cell x-1 moved to x
    right = g >> _ONE           # cell x+1 moved to x
    if w > 1:
        left[:, 1:] |= g[:, :-1] >> _S63
        right[:, :-1] |= g[:, 1:] << _S63
    if torus:
        left[:, 0] |= g[:, -1] >> _S63
        right[:, -1] |= g[:, 0] << _S63
    # per-row 3-cell sums (including the centre): s0 = ones bit, s1 = twos bit
    s0 = left ^ right
    s1 = left & right
    s1 |= s0 & g
    s0 ^= g
    del left, right
    # vertical neighbours of the row sums
    u0 = np.empty_like(s0); u1 = np.empty_like(s1)
    d0 = np.empty_like(s0); d1 = np.empty_like(s1)
    u0[1:] = s0[:-1]; u1[1:] = s1[:-1]
    d0[:-1] = s0[1:]; d1[:-1] = s1[1:]
    if torus:
        u0[0] = s0[-1]; u1[0] = s1[-1]; d0[-1] = s0[0]; d1[-1] = s1[0]
    else:
        u0[0] = 0; u1[0] = 0; d0[-1] = 0; d1[-1] = 0
    # ones column: t0 = parity, c0 = carry into the twos column
    t0 = u0 ^ d0
    c0 = u0 & d0
    c0 |= t0 & s0
    t0 ^= s0
    # twos column: V = u1 + s1 + d1 + c0  (0..4)
    p = u1 ^ s1
    g1 = u1 & s1
    q = d1 ^ c0
    g2 = d1 & c0
    v0 = p ^ q
    p &= q
    g1 ^= g2
    g1 ^= p                     # v1 (twos bit of V)
    # total (incl. centre) == 3  <=> t0 & V==1 ; == 4 <=> ~t0 & V==2
    born = t0 & v0
    born &= ~g1
    keep = g1 & ~v0
    keep &= ~t0
    keep &= g
    born |= keep
    return born


def run(g: np.ndarray, n: int, torus: bool = False) -> np.ndarray:
    for _ in range(n):
        g = step(g, torus)
    return g


def pack(bits: np.ndarray) -> np.ndarray:
    """(H, W) array of 0/1  ->  packed (H, ceil(W/64)) uint64."""
    h, w = bits.shape
    ww = (w + 63) // 64
    if w == ww * 64:
        buf = np.ascontiguousarray(bits, dtype=np.uint8)
    else:
        buf = np.zeros((h, ww * 64), dtype=np.uint8)
        buf[:, :w] = bits
    by = np.packbits(buf, axis=1, bitorder="little")
    return by.view("<u8").reshape(h, ww).copy()


def unpack(g: np.ndarray, w: int | None = None) -> np.ndarray:
    """packed -> (H, W) uint8 array of 0/1."""
    h, ww = g.shape
    by = np.ascontiguousarray(g.astype("<u8", copy=False)).view(np.uint8).reshape(h, ww * 8)
    bits = np.unpackbits(by, axis=1, bitorder="little")
    return bits if w is None else bits[:, :w]


def parse_rle(text: str):
    """Parse RLE (or a bare RLE body).  Returns list of (x, y) live cells."""
    body = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("x") and "=" in s:
            continue
        body.append(s)
    data = "".join(body)
    cells = []
    x = y = 0
    n = 0
    for ch in data:
        if ch.isdigit():
            n = n * 10 + ord(ch) - 48
        elif ch == "b" or ch == ".":
            x += n or 1; n = 0
        elif ch == "$":
            y += n or 1; x = 0; n = 0
        elif ch == "!":
            break
        elif ch.isalpha():
            k = n or 1
            cells.extend((x + i, y) for i in range(k))
            x += k; n = 0
    return cells


def to_rle(cells, rule: str = "B3/S23") -> str:
    """Encode an iterable of (x, y) live cells as RLE text."""
    cells = sorted(set(cells), key=lambda c: (c[1], c[0]))
    if not cells:
        return f"x = 0, y = 0, rule = {rule}\n!\n"
    x0 = min(c[0] for c in cells); y0 = min(c[1] for c in cells)
    w = max(c[0] for c in cells) - x0 + 1
    h = max(c[1] for c in cells) - y0 + 1
    out = []

    def emit(n, ch):
        if n > 0:
            out.append((str(n) if n > 1 else "") + ch)

    cy = y0; cx = x0; run_len = 0
    for (x, y) in cells:
        if y != cy:
            emit(run_len, "o"); run_len = 0
            emit(y - cy, "$"); cy = y; cx = x0
        if x != cx:
            emit(run_len, "o"); run_len = 0
            emit(x - cx, "b"); cx = x
        run_len += 1; cx += 1
    emit(run_len, "o"); out.append("!")
    lines = [f"x = {w}, y = {h}, rule = {rule}"]
    cur = ""
    for tok in out:
        if len(cur) + len(tok) > 70:
            lines.append(cur); cur = ""
        cur += tok
    lines.append(cur)
    return "\n".join(lines) + "\n"
