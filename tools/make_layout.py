"""Find a torus of meta-cells that contains every metapixel context.

A context is (3x3 neighbourhood of current states, own previous state, own
honeybit) = 2**11 = 2048 possibilities.  We anneal a binary NX x NY torus so
that each of the 512 neighbourhoods occurs at least 4 times, then hand out the
four (prev, honeybit) combinations among those occurrences.

usage: make_layout.py NX NY seed out_layout.bin
Writes one byte per tile (row-major): variant = prev + 2*cur + 4*honeybit.
"""
import math
import random
import sys

import numpy as np

NX, NY, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
random.seed(seed)
g = [[random.getrandbits(1) for _ in range(NX)] for _ in range(NY)]


def key(y, x):
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            k = k * 2 + g[(y + dy) % NY][(x + dx) % NX]
    return k


cnt = [0] * 512
for y in range(NY):
    for x in range(NX):
        cnt[key(y, x)] += 1
score = sum(min(c, 4) for c in cnt)
T = 1.0
ITERS = 6_000_000
for it in range(ITERS):
    y = random.randrange(NY); x = random.randrange(NX)
    cells = [((y + dy) % NY, (x + dx) % NX) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    old = [key(a, b) for a, b in cells]
    d = 0
    for k in old:
        if cnt[k] <= 4: d -= 1
        cnt[k] -= 1
    g[y][x] ^= 1
    new = [key(a, b) for a, b in cells]
    for k in new:
        if cnt[k] < 4: d += 1
        cnt[k] += 1
    if d >= 0 or random.random() < math.exp(d / T):
        score += d
    else:
        for k in new: cnt[k] -= 1
        g[y][x] ^= 1
        for k in old: cnt[k] += 1
    T = max(0.05, T * (1 - 4.0 / ITERS))
    if score == 2048:
        break
print("contexts covered:", score, "of 2048 after", it, "iterations", file=sys.stderr)

rng = np.random.default_rng(seed)
var = np.zeros((NY, NX), np.uint8)
seen = {}
for y in range(NY):
    for x in range(NX):
        k = key(y, x)
        i = seen.get(k, 0); seen[k] = i + 1
        ph = i if i < 4 else int(rng.integers(4))       # (prev, hb) combo
        prev, hb = ph & 1, ph >> 1
        var[y, x] = prev + 2 * g[y][x] + 4 * hb
var.tofile(out)
