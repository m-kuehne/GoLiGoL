import numpy as np

from goligol import life


def naive(b, torus=False):
    if torus:
        n = sum(np.roll(np.roll(b, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy)
    else:
        p = np.pad(b, 1)
        n = sum(p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy)
    return ((n == 3) | ((n == 2) & (b == 1))).astype(np.uint8)


def test_step_matches_naive():
    rng = np.random.default_rng(1)
    for torus in (False, True):
        b = (rng.random((96, 192)) < 0.4).astype(np.uint8)
        g = life.pack(b)
        for _ in range(25):
            g = life.step(g, torus)
            b = naive(b, torus)
            assert (life.unpack(g) == b).all()


def test_rle_roundtrip():
    cells = [(0, 0), (5, 0), (6, 0), (2, 3), (0, 9), (70, 2)]
    assert sorted(life.parse_rle(life.to_rle(cells))) == sorted(cells)
    assert sorted(life.parse_rle("x = 3, y = 3\nbob$2bo$3o!")) == [(0, 2), (1, 0), (1, 2), (2, 1), (2, 2)]
