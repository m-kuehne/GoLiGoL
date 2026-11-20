"""Built-in starting patterns for the top level (RLE)."""
import numpy as np

from .life import parse_rle

PRESETS = [
    ("Glider", "bob$2bo$3o!"),
    ("Gosper glider gun", "24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$10bo5bo7bo$11bo3bo$12b2o!"),
    ("R-pentomino", "b2o$2o$bo!"),
    ("Acorn", "bo5b$3bo3b$2o2b3o!"),
    ("Pulsar", "2b3o3b3o2$o4bobo4bo$o4bobo4bo$o4bobo4bo$2b3o3b3o2$2b3o3b3o$o4bobo4bo$o4bobo4bo$o4bobo4bo2$2b3o3b3o!"),
    ("Lightweight spaceship", "b4o$o3bo$4bo$o2bo!"),
    ("Pentadecathlon", "2bo4bo$2ob4ob2o$2bo4bo!"),
    ("Diehard", "6bo$2o$bo3b3o!"),
    ("Blinker", "3o!"),
]


def cells_to_array(cells, size: int) -> np.ndarray:
    """Centre a list of (x, y) cells in a size x size array."""
    a = np.zeros((size, size), np.uint8)
    if not cells:
        return a
    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    w = max(xs) - min(xs) + 1; h = max(ys) - min(ys) + 1
    ox = size // 2 - w // 2 - min(xs); oy = size // 2 - h // 2 - min(ys)
    for x, y in cells:
        a[(y + oy) % size, (x + ox) % size] = 1
    return a


def preset(i: int, size: int) -> np.ndarray:
    return cells_to_array(parse_rle(PRESETS[i][1]), size)


def soup(size: int, extent: int = 48, density: float = 0.35, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros((size, size), np.uint8)
    o = size // 2 - extent // 2
    a[o:o + extent, o:o + extent] = rng.random((extent, extent)) < density
    return a
