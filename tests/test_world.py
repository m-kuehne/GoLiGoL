"""Consistency tests of the metapixel tower (need the bundled database)."""
import pytest

from goligol.patterns import preset
from goligol.tiledb import PERIOD, TileDB, default_path
from goligol.world import World


@pytest.fixture(scope="module")
def db():
    return TileDB(default_path())


def _bits(reg, arr, rect):
    return reg.bits(getattr(reg, arr), *rect)


def test_database_shape(db):
    assert db.roots.shape == (2048, PERIOD // db.K, 2)
    assert db.have.all()
    # phase-0 tiles depend only on (prev, cur, honeybit)
    for v in range(8):
        r = {int(db.roots[c, 0, 0]) for c in range(2048)
             if (c & 1) + 2 * ((c >> 6) & 1) + 4 * ((c >> 1) & 1) == v}
        assert len(r) == 1


@pytest.mark.parametrize("level,start", [(-1, PERIOD - 3), (-1, 5000), (-2, PERIOD - 2), (-2, 12)])
def test_stepping_equals_direct(db, level, start):
    """Running Life on a region == asking the tower for the later time,
    also across a period boundary (where the parent cell changes state)."""
    w = World(db, top_size=64)
    w.set_pattern(preset(0, 64))          # glider
    w.advance(0, 7)
    w.digits[-1] = 20000 if level == -2 else start
    if level == -2:
        w.digits[-2] = start
    x0, y0 = 32 * 2048 + 900, 32 * 2048 + 950
    if level == -2:
        x0, y0 = x0 * 2048 + 700, y0 * 2048 + 1200
    rect = (x0, y0, x0 + 300, y0 + 200)
    n = 5
    big = w.region(level, rect[0] - n, rect[1] - n, rect[2] + n, rect[3] + n)
    big = big.crop(rect[0] - n, rect[1] - n, rect[2] + n, rect[3] + n)
    big.step(n)
    w.advance(level, n)
    direct = w.region(level, *rect)
    for arr in ("cur", "prev"):
        assert (_bits(big, arr, rect) == _bits(direct, arr, rect)).all(), arr


def test_tower_self_similar(db):
    w = World(db)
    w.set_tower()
    a = w.region(3, -300, -200, 300, 200)
    b = w.region(11, -300, -200, 300, 200)
    rect = (-300, -200, 300, 200)
    assert (_bits(a, "cur", rect) == _bits(b, "cur", rect)).all()
    # the hard-coded chain neighbourhood equals what the parent tiles produce
    r = w.region(4, -40, -40, 40, 40)
    inner = (-28, -28, 28, 28)
    for arr in ("cur", "prev", "hb"):
        assert (_bits(r, arr, inner) == _bits(w.base, arr, inner)).all()
