"""Build the OTCA metapixel OFF / ON tiles (Brice Due, 2006).

The construction recipe is the one in Golly's ``metafier.py`` (Dave Greene /
Andrew Trevorrow).  That script is treated purely as *data*: it is parsed with
``ast`` and only the handful of constructs the recipe needs (pattern literals,
translation / rotation, union, "advance n generations", simple ``for`` loops)
are interpreted here -- nothing in it is executed as Python.

usage: build_metacell.py path/to/metafier.py outdir
Output: otca_off.npy / otca_on.npy -- arrays of (x, y) live cells; the tile
occupies [0,2048) x [0,2048) (a few cells spill over into the neighbours).
"""
import ast
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from goligol import life  # noqa: E402

TRANSFORMS = dict(
    identity=(1, 0, 0, 1), flip=(-1, 0, 0, -1), flip_x=(-1, 0, 0, 1),
    flip_y=(1, 0, 0, -1), swap_xy=(0, 1, 1, 0), swap_xy_flip=(0, -1, -1, 0),
    rcw=(0, -1, 1, 0), rccw=(0, 1, -1, 0))


def evolve(cells, n):
    if n == 0 or not cells:
        return set(cells)
    arr = np.array(list(cells))
    m = n + 2
    x0 = arr[:, 0].min() - m; y0 = arr[:, 1].min() - m
    w = arr[:, 0].max() - x0 + m + 1; h = arr[:, 1].max() - y0 + m + 1
    bits = np.zeros((h, w), np.uint8)
    bits[arr[:, 1] - y0, arr[:, 0] - x0] = 1
    g = life.run(life.pack(bits), n)
    yy, xx = np.nonzero(life.unpack(g, w))
    return set(zip((xx + x0).tolist(), (yy + y0).tolist()))


class Pattern:
    def __init__(self, P=(), x0=0, y0=0, A=TRANSFORMS["identity"]):
        if isinstance(P, str):
            P = life.parse_rle(P)
        elif isinstance(P, Pattern):
            P = P.cells
        a, b, c, d = A
        self.cells = {(x0 + x * a + y * b, y0 + x * c + y * d) for (x, y) in P}

    def add(self, o):
        r = Pattern(); r.cells = self.cells | o.cells; return r

    def advance(self, n):
        r = Pattern(); r.cells = evolve(self.cells, n); return r


class Interp:
    """Interprets the tiny subset of Python used by the tile recipe."""

    def __init__(self):
        self.env = dict(TRANSFORMS)

    def run(self, stmts):
        for s in stmts:
            if isinstance(s, ast.Assign):
                assert len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                self.env[s.targets[0].id] = self.ev(s.value)
            elif isinstance(s, ast.AugAssign):
                assert isinstance(s.op, ast.Add) and isinstance(s.target, ast.Name)
                self.env[s.target.id] = self.plus(self.env[s.target.id], self.ev(s.value))
            elif isinstance(s, ast.For):
                assert isinstance(s.target, ast.Name)
                for v in self.ev(s.iter):
                    self.env[s.target.id] = v
                    self.run(s.body)
            elif isinstance(s, (ast.If, ast.Expr)):
                continue        # file caching / status messages: not needed
            else:
                raise ValueError("unsupported statement " + ast.dump(s)[:80])

    @staticmethod
    def plus(a, b):
        if isinstance(a, Pattern) and isinstance(b, Pattern):
            return a.add(b)
        if isinstance(a, (str, int)) and type(a) is type(b):
            return a + b
        raise ValueError("bad +")

    def ev(self, e):
        if isinstance(e, ast.Constant) and isinstance(e.value, (str, int)):
            return e.value
        if isinstance(e, ast.Name):
            return self.env[e.id]
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            return -self.ev(e.operand)
        if isinstance(e, ast.BinOp):
            a, b = self.ev(e.left), self.ev(e.right)
            if isinstance(e.op, ast.Add):
                return self.plus(a, b)
            if isinstance(e.op, ast.Mult) and isinstance(b, int) and isinstance(a, (str, int)):
                return a * b
            if isinstance(e.op, ast.Mod) and isinstance(a, int) and isinstance(b, int):
                return a % b
            raise ValueError("unsupported operator")
        if isinstance(e, ast.Subscript):
            p, n = self.ev(e.value), self.ev(e.slice)
            assert isinstance(p, Pattern) and isinstance(n, int)
            return p.advance(n)
        if isinstance(e, ast.Call):
            assert not e.keywords
            args = [self.ev(a) for a in e.args]
            if isinstance(e.func, ast.Name) and e.func.id == "pattern":
                return Pattern(*args)
            if isinstance(e.func, ast.Name) and e.func.id == "range":
                assert all(isinstance(a, int) for a in args)
                return range(*args)
            f = self.ev(e.func)
            assert isinstance(f, Pattern)
            return Pattern(f, *args)
        raise ValueError("unsupported expression " + ast.dump(e)[:80])


def build(metafier_path, bvals="3", svals="23"):
    tree = ast.parse(open(metafier_path).read())
    recipe = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and node.orelse:
            first = ast.Module(body=node.orelse[:3], type_ignores=[])
            if any(isinstance(n, ast.Name) and n.id == "LWSS" for n in ast.walk(first)):
                recipe = node.orelse
                break
    assert recipe is not None, "tile recipe not found"
    it = Interp()
    it.run(recipe)
    Brle = Srle = "b10$b10$b26$b10$b10$b26$b10$b10$b!"
    for ch in bvals:
        ind = 32 - int(ch) * 4
        Brle = Brle[:ind] + "o" + Brle[ind + 1:]
    for ch in svals:
        ind = 32 - int(ch) * 4
        Srle = Srle[:ind] + "o" + Srle[ind + 1:]
    rule_bits = Pattern(Brle, 148, 1404).add(Pattern(Srle, 162, 1406))
    off = Pattern(it.env["OFFcell"].add(rule_bits), -5, -5)
    on = Pattern(it.env["ONcell"].add(rule_bits), -5, -5)
    return off.cells, on.cells


if __name__ == "__main__":
    src, outdir = sys.argv[1], sys.argv[2]
    off, on = build(src)
    for name, c in (("off", off), ("on", on)):
        arr = np.array(sorted(c), dtype=np.int32)
        print(name, "population", len(arr), "bbox", arr.min(0), arr.max(0))
        np.save(os.path.join(outdir, f"otca_{name}.npy"), arr)
