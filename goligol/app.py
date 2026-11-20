"""GoLiGoL - interactive, infinitely zoomable Game of Life in Game of Life."""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from . import life
from .patterns import PRESETS, preset, soup, cells_to_array
from .tiledb import PERIOD, TILE, TileDB, default_path
from .world import World

ALPHA_START, ALPHA_FULL = 3.0, 24.0
DISPLAY_SWITCH = 6600   # phase at which an OTCA metapixel's display has mostly switched
AUTO_A, AUTO_EXP = 181.0, math.log(PERIOD) / math.log(TILE)   # speed continuity across levels

HELP = """\
Mouse wheel / PgUp PgDn   zoom (towards the cursor) - zoom in forever
Left drag                 pan
Right drag                draw / erase cells (top level of pattern mode)
Z / X                     auto-zoom in / out (toggle)      [ / ]  zoom rate
Space                     pause / run          .  single generation
+ / -                     faster / slower      A  auto speed (follows zoom)
1 .. 9                    load preset          R  random soup      C  clear
T                         toggle infinite tower mode (zoom out forever)
Ctrl+S                    save level-0 pattern to saves/ (pattern mode only)
Ctrl+O                    open RLE (or drop an .rle file on the window)
G                         grid lines on/off    Home  reset view
F11                       fullscreen           H  this help     Esc  quit
"""


def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


class Camera:
    """Position = xi + xf (Python int + float in [0,1)) in cells of level V."""

    def __init__(self, V=0, x=0.0, y=0.0, s=8.0):
        self.V = V
        self.xi, self.xf = int(math.floor(x)), x - math.floor(x)
        self.yi, self.yf = int(math.floor(y)), y - math.floor(y)
        self.s = s

    def move(self, dx, dy):
        self.xf += dx; self.yf += dy
        fx, fy = math.floor(self.xf), math.floor(self.yf)
        self.xi += int(fx); self.xf -= fx
        self.yi += int(fy); self.yf -= fy

    def cell_at(self, sx, sy, W, H):
        fx = self.xf + (sx - W / 2) / self.s
        fy = self.yf + (sy - H / 2) / self.s
        return self.xi + math.floor(fx), self.yi + math.floor(fy)

    def to_child(self, off):
        ox, oy = off
        for a in ("x", "y"):
            i, f = getattr(self, a + "i"), getattr(self, a + "f")
            o = ox if a == "x" else oy
            g = f * TILE
            gi = math.floor(g)
            setattr(self, a + "i", i * TILE - o + gi)
            setattr(self, a + "f", g - gi)
        self.V -= 1
        self.s /= TILE

    def to_parent(self, off):
        ox, oy = off
        for a in ("x", "y"):
            i, f = getattr(self, a + "i"), getattr(self, a + "f")
            o = ox if a == "x" else oy
            p = (i + o) // TILE
            rem = i + o - p * TILE
            setattr(self, a + "i", p)
            setattr(self, a + "f", (rem + f) / TILE)
        self.V += 1
        self.s *= TILE


class Engine:
    """Keeps the exact state of the viewed level in sync with time and camera."""

    def __init__(self, world: World, renderer):
        self.w = world
        self.r = renderer
        self.K = world.K
        self.cam = Camera()
        self.acc = 0.0            # fraction of the current view-level generation
        self.paused = False
        self.auto = True
        self.mult = 1.0
        self.manual = 8.0
        self.grid = False
        self.R = None
        self.R_rect = None
        self.steps = 0
        self.top_dirty = True
        self.ov_key = None
        self.ov_entries = {}
        self.ov_res = 0
        self.ov_last = 0.0
        self.zoom_rate = 0.0
        self.zoom_speed = 1.6
        self.status = ""
        self.last_pattern = None

    # ------------------------------------------------------------ helpers
    @property
    def off(self):
        o = self.w.off
        return o if isinstance(o, tuple) else (o, o)

    def is_top(self):
        return self.w.mode == "pattern" and self.cam.V >= 0

    def speed(self):
        if self.auto:
            v = AUTO_A * self.cam.s ** (-AUTO_EXP) * self.mult
        else:
            v = self.manual * self.mult
        if self.is_top():
            v = min(v, 40.0)
        return v

    def reset_view(self):
        self.R = None
        self.top_dirty = True
        self.ov_key = None
        self.ov_entries = {}

    # ------------------------------------------------------------ modes
    def load_pattern(self, cells: np.ndarray, name=""):
        n = self.w.top_size
        if cells.shape[0] > n or cells.shape[1] > n:
            n = min(4096, -(-max(cells.shape) // 64) * 64 + 64)
            self.w.top_size = n
        a = np.zeros((n, n), np.uint8)
        a[:cells.shape[0], :cells.shape[1]] = cells
        self.w.set_pattern(a)
        self.last_pattern = a
        self.acc = 0.0
        self.cam = Camera(0, n / 2, n / 2, 1.0)
        self.fit_top()
        ys, xs = np.nonzero(a)
        if len(xs):
            bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            self.cam = Camera(0, (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2, 1.0)
            W, H = self.screen
            self.cam.s = max(0.9 * min(W, H) / n, min(0.5 * W / bw, 0.5 * H / bh, 20.0))
        self.reset_view()
        self.status = f"loaded {name}" if name else ""

    def fit_top(self, W=None, H=None):
        W = W or self.screen[0]; H = H or self.screen[1]
        self.cam.s = max(1.0, 0.9 * min(W, H) / self.w.top_size)

    def tower(self):
        if self.w.mode == "tower":
            self.load_pattern(self.last_pattern if self.last_pattern is not None else
                              preset(1, self.w.top_size), "pattern mode")
            return
        self.w.set_tower()
        self.acc = 0.0
        self.cam = Camera(0, 0.5, 0.5, 6.0)
        self.reset_view()
        self.status = "infinite tower: every level is the same metapixel chain - zoom out forever"

    # ------------------------------------------------------------ zoom
    def zoom(self, factor, sx, sy):
        W, H = self.screen
        c = self.cam
        dx, dy = (sx - W / 2), (sy - H / 2)
        c.move(dx / c.s, dy / c.s)
        c.s *= factor
        c.move(-dx / c.s, -dy / c.s)
        self.normalize()

    def normalize(self):
        c = self.cam
        while c.s >= TILE:
            f = self.acc * PERIOD
            d = int(f)
            self.acc = f - d
            self.w.digits[c.V - 1] = d
            c.to_child(self.off)
            self.reset_view()
        while c.s < 1.0 and not self.is_top():
            d = self.w.digit(c.V)
            self.acc = (d + self.acc) / PERIOD
            self.w.digits.pop(c.V, None)
            c.to_parent(self.off)
            self.reset_view()
        if self.is_top():
            smin = min(1.0, 0.25 * min(self.screen) / self.w.top_size)
            c.s = max(c.s, smin)

    # ------------------------------------------------------------ time
    def advance(self, n: int):
        if n <= 0:
            return
        c = self.cam
        if self.is_top():
            self.w.advance(c.V, n)
            self.top_dirty = True
            return
        if self.R is not None:
            cells = self.R.cur.size * 64
            allowed = max(1, int(0.03 / (cells * 0.5e-9 + 1e-6)))
            vx0, vy0, vx1, vy1 = self.visible_rect()
            rx0, ry0, rx1, ry1 = self.R_rect
            m = min(vx0 - rx0, vy0 - ry0, rx1 - vx1, ry1 - vy1) - self.steps
            if n <= allowed and n < m:
                for _ in range(n):
                    self.R.step(1)
                    self.w.advance(c.V, 1)
                    self.steps += 1
                    if self.w.digit(c.V) == 0:
                        self.R.par[:] = 0
                        self.R.hb[:] = 0
                return
        self.w.advance(c.V, n)
        self.R = None

    def visible_rect(self):
        W, H = self.screen
        c = self.cam
        hw, hh = W / 2 / c.s, H / 2 / c.s
        return (c.xi + math.floor(c.xf - hw) - 2, c.yi + math.floor(c.yf - hh) - 2,
                c.xi + math.ceil(c.xf + hw) + 2, c.yi + math.ceil(c.yf + hh) + 2)

    def ensure_region(self):
        c = self.cam
        if self.is_top():
            if self.top_dirty or self.R is None:
                n = self.w.top_size
                cur, prev, hb = self.w.top_hist[-1]
                z = life.pack(np.zeros((n, n), np.uint8))
                from .world import Region
                self.R = Region(0, 0, life.pack(cur), life.pack(prev), z, life.pack(hb))
                self.top_dirty = False
                self.r.set_cells(self.R.cur, self.R.prev, self.R.hb)
                self.uploaded = True
            return
        vx0, vy0, vx1, vy1 = self.visible_rect()
        if self.R is not None:
            rx0, ry0, rx1, ry1 = self.R_rect
            k = self.steps
            if rx0 + k <= vx0 and ry0 + k <= vy0 and vx1 <= rx1 - k and vy1 <= ry1 - k:
                return
        # rebuild: snap the view level to a generation 1 (mod K) after a checkpoint
        g = self.w.digit(c.V)
        delta = (1 - g) % self.K
        if delta:
            self.w.advance(c.V, delta)
        M = self.K + 8 + min(512, int(256 / max(c.s, 0.5)))
        rect = (vx0 - M, vy0 - M, vx1 + M, vy1 + M)
        self.R = self.w.region(c.V, *rect)
        self.R_rect = rect
        self.steps = 0

    # ------------------------------------------------------------ children overlay
    def overlay(self, alpha):
        db = self.w.db
        lut = np.full(2048, -1, np.int32)
        if alpha <= 0:
            return lut
        c = self.cam
        res = int(min(TILE, max(4, 2 ** math.ceil(math.log2(max(c.s, 1.0))))))
        cp = min(db.M - 1, int(self.acc * PERIOD) // self.K)
        now = time.perf_counter()
        if res <= 32:
            key = (res, cp)
            if key != self.ov_key and (now - self.ov_last > 0.06 or self.ov_res != res):
                self.r.ensure_layers(res, 2048)
                imgs = db.density_many(db.roots[:, cp, 0], res)
                self.r.write_layers(0, imgs)
                self.ov_key, self.ov_last, self.ov_res = key, now, res
                self.ov_entries = {}
            return np.arange(2048, dtype=np.int32) if self.ov_res == res else lut
        # large cells: only the contexts that are visible
        cap = int(min(2048, (1 << 27) // (res * res * 2)))
        if self.ov_res != res or not isinstance(self.ov_entries, dict) or self.ov_key is None:
            self.r.ensure_layers(res, cap)
            self.ov_entries = {}
            self.ov_res = res
            self.ov_key = (res, -1)
        vx0, vy0, vx1, vy1 = self.visible_rect()
        ctx = self.visible_contexts(vx0 + 1, vy0 + 1, vx1 - 1, vy1 - 1)
        uniq, counts = np.unique(ctx, return_counts=True)
        uniq = uniq[np.argsort(-counts)]
        want = db.roots[uniq, cp, 0]
        budget = 0.025 if res <= 512 else 0.045
        t0 = time.perf_counter()
        used = {v[1] for v in self.ov_entries.values()}
        free = [i for i in range(cap) if i not in used]
        vis = set(uniq.tolist())
        # missing contexts first, then the stalest ones
        todo = [(0.0 if k not in self.ov_entries else self.ov_entries[k][2], k, r)
                for k, r in zip(uniq.tolist(), want.tolist())
                if k not in self.ov_entries or self.ov_entries[k][0] != r]
        todo.sort(key=lambda t: t[0])
        for _, k, root in todo:
            if time.perf_counter() - t0 > budget:
                break
            e = self.ov_entries.get(k)
            if e is None:
                if not free:
                    victim = next((kk for kk in self.ov_entries if kk not in vis), None)
                    if victim is None:
                        break
                    free.append(self.ov_entries.pop(victim)[1])
                layer = free.pop()
            else:
                layer = e[1]
            img = db.density_many([root], res)[0]
            self.r.write_layer(layer, img)
            self.ov_entries[k] = (root, layer, time.perf_counter())
        for k, (_, layer, _t) in self.ov_entries.items():
            lut[k] = layer
        return lut

    def visible_contexts(self, x0, y0, x1, y1):
        if self.is_top():
            n = self.w.top_size
            cur, prev, hb = (a.astype(np.int32) for a in self.w.top_hist[-1])
            k = np.zeros((n, n), np.int32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    k = (k << 1) | np.roll(np.roll(cur, -dy, 0), -dx, 1)
            ctx = (k << 2) | prev | (hb << 1)
            ys = np.arange(y0, y1) % n
            xs = np.arange(x0, x1) % n
            return ctx[np.ix_(ys, xs)].ravel()
        R = self.R
        x0 = max(x0, R.ox + 1); y0 = max(y0, R.oy + 1)
        x1 = min(x1, R.ox + R.cur.shape[1] * 64 - 1); y1 = min(y1, R.oy + R.cur.shape[0] - 1)
        from .world import contexts
        return contexts(R, x0, y0, x1, y1).ravel()

    # ------------------------------------------------------------ frame
    def frame(self, dt, screen):
        self.screen = screen
        c = self.cam
        if self.zoom_rate:
            self.zoom(self.zoom_speed ** (self.zoom_rate * dt), screen[0] / 2, screen[1] / 2)
        if not self.paused:
            self.acc += self.speed() * dt
            n = int(self.acc)
            self.acc -= n
            self.advance(n)
        self.ensure_region()
        if not self.is_top():
            self.r.set_cells(self.R.cur, self.R.prev, self.R.hb)
        alpha = smoothstep(ALPHA_START, ALPHA_FULL, c.s)
        lut = self.overlay(alpha)
        self.r.set_lut(lut)
        R = self.R
        if self.is_top():
            n = self.w.top_size
            cam = ((c.xi % n) + c.xf, (c.yi % n) + c.yf)
            size = (n, n)
            wrap = n
        else:
            cam = ((c.xi - R.ox) + c.xf, (c.yi - R.oy) + c.yf)
            size = (R.cur.shape[1] * 64, R.cur.shape[0])
            wrap = 0
        grid = smoothstep(3.0, 8.0, c.s) if self.grid else 0.0
        tint = 0.22 * (1.0 - smoothstep(48.0, 900.0, c.s))
        gain = 10.0 if c.s < 64 else 6.0
        # a metapixel shows its previous state until its display is redrawn mid-period
        tint_prev = self.acc * PERIOD < DISPLAY_SWITCH
        self.r.draw(screen, cam, c.s, size, alpha, c.V, grid, wrap, gain, tint, tint_prev)

    # ------------------------------------------------------------ HUD text
    def hud_lines(self):
        c = self.cam
        w = self.w
        lines = ["GoLiGoL - Game of Life in Game of Life      [H] help"]
        if w.mode == "pattern":
            lines.append(f"pattern mode: level 0 = your {w.top_size}x{w.top_size} torus, "
                         f"every cell is an OTCA metapixel of the level below")
            depth = f"level {c.V}" + ("  (top)" if c.V == 0 else f"  ({-c.V} metapixel levels below your pattern)")
        else:
            lines.append("infinite tower: every level is made of OTCA metapixels of the level below")
            depth = f"level {c.V}"
        lines.append(depth)
        mag = math.log2(c.s) / 11.0 - c.V
        lines.append(f"zoom: 1 cell = {c.s:.3g} px    (2048^{mag:+.2f} relative to level 0)")
        sp = self.speed()
        mode = "auto" if self.auto else "manual"
        st = "PAUSED" if self.paused else f"{sp:.3g} gen/s"
        lines.append(f"speed: {st} at this level ({mode}, x{self.mult:g})")
        parts = []
        if w.mode == "pattern":
            parts.append(f"L0 gen {w.G}")
            lo = c.V
            for L in range(-1, lo - 1, -1):
                parts.append(f"L{L} {w.digit(L)}")
        else:
            for L in range(c.V + 3, c.V - 1, -1):
                parts.append(f"L{L} {w.digit(L)}")
        child = int(self.acc * PERIOD)
        parts.append(f"L{c.V - 1} {child}")
        lines.append("time: " + " | ".join(parts[-7:]) + f"   (period {PERIOD})")
        if self.status:
            lines.append(self.status)
        return lines


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Game of Life in Game of Life, infinitely zoomable")
    ap.add_argument("rle", nargs="?", help="RLE pattern file to start with")
    ap.add_argument("--db", default=default_path(), help="context database (.npz)")
    ap.add_argument("--size", type=int, default=256, help="top level torus size (multiple of 64)")
    ap.add_argument("--tower", action="store_true", help="start in infinite tower mode")
    ap.add_argument("--window", default="1600x900")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--screenshot", help="render a few frames, save a PNG and exit")
    ap.add_argument("--script", help="(testing) semicolon-separated actions")
    args = ap.parse_args(argv)

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    import moderngl
    from .render import Renderer

    print("loading metapixel database ...", flush=True)
    t = time.time()
    db = TileDB(args.db)
    db._populations()
    print(f"  {len(db.blocks)} blocks, {db.M} checkpoints, {time.time() - t:.1f}s", flush=True)
    world = World(db, top_size=max(64, args.size // 64 * 64))

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    W, H = (int(v) for v in args.window.lower().split("x"))
    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    if args.fullscreen:
        info = pygame.display.Info()
        W, H = info.current_w, info.current_h
        flags |= pygame.FULLSCREEN
    try:
        pygame.display.set_mode((W, H), flags, vsync=1)
    except pygame.error:
        pygame.display.set_mode((W, H), flags)
    pygame.display.set_caption("GoLiGoL - Game of Life in Game of Life")
    ctx = moderngl.create_context()
    rend = Renderer(ctx)
    eng = Engine(world, rend)
    eng.screen = (W, H)
    font = pygame.font.SysFont("consolas,menlo,dejavusansmono,couriernew,monospace", max(14, H // 60))

    if args.tower:
        eng.tower()
    elif args.rle:
        cells = life.parse_rle(open(args.rle).read())
        eng.load_pattern(_cells_array(cells), os.path.basename(args.rle))
    else:
        eng.load_pattern(preset(1, world.top_size), PRESETS[1][0])

    show_help = False
    hud_t = 0.0
    hud_cache = None
    drag = None
    paint = None
    clock = pygame.time.Clock()
    script = [s.strip() for s in args.script.split(";")] if args.script else []
    frames = 0
    running = True
    while running:
        dt = min(clock.tick(120) / 1000.0, 0.1)
        frames += 1
        if script and frames % 3 == 0:
            _run_script_step(eng, script.pop(0), (W, H))
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                eng.zoom(1.18 ** ev.y, mx, my)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button in (1, 2):
                    drag = ev.pos
                elif ev.button == 3:
                    paint = _paint(eng, ev.pos, None, (W, H))
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button in (1, 2):
                    drag = None
                elif ev.button == 3:
                    paint = None
            elif ev.type == pygame.MOUSEMOTION:
                if drag is not None:
                    dx, dy = ev.pos[0] - drag[0], ev.pos[1] - drag[1]
                    eng.cam.move(-dx / eng.cam.s, -dy / eng.cam.s)
                    drag = ev.pos
                if paint is not None:
                    _paint(eng, ev.pos, paint, (W, H))
            elif ev.type == pygame.DROPFILE:
                _load_file(eng, ev.file)
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                mods = pygame.key.get_mods()
                ctrl = mods & pygame.KMOD_CTRL
                if k == pygame.K_ESCAPE:
                    if show_help:
                        show_help = False
                    else:
                        running = False
                elif k == pygame.K_h:
                    show_help = not show_help
                elif k == pygame.K_SPACE:
                    eng.paused = not eng.paused
                elif k == pygame.K_PERIOD:
                    eng.paused = True
                    eng.advance(1)
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    eng.mult *= 2
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    eng.mult /= 2
                elif k == pygame.K_a:
                    eng.auto = not eng.auto
                    if not eng.auto:
                        eng.manual = eng.speed() / eng.mult
                    eng.status = "auto speed: speed follows zoom" if eng.auto else "manual speed"
                elif k == pygame.K_g:
                    eng.grid = not eng.grid
                    eng.status = "grid on (visible from 3 px per cell)" if eng.grid else "grid off"
                elif k == pygame.K_z:
                    eng.zoom_rate = 0.0 if eng.zoom_rate > 0 else 1.0
                elif k == pygame.K_x:
                    eng.zoom_rate = 0.0 if eng.zoom_rate < 0 else -1.0
                elif k == pygame.K_RIGHTBRACKET:
                    eng.zoom_speed = min(64.0, eng.zoom_speed * 1.25)
                elif k == pygame.K_LEFTBRACKET:
                    eng.zoom_speed = max(1.05, eng.zoom_speed / 1.25)
                elif k == pygame.K_PAGEUP:
                    eng.zoom(2.0, W / 2, H / 2)
                elif k == pygame.K_PAGEDOWN:
                    eng.zoom(0.5, W / 2, H / 2)
                elif k == pygame.K_t:
                    eng.tower()
                elif k == pygame.K_HOME:
                    if world.mode == "pattern":
                        eng.cam = Camera(0, world.top_size / 2, world.top_size / 2, 1.0)
                        eng.fit_top(W, H)
                    else:
                        eng.cam = Camera(eng.cam.V, 0.5, 0.5, 6.0)
                    eng.reset_view()
                elif k == pygame.K_o and ctrl:
                    f, err = _ask_file()
                    if f:
                        _load_file(eng, f)
                    elif err:
                        eng.status = f"open dialog failed ({err}) - drop the .rle file onto the window instead"
                elif k == pygame.K_s and ctrl:
                    eng.status = _save_pattern(world)
                elif k == pygame.K_r:
                    eng.load_pattern(soup(world.top_size), "random soup")
                elif k == pygame.K_c:
                    eng.load_pattern(np.zeros((world.top_size, world.top_size), np.uint8), "empty universe")
                elif pygame.K_1 <= k <= pygame.K_9:
                    i = k - pygame.K_1
                    if i < len(PRESETS):
                        eng.load_pattern(preset(i, world.top_size), PRESETS[i][0])
                elif k == pygame.K_F11:
                    try:
                        pygame.display.toggle_fullscreen()
                        W, H = pygame.display.get_window_size()
                    except Exception:
                        pass
        W, H = pygame.display.get_window_size() if hasattr(pygame.display, "get_window_size") else (W, H)
        ctx.screen.use()
        eng.frame(dt, (W, H))
        hud_t -= dt
        if hud_t <= 0 or hud_cache is None:
            hud_t = 0.2
            lines = eng.hud_lines() + ([""] + HELP.splitlines() if show_help else [])
            fps = clock.get_fps()
            lines[0] += f"      {fps:4.0f} fps"
            hud_cache = _hud_surface(pygame, font, lines)
            rend.set_hud(pygame.image.tostring(hud_cache, "RGBA"), hud_cache.get_size())
        rend.draw_hud((W, H), (10, 10), hud_cache.get_size())
        pygame.display.flip()
        if args.screenshot and not script and frames > 30:
            data = ctx.screen.read(components=3)
            img = pygame.image.frombuffer(data, (W, H), "RGB")
            pygame.image.save(pygame.transform.flip(img, False, True), args.screenshot)
            running = False
    pygame.quit()


def _cells_array(cells):
    if not cells:
        return np.zeros((64, 64), np.uint8)
    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    w = max(xs) - min(xs) + 1; h = max(ys) - min(ys) + 1
    n = max(256, -(-(max(w, h) + 64) // 64) * 64)
    return cells_to_array(cells, n)


def _load_file(eng, path):
    try:
        cells = life.parse_rle(open(path).read())
        eng.load_pattern(_cells_array(cells), os.path.basename(path))
    except Exception as e:  # noqa: BLE001
        eng.status = f"could not load {path}: {e}"


SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saves")


def _save_pattern(world):
    """Save the top-level pattern as RLE into saves/ (no dialog)."""
    if world.mode != "pattern":
        return "tower mode cannot be saved - only pattern mode (your level-0 universe) can"
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        f = os.path.join(SAVE_DIR, time.strftime("pattern_%Y%m%d_%H%M%S.rle"))
        ys, xs = np.nonzero(world.top_cur)
        with open(f, "w") as fh:
            fh.write(f"#C GoLiGoL level-0 pattern, generation {world.G}\n")
            fh.write(life.to_rle(list(zip(xs.tolist(), ys.tolist()))))
        return f"saved {f}"
    except Exception as e:  # noqa: BLE001
        return f"save failed: {e}"


def _ask_file():
    """Open dialog (kept on top of the game window). Returns (path, error)."""
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        f = filedialog.askopenfilename(parent=root, initialdir=SAVE_DIR if os.path.isdir(SAVE_DIR) else None,
                                       filetypes=[("RLE pattern", "*.rle"), ("All files", "*.*")])
        root.destroy()
        return f, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def _paint(eng, pos, value, screen):
    if eng.w.mode != "pattern" or eng.cam.V != 0:
        eng.status = "drawing works on the top level of pattern mode (zoom out to level 0)"
        return None
    W, H = screen
    x, y = eng.cam.cell_at(pos[0], pos[1], W, H)
    n = eng.w.top_size
    cur = eng.w.top_cur.copy()
    if value is None:
        value = 1 - int(cur[y % n, x % n])
    if cur[y % n, x % n] != value:
        cur[y % n, x % n] = value
        eng.w.set_pattern(cur)
        eng.last_pattern = cur
        eng.acc = 0.0
        eng.reset_view()
    return value


def _hud_surface(pygame, font, lines):
    surfs = [font.render(l, True, (235, 235, 240)) for l in lines]
    lh = font.get_linesize()
    w = max(s.get_width() for s in surfs) + 16
    h = lh * len(surfs) + 10
    bg = pygame.Surface((w, h), pygame.SRCALPHA)
    bg.fill((8, 10, 18, 170))
    for i, s in enumerate(surfs):
        bg.blit(s, (8, 5 + i * lh))
    return bg


def _run_script_step(eng, step, screen):
    """Tiny automation hook used for headless testing."""
    if not step:
        return
    cmd, *a = step.split()
    if cmd == "zoom":
        eng.zoom(float(a[0]), screen[0] / 2, screen[1] / 2)
    elif cmd == "pause":
        eng.paused = True
    elif cmd == "run":
        eng.paused = False
    elif cmd == "tower":
        eng.tower()
    elif cmd == "preset":
        eng.load_pattern(preset(int(a[0]), eng.w.top_size), PRESETS[int(a[0])][0])
    elif cmd == "move":
        eng.cam.move(float(a[0]), float(a[1]))
    elif cmd == "goto":
        V, target = int(a[0]), float(a[1])
        for _ in range(10000):
            lv = eng.cam.V * math.log(TILE) - math.log(eng.cam.s)
            lt = V * math.log(TILE) - math.log(target)
            d = lt - lv
            if abs(d) < 1e-9:
                break
            eng.zoom(math.exp(-max(-2.0, min(2.0, d))), screen[0] / 2, screen[1] / 2)
    elif cmd == "mult":
        eng.mult = float(a[0])


if __name__ == "__main__":
    main()
