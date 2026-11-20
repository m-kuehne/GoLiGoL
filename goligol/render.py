"""OpenGL (moderngl) renderer for one level of the tower plus its children."""
from __future__ import annotations

import numpy as np
import moderngl

VERT = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() { v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FRAG = """
#version 330
uniform usampler2D tCur;
uniform usampler2D tPrev;
uniform usampler2D tHb;
uniform isampler2D tLut;
uniform sampler2DArray tLayers;
uniform vec2 uScreen;
uniform vec2 uCam;          // camera centre, in cells, relative to region origin
uniform float uScale;       // pixels per cell
uniform ivec2 uRegSize;     // region size in cells
uniform float uAlpha;       // children overlay opacity
uniform float uGain;
uniform float uTint;
uniform int uTintPrev;
uniform vec3 uDead, uAlive, uCDead, uCAlive;
uniform float uGrid;
uniform int uWrap;          // >0: region is a torus of this many cells (top level)
out vec4 fragColor;

uint bitAt(usampler2D t, ivec2 c) {
    if (uWrap > 0) { c = ((c % uWrap) + uWrap) % uWrap; }
    if (c.x < 0 || c.y < 0 || c.x >= uRegSize.x || c.y >= uRegSize.y) return 0u;
    uint w = texelFetch(t, ivec2(c.x >> 5, c.y), 0).r;
    return (w >> uint(c.x & 31)) & 1u;
}

void main() {
    vec2 sp = vec2(gl_FragCoord.x, uScreen.y - gl_FragCoord.y);
    vec2 p = uCam + (sp - 0.5 * uScreen) / uScale;
    ivec2 c = ivec2(floor(p));
    uint alive = bitAt(tCur, c);
    vec3 col = alive == 1u ? uAlive : uDead;
    if (uAlpha > 0.0) {
        int n = 0;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
                n = (n << 1) | int(bitAt(tCur, c + ivec2(dx, dy)));
        int ctx = (n << 2) | int(bitAt(tPrev, c)) | (int(bitAt(tHb, c)) << 1);
        int layer = texelFetch(tLut, ivec2(ctx, 0), 0).r;
        if (layer >= 0) {
            float d = texture(tLayers, vec3(fract(p), float(layer))).r;
            float b = clamp(sqrt(d * uGain), 0.0, 1.0);
            vec3 o = mix(uCDead, uCAlive, b);
            // keep a memory of the parent state inside the cell
            uint shown = uTintPrev == 1 ? bitAt(tPrev, c) : alive;   // what the metapixel displays now
            if (shown == 1u) o = mix(o, uAlive, uTint);
            col = mix(col, o, uAlpha);
        }
    }
    if (uGrid > 0.0) {
        vec2 f = fract(p) * uScale;
        float e = min(min(f.x, f.y), min(uScale - f.x, uScale - f.y));
        col = mix(col, vec3(0.55, 0.62, 0.80), 0.55 * uGrid * (1.0 - smoothstep(0.0, 1.0, e)));
    }
    if (uWrap > 0) {   // outline of the top-level torus
        vec2 q = abs(p - floor(p / float(uWrap) + 0.5) * float(uWrap)) * uScale;
        if (min(q.x, q.y) < 1.0) col = mix(col, vec3(0.35, 0.4, 0.55), 0.6);
    }
    fragColor = vec4(pow(col, vec3(1.0 / 1.1)), 1.0);
}
"""

HUD_VERT = """
#version 330
in vec2 in_pos;
uniform vec4 uRect;   // x, y, w, h in NDC
out vec2 v_uv;
void main() {
    v_uv = vec2(in_pos.x * 0.5 + 0.5, 0.5 - in_pos.y * 0.5);
    vec2 q = uRect.xy + (in_pos * 0.5 + 0.5) * uRect.zw;
    gl_Position = vec4(q, 0.0, 1.0);
}
"""
HUD_FRAG = """
#version 330
uniform sampler2D tex;
in vec2 v_uv;
out vec4 fragColor;
void main() { fragColor = texture(tex, v_uv); }
"""

PALETTE = [  # (dead, alive) per level, cycling
    ((0.035, 0.045, 0.075), (1.00, 0.80, 0.42)),
    ((0.045, 0.030, 0.070), (0.55, 0.85, 1.00)),
    ((0.030, 0.060, 0.050), (1.00, 0.55, 0.75)),
    ((0.060, 0.045, 0.030), (0.62, 1.00, 0.62)),
]


def palette(level: int):
    return PALETTE[level % len(PALETTE)]


class Renderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.hud_prog = ctx.program(vertex_shader=HUD_VERT, fragment_shader=HUD_FRAG)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self.vbo = ctx.buffer(quad.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])
        self.hud_vao = ctx.vertex_array(self.hud_prog, [(self.vbo, "2f", "in_pos")])
        self.cell_tex = [None, None, None]
        self.cell_shape = None
        self.lut = ctx.texture((2048, 1), 1, dtype="i4")
        self.lut.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.layers = None
        self.layer_res = 0
        self.layer_cap = 0
        self.hud_tex = None
        for i, name in enumerate(("tCur", "tPrev", "tHb", "tLut", "tLayers")):
            self.prog[name].value = i
        self._ensure_layers(4, 1)

    # ------------------------------------------------------------ data upload
    def set_cells(self, cur: np.ndarray, prev: np.ndarray, hb: np.ndarray):
        rows, words = cur.shape
        shape = (words * 2, rows)
        if self.cell_shape != shape:
            for t in self.cell_tex:
                if t is not None:
                    t.release()
            self.cell_tex = []
            for _ in range(3):
                t = self.ctx.texture(shape, 1, dtype="u4")
                t.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self.cell_tex.append(t)
            self.cell_shape = shape
        for t, a in zip(self.cell_tex, (cur, prev, hb)):
            t.write(np.ascontiguousarray(a, dtype="<u8").view("<u4").tobytes())

    def _ensure_layers(self, res: int, cap: int):
        if self.layers is not None and self.layer_res == res and self.layer_cap >= cap:
            return False
        if self.layers is not None:
            self.layers.release()
        self.layers = self.ctx.texture_array((res, res, cap), 1, dtype="f2")
        self.layers.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.layers.repeat_x = False
        self.layers.repeat_y = False
        self.layer_res, self.layer_cap = res, cap
        return True

    def ensure_layers(self, res: int, cap: int) -> bool:
        """(Re)allocate the children texture array; True if it was recreated."""
        return self._ensure_layers(res, cap)

    def write_layers(self, start: int, imgs: np.ndarray):
        n, r, _ = imgs.shape
        self.layers.write(imgs.astype(np.float16).tobytes(), viewport=(0, 0, start, r, r, n))

    def write_layer(self, layer: int, img: np.ndarray):
        r = img.shape[0]
        self.layers.write(img.astype(np.float16).tobytes(), viewport=(0, 0, layer, r, r, 1))

    def set_lut(self, lut: np.ndarray):
        self.lut.write(np.ascontiguousarray(lut, dtype="<i4").tobytes())

    def set_hud(self, rgba: bytes, size):
        if self.hud_tex is None or self.hud_tex.size != size:
            if self.hud_tex is not None:
                self.hud_tex.release()
            self.hud_tex = self.ctx.texture(size, 4)
            self.hud_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.hud_tex.write(rgba)

    # ------------------------------------------------------------ draw
    def draw(self, screen, cam, scale, reg_size, alpha, level, grid, wrap=0, gain=6.0, tint=0.0, tint_prev=False):
        p = self.prog
        p["uTintPrev"].value = int(tint_prev)
        p["uTint"].value = float(tint)
        p["uScreen"].value = tuple(float(v) for v in screen)
        p["uCam"].value = (float(cam[0]), float(cam[1]))
        p["uScale"].value = float(scale)
        p["uRegSize"].value = (int(reg_size[0]), int(reg_size[1]))
        p["uAlpha"].value = float(alpha)
        p["uGain"].value = float(gain)
        dead, alive = palette(level)
        cdead, calive = palette(level - 1)
        p["uDead"].value, p["uAlive"].value = dead, alive
        p["uCDead"].value, p["uCAlive"].value = cdead, calive
        p["uGrid"].value = float(grid)
        p["uWrap"].value = int(wrap)
        for i, t in enumerate(self.cell_tex):
            t.use(i)
        self.lut.use(3)
        self.layers.use(4)
        self.ctx.viewport = (0, 0, int(screen[0]), int(screen[1]))
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def draw_hud(self, screen, pos, size):
        if self.hud_tex is None:
            return
        W, H = screen
        x, y = pos
        w, h = size
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.hud_prog["uRect"].value = (2 * x / W - 1, 1 - 2 * (y + h) / H, 2 * w / W, 2 * h / H)
        self.hud_tex.use(0)
        self.hud_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
