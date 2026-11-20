/* Generate the metapixel context database with HashLife.
 *
 *   hl_db layout.bin variants.bin K outprefix
 *
 * Simulates a 64 x 64 torus of OTCA metapixel tiles (each 2048 x 2048 cells,
 * so the torus is 2^17 cells wide) for exactly one metapixel period (35328
 * generations), starting from the phase-0 tile bitmaps in variants.bin
 * (8 variants, 2048 rows x 32 uint64, bit i of word w = column 64w+i).
 *
 * The automaton that is simulated has two bit-planes per cell: the Life
 * state s and the birth parity p (p' = p xor (s' and not s)).  This is still
 * a local, deterministic rule, so HashLife applies unchanged; time advances
 * 4 generations per HashLife step.
 *
 * Every K generations each tile's s-plane and p-plane are converted into the
 * database quadtree (leaves = 64x64 blocks, levels 1..5 above them):
 *   outprefix.blocks : unique leaf blocks, 64 uint64 each
 *   outprefix.n1..n5 : unique nodes per level (4 uint32: NW NE SW SE)
 *   outprefix.roots  : per checkpoint, per tile: uint32 s-root, uint32 p-root
 *   outprefix.final  : per tile s-root at generation 35328
 *   outprefix.vroots : s-root of each of the 8 phase-0 variants
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;
typedef uint32_t u32;
#define PERIOD 35328
#define NT 64            /* tiles per torus side */
#define TOP 17           /* torus level: 2^17 = 64 * 2048 */
#define UNSET 0xFFFFFFFFu
#define MAXL 19

static inline u64 mix(u64 h) { h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL; h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ULL; h ^= h >> 33; return h; }

/* ---------------- generic open-addressing tables ---------------- */
typedef struct { u64 s, p; } Leaf;
typedef struct { u32 c[4]; u32 res, dbs, dbp; } Node;

static Leaf *leaf; static u32 nleaf, capleaf; static u32 *leaft; static u64 leafmask;
static Node *nd[MAXL]; static u32 nn[MAXL], capn[MAXL]; static u32 *ndt[MAXL]; static u64 ndmask[MAXL];

static inline u64 lhash(u64 s, u64 p) { return mix(s * 0x9E3779B97F4A7C15ULL ^ mix(p + 0x632BE59BD9B4E019ULL)); }
static inline u64 nhash(const u32 *c) { return mix(((u64)c[0] << 32 | c[1]) ^ mix((u64)c[2] << 32 | c[3])); }

static void leaf_rehash(void) {
    u64 sz = 1 << 16; while (sz < 2ULL * capleaf) sz <<= 1;
    free(leaft); leaft = malloc(sz * 4); memset(leaft, 0xFF, sz * 4); leafmask = sz - 1;
    for (u32 i = 0; i < nleaf; i++) { u64 h = lhash(leaf[i].s, leaf[i].p) & leafmask; while (leaft[h] != UNSET) h = (h + 1) & leafmask; leaft[h] = i; }
}
static u32 mkleaf(u64 s, u64 p) {
    u64 h = lhash(s, p) & leafmask;
    while (leaft[h] != UNSET) { Leaf *l = &leaf[leaft[h]]; if (l->s == s && l->p == p) return leaft[h]; h = (h + 1) & leafmask; }
    if (nleaf == capleaf) { capleaf *= 2; leaf = realloc(leaf, (size_t)capleaf * sizeof(Leaf)); leaf[nleaf].s = s; leaf[nleaf].p = p; nleaf++; leaf_rehash(); return nleaf - 1; }
    leaf[nleaf].s = s; leaf[nleaf].p = p; leaft[h] = nleaf; return nleaf++;
}
static void node_rehash(int L) {
    u64 sz = 1 << 12; while (sz < 2ULL * capn[L]) sz <<= 1;
    free(ndt[L]); ndt[L] = malloc(sz * 4); memset(ndt[L], 0xFF, sz * 4); ndmask[L] = sz - 1;
    for (u32 i = 0; i < nn[L]; i++) { u64 h = nhash(nd[L][i].c) & ndmask[L]; while (ndt[L][h] != UNSET) h = (h + 1) & ndmask[L]; ndt[L][h] = i; }
}
static u32 mk(int L, u32 a, u32 b, u32 c, u32 d) {
    u32 k[4] = {a, b, c, d};
    u64 h = nhash(k) & ndmask[L];
    while (ndt[L][h] != UNSET) { Node *n = &nd[L][ndt[L][h]]; if (!memcmp(n->c, k, 16)) return ndt[L][h]; h = (h + 1) & ndmask[L]; }
    Node v = {{a, b, c, d}, UNSET, UNSET, UNSET};
    if (nn[L] == capn[L]) { capn[L] *= 2; nd[L] = realloc(nd[L], (size_t)capn[L] * sizeof(Node)); nd[L][nn[L]++] = v; node_rehash(L); return nn[L] - 1; }
    nd[L][nn[L]] = v; ndt[L][h] = nn[L]; return nn[L]++;
}
#define CH(L, id, q) (nd[L][id].c[q])

/* ---------------- base case: level 5 (32x32), 4 generations ---------------- */
static void step32(const u32 *in, u32 *out) {
    for (int y = 0; y < 32; y++) {
        u32 up = y ? in[y - 1] : 0, mid = in[y], dn = y < 31 ? in[y + 1] : 0;
        /* 8 neighbour bitboards */
        u32 nb[8] = {up << 1, up, up >> 1, mid << 1, mid >> 1, dn << 1, dn, dn >> 1};
        u32 s0 = 0, s1 = 0, s2 = 0;   /* bit-sliced count (s2 = "4 or more" sticky) */
        for (int i = 0; i < 8; i++) {
            u32 x = nb[i];
            u32 c0 = s0 & x; s0 ^= x;
            u32 c1 = s1 & c0; s1 ^= c0;
            s2 |= c1;
        }
        /* count == 3: s0 & s1 & !s2 ; count == 2: !s0 & s1 & !s2 */
        out[y] = s1 & ~s2 & (s0 | mid);
    }
}
static u32 base5(u32 id) {
    u32 s[32] = {0}, p[32] = {0};
    for (int q = 0; q < 4; q++) {
        u32 n4 = CH(5, id, q);
        for (int r = 0; r < 4; r++) {
            Leaf lf = leaf[CH(4, n4, r)];
            int lx = (q & 1) * 2 + (r & 1), ly = (q >> 1) * 2 + (r >> 1);
            for (int row = 0; row < 8; row++) {
                s[ly * 8 + row] |= (u32)((lf.s >> (8 * row)) & 0xFF) << (8 * lx);
                p[ly * 8 + row] |= (u32)((lf.p >> (8 * row)) & 0xFF) << (8 * lx);
            }
        }
    }
    u32 t[32];
    for (int g = 0; g < 4; g++) {
        step32(s, t);
        for (int y = 0; y < 32; y++) { p[y] ^= t[y] & ~s[y]; s[y] = t[y]; }
    }
    u32 lv[4];
    for (int q = 0; q < 4; q++) {
        int qx = q & 1, qy = q >> 1;
        u64 ls = 0, lp = 0;
        for (int row = 0; row < 8; row++) {
            ls |= (u64)((s[8 + qy * 8 + row] >> (8 + 8 * qx)) & 0xFF) << (8 * row);
            lp |= (u64)((p[8 + qy * 8 + row] >> (8 + 8 * qx)) & 0xFF) << (8 * row);
        }
        lv[q] = mkleaf(ls, lp);
    }
    return mk(4, lv[0], lv[1], lv[2], lv[3]);
}

/* ---------------- recursive step: level k -> centre (level k-1) + 4 gens ---------------- */
static u32 result(int k, u32 id) {
    if (nd[k][id].res != UNSET) return nd[k][id].res;
    u32 r;
    if (k == 5) r = base5(id);
    else {
        u32 a = CH(k, id, 0), b = CH(k, id, 1), c = CH(k, id, 2), d = CH(k, id, 3);
        int m = k - 1, g = k - 2;
        u32 an = CH(m, a, 0), ae = CH(m, a, 1), as = CH(m, a, 2), aw = CH(m, a, 3);
        u32 bn = CH(m, b, 0), be = CH(m, b, 1), bs = CH(m, b, 2), bw = CH(m, b, 3);
        u32 cn = CH(m, c, 0), ce = CH(m, c, 1), cs = CH(m, c, 2), cw = CH(m, c, 3);
        u32 dn = CH(m, d, 0), de = CH(m, d, 1), ds = CH(m, d, 2), dw = CH(m, d, 3);
        /* naming: xn = nw child, xe = ne child, xs = sw child, xw = se child */
#define C(n, q) CH(g, n, q)
        u32 t00 = mk(g, C(an, 3), C(ae, 2), C(as, 1), C(aw, 0));
        u32 t01 = mk(g, C(ae, 3), C(bn, 2), C(aw, 1), C(bs, 0));
        u32 t02 = mk(g, C(bn, 3), C(be, 2), C(bs, 1), C(bw, 0));
        u32 t10 = mk(g, C(as, 3), C(aw, 2), C(cn, 1), C(ce, 0));
        u32 t11 = mk(g, C(aw, 3), C(bs, 2), C(ce, 1), C(dn, 0));
        u32 t12 = mk(g, C(bs, 3), C(bw, 2), C(dn, 1), C(de, 0));
        u32 t20 = mk(g, C(cn, 3), C(ce, 2), C(cs, 1), C(cw, 0));
        u32 t21 = mk(g, C(ce, 3), C(dn, 2), C(cw, 1), C(ds, 0));
        u32 t22 = mk(g, C(dn, 3), C(de, 2), C(ds, 1), C(dw, 0));
#undef C
        u32 q0 = mk(m, t00, t01, t10, t11), q1 = mk(m, t01, t02, t11, t12);
        u32 q2 = mk(m, t10, t11, t20, t21), q3 = mk(m, t11, t12, t21, t22);
        u32 r0 = result(m, q0), r1 = result(m, q1), r2 = result(m, q2), r3 = result(m, q3);
        r = mk(m, r0, r1, r2, r3);
    }
    nd[k][id].res = r;
    return r;
}

/* ---------------- database (output) tables ---------------- */
static u64 *blk; static u32 nblk, capblk; static u32 *blkt; static u64 blkmask;
static u64 bhash(const u64 *b) { u64 h = 0x9E3779B97F4A7C15ULL; for (int i = 0; i < 64; i++) h = mix(h ^ b[i]) + i; return h; }
static void blk_rehash(void) {
    u64 sz = 1 << 16; while (sz < 2ULL * capblk) sz <<= 1;
    free(blkt); blkt = malloc(sz * 4); memset(blkt, 0xFF, sz * 4); blkmask = sz - 1;
    for (u32 i = 0; i < nblk; i++) { u64 h = bhash(blk + (size_t)i * 64) & blkmask; while (blkt[h] != UNSET) h = (h + 1) & blkmask; blkt[h] = i; }
}
static u32 blk_id(const u64 *b) {
    u64 h = bhash(b) & blkmask;
    while (blkt[h] != UNSET) { if (!memcmp(blk + (size_t)blkt[h] * 64, b, 512)) return blkt[h]; h = (h + 1) & blkmask; }
    if (nblk == capblk) { capblk *= 2; blk = realloc(blk, (size_t)capblk * 512); memcpy(blk + (size_t)nblk * 64, b, 512); nblk++; blk_rehash(); return nblk - 1; }
    memcpy(blk + (size_t)nblk * 64, b, 512); blkt[h] = nblk; return nblk++;
}
typedef struct { u32 k[4]; } Key4;
static Key4 *dn_[6]; static u32 ndn[6], capdn[6]; static u32 *dnt[6]; static u64 dnmask[6];
static void dn_rehash(int L) {
    u64 sz = 1 << 12; while (sz < 2ULL * capdn[L]) sz <<= 1;
    free(dnt[L]); dnt[L] = malloc(sz * 4); memset(dnt[L], 0xFF, sz * 4); dnmask[L] = sz - 1;
    for (u32 i = 0; i < ndn[L]; i++) { u64 h = nhash(dn_[L][i].k) & dnmask[L]; while (dnt[L][h] != UNSET) h = (h + 1) & dnmask[L]; dnt[L][h] = i; }
}
static u32 dn_id(int L, u32 a, u32 b, u32 c, u32 d) {
    Key4 k = {{a, b, c, d}};
    u64 h = nhash(k.k) & dnmask[L];
    while (dnt[L][h] != UNSET) { if (!memcmp(&dn_[L][dnt[L][h]], &k, 16)) return dnt[L][h]; h = (h + 1) & dnmask[L]; }
    if (ndn[L] == capdn[L]) { capdn[L] *= 2; dn_[L] = realloc(dn_[L], (size_t)capdn[L] * 16); dn_[L][ndn[L]++] = k; dn_rehash(L); return ndn[L] - 1; }
    dn_[L][ndn[L]] = k; dnt[L][h] = ndn[L]; return ndn[L]++;
}
/* gather plane (0 = s, 1 = p) of hashlife node (level L) into 64-wide rows */
static void gather(int L, u32 id, int x0, int y0, u64 *rows, int plane) {
    if (L == 3) {
        u64 v = plane ? leaf[id].p : leaf[id].s;
        for (int r = 0; r < 8; r++) rows[y0 + r] |= ((v >> (8 * r)) & 0xFF) << x0;
        return;
    }
    int h = 1 << (L - 1);
    for (int q = 0; q < 4; q++) gather(L - 1, CH(L, id, q), x0 + (q & 1) * h, y0 + (q >> 1) * h, rows, plane);
}
static u32 to_db(int L, u32 id, int plane) {      /* L in 6..11 */
    u32 *slot = plane ? &nd[L][id].dbp : &nd[L][id].dbs;
    if (*slot != UNSET) return *slot;
    u32 r;
    if (L == 6) { u64 rows[64] = {0}; gather(6, id, 0, 0, rows, plane); r = blk_id(rows); }
    else {
        u32 c[4];
        for (int q = 0; q < 4; q++) c[q] = to_db(L - 1, CH(L, id, q), plane);
        r = dn_id(L - 6, c[0], c[1], c[2], c[3]);
    }
    slot = plane ? &nd[L][id].dbp : &nd[L][id].dbs;
    *slot = r;
    return r;
}
static u32 tile_of(u32 T, int tx, int ty) {
    u32 id = T;
    for (int L = TOP; L > 11; L--) {
        int bit = L - 12;
        int q = ((ty >> bit) & 1) * 2 + ((tx >> bit) & 1);
        id = CH(L, id, q);
    }
    return id;
}

/* ---------------- garbage collection ---------------- */
static u32 *remap[MAXL]; static u32 *leafremap;
static void mark(int L, u32 id) {
    if (L == 3) { leafremap[id] = 1; return; }
    if (remap[L][id]) return;
    remap[L][id] = 1;
    for (int q = 0; q < 4; q++) mark(L - 1, CH(L, id, q));
}
static u32 gc(u32 T) {
    leafremap = calloc(nleaf, 4);
    for (int L = 4; L <= TOP + 1; L++) remap[L] = calloc(nn[L] ? nn[L] : 1, 4);
    mark(TOP, T);
    u32 k = 0;
    for (u32 i = 0; i < nleaf; i++) if (leafremap[i]) { leaf[k] = leaf[i]; leafremap[i] = k++; } else leafremap[i] = UNSET;
    nleaf = k; leaf_rehash();
    for (int L = 4; L <= TOP + 1; L++) {
        k = 0;
        for (u32 i = 0; i < nn[L]; i++) if (remap[L][i]) {
            Node n = nd[L][i];
            for (int q = 0; q < 4; q++) n.c[q] = (L == 4) ? leafremap[n.c[q]] : remap[L - 1][n.c[q]];
            n.res = UNSET;
            nd[L][k] = n; remap[L][i] = k++;
        } else remap[L][i] = UNSET;
        nn[L] = k; node_rehash(L);
    }
    u32 nT = remap[TOP][T];
    free(leafremap); for (int L = 4; L <= TOP + 1; L++) free(remap[L]);
    return nT;
}
static u64 total_nodes(void) { u64 t = nleaf; for (int L = 4; L <= TOP + 1; L++) t += nn[L]; return t; }

static void save(const char *prefix, const char *ext, const void *p, size_t sz, size_t n) {
    char fn[600]; snprintf(fn, sizeof fn, "%s.%s", prefix, ext);
    FILE *f = fopen(fn, "wb"); if (!f) { perror(fn); exit(1); }
    fwrite(p, sz, n, f); fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: hl_db layout.bin variants.bin K outprefix\n"); return 1; }
    int K = atoi(argv[3]); const char *prefix = argv[4];
    if (K % 4 || PERIOD % K) { fprintf(stderr, "K must be a multiple of 4 dividing %d\n", PERIOD); return 1; }
    u64 maxnodes = getenv("HL_MAXNODES") ? strtoull(getenv("HL_MAXNODES"), 0, 10) : 60000000ULL;

    capleaf = 1 << 16; leaf = malloc((size_t)capleaf * sizeof(Leaf)); leaf_rehash();
    for (int L = 4; L <= TOP + 1; L++) { capn[L] = 1 << 10; nd[L] = malloc((size_t)capn[L] * sizeof(Node)); node_rehash(L); }
    capblk = 1 << 14; blk = malloc((size_t)capblk * 512); blk_rehash();
    for (int L = 1; L <= 5; L++) { capdn[L] = 1 << 10; dn_[L] = malloc((size_t)capdn[L] * 16); dn_rehash(L); }
    { u64 z[64] = {0}; blk_id(z); u32 n = 0; for (int L = 1; L <= 5; L++) n = dn_id(L, n, n, n, n); }

    unsigned char lay[NT * NT];
    FILE *f = fopen(argv[1], "rb"); if (!f || fread(lay, 1, NT * NT, f) != NT * NT) { perror("layout"); return 1; } fclose(f);
    u64 *vb = malloc((size_t)8 * 2048 * 32 * 8);
    f = fopen(argv[2], "rb"); if (!f || fread(vb, 2048 * 32 * 8, 8, f) != 8) { perror("variants"); return 1; } fclose(f);

    /* build the 8 variant tiles (level 11) */
    u32 vt[8];
    static u32 grid[256 * 256];
    for (int v = 0; v < 8; v++) {
        const u64 *bm = vb + (size_t)v * 2048 * 32;
        for (int ly = 0; ly < 256; ly++) for (int lx = 0; lx < 256; lx++) {
            u64 s = 0;
            for (int r = 0; r < 8; r++) s |= ((bm[(ly * 8 + r) * 32 + lx / 8] >> (8 * (lx % 8))) & 0xFF) << (8 * r);
            grid[ly * 256 + lx] = mkleaf(s, 0);
        }
        int n = 256;
        for (int L = 4; L <= 11; L++) {
            int m = n / 2;
            for (int y = 0; y < m; y++) for (int x = 0; x < m; x++)
                grid[y * m + x] = mk(L, grid[2 * y * n + 2 * x], grid[2 * y * n + 2 * x + 1], grid[(2 * y + 1) * n + 2 * x], grid[(2 * y + 1) * n + 2 * x + 1]);
            n = m;
        }
        vt[v] = grid[0];
    }
    u32 vroots[8];
    for (int v = 0; v < 8; v++) vroots[v] = to_db(11, vt[v], 0);
    save(prefix, "vroots", vroots, 4, 8);
    /* torus */
    for (int i = 0; i < NT * NT; i++) grid[i] = vt[lay[i]];
    {
        int n = NT;
        for (int L = 12; L <= TOP; L++) {
            int m = n / 2;
            for (int y = 0; y < m; y++) for (int x = 0; x < m; x++)
                grid[y * m + x] = mk(L, grid[2 * y * n + 2 * x], grid[2 * y * n + 2 * x + 1], grid[(2 * y + 1) * n + 2 * x], grid[(2 * y + 1) * n + 2 * x + 1]);
            n = m;
        }
    }
    u32 T = grid[0];

    char fn[600]; snprintf(fn, sizeof fn, "%s.roots", prefix);
    FILE *fr = fopen(fn, "wb");
    static u32 roots[2 * NT * NT];
    long GMAX = getenv("HL_GENS") ? atol(getenv("HL_GENS")) : PERIOD;
    for (long g = 0; g <= PERIOD; g += 4) {
        if (g > GMAX) break;
        if (g % K == 0) {
            if (g < PERIOD) {
                for (int i = 0; i < NT * NT; i++) {
                    u32 t = tile_of(T, i % NT, i / NT);
                    roots[2 * i] = to_db(11, t, 0);
                    roots[2 * i + 1] = to_db(11, t, 1);
                }
                fwrite(roots, 4, 2 * NT * NT, fr); fflush(fr);
            } else {
                for (int i = 0; i < NT * NT; i++) roots[i] = to_db(11, tile_of(T, i % NT, i / NT), 0);
                save(prefix, "final", roots, 4, NT * NT);
            }
            fprintf(stderr, "gen %ld hl-nodes %llu db-blocks %u db-nodes %u %u %u %u %u\n", g,
                    (unsigned long long)total_nodes(), nblk, ndn[1], ndn[2], ndn[3], ndn[4], ndn[5]);
            if (total_nodes() > maxnodes) { T = gc(T); fprintf(stderr, "  gc -> %llu\n", (unsigned long long)total_nodes()); }
        }
        if (g == PERIOD) break;
        u32 R = mk(TOP + 1, T, T, T, T);
        u32 c = result(TOP + 1, R);
        T = mk(TOP, CH(TOP, c, 3), CH(TOP, c, 2), CH(TOP, c, 1), CH(TOP, c, 0));
    }
    fclose(fr);
    save(prefix, "blocks", blk, 512, nblk);
    for (int L = 1; L <= 5; L++) { char e[8]; snprintf(e, sizeof e, "n%d", L); save(prefix, e, dn_[L], 16, ndn[L]); }
    fprintf(stderr, "done\n");
    return 0;
}
