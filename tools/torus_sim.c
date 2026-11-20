/* Dense bit-parallel Life on a torus of N x N OTCA metapixel tiles.
 *
 * Research / dataset tool (not needed to run the app).
 *
 *   torus_sim N state.bin on.i32 off.i32 GENS EVERY hashes.bin [tiles.bin K0 K1 ...]
 *
 * state.bin : N*N bytes (row-major, y then x), 1 = start from the ON tile.
 * on/off    : int32 (x,y) pairs of live cells of the pristine tiles.
 * Every EVERY generations (including generation 0) a 64-bit hash of each
 * tile's 2048x2048 bitmap is appended to hashes.bin.  If tiles.bin is given,
 * the raw bitmaps (2048 rows x 32 uint64 per tile, all tiles) are appended at
 * each listed checkpoint index K.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;
#define T 2048
#define TW 32

static int N, H, W;            /* H rows, W words per row */
static u64 *A, *B;             /* (H+2) x (W+2) with ghost border */
#define AT(P, y, x) (P)[(size_t)((y) + 1) * (W + 2) + (x) + 1]

static void wrap(u64 *P) {
    for (int y = 0; y < H; y++) { AT(P, y, -1) = AT(P, y, W - 1); AT(P, y, W) = AT(P, y, 0); }
    memcpy(&AT(P, -1, -1), &AT(P, H - 1, -1), sizeof(u64) * (W + 2));
    memcpy(&AT(P, H, -1), &AT(P, 0, -1), sizeof(u64) * (W + 2));
}

static void step(const u64 *restrict P, u64 *restrict Q) {
#pragma omp parallel
    {
        u64 *s0 = malloc(sizeof(u64) * 3 * W), *s1 = malloc(sizeof(u64) * 3 * W);
#pragma omp for schedule(static)
        for (int chunk = 0; chunk < H; chunk += 256) {
            int y1 = chunk + 256 < H ? chunk + 256 : H;
            for (int y = chunk - 1; y <= y1; y++) {
                const u64 *r = &AT(P, y, 0);
                u64 *a0 = s0 + ((y + 3) % 3) * W, *a1 = s1 + ((y + 3) % 3) * W;
                for (int x = 0; x < W; x++) {
                    u64 c = r[x], l = (c << 1) | (r[x - 1] >> 63), rr = (c >> 1) | (r[x + 1] << 63);
                    u64 t = l ^ rr;
                    a1[x] = (l & rr) | (t & c);
                    a0[x] = t ^ c;
                }
                if (y < chunk + 1) continue;
                int yc = y - 1;
                const u64 *u0 = s0 + ((yc - 1 + 3) % 3) * W, *u1 = s1 + ((yc - 1 + 3) % 3) * W;
                const u64 *m0 = s0 + ((yc + 3) % 3) * W, *m1 = s1 + ((yc + 3) % 3) * W;
                const u64 *d0 = s0 + ((yc + 1 + 3) % 3) * W, *d1 = s1 + ((yc + 1 + 3) % 3) * W;
                const u64 *cen = &AT(P, yc, 0);
                u64 *out = &AT(Q, yc, 0);
                for (int x = 0; x < W; x++) {
                    u64 t0 = u0[x] ^ d0[x];
                    u64 c0 = (u0[x] & d0[x]) | (t0 & m0[x]);
                    t0 ^= m0[x];
                    u64 p = u1[x] ^ m1[x], g1 = u1[x] & m1[x];
                    u64 q = d1[x] ^ c0, g2 = d1[x] & c0;
                    u64 v0 = p ^ q, v1 = g1 ^ g2 ^ (p & q);
                    out[x] = (t0 & v0 & ~v1) | (~t0 & ~v0 & v1 & cen[x]);
                }
            }
        }
        free(s0); free(s1);
    }
}

static void put(const int32_t *cells, long n, int tx, int ty) {
    for (long i = 0; i < n; i++) {
        long x = (cells[2 * i] + (long)tx * T + (long)N * T) % ((long)N * T);
        long y = (cells[2 * i + 1] + (long)ty * T + (long)N * T) % ((long)N * T);
        AT(A, y, x >> 6) |= 1ULL << (x & 63);
    }
}

static int32_t *load(const char *fn, long *n) {
    FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    int32_t *p = malloc(sz); if (fread(p, 1, sz, f) != (size_t)sz) exit(1);
    fclose(f); *n = sz / 8; return p;
}

static u64 tile_hash(int tx, int ty) {
    u64 h = 0x9E3779B97F4A7C15ULL;
    for (int y = 0; y < T; y++) {
        const u64 *r = &AT(A, ty * T + y, tx * TW);
        for (int x = 0; x < TW; x++) {
            h ^= r[x]; h *= 0xFF51AFD7ED558CCDULL; h ^= h >> 29;
        }
    }
    return h;
}

/* ---------- hash-consed quadtree (leaf = 8x8 cells in one u64) ---------- */
#define LEVELS 9               /* level 0 = 8x8 leaf ... level 8 = 2048 tile */
typedef struct { uint32_t k[4]; } Key4;
static u64 *leafv; static uint32_t nleaf, capleaf; static uint32_t *leaft; static u64 leafmask;
static Key4 *nodev[LEVELS]; static uint32_t nnode[LEVELS], capnode[LEVELS];
static uint32_t *nodet[LEVELS]; static u64 nodemask[LEVELS];
static inline u64 mix(u64 h) { h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL; h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ULL; h ^= h >> 33; return h; }
static void leaf_rehash(void) {
    u64 sz = 1024; while (sz < 3ULL * capleaf) sz <<= 1;
    free(leaft); leaft = malloc(sz * 4); memset(leaft, 0xFF, sz * 4); leafmask = sz - 1;
    for (uint32_t i = 0; i < nleaf; i++) { u64 h = mix(leafv[i]) & leafmask; while (leaft[h] != 0xFFFFFFFFu) h = (h + 1) & leafmask; leaft[h] = i; }
}
static uint32_t leaf_id(u64 v) {
    u64 h = mix(v) & leafmask;
    while (leaft[h] != 0xFFFFFFFFu) { if (leafv[leaft[h]] == v) return leaft[h]; h = (h + 1) & leafmask; }
    if (nleaf == capleaf) { capleaf *= 2; leafv = realloc(leafv, (size_t)capleaf * 8); leafv[nleaf++] = v; leaf_rehash(); return nleaf - 1; }
    leafv[nleaf] = v; leaft[h] = nleaf; return nleaf++;
}
static inline u64 khash(const Key4 *k) { return mix(((u64)k->k[0] << 32 | k->k[1]) * 0x9E3779B97F4A7C15ULL ^ mix((u64)k->k[2] << 32 | k->k[3])); }
static void node_rehash(int L) {
    u64 sz = 1024; while (sz < 3ULL * capnode[L]) sz <<= 1;
    free(nodet[L]); nodet[L] = malloc(sz * 4); memset(nodet[L], 0xFF, sz * 4); nodemask[L] = sz - 1;
    for (uint32_t i = 0; i < nnode[L]; i++) { u64 h = khash(&nodev[L][i]) & nodemask[L]; while (nodet[L][h] != 0xFFFFFFFFu) h = (h + 1) & nodemask[L]; nodet[L][h] = i; }
}
static uint32_t node_id(int L, Key4 k) {
    u64 h = khash(&k) & nodemask[L];
    while (nodet[L][h] != 0xFFFFFFFFu) { if (!memcmp(&nodev[L][nodet[L][h]], &k, 16)) return nodet[L][h]; h = (h + 1) & nodemask[L]; }
    if (nnode[L] == capnode[L]) { capnode[L] *= 2; nodev[L] = realloc(nodev[L], (size_t)capnode[L] * 16); nodev[L][nnode[L]++] = k; node_rehash(L); return nnode[L] - 1; }
    nodev[L][nnode[L]] = k; nodet[L][h] = nnode[L]; return nnode[L]++;
}
static void canon_init(const char *prefix) {
    char fn[512];
    capleaf = 1 << 16; leafv = malloc((size_t)capleaf * 8); nleaf = 0;
    snprintf(fn, sizeof fn, "%s.L0", prefix);
    FILE *f = fopen(fn, "rb");
    if (f) { fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); nleaf = sz / 8;
        while (capleaf < nleaf + 16) capleaf *= 2; leafv = realloc(leafv, (size_t)capleaf * 8);
        if (fread(leafv, 8, nleaf, f) != nleaf) exit(1); fclose(f); }
    leaf_rehash();
    for (int L = 1; L < LEVELS; L++) {
        capnode[L] = 1 << 16; nodev[L] = malloc((size_t)capnode[L] * 16); nnode[L] = 0;
        snprintf(fn, sizeof fn, "%s.L%d", prefix, L);
        f = fopen(fn, "rb");
        if (f) { fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); nnode[L] = sz / 16;
            while (capnode[L] < nnode[L] + 16) capnode[L] *= 2; nodev[L] = realloc(nodev[L], (size_t)capnode[L] * 16);
            if (fread(nodev[L], 16, nnode[L], f) != nnode[L]) exit(1); fclose(f); }
        node_rehash(L);
    }
}
static void canon_save(const char *prefix) {
    char fn[512];
    snprintf(fn, sizeof fn, "%s.L0", prefix);
    FILE *f = fopen(fn, "wb"); fwrite(leafv, 8, nleaf, f); fclose(f);
    for (int L = 1; L < LEVELS; L++) {
        snprintf(fn, sizeof fn, "%s.L%d", prefix, L);
        f = fopen(fn, "wb"); fwrite(nodev[L], 16, nnode[L], f); fclose(f);
    }
}
static uint32_t canon_tile(int tx, int ty) {
    static uint32_t ids[256 * 256];
    for (int by = 0; by < 256; by++)
        for (int bx = 0; bx < 256; bx++) {
            u64 v = 0;
            for (int r = 0; r < 8; r++) {
                u64 w = AT(A, ty * T + by * 8 + r, tx * TW + (bx >> 3));
                v |= ((w >> ((bx & 7) * 8)) & 0xFF) << (8 * r);
            }
            ids[by * 256 + bx] = leaf_id(v);
        }
    int n = 256;
    for (int L = 1; L < LEVELS; L++) {
        int m = n / 2;
        for (int y = 0; y < m; y++) for (int x = 0; x < m; x++) {
            Key4 k = {{ids[(2 * y) * n + 2 * x], ids[(2 * y) * n + 2 * x + 1], ids[(2 * y + 1) * n + 2 * x], ids[(2 * y + 1) * n + 2 * x + 1]}};
            ids[y * m + x] = node_id(L, k);
        }
        n = m;
    }
    return ids[0];
}

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage\n"); return 1; }
    N = atoi(argv[1]); H = N * T; W = N * TW;
    long non, noff; int32_t *on = load(argv[3], &non), *off = load(argv[4], &noff);
    long gens = atol(argv[5]); int every = atoi(argv[6]);
    A = calloc((size_t)(H + 2) * (W + 2), sizeof(u64));
    B = calloc((size_t)(H + 2) * (W + 2), sizeof(u64));
    unsigned char *st = malloc(N * N);
    FILE *f = fopen(argv[2], "rb"); if (!f || fread(st, 1, N * N, f) != (size_t)(N * N)) { perror("state"); return 1; }
    fclose(f);
    if (getenv("VARIANTS")) {   /* state.bin holds variant indices; compose tile bitmaps */
        FILE *fv = fopen(getenv("VARIANTS"), "rb"); if (!fv) { perror("variants"); return 1; }
        fseek(fv, 0, SEEK_END); long nv = ftell(fv) / (T * TW * 8); fseek(fv, 0, SEEK_SET);
        u64 *vb = malloc((size_t)nv * T * TW * 8);
        if (fread(vb, (size_t)T * TW * 8, nv, fv) != (size_t)nv) return 1; fclose(fv);
        for (int ty = 0; ty < N; ty++) for (int tx = 0; tx < N; tx++)
            for (int y = 0; y < T; y++)
                memcpy(&AT(A, ty * T + y, tx * TW), vb + ((size_t)st[ty * N + tx] * T + y) * TW, TW * 8);
        free(vb);
    } else
    for (int ty = 0; ty < N; ty++) for (int tx = 0; tx < N; tx++)
        if (st[ty * N + tx]) put(on, non, tx, ty); else put(off, noff, tx, ty);
    FILE *fh = fopen(argv[7], "wb");
    FILE *ft = argc > 8 ? fopen(argv[8], "wb") : NULL;
    u64 *hs = malloc(sizeof(u64) * N * N);
    const char *qprefix = getenv("QT_PREFIX"), *qroots = getenv("QT_ROOTS");
    long qstart = getenv("QT_START") ? atol(getenv("QT_START")) : 0;
    FILE *fq = NULL; uint32_t *rs = malloc(4 * N * N);
    if (qprefix) { canon_init(qprefix); fq = fopen(qroots, "wb"); }
    for (long g = 0; g <= gens; g++) {
        if (g % every == 0) {
            long k = g / every;
            for (int i = 0; i < N * N; i++) hs[i] = tile_hash(i % N, i / N);
            fwrite(hs, sizeof(u64), N * N, fh); fflush(fh);
            if (fq && k >= qstart) {
                for (int i = 0; i < N * N; i++) rs[i] = canon_tile(i % N, i / N);
                fwrite(rs, 4, N * N, fq); fflush(fq);
            }
            for (int a = 9; a < argc; a++) if (atol(argv[a]) == k) {
                for (int i = 0; i < N * N; i++)
                    for (int y = 0; y < T; y++)
                        fwrite(&AT(A, (i / N) * T + y, (i % N) * TW), sizeof(u64), TW, ft);
                fflush(ft);
            }
        }
        if (g == gens) break;
        wrap(A); step(A, B);
        u64 *t = A; A = B; B = t;
    }
    fclose(fh); if (ft) fclose(ft);
    if (fq) { fclose(fq); canon_save(qprefix);
        fprintf(stderr, "leaves %u nodes", nleaf); for (int L = 1; L < LEVELS; L++) fprintf(stderr, " %u", nnode[L]); fprintf(stderr, "\n"); }
    return 0;
}
