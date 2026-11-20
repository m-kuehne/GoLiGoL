/* Generate the metapixel context database.
 *
 *   gen_db NX NY layout.bin variants.bin K outprefix
 *
 * Simulates an NX x NY torus of OTCA metapixel tiles (each 2048x2048 cells)
 * for exactly one metapixel period (35328 generations), starting from the
 * phase-0 tile bitmaps in variants.bin (8 variants, 2048 rows x 32 uint64).
 *
 * Every K generations, each tile's 2048x2048 bitmap and its "birth parity"
 * bitmap (parity of the number of births of each cell since generation 0) are
 * hash-consed into a quadtree whose leaves are 64x64 blocks:
 *   outprefix.blocks : unique leaf blocks, 64 uint64 each (row-major rows)
 *   outprefix.n1..n5 : unique nodes per level (4 uint32 child ids: NW NE SW SE)
 *   outprefix.roots  : per checkpoint, per tile: uint32 cur-root, uint32 parity-root
 *   outprefix.final  : 64-bit hashes of each tile at generation 35328
 * Block id 0 / node id 0 at every level are the all-dead block/node.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;
#define T 2048
#define TW 32
#define PERIOD 35328

static int NX, NY, H, W;
static u64 *A, *B, *Pb;
#define AT(P, y, x) (P)[(size_t)((y) + 1) * (W + 2) + (x) + 1]

static void wrap(u64 *P) {
    for (int y = 0; y < H; y++) { AT(P, y, -1) = AT(P, y, W - 1); AT(P, y, W) = AT(P, y, 0); }
    memcpy(&AT(P, -1, -1), &AT(P, H - 1, -1), sizeof(u64) * (W + 2));
    memcpy(&AT(P, H, -1), &AT(P, 0, -1), sizeof(u64) * (W + 2));
}

static void step(const u64 *restrict P, u64 *restrict Q, u64 *restrict PAR) {
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
                const u64 *u0 = s0 + ((yc + 2) % 3) * W, *u1 = s1 + ((yc + 2) % 3) * W;
                const u64 *m0 = s0 + ((yc + 3) % 3) * W, *m1 = s1 + ((yc + 3) % 3) * W;
                const u64 *d0 = s0 + ((yc + 4) % 3) * W, *d1 = s1 + ((yc + 4) % 3) * W;
                const u64 *cen = &AT(P, yc, 0);
                u64 *out = &AT(Q, yc, 0), *par = &AT(PAR, yc, 0);
                for (int x = 0; x < W; x++) {
                    u64 t0 = u0[x] ^ d0[x];
                    u64 c0 = (u0[x] & d0[x]) | (t0 & m0[x]);
                    t0 ^= m0[x];
                    u64 p = u1[x] ^ m1[x], g1 = u1[x] & m1[x];
                    u64 q = d1[x] ^ c0, g2 = d1[x] & c0;
                    u64 v0 = p ^ q, v1 = g1 ^ g2 ^ (p & q);
                    u64 nw = (t0 & v0 & ~v1) | (~t0 & ~v0 & v1 & cen[x]);
                    out[x] = nw;
                    par[x] ^= nw & ~cen[x];
                }
            }
        }
        free(s0); free(s1);
    }
}

static inline u64 mix(u64 h) { h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL; h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ULL; h ^= h >> 33; return h; }

/* ---- leaf blocks ---- */
static u64 *blk; static uint32_t nblk, capblk; static uint32_t *blkt; static u64 blkmask;
static u64 bhash(const u64 *b) { u64 h = 0x9E3779B97F4A7C15ULL; for (int i = 0; i < 64; i++) h = mix(h ^ b[i]) + i; return h; }
static void blk_rehash(void) {
    u64 sz = 1 << 16; while (sz < 2ULL * capblk) sz <<= 1;
    free(blkt); blkt = malloc(sz * 4); memset(blkt, 0xFF, sz * 4); blkmask = sz - 1;
    for (uint32_t i = 0; i < nblk; i++) { u64 h = bhash(blk + (size_t)i * 64) & blkmask; while (blkt[h] != 0xFFFFFFFFu) h = (h + 1) & blkmask; blkt[h] = i; }
}
static uint32_t blk_id(const u64 *b) {
    u64 h = bhash(b) & blkmask;
    while (blkt[h] != 0xFFFFFFFFu) { if (!memcmp(blk + (size_t)blkt[h] * 64, b, 512)) return blkt[h]; h = (h + 1) & blkmask; }
    if (nblk == capblk) { capblk *= 2; blk = realloc(blk, (size_t)capblk * 512); memcpy(blk + (size_t)nblk * 64, b, 512); nblk++; blk_rehash(); return nblk - 1; }
    memcpy(blk + (size_t)nblk * 64, b, 512); blkt[h] = nblk; return nblk++;
}
/* ---- nodes, levels 1..5 ---- */
typedef struct { uint32_t k[4]; } Key4;
static Key4 *nodev[6]; static uint32_t nnode[6], capnode[6]; static uint32_t *nodet[6]; static u64 nodemask[6];
static inline u64 khash(const Key4 *k) { return mix(((u64)k->k[0] << 32 | k->k[1]) ^ mix((u64)k->k[2] << 32 | k->k[3])); }
static void node_rehash(int L) {
    u64 sz = 1 << 12; while (sz < 2ULL * capnode[L]) sz <<= 1;
    free(nodet[L]); nodet[L] = malloc(sz * 4); memset(nodet[L], 0xFF, sz * 4); nodemask[L] = sz - 1;
    for (uint32_t i = 0; i < nnode[L]; i++) { u64 h = khash(&nodev[L][i]) & nodemask[L]; while (nodet[L][h] != 0xFFFFFFFFu) h = (h + 1) & nodemask[L]; nodet[L][h] = i; }
}
static uint32_t node_id(int L, Key4 k) {
    u64 h = khash(&k) & nodemask[L];
    while (nodet[L][h] != 0xFFFFFFFFu) { if (!memcmp(&nodev[L][nodet[L][h]], &k, 16)) return nodet[L][h]; h = (h + 1) & nodemask[L]; }
    if (nnode[L] == capnode[L]) { capnode[L] *= 2; nodev[L] = realloc(nodev[L], (size_t)capnode[L] * 16); nodev[L][nnode[L]++] = k; node_rehash(L); return nnode[L] - 1; }
    nodev[L][nnode[L]] = k; nodet[L][h] = nnode[L]; return nnode[L]++;
}
static uint32_t canon(const u64 *P, int tx, int ty) {
    uint32_t ids[32 * 32]; u64 b[64];
    for (int by = 0; by < 32; by++)
        for (int bx = 0; bx < 32; bx++) {
            for (int r = 0; r < 64; r++) b[r] = AT(P, ty * T + by * 64 + r, tx * TW + bx);
            ids[by * 32 + bx] = blk_id(b);
        }
    int n = 32;
    for (int L = 1; L <= 5; L++) {
        int m = n / 2;
        for (int y = 0; y < m; y++) for (int x = 0; x < m; x++) {
            Key4 k = {{ids[(2 * y) * n + 2 * x], ids[(2 * y) * n + 2 * x + 1], ids[(2 * y + 1) * n + 2 * x], ids[(2 * y + 1) * n + 2 * x + 1]}};
            ids[y * m + x] = node_id(L, k);
        }
        n = m;
    }
    return ids[0];
}
static u64 tile_hash(const u64 *P, int tx, int ty) {
    u64 h = 0x9E3779B97F4A7C15ULL;
    for (int y = 0; y < T; y++) {
        const u64 *r = &AT(P, ty * T + y, tx * TW);
        for (int x = 0; x < TW; x++) { h ^= r[x]; h *= 0xFF51AFD7ED558CCDULL; h ^= h >> 29; }
    }
    return h;
}

static void save(const char *prefix, const char *ext, const void *p, size_t sz, size_t n) {
    char fn[600]; snprintf(fn, sizeof fn, "%s.%s", prefix, ext);
    FILE *f = fopen(fn, "wb"); if (!f) { perror(fn); exit(1); }
    fwrite(p, sz, n, f); fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: gen_db NX NY layout.bin variants.bin K outprefix\n"); return 1; }
    NX = atoi(argv[1]); NY = atoi(argv[2]); H = NY * T; W = NX * TW;
    int K = atoi(argv[5]); const char *prefix = argv[6];
    if (PERIOD % K) { fprintf(stderr, "K must divide %d\n", PERIOD); return 1; }
    size_t cells = (size_t)(H + 2) * (W + 2);
    A = calloc(cells, 8); B = calloc(cells, 8); Pb = calloc(cells, 8);
    if (!A || !B || !Pb) { fprintf(stderr, "out of memory\n"); return 1; }
    unsigned char *lay = malloc(NX * NY);
    FILE *f = fopen(argv[3], "rb"); if (!f || fread(lay, 1, NX * NY, f) != (size_t)(NX * NY)) { perror("layout"); return 1; } fclose(f);
    f = fopen(argv[4], "rb"); if (!f) { perror("variants"); return 1; }
    u64 *vb = malloc((size_t)8 * T * TW * 8);
    if (fread(vb, (size_t)T * TW * 8, 8, f) != 8) { perror("variants"); return 1; } fclose(f);
    for (int ty = 0; ty < NY; ty++) for (int tx = 0; tx < NX; tx++)
        for (int y = 0; y < T; y++)
            memcpy(&AT(A, ty * T + y, tx * TW), vb + ((size_t)lay[ty * NX + tx] * T + y) * TW, TW * 8);
    free(vb);

    capblk = 1 << 16; blk = malloc((size_t)capblk * 512); nblk = 0; blk_rehash();
    for (int L = 1; L <= 5; L++) { capnode[L] = 1 << 12; nodev[L] = malloc((size_t)capnode[L] * 16); nnode[L] = 0; node_rehash(L); }
    { u64 z[64] = {0}; blk_id(z); uint32_t n = 0; for (int L = 1; L <= 5; L++) { Key4 k = {{n, n, n, n}}; n = node_id(L, k); } }

    char fn[600]; snprintf(fn, sizeof fn, "%s.roots", prefix);
    FILE *fr = fopen(fn, "wb");
    uint32_t *roots = malloc(8 * NX * NY);
    for (long g = 0; g <= PERIOD; g++) {
        if (g % K == 0 && g < PERIOD) {
            for (int i = 0; i < NX * NY; i++) {
                roots[2 * i] = canon(A, i % NX, i / NX);
                roots[2 * i + 1] = canon(Pb, i % NX, i / NX);
            }
            fwrite(roots, 4, 2 * NX * NY, fr); fflush(fr);
            fprintf(stderr, "gen %ld blocks %u nodes %u %u %u %u %u\n", g, nblk, nnode[1], nnode[2], nnode[3], nnode[4], nnode[5]);
        }
        if (g == PERIOD) break;
        wrap(A); step(A, B, Pb);
        u64 *t = A; A = B; B = t;
    }
    fclose(fr);
    u64 *fh = malloc(8 * NX * NY);
    for (int i = 0; i < NX * NY; i++) fh[i] = tile_hash(A, i % NX, i / NX);
    save(prefix, "final", fh, 8, NX * NY);
    save(prefix, "blocks", blk, 512, nblk);
    for (int L = 1; L <= 5; L++) { char e[8]; snprintf(e, sizeof e, "n%d", L); save(prefix, e, nodev[L], 16, nnode[L]); }
    fprintf(stderr, "done\n");
    return 0;
}
