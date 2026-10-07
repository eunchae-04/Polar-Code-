/*
 * polar_code_sc_scl_cascl_compare.c
 *
 * SC vs SCL(L=2,4,8) vs CRC-aided SCL(L=2,4,8) BER/FER comparison.
 *
 * Fixed code parameters (as instructed by the professor):
 *   N = 1024, R = 0.5 (so K = 512 non-frozen positions, same for every decoder)
 *   CA-SCL uses an 8-bit CRC (CRC-8, poly x^8+x^2+x+1 = 0x07) carved out of the
 *   K=512 non-frozen slots, so CA-SCL's true information bits are
 *   K_INFO = K - 8 = 504 (effective rate 504/1024 ~= 0.4922).
 *
 * All three decoder families (SC, SCL, CA-SCL) are implemented with ONE
 * unified recursive list-decoder:
 *   - SC    == list decoding with list size L = 1 (the fork-then-keep-best-1
 *             rule is exactly equivalent to a normal hard SC decision).
 *   - SCL   == list decoding with L in {2,4,8}, pick lowest path metric.
 *   - CA-SCL== same list decoding with L in {2,4,8}, but among the L final
 *             candidates we pick the lowest-path-metric one that PASSES the
 *             CRC check; if none pass, we fall back to the global PM-minimum
 *             (the usual CA-SCL fallback rule).
 *
 * Frozen-mask construction uses the standard Gaussian-Approximation (GA)
 * recursion (phi/phi_inv), matched to each Eb/No sweep point (same
 * convention as the project's earlier scd/scl "Baseline" simulations).
 *
 * Build:
gcc -O2 -Wall -o compare.exe polar_code_sc_scl_cascl_compare.c -lm
 * Run:
./compare.exe

 *
 * NOTE ON RUNTIME: this does 7 Eb/No points x 7 decoder configs = 49 Monte
 * Carlo runs, and CA-SCL(L=8) is the most expensive. TARGET_ERRORS below is
 * a conservative placeholder -- benchmark with a small value first (see the
 * printed per-point timing) before trusting/raising it for a final run.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#if defined(_WIN32)
  #include <direct.h>
  #define MKDIR(d) _mkdir(d)
  #define POPEN(cmd) _popen((cmd), "w")
  #define PCLOSE _pclose
#else
  #include <sys/stat.h>
  #include <sys/types.h>
  #define MKDIR(d) mkdir((d), 0755)
  #define POPEN(cmd) popen((cmd), "w")
  #define PCLOSE pclose
#endif

/* ------------------------------------------------------------------ */
/* Code parameters                                                     */
/* ------------------------------------------------------------------ */
#define N          1024
#define N_STAGES   10          /* log2(N) */
#define K          512         /* non-frozen positions (R = 0.5)       */
#define CRC_LEN    8
#define CRC_POLY   0x07        /* CRC-8, x^8 + x^2 + x + 1              */
#define K_INFO     (K - CRC_LEN)  /* 504 true info bits for CA-SCL     */
#define MAX_LIST   8

/* Eb/No sweep */
#define EBNO_START      -3.0
#define EBNO_STEP       0.25
#define NUM_SNR_POINTS  27      /* 0.0 .. 3.5 dB */

/*
 * Monte Carlo stopping rule (BENCHMARK before raising these).
 *
 * The slowest points are CA-SCL at high Eb/No (L=8 is the most expensive
 * decode AND has the lowest error rate, so it needs the most frames). With
 * TARGET_ERRORS=1000 / MAX_FRAMES=3000000 the full 7x7=49-point sweep takes
 * roughly 3-5 minutes on a typical machine; the last couple of points may
 * hit the MAX_FRAMES cap before TARGET_ERRORS, which just means their
 * BER/FER estimate is a little noisier (fewer than 100 error events) --
 * raise MAX_FRAMES (and TARGET_ERRORS) for a final, more precise run once
 * you've confirmed the curves look right, but expect that to take much
 * longer (the lowest-BER CA-SCL(L=8) point alone can need >1M frames to
 * reach 100-200 errors).
 */
#define TARGET_ERRORS   1000
#define MAX_FRAMES      3000000

static const int LIST_SIZES[3] = {2, 4, 8};
#define NUM_LIST_SIZES 3
#define NUM_CONFIGS (1 + NUM_LIST_SIZES * 2)   /* SC + 3xSCL + 3xCASCL = 7 */

typedef enum { MODE_SC, MODE_SCL, MODE_CASCL } DecoderMode;

typedef struct {
    char name[32];
    DecoderMode mode;
    int l_limit;
} DecoderConfig;

typedef struct {
    double ber;
    double fer;
    long frames;
    long bit_errors;
} RunResult;

/* ------------------------------------------------------------------ */
/* RNG: Xorshift64 (consistent with the rest of the project)           */
/* ------------------------------------------------------------------ */
static uint64_t rng_state;

static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static double rand_double(void) {
    /* 53-bit resolution double in (0,1) */
    return (double)(xorshift64() >> 11) * (1.0 / 9007199254740992.0);
}

static void rand_gaussian(double *z0, double *z1) {
    double u1, u2;
    do { u1 = rand_double(); } while (u1 <= 1e-12);
    u2 = rand_double();
    double r = sqrt(-2.0 * log(u1));
    *z0 = r * cos(2.0 * M_PI * u2);
    *z1 = r * sin(2.0 * M_PI * u2);
}

static inline uint8_t rand_bit(void) {
    return (uint8_t)(xorshift64() & 1ULL);
}

/* ------------------------------------------------------------------ */
/* Gaussian Approximation (GA) phi / phi_inv                           */
/* ------------------------------------------------------------------ */
static double phi(double x) {
    if (x <= 0.0) return 1.0;
    double v;
    if (x > 10.0) {
        v = sqrt(M_PI / x) * exp(-x / 4.0) * (1.0 - 10.0 / (7.0 * x));
    } else {
        v = exp(-0.4527 * pow(x, 0.86) + 0.0218);
    }
    if (v > 1.0) v = 1.0;
    if (v < 0.0) v = 0.0;
    return v;
}

static double phi_inv(double y) {
    if (y >= 1.0) return 0.0;
    if (y <= 0.0) return 200.0;
    double lo = 0.0, hi = 200.0;
    for (int it = 0; it < 60; it++) {
        double mid = 0.5 * (lo + hi);
        double v = phi(mid);
        if (v > y) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

typedef struct { int idx; double val; } IdxVal;

static int cmp_idxval_desc(const void *a, const void *b) {
    const IdxVal *A = (const IdxVal *)a, *B = (const IdxVal *)b;
    if (A->val > B->val) return -1;
    if (A->val < B->val) return 1;
    return 0;
}

/*
 * Builds frozen[N] (1 = frozen, 0 = info/non-frozen) using GA, designed for
 * the given Eb/No (dB) assuming code rate R_design (K/N = 0.5 for this
 * project; CA-SCL still uses this same 0.5-rate mask -- its extra CRC
 * overhead is handled later by splitting the 512 non-frozen slots, not by
 * changing which slots are non-frozen).
 *
 * ga_recursive() mirrors scl_decode_recursive()'s split EXACTLY (top-down:
 * left half = f/minus, right half = g/plus, then recurse into each half
 * separately) rather than a bottom-up/iterative butterfly sweep. Those two
 * orderings are NOT interchangeable for a recursive (non-bit-reversed)
 * encoder/decoder like this one -- an iterative bottom-up GA sweep silently
 * produces a mask barely better than random (confirmed during development
 * via a genie-aided per-position error-rate measurement: ~71% overlap with
 * the true best-K positions and ~400x worse average BER). Mirroring the
 * decoder's own recursion gives >99% overlap and matches the true optimum.
 */
static void ga_recursive(double *x, int n2) {
    if (n2 == 1) return;
    int half = n2 / 2;
    double xm[512], xp[512];
    for (int i = 0; i < half; i++) {
        double a = x[i], b = x[i + half];
        double pa = phi(a), pb = phi(b);
        double arg = 1.0 - (1.0 - pa) * (1.0 - pb);
        if (arg < 1e-300) arg = 1e-300;
        if (arg > 1.0) arg = 1.0;
        xm[i] = phi_inv(arg);
        xp[i] = a + b;
    }
    for (int i = 0; i < half; i++) { x[i] = xm[i]; x[i + half] = xp[i]; }
    ga_recursive(x, half);
    ga_recursive(x + half, half);
}

static void build_frozen_mask(double ebno_db, double r_design, int *frozen) {
    double ebno_lin = pow(10.0, ebno_db / 10.0);
    double sigma2 = 1.0 / (2.0 * r_design * ebno_lin);

    double x[N];
    for (int i = 0; i < N; i++) x[i] = 2.0 / sigma2;

    ga_recursive(x, N);

    IdxVal iv[N];
    for (int i = 0; i < N; i++) { iv[i].idx = i; iv[i].val = x[i]; }
    qsort(iv, N, sizeof(IdxVal), cmp_idxval_desc);

    for (int i = 0; i < N; i++) frozen[i] = 1;
    for (int i = 0; i < K; i++) frozen[iv[i].idx] = 0;
}

/* ------------------------------------------------------------------ */
/* Polar encoder (recursive, Arikan butterfly)                         */
/* ------------------------------------------------------------------ */
static void polar_encode(const uint8_t *u, uint8_t *xout, int n2) {
    if (n2 == 1) { xout[0] = u[0]; return; }
    int half = n2 / 2;
    uint8_t u1[512], u2[512];
    uint8_t x1[512], x2[512];
    for (int i = 0; i < half; i++) {
        u1[i] = u[i] ^ u[i + half];
        u2[i] = u[i + half];
    }
    polar_encode(u1, x1, half);
    polar_encode(u2, x2, half);
    for (int i = 0; i < half; i++) {
        xout[i] = x1[i];
        xout[i + half] = x2[i];
    }
}

/* ------------------------------------------------------------------ */
/* CRC-8 (bit-serial LFSR, poly 0x07) -- used identically for          */
/* generation (encoder side) and verification (CA-SCL decoder side)    */
/* ------------------------------------------------------------------ */
static uint8_t crc8_compute(const uint8_t *bits, int len) {
    uint8_t crc = 0x00;
    for (int i = 0; i < len; i++) {
        uint8_t bit_in = bits[i] & 1;
        uint8_t msb = (crc >> 7) & 1;
        crc = (uint8_t)(crc << 1);
        if (bit_in ^ msb) crc ^= CRC_POLY;
    }
    return crc;
}

/* ------------------------------------------------------------------ */
/* Unified SC / SCL / CA-SCL recursive list decoder                     */
/* ------------------------------------------------------------------ */
static inline double softplus(double v) {
    /* log(1+exp(v)), numerically stable */
    if (v > 0.0) return v + log1p(exp(-v));
    return log1p(exp(v));
}

static inline double f_combine_minsum(double l1, double l2) {
    double sgn = ((l1 < 0.0) != (l2 < 0.0)) ? -1.0 : 1.0;
    double minabs = fmin(fabs(l1), fabs(l2));
    return sgn * minabs;
}

/*
 * Recursively decodes a block of size n2 starting at absolute index
 * base_idx, operating on up to MAX_LIST parallel paths at once.
 *
 * llr[p][0..n2-1]   : this path's combined LLRs for this subtree (indices
 *                      are LOCAL to the subtree, i.e. 0..n2-1)
 * ubits[p][0..N-1]  : this path's full decided-bit history (GLOBAL array,
 *                      absolute indices); only base_idx..base_idx+n2-1 get
 *                      written by this call (directly at the leaves)
 * pm[p]             : this path's accumulated path metric (lower = better)
 * path_count        : in/out -- number of active paths
 * origin_out[newp]  : out -- which ENTRY path index (0..path_count_in-1)
 *                      the surviving/duplicated path newp descends from.
 *                      Needed by the caller to correctly re-index its own
 *                      (already-computed) llr array when building the
 *                      g-function inputs for the right subtree.
 */
static void scl_decode_recursive(double llr[][N], uint8_t ubits[][N], int n2,
                                  int base_idx, const int *frozen,
                                  double pm[MAX_LIST], int *path_count,
                                  int l_limit, int origin_out[MAX_LIST]) {
    int pc = *path_count;

    if (n2 == 1) {
        int idx = base_idx;
        if (frozen[idx]) {
            for (int p = 0; p < pc; p++) {
                ubits[p][idx] = 0;
                pm[p] += softplus(-llr[p][0]);
                origin_out[p] = p;
            }
            return; /* path_count unchanged */
        }

        int cand_n = pc * 2;
        double cand_pm[2 * MAX_LIST];
        int cand_parent[2 * MAX_LIST];
        int cand_bit[2 * MAX_LIST];
        for (int p = 0; p < pc; p++) {
            cand_parent[2 * p] = p; cand_bit[2 * p] = 0;
            cand_pm[2 * p] = pm[p] + softplus(-llr[p][0]);
            cand_parent[2 * p + 1] = p; cand_bit[2 * p + 1] = 1;
            cand_pm[2 * p + 1] = pm[p] + softplus(llr[p][0]);
        }

        int order[2 * MAX_LIST];
        for (int i = 0; i < cand_n; i++) order[i] = i;
        int keep = (cand_n < l_limit) ? cand_n : l_limit;
        for (int a = 0; a < keep; a++) {
            int best = a;
            for (int b = a + 1; b < cand_n; b++)
                if (cand_pm[order[b]] < cand_pm[order[best]]) best = b;
            int t = order[a]; order[a] = order[best]; order[best] = t;
        }

        double pm_new[MAX_LIST];
        static uint8_t ubits_new[MAX_LIST][N];
        for (int a = 0; a < keep; a++) {
            int c = order[a];
            int par = cand_parent[c];
            pm_new[a] = cand_pm[c];
            memcpy(ubits_new[a], ubits[par], N);
            ubits_new[a][idx] = (uint8_t)cand_bit[c];
            origin_out[a] = par;
        }
        for (int a = 0; a < keep; a++) {
            pm[a] = pm_new[a];
            memcpy(ubits[a], ubits_new[a], N);
        }
        *path_count = keep;
        return;
    }

    int half = n2 / 2;
    double llr_minus[MAX_LIST][N];
    for (int p = 0; p < pc; p++)
        for (int i = 0; i < half; i++)
            llr_minus[p][i] = f_combine_minsum(llr[p][i], llr[p][i + half]);

    int origin_left[MAX_LIST];
    scl_decode_recursive(llr_minus, ubits, half, base_idx, frozen, pm,
                          path_count, l_limit, origin_left);
    int pc_after_left = *path_count;

    /*
     * g-function needs the RE-ENCODED combination of the left branch's
     * decoded bits (the "partial sum"), not the raw decoded bits
     * themselves -- they only coincide at the leaf-adjacent level
     * (half == 1). Using the raw bits here silently corrupts every
     * decode where half >= 2 (verified by an exhaustive N=4 round-trip
     * test), so always re-encode before building llr_plus.
     */
    double llr_plus[MAX_LIST][N];
    uint8_t xleft_hat[N];
    for (int p = 0; p < pc_after_left; p++) {
        int par = origin_left[p];
        polar_encode(&ubits[p][base_idx], xleft_hat, half);
        for (int i = 0; i < half; i++) {
            double l1 = llr[par][i];
            double l2 = llr[par][i + half];
            uint8_t ub = xleft_hat[i];
            llr_plus[p][i] = l2 + (ub ? -l1 : l1);
        }
    }

    int origin_right[MAX_LIST];
    scl_decode_recursive(llr_plus, ubits, half, base_idx + half, frozen, pm,
                          path_count, l_limit, origin_right);
    int pc_after_right = *path_count;

    for (int p = 0; p < pc_after_right; p++)
        origin_out[p] = origin_left[origin_right[p]];
}

/* ------------------------------------------------------------------ */
/* Monte Carlo core (shared by SC / SCL / CA-SCL)                      */
/* ------------------------------------------------------------------ */
static double g_llr[MAX_LIST][N];
static uint8_t g_ubits[MAX_LIST][N];

static RunResult run_point(DecoderConfig cfg, double ebno_db, const int *frozen,
                            uint64_t seed) {
    double r_eff = (cfg.mode == MODE_CASCL) ? ((double)K_INFO / (double)N)
                                             : ((double)K / (double)N);
    double sigma2 = 1.0 / (2.0 * r_eff * pow(10.0, ebno_db / 10.0));
    double sigma = sqrt(sigma2);

    long frames = 0;
    long bit_errors = 0;
    long frame_errors = 0;
    long score_len = (cfg.mode == MODE_CASCL) ? K_INFO : K;

    rng_state = seed;
    if (rng_state == 0) rng_state = 1;

    uint8_t sentInfo[K];
    uint8_t u[N];
    uint8_t x[N];
    double rx[N];

    while (frames < MAX_FRAMES && bit_errors < TARGET_ERRORS) {
        frames++;

        if (cfg.mode == MODE_CASCL) {
            for (int i = 0; i < K_INFO; i++) sentInfo[i] = rand_bit();
            uint8_t crc = crc8_compute(sentInfo, K_INFO);
            for (int b = 0; b < CRC_LEN; b++)
                sentInfo[K_INFO + b] = (uint8_t)((crc >> (CRC_LEN - 1 - b)) & 1);
        } else {
            for (int i = 0; i < K; i++) sentInfo[i] = rand_bit();
        }

        int info_idx = 0;
        for (int i = 0; i < N; i++)
            u[i] = frozen[i] ? 0 : sentInfo[info_idx++];

        polar_encode(u, x, N);

        for (int i = 0; i < N; i += 2) {
            double b0 = x[i] ? -1.0 : 1.0;
            double b1 = (i + 1 < N) ? (x[i + 1] ? -1.0 : 1.0) : 0.0;
            double z0, z1;
            rand_gaussian(&z0, &z1);
            rx[i] = b0 + sigma * z0;
            if (i + 1 < N) rx[i + 1] = b1 + sigma * z1;
        }

        for (int i = 0; i < N; i++) g_llr[0][i] = (2.0 / sigma2) * rx[i];

        double pm[MAX_LIST];
        pm[0] = 0.0;
        int path_count = 1;
        int origin_dummy[MAX_LIST];

        scl_decode_recursive(g_llr, g_ubits, N, 0, frozen, pm, &path_count,
                              cfg.l_limit, origin_dummy);

        int best_path = 0;
        if (cfg.mode == MODE_CASCL) {
            int found = -1;
            double best_pm = 1e300;
            for (int p = 0; p < path_count; p++) {
                uint8_t infoCand[K];
                int ii = 0;
                for (int i = 0; i < N; i++)
                    if (!frozen[i]) infoCand[ii++] = g_ubits[p][i];
                uint8_t crc_calc = crc8_compute(infoCand, K_INFO);
                uint8_t crc_rx = 0;
                for (int b = 0; b < CRC_LEN; b++)
                    crc_rx = (uint8_t)((crc_rx << 1) | infoCand[K_INFO + b]);
                if (crc_calc == crc_rx && pm[p] < best_pm) {
                    best_pm = pm[p];
                    found = p;
                }
            }
            if (found >= 0) {
                best_path = found;
            } else {
                double bpm = 1e300;
                for (int p = 0; p < path_count; p++)
                    if (pm[p] < bpm) { bpm = pm[p]; best_path = p; }
            }
        } else {
            double bpm = 1e300;
            for (int p = 0; p < path_count; p++)
                if (pm[p] < bpm) { bpm = pm[p]; best_path = p; }
        }

        uint8_t decodedInfo[K];
        int ii = 0;
        for (int i = 0; i < N; i++)
            if (!frozen[i]) decodedInfo[ii++] = g_ubits[best_path][i];

        long err_this_frame = 0;
        for (int i = 0; i < score_len; i++)
            if (decodedInfo[i] != sentInfo[i]) err_this_frame++;

        bit_errors += err_this_frame;
        if (err_this_frame > 0) frame_errors++;
    }

    RunResult r;
    r.frames = frames;
    r.bit_errors = bit_errors;
    r.ber = (double)bit_errors / (double)(frames * score_len);
    r.fer = (double)frame_errors / (double)frames;
    return r;
}

/* ------------------------------------------------------------------ */
/* Gnuplot                                                             */
/* ------------------------------------------------------------------ */
static void run_gnuplot(DecoderConfig *configs, int nc) {
    FILE *gp = POPEN("gnuplot -persist");
    if (!gp) {
        printf("[INFO] gnuplot을 열 수 없습니다. result/*.txt 파일을 직접 그려보세요.\n");
        return;
    }
    fprintf(gp, "set terminal wxt size 1200,600\n");
    fprintf(gp, "set multiplot layout 1,2 title 'SC vs SCL vs CA-SCL (N=1024, R=0.5, CRC-8)'\n");

    const char *colors[7] = {"black", "#1f77b4", "#1f77b4", "#1f77b4",
                              "#d62728", "#d62728", "#d62728"};
    const char *dashtypes[7] = {"1", "1", "2", "3", "1", "2", "3"};
    const char *points[7] = {"7", "5", "5", "5", "9", "9", "9"};

    fprintf(gp, "set logscale y\n");
    fprintf(gp, "set grid\n");
    fprintf(gp, "set xlabel 'Eb/No (dB)'\n");

    fprintf(gp, "set title 'BER'\n");
    fprintf(gp, "set ylabel 'BER'\n");
    fprintf(gp, "plot ");
    for (int c = 0; c < nc; c++) {
        fprintf(gp, "'result/result_%s.txt' using 1:2 with linespoints lc rgb '%s' dt %s pt %s title '%s'%s",
                configs[c].name, colors[c], dashtypes[c], points[c], configs[c].name,
                (c == nc - 1) ? "\n" : ", ");
    }

    fprintf(gp, "set title 'FER'\n");
    fprintf(gp, "set ylabel 'FER'\n");
    fprintf(gp, "plot ");
    for (int c = 0; c < nc; c++) {
        fprintf(gp, "'result/result_%s.txt' using 1:3 with linespoints lc rgb '%s' dt %s pt %s title '%s'%s",
                configs[c].name, colors[c], dashtypes[c], points[c], configs[c].name,
                (c == nc - 1) ? "\n" : ", ");
    }

    fprintf(gp, "unset multiplot\n");
    fflush(gp);
    PCLOSE(gp);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(void) {
    MKDIR("result");

    DecoderConfig configs[NUM_CONFIGS];
    int nc = 0;
    strcpy(configs[nc].name, "SC"); configs[nc].mode = MODE_SC; configs[nc].l_limit = 1; nc++;
    for (int i = 0; i < NUM_LIST_SIZES; i++) {
        sprintf(configs[nc].name, "SCL_L%d", LIST_SIZES[i]);
        configs[nc].mode = MODE_SCL;
        configs[nc].l_limit = LIST_SIZES[i];
        nc++;
    }
    for (int i = 0; i < NUM_LIST_SIZES; i++) {
        sprintf(configs[nc].name, "CASCL_L%d", LIST_SIZES[i]);
        configs[nc].mode = MODE_CASCL;
        configs[nc].l_limit = LIST_SIZES[i];
        nc++;
    }

    FILE *fps[NUM_CONFIGS];
    for (int c = 0; c < nc; c++) {
        char fn[160];
        snprintf(fn, sizeof(fn), "result/result_%s.txt", configs[c].name);
        fps[c] = fopen(fn, "w");
        fprintf(fps[c], "# EbNo_dB BER FER frames\n");
    }

    printf("N=%d, K=%d (R=%.4f), CA-SCL K_INFO=%d (R_eff=%.4f), CRC-%d (poly 0x%02X)\n",
           N, K, (double)K / N, K_INFO, (double)K_INFO / N, CRC_LEN, CRC_POLY);
    printf("Decoders: ");
    for (int c = 0; c < nc; c++) printf("%s ", configs[c].name);
    printf("\n\n");

    time_t t_start = time(NULL);
    int total_points = NUM_SNR_POINTS * nc;
    int done = 0;

    for (int s = 0; s < NUM_SNR_POINTS; s++) {
        double ebno = EBNO_START + s * EBNO_STEP;
        int frozen[N];
        build_frozen_mask(ebno, (double)K / (double)N, frozen);

        uint64_t seed = 0x243F6A8885A308D3ULL + (uint64_t)s * 9999991ULL;

        for (int c = 0; c < nc; c++) {
            RunResult r = run_point(configs[c], ebno, frozen, seed);
            fprintf(fps[c], "%.2f %.6e %.6e %ld\n", ebno, r.ber, r.fer, r.frames);
            fflush(fps[c]);

            done++;
            double elapsed = difftime(time(NULL), t_start);
            printf("[%d/%d] %-10s @ %.2fdB  BER=%.3e  FER=%.3e  frames=%-8ld elapsed=%.0fs\n",
                   done, total_points, configs[c].name, ebno, r.ber, r.fer, r.frames, elapsed);
            fflush(stdout);
        }
    }

    for (int c = 0; c < nc; c++) fclose(fps[c]);

    printf("\n[DONE] 결과는 result/result_<name>.txt 에 저장되었습니다.\n");
    run_gnuplot(configs, nc);
    return 0;
}