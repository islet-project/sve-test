/*
 * sve-test - SVE context save/restore stress test for a Realm Linux VM
 *
 * The test verifies that the SVE register file (Z0-Z31, P0-P15, FFR) and
 * effective vector length (ZCR_EL1) survive context switches and CPU
 * migrations without corruption. It works in two layers:
 *
 *  1. Pattern layer: all Z registers, all P registers, and FFR are filled
 *     with deterministic byte patterns that vary across register number,
 *     byte lane, thread ID, and iteration. Fill, busy-spin preemption, VL
 *     verification, and register read-back execute inside a single
 *     asm volatile block with full SVE clobbers so the compiler cannot
 *     schedule any FP/NEON/SVE instructions inside the window.
 *
 *  2. Compute layer: SVE intrinsic kernels (predicated uint8 add + store +
 *     reduction, and float32 FMA reduction) are executed and compared
 *     against reference results.
 *
 * Multiple worker threads plus an asynchronous CPU migrator thread force
 * guest-kernel SVE context switching, IPI-driven cross-vCPU migrations while
 * SVE state is live in EL0, and (on Realm exits) RMM-level SIMD save/restore.
 *
 * Usage: ./sve-test [iterations] [threads]
 * Exit code: 0 = PASS, 1 = FAIL
 */

#define _GNU_SOURCE

#include <arm_sve.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

#define NUM_Z_REGS 32
#define NUM_P_REGS 16
#define Z_PATTERN_BASE 0x40
#define P_PATTERN_BASE 0x80
#define SPIN_ITERS 5000000UL

static size_t vl_bytes; /* current SVE vector length in bytes */
static size_t ps_bytes; /* predicate register size in bytes (VL/8) */

/* compute-layer reference data (read-only after main() init) */
#define DATA_LEN (1 << 20)
static uint8_t *data_a, *data_b, *expected_dst_u8;
static uint64_t expected_u8_sum;
static float *fa, *fb;
#define F32_LEN (DATA_LEN / 4) /* fa/fb hold F32_LEN floats */
static float expected_f32_sum;

/* failure accounting */
static pthread_mutex_t fail_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned long fails;
static unsigned long iters_done;

#define STR2(x) #x
#define STR(x) STR2(x)

static uint64_t rdvl_bytes(void)
{
    uint64_t v;
    asm volatile("rdvl %0, #1" : "=r"(v));
    return v;
}

static void dump_hex(const uint8_t *p, size_t len)
{
    size_t i;
    for (i = 0; i < len && i < 64; i++)
        printf("%02x ", p[i]);
    if (len > 64)
        printf("...");
    printf("\n");
}

/*
 * Generate per-(thread, iter, reg, byte_lane) expected patterns:
 *  - Z[n][b]: non-zero byte varying with register index, byte offset,
 *    thread_id, and iteration so cross-thread swaps, stale restores,
 *    and intra-register lane permutations are all detected.
 *  - P[n][b]: non-zero byte varying with predicate index, byte offset,
 *    thread_id, and iteration.
 *  - FFR: architecturally canonical monotonic bitmask (bits 0..k-1 set to 1,
 *    remaining bits 0, with 1 <= k <= ps_bytes*8) varying with thread_id and
 *    iteration. (Arm SVE requires WRFFR inputs to be monotonic from bit 0,
 *    otherwise RDFFR is CONSTRAINED UNPREDICTABLE.)
 */
static void build_patterns(int thread_id, unsigned long iter, uint8_t *zp,
                           uint8_t *pp, uint8_t *fp)
{
    size_t n, b;
    size_t total_ffr_bits = ps_bytes * 8;
    size_t ones_bits =
        1 + (((size_t)thread_id * 13U + iter * 7U) % total_ffr_bits);
    uint8_t z_seed = (uint8_t)((unsigned)thread_id * 37U + iter * 13U);
    uint8_t p_seed = (uint8_t)((unsigned)thread_id * 29U + iter * 19U);

    for (n = 0; n < NUM_Z_REGS; n++) {
        uint8_t *dst = zp + n * vl_bytes;
        for (b = 0; b < vl_bytes; b++) {
            uint8_t v =
                (uint8_t)((Z_PATTERN_BASE + n) + (b * 7U) + z_seed);
            dst[b] = (v != 0) ? v : 0xffU;
        }
    }

    for (n = 0; n < NUM_P_REGS; n++) {
        uint8_t *dst = pp + n * ps_bytes;
        for (b = 0; b < ps_bytes; b++) {
            uint8_t v =
                (uint8_t)((P_PATTERN_BASE | n) + (b * 11U) + p_seed);
            dst[b] = (v != 0) ? v : (uint8_t)(P_PATTERN_BASE | n);
        }
    }

    for (b = 0; b < ps_bytes; b++) {
        if ((b + 1) * 8 <= ones_bits)
            fp[b] = 0xffU;
        else if (b * 8 >= ones_bits)
            fp[b] = 0x00U;
        else
            fp[b] = (uint8_t)((1U << (ones_bits - b * 8)) - 1U);
    }
}

/*
 * Single self-contained asm volatile window:
 *  1. Fill Z0-Z31 from zp
 *  2. Fill FFR from fp (using p0 as scratch before loading P0-P15)
 *  3. Fill P0-P15 from pp
 *  4. Busy-spin for spin_iters iterations (pure integer register loop;
 *     no syscalls, no stack accesses, no compiler-inserted instructions)
 *  5. Check effective VL via rdvl; if it changed, skip storing Z/P/FFR
 *     (avoids buffer overflow if a broken hypervisor increased ZCR_EL1.LEN)
 *  6. Store Z0-Z31 to zv, P0-P15 to pv, and FFR to fv
 *
 * Declaring all Z0-Z31, P0-P15, and FFR in the clobber list informs GCC that
 * callee-saved d8-d15 (low 64 bits of z8-z15) are modified here, preventing
 * register-allocation corruption across the call while keeping all compiler
 * save/restore code strictly outside the fill -> spin -> readback window.
 */
__attribute__((noinline))
static uint64_t sve_fill_spin_readback(const uint8_t *zp, const uint8_t *pp,
                                       const uint8_t *fp, uint8_t *zv,
                                       uint8_t *pv, uint8_t *fv,
                                       unsigned long spin_iters)
{
    uint64_t ptr, vl_out;
    unsigned long cnt = spin_iters;

#define LOAD_Z_STEP(n)                                                         \
    "ld1b {z" STR(n) ".b}, p0/z, [%[ptr]]\n\t"                                 \
    "add %[ptr], %[ptr], %[vl]\n\t"
#define LOAD_P_STEP(n)                                                         \
    "ldr p" STR(n) ", [%[ptr]]\n\t"                                            \
    "add %[ptr], %[ptr], %[ps]\n\t"
#define STORE_Z_STEP(n)                                                        \
    "str z" STR(n) ", [%[ptr]]\n\t"                                            \
    "add %[ptr], %[ptr], %[vl]\n\t"
#define STORE_P_STEP(n)                                                        \
    "str p" STR(n) ", [%[ptr]]\n\t"                                            \
    "add %[ptr], %[ptr], %[ps]\n\t"

    asm volatile(
        /* 1. Load Z0-Z31 */
        "ptrue p0.b\n\t"
        "mov %[ptr], %[zp]\n\t"
        LOAD_Z_STEP(0)  LOAD_Z_STEP(1)  LOAD_Z_STEP(2)  LOAD_Z_STEP(3)
        LOAD_Z_STEP(4)  LOAD_Z_STEP(5)  LOAD_Z_STEP(6)  LOAD_Z_STEP(7)
        LOAD_Z_STEP(8)  LOAD_Z_STEP(9)  LOAD_Z_STEP(10) LOAD_Z_STEP(11)
        LOAD_Z_STEP(12) LOAD_Z_STEP(13) LOAD_Z_STEP(14) LOAD_Z_STEP(15)
        LOAD_Z_STEP(16) LOAD_Z_STEP(17) LOAD_Z_STEP(18) LOAD_Z_STEP(19)
        LOAD_Z_STEP(20) LOAD_Z_STEP(21) LOAD_Z_STEP(22) LOAD_Z_STEP(23)
        LOAD_Z_STEP(24) LOAD_Z_STEP(25) LOAD_Z_STEP(26) LOAD_Z_STEP(27)
        LOAD_Z_STEP(28) LOAD_Z_STEP(29) LOAD_Z_STEP(30)
        "ld1b {z31.b}, p0/z, [%[ptr]]\n\t"

        /* 2. Load FFR before P0-P15 (uses p0 as scratch) */
        "ldr p0, [%[fp]]\n\t"
        "wrffr p0.b\n\t"

        /* 3. Load P0-P15 */
        "mov %[ptr], %[pp]\n\t"
        LOAD_P_STEP(0)  LOAD_P_STEP(1)  LOAD_P_STEP(2)  LOAD_P_STEP(3)
        LOAD_P_STEP(4)  LOAD_P_STEP(5)  LOAD_P_STEP(6)  LOAD_P_STEP(7)
        LOAD_P_STEP(8)  LOAD_P_STEP(9)  LOAD_P_STEP(10) LOAD_P_STEP(11)
        LOAD_P_STEP(12) LOAD_P_STEP(13) LOAD_P_STEP(14)
        "ldr p15, [%[ptr]]\n\t"

        /* 4. Preemption busy-spin (no syscalls, no FP/SIMD) */
        "cbz %[cnt], 2f\n\t"
        "1:\n\t"
        "subs %[cnt], %[cnt], #1\n\t"
        "b.ne 1b\n\t"
        "2:\n\t"

        /* 5. Check effective VL before storing to zv/pv/fv */
        "rdvl %[vl_out], #1\n\t"
        "cmp %[vl_out], %[vl]\n\t"
        "b.ne 3f\n\t"

        /* 6. Store Z0-Z31, P0-P15, and FFR */
        "mov %[ptr], %[zv]\n\t"
        STORE_Z_STEP(0)  STORE_Z_STEP(1)  STORE_Z_STEP(2)  STORE_Z_STEP(3)
        STORE_Z_STEP(4)  STORE_Z_STEP(5)  STORE_Z_STEP(6)  STORE_Z_STEP(7)
        STORE_Z_STEP(8)  STORE_Z_STEP(9)  STORE_Z_STEP(10) STORE_Z_STEP(11)
        STORE_Z_STEP(12) STORE_Z_STEP(13) STORE_Z_STEP(14) STORE_Z_STEP(15)
        STORE_Z_STEP(16) STORE_Z_STEP(17) STORE_Z_STEP(18) STORE_Z_STEP(19)
        STORE_Z_STEP(20) STORE_Z_STEP(21) STORE_Z_STEP(22) STORE_Z_STEP(23)
        STORE_Z_STEP(24) STORE_Z_STEP(25) STORE_Z_STEP(26) STORE_Z_STEP(27)
        STORE_Z_STEP(28) STORE_Z_STEP(29) STORE_Z_STEP(30)
        "str z31, [%[ptr]]\n\t"

        "mov %[ptr], %[pv]\n\t"
        STORE_P_STEP(0)  STORE_P_STEP(1)  STORE_P_STEP(2)  STORE_P_STEP(3)
        STORE_P_STEP(4)  STORE_P_STEP(5)  STORE_P_STEP(6)  STORE_P_STEP(7)
        STORE_P_STEP(8)  STORE_P_STEP(9)  STORE_P_STEP(10) STORE_P_STEP(11)
        STORE_P_STEP(12) STORE_P_STEP(13) STORE_P_STEP(14)
        "str p15, [%[ptr]]\n\t"

        "rdffr p0.b\n\t"
        "str p0, [%[fv]]\n\t"
        "3:\n\t"
        : [ptr] "=&r"(ptr), [vl_out] "=&r"(vl_out), [cnt] "+r"(cnt)
        : [zp] "r"(zp), [pp] "r"(pp), [fp] "r"(fp),
          [zv] "r"(zv), [pv] "r"(pv), [fv] "r"(fv),
          [vl] "r"(vl_bytes), [ps] "r"(ps_bytes)
        : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15",
          "ffr", "cc", "memory");

#undef LOAD_Z_STEP
#undef LOAD_P_STEP
#undef STORE_Z_STEP
#undef STORE_P_STEP

    return vl_out;
}

static const char *const z_reg_names[NUM_Z_REGS] = {
    "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
    "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
    "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
    "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
};
static const char *const p_reg_names[NUM_P_REGS] = {
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
    "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15",
};

static void report_fail(const char *what, int thread_id, unsigned long iter,
                        const char *regname, const uint8_t *expect,
                        const uint8_t *got, size_t len)
{
    size_t off = 0;

    while (off < len && expect[off] == got[off])
        off++;

    pthread_mutex_lock(&fail_lock);
    fails++;
    printf("[FAIL] t%d iter %lu: %s: %s corrupted (len %zu, first diff at "
           "byte %zu)\n",
           thread_id, iter, what, regname, len, off);
    printf("  expected: ");
    dump_hex(expect, len);
    printf("  got     : ");
    dump_hex(got, len);
    pthread_mutex_unlock(&fail_lock);
}

/* Compare the read-back buffers against the expected patterns. */
__attribute__((noinline))
static int sve_compare_patterns(const char *what, int thread_id,
                                unsigned long iter, const uint8_t *zp,
                                const uint8_t *pp, const uint8_t *fp,
                                const uint8_t *zv, const uint8_t *pv,
                                const uint8_t *fv)
{
    int bad = 0, n;

    for (n = 0; n < NUM_Z_REGS; n++) {
        const uint8_t *e = zp + n * vl_bytes;
        const uint8_t *g = zv + n * vl_bytes;
        if (memcmp(e, g, vl_bytes) != 0) {
            report_fail(what, thread_id, iter, z_reg_names[n], e, g, vl_bytes);
            bad++;
        }
    }
    for (n = 0; n < NUM_P_REGS; n++) {
        const uint8_t *e = pp + n * ps_bytes;
        const uint8_t *g = pv + n * ps_bytes;
        if (memcmp(e, g, ps_bytes) != 0) {
            report_fail(what, thread_id, iter, p_reg_names[n], e, g, ps_bytes);
            bad++;
        }
    }
    if (memcmp(fp, fv, ps_bytes) != 0) {
        report_fail(what, thread_id, iter, "FFR", fp, fv, ps_bytes);
        bad++;
    }
    return bad;
}

/* Functional SVE kernels ------------------------------------------------- */

__attribute__((noinline))
static uint64_t sve_compute_u8(const uint8_t *a, const uint8_t *b,
                               uint8_t *dst, size_t len)
{
    uint64_t sum = 0;
    size_t i;
    for (i = 0; i < len;) {
        svbool_t pg = svwhilelt_b8(i, len);
        svuint8_t va = svld1_u8(pg, a + i);
        svuint8_t vb = svld1_u8(pg, b + i);
        svuint8_t vd = svadd_u8_x(pg, va, vb);
        svst1_u8(pg, dst + i, vd);
        sum += svaddv_u8(pg, vd);
        i += svcntb();
    }
    return sum;
}

__attribute__((noinline))
static float sve_compute_f32(const float *a, const float *b, size_t len)
{
    svfloat32_t acc = svdup_f32(0.0f);
    size_t i;
    for (i = 0; i < len;) {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        acc = svmla_f32_m(pg, acc, va, vb);
        i += svcntb() / 4;
    }
    return svaddv_f32(svptrue_b32(), acc);
}

static void report_compute_u64_fail(int thread_id, unsigned long iter,
                                    const char *what,
                                    unsigned long long expect,
                                    unsigned long long got)
{
    pthread_mutex_lock(&fail_lock);
    fails++;
    printf("[FAIL] t%d iter %lu: %s result corrupted: expected %llu, got %llu\n",
           thread_id, iter, what, expect, got);
    pthread_mutex_unlock(&fail_lock);
}

static void report_compute_f32_fail(int thread_id, unsigned long iter,
                                    const char *what, float expect, float got)
{
    uint32_t exp_bits, got_bits;
    memcpy(&exp_bits, &expect, sizeof(exp_bits));
    memcpy(&got_bits, &got, sizeof(got_bits));

    pthread_mutex_lock(&fail_lock);
    fails++;
    printf("[FAIL] t%d iter %lu: %s result corrupted: expected %.6f (0x%08x), "
           "got %.6f (0x%08x)\n",
           thread_id, iter, what, (double)expect, exp_bits, (double)got,
           got_bits);
    pthread_mutex_unlock(&fail_lock);
}

/* Preemption / cross-vCPU migration pressure -----------------------------
 *
 * IMPORTANT: no syscalls are allowed inside the pattern window.
 * Current arm64 kernels run fp_user_discard() from el0_svc() on EVERY
 * syscall: sve_flush_live() zeroes bits >= 128 of every Z register and
 * zeroes P0-P15/FFR.
 *
 * Moreover, calling sched_setaffinity(0, ...) from the worker thread before
 * filling patterns migrates the calling thread synchronously inside the
 * syscall before returning to userspace, and pinning to a single CPU would
 * forbid migration during the spin window.
 *
 * Instead, when running multi-threaded on >1 CPU, a dedicated migrator
 * thread asynchronously calls sched_setaffinity(worker_tid, ...) from the
 * outside while workers are mid-spin in EL0. This sends a kernel migration
 * IPI that preempts the running worker via el0_irq (preserving full SVE
 * state via fpsimd_save_user_state) and resumes it on another vCPU, while
 * alternating phases that pack multiple workers onto the same vCPU to force
 * same-CPU timer-tick preemption between threads with distinct SVE patterns.
 */
static int nr_cpus = 1;

struct thread_arg {
    int id;
    unsigned long iters;
    pid_t tid; /* published via atomic store while worker is active */
};

struct migrator_arg {
    struct thread_arg *targs;
    int nthreads;
    int stop;
};

static void *migrator_fn(void *argp)
{
    struct migrator_arg *m = argp;
    unsigned long step = 0;
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 500000L }; /* 500 us */

    while (!__atomic_load_n(&m->stop, __ATOMIC_ACQUIRE)) {
        int i;
        for (i = 0; i < m->nthreads; i++) {
            pid_t tid = __atomic_load_n(&m->targs[i].tid, __ATOMIC_ACQUIRE);
            cpu_set_t set;
            int cpu;

            if (tid <= 0)
                continue;

            CPU_ZERO(&set);
            if ((step & 1UL) == 0) {
                /* Pack all workers onto the same vCPU to force timer-IRQ
                 * context switches between threads with different patterns. */
                cpu = (int)((step / 2) % (unsigned)nr_cpus);
            } else {
                /* Rotate each worker onto a different vCPU via IPI while
                 * it is spinning in userspace with live SVE registers. */
                cpu = (int)((i + step) % (unsigned)nr_cpus);
            }
            CPU_SET(cpu, &set);
            sched_setaffinity(tid, sizeof(set), &set);
        }
        step++;
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static void *worker(void *argp)
{
    struct thread_arg *a = argp;
    unsigned long iter;
    uint8_t *zp, *pp, *fp;
    uint8_t *zv, *pv, *fv;
    uint8_t *dst_u8;

    /* Per-thread expected pattern buffers, read-back buffers, and compute
     * output buffer so no shared writable state races across threads. */
    zp = malloc(NUM_Z_REGS * vl_bytes);
    pp = malloc(NUM_P_REGS * ps_bytes);
    fp = malloc(ps_bytes);
    zv = malloc(NUM_Z_REGS * vl_bytes);
    pv = malloc(NUM_P_REGS * ps_bytes);
    fv = malloc(ps_bytes);
    dst_u8 = malloc(DATA_LEN);
    if (!zp || !pp || !fp || !zv || !pv || !fv || !dst_u8) {
        pthread_mutex_lock(&fail_lock);
        fails++;
        printf("[FAIL] t%d: out of memory allocating thread buffers\n", a->id);
        pthread_mutex_unlock(&fail_lock);
        free(zp);
        free(pp);
        free(fp);
        free(zv);
        free(pv);
        free(fv);
        free(dst_u8);
        return NULL;
    }

    __atomic_store_n(&a->tid, gettid(), __ATOMIC_RELEASE);

    for (iter = 0; iter < a->iters; iter++) {
        uint64_t vl_now, sum_u8;
        float sum_f32;

        /* 1. Build unique expected patterns for (thread_id, iter) */
        build_patterns(a->id, iter, zp, pp, fp);

        /* 2. On the first iteration, sanity-check the fill/readback
         * machinery itself with spin_iters = 0 before spinning. */
        if (iter == 0) {
            vl_now = sve_fill_spin_readback(zp, pp, fp, zv, pv, fv, 0);
            if (vl_now != vl_bytes) {
                pthread_mutex_lock(&fail_lock);
                fails++;
                printf("[FAIL] t%d iter %lu: selfcheck VL mismatch: expected "
                       "%zu, got %llu\n",
                       a->id, iter, vl_bytes, (unsigned long long)vl_now);
                pthread_mutex_unlock(&fail_lock);
                continue;
            }
            if (sve_compare_patterns("selfcheck", a->id, iter, zp, pp, fp, zv,
                                     pv, fv) != 0)
                continue;
        }

        /* 3. Fill SVE state, busy-spin for preemption/migration, verify VL,
         * and read back Z0-Z31, P0-P15, and FFR in a single asm block. */
        vl_now = sve_fill_spin_readback(zp, pp, fp, zv, pv, fv, SPIN_ITERS);
        if (vl_now != vl_bytes) {
            pthread_mutex_lock(&fail_lock);
            fails++;
            printf("[FAIL] t%d iter %lu: effective VL changed: expected "
                   "%zu, got %llu (hypervisor/RMM likely clobbered ZCR_EL1)\n",
                   a->id, iter, vl_bytes, (unsigned long long)vl_now);
            pthread_mutex_unlock(&fail_lock);
            continue;
        }

        /* 4. Compare read-back buffers against expected patterns */
        sve_compare_patterns("after-switch", a->id, iter, zp, pp, fp, zv, pv,
                             fv);

        /* 5. Functional SVE compute, compared against reference results */
        sum_u8 = sve_compute_u8(data_a, data_b, dst_u8, DATA_LEN);
        if (sum_u8 != expected_u8_sum)
            report_compute_u64_fail(a->id, iter, "sve_compute_u8 sum",
                                    expected_u8_sum, sum_u8);
        if (memcmp(dst_u8, expected_dst_u8, DATA_LEN) != 0)
            report_fail("sve_compute_u8 dst", a->id, iter, "dst_u8",
                        expected_dst_u8, dst_u8, DATA_LEN);

        sum_f32 = sve_compute_f32(fa, fb, F32_LEN);
        if (sum_f32 != expected_f32_sum)
            report_compute_f32_fail(a->id, iter, "sve_compute_f32",
                                    expected_f32_sum, sum_f32);

        /* 6. Verify the vector length did not change after compute */
        vl_now = rdvl_bytes();
        if (vl_now != vl_bytes) {
            pthread_mutex_lock(&fail_lock);
            fails++;
            printf("[FAIL] t%d iter %lu: VL changed: was %zu, now %llu\n",
                   a->id, iter, vl_bytes, (unsigned long long)vl_now);
            pthread_mutex_unlock(&fail_lock);
        }

        pthread_mutex_lock(&fail_lock);
        iters_done++;
        pthread_mutex_unlock(&fail_lock);
    }

    __atomic_store_n(&a->tid, 0, __ATOMIC_RELEASE);

    free(zp);
    free(pp);
    free(fp);
    free(zv);
    free(pv);
    free(fv);
    free(dst_u8);
    return NULL;
}

int main(int argc, char **argv)
{
    unsigned long iters = 1000;
    int nthreads;
    pthread_t *tids;
    struct thread_arg *targs;
    pthread_t migrator_tid;
    struct migrator_arg marg;
    int migrator_running = 0;
    int i, ret = 0;

    setvbuf(stdout, NULL, _IONBF, 0);

    nr_cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nr_cpus < 1)
        nr_cpus = 1;

    /* Default to 2x online CPUs (min 2) so multiple threads contend on each
     * vCPU unless overridden on the command line. */
    nthreads = (nr_cpus > 1) ? (nr_cpus * 2) : 2;

    if (argc > 1)
        iters = strtoul(argv[1], NULL, 0);
    if (argc > 2)
        nthreads = atoi(argv[2]);
    if (nthreads < 1)
        nthreads = 1;

    if (!(getauxval(AT_HWCAP) & (1UL << 22)) /* HWCAP_SVE */) {
        printf("SVE not available (HWCAP_SVE missing)\n");
        return 1;
    }

    /* Request the largest VL the kernel/EL2/RMM allows and mark it for
     * inheritance across pthread_create(). Note that PR_SVE_SET_VL returns
     * the new nonnegative configuration mask (not 0) on success. */
    {
        int set_vl = prctl(PR_SVE_SET_VL,
                           PR_SVE_VL_LEN_MASK | PR_SVE_VL_INHERIT, 0, 0, 0);
        if (set_vl < 0) {
            int cur_vl = prctl(PR_SVE_GET_VL, 0, 0, 0, 0);
            if (cur_vl >= 0) {
                int want = (cur_vl & PR_SVE_VL_LEN_MASK) | PR_SVE_VL_INHERIT;
                (void)prctl(PR_SVE_SET_VL, want, 0, 0, 0);
            }
        }
        vl_bytes = rdvl_bytes();
        ps_bytes = vl_bytes / 8;
        printf("SVE VL: %zu bytes (%zu-bit), predicate: %zu bytes\n", vl_bytes,
               vl_bytes * 8, ps_bytes);
    }

    /* compute-layer reference data */
    data_a = malloc(DATA_LEN);
    data_b = malloc(DATA_LEN);
    expected_dst_u8 = malloc(DATA_LEN);
    fa = malloc(F32_LEN * sizeof(float));
    fb = malloc(F32_LEN * sizeof(float));
    if (!data_a || !data_b || !expected_dst_u8 || !fa || !fb) {
        printf("out of memory\n");
        return 1;
    }
    for (i = 0; i < DATA_LEN; i++) {
        data_a[i] = (uint8_t)(i * 7 + 1);
        data_b[i] = (uint8_t)(i * 13 + 3);
        if (i < F32_LEN) {
            fa[i] = (float)(i % 97) * 0.25f;
            fb[i] = (float)(i % 31) * 0.5f;
        }
    }
    expected_u8_sum = sve_compute_u8(data_a, data_b, expected_dst_u8, DATA_LEN);
    expected_f32_sum = sve_compute_f32(fa, fb, F32_LEN);
    printf("expected u8 sum: %llu, f32 sum: %.4f\n",
           (unsigned long long)expected_u8_sum, (double)expected_f32_sum);

    printf("running %d threads x %lu iterations across %d online CPU(s)...\n",
           nthreads, iters, nr_cpus);

    tids = calloc(nthreads, sizeof(*tids));
    targs = calloc(nthreads, sizeof(*targs));
    if (!tids || !targs) {
        printf("out of memory\n");
        return 1;
    }
    for (i = 0; i < nthreads; i++) {
        targs[i].id = i;
        targs[i].iters = iters;
        targs[i].tid = 0;
        if (pthread_create(&tids[i], NULL, worker, &targs[i]) != 0) {
            printf("pthread_create failed\n");
            return 1;
        }
    }

    if (nthreads > 1 && nr_cpus > 1) {
        marg.targs = targs;
        marg.nthreads = nthreads;
        marg.stop = 0;
        if (pthread_create(&migrator_tid, NULL, migrator_fn, &marg) == 0)
            migrator_running = 1;
    }

    for (i = 0; i < nthreads; i++)
        pthread_join(tids[i], NULL);

    if (migrator_running) {
        __atomic_store_n(&marg.stop, 1, __ATOMIC_RELEASE);
        pthread_join(migrator_tid, NULL);
    }

    printf("iterations completed: %lu, corruptions: %lu\n", iters_done, fails);
    if (fails == 0) {
        printf("SVE SAVE/RESTORE TEST: PASS\n");
    } else {
        printf("SVE SAVE/RESTORE TEST: FAIL\n");
        ret = 1;
    }
    return ret;
}
