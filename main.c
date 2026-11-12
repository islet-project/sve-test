/*
 * sve-test - SVE context save/restore stress test for a Realm Linux VM
 *
 * The test verifies that the SVE register file (Z0-Z31, P0-P15, FFR)
 * survives context switches without corruption. It works in two layers:
 *
 *  1. Pattern layer: all Z registers are filled with a per-register byte
 *     pattern, all P registers with a per-register pattern and FFR with
 *     all-ones. After a preemption point (syscall / yield / sleep / thread
 *     switch) the patterns are read back and compared. Any mismatch means
 *     the SVE context was not saved/restored correctly somewhere in the
 *     stack (guest kernel, hypervisor or RMM).
 *
 *  2. Compute layer: an SVE intrinsic kernel (integer add + float FMA over
 *     arrays) is executed and its result compared against the expected
 *     constant. This catches wrong data being restored into registers even
 *     when the pattern layer would happen to match.
 *
 * Multiple threads are used to force guest-kernel SVE context switching,
 * CPU migrations and (on realm exits) RMM-level SIMD save/restore.
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
#define Z_PATTERN_BASE 0x40 /* z[n] filled with 0x40 + n */
#define P_PATTERN_BASE 0x80 /* p[n] bytes filled with 0x80 | n */

static size_t vl_bytes;     /* current SVE vector length in bytes */
static size_t ps_bytes;     /* predicate register size in bytes (VL/8) */
static uint8_t *z_patterns; /* NUM_Z_REGS * vl_bytes (read-only after init) */
static uint8_t *p_patterns; /* NUM_P_REGS * ps_bytes (read-only after init) */

/* compute-layer data */
#define DATA_LEN (1 << 20)
static uint8_t *data_a, *data_b, *data_dst;
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
    if (len > 32)
        printf("...");
    printf("\n");
}

/* Fill Z0-Z31 with per-register patterns and P0-P15 with per-register
 * patterns. One asm per register with an explicit address operand, so no
 * MUL VL offset arithmetic is needed and the compiler cannot reorder. */
__attribute__((noinline))
static void sve_fill_patterns(void)
{
#define FILL_Z(n)                                                              \
    asm volatile("ld1b z" STR(n) ".b, p0/z, [%0]\n"                            \
                 :                                                             \
                 : "r"(z_patterns + (n) * vl_bytes)                            \
                 : "memory")
    asm volatile("ptrue p0.b\n");
    FILL_Z(0);
    FILL_Z(1);
    FILL_Z(2);
    FILL_Z(3);
    FILL_Z(4);
    FILL_Z(5);
    FILL_Z(6);
    FILL_Z(7);
    FILL_Z(8);
    FILL_Z(9);
    FILL_Z(10);
    FILL_Z(11);
    FILL_Z(12);
    FILL_Z(13);
    FILL_Z(14);
    FILL_Z(15);
    FILL_Z(16);
    FILL_Z(17);
    FILL_Z(18);
    FILL_Z(19);
    FILL_Z(20);
    FILL_Z(21);
    FILL_Z(22);
    FILL_Z(23);
    FILL_Z(24);
    FILL_Z(25);
    FILL_Z(26);
    FILL_Z(27);
    FILL_Z(28);
    FILL_Z(29);
    FILL_Z(30);
    FILL_Z(31);
#undef FILL_Z
    /* FFR = all ones. Must happen BEFORE the P patterns are loaded, since
     * p0 is used as the all-true scratch predicate above and wrffr needs
     * one; ldr/str on predicate registers are unpredicated. */
    asm volatile("wrffr p0.b\n");
#define FILL_P(n)                                                              \
    asm volatile("ldr p" STR(n) ", [%0]\n"                                     \
                 :                                                             \
                 : "r"(p_patterns + (n) * ps_bytes)                            \
                 : "memory")
    FILL_P(0);
    FILL_P(1);
    FILL_P(2);
    FILL_P(3);
    FILL_P(4);
    FILL_P(5);
    FILL_P(6);
    FILL_P(7);
    FILL_P(8);
    FILL_P(9);
    FILL_P(10);
    FILL_P(11);
    FILL_P(12);
    FILL_P(13);
    FILL_P(14);
    FILL_P(15);
#undef FILL_P
}

/* Read back Z, P and FFR into the verify buffers. The buffers are
 * per-thread (no other thread may write them concurrently). This must be
 * called with NOTHING that uses SVE/NEON registers in between the pattern
 * fill and this read-back (not even glibc memcmp/printf/memcpy, whose
 * NEON registers alias the low 128 bits of Z0-Z31). */
__attribute__((noinline))
static void sve_readback_patterns(uint8_t *zv, uint8_t *pv, uint8_t *fv)
{
#define READ_Z(n)                                                              \
    asm volatile("str z" STR(n) ", [%0]\n"                                     \
                 :                                                             \
                 : "r"(zv + (n) * vl_bytes)                                    \
                 : "memory")
    READ_Z(0);
    READ_Z(1);
    READ_Z(2);
    READ_Z(3);
    READ_Z(4);
    READ_Z(5);
    READ_Z(6);
    READ_Z(7);
    READ_Z(8);
    READ_Z(9);
    READ_Z(10);
    READ_Z(11);
    READ_Z(12);
    READ_Z(13);
    READ_Z(14);
    READ_Z(15);
    READ_Z(16);
    READ_Z(17);
    READ_Z(18);
    READ_Z(19);
    READ_Z(20);
    READ_Z(21);
    READ_Z(22);
    READ_Z(23);
    READ_Z(24);
    READ_Z(25);
    READ_Z(26);
    READ_Z(27);
    READ_Z(28);
    READ_Z(29);
    READ_Z(30);
    READ_Z(31);
#undef READ_Z
#define READ_P(n)                                                              \
    asm volatile("str p" STR(n) ", [%0]\n"                                     \
                 :                                                             \
                 : "r"(pv + (n) * ps_bytes)                                    \
                 : "memory")
    READ_P(0);
    READ_P(1);
    READ_P(2);
    READ_P(3);
    READ_P(4);
    READ_P(5);
    READ_P(6);
    READ_P(7);
    READ_P(8);
    READ_P(9);
    READ_P(10);
    READ_P(11);
    READ_P(12);
    READ_P(13);
    READ_P(14);
    READ_P(15);
#undef READ_P
    asm volatile("rdffr p0.b\n"
                 "str p0, [%0]\n"
                 :
                 : "r"(fv)
                 : "memory");
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

/* Compare the read-back buffers against the expected patterns. This only
 * touches memory: it must be callable after the SVE registers have been
 * clobbered (e.g. by glibc memcmp itself, which uses NEON registers that
 * alias the low 128 bits of Z0-Z31). */
__attribute__((noinline))
static int sve_compare_patterns(const char *what, int thread_id,
                                unsigned long iter, const uint8_t *zv,
                                const uint8_t *pv, const uint8_t *fv)
{
    int bad = 0, n;

    for (n = 0; n < NUM_Z_REGS; n++) {
        const uint8_t *e = z_patterns + n * vl_bytes;
        const uint8_t *g = zv + n * vl_bytes;
        if (memcmp(e, g, vl_bytes) != 0) {
            report_fail(what, thread_id, iter, z_reg_names[n], e, g, vl_bytes);
            bad++;
        }
    }
    for (n = 0; n < NUM_P_REGS; n++) {
        const uint8_t *e = p_patterns + n * ps_bytes;
        const uint8_t *g = pv + n * ps_bytes;
        if (memcmp(e, g, ps_bytes) != 0) {
            report_fail(what, thread_id, iter, p_reg_names[n], e, g, ps_bytes);
            bad++;
        }
    }
    {
        uint8_t allones[64];
        memset(allones, 0xff, sizeof(allones));
        if (memcmp(allones, fv,
                   ps_bytes < sizeof(allones) ? ps_bytes : sizeof(allones)) !=
            0) {
            report_fail(what, thread_id, iter, "FFR", allones, fv, ps_bytes);
            bad++;
        }
    }
    return bad;
}

/* Functional SVE kernels ------------------------------------------------- */

/* noinline: keep GCC's SVE codegen (ptrue etc.) out of worker() so it can
 * not be scheduled between the pattern fill and the pattern verify. */
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

static void report_compute_fail(int thread_id, unsigned long iter,
                                const char *what, unsigned long long expect,
                                unsigned long long got)
{
    pthread_mutex_lock(&fail_lock);
    fails++;
    printf("[FAIL] t%d iter %lu: %s result corrupted: expected %llu, got %llu\n",
           thread_id, iter, what, expect, got);
    pthread_mutex_unlock(&fail_lock);
}

/* Preemption / world-switch pressure -------------------------------------
 *
 * IMPORTANT: no syscalls are allowed inside the pattern window.
 * Current arm64 kernels run fp_user_discard() from el0_svc() on EVERY
 * syscall: sve_flush_live() executes 'mov vN.16b, vN.16b' for every Z
 * register - which keeps the low 128 bits but zeroes everything above it
 * (FP-write semantics) - and pfalse's all predicate registers.  So
 * "fill SVE registers, make a syscall, read them back" always shows
 * "first 16 bytes valid, rest zero".  That is the documented SVE syscall
 * ABI, not a save/restore bug.
 *
 * The paths we actually want to stress - the guest kernel's full-SVE
 * fpsimd_save()/task_fpsimd_load() and the RMM's REC world switches - are
 * exercised by timer-IRQ-driven preemption and CPU migration, so force
 * those with a long busy-spin (no syscalls, no FP instructions) plus
 * affinity changes made BEFORE the pattern fill (the migration then
 * happens on a timer tick, inside the pattern window).
 */
static int nr_cpus = 1;

static void preempt_spin(void)
{
	/* Volatile busy loop roughly a few scheduler timeslices long.
	 * No syscalls, no FP/SVE instructions - they would clobber state. */
	volatile unsigned long i;

	for (i = 0; i < 5000000UL; i++)
		__asm__ __volatile__("" ::: "memory");
}

static void migrate_hint(int thread_id, unsigned long iter)
{
	cpu_set_t set;
	int cpu = (thread_id + (int)iter) % (nr_cpus > 0 ? nr_cpus : 1);

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	sched_setaffinity(0, sizeof(set), &set);
}

struct thread_arg {
    int id;
    unsigned long iters;
};

static void *worker(void *argp)
{
    struct thread_arg *a = argp;
    unsigned long iter;
    uint8_t *zv, *pv, *fv;

    /* per-thread read-back buffers: shared buffers would race between
     * threads (one thread's compare vs another thread's read-back) and
     * produce bogus corruption reports */
    zv = malloc(NUM_Z_REGS * vl_bytes);
    pv = malloc(NUM_P_REGS * ps_bytes);
    fv = malloc(ps_bytes);
    if (!zv || !pv || !fv)
        return NULL;

    for (iter = 0; iter < a->iters; iter++) {
        uint64_t sum_u8;
        float sum_f32;

        /* 0. request a CPU affinity change: the actual migration happens
         * on the next scheduler tick, i.e. inside the pattern window
         * below (sched_setaffinity is a syscall, so it must run before
         * the fill, never inside the pattern window). */
        migrate_hint(a->id, iter);

        /* 1. fill all SVE registers with patterns */
        sve_fill_patterns();

        /* 2. on the first iteration, sanity-check the fill machinery
         * itself: read back and compare immediately, then re-fill */
        if (iter == 0) {
            sve_readback_patterns(zv, pv, fv);
            if (sve_compare_patterns("selfcheck", a->id, iter, zv, pv, fv) !=
                0)
                continue;
            sve_fill_patterns();
        }

        /* 3. force timer-driven preemption / migration. NO SYSCALLS here:
         * the arm64 syscall entry path zeroes SVE state above the low
         * 128 bits of every Z register by design (fp_user_discard). */
        preempt_spin();

        /* 3b. The effective VL must still be what main() measured. If the
         * hypervisor/RMM restores a stale ZCR_EL1, every SVE instruction in
         * this entry runs at the wrong (typically minimum, 16-byte) VL:
         * ld1b then loads only 16 bytes and zeroes the rest of each Z
         * register (SVE inactive-element semantics), and the pattern
         * comparisons below would only produce meaningless hex-dump noise.
         * Detect and report the VL change explicitly instead. rdvl writes no
         * Z/P data registers, so it is safe between fill and read-back. */
        {
            uint64_t vl_now = rdvl_bytes();

            if (vl_now != vl_bytes) {
                pthread_mutex_lock(&fail_lock);
                fails++;
                printf("[FAIL] t%d iter %lu: effective VL changed: expected "
                       "%zu, got %llu (hypervisor likely clobbered ZCR_EL1)\n",
                       a->id, iter, vl_bytes,
                       (unsigned long long)vl_now);
                pthread_mutex_unlock(&fail_lock);
                continue;
            }
        }

        /* 4. read the patterns back out. Nothing may touch the SVE
         * registers between the fill and this read-back. */
        sve_readback_patterns(zv, pv, fv);

        /* 5. compare on memory only (memcmp may clobber SVE regs) */
        sve_compare_patterns("after-switch", a->id, iter, zv, pv, fv);

        /* 5. functional SVE compute, compared against the constant result */
        sum_u8 = sve_compute_u8(data_a, data_b, data_dst, DATA_LEN);
        if (sum_u8 != expected_u8_sum)
            report_compute_fail(a->id, iter, "sve_compute_u8", expected_u8_sum,
                                sum_u8);
        sum_f32 = sve_compute_f32(fa, fb, F32_LEN);
        if (sum_f32 != expected_f32_sum)
            report_compute_fail(a->id, iter, "sve_compute_f32",
                                (unsigned long long)expected_f32_sum,
                                (unsigned long long)sum_f32);

        /* 6. verify the vector length did not change behind our back */
        if (rdvl_bytes() != vl_bytes) {
            pthread_mutex_lock(&fail_lock);
            fails++;
            printf("[FAIL] t%d iter %lu: VL changed: was %zu, now %llu\n",
                   a->id, iter, vl_bytes, (unsigned long long)rdvl_bytes());
            pthread_mutex_unlock(&fail_lock);
        }

        pthread_mutex_lock(&fail_lock);
        iters_done++;
        pthread_mutex_unlock(&fail_lock);
    }
    free(zv);
    free(pv);
    free(fv);
    return NULL;
}

int main(int argc, char **argv)
{
    unsigned long iters = 1000;
    int nthreads = 2;
    pthread_t *tids;
    struct thread_arg *targs;
    int i, ret = 0;

    if (argc > 1)
        iters = strtoul(argv[1], NULL, 0);
    if (argc > 2)
        nthreads = atoi(argv[2]);
    if (nthreads < 1)
        nthreads = 1;

    setvbuf(stdout, NULL, _IONBF, 0);

    nr_cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nr_cpus < 1)
        nr_cpus = 1;

    if (!(getauxval(AT_HWCAP) & (1UL << 22)) /* HWCAP_SVE */) {
        printf("SVE not available (HWCAP_SVE missing)\n");
        return 1;
    }

    vl_bytes = rdvl_bytes();
    ps_bytes = vl_bytes / 8;

    /* try to run at the largest VL the kernel/EL2 allows */
    {
        int cur_vl = prctl(PR_SVE_GET_VL, 0, 0, 0, 0);
        if (cur_vl >= 0) {
            int want = (cur_vl & PR_SVE_VL_LEN_MASK) | PR_SVE_VL_INHERIT;
            if (prctl(PR_SVE_SET_VL, want, 0, 0, 0) == 0)
                vl_bytes = rdvl_bytes();
        }
        printf("SVE VL: %zu bytes (%d-bit), predicate: %zu bytes\n", vl_bytes,
               (int)(vl_bytes * 8), ps_bytes);
    }

    /* allocate pattern buffers (read-only after init; verify buffers are
     * per-thread and allocated in the worker) */
    z_patterns = malloc(NUM_Z_REGS * vl_bytes);
    p_patterns = malloc(NUM_P_REGS * ps_bytes);
    if (!z_patterns || !p_patterns) {
        printf("out of memory\n");
        return 1;
    }
    for (i = 0; i < NUM_Z_REGS; i++)
        memset(z_patterns + i * vl_bytes, Z_PATTERN_BASE + i, vl_bytes);
    for (i = 0; i < NUM_P_REGS; i++)
        memset(p_patterns + i * ps_bytes, P_PATTERN_BASE | (unsigned)i,
               ps_bytes);

    /* compute-layer data */
    data_a = malloc(DATA_LEN);
    data_b = malloc(DATA_LEN);
    data_dst = malloc(DATA_LEN);
    fa = malloc(F32_LEN * sizeof(float));
    fb = malloc(F32_LEN * sizeof(float));
    if (!data_a || !data_b || !data_dst || !fa || !fb) {
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
    expected_u8_sum = sve_compute_u8(data_a, data_b, data_dst, DATA_LEN);
    expected_f32_sum = sve_compute_f32(fa, fb, F32_LEN);
    printf("expected u8 sum: %llu, f32 sum: %.4f\n",
           (unsigned long long)expected_u8_sum, expected_f32_sum);

    printf("running %d threads x %lu iterations...\n", nthreads, iters);

    tids = calloc(nthreads, sizeof(*tids));
    targs = calloc(nthreads, sizeof(*targs));
    if (!tids || !targs) {
        printf("out of memory\n");
        return 1;
    }
    for (i = 0; i < nthreads; i++) {
        targs[i].id = i;
        targs[i].iters = iters;
        if (pthread_create(&tids[i], NULL, worker, &targs[i]) != 0) {
            printf("pthread_create failed\n");
            return 1;
        }
    }
    for (i = 0; i < nthreads; i++)
        pthread_join(tids[i], NULL);

    printf("iterations completed: %lu, corruptions: %lu\n", iters_done, fails);
    if (fails == 0) {
        printf("SVE SAVE/RESTORE TEST: PASS\n");
    } else {
        printf("SVE SAVE/RESTORE TEST: FAIL\n");
        ret = 1;
    }
    return ret;
}
