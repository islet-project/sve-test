# sve-test — SVE context save/restore stress test for a Realm Linux VM

## What it does

Statically-linked AArch64 program that hammers the SVE register file
(Z0–Z31, P0–P15, FFR) to detect broken SIMD context save/restore anywhere
in the stack: guest kernel, hypervisor/KVM, or the RMM.

Per iteration and thread:

1. **Fill** — all Z registers are loaded with a per-register byte pattern
   (Z_n = 0x40+n), all P registers with 0x80|n, FFR with all-ones
   (`ptrue`/`wrffr`).
2. **Preempt** — `sched_yield()`, short `nanosleep()`, `getpid()` or an
   occasional 200 µs sleep, plus thread switching between 1–4 worker
   threads, to force context switches and realm/world exits.
3. **Read back** — all Z/P registers are stored out with `str` and FFR is
   read with `rdffr`; results are compared against the patterns.
4. **Functional compute** — an SVE intrinsic kernel (u8 add over 1 MB +
   f32 FMA) is checked against a constant expected result.
5. **VL check** — `rdvl` must not change behind the program's back.

Any mismatch is reported as
`[FAIL] t<thread> iter <n>: <where>: <reg> corrupted (len <L>, first diff
at byte <off>)` followed by hex dumps of the full expected/got register
contents; the program exits non-zero on any corruption.

## Diagnosing failures

The `first diff at byte <off>` and full dumps localize the corruption:

* **Diff only in bytes 0..15** — the low 128-bit (NEON-visible) portion was
  lost: something restored only the FPSIMD part of the SVE registers.
* **Diff only at bytes ≥ 16 or ≥ 32** — only a short prefix of each Z
  register survived: a save/restore path that saves less than the full
  vector length.
* **Only P registers or FFR** — the predicate/FFR part of the context was
  not saved or restored, or was saved with the wrong size (VL/8 bytes per
  predicate).
* **All bytes shifted/swapped between registers** — an off-by-one or
  stride error in the save/restore buffer layout.

Run single-threaded first (`./sve-test 10000 1`) to separate per-thread
context restore bugs from cross-thread/preemption issues; then increase
the thread count.

## Build

    make            # uses aarch64-linux-gnu-gcc, -march=armv8.2-a+sve, -static

## Run (in the Realm VM)

    ./sve-test [iterations] [threads]
    ./sve-test 10000 2      # recommended soak: 2 threads x 10k iterations

Exit code 0 = PASS, 1 = FAIL. Let it run for a while (minutes+) to hit
real preemption/world-switch windows; the RMM-level save/restore is
exercised whenever the host preempts the vCPU or the host side uses SVE.

## Implementation notes

* Fill/read-back use one `asm volatile` per register with an explicit
  address operand (no `MUL VL` offset arithmetic) and live in `noinline`
  helpers, so no compiler-generated SVE code can be scheduled between
  filling and reading back the patterns.
* The read-back buffers are **per-thread**; the comparison is done on
  memory only, in a separate function: glibc's `memcmp`/`memcpy`/`printf`
  use NEON registers, and NEON v0–v31 alias the low 128 bits of Z0–Z31 —
  a `memcmp` between fill and read-back (or shared read-back buffers
  raced between threads) would otherwise produce bogus corruption
  reports whose expected/got dumps look identical.
* FFR is written (`wrffr p0.b`) *before* the P-register patterns are
  loaded, because P0 doubles as the all-true scratch predicate.

## Pre-validating under qemu-user (optional)

    qemu-aarch64 ./sve-test 1000 2

Known artifact: `qemu-aarch64` 8.2.x has a signal-frame SVE save/restore
bug; with ≥4 threads you may see a handful of failures in the thread doing
the long `nanosleep` (iter % 64 == 0), always Z0, with the "got" value
being the timespec `{tv_sec=0, tv_nsec=200000}`. That is a qemu-user bug,
not a real corruption — the authoritative result is the one from the real
Realm VM, where any mismatch is a genuine save/restore bug.

## Why the preemption points avoid syscalls

Modern arm64 kernels run `fp_user_discard()` from `el0_svc()` on **every
syscall**: `sve_flush_live()` executes `mov vN.16b, vN.16b` for every Z
register — which keeps the low 128 bits but zeroes everything above it —
and `pfalse`s all predicate registers. Filling the SVE registers and then
making a syscall therefore *always* "corrupts" the patterns (first 16
bytes valid, rest zero). That is the documented SVE syscall ABI, not a
save/restore bug, and it happens identically under any RMM (observed
the same under both Islet RMM and TF-RMM).

The test instead forces context switches through timer-IRQ-driven
preemption (long busy-spins, no syscalls) and CPU migration
(`sched_setaffinity` done *before* the pattern fill, taking effect on a
later scheduler tick). Those paths exercise the full-width SVE
save/restore (guest kernel `fpsimd_save()`/`task_fpsimd_load()` plus the
RMM REC world switches), which is what this test validates. **Never add
syscalls inside the fill → readback window.**

