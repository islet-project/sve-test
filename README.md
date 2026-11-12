# sve-test — SVE context save/restore stress test for a Realm Linux VM

## What it does

Statically-linked AArch64 program that hammers the SVE register file
(Z0–Z31, P0–P15, FFR) to detect broken SIMD context save/restore anywhere
in the stack: guest kernel, hypervisor/KVM, or the RMM.

Per iteration and thread:

1. **Fill** — all Z registers (`Z0–Z31`), all P registers (`P0–P15`), and
   `FFR` are loaded with deterministic non-zero patterns that vary across
   register index, byte lane, thread ID, and iteration (so cross-thread
   context swaps, stale restores, and intra-register lane permutations are
   all detected).
2. **Preempt & Migrate** — a pure integer busy-spin loop (no syscalls, no
   FP/SIMD instructions) runs inside the same `asm volatile` block while an
   asynchronous migrator thread calls `sched_setaffinity(worker_tid, ...)`
   to force both same-vCPU timer-IRQ preemption and IPI-driven cross-vCPU
   migrations while SVE state is live in EL0.
3. **VL check & Read back** — inside the same `asm volatile` block, `rdvl`
   verifies the effective vector length (`ZCR_EL1`) before storing all Z/P
   registers (`str`) and `FFR` (`rdffr`); results are then compared against
   the thread's expected patterns.
4. **Functional compute** — SVE intrinsic kernels (predicated `u8` add +
   store + reduction over 1 MB, and `f32` FMA reduction) are checked against
   reference results and output buffers.
5. **Post-compute VL check** — `rdvl` is checked again after the compute
   kernels.

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

* Fill, busy-spin, `rdvl` verification, and read-back execute inside a
  **single `asm volatile` block** (`sve_fill_spin_readback`) with `z0–z31`,
  `p0–p15`, and `ffr` declared in the clobber list. This prevents the
  compiler from scheduling any FP/NEON/SVE code or register spills inside
  the pattern window while properly informing the compiler that callee-saved
  `d8–d15` (low 64 bits of `z8–z15`) are modified.
* Expected pattern buffers, read-back buffers, and `u8` compute output
  buffers are **per-thread**; comparisons are performed on memory only after
  read-back completes, avoiding NEON `v0–v31` aliasing with `Z0–Z31`.
* `FFR` is loaded with a canonical monotonic predicate bitmask (`wrffr p0.b`)
  *before* `P0–P15` are loaded, using `P0` as scratch.

## Pre-validating under qemu-user (optional)

    qemu-aarch64 ./sve-test 100 4

## Why the preemption points avoid syscalls

Modern arm64 kernels run `fp_user_discard()` from `el0_svc()` on **every
syscall**: `sve_flush_live()` executes `mov vN.16b, vN.16b` for every Z
register — which keeps the low 128 bits but zeroes everything above it —
and `pfalse`s all predicate registers. Filling the SVE registers and then
making a syscall therefore *always* "corrupts" the patterns (first 16
bytes valid, rest zero). That is the documented SVE syscall ABI, not a
save/restore bug, and it happens identically under any RMM (observed
the same under both Islet RMM and TF-RMM).

Moreover, calling `sched_setaffinity(0, ...)` from a worker thread before
filling patterns migrates the calling thread synchronously inside the
syscall before returning to userspace, and pinning to a single CPU would
prevent migration during the spin window. Instead, the test uses a pure
integer busy-spin inside the pattern window while a separate migrator
thread calls `sched_setaffinity(worker_tid, ...)` from the outside. That
triggers kernel migration IPIs and timer-IRQ preemption via `el0_irq`,
exercising full-width SVE save/restore (`fpsimd_save()`/`task_fpsimd_load()`
and RMM REC world switches). **Never add syscalls inside the fill → readback
window.**

