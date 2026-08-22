# Why our kernels use 130 registers and NVSHMEM's use 87

Measured 2026-08-22, sm_89, real headers and real build flags. Ladder source:
`context-transfer-engine/adapter/gpu_vector/benchmark/coro_reg_ladder.cc`.
Each rung compiled in **its own translation unit** (see the trap in section 4).

---

## 1. The measurement

Each rung adds exactly one ingredient to the one below it.

| rung | what it is | REG | delta |
|---|---|---|---|
| R0 | raw pointer loop, no clio at all | **14** | -- |
| R1 | + `CLIO_GPU_INIT` + `YieldTlsPublish` | **22** | +8 |
| R2 | + an **empty coroutine** (`co_return` only) | **28** | **+6** |
| R3 | + the loop moved *inside* the coroutine | **28** | +0 |
| R6 | + `TryHoldFast` -- the **resident-hit probe only**, no fetch | **72** | **+44** |
| R4 | + full `HoldPage` (probe **and** the miss path) | **106** | **+34** |
| R5 | + `HoldPage` in a loop (what the MD bench ships) | **106** | +0 |

## 2. The answer

**Coroutines are not the problem. They cost 6 registers.**

The paging machinery costs 78, and it splits almost evenly:

- **+44 registers: the resident-hit probe** (`TryHoldFast` -> `ProbeHold`).
  This is paid on *every* hold, including the fast path that never suspends.
- **+34 registers: the miss path** (`HoldPageCoro` -- fetch, evict, flush).
  This is inlined into every kernel even though it executes rarely.

**Why NVSHMEM/MPI/NCCL are cheaper (50-87):** their kernels have no page
table. They index a resident array. There is no probe to run and no fetch path
to inline. We embed a software page cache into every kernel, and the register
file pays for it whether or not a page ever misses. That is the entire
difference. It is not a compiler defect and it is not the coroutine lowering.

**Second, separate effect (already fixed).** Resuming a coroutine is an
indirect call, and an indirect call makes ptxas allocate the kernel for the
worst case over every address-taken function in the module -- which flattened
all eleven kernels to one number regardless of what they do. `CoroDevirt`
(committed, `tools/coro_regcap/CoroDevirt.cpp`) takes the real build from 63
indirect calls to 0. It is a prerequisite for any per-kernel number, but on its
own it changed no register counts, because the dominant cost above is genuinely
present in every kernel.

## 3. Why "exact register usage" has no compile-time answer

Unconstrained, ptxas picks 130. Capped at 64 the same code compiles
**spill-free** (`LOCAL:0`, +1.4% instructions) and runs at the same speed. Both
are honest. ptxas allocates greedily when no occupancy target is stated, and it
cannot infer one, because occupancy depends on block size and block size is a
runtime value here (`--threads`).

So there is no compiler-side "exact" number to compute. **The only way to get
the register count down is to make the code genuinely need fewer registers.**
That is what section 5 is about.

## 4. Measurement trap, recorded because it produced a wrong answer

Compiling all rungs in one TU reported **106 for every rung including the empty
coroutine** -- which reads as "coroutines cost 106" and is wrong. That is the
indirect-call effect of section 2: within one module every kernel inherits the
worst coroutine present. One rung per TU is mandatory here.

---

## 5. Candidate solutions

### A. Move the miss path out of the compute kernel

**Saves: ~34 registers (106 -> 72).**

The miss path already exits the kernel -- a park returns from the kernel and
the servicer runs before relaunch. Yet `HoldPageCoro`'s fetch/evict/flush state
machine is compiled *into* every compute kernel. If a miss instead recorded
"block B needs page P", parked, and let a separate servicer kernel do the
fetch, the compute kernel would contain only the probe.

- *Type:* architectural, our code, no compiler work.
- *Risk:* medium. Changes the fault protocol; the servicer must be able to do
  everything the device-side fetch does today (batched fetch, eviction,
  writeback of dirty victims).
- *Gate:* MD statics/resort/NVE digit-identical, resident and out of core.

### B. Shrink the resident-hit probe

**Saves: up to ~44 registers (72 -> ~28).**

This is the bigger half and it is paid on every hold. `ProbeHold` does a
lock-free open-addressed scan with a Dekker handshake and a block-wide vote,
per thread. Concrete directions:

- **Have one thread probe and broadcast the result.** The probe is already
  block-uniform by construction (`__syncthreads_and` votes the outcome), so
  every thread is computing the same answer with its own live values. Thread 0
  probing into shared memory and broadcasting would cut the per-thread live set
  hard. *Caveat: shared memory does not survive a park, so the broadcast must
  be re-done after resume, not cached across one.*
- **Shorten the scan.** Fewer probes per lookup, or a smaller resident
  descriptor.
- **Cache the last resolved page per block in registers** rather than
  re-deriving it -- the `HoldPage`-in-a-loop pattern (R5) re-probes per page.

- *Type:* our code, local to `device_vector.h`.
- *Risk:* high on correctness, low on architecture. This code has a history of
  block-uniformity bugs (the mismatched-BAR crash, the 8-vs-123 writeback bug
  quoted in its own comments). Every change needs the out-of-core gates.

### C. Keep a register cap

**Saves: everything, at an occupancy target you must choose.**

`--maxrregcount=64` works today, spill-free, same speed. It is not a hack --
it is the mechanism for stating an occupancy target that the compiler cannot
infer. Its defect is that the number is written by hand and is wrong for block
sizes other than the one it was tuned for.

- *Type:* build config.
- *Risk:* none technically; the objection is that it is a magic constant.

### D. Split the kernel: resolve, then compute

**Saves: potentially all 78, in the compute kernel.**

A resolve kernel walks the pages a block will touch and writes an array of raw
pointers; the compute kernel reads pointers and does arithmetic with no paging
code in it at all. The compute kernel then looks like the MPI one (~50).

- *Type:* architectural, largest change.
- *Risk:* high. Requires knowing the access set ahead of time, which the
  Verlet-list force loop does (it is a gather over a known list) but a general
  workload may not. Loses the ability to fault mid-kernel, which is the point
  of the abstraction.

---

## 6. Comparison

| | A. miss path out | B. shrink probe | C. cap | D. resolve/compute split |
|---|---|---|---|---|
| registers saved | ~34 | up to ~44 | all (by fiat) | up to 78 |
| where the work is | fault protocol | `device_vector.h` | build file | kernel structure |
| compiler work | none | none | none | none |
| correctness risk | medium | **high** | none | high |
| keeps mid-kernel faulting | yes | yes | yes | **no** |
| magic constants | none | none | **one** | none |
| effort | days | days | done | weeks |

**Recommended order: B, then A.** They are independent and they attack the two
halves of the 78. B is the bigger prize and is paid on the hot path (every
resident hit), so it also buys latency, not just occupancy. A is more
mechanical and lower risk. Together they target 106 -> ~30, which is the range
where the transports live and where no cap is needed at all.

C stays in place meanwhile -- it is what makes the system usable today, and it
should be removed only once B and A make it unnecessary, not before.

D is a last resort: it buys the most but gives up mid-kernel faulting, which is
the feature the abstraction exists to provide.

**What is explicitly NOT on this list:** any further compiler plugin work for
registers. The ladder shows the compiler is not the problem -- 6 registers for
the coroutine lowering. `CoroDevirt` is already committed and is worth keeping
for the indirect-call elimination, but no additional pass will move these
numbers, because the registers are being used by code we wrote.
