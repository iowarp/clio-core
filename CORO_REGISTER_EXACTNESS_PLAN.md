# Plan: Exact, Automatic Register Allocation for Device-Coroutine Kernels

**Status:** proposed, not started
**Owner:** core / gpu_vector
**Files in scope:** `tools/coro_regcap/`, `cmake/ClioCoroRegCap.cmake`,
`context-transfer-engine/adapter/gpu_vector/benchmark/CMakeLists.txt`,
`external/lammps/lib/eternia/CMakeLists.txt` and its copy of the pass

---

## 1. Requirement

Register usage must be **calculated exactly and automatically at compile
time**, per kernel, from the program itself.

Explicitly forbidden:

- assumed register counts (`64`, `128`, `193`, `65536`)
- assumed thread counts (`256`) or blocks-per-SM targets (`2`, `4`)
- hand-written ceilings of any kind: `--maxrregcount`, `__launch_bounds__`,
  `nvvm.maxnreg`, `.maxntid`, `.minnctapersm`
- anything a user must set, sweep, or tune

Toolchain choice is unconstrained: clang, nvcc, a different lowering, or a
change to how the yield mechanism is compiled are all permitted.

---

## 2. Why every ceiling-based approach is disqualified

A register ceiling has no meaning on its own. It is only interpretable against
an occupancy goal:

```
blocks_per_SM = regs_per_SM / (registers_per_thread * threads_per_block)
```

Any chosen cap therefore encodes an assumed `threads_per_block` and an assumed
target `blocks_per_SM`. In this tree `threads_per_block` is a **runtime flag**
(`--threads`), so no compile-time constant can be correct across launches: a
cap derived for 256 threads is simply wrong at 64 or 512.

Every iteration attempted so far smuggles in exactly those two assumptions:

| attempt | where the assumption lives |
|---|---|
| `-Xcuda-ptxas --maxrregcount=64` (`MD_MAXRREG`) | the literal `64`, TU-wide, clamps plain kernels too |
| `NVPTXCoroCap` with `Cap = 128` | the literal `128` |
| `ClioCoroRegCap.cmake` measured budget | `65536 / (256 * 4) = 64` -- three constants that multiply out to the same number |

The third is the most misleading, because it *looks* derived. It measures which
kernels exceed a budget, but the budget itself is still assumed. It must go
with the others.

**What this leaves is the correct target.** ptxas already computes each
kernel's register requirement exactly and automatically. It is not failing to
do so; it is being handed a function whose register pressure has been
artificially inflated before it ever sees it. Remove the inflation and the
exact number is produced for free, per kernel, with no constant anywhere in the
tree.

---

## 3. Root cause  **[SUPERSEDED BY PHASE 0 -- SEE SECTION 11]**

> The mechanism proposed in 3.2 below (liveness union across suspend points)
> was **refuted** by Phase 0. It is kept only so the reasoning that led to the
> wrong model stays visible. The verified mechanism is in section 11.

### 3.1 The observation

Measured on the uncapped probe build of `clio_gpu_vector_md_bench`
(`cuobjdump -res-usage`, 12 kernels):

| kernel | REG |
|---|---|
| `YieldStackInitKernel` (plain, no coroutine) | **8** |
| `ForceKernel` | 138 |
| `ListForceKernel` | 138 |
| `BuildListKernel` | 138 |
| `RebinKernel` | 138 |
| `ThermoKernel` | 138 |
| `GatherKernel` | 138 |
| `IntegrateKernel` | 138 |
| `MDIntegrateKernel` | 138 |
| `SentinelKernel` | 138 |
| `ReadProbeKernel` | 138 |
| `FlushAllKernel` | 138 |

Eleven kernels with unrelated bodies -- a full LJ force loop, a thermodynamic
reduction, a one-line sentinel -- allocate **identically**. Register count that
does not depend on the body is only possible if the allocator is not seeing the
body's real pressure.

### 3.2 The mechanism

1. clang switch-lowers each device coroutine. CoroSplit emits a resume function
   whose body is a `switch` over the suspend index, one case per segment.
2. On NVPTX that resume function is **inlined into the kernel** -- no distinct
   resume symbol appears in `cuobjdump`, and the kernel is ~27k SASS
   instructions.
3. Coro-frame reloads are materialised at the **dispatch**, so the live set at
   that point is the **union of every segment's live-in set**, not the maximum.
4. ptxas allocates for that union. The union is dominated by the coroutine
   machinery, which is identical in all eleven kernels -- producing the
   identical 138.

### 3.3 Corroborating evidence already in hand

- An empty `co_return`-only coroutine measures the same as the full paging
  machinery (193 in the eternia ladder; 138 here) -- body-independence.
- Under any explicit ceiling ptxas recompresses the same code **spill-free**
  (`LOCAL:0`) at +1.4% instructions, proving 138 is allocator laziness and not
  demand.
- Every optimisation knob is a null result: ptxas `-O0/-O1/-O2/-O3`, clang
  `-O1/-O2/-Os` (`-Os` shrinks SASS 27.4k -> 15.6k instructions with registers
  unchanged), machine scheduler off, `__noinline__` fences.

The task is therefore to **collapse the union back to a maximum**.

---

## 4. Phase 0 -- Prove the mechanism before changing anything

The diagnosis in section 3.2 is inference from register counts. It must be
confirmed against artefacts before any code is written; the last several
conclusions in this area were wrong and were caught only by measurement.

**0.1 Emit and retain intermediates** for one representative TU:

```
-save-temps
-mllvm -print-after=coro-split -mllvm -print-module-scope
cuobjdump -ptx -sass
```

**0.2 Answer three questions with artefacts, not reasoning:**

- **Is the resume function inlined into the kernel?** Look for a distinct
  resume symbol in PTX and SASS. If it is a separate function, step 2 of the
  mechanism is wrong.
- **Where are the coro-frame reloads?** If they sit in the dispatch block
  rather than in the segments that consume them, the union is confirmed and
  Phase 1a is the correct fix. If they are already sunk, the pressure comes
  from somewhere else and this plan is rewritten.
- **What is the true max-over-segments pressure?** Compute MAXLIVE per segment
  on the post-CoroSplit IR and compare to 138. That number is what the kernel
  *should* allocate, and it becomes the acceptance target for Phase 3.

**0.3 Falsification check.** Build a two-kernel TU: one coroutine with a
deliberately tiny frame, one with a deliberately huge frame. If both still
report the same REG, the union hypothesis holds. If they differ, the diagnosis
is wrong and the plan restarts from measurement.

**Exit criterion:** a written statement of the mechanism, with the IR/SASS
excerpt that proves it and the per-segment MAXLIVE figure.

---

## 5. Phase 1 -- Collapse the union  **[SUPERSEDED -- SEE SECTION 12]**

Choose the variant Phase 0 indicates. They compose, and are ordered by
risk/benefit.

### 1a. Sink coro-frame reloads into their segments (preferred)

If reloads are hoisted to the dispatch, an LLVM pass that sinks each frame load
to its use collapses the live-in union to a per-segment maximum with **no ABI
calls, no outlining, and no code duplication**. It fixes the pressure at its
source and is the lowest-risk option.

- Runs after CoroSplit; NVPTX only, guarded by triple.
- Sinks `load`s from the coro frame to the nearest point dominating their
  uses; never across side effects, never through the suspend dispatch.
- Values used by several segments stay at a point dominating all uses.

### 1b. Outline each switch case into its own function

If segments share a register file purely because they are one function,
outline each case so each is allocated independently. Costs NVPTX ABI calls
per resume. **Verify ptxas reports a max over the call graph and not a sum**
before committing to this.

### 1c. Keep the resume function out of the kernel

Mark the CoroSplit-generated resume/destroy functions `noinline` on NVPTX so
they are allocated separately from the ramp. Cheapest to try; confirm it
changes what ptxas reports rather than merely relocating code.

**Deliverable:** one pass under `tools/coro_regcap/` (renamed -- it no longer
caps anything), loaded via `-fpass-plugin`, NVPTX-guarded, **no options, no
tunables, no numbers**.

**Exit criterion:** the eleven kernels no longer report an identical register
count, and each sits at or near its per-segment MAXLIVE from 0.2.

---

## 6. Phase 2 -- Delete every assumed constant

After this phase nothing in the tree names a register count, a thread count, or
an occupancy target.

- `context-transfer-engine/adapter/gpu_vector/benchmark/CMakeLists.txt`:
  remove `MD_MAXRREG` and the `clio_coro_regcap()` call.
- `cmake/ClioCoroRegCap.cmake`: delete `CLIO_CORO_REGS_PER_SM`,
  `CLIO_CORO_REF_THREADS`, `CLIO_CORO_TARGET_BLOCKS`, the probe target, the cap
  file, and `tools/coro_regcap/derive_caps.py`. The entire two-pass build goes
  away -- it exists only to serve a cap.
- `tools/coro_regcap/NVPTXCoroCap.cpp`: remove `nvvm.maxnreg` stamping and both
  `-clio-coro-*` options.
- `external/lammps/lib/eternia/CMakeLists.txt` and its copy of the pass: remove
  the `Cap = 128` default and `CLIO_CORO_MAXNREG`.
- **Grep gate:** `maxrregcount|maxnreg|launch_bounds|MAXRREG|minnctapersm` must
  return nothing outside third-party trees.

---

## 7. Phase 3 -- Verification

Correctness first. A register win that changes the physics is not a win.

**3.1 Exactness.** Per-kernel REG now varies with body, and each kernel sits at
its Phase 0.2 MAXLIVE. `LOCAL:0` everywhere -- a spill means the collapse was
done wrong, not that a budget needs raising.

**3.2 Correctness gates.** `clio_gpu_vector_md_bench` statics/resort/NVE gates
at lattice 100 and 112, resident and out-of-core; PE/atom digit-identical to
the pre-change build; the gpu_vector ctest suite (14) and the cte suite (45)
green. GPU tests run **sequentially** -- `-j` produces false failures.

**3.3 Performance, at matched geometry.** Compare against the current
`--maxrregcount=64` build **at several `--threads` values**, not one: the whole
point is that the result must not be tuned to a single block size. Median of 5,
warmed GPU, per-kernel REG reported alongside ms/step.

**3.4 Cross-consumer.** Rebuild the LAMMPS eternia stack and re-run its parity
gate; that tree carries the other copy of the pass. `lmp` does **not** relink
after `lib/eternia` changes -- delete `lmp-build/lmp` first.

**3.5 Regression guard.** A CI check asserting the coroutine kernels in the MD
bench do **not** all report the same register count. That identity is the
signature of this bug and must not silently return.

---

## 8. If Phase 1 fails

If none of 1a/1c/1b collapses the union -- for instance if the pressure proves
inherent to switch-lowered coroutines on a target without tail calls -- then
the exact number cannot be produced by this toolchain, and that will be
reported as such rather than papered over with a reintroduced constant.

The fallback is a **toolchain change**, which the requirement permits: lower
the yield mechanism to a structure whose register allocation is per-segment by
construction -- an explicit state machine with state in the frame and each
state its own function. That is what the coroutine is being compiled into
already, minus the union. It is a larger change to the `context-runtime` yield
driver and would be scoped as its own plan.

---

## 9. Explicitly out of scope

**Autotuning.** Compiling at several candidate caps and picking the best by
runtime is *optimisation*, not exact calculation, and it reintroduces an
occupancy objective through the back door. It is not a substitute for fixing
the allocation.

---

## 10. Risks

| risk | handling |
|---|---|
| Diagnosis wrong (pressure is not the union) | Phase 0.3 falsification check gates all implementation work |
| 1b/1c trade registers for ABI call cost | measure ms/step, not just REG; reject on runtime regression |
| Sinking changes semantics around side effects | restrict to coro-frame loads; gates 3.2 are digit-exact |
| Lower registers destabilise the fetch machinery | a known prior effect at 4 blocks/SM (intermittent stale pages); 3.2 runs out-of-core specifically to surface it |
| Two consumers drift | Phase 2 deletes the duplicate pass copy; 3.4 rebuilds LAMMPS |

---

# 11. PHASE 0 RESULTS -- the verified root cause

Run 2026-08-21. Every claim below is backed by an artefact, and the section-3
hypothesis was refuted in the process.

## 11.1 What was refuted

**The liveness-union model (section 3.2) is wrong**, on two counts:

1. *"The resume function is inlined into the kernel."* **False.** The device
   PTX contains 69 `.func` definitions including distinct `$_resume` and
   `$_destroy` symbols per coroutine (`GatherCoro_$_resume`,
   `ListForceCoro_$_resume`, ...). They are separate functions, not inlined.
2. *"Register count is the union of segment live sets."* **False.** An empty
   `co_return`-only coroutine has no suspend points and therefore no union to
   take, yet it costs the same as the full paging machinery -- evidence that
   was already in hand and that the union model cannot explain.

## 11.2 The verified mechanism

**An indirect call forces ptxas to allocate the calling kernel for the worst
case over every address-taken function in the module.**

`CLIO_YCORO_RUN` resumes through a runtime pointer:

```cpp
__builtin_coro_resume(reinterpret_cast<void *>(_yc_lane->coro_resume_));
```

which lowers to an indirect call (the device PTX carries 63 `.callprototype`
sites). Every coroutine kernel therefore inherits the maximum register
requirement over *all* `$_resume` bodies in the translation unit -- including
the resume functions of coroutines it can never reach. That is exactly why
eleven kernels with unrelated bodies report an identical count.

Measured on the real TU (`ptxas -arch=sm_89 -O3 -v`): all eleven coroutine
kernels **122**, the one plain kernel **8**. Only entry functions report
registers; the `.func` resume bodies do not, because their cost is folded into
each entry's requirement.

## 11.3 The controlled experiment (no coroutines involved)

`scratchpad/indirect{,2,3}.cu` -- plain CUDA, one address-taken heavy function:

| kernel | heavy fn address-taken | heavy fn NOT address-taken |
|---|---|---|
| `K_TrivialBodyIndirect` (body = one store, + indirect call) | **40** | **24** |
| `K_TrivialNoIndirect` (identical body, no call) | 8 | 8 |
| `K_CallsHeavyDirectly` | -- | 40 (honest: it really calls it) |

A kernel whose entire body is a single store pays 40 registers for a function
it never calls, purely because that function's address escaped somewhere in
the module. Removing the escape drops it to 24. **This is the whole bug, in
miniature.**

Two further controls confirm the cost is not the ABI itself:
`K_IndirectBig` (34) equals `K_BigNoCall` (34) -- an indirect call adds nothing
over the body's own need when the callee set is light; and a direct call to a
known callee is free (`K_DirectCall` 8, inlined).

## 11.4 What does NOT work: `.calltargets`

PTX can annotate an indirect call site with its candidate targets. ptxas
**parses and validates** the directive -- a bogus name gives
`error: Illegal call target, device function expected` -- but it does **not**
use it to narrow register allocation:

| call site annotation | REG |
|---|---|
| `.callprototype` (no targets) | 40 |
| `.calltargets` naming only the light callee | **40** |
| `.calltargets` naming both callees | 40 |

Per-call-site narrowing is therefore not available in ptxas 13.2. The only
lever that moved the number was changing the **module-wide address-taken set**.

## 11.5 Consequence for the fix

The register count a kernel *should* have is the max over the resume functions
it can actually reach. ptxas already computes exactly that -- it is simply
being given too large a candidate set, because the whole benchmark lives in one
translation unit. Shrink the set to what each kernel can reach and the number
becomes exact, per-kernel, and automatic, with no constant anywhere.

---

# 12. Phase 1 (revised) -- narrow the indirect callee set

Replaces section 5, which was written against the refuted model.

**12.1 Partition the module.** Compile each kernel together with only its own
coroutine chain, so the address-taken set ptxas sees per module is the set that
kernel can actually reach. With `-fgpu-rdc` ptxas runs per TU and nvlink
combines cubins afterwards, so per-TU partitioning yields per-TU allocation.
Two candidate mechanisms, to be chosen on measurement:

- *Source-level*: split `clio_gpu_vector_md_bench.cc` so each kernel and its
  coroutines form a TU. Simple, no tooling, but it is a source restructuring
  and it is the benchmark's layout, not a general fix for consumers.
- *IR-level (preferred)*: an LLVM module-splitting pass that partitions the
  device module by kernel reachability before codegen. General -- every
  consumer of the yield machinery benefits without touching its sources.

**12.2 Devirtualize where the target is static.** Where the resume target is
statically known (a chain of depth one, the common case for the leaf
coroutines), replace the indirect resume with a direct call so the callee never
escapes at all. `K_DirectCall` measured 8 -- a direct call to a known callee
costs nothing. This is complementary to 12.1 and shrinks the residual set.

**12.3 Acceptance.** Per-kernel REG varies with body; `SentinelKernel` (a
one-line coroutine) must not report the same count as `ListForceKernel`. That
inequality is the test that the bug is fixed, and it is the CI guard in 3.5.

**12.4 Cost to weigh.** Partitioning trades whole-program inlining across
kernels for correct allocation. Measure ms/step, not only REG: a kernel that
allocates 40 instead of 122 but loses an inlining opportunity may not be
faster. Section 3.3's multi-`--threads` protocol applies.
