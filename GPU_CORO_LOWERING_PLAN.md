# Plan: a GPU-native coroutine lowering (LLVM plugin) for CUDA and ROCm

**Status:** proposed
**Motivation:** `CORO_REGISTER_EXACTNESS_PLAN.md` section 11 (verified Phase 0)
**Targets:** NVPTX (sm_80+) and AMDGPU
**Vehicle:** an LLVM pass plugin (`-fpass-plugin=`), the same delivery mechanism
already wired into this tree

---

## 1. Why the standard lowering is wrong for GPUs

C++20 coroutines were specified for a CPU: type-erased handles, cheap indirect
calls, a real stack, and heap frames. Every one of those is a liability on a
GPU. Phase 0 measured the consequence:

- `std::coroutine_handle<>` is a bare frame pointer, and the frame's first
  words hold pointers to that coroutine's resume/destroy functions. Resuming is
  therefore **always an indirect call**.
- An indirect call makes ptxas allocate the *calling kernel* for the worst case
  over **every address-taken function in the module**. Measured: a kernel whose
  entire body is one store costs **40** registers when a heavy function's
  address escapes elsewhere, **24** when it does not, **8** with no indirect
  call at all.
- Consequence in the real build: eleven coroutine kernels with unrelated bodies
  all allocate an identical **122** registers, because each inherits the maximum
  over every `$_resume` body in the translation unit -- including coroutines it
  can never reach.

Measured alternative, same module, same functions -- a switch over **direct**
calls pays only for what it can reach:

| form | REG |
|---|---|
| indirect call through a table of three | 40 |
| `switch` over direct calls to the two reachable ones | **24** |

The type erasure buys genericity we do not use: a GPU kernel's coroutine chain
is statically known. **We are paying the full cost of dynamic dispatch for a
static call graph.**

Two further GPU-specific mismatches the standard lowering does not model:

- **Park exits the kernel.** Our yield driver returns from the kernel on a
  fault and relaunches it. So `__shared__` state does not survive a suspend --
  a fact that has already caused silent wrong answers. A GPU-native lowering
  should make that a checked property, not a comment.
- **Suspension is block-collective.** All threads in a block suspend together
  (`__syncthreads_or` vote). The standard lowering has no notion of a
  collective suspend and cannot verify it.

---

## 2. Goals and non-goals

**Goals**

- G1. **No indirect calls in device code.** The verification pass fails the
  build if one survives.
- G2. **Exact per-kernel register allocation**, computed by the backend, with
  no ceiling, no `maxnreg`, no assumed block size.
- G3. **Keep `co_await` syntax.** Reuse clang's coroutine front end; replace
  only the lowering. No source churn across gpu_vector, eternia, LBANN.
- G4. **Target-portable**: one IR-level lowering for NVPTX and AMDGPU.
- G5. **Statically bounded chains** verified at compile time.

**Non-goals**

- Host coroutines -- untouched, this is device-only.
- Dynamically-dispatched chains. If the reachable coroutine set is not
  statically resolvable the pass **errors**; it never falls back to indirect
  dispatch, because a silent fallback restores the full register cost with no
  visible symptom.
- Symmetric transfer / arbitrary `await_suspend` returning a handle.
- Exceptions (already unavailable on device).

---

## 3. Architecture: clang's front end, our lowering

The front end is worth keeping -- it parses `co_await`, builds the promise
protocol, computes what crosses a suspend, and reports errors well. Only the
*lowering* is wrong for us.

```
clang parse/sema  ->  presplitcoroutine functions in IR
                          |
              [OUR PASS runs here, at pipeline start]
                          |
        GPU state-machine lowering (no CoroSplit)
                          |
              strip presplitcoroutine attribute
                          |
     rest of the LLVM pipeline (inlining, codegen)  -> PTX / GCN
```

The existing plugin already proves the hook point works: it finds
`presplitcoroutine` functions before CoroSplit erases them and taints callers
up the call graph. That discovery machinery is reused wholesale.

**Critical ordering constraint:** we must run before `CoroEarly`/`CoroSplit`
and must clear `presplitcoroutine` on everything we lower, or the standard
pipeline will lower it a second time.

---

## 4. The lowering, step by step

For each kernel that transitively reaches a coroutine:

**4.1 Build the chain graph.** Walk direct calls from the kernel; collect
reachable coroutines. Every edge must be a direct call. Error with a source
location when the set cannot be closed: a resume through a pointer, a handle
escaping to a function whose body is not visible, or a coroutine whose
definition is not in this TU (see 5.4 on why that does not happen with
header-defined templates).

Note that a **cycle** in the coroutine graph is NOT a problem for the switch:
the reachable *set* stays finite even when the depth does not, so the switch is
still complete. Cycles bound frame depth, not the candidate set, and are
therefore handled by the frame-budget check (C3) rather than here.

**4.2 Number the states.** Assign each `(coroutine, suspend point)` pair a
dense integer ID, unique **within the kernel**. IDs are per-kernel so a kernel
never carries states it cannot reach -- this is what makes register allocation
per-kernel exact.

**4.3 Compute frame layout.** For each coroutine, the set of values live across
each suspend. Union per coroutine gives the frame; the pass emits an explicit
struct type rather than an opaque blob, so the frame is inspectable and
debuggable. Record the exact size for 4.5.

**4.4 Rewrite suspends.** At each `llvm.coro.suspend`:
- store live-across values into the frame,
- store the state ID into the frame header,
- `__syncthreads()` and return from the kernel (park), or fall through to the
  driver loop for a non-parking suspend.

**4.5 Rewrite resume as a switch.** At kernel entry, load the top frame's state
ID and `switch` on it, one case per state, branching directly to that
resumption block. `default: unreachable` -- **the switch must be complete**, or
the original indirect call survives and takes the inflated candidate set with
it (section 1). Reloads from the frame are **sunk into their case**, not
hoisted to the dispatch, so registers are max-over-states, not union.

**4.6 Chain representation.** The lane region already holds a bump-allocated
frame stack (`YCoroAlloc`). Keep it. The frame header becomes
`{ state_id : u32, parent_off : u32 }` -- an offset, not a pointer, so nothing
address-taken exists anywhere. Resuming the deepest frame is a switch on its
`state_id`; returning to a parent is a switch on the parent's.

**4.7 Discriminator choice (settle by measurement first).** Two candidates:
- **integer state ID in the frame** (above). Nothing takes a function address.
- **compare the frame's function pointer** against known resume functions
  (indirect-call promotion). Cheaper to implement, but writing `fp == F` *takes
  F's address*, which is the exact thing that inflates the candidate set.

The ID form is correct by construction; the pointer form may be self-defeating.
**M1 below settles this empirically before any design commits to it.**

---

## 5. GPU-specific properties the pass can now check

Things the standard lowering cannot express, and which have each already cost
us a debugging session:

- **C1. No `__shared__` value may be live across a suspend.** The kernel exits
  on park, so shared memory is garbage on resume. Today this is a comment and a
  class of silent wrong answers; the pass can make it a compile error.
- **C2. Collective suspends.** A suspend inside divergent control flow is a
  hang. The pass can require that every suspend is block-uniform (reachable
  only under block-uniform conditions) and error otherwise.
- **C3. Frame budget.** The lane region is fixed. Summing max chain depth x
  frame size gives a compile-time bound; overflow becomes a build error instead
  of a runtime corruption. This is also where a **cycle** in the coroutine
  graph is handled: a cycle leaves the candidate set finite (so the switch is
  unaffected, 4.1) but makes depth unbounded, so a cycle without a
  statically-provable depth bound is rejected here.

C1 and C2 are arguably worth more than the register win.

---

## 5.4 What kernels this supports (the envelope)

The lowering rewrites only the resume dispatch. **Kernel bodies are untouched**
-- arbitrary control flow, shared memory, atomics, arbitrary `co_await` nesting
depth. There is no constraint on what a kernel computes. The single requirement
is that the set of coroutines a kernel can reach be **visible at compile time**.

| Supported | Rejected (compile error, with source location) |
|---|---|
| Arbitrary kernel logic and control flow | Resuming a handle fetched from a data structure at runtime |
| Deep `co_await` nesting (chains) | Resuming through a function pointer or virtual dispatch |
| **Templated coroutines** -- see below | A handle escaping into a body the pass cannot see |
| Cycles in the coroutine graph (finite set; C3 bounds depth) | A coroutine only *declared* in this TU |
| Many distinct coroutines -- measured to 16 cases, see 5.5 | |

### Templated coroutines are a first-class supported case

They already work, and this tree already depends on it. The 18 resume symbols
in the current device PTX include **two instantiations of the same template**:

```
clio::cte::gpu_vector::DeviceVector<float>::HoldPageCoro_$_resume
clio::cte::gpu_vector::DeviceVector<int>::HoldPageCoro_$_resume
```

By the time the pass runs **there are no templates left**. Instantiation
happens in the front end; the pass sees ordinary concrete functions with
mangled names, so each instantiation is simply one more switch case. Nothing
about templates is special to the lowering.

This also scales without maintenance: adding `DeviceVector<double>` produces a
new instantiation that the reachability walk finds on its own. There is no
registration list to keep in sync.

**The requirement templates impose: definitions must be visible in the TU that
uses them.** A coroutine template instantiated in another TU and only declared
here could not be proven complete, and the pass would (correctly) error. This
does not arise in practice here because `HoldPageCoro`, `EnterHoldSet` and
`AwaitFlush` are **header-defined** in `device_vector.h`, so every
instantiation lands in the same TU as the kernel using it. The complete-switch
requirement is satisfiable by construction -- but it is a real constraint and
must be documented for consumers, not assumed.

### The cost model to design against

Registers become **max over what the kernel can reach** -- not over the module
(today's behaviour), and not over what it happens to execute on a given run. A
kernel touching both a `float` and an `int` vector reaches both `HoldPageCoro`
instantiations and pays the larger.

The practical consequence is that the tuning knob changes character: from "cap
the registers" (a number nobody can set correctly, since block size is a
runtime flag) to "**do not pull a heavy coroutine into a light kernel's
reachable set**" -- a design property visible in the source and under the
author's control.

## 5.5 Switch scaling: measured, not assumed

The obvious failure mode is that a large switch gets lowered back into a
function-pointer table -- reintroducing the indirect call and the whole bug.
Measured at 16 cases of direct calls to distinct `__noinline__ ` functions
(`scratchpad/bigswitch.cu`, sm_89):

| metric | result |
|---|---|
| `callprototype` (indirect call sites) | **0** -- stayed direct |
| `brx.idx` (jump table) | 1 |
| kernel registers | 18 |

The switch does get a jump table, but that is a branch among **known labels**,
not an indirect call, so it does not trigger the callee-set conservatism. V1
guards this permanently by failing the build on any surviving indirect call.

---

## 6. Targets

**NVPTX** is the reference target: all Phase 0 evidence is ptxas.

**AMDGPU is unverified.** The register-pressure story is *assumed* to be
analogous (AMDGPU also allocates conservatively around indirect calls) but this
has not been measured. **M0 measures it before any ROCm claim is made.** The
lowering itself is target-independent -- it is plain IR -- so if AMDGPU does
not have the problem, the pass is still correct there, just not a win.

---

## 7. Milestones

Each milestone is independently shippable and independently revertible.

**M0 -- Evidence (0.5 day).**
Reproduce the Phase 0 indirect-call experiment on AMDGPU. Deliverable: the same
three-row table for `amdgcn`. Decides whether G4 is a real goal or NVPTX-only.

**M1 -- Devirtualize, no new lowering (2-3 days). THE BIG WIN, CHEAPLY.**
Keep CoroSplit. Add a pass that runs *after* it, computes each kernel's
reachable resume set, and replaces the indirect `coro.resume` with a complete
switch over direct calls. Settles 4.7 by measuring both discriminators.
- *Exit:* per-kernel REG varies with body -- `SentinelKernel` no longer equals
  `ListForceKernel`; all MD gates digit-identical; ms/step measured at several
  `--threads`.
- If M1 delivers exact per-kernel allocation, **M2-M4 may not be worth
  building.** Decide there, with numbers.

**M2 -- Custom lowering behind a flag (2-3 weeks).**
Implement sections 4.1-4.6 as a real lowering, opt-in per target, with the
stock path still available. Golden test: bit-identical results to the stock
lowering on the full gpu_vector suite.

**M3 -- Safety checks (1 week).** C1, C2, C3 as build errors, each with a test
that is *supposed* to fail to compile.

**M4 -- Make it the default (1 week).** Flip gpu_vector, eternia, LBANN.
Delete the cap machinery (`tools/coro_regcap`, `ClioCoroRegCap.cmake`,
`MD_MAXRREG`, eternia's `Cap = 128`) -- Phase 2 of the other plan.

**M5 -- ROCm (scoped only if M0 says yes).**

---

## 8. Verification

- **V1.** No indirect calls in device code: scan the emitted PTX/GCN for
  `callprototype` / equivalent. A build-failing check, not a report.
- **V2.** Per-kernel register counts differ across kernels of differing
  complexity. The all-identical signature is the bug and must not return.
- **V3.** Physics gates digit-identical: MD statics/resort/NVE at lattice 100
  and 112, resident and out-of-core; gpu_vector suite (14); cte suite (45).
  GPU tests run sequentially -- `-j` gives false failures.
- **V4.** Performance at matched geometry across several `--threads` values,
  median of 5, warmed GPU. Registers are the mechanism, ms/step is the result.
- **V5.** Cross-consumer: LAMMPS eternia parity gate and LBANN digit gate.
  `lmp` does not relink after `lib/eternia` changes -- delete `lmp-build/lmp`.

---

## 9. Risks

| risk | handling |
|---|---|
| M2 is a large custom lowering with subtle miscompiles | M1 first; M2 only if M1's numbers justify it; golden bit-identity vs stock |
| Discriminator by pointer comparison re-escapes addresses | measured in M1 before design commits (4.7) |
| Losing inlining across the switch | V4 measures ms/step, not registers |
| LLVM version churn (plugin ABI, `presplitcoroutine`) | pin llvm-22; the plugin already vendors the PassPlugin struct |
| AMDGPU may not share the problem | M0 gates the ROCm claim |
| Complete-switch requirement violated silently | V1 fails the build if any indirect call survives |

---

## 10. The honest framing

M1 is a targeted fix to a measured defect and is very likely sufficient for the
register problem. M2-M4 -- "our own coroutine system" -- is justified by the
*safety* properties (C1-C3), not by registers: shared-memory-across-suspend and
divergent-suspend bugs have each cost real debugging time and are invisible to
the stock lowering. That is the argument for building it, and it should be
made on those grounds rather than on performance.
