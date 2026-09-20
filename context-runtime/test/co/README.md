# Portable GPU coroutines — `CO_AWAIT` on SYCL, CUDA and ROCm

`CO_AWAIT(...)` in device code, transpiled to a group-scoped state machine that
parks and resumes across kernel launches. No C++20 coroutines, no function
pointers, no recursion, no global device state, and **no backend token in the
generated code** — so one transpiled source serves all three backends.

Design: `$HOME/coroutines.md`.

## Run it

```bash
./tools/coroc/build.sh                      # build the transpiler (needs an LLVM with dev files)
./context-runtime/test/co/run_coroc.sh      # transpile + differential test + SYCL + AOT
```

## What each piece is

| file | role |
|---|---|
| `include/clio_runtime/co/coro.h` | device runtime: `Item` (the five-operation backend seam), `Ctx`, `Frame`, `Scope`, and the `CO_AWAIT` marker |
| `include/clio_runtime/co/driver.h` | host side: stack layout and the relaunch loop. Allocates nothing, so it has no per-backend conditional |
| `tools/coroc/main.cc` | the transpiler, on clang LibTooling |
| `coro_workload.h` | **the input.** What a user writes, and all of it |
| `coro_types.h` | data and awaiters. No `CO_AWAIT`, so never rewritten |
| `coro_ref.cc` / `coro_gen.cc` | the differential test's two halves |
| `coro_sycl.cc` | the same generated header, on SYCL |
| `bad_missing_await.h` | negative test: rule R1 must reject it |
| `spike_*.{h,cc}` | the P0 spikes — the output shape written by hand, before the tool existed |

## Results — 2026-09-20, Aurora login node

oneAPI 2025.3.2 (clang 21) for SYCL; GCC 13.4 and LLVM 22 for the tool.

```
PASS  host: transpiled == source (576 values)
PASS  sycl: transpiled == source, on device
PASS  sycl: IGC accepts the generated shape for pvc
PASS  R1: unwrapped call to a suspending function is an error
```

```
gen:  launches=13 fetches=18 flushes=18 max_park_depth=2 hwm=248/512 bytes
sycl: launches=13 fetches=18 flushes=18 max_park_depth=2 hwm=240/512 bytes
```

Thirteen launches is `npages*2 + 1` — every page faults once on the fetch and
once on the flush, and the last launch finds everything ready. `max_park_depth=2`
means the three-level chain really did park at its leaf, so the replay descent
reconstructed `StreamTile > HoldPage > Fetch` and every frame landed back at the
offset it had. The stack is poisoned with `0xA5` before each run, so a value
missing from a save list comes back as garbage rather than a plausible zero.

### The premise, confirmed

A five-line C++20 device coroutine does not merely fail to compile for spir64 —
clang **crashes** on it:

```
3. coro_probe.cc:15:6: Generating code for declaration 'DeviceCoro'
 #4 llvm::Value::setName(llvm::Twine const&)
 #5 (anonymous namespace)::PromoteMem2Reg::run()
```

This design emits no coroutine, so it cannot meet that bug or its successors.

## What the transpiler does

Five edits, and nothing else:

| | edit |
|---|---|
| E1 | append `clio::co::Ctx &_cy` to every suspending function's signature |
| E2 | append `_cy` at every call to a suspending function |
| E3 | insert the `Frame`, the hoisted declarations and the dispatching `switch` |
| E4 | insert `case N:` + replay + park guard at each `CO_AWAIT` |
| E5 | close the switch and call `Done()` before every return |

A function is suspending **iff its body contains `CO_AWAIT`** — local and
explicit, never inferred transitively, so there is no whole-program fixed point
and a TU can be processed alone.

The hoisting rule is one sentence: *a declaration moves to function scope iff
its block lexically contains a `CO_AWAIT`.* It moves the **declaration**, not the
variable — the hoisted variable is an ordinary automatic, so loop counters stay
in registers. It exists only so the dispatch can jump past them without entering
the scope of a variable with non-vacuous initialization, which is ill-formed.
That also means **a missed hoist is impossible**: it would be a compile error in
the generated file, not wrong data.

## Not yet covered

- **CUDA and ROCm.** Neither `nvcc` nor `hipcc` is on this node, so the CUDA/HIP
  branch of `Item` is written but unexercised. Five token-identical lines, but
  unexercised is unexercised.
- **A real GPU.** AOT proves IGC accepts the generated code; it does not prove a
  PVC executes it. Run `coro_sycl` on a compute node.
- **Precise liveness.** The save list is every parameter plus every hoisted
  variable — a superset of the live set, so always correct, costing frame bytes
  at a park and nothing on the fast path. `clang::LiveVariables` would shrink it.
- **Expression-position awaits**, range-`for` desugaring, `CO_AWAIT` directly in
  a kernel lambda. All diagnosed rather than mis-compiled.
- **Randomized park schedules** in the differential test, and the
  divergent-`CO_AWAIT` check behind `-DCLIO_CO_VERIFY`.

## Five things found by building rather than by thinking

Each is now a requirement on the emitter or on the runtime, and each was a bug
first:

1. **`Await(T&&)` cannot take a void await.** `CO_AWAIT(Fetch(c, page))` awaits
   a void-returning call, and no function can take a void argument. The marker
   is a comma expression — `(AwaitMark(), (e))` — because the built-in comma
   accepts a void operand and yields its right operand with the same type *and*
   value category.
2. **Macro-argument text needs spelling locations, not expansion.** The operand
   of `CO_AWAIT` is a macro argument whose expansion range is the whole
   invocation, so asking for it yields the marker back. This first appeared as
   generated code containing `CO_AWAIT(Fetch(c, page), _cy)`.
3. **A statement ending inside a macro has an unreliable expansion end.** The
   park guard must be anchored on the `CO_AWAIT` itself; anchored on the
   enclosing statement it skipped past its own semicolon and landed in the
   middle of the next line's `for`-init.
4. **Emit `[[fallthrough]];`** at the fresh-entry-into-first-await edge, or every
   generated file trips `-Wimplicit-fallthrough` and a `-Werror` build fails.
5. **Value-initialize hoisted declarations, and emit an unreachable `return {}`.**
   The dispatch shape defeats definite-assignment and reachability analysis, so
   hoisted variables draw `-Wmaybe-uninitialized` and non-void functions draw
   `-Wreturn-type` even where both are provably fine.
