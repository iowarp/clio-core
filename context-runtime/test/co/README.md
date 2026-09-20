# P0 spikes — portable GPU coroutines

These validate the *output shape* of the `clio-coroc` design (`$HOME/coroutines.md`)
before the transpiler is written. Everything here is hand-written in exactly the
form the tool will emit, because the assumptions that can sink the design are
about backends, not about tooling — and no amount of tool engineering fixes a
shape a device compiler will not accept.

| file | what it is |
|---|---|
| `spike_workload.h` | the workload in both forms: `src` (what a user writes) and `gen` (what the tool emits). Read them side by side. |
| `spike_host.cc` | S2–S5 on the host backend: a work-group emulated with real threads and a real `std::barrier`. |
| `spike_sycl.cc` | S1 on SYCL: JIT/run, plus AOT through IGC for Aurora's PVC. |

The call graph is deliberately awkward, because a flat one would prove nothing:

```
StreamTile   depth 0   two suspend points, one inside a loop
  HoldPage   depth 1   one suspend point, inside an `if`
    Fetch    depth 2   one leaf await
  Flush      depth 1   one leaf await      <- second child at the same depth
```

## Building and running

```bash
# Host backend (S2-S5). No GPU, no SYCL.
g++ -std=c++20 -O1 -Wall -Wextra -pthread \
    -I context-runtime/include -I context-runtime/test/co \
    context-runtime/test/co/spike_host.cc -o build-spike/spike_host
./build-spike/spike_host

# SYCL, JIT + run on whatever device is present (S1 semantics)
icpx -fsycl -std=c++20 -O2 -Wall \
     -I context-runtime/include -I context-runtime/test/co \
     context-runtime/test/co/spike_sycl.cc -o build-spike/spike_sycl
ONEAPI_DEVICE_SELECTOR=opencl:cpu ./build-spike/spike_sycl

# SYCL, AOT for Aurora's GPU (S1 codegen -- runs IGC, needs no GPU)
icpx -fsycl -std=c++20 -O2 -fsycl-targets=spir64_gen -Xs "-device pvc" \
     -I context-runtime/include -I context-runtime/test/co \
     context-runtime/test/co/spike_sycl.cc -o build-spike/spike_sycl_pvc
```

## Results — 2026-09-20, Aurora login node, oneAPI 2025.3.2 (clang 21)

| spike | what it falsifies | result |
|---|---|---|
| **S1** codegen | the generated `switch`-inside-a-loop (an irreducible CFG) survives a real device backend | **PASS** — `icpx -fsycl-targets=spir64_gen -Xs "-device pvc"` reports `Build succeeded`. IGC accepts it. No `--emit=flat` fallback needed. |
| **S1** semantics | the state machine computes what the source computes, on a device | **PASS** — bit-identical to the synchronous form on the OpenCL CPU device. |
| **S2** barriers | a barrier between two suspend points is legal, crossed after a resume (invariant I2/I3) | **PASS** — no hang; launch count exactly `npages*2 + 1`. |
| **S3** nesting | frame offsets reproduce on a three-level replay descent (invariant I1) | **PASS** — `max_park_depth == 2`; a wrong offset would make `Pop` read another frame and corrupt the output. |
| **S4** persistence | state survives the kernel exit with no fence and no atomic (invariant I8) | **PASS** — each group fetched and flushed each page exactly once across 13 launches. |
| **S5** differential | nothing is missing from a save list | **PASS** — parking run bit-identical to the synchronous source, over a poisoned (`0xA5`) stack. |

Numbers from both backends, which agree exactly:

```
stack bytes per work-item (deepest chain): 184
launches=13 parks=36 fetches=18 flushes=18 max_park_depth=2
```

### The premise, confirmed

The reason this mechanism exists is that C++20 device coroutines do not compile
for spir64. Verified on this toolchain with a five-line coroutine: clang does not
merely error, it **crashes**.

```
3. coro_probe.cc:15:6: Generating code for declaration 'DeviceCoro'
 #4 llvm::Value::setName(llvm::Twine const&)
 #5 (anonymous namespace)::PromoteMem2Reg::run()
```

(The design doc records an earlier symptom — a PHI operand-type assert in
`EmitCoroutineBody`. Same class, different crash site; clang 21 gets further
before falling over.) The design emits no coroutine, so it cannot meet this bug
or any successor of it.

## Not yet covered

- **CUDA and ROCm.** Neither `nvcc` nor `hipcc` exists on this node, so the
  CUDA/HIP branch of `Item` is written but unexercised. It is five lines and
  token-identical between the two, but *unexercised is unexercised* — S1 must be
  re-run on a machine with those toolchains before the design's portability claim
  is anything more than an argument.
- **A real GPU.** AOT compilation proves IGC accepts the code; it does not prove
  the PVC executes it correctly. Run `spike_sycl` on a compute node.
- **Divergent `CO_AWAIT`** (rule R3). The workload is uniformly convergent. A
  spike that deliberately violates it, to check that `-DCLIO_CO_VERIFY` catches
  it rather than hanging, is still to write.

## Two things the spikes changed in the design

Both were found by compiling, not by thinking, and both are requirements on the
transpiler's emitter:

1. **Emit `[[fallthrough]];`** at the fresh-entry-into-first-await edge.
   Otherwise every generated file trips `-Wimplicit-fallthrough`, and a
   `-Werror` build fails.
2. **Value-initialize hoisted declarations** (`u64 i{};`). The dispatch shape
   defeats the compiler's definite-assignment analysis, so hoisted variables draw
   `-Wmaybe-uninitialized` even where they are provably assigned before use. This
   is legal precisely because hoisted declarations sit *above* the switch, where
   no jump crosses their initialization — and it is the one place the in-place
   hoisting refinement (design doc §8.4) cannot be used, since that refinement
   requires vacuous initialization.
