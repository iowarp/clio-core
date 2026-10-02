# CUDA: the leg that was written but unexercised

`README.md` listed the CUDA/HIP arm of `Item` as "five token-identical lines,
but unexercised is unexercised". This is the exercise. It covers semantics on
a real NVIDIA GPU and the occupancy the mechanism costs, which are separate
questions and are kept separate below.

Host: RTX 5080, sm_120, 84 SMs, 48 warps/SM, 65536 registers/SM.
Compiler: clang 23 (`-x cuda`) with CUDA 13.2, `-O2`, inside
`iowarp/clio-core-devcontainer`. Cross-checked against nvcc 13.3 + MSVC on
the Windows host, which agrees to within two registers (22/78 vs 26/80).

## Four bugs, all in the arm that had never been compiled

Each was a hard error, not a warning, and none of them can appear on SYCL or
on the host backend:

1. **`StackView::Header` was `__device__`, and `driver.h` is host code.** The
   host driver lays the stack out and reads the headers back, so the one piece
   of pure address arithmetic both sides use has to be `__host__ __device__`.
   Now spelled `CLIO_CO_HD`, with the rule that anything wearing it touches no
   builtin and no group state.
2. **`PackBytes` / `MaxOf` were plain `constexpr`, called from `__device__`
   code.** nvcc rejects that without `--expt-relaxed-constexpr`, and a header
   should not make every consumer pass an experimental flag. Now `CLIO_CO_CE`.
3. **`__builtin_memcpy` in the save/restore path.** A GCC/Clang spelling, so it
   covered icpx, hipcc and nvcc-on-Linux, and not nvcc with MSVC as host
   compiler. Now `CopyBytes`, over plain `memcpy`, which all three device
   libraries have.
4. **nvcc cannot compile device coroutines at all** — `error: device code does
   not support coroutines`. Not a bug in this design; a fact about the
   incumbent, and the reason `ClioCoroRegCap.cmake` requires `clang -x cuda`.
   It is also why the C++20 comparison below is built with clang.

## Semantics: green, and identical to SYCL

```
cuda-coroc: launches=13 fetches=18 flushes=18 max_park_depth=2 hwm=240/512 bytes
```

Bit-identical stdout to the synchronous source over 576 values from a
`0xA5`-poisoned stack, and the same statistics the SYCL run produced on Aurora
down to the high-water mark. Three backends, one generated header, same
numbers.

| backend | launches | fetches | flushes | park depth | hwm |
|---|---|---|---|---|---|
| host | 13 | 18 | 18 | 2 | 248 |
| sycl (Aurora PVC) | 13 | 18 | 18 | 2 | 240 |
| **cuda, clang `-x cuda`, sm_120** | **13** | **18** | **18** | **2** | **240** |
| **cuda, nvcc 13.3 + MSVC, sm_120** | **13** | **18** | **18** | **2** | **240** |
| **sycl -> nvptx64, on the RTX 5080** | **13** | **18** | **18** | **2** | **240** |

The last row is the one that would have been easiest to assume and hardest to
justify assuming: the SYCL edition, retargeted at NVIDIA through
`-fsycl-targets=nvptx64-nvidia-cuda`, running the same generated header on the
same GPU as the CUDA edition. Two compilers and two backends over one source,
agreeing on the high-water mark to the byte.

## Occupancy

All five editions compute the same 576 values. `local` is per-thread local
memory, which is the column that decides whether a low register count is real.

| | edition | regs | local | stack | occupancy @256 |
|---|---|---|---|---|---|
| A | synchronous source, no mechanism | 26 | 0 B | 0 | **100%** |
| B | clio-coroc transpiled | 80 | **0 B** | 0 | 50% |
| C | B + `__launch_bounds__(256,4)` | 64 | 72 B | 72 | 66.7% |
| D | C++20 device coroutine, outlined resumes | 38 | 48 B | 48 | **100%** |
| E | D + `__launch_bounds__(256,4)` | 38 | 48 B | 48 | 100% |

Occupancy against block size, for the two that differ:

```
A/D  block=64/128/256/512  ->  48 warps/SM  100%
B    block=64/128/256      ->  24 warps/SM   50%    block=512 -> 33.3%
```

### What the numbers say

**The transpiled state machine keeps every live value in registers.** B is the
only edition with `local=0`, and it holds it across all six suspend points.
That is the design's central claim and it is true as stated.

**It pays for that in occupancy.** 26 → 80 registers halves resident warps at
every block size from 64 up. 80 registers is 2 blocks/SM short of the 40 it
would need for 48 warps.

**`__launch_bounds__` does not buy the cut for free here.** The delta pipeline
got the coroutine paged benches from 192 to 64 registers with `LOCAL:0`. The
same annotation on B reaches 64 registers but spills 72 bytes, because 80 is
what the live set actually needs rather than an artefact of the lowering. Row
C trades local-memory traffic for 16 more warps; whether that is a win is a
per-workload question, not a general one.

**`__forceinline__` is not the cause.** The obvious hypothesis was that
`CLIO_CO_FUN`'s `__forceinline__` inlines the whole chain into the entry point
and inflates the live set, so `CLIO_CO_NO_FORCEINLINE` was added and measured:
**still exactly 80 registers.** The functions are called statically, so clang
inlines them at `-O2` regardless. The register count is a property of the
generated shape, not of the annotation, and the knob is kept only because it
is now a measured fact rather than an assumption.

### The comparison is narrower than it looks

Row D is **not** the incumbent. It is a C++20 coroutine driver written for this
file with an explicit resume stack and no symmetric transfer, which leaves the
resume segments outlined — hence low entry registers and a real ABI stack with
48 bytes of local memory per thread and an indirect call per resume.

The shipping path measures far worse than either: `ClioCoroRegCap.cmake`
records **138 registers for every coroutine kernel in the MD bench against 8
for the plain kernel in the same module**, and the delta pipeline measures
**192**, in both cases near-independent of the kernel body, because NVPTX has
no tail calls and CoroSplit takes the liveness union across all suspend points.

So, on registers alone, against measured numbers:

```
plain kernel, MD bench                    8
synchronous source, this workload        26
clio-coroc, this workload                80      <- body-dependent
C++20 coroutine, MD bench               138      <- body-INDEPENDENT
C++20 coroutine, delta paged benches    192      <- body-INDEPENDENT
```

The interesting property is not that 80 < 138. It is that **80 is a function of
this workload and 138 is not.** A flat penalty on every coroutine kernel is
what makes the incumbent expensive on simple kernels; a body-dependent cost is
what makes the transpiler's ceiling a thing a workload can be optimised
against. Whether that holds at the scale of `clio_lammps_md_paged_bench` is
unmeasured, and is the next thing worth measuring.

## Reproducing

```bash
# transpile (needs clio-coroc; see README)
clio-coroc context-runtime/test/co/coro_workload.h \
  --rewrite-root=$PWD/context-runtime --mirror-to=$PWD/build-gen/context-runtime \
  -- -x c++ -std=c++20 -Icontext-runtime/include -Icontext-runtime/test/co

# build and run the five editions
docker run --rm --gpus all -v "$PWD:/workspace" \
  iowarp/clio-core-devcontainer:latest bash /workspace/context-runtime/test/co/run_cuda.sh
```

## Still not covered

- **ROCm.** The HIP branch shares CUDA's code path and all four fixes above
  apply to it, but no hipcc has compiled it.
- **A real workload.** Every number here is from a six-suspend-point toy. The
  claim that the register cost is body-dependent predicts something specific
  and checkable about a transpiled `clio_gmx_paged_bench`, and that prediction
  is the cheapest way to find out whether the refactor pays.
- **Runtime, as opposed to occupancy.** Nothing here is timed. The delta
  pipeline's finding that raising occupancy from 12.5% to 50% left k-means
  slightly *slower* is the standing warning against reading a register count
  as a performance result.
