# Can the yieldable-kernel coroutine machinery run under SYCL?

**Yes on SYCL→CUDA (NVPTX). No on SYCL→SPIR-V — DPC++ crashes in codegen on
any coroutine at all.**

Three spikes in this directory, all measured on an RTX 4070 Laptop with
DPC++ nightly-2026-08-27 (`intel/llvm`, CUDA adapter), all PASS:

| spike | what it proves |
| --- | --- |
| `spike_sycl_coroutine.cpp` | the primitive: a coroutine frame in OUR memory, suspended inside a loop, resumed across 6 separate kernel launches with ordinary locals intact |
| `spike_sycl_lane_coroutine.cpp` | the full `yield_coro.h` shape: per-work-item chains, block-collective voted suspend, the trampoline, a value-returning child, a group barrier inside a coroutine |
| `spike_sycl_implicit_lane.cpp` | that the port is macro substitution, not a signature rewrite |

Build/run everything: `./build_sycl_spikes.sh`.

## What ports for free, and what SYCL actually makes *easier*

The CUDA implementation pays two taxes that SYCL simply does not charge:

1. **No `<coroutine>` shim.** `spike_device_coroutine.cu` had to re-declare
   the whole of `std::coroutine_handle` with `__device__` on every member,
   because libstdc++ marks them host-only, and then go through
   `__builtin_coro_*` by hand for the rest of `yield_coro.h`. SYCL is
   single-source with no host/device function attributes, so stock
   libstdc++ `<coroutine>` is callable from device code exactly as written —
   `h.resume()`, `h.done()`, `h.promise()`, `from_address`, all of it. Every
   `__builtin_coro_*` call in `yield_coro.h` exists only to work around a
   restriction that does not apply here.

2. **No signature churn.** SYCL has no `threadIdx`, which at first forces a
   context parameter through every yieldable function — and worse, through
   `promise_type::operator new`, whose signature must match the coroutine's
   parameters exactly. `spike_sycl_implicit_lane.cpp` shows the way out:
   `sycl::ext::oneapi::this_work_item::get_nd_item<1>()` is a free-function
   work-item query, so the CUDA spellings become one-line macros

   ```
   #define SY_THREADIDX_X    (twi::get_nd_item<1>().get_local_id(0))
   #define SY_SYNCTHREADS()  sycl::group_barrier(twi::get_work_group<1>())
   #define SY_SYNCTHREADS_OR(c) sycl::any_of_group(twi::get_work_group<1>(), (c))
   ```

   and `YieldLane()` derives the lane header from those plus a
   `device_global` arena base. `promise_type::operator new` goes back to the
   ordinary one-argument form and every yieldable function keeps its
   signature.

The rest of the device stack maps 1:1: `__threadfence_system` →
`sycl::atomic_fence(…, memory_scope::system)`, `atomicCAS`/`atomicAdd` →
`sycl::atomic_ref`, `__trap` → `assert(false)` or a device-side abort. The
constructs the yieldable stack actually uses (counted across `yield_coro.h`,
`yield_stack.h`, `device_vector.h`) are exactly these; there is no CUDA
intrinsic in that path without a SYCL equivalent.

## The SPIR-V wall

```
clang-24: llvm/include/llvm/IR/Instructions.h:2790:
  Assertion `getType() == V->getType() &&
             "All operands to PHI node must be the same type as the PHI node!"'
  failed.
...
3. Generating code for declaration '(anonymous namespace)::Walk'
11 clang::CodeGen::CodeGenFunction::EmitCoroutineBody(...)
```

This is not our code. A four-line coroutine with no custom allocator, no
arena and no SYCL beyond the kernel launch reproduces it. It is
unconditional: `-O0`, `-O2`, and toggling
`-foffload-use-alloca-addrspace-for-srets` all crash identically, and the
same file compiles clean with `-fsycl-host-only`. The shape of the assertion
(a PHI over pointers of mismatched type, in the coroutine-frame emission for
a target whose allocas are not in address space 0) says the SPIR-V target's
address spaces and clang's coroutine frame lowering have never been made to
agree. Worth an `intel/llvm` issue; not something a port can work around.

Beyond the crash there is a second, more fundamental problem waiting: a
coroutine resume is an **indirect call** through a pointer in the frame, and
the emitted PTX confirms the shape — 6 `.callprototype` sites, with the
coroutine bodies emitted as separate out-of-line `Walk_$_resume` /
`Child_$_resume` functions rather than inlined into the kernel. SPIR-V has
no function pointers without `SPV_INTEL_function_pointers`, so even with the
codegen crash fixed, the Intel-GPU path would depend on that extension.

**So: SYCL as a portability layer over NVIDIA works today. SYCL as the route
to Intel GPUs does not, and the blocker is upstream.**

## Registers

DPC++ never runs `ptxas` for the NVPTX target — it embeds PTX and lets the
CUDA driver JIT it, so `-Xsycl-target-backend "--ptxas-options=-v"` is
silently inert and there is no build-time register report. `maxrregcount` is
still reachable (the CUDA adapter parses it out of build options into a
`CUjit_option`, settable via `SYCL_PROGRAM_COMPILE_OPTIONS`), but its effect
cannot be measured at build time. To get numbers, dump the PTX and run
`ptxas` by hand:

```
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -save-offload-code=. …
ptxas -arch=sm_89 -v spike_*.s -o /dev/null
```

On the lane spike that gives 30 registers for the kernel entry, with the
coroutine bodies out-of-line and spilling (48 B frame in `Walk_$_resume`,
24 B in `Child_$_resume`). This is the same indirect-call-driven register
profile the CUDA build has, so the known register ceiling is neither fixed
nor made worse by SYCL — but the tooling to *see* it is worse, and losing
build-time `ptxas -v` on a codebase whose main GPU ceiling is register
pressure is a real cost.

## What was NOT established

These spikes prove the *mechanism*. They do not touch the much larger
question of whether the clio device stack as a whole compiles under DPC++:
`device_vector.h` and the page-cache machinery reach into runtime headers,
shared-memory allocators and task types that have only ever been through
clang-CUDA. `CLIO_CORE_ENABLE_SYCL` exists in the build but the `gpu_vector`
adapter is CUDA-only. Nothing here says how big that job is.

## Reproducing

Toolchain (no sudo, ~424 MB download, extracts to `~/opt/dpcpp`):

```
curl -L -o sycl.tgz \
  https://github.com/intel/llvm/releases/download/nightly-2026-08-27/sycl_linux.tar.gz
mkdir -p ~/opt/dpcpp && tar xzf sycl.tgz -C ~/opt/dpcpp
export PATH=~/opt/dpcpp/bin:$PATH LD_LIBRARY_PATH=~/opt/dpcpp/lib:$LD_LIBRARY_PATH
sycl-ls   # should list [cuda:gpu]
```

Note the nightly ships **only** the CUDA adapter — no OpenCL or Level Zero —
so an Intel GPU could not have been tested here even if the SPIR-V target
compiled.
