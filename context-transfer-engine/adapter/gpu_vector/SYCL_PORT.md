# gpu_vector on SYCL

The device-paged vector runs on SYCL. `test_gpu_vector_smoke_sycl` writes
every element through a held page, flushes it to the CTE, drops the cache,
reads it all back through real page faults, and reports zero mismatches --
including the oversubscribed case where 16 pages go through 4 frames and
every page is evicted and re-fetched.

```
[sycl-smoke] faults=8  puts=4  evicts=0  get_err=0 put_err=0   mismatches=0 / 4096
[sycl-evict] faults=32 puts=16 evicts=24 get_err=0 put_err=0   mismatches=0 / 16384
[sycl-multi-2]                                                 mismatches=0 / 4096
[sycl-multi-8]                                                 mismatches=0 / 16384
Passed: 1  Failed: 0
```

## One implementation, not two

`device_vector.h` is **unchanged**. So are `yield_coro.h`'s coroutine
machinery, the lane stack, the host relaunch driver, the page cache, the CTE
client and the task-submission path. The CUDA build still passes its own
smoke test through 128 blocks.

That is possible because the CUDA surface those 4000 lines actually use is
twelve names, and `clio_ctp/util/sycl_cuda_compat.h` supplies them for SYCL:

```
threadIdx blockIdx blockDim   __syncthreads __syncthreads_or
atomicAdd atomicCAS atomicSub atomicExch atomicOr atomicAnd atomicXor
__threadfence __threadfence_system __trap __nanosleep clock64 __shfl_sync
printf   __device__ __host__ __global__ __align__ __forceinline__   dim3
```

The header is inert unless `CTP_ENABLE_SYCL`, so the CUDA path is touched in
zero places. What makes it work at all is
`sycl::ext::oneapi::this_work_item::get_nd_item()` -- a free-function
work-item query. Without it every one of those names would have to become a
parameter, and a shared `device_vector.h` would be impossible.

Two of the twelve are not cosmetic:

- **`clock64()`** is the LRU tiebreak in the eviction policy. Returning a
  constant compiles fine and silently changes which frame gets evicted, so
  it uses the NVPTX `%clock64` register and says so loudly where it cannot.
- **`atomicCAS`** on 64-bit types uses inline PTX, because
  `sycl::atomic_ref::compare_exchange_strong` emits a call to
  `__clc_atomic_compare_exchange` that libspirv does not define for
  `nvptx64`. Shared-library device linking dies at
  `ptxas fatal: Unresolved extern function '_Z29__clc_atomic_compare_exchangePmmmiii'`.
  32-bit compare-exchange links fine; this is a DPC++ bug
  (nightly-2026-08-27), reproduced on a four-line kernel.

## What could NOT be shared, and why

**The kernel bodies' outermost layer, and the translation unit boundary.**

A CUDA kernel is a `__global__` function at namespace scope with the host
test beneath it, so host-only code is removed from the device pass with
`#if !CTP_IS_DEVICE_PASS`. A SYCL kernel is a **lambda inside the function
that submits it**, and DPC++ member-checks the entire translation unit in
its device pass. There is no preprocessor line that keeps the kernel and
drops the host code around it -- they are the same statement.

So the SYCL side is two files:

- `test/sycl/gv_sycl_kernels.cc` -- compiled with `-fsycl`, holds the
  coroutines and the launchers. **The coroutines are the CUDA ones
  verbatim**; compare `FillCoro` here with `FillCoro` in
  `test/test_gpu_vector_smoke_gpu.cc`.
- `test/sycl/test_gpu_vector_smoke_sycl.cc` -- ordinary C++, holds the host
  test and drives the launchers through `gv_sycl_kernels.h`.

The same rule bit the runtime: `gpu2cpu_init_sycl.cc` had its kernel inside
`#if CTP_IS_HOST`, so no device code was emitted for it and the host pass
then submitted a kernel the runtime had never heard of --
`Assertion 'It != m_DeviceKernelInfoMap.end()'`. The kernel is now hoisted
into a function that touches no host-only state.

## Bugs this found in the existing SYCL path

None of these had ever been executed. Each one was silent.

| Where | What |
| --- | --- |
| `macros.h` | `CTP_IS_HOST` stayed 1 in the SYCL device pass, so a dozen `#if !CTP_IS_HOST` guards -- the ones that keep host-only bodies out of the device pass -- did nothing. |
| `gpu_api.h` | `MemcpyAsync` had CUDA and ROCm branches only: **a silent no-op under SYCL**. The RAM bdev writes pages back with it, so every flushed gpu_vector page kept its old contents, with `rc=0` and `put_errors=0`. |
| `gpu_api.h` | `DeviceAwareMemcpy` fell through to `std::memcpy` on device pointers -- segfaulted the CPU worker staging a task. |
| `gpu_api.h` | `IsDeviceAccessiblePointer` answered "false" for real device memory, so every caller that uses it to pick a GPU path took the host one. |
| `gpu_api.h` | `MemsetAsync` did not wait, so it was ordered with respect to nothing. |
| `gpu_vector.h` | 14 `#if CTP_ENABLE_CUDA` guards around bodies that are pure `ctp::GpuApi` -- including `PublishHeader`, so the device `VecHeader` was never allocated and the first kernel dereferenced null. |
| `gpu_device_ring.h` | `Push` was `CTP_IS_GPU_COMPILER`-only; rewritten against the portable intrinsics so one body serves all three backends. |
| build | `CTP_ENABLE_SYCL` was per-target. It selects between two bodies of the same inline functions in `gpu_api.h`, so an uneven definition is an ODR violation -- and that is exactly how `mem_bdev_transport.cc` got the no-op `MemcpyAsync`. It is now global, with `libsycl` linked globally to match. |

## Backend matrix

Where each workload stands. "paged" is the gpu_vector row; MPI/Kokkos are
baselines that link nothing from clio.

| workload  | paged CUDA | paged SYCL | MPI | Kokkos | paged: multi-node | baselines: ranks |
|-----------|-----------|-----------|-----|--------|-------------------|------------------|
| kmeans    | yes | yes | yes | yes | n/a -- single-node bench | 1 + 2 |
| grayscott | yes | yes | yes | yes | n/a -- single-node bench | 1 + 2 |
| weights   | yes | yes | yes | yes | n/a -- single-node bench | 1 + 2 |
| lbann     | yes | yes | yes | yes | n/a -- single-node bench | 1 + 2 |
| gmx       | yes | yes | yes | yes | n/a -- single-node bench | 1 + 2 + 4 |
| lammps_md | yes | yes | yes | yes | **2 nodes, BOTH backends** | 1 + 2 + 4 |

READ THE TWO RIGHT-HAND COLUMNS SEPARATELY. Only lammps_md's paged bench has
`--nodes/--node`; the other five paged benches are single-node BY DESIGN, on
CUDA as much as on SYCL, so "no distributed SYCL row" there is not a porting
gap and nothing is missing. The rank counts in the last column belong to the
MPI and Kokkos BASELINES, which are separate programs that link nothing from
clio -- they say nothing about the paged rows.

All six workloads now build and run on both backends. lammps_md was the last,
and it needed a seam split of its own (md_common.h / md_kernels.h /
md_launch.h + cuda/ + sycl/) because its `__global__` trampolines were
interleaved among the device coroutines.

**What the lammps_md SYCL row is validated against.** The CUDA row on
identical flags (`--md --lattice 20 --steps 20 --blocks 8 --slots 8`): all
four gates pass and every number matches digit for digit, including the paging
counters -- `PE/atom -6.7733683`, `pairs 864000`, `W -709062`, resort PE
`-216747.785344` at `rel=0.00e+00`, `E0=-215787.790233 En=-215787.666467
drift=5.74e-07`, `x faults=22 evicts=0, x puts=493`.

**Distributed, both backends.** The decomposed 2-node harness (one domain
split across nodes, halo carried by the generational exchange -- not two
independent decks) passes on CUDA and on SYCL with IDENTICAL numbers, and both
match the harness's own documented reference:

    E0 = -592121.595111  En = -592121.125973  drift 7.92e-07
    pairs 2370816        RESORT PE -594755.829466 at rel=0.00e+00

    GVMB_TAG1= GVMB_TAG2= GVMB_DECOMP1="--nodes 2 --node 0" \
    GVMB_DECOMP2="--nodes 2 --node 1" \
    GVMB_ARGS="--lattice 28 --steps 20 --blocks 8" \
    BUILD_DIR=build-cuda ./run_md_bench_distributed.sh

**RUNNING THE SYCL BUILD IN THAT HARNESS**, which the deps-cpu image cannot do
on its own -- it has no DPC++ -- WITHOUT editing the compose file:

    ln -sf clio_lammps_md_paged_bench_sycl \
           build-sycl/bin/clio_lammps_md_paged_bench     # the command hardcodes the name
    CUSZP_LIB_DIR=$HOME/opt/dpcpp/lib BUILD_DIR=build-sycl ...same as above...

`/opt/cuszp-lib` is already on the container's LD_LIBRARY_PATH, so pointing
CUSZP_LIB_DIR at the DPC++ lib dir mounts libsycl AND libur_adapter_cuda.so
where the loader finds them. Reusing an existing mount slot beats adding one.

**Why two nodes on ONE host is not an option.** The runtime deliberately drops
a configured hostfile when the bind address is loopback ("single node, this
machine only"), so 127.0.0.1/127.0.0.2 aliases collapse to a single-node run
rather than a two-node one. Non-loopback local addresses need root. Docker is
the route.

**Re-verified after the shared compat header changed.** Adding `__fmaf_rn`
and turning `__ldcg`/`__ldcv` into volatile loads touched a header ALL SIX
rows include, so all five previously-validated rows were rebuilt and re-run
against their CUDA twins on identical flags:

| row | result |
|-----|--------|
| grayscott | `v_checksum=12381560.488928` bit-identical |
| lbann     | LOSS and WEIGHT gates pass on both |
| gmx       | CONSERVATION, MESH and GATHER pass on both |
| weights   | `checksum=OK`, `get_errors=0 put_errors=0` on both |
| kmeans    | counters exact; checksum not reproducible -- see below |

**DO NOT COMPARE kmeans BY ITS CENTROID CHECKSUM.** It is not reproducible
run to run on a SINGLE binary: the CUDA build alone gave `30720.001988` then
`30720.000665` on two consecutive runs, because centroid accumulation is
atomic-ordered. Comparing that number across backends manufactures a
"regression" that is not there. Compare `faults`/`evicts`/`get_errors`/
`put_errors`, which are exact (131072/131072/0/0 on both). weights'
fault/evict counts likewise track the paging schedule and drift a little
(410/346 vs 401/338); its correctness signal is `checksum=OK` with zero
errors.

**Eviction IS exercised on SYCL** -- kmeans evicts 131072 pages and weights
346, both with zero get/put errors. The eviction gap noted below is specific
to the MD deck's block geometry, not a property of the port.

**What it is NOT validated against, and why.** EVICTION. The ctest entry named
`_ooc` does not exercise it: `--slots 8` is clamped up to the 28-frame pin
floor, so both backends report `evicts=0` and "resident contract HELD". Real
eviction needs a larger working set, and at `--blocks 8` the CUDA row itself
dies with `FATAL set full` for lattice 24, 26, 28 and 32 -- a shared
set-associativity limit at that block count, not a SYCL gap. Faults are
exercised (22 of them, plus 493 puts); eviction is not, on EITHER backend, at
8 blocks. Raising the SYCL block ceiling (see Known limits) is what would let
this be tested.

Two facts about the Kokkos rows that are easy to misread:

- **Kokkos is a programming model here, not a transport.** Every Kokkos row
  still moves its halo over two-sided host-staged MPI, exactly as its MPI
  sibling does. What changes is how the device side is expressed.
- **Kokkos rows do not match the nvcc rows bit for bit, and should not be
  expected to.** gmx makes this concrete: its mesh checksum differs from the
  nvcc baseline in a handful of last units purely because nvcc defaults to
  `-fmad=true` and clang-CUDA contracts differently. Rebuilding the baseline
  with `nvcc -fmad=false` and the Kokkos row with `-ffp-contract=off` makes
  the two agree bit for bit -- measured, not assumed. Use integer-exact gates
  (conservation, pair counts, migrant counts) and rank-count invariance for
  cross-build comparison; use float checksums only within one build.

## Known limits

- **8 blocks.** At 32 the multi-block phase wedges: the driver relaunches and
  no block completes. The CUDA build carries the same workload to 128, but it
  submits through the **batched device ring**, and that transport is not
  ported -- `ServerInitGpuQueues (SYCL)` leaves `ring.dev_ring` null and every
  submission falls back to the legacy per-task `GpuTaskQueue`, built with one
  lane. `RingNext` is a documented stub. Porting the ring is the same work
  item as raising this bound, and it is mechanical: SYCL's `malloc_host` is
  directly device-addressable, so it needs no `cudaHostGetDevicePointer` step.
- **No performance claim.** `MemcpyAsync` is synchronous under SYCL and the
  ring is absent, so both halves of the writeback pipeline are serialized.
  Nothing here has been measured against the CUDA path and it should not be.
- **NVPTX only.** SPIR-V cannot compile a coroutine at all (see
  `context-runtime/test/unit/gpu/spike/SYCL_COROUTINES.md`), so
  `test/sycl/CMakeLists.txt` refuses to build a target that would only crash.
  Configure with `-DSYCL_TARGET=nvptx64-nvidia-cuda -DSYCL_CUDA_ARCH=sm_XX`.
- **GPU codecs.** The compressed page path is nvcomp, which has no SYCL
  equivalent; this build runs with `CLIO_CTE_ENABLE_COMPRESS=OFF`.
- Stats are opt-in (`EnableStats()`), and the SYCL test turns them on --
  `get_errors` / `put_errors` are the only place a failed transfer shows up.

## Building

```
cmake -B build-sycl \
  -DCMAKE_C_COMPILER=$DPCPP/bin/clang -DCMAKE_CXX_COMPILER=$DPCPP/bin/clang++ \
  -DCLIO_CORE_ENABLE_SYCL=ON -DSYCL_TARGET=nvptx64-nvidia-cuda \
  -DSYCL_CUDA_ARCH=sm_89 -DCLIO_CTE_ENABLE_COMPRESS=OFF \
  -DCLIO_CORE_ENABLE_IO_URING=OFF -DCLIO_CORE_ENABLE_TESTS=ON
cmake --build build-sycl --target test_gpu_vector_smoke_sycl
ctest --test-dir build-sycl -R cte_gpu_vector_smoke_sycl
```
