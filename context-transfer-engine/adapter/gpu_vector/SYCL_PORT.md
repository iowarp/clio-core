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
