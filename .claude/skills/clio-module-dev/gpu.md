# GPU: the producer-only model and GPU code rules

Paths are relative to `context-runtime/` unless noted (`CTP/` =
`context-transport-primitives/`). Line numbers were checked on branch
`cte-shm-cache`. For paged out-of-core vectors on the GPU (`gv::Vector`)
there is a separate `gpu-vector` skill.

## 1. The model

The GPU side of the runtime is a **pure task producer**. Kernels do not
allocate tasks, completion records or data buffers, and modules have no GPU
handlers: there is no "GPU runtime" and no per-module `_gpu` library any more
(`cmake/ClioCoreCommon.cmake`, comment in `add_clio_module_runtime`). Every
handler runs on the CPU. All allocation happens on the host before the kernel
launch, inside client-owned device-visible memory backends registered with the
runtime. Inside a kernel the one operation is **`Send`**: push a pre-built task
onto the per-device `gpu2cpu` queue, wait for it, read the results.

## 2. Host lifecycle

1. **Runtime init**: `IpcManager::ServerInit` bootstraps GPU queues
   (`src/ipc_manager.cc:576-627`): `gpu::IpcManager::ServerInitGpuQueues(queue_depth)`
   (`include/clio_runtime/gpu/gpu_ipc_manager.h:315`; implementations in
   `src/gpu/gpu2cpu_init_hip.cc:102`, `src/gpu/gpu2cpu_init_sycl.cc:41`)
   enumerates devices and creates one pinned-host `gpu2cpu` queue per device.
   The scheduler's **GPU worker** (worker N-3 when N >= 5, else worker 0;
   `src/scheduler/default_sched.cc:132-136`) polls the queues and never sleeps.
2. **Client backend allocation**:

   ```cpp
   auto *ipc = CLIO_IPC;
   char *base = nullptr;
   auto alloc_id = ipc->AllocateAndRegisterGpuBackend(
       gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost, bytes, &base);
   ```

   (`include/clio_runtime/ipc_manager.h:1581`, `src/ipc_manager.cc:5055-5075`;
   registered with the runtime by `admin::RegisterMemoryTask`,
   `src/ipc_manager.cc:5122`, `modules/admin/include/clio_runtime/admin/admin_tasks.h:1629`.)
   `MemKind` (`gpu_ipc_manager.h:199-203`):
   - `kPinnedHost` (0): cudaHostAlloc / hipHostMalloc / sycl::malloc_host -- fastest, the default.
   - `kManagedUvm` (1): cudaMallocManaged / sycl::malloc_shared.
   - `kDeviceMem` (2): device-only; the worker copies the POD task
     device->host on pop and the result back host->device on completion
     (`src/ipc/ipc_gpu2cpu.cc:394-406`).
3. **Pre-construct tasks** in the registered backend with placement new.
   GPU-submitted tasks are **POD with identical layout on CPU and GPU**:
   fixed-size fields (e.g. `fixed_string<32>`), no SSO strings or vectors, no
   host pointers; members callable on both sides are `CTP_CROSS_FUN`. The CTE
   has dedicated POD variants for this (`kPodPutBlob`, `kPodGetBlob`,
   `kPodMulti*` in `context-transfer-engine/core/clio_mod.yaml`). The task
   carries its own completion record (`FutureInfo fut_`, `task.h:311-334`).

## 3. Kernel lifecycle

```cpp
__global__ void MyKernel(clio::run::IpcManagerGpuInfo info,
                         ctp::ipc::FullPtr<MyTaskT> task) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (threadIdx.x == 0) {
    task->some_input_ = ...;          // mutate POD fields only
    auto fut = CLIO_IPC->Send(task);  // push onto gpu2cpu_queue
    fut.Wait();
    ... = task->some_output_;         // read results from the same task
  }
}
```

- `CLIO_GPU_INIT(gpu_info, ipc_ptr)` covers both backends
  (`gpu_ipc_manager.h:425-450`): CUDA/ROCm use a per-block `__shared__`
  `IpcManager` (`GetBlockIpcManager`, `:172`) initialized by thread 0 then
  `__syncthreads()`; SYCL uses a per-block record in USM that the host
  stamped (`:191`).
- `gpu::IpcManager::Send(task_ptr)` (`gpu_ipc_manager.h:154`) ->
  `IpcGpu2Cpu::SendIn`; only one thread per block should send.
- Use `PoolQuery::Local()` (or another CPU-side query) for kernel ->
  runtime submission. `PoolQuery::ToLocalGpu` / `LocalGpuBcast` are removed
  and no longer honored (`src/ipc_manager.cc:5600,5617-5619`).

## 4. Worker pop path

`Worker::ProcessNewTaskGpu` (`src/worker.cc:635-644`) delegates to
`IpcGpu2Cpu::RecvIn` (`src/ipc/ipc_gpu2cpu.cc:406`): pop the task, resolve its
pointer in the registered client backend (`gpu::IpcManager::FindClientBackend`,
`gpu_ipc_manager.h:357`), copy it to the host if it lives in device memory,
dispatch the module method on the CPU runtime like any other task, then copy
results back and set the completion bit (`FUTURE_COMPLETE`, `task.h:1127`)
that the kernel polls.

## 5. Forbidden on the GPU

- `CLIO_IPC->NewTask(...)`, `NewObj(...)`, `AllocateBuffer(...)` -- host only.
- `PoolQuery::ToLocalGpu(...)`, `PoolQuery::LocalGpuBcast()` -- removed.
- `IpcCpu2Gpu`, `IpcGpu2Gpu` -- removed with the GPU runtime concept. (A few
  names linger: `friend struct IpcCpu2Gpu;` at `include/clio_runtime/future.h:97`,
  `WaitCpu2Gpu`/`WaitGpu2Gpu` declarations at `future.h:463,472`; do not build
  on them.)

## 6. GPU compiler macro rule

**Never** use raw compiler-detection macros (`__CUDACC__`, `__HIPCC__`,
`__HIP__`, `__CUDA_ARCH__`, `__HIP_DEVICE_COMPILE__`, `__SYCL_DEVICE_ONLY__`)
anywhere except
`context-transport-primitives/constants/include/clio_ctp/constants/macros.h`.
Code must not compile a GPU path merely because a GPU compiler is in use; the
explicit build flags (`CTP_ENABLE_CUDA`, `CTP_ENABLE_ROCM`, `CTP_ENABLE_SYCL`,
set by `CLIO_CORE_ENABLE_CUDA` / `_ROCM` / `_SYCL`) must also be on. Use
(`macros.h`):

| macro | meaning | replaces |
|---|---|---|
| `CTP_IS_GPU` | compiling CUDA/ROCm device code | `__CUDA_ARCH__`, `__HIP_DEVICE_COMPILE__` |
| `CTP_IS_HOST` | compiling host code (0 in any device pass) | |
| `CTP_IS_DEVICE_PASS` | any device pass incl. SYCL device | |
| `CTP_IS_GPU_COMPILER` | nvcc/hipcc **and** the build flag | `__CUDACC__`, `__HIPCC__` |
| `CTP_IS_CUDA_COMPILER` | `CTP_ENABLE_CUDA && __CUDACC__` | |
| `CTP_IS_ROCM_COMPILER` | `CTP_ENABLE_ROCM && (__HIPCC__ \|\| __CUDACC__)` | |
| `CTP_IS_SYCL_COMPILER` / `CTP_IS_SYCL_DEVICE` | SYCL with the flag / its device pass | |

## 7. Building GPU code

- Presets: `cuda-release`, `cuda-clang`, `rocm-debug`, `sycl-debug`
  (`CMakePresets.json`).
- Link GPU kernel code against `ctp::cuda_cxx` / `ctp::rocm_cxx` (available
  only when the backend is enabled); host code against `ctp::cxx`.
- The CMake helpers `add_cuda_library`, `add_cuda_executable`,
  `add_rocm_gpu_library`, `add_sycl_library`, ... live in
  `cmake/ClioCoreCommon.cmake:466-830`.
- GPU tests: `.github/workflows/gpu-tests.yml`,
  `context-runtime/test/unit/gpu/`.
- A GPU page fault costs ~110 us of round trip against ~6 us for the 256 KB
  copy it performs; batch requests (one task carrying many pages) to amortize
  the trip (`context-transfer-engine/core/clio_mod.yaml`, batched POD paging
  comment).
