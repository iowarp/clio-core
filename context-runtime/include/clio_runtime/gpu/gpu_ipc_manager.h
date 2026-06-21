/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CLIO_RUNTIME_INCLUDE_GPU_IPC_MANAGER_H_
#define CLIO_RUNTIME_INCLUDE_GPU_IPC_MANAGER_H_

#include "clio_runtime/api.h"
#include "clio_runtime/types.h"
#include "clio_runtime/task.h"
#include "clio_runtime/gpu/gpu_info.h"
#include "clio_runtime/gpu/future.h"
#include "clio_runtime/ipc/ipc_gpu2cpu.h"

#if CTP_ENABLE_GPU

#include <memory>
#include <unordered_map>
#include <vector>

namespace clio::run {
namespace gpu {

/**
 * Producer-only GPU IPC infrastructure manager.
 *
 * After the GPU runtime concept was deleted, the GPU is a pure task
 * producer: kernels do not allocate. The host pre-allocates Task and
 * data buffers in client-owned device-memory backends and registers
 * them with the runtime via the admin RegisterMemory API. Inside a
 * kernel the only operation this class exposes is `Send` — pack a
 * pre-allocated task and push it onto the per-device gpu2cpu_queue.
 *
 * Host-side (CPU) state held here:
 *   - `per_gpu_devices_`: one PerGpuDeviceState per physical GPU,
 *     each holding the gpu2cpu_queue + queue backend. Replaces the
 *     previous gpu_queues_ vector on clio::run::IpcManager.
 *   - `client_backends_`: map from AllocatorId to a registered
 *     client-side device-memory backend. Populated by the admin
 *     RegisterMemory handler. The CPU GPU worker uses this map to
 *     resolve ShmPtrs popped off gpu2cpu_queue into host-readable
 *     pointers (direct read for kPinnedHost; cudaMemcpy for kDeviceMem
 *     and kManagedUvm).
 *
 * Device-side (GPU) state held here:
 *   - `gpu_info_`: copy of the IpcManagerGpuInfo passed by value into
 *     the kernel via CLIO_GPU_INIT.
 */
class IpcManager {
 public:
  // ================================================================
  // Device-side GPU thread topology helpers
  // ================================================================

#if CTP_IS_GPU_COMPILER
  static CTP_GPU_FUN inline int GetGpuThreadId() {
    return threadIdx.x + threadIdx.y * blockDim.x +
           threadIdx.z * blockDim.x * blockDim.y;
  }
  static CTP_GPU_FUN inline int GetGpuNumThreads() {
    return blockDim.x * blockDim.y * blockDim.z;
  }
#elif CTP_IS_SYCL_COMPILER
  /** SYCL: producer kernels run as q.single_task — topology is constant. */
  static inline int GetGpuThreadId() { return 0; }
  static inline int GetGpuNumThreads() { return 1; }
#endif

  // ================================================================
  // Device-side init
  // ================================================================

  /**
   * Initialize the per-block IpcManager from the host-supplied gpu_info.
   * Called by the CLIO_GPU_INIT macro at kernel entry.
   */
  CTP_GPU_FUN void ClientInitGpu(const IpcManagerGpuInfo &gpu_info) {
    gpu_info_ = gpu_info;
  }

  // ================================================================
  // Device-side Send
  // ================================================================

  /**
   * Push a pre-allocated task onto the gpu2cpu_queue.
   *
   * The task and its co-located gpu::FutureShm must already live in a
   * registered device-memory backend (admin RegisterMemory). Only one
   * thread per CUDA block actually performs the enqueue; other threads
   * receive an empty future (mirrors today's threadIdx==0 guard).
   */
  template <typename TaskT>
  CTP_CROSS_FUN gpu::Future<TaskT> Send(
      const ctp::ipc::FullPtr<TaskT> &task_ptr) {
#if CTP_IS_GPU || CTP_IS_SYCL_DEVICE
    return IpcGpu2Cpu::ClientSend(this, task_ptr);
#else
    (void)task_ptr;
    return gpu::Future<TaskT>();
#endif
  }

#if CTP_IS_GPU_COMPILER
  /**
   * CUDA/ROCm only: per-block IpcManager lives in __shared__ storage so
   * helpers reachable from the kernel (e.g. IpcGpu2Cpu::ClientSend) can
   * resolve it via plain symbol lookup. SYCL achieves the same with a
   * kernel-scope local + `g_ipc_manager_ptr` name-lookup trick (see the
   * SYCL CLIO_GPU_INIT macro below).
   */
  static CTP_GPU_FUN __noinline__ IpcManager *GetBlockIpcManager() {
    __shared__ char s_ipc_bytes[sizeof(IpcManager)];
    return reinterpret_cast<IpcManager *>(s_ipc_bytes);
  }
#endif  // CTP_IS_GPU_COMPILER

  /** Kind of memory a client-registered backend lives in. Visible from
   *  both host and device passes so clio::run::IpcManager helper signatures can
   *  reference it without CTP_IS_HOST gating. */
  enum class MemKind : unsigned char {
    kPinnedHost = 0,    ///< cudaHostAlloc / hipHostMalloc / sycl::malloc_host
    kManagedUvm = 1,    ///< cudaMallocManaged / hipMallocManaged / sycl::malloc_shared
    kDeviceMem  = 2,    ///< cudaMalloc / hipMalloc / sycl::malloc_device
  };

  // ================================================================
  // Fields visible to device + host
  // ================================================================

  IpcManagerGpuInfo gpu_info_;

  // ================================================================
  // Host-only: per-device queues + client backend registration
  // ================================================================

#if CTP_IS_HOST

  /** Registration record for a client-owned device-memory backend. */
  struct ClientBackend {
    ctp::ipc::AllocatorId alloc_id;
    char *host_view = nullptr;  ///< CPU-readable pointer (pinned/UVM only)
    char *device_ptr = nullptr; ///< Raw device pointer (for kDeviceMem D2H copy)
    size_t capacity = 0;
    clio::run::u32 gpu_id = 0;
    MemKind kind = MemKind::kPinnedHost;
  };

  /** Per-physical-GPU state: queue + queue backend + client backends. */
  struct PerGpuDeviceState {
    /** Pinned host backend holding the GpuTaskQueue. */
    char *queue_backend = nullptr;
    size_t queue_backend_size = 0;
    /** The actual GpuTaskQueue object, constructed inside queue_backend. */
    ctp::ipc::FullPtr<clio::run::GpuTaskQueue> gpu2cpu_queue;
    /** AllocatorId → registered client backend. */
    std::unordered_map<u64, ClientBackend> client_backends;
    u32 gpu_id = 0;
  };

  std::vector<PerGpuDeviceState> per_gpu_devices_;

  /**
   * Initialize per-device gpu2cpu queues. Implemented in
   * src/gpu/gpu2cpu_init_hip.cc (CUDA/ROCm) or
   * src/gpu/gpu2cpu_init_sycl.cc (SYCL).
   *
   * @param queue_depth Ring-buffer depth per lane.
   * @return true on success.
   */
  CLIO_RUN_GPU_API bool ServerInitGpuQueues(u32 queue_depth);

  /** Free per-device queues. */
  CLIO_RUN_GPU_API void FinalizeGpuQueues();

  /** Number of registered GPU devices. */
  size_t GetGpuQueueCount() const { return per_gpu_devices_.size(); }

  /** Get gpu2cpu queue for a given device (CPU worker polls this). */
  GpuTaskQueue *GetGpuQueue(u32 gpu_id) const {
    if (gpu_id >= per_gpu_devices_.size()) return nullptr;
    return per_gpu_devices_[gpu_id].gpu2cpu_queue.ptr_;
  }

  /** Build the kernel-facing IpcManagerGpuInfo for a given device. */
  IpcManagerGpuInfo GetGpuInfo(u32 gpu_id) const {
    IpcManagerGpuInfo info;
    if (gpu_id >= per_gpu_devices_.size()) return info;
    info.gpu2cpu_queue = per_gpu_devices_[gpu_id].gpu2cpu_queue.ptr_;
    info.gpu_id = gpu_id;
    return info;
  }

  /**
   * Register a client-owned device-memory backend. Called by the admin
   * RegisterMemory handler on the runtime side; the client side wraps
   * this in IpcManager::AllocateAndRegisterGpuBackend (host helper).
   */
  CLIO_RUN_GPU_API bool RegisterClientBackend(const ClientBackend &b);

  /** Unregister a client backend. */
  CLIO_RUN_GPU_API void UnregisterClientBackend(
      u32 gpu_id, const ctp::ipc::AllocatorId &alloc_id);

  /**
   * Resolve an AllocatorId to its registered ClientBackend record.
   * Used by IpcManager::ToFullPtr when a CPU SHM allocator lookup
   * misses, so this must be inline (header-only) — callers like
   * clio_commands link without libclio_run_cxx_gpu.
   * Returns nullptr if unknown.
   */
  inline const ClientBackend *FindClientBackend(
      u32 gpu_id, const ctp::ipc::AllocatorId &alloc_id) const {
    if (gpu_id >= per_gpu_devices_.size()) return nullptr;
    u64 key = (static_cast<u64>(alloc_id.major_) << 32) |
              static_cast<u64>(alloc_id.minor_);
    const auto &dev = per_gpu_devices_[gpu_id];
    auto it = dev.client_backends.find(key);
    if (it == dev.client_backends.end()) return nullptr;
    return &it->second;
  }

#endif  // CTP_IS_HOST

  IpcManager() = default;
  ~IpcManager() = default;
};

}  // namespace gpu
}  // namespace clio::run

// gpu::Future::Wait + IpcGpu2Cpu::ClientSend (both reach into gpu_info_).
#include "clio_runtime/ipc/ipc_gpu2cpu_impl.h"

// ================================================================
// Single CLIO_GPU_INIT macro (replaces the 5 legacy variants)
// ================================================================
//
// CUDA/ROCm: per-block IpcManager lives in __shared__ storage so any
// CTP_GPU_FUN helper reachable from the kernel can look it up via
// CLIO_IPC = GetBlockIpcManager(). SYCL: kernel functor captures a
// pointer to a host-allocated USM IpcManager; the macro names that
// pointer `g_ipc_manager_ptr` so the SYCL CLIO_IPC macro (in ipc_manager.h)
// can resolve it via plain C++ name lookup.

#if CTP_IS_SYCL_COMPILER

#define CLIO_GPU_INIT(gpu_info, ipc_ptr)                                  \
  clio::run::gpu::IpcManager *g_ipc_manager_ptr = (ipc_ptr);                        \
  g_ipc_manager_ptr->ClientInitGpu(gpu_info);                                 \
  clio::run::gpu::IpcManager &g_ipc_manager = *g_ipc_manager_ptr

#else  // CUDA / ROCm

#define CLIO_GPU_INIT(gpu_info, ipc_ptr)                                  \
  (void)(ipc_ptr);                                                            \
  clio::run::gpu::IpcManager *g_ipc_manager_ptr =                                   \
      clio::run::gpu::IpcManager::GetBlockIpcManager();                             \
  if (clio::run::gpu::IpcManager::GetGpuThreadId() == 0) {                          \
    g_ipc_manager_ptr->ClientInitGpu(gpu_info);                               \
  }                                                                           \
  __syncthreads();                                                            \
  clio::run::gpu::IpcManager &g_ipc_manager = *g_ipc_manager_ptr

#endif  // CTP_IS_SYCL_COMPILER

#endif  // CTP_ENABLE_GPU
#endif  // CLIO_RUNTIME_INCLUDE_GPU_IPC_MANAGER_H_
