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
#include "clio_runtime/gpu/gpu_device_ring.h"
// blockIdx, for the per-block GetBlockIpcManager under SYCL. Inert on CUDA.
#include "clio_ctp/util/sycl_cuda_compat.h"
#include "clio_runtime/gpu/future.h"
#include "clio_runtime/ipc/ipc_gpu2cpu.h"

#if CTP_ENABLE_GPU

#include <memory>
#include <unordered_map>
#include <vector>

namespace clio::run {
namespace gpu {

#if CTP_ENABLE_SYCL
/** Base of the per-block IpcManager array; see GetBlockIpcManager below.
 *  Installed by the host with SyclInitBlockIpcManagers.
 *
 *  Defined only by the TU that owns the kernels -- see the same rule, and
 *  the DPC++ double-registration abort that motivates it, in yield_stack.h. */
#if defined(CLIO_SYCL_KERNEL_TU)
// device_image_scope for the same reason as the yield globals; see
// yield_stack.h.
inline ::sycl::ext::oneapi::experimental::device_global<
    char *, decltype(::sycl::ext::oneapi::experimental::properties(
                ::sycl::ext::oneapi::experimental::device_image_scope))>
    g_sycl_block_ipc;
inline char *SyclBlockIpcBase() { return g_sycl_block_ipc.get(); }
#else
inline char *SyclBlockIpcBase() { return nullptr; }
#endif
#endif

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
  template <bool Probing = true, typename TaskT>
  CTP_CROSS_FUN gpu::Future<TaskT> Send(
      const ctp::ipc::FullPtr<TaskT> &task_ptr) {
#if CTP_IS_GPU || CTP_IS_SYCL_DEVICE
    return IpcGpu2Cpu::SendIn<Probing>(this, task_ptr);
#else
    (void)task_ptr;
    return gpu::Future<TaskT>();
#endif
  }

#if CTP_IS_GPU_COMPILER
  /**
   * CUDA/ROCm only: per-block IpcManager lives in __shared__ storage so
   * helpers reachable from the kernel (e.g. IpcGpu2Cpu::SendIn) can
   * resolve it via plain symbol lookup. SYCL achieves the same with a
   * kernel-scope local + `g_ipc_manager_ptr` name-lookup trick (see the
   * SYCL CLIO_GPU_INIT macro below).
   */
  static CTP_GPU_FUN __noinline__ IpcManager *GetBlockIpcManager() {
    __shared__ char s_ipc_bytes[sizeof(IpcManager)];
    return reinterpret_cast<IpcManager *>(s_ipc_bytes);
  }
#elif CTP_ENABLE_SYCL
  /**
   * SYCL: the same PER-BLOCK IpcManager, in global memory.
   *
   * Callers reachable from a kernel (IpcGpu2Cpu::SendIn, and every submit
   * site in device_vector.h) resolve it by plain symbol lookup with no
   * parameter, which is the whole reason the CUDA version is `__shared__`
   * rather than a kernel argument. SYCL has no `__shared__` a free function
   * can name, so the array lives in USM and the host installs its base --
   * exactly the arrangement YieldTls uses for YieldSmem.
   *
   * ONE PER BLOCK, not one global: `probe_slot_` is per-block mutable state
   * that SendIn stamps for the submission in flight. Sharing a single record
   * across the grid would let blocks overwrite each other's probe slot.
   */
  static IpcManager *GetBlockIpcManager() {
    return reinterpret_cast<IpcManager *>(SyclBlockIpcBase()) + blockIdx.x;
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

  /** Submit-probe record the caller claimed for the Send now in flight.
   *  The IpcManager is __shared__ (per CUDA block) and only thread 0 submits, so
   *  the producer claims a slot, parks it here, and SendIn stamps into it — no
   *  signature change to Send()/SendIn(). Meaningless when the probe is off. */
  u32 probe_slot_ = gpu::kProbeNoSlot;

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

  /**
   * Host-side mirror of a device-memory submission ring.
   *
   * The ring lives on the GPU, so the CPU cannot Pop() from it -- it copies.
   * One D2H copy brings back a whole span of submissions, which is the entire
   * point: per-request bus crossings become per-BATCH crossings. Drained
   * entries are held here and handed out one at a time, so the worker keeps
   * its existing one-task-per-poll pacing (batching the WORKER's dequeue is
   * what deadlocked in d265bdb3 -- this batches the TRANSPORT only).
   */
  struct GpuRingMirror {
    clio::run::GpuDeviceRing *dev_ring = nullptr;  ///< device address of the ring
    void *stream = nullptr;                    ///< dedicated copy stream
    /** Slots consumed so far; the device sees this via a published tail_. */
    unsigned long long tail = 0;
    /** Entries drained but not yet handed to a worker. */
    std::vector<clio::run::GpuRingEntry> pending;
    size_t pending_pos = 0;                    ///< read cursor into `pending`
    /**
     * The ring payload, in PINNED HOST memory. The host reads these directly;
     * the device writes them through the mapped device pointers stored in the
     * ring. No staging and no copy is involved on the read path any more.
     */
    clio::run::GpuRingEntry *host_entries = nullptr;
    unsigned int *host_ready = nullptr;
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
    /** Device-ring state; dev_ring stays null when the legacy queue is used. */
    GpuRingMirror ring;
  };

  /**
   * Drain up to `max_out` submissions from the device ring into the mirror's
   * pending list, and return the next one. Returns false when nothing is
   * ready. Host-only; implemented in gpu2cpu_init_hip.cc beside the ring's
   * allocation so the copy logic sits next to the layout it depends on.
   */
  CLIO_RUN_GPU_API bool RingNext(u32 gpu_id, clio::run::GpuRingEntry *out);

  /**
   * The dedicated copy stream for a device's ring, or null.
   *
   * SendOut publishes completions on it: issuing the POD writeback and then
   * the completion flag as two async copies on ONE stream keeps
   * payload-before-flag ordering (the invariant the kernel's release signal
   * depends on) without the caller blocking on either.
   */
  void *GetRingStream(u32 gpu_id) const {
    if (gpu_id >= per_gpu_devices_.size()) return nullptr;
    return per_gpu_devices_[gpu_id].ring.stream;
  }

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
    info.gpu2cpu_ring = per_gpu_devices_[gpu_id].ring.dev_ring;
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

// gpu::Future::Wait + IpcGpu2Cpu::SendIn (both reach into gpu_info_).
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

/**
 * Allocate and publish the per-block IpcManager array. Call ONCE on the host
 * before launching any kernel that submits tasks, with the widest grid those
 * kernels will use.
 *
 * gpu_info is stamped here rather than at kernel entry: it is the same value
 * for every block and every launch, so writing it once on the host removes a
 * store and a barrier from the kernel prologue. The CUDA path cannot do this
 * -- its IpcManager is __shared__ and therefore born fresh, and uninitialized,
 * at every launch.
 */
namespace clio::run::gpu {
inline void SyclInitBlockIpcManagers(clio::run::u32 nblocks,
                                     const clio::run::IpcManagerGpuInfo &gpu_info) {
  auto &q = ctp::GpuApi::SyclQueue();
  const size_t bytes = static_cast<size_t>(nblocks) * sizeof(IpcManager);
  char *base = static_cast<char *>(sycl::malloc_device(bytes, q));
  q.memset(base, 0, bytes).wait();
  clio::run::IpcManagerGpuInfo info = gpu_info;
  q.parallel_for(sycl::range<1>(nblocks), [=](sycl::id<1> i) {
     reinterpret_cast<IpcManager *>(base)[i[0]].gpu_info_ = info;
     reinterpret_cast<IpcManager *>(base)[i[0]].probe_slot_ = kProbeNoSlot;
   }).wait();
#if defined(CLIO_SYCL_KERNEL_TU)
  q.copy(&base, g_sycl_block_ipc, 1).wait();
#else
  // Unreachable in practice: this function submits kernels, so only a
  // -fsycl TU can call it, and the one that does owns the device global.
  (void)base;
#endif
}
}  // namespace clio::run::gpu

/**
 * SYCL: nothing to do at kernel entry -- SyclInitBlockIpcManagers already
 * stamped gpu_info into every block's record on the host, and
 * GetBlockIpcManager finds this block's by symbol lookup. The names are kept
 * so a kernel body reads the same as its CUDA twin.
 */
#define CLIO_GPU_INIT(gpu_info, ipc_ptr)                                      \
  (void)(gpu_info);                                                           \
  (void)(ipc_ptr);                                                            \
  clio::run::gpu::IpcManager *g_ipc_manager_ptr =                             \
      clio::run::gpu::IpcManager::GetBlockIpcManager();                       \
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
