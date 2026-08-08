/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Slim CUDA / ROCm gpu2cpu init.
 *
 * Producer-only design: per detected GPU, allocate one pinned-host backend
 * holding a clio::run::GpuTaskQueue. The CPU GPU worker polls each queue;
 * kernels push admin-registered task allocations onto them via
 * IpcGpu2Cpu::SendIn. There is no longer a per-device "copy_backend"
 * — clients allocate their own task and data backends on the host and
 * register them through admin RegisterMemory.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/gpu/gpu_ipc_manager.h"
#include "clio_runtime/gpu/gpu_device_ring.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/singletons.h"
#include "clio_ctp/util/gpu_api.h"
#include "clio_ctp/util/logging.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <new>

namespace clio::run {

// Note: queue construction happens host-side (see ServerInitGpuQueues).
// We previously launched a single-thread kernel to construct the
// BuddyAllocator + GpuTaskQueue in device memory, but the kernel had no
// host/device asymmetry that required device-side construction (the
// queue_backend is MANAGED memory, addressable 1:1 from host and device
// space) and the cross-shared-library kernel registration was unreliable
// under HIP-NVCC ("cudaErrorInvalidDeviceFunction" on launch).

#if CTP_IS_HOST

bool gpu::IpcManager::ServerInitGpuQueues(u32 queue_depth) {
  if (!per_gpu_devices_.empty()) {
    return true;  // already initialized
  }

  int device_count = static_cast<int>(ctp::GpuApi::GetDeviceCount());
  if (device_count <= 0) {
    // CPU-only deployment: leave per_gpu_devices_ empty. Consumers
    // (GetGpuInfo, GetGpuQueue, RegisterClientBackend) all bounds-check
    // against per_gpu_devices_.size() and gracefully return null for
    // unknown gpu_ids, so a runtime without GPUs operates correctly —
    // it just never services GPU→CPU tasks. Returning true here lets
    // CLIO_INIT complete on hosts and CI containers without a
    // visible CUDA device.
    HLOG(kInfo, "ServerInitGpuQueues: no GPU devices detected — "
         "GPU queues will not be initialized (CPU-only mode)");
    return true;
  }
  per_gpu_devices_.resize(device_count);

  constexpr size_t kQueueBackendBytes = 16 * 1024 * 1024;

  for (int gpu_id = 0; gpu_id < device_count; ++gpu_id) {
    PerGpuDeviceState &dev = per_gpu_devices_[gpu_id];
    dev.gpu_id = static_cast<u32>(gpu_id);

    ctp::GpuApi::SetDevice(gpu_id);

    // Pre-create the device-I/O stream pool HERE, while no user kernel can be
    // resident. Creating a stream later, from the bdev's GPU read/write path,
    // takes the CUDA context write lock and blocks until the device is free --
    // and the kernels that path serves are demand-paged ones spinning until
    // that very I/O completes, so the two deadlock. Measured with a paged GPU
    // vector: every compute worker parked in pthread_rwlock_wrlock under
    // cuStreamCreate/cuStreamDestroy and the runtime reported "ALL compute
    // workers stalled". See ctp::GpuApi::BorrowStream.
    constexpr int kIoStreamPoolSize = 64;
    ctp::GpuApi::WarmStreamPool(kIoStreamPoolSize);

    // Device-memory submission ring (CLIO_GPU_DEVRING=1).
    //
    // Its copy stream is created HERE, at init, and is NEVER the bdev I/O
    // pool's. Two reasons, both already paid for once: creating a stream while
    // a kernel is resident blocks until that kernel finishes (see
    // GpuApi::BorrowStream), and a queue copy queued behind a multi-megabyte
    // data copy would add that copy's latency to every task on the GPU.
    if (const char *dr = std::getenv("CLIO_GPU_DEVRING")) {
      if (*dr != '\0' && std::string(dr) != "0" && std::string(dr) != "false") {
        void *ring_mem = nullptr;
#if CTP_ENABLE_CUDA
        if (cudaMalloc(&ring_mem, sizeof(clio::run::GpuDeviceRing)) !=
            cudaSuccess) {
          ring_mem = nullptr;
        }
#endif
        if (ring_mem == nullptr) {
          HLOG(kError, "ServerInitGpuQueues: device ring alloc failed "
               "(gpu_id={})", gpu_id);
          FinalizeGpuQueues();
          return false;
        }
        // Construct on the host, then upload once: head_/tail_ zeroed and all
        // ready stamps cleared, which is the state the protocol assumes.
        {
          auto *init = new clio::run::GpuDeviceRing();
#if CTP_ENABLE_CUDA
          cudaMemcpy(ring_mem, init, sizeof(*init), cudaMemcpyHostToDevice);
#endif
          delete init;
        }
        dev.ring.dev_ring = static_cast<clio::run::GpuDeviceRing *>(ring_mem);
        dev.ring.stream = ctp::GpuApi::CreateStream();
        dev.ring.tail = 0;
#if CTP_ENABLE_CUDA
        cudaHostAlloc(&dev.ring.stage_entries,
                      clio::run::kGpuRingCapacity *
                          sizeof(clio::run::GpuRingEntry), 0);
        cudaHostAlloc(&dev.ring.stage_ready,
                      clio::run::kGpuRingCapacity * sizeof(unsigned int), 0);
#endif
        HLOG(kInfo, "ServerInitGpuQueues: gpu_id={} DEVICE ring at {} "
             "(capacity {})", gpu_id, ring_mem,
             clio::run::kGpuRingCapacity);
      }
    }

    // MANAGED, not pinned host memory. The device pushes to this queue with a
    // SYSTEM-SCOPED atomic (the ring head), and this GPU reports
    // cudaDevAttrHostNativeAtomicSupported = 0 -- it cannot perform atomics on
    // host memory at all, so that atomic was invalid on every single device
    // task submission: every page fault, every flush, every prefetch.
    // compute-sanitizer flagged it in EVERY gpu_vector binary, including tests
    // that pass, which were relying on undefined behaviour that usually
    // happened to work. No cudaHostAlloc flag can fix that; the memory simply
    // must not be host memory. Managed memory does support system-wide atomics
    // here (cudaDevAttrConcurrentManagedAccess = 1) and keeps the single
    // address space the queue's construction relies on.
    dev.queue_backend = ctp::GpuApi::MallocManaged<char>(kQueueBackendBytes);
    if (!dev.queue_backend) {
      HLOG(kError, "ServerInitGpuQueues: MallocManaged for queue backend "
           "failed (gpu_id={})", gpu_id);
      FinalizeGpuQueues();
      return false;
    }
    dev.queue_backend_size = kQueueBackendBytes;
    std::memset(dev.queue_backend, 0, kQueueBackendBytes);

    // Host-side construction. queue_backend is managed memory mapped
    // into device address space at the same virtual address, so the
    // BuddyAllocator's internal offset-based bookkeeping is safe to set
    // up from the host. We previously constructed inside a single-thread
    // CUDA kernel; under HIP-NVCC that path hit "invalid device function"
    // intermittently, and the kernel had no host/device asymmetry that
    // required device-side construction in the first place.
    size_t queue_off = static_cast<size_t>(-1);
    {
      ctp::ipc::MemoryBackend proxy;
      proxy.data_ = dev.queue_backend;
      proxy.data_capacity_ = kQueueBackendBytes;
      CLIO_QUEUE_ALLOC_T *alloc = proxy.MakeAlloc<CLIO_QUEUE_ALLOC_T>();
      if (alloc) {
        ctp::ipc::FullPtr<clio::run::GpuTaskQueue> queue =
            alloc->NewObj<clio::run::GpuTaskQueue>(
                alloc, /*num_lanes=*/1u, /*num_prio=*/2u, queue_depth);
        if (!queue.IsNull()) {
          queue_off = queue.shm_.off_.load();
        }
      }
    }
    if (queue_off == static_cast<size_t>(-1)) {
      HLOG(kError, "ServerInitGpuQueues: queue construction failed "
           "(gpu_id={})", gpu_id);
      FinalizeGpuQueues();
      return false;
    }
    dev.gpu2cpu_queue.shm_.off_ = queue_off;
    dev.gpu2cpu_queue.shm_.alloc_id_ = ctp::ipc::AllocatorId{0, 0};
    dev.gpu2cpu_queue.ptr_ = reinterpret_cast<clio::run::GpuTaskQueue *>(
        dev.queue_backend + queue_off);

    HLOG(kInfo, "ServerInitGpuQueues: gpu_id={} queue at {} ({}MB)",
         gpu_id, static_cast<void *>(dev.gpu2cpu_queue.ptr_),
         kQueueBackendBytes / (1024 * 1024));
  }

  // Device-aware memcpy / device-pointer detection are now plain header
  // functions (ctp::DeviceAwareMemcpy / ctp::IsDevicePointer in gpu_api.h),
  // compiled directly into each caller — no runtime hook to install here.
  return true;
}

/**
 * Hand back the next device-ring submission, refilling from the GPU in a
 * BATCH when the local buffer runs dry.
 *
 * One D2H copy brings back every submission that arrived since the last
 * refill; the worker then consumes them one per poll. That split is
 * deliberate: batching the transport is the win, while batching the WORKER's
 * dequeue is what deadlocked the runtime in d265bdb3 (one lane serves the
 * whole GPU, so a batch piles onto a single worker and every compute worker
 * ends up parked on a device task).
 */
bool gpu::IpcManager::RingNext(u32 gpu_id, clio::run::GpuRingEntry *out) {
#if CTP_ENABLE_CUDA
  if (gpu_id >= per_gpu_devices_.size()) return false;
  auto &m = per_gpu_devices_[gpu_id].ring;
  if (m.dev_ring == nullptr) return false;

  // Serve from the already-drained batch first.
  if (m.pending_pos < m.pending.size()) {
    *out = m.pending[m.pending_pos++];
    return true;
  }
  m.pending.clear();
  m.pending_pos = 0;

  auto *stream = static_cast<cudaStream_t>(m.stream);

  // 1. How far has the producer claimed?
  unsigned long long head = 0;
  cudaMemcpyAsync(&head, &m.dev_ring->head_, sizeof(head),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  if (head <= m.tail) return false;

  unsigned long long n = head - m.tail;
  if (n > clio::run::kGpuRingCapacity) n = clio::run::kGpuRingCapacity;
  const u32 begin = static_cast<u32>(m.tail) & clio::run::kGpuRingMask;

  // 2. Copy the span. It wraps at most once, so at most two copies -- still
  //    O(1) bus crossings for O(n) submissions, which is the whole point.
  auto *ents = static_cast<clio::run::GpuRingEntry *>(m.stage_entries);
  auto *rdy = static_cast<unsigned int *>(m.stage_ready);
  const u32 first = std::min<u32>(static_cast<u32>(n),
                                  clio::run::kGpuRingCapacity - begin);
  const u32 second = static_cast<u32>(n) - first;
  cudaMemcpyAsync(ents, &m.dev_ring->entries_[begin],
                  first * sizeof(clio::run::GpuRingEntry),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(rdy, &m.dev_ring->ready_[begin],
                  first * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
  if (second) {
    cudaMemcpyAsync(ents + first, &m.dev_ring->entries_[0],
                    second * sizeof(clio::run::GpuRingEntry),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(rdy + first, &m.dev_ring->ready_[0],
                    second * sizeof(unsigned int), cudaMemcpyDeviceToHost,
                    stream);
  }
  cudaStreamSynchronize(stream);

  // 3. Accept the contiguous READY prefix only. A claimed-but-unwritten slot
  //    stops the batch; it will be there next poll. The stamp must match the
  //    slot's generation, so a wrapped slot cannot masquerade as a fresh one.
  u32 accepted = 0;
  for (u32 i = 0; i < n; ++i) {
    const unsigned long long slot = m.tail + i;
    const unsigned int want =
        static_cast<unsigned int>(slot / clio::run::kGpuRingCapacity) + 1u;
    if (rdy[i] != want) break;
    m.pending.push_back(ents[i]);
    ++accepted;
  }
  if (accepted == 0) return false;

  // 4. Publish the new tail so producers blocked on a full ring advance.
  m.tail += accepted;
  cudaMemcpyAsync(&m.dev_ring->tail_, &m.tail, sizeof(m.tail),
                  cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  *out = m.pending[m.pending_pos++];
  return true;
#else
  (void)gpu_id; (void)out;
  return false;
#endif
}

void gpu::IpcManager::FinalizeGpuQueues() {
  for (auto &dev : per_gpu_devices_) {
    if (dev.queue_backend) {
      ctp::GpuApi::Free(dev.queue_backend);
      dev.queue_backend = nullptr;
    }
    if (dev.ring.dev_ring) {
      ctp::GpuApi::Free(reinterpret_cast<char *>(dev.ring.dev_ring));
      dev.ring.dev_ring = nullptr;
    }
#if CTP_ENABLE_CUDA
    if (dev.ring.stage_entries) {
      cudaFreeHost(dev.ring.stage_entries);
      dev.ring.stage_entries = nullptr;
    }
    if (dev.ring.stage_ready) {
      cudaFreeHost(dev.ring.stage_ready);
      dev.ring.stage_ready = nullptr;
    }
#endif
    dev.gpu2cpu_queue = ctp::ipc::FullPtr<clio::run::GpuTaskQueue>::GetNull();
    dev.client_backends.clear();
  }
  per_gpu_devices_.clear();
}

bool gpu::IpcManager::RegisterClientBackend(const ClientBackend &b) {
  if (b.gpu_id >= per_gpu_devices_.size()) return false;
  u64 key = (static_cast<u64>(b.alloc_id.major_) << 32) |
            static_cast<u64>(b.alloc_id.minor_);
  auto &dev = per_gpu_devices_[b.gpu_id];
  dev.client_backends[key] = b;
  return true;
}

void gpu::IpcManager::UnregisterClientBackend(
    u32 gpu_id, const ctp::ipc::AllocatorId &alloc_id) {
  if (gpu_id >= per_gpu_devices_.size()) return;
  u64 key = (static_cast<u64>(alloc_id.major_) << 32) |
            static_cast<u64>(alloc_id.minor_);
  per_gpu_devices_[gpu_id].client_backends.erase(key);
}

// FindClientBackend is now inline in gpu_ipc_manager.h so it's
// available without linking libclio_run_cxx_gpu (used by ToFullPtr).

CLIO_RUN_GPU_API bool ChiServerBootstrapHipGpu(IpcManager *self,
                                               clio::run::u32 queue_depth,
                                               std::size_t backend_bytes) {
  (void)backend_bytes;  // No host-managed copy_backend in producer-only model.
  if (!self) return false;
  if (!self->gpu_ipc_) {
    self->gpu_ipc_ = std::make_unique<gpu::IpcManager>();
  }
  return self->gpu_ipc_->ServerInitGpuQueues(queue_depth);
}

#endif  // CTP_IS_HOST

}  // namespace clio::run

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
