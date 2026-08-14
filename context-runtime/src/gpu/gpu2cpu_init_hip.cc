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
#include <atomic>
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
    // Sized by env so the exhaustion deadlock can be measured against pool
    // size. 64 was measured EXHAUSTED (outstanding=64 of warmed=64) at the
    // moment of a wedge: every stream in flight, three worker threads spinning
    // in BorrowStream's unbounded retry, lanes undrained (one worker held 45
    // queued tasks), and the faulting kernel waiting on those very tasks.
    int io_pool = 64;
    if (const char *e = std::getenv("CLIO_GPU_STREAM_POOL")) {
      const int v = std::atoi(e);
      if (v > 0) io_pool = v;
    }
    ctp::GpuApi::WarmStreamPool(io_pool);
    HLOG(kInfo, "GPU I/O stream pool: {} streams (CLIO_GPU_STREAM_POOL)",
         io_pool);

    // Device-memory submission ring (CLIO_GPU_DEVRING=1).
    //
    // Its copy stream is created HERE, at init, and is NEVER the bdev I/O
    // pool's. Two reasons, both already paid for once: creating a stream while
    // a kernel is resident blocks until that kernel finishes (see
    // GpuApi::BorrowStream), and a queue copy queued behind a multi-megabyte
    // data copy would add that copy's latency to every task on the GPU.
    // ON BY DEFAULT. It cleared the gate the design set: 23/23 tests,
    // compute-sanitizer 0 errors, correct through 64 blocks and
    // multi-oversub-64, and faster on every workload measured (sync demand
    // faults 493 -> 801 MB/s). CLIO_GPU_DEVRING=0 falls back to the managed
    // queue for bisecting a regression to this transport.
    {
      const char *dr = std::getenv("CLIO_GPU_DEVRING");
      const bool use_ring =
          (dr == nullptr) || (*dr != '\0' && std::string(dr) != "0" &&
                              std::string(dr) != "false");
      if (use_ring) {
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
        // The PAYLOAD lives in mapped, pinned host memory so the consumer can
        // read submissions with plain loads. cudaHostAllocMapped is what makes
        // the same bytes addressable from a kernel; cudaHostGetDevicePointer
        // yields the address the device must use.
#if CTP_ENABLE_CUDA
        void *h_ents = nullptr, *h_rdy = nullptr;
        void *d_ents = nullptr, *d_rdy = nullptr;
        bool host_ok =
            cudaHostAlloc(&h_ents,
                          clio::run::kGpuRingCapacity *
                              sizeof(clio::run::GpuRingEntry),
                          cudaHostAllocMapped) == cudaSuccess &&
            cudaHostAlloc(&h_rdy,
                          clio::run::kGpuRingCapacity * sizeof(unsigned int),
                          cudaHostAllocMapped) == cudaSuccess;
        if (host_ok) {
          // Stamps must read "not ready" before any producer runs, or the
          // consumer would accept whatever the allocation happened to contain.
          std::memset(h_rdy, 0,
                      clio::run::kGpuRingCapacity * sizeof(unsigned int));
          std::memset(h_ents, 0,
                      clio::run::kGpuRingCapacity *
                          sizeof(clio::run::GpuRingEntry));
          host_ok = cudaHostGetDevicePointer(&d_ents, h_ents, 0) == cudaSuccess &&
                    cudaHostGetDevicePointer(&d_rdy, h_rdy, 0) == cudaSuccess;
        }
        if (!host_ok) {
          HLOG(kError, "ServerInitGpuQueues: mapped host ring alloc failed "
               "(gpu_id={})", gpu_id);
          FinalizeGpuQueues();
          return false;
        }
        dev.ring.host_entries = static_cast<clio::run::GpuRingEntry *>(h_ents);
        dev.ring.host_ready = static_cast<unsigned int *>(h_rdy);
#endif
        // Construct on the host, then upload once: head_/tail_ zeroed and the
        // payload pointers set to the DEVICE-side addresses of the pinned
        // allocations above.
        {
          auto *init = new clio::run::GpuDeviceRing();
#if CTP_ENABLE_CUDA
          init->entries_ = static_cast<clio::run::GpuRingEntry *>(d_ents);
          init->ready_ = static_cast<unsigned int *>(d_rdy);
          cudaMemcpy(ring_mem, init, sizeof(*init), cudaMemcpyHostToDevice);
#endif
          delete init;
        }
        dev.ring.dev_ring = static_cast<clio::run::GpuDeviceRing *>(ring_mem);
        dev.ring.stream = ctp::GpuApi::CreateStream();
        dev.ring.tail = 0;
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
#if CTP_ENABLE_CUDA
    // Pin the managed queue's RESIDENCE to the host and give the GPU a fixed
    // remote mapping instead of migration rights. Concurrent CPU drain +
    // device atomics on migratable managed pages under a RESIDENT kernel
    // stalls UVM migrations, and migrations share the copy engines — one
    // stalled migration froze every CE transfer on the device (captured:
    // cuStreamSynchronize wedged, 0% MEM util, free SMs irrelevant). With
    // AccessedBy there is nothing to migrate, ever.
    {
      int dev_id = 0;
      cudaGetDevice(&dev_id);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
      // CUDA 13 retired the (advice, int device) overload: the location is now
      // a cudaMemLocation, so passing a device id straight through no longer
      // compiles. Same advice, spelled for the current toolkit.
      cudaMemLocation host_loc{};
      host_loc.type = cudaMemLocationTypeHost;
      host_loc.id = 0;
      cudaMemLocation dev_loc{};
      dev_loc.type = cudaMemLocationTypeDevice;
      dev_loc.id = dev_id;
      cudaMemAdvise(dev.queue_backend, kQueueBackendBytes,
                    cudaMemAdviseSetPreferredLocation, host_loc);
      cudaMemAdvise(dev.queue_backend, kQueueBackendBytes,
                    cudaMemAdviseSetAccessedBy, dev_loc);
#else
      cudaMemAdvise(dev.queue_backend, kQueueBackendBytes,
                    cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
      cudaMemAdvise(dev.queue_backend, kQueueBackendBytes,
                    cudaMemAdviseSetAccessedBy, dev_id);
#endif
    }
#endif

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
  if (m.dev_ring == nullptr || m.host_ready == nullptr) return false;

  // Serve from the already-drained batch first.
  if (m.pending_pos < m.pending.size()) {
    *out = m.pending[m.pending_pos++];
    return true;
  }
  m.pending.clear();
  m.pending_pos = 0;

  // THE STAMP IS THE ARRIVAL SIGNAL. head_ is never probed: it lives in device
  // memory and reading it cost a copy plus a synchronize on EVERY poll,
  // including the overwhelming majority that find nothing. A stamp carrying
  // the generation this slot is due is proof both that a producer claimed it
  // and that it finished writing, which is strictly more than head_ told us.
  //
  // Reading the stamp before the entry is still what makes this safe, and it
  // is now free: the producer's __threadfence_system() orders its entry write
  // ahead of its stamp write, so a stamp we observe as ready guarantees the
  // entry beside it is complete.
  auto *rdy = static_cast<volatile unsigned int *>(
      static_cast<void *>(m.host_ready));
  u32 accepted = 0;
  while (accepted < clio::run::kGpuRingCapacity) {
    const unsigned long long slot = m.tail + accepted;
    const u32 idx = static_cast<u32>(slot) & clio::run::kGpuRingMask;
    const unsigned int want =
        static_cast<unsigned int>(slot / clio::run::kGpuRingCapacity) + 1u;
    if (rdy[idx] != want) break;   // nothing there, or not finished yet
    // ACQUIRE between the stamp and the entry. The stamp is volatile but the
    // entry is not, and nothing here stopped the COMPILER from hoisting the
    // entry load above the stamp check -- accepting a stale entry whose stamp
    // lands just in time. A stale entry is the PREVIOUS generation's task
    // address: a completed task whose device POD carries the freed RunContext
    // pointer SendOut wrote back, which the staged copy then presents as live
    // ("SetAwaitedFshm: null RunContext" mid-execution under the
    // CLIO_FUSE_SCALE reproducer). The old device-memory ring had this
    // ordering physically, as two separate copy batches; the pinned rewrite
    // must state it.
    std::atomic_thread_fence(std::memory_order_acquire);
    GpuRingEntry e = m.host_entries[idx];
    // Seqlock-style recheck: if the stamp no longer matches, the entry bytes
    // we read may be torn; leave the slot for the next poll.
    std::atomic_thread_fence(std::memory_order_acquire);
    if (rdy[idx] != want) break;
    m.pending.push_back(e);
    ++accepted;
  }
  if (accepted == 0) return false;

  // Publish the new tail so producers blocked on a full ring advance. This is
  // the only CUDA call left on this path, and it is paid per BATCH of real
  // work rather than per poll.
  m.tail += accepted;
  auto *stream = static_cast<cudaStream_t>(m.stream);
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
    if (dev.ring.host_entries) {
      cudaFreeHost(dev.ring.host_entries);
      dev.ring.host_entries = nullptr;
    }
    if (dev.ring.host_ready) {
      cudaFreeHost(dev.ring.host_ready);
      dev.ring.host_ready = nullptr;
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


// CPU-launched copy KERNEL service (see mem_bdev_transport bounce): engine
// copies of device memory stall in channel order behind a resident kernel;
// kernels are SM-scheduled and run on free SMs. C linkage so the plain-C++
// transport TU can call it.
extern "C" void ctp_copy_kernel_launch(char *dst, const char *src, size_t n,
                                       void *stream) {
#if CTP_ENABLE_CUDA
  ctp::CtpCopyKernel<uint4><<<64, 256, 0, (cudaStream_t) stream>>>(dst, src,
                                                                   n);
#endif
}
