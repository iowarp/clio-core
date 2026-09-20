/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * SYCL gpu2cpu init.
 *
 * Producer-only design (mirror of gpu2cpu_init_hip.cc): per detected SYCL
 * GPU, allocate one pinned-host (sycl::malloc_host) backend holding a
 * GpuTaskQueue. Clients allocate their own task / data backends and
 * register them through admin RegisterMemory.
 */

#if CTP_ENABLE_SYCL && !(CTP_ENABLE_CUDA || CTP_ENABLE_ROCM)

#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/gpu/gpu_ipc_manager.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/singletons.h"
#include "clio_ctp/util/gpu_api.h"
#include "clio_ctp/util/logging.h"
#include "clio_runtime/gpu/gpu_device_ring.h"
#include <atomic>
#include <mutex>
#include <cstdlib>
#include <string>

#include <sycl/sycl.hpp>

#include <cstring>
#include <memory>
#include <new>

namespace clio::run {

#if CTP_IS_HOST

bool gpu::IpcManager::ServerInitGpuQueues(u32 queue_depth) {
  if (!per_gpu_devices_.empty()) return true;

  auto sycl_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (sycl_devices.empty()) {
    HLOG(kInfo, "ServerInitGpuQueues (SYCL): no GPU devices detected — "
         "GPU queues will not be initialized (CPU-only mode)");
    return true;
  }
  per_gpu_devices_.resize(sycl_devices.size());

  constexpr size_t kQueueBackendBytes = 16 * 1024 * 1024;
  auto &q = ctp::GpuApi::SyclQueue();

  for (size_t gpu_id = 0; gpu_id < sycl_devices.size(); ++gpu_id) {
    PerGpuDeviceState &dev = per_gpu_devices_[gpu_id];
    dev.gpu_id = static_cast<u32>(gpu_id);

    // THE DEVICE RING, ported from gpu2cpu_init_hip.cc. It was built for a
    // CUDA GPU reporting HostNativeAtomicSupported = 0, and Intel's Data
    // Center GPU Max 1550 is the same case: a device atomic to host USM faults
    // (benchmark/usm_atomic_probe.cc: usm_atomic_host_allocations=0,
    // AtomicAccessViolation on malloc_host). The ring splits by who needs
    // what -- head_/tail_ (the atomics) in DEVICE memory, entries_/ready_
    // (the payload) in HOST memory written by plain stores behind a system
    // fence and read by ordinary host loads. Nothing in it depends on a host
    // atomic. Without it the kernel takes the legacy GpuTaskQueue path and
    // pushes with device atomics into the 16 MB backend below, which is where
    // every paged benchmark on Aurora died: a fault at a 16 MB-aligned address
    // at PDE level, on the ring's first push (evlog: last 0 events).
    //
    // No cudaHostGetDevicePointer step: SYCL host USM is device-addressable at
    // the same virtual address, so the host pointers ARE the device pointers.
    // CLIO_GPU_DEVRING=0 falls back to the legacy queue, as on CUDA.
    {
      const char *dr = std::getenv("CLIO_GPU_DEVRING");
      const bool use_ring =
          (dr == nullptr) || (*dr != '\0' && std::string(dr) != "0" &&
                              std::string(dr) != "false");
      if (use_ring) {
        void *ring_mem = sycl::malloc_device(sizeof(clio::run::GpuDeviceRing), q);
        auto *h_ents = static_cast<clio::run::GpuRingEntry *>(sycl::malloc_host(
            clio::run::kGpuRingCapacity * sizeof(clio::run::GpuRingEntry), q));
        auto *h_rdy = static_cast<unsigned int *>(sycl::malloc_host(
            clio::run::kGpuRingCapacity * sizeof(unsigned int), q));
        if (ring_mem == nullptr || h_ents == nullptr || h_rdy == nullptr) {
          HLOG(kError, "ServerInitGpuQueues (SYCL): device ring alloc failed "
               "(gpu_id={})", gpu_id);
          if (ring_mem) sycl::free(ring_mem, q);
          if (h_ents) sycl::free(h_ents, q);
          if (h_rdy) sycl::free(h_rdy, q);
          FinalizeGpuQueues();
          return false;
        }
        // Stamps must read "not ready" before any producer runs, or the
        // consumer would accept whatever the allocation happened to contain.
        std::memset(h_rdy, 0, clio::run::kGpuRingCapacity * sizeof(unsigned int));
        std::memset(h_ents, 0,
                    clio::run::kGpuRingCapacity * sizeof(clio::run::GpuRingEntry));
        // Construct on the host, then upload once: head_/tail_ zeroed and the
        // payload pointers set to the (shared-VA) host allocations above.
        {
          clio::run::GpuDeviceRing init;
          init.entries_ = h_ents;
          init.ready_ = h_rdy;
          q.memcpy(ring_mem, &init, sizeof(init)).wait();
        }
        dev.ring.dev_ring = static_cast<clio::run::GpuDeviceRing *>(ring_mem);
        dev.ring.host_entries = h_ents;
        dev.ring.host_ready = h_rdy;
        dev.ring.stream = ctp::GpuApi::CreateStream();
        dev.ring.tail = 0;
        HLOG(kInfo, "ServerInitGpuQueues (SYCL): gpu_id={} DEVICE ring at {} "
             "(capacity {})", gpu_id, ring_mem, clio::run::kGpuRingCapacity);
      }
    }

    // The legacy queue backend. With the ring live the device never touches
    // it -- IpcGpu2Cpu::SendIn takes the ring path -- so it is plain host
    // memory, exactly as on CUDA (commit 6f70bf00: "the legacy queue must not
    // be managed memory when the ring is live"). Without the ring the device
    // pushes here with atomics, and on this GPU that is only legal on shared
    // USM (usm_atomic_probe.cc: malloc_shared takes the atomic, malloc_host
    // faults), so the fallback is malloc_shared rather than malloc_host.
    const bool ring_live = (dev.ring.dev_ring != nullptr);
    dev.queue_backend = static_cast<char *>(
        ring_live ? sycl::malloc_host(kQueueBackendBytes, q)
                  : sycl::malloc_shared(kQueueBackendBytes, q));
    dev.queue_backend_pinned = ring_live;
    if (!dev.queue_backend) {
      HLOG(kError, "ServerInitGpuQueues (SYCL): {} failed (gpu_id={})",
           ring_live ? "malloc_host" : "malloc_shared",
           gpu_id);
      FinalizeGpuQueues();
      return false;
    }
    dev.queue_backend_size = kQueueBackendBytes;
    std::memset(dev.queue_backend, 0, kQueueBackendBytes);

    // Host-side construction (mirrors gpu2cpu_init_hip.cc). queue_backend is
    // pinned host memory mapped into the device address space at the same
    // virtual address, so the BuddyAllocator's offset-based bookkeeping is
    // safe to set up from the host.
    //
    // This MUST NOT run in a kernel on Intel GPUs. BuddyAllocator::Allocate
    // takes ctp::Mutex, whose Lock() opens with lock_.fetch_add(1) -- and in
    // the SYCL device pass CTP_IS_GPU==0, so ctp::ipc::atomic resolves to
    // std_atomic (plain std::atomic). That is a device-side atomic RMW
    // against host USM, which Ponte Vecchio does not support at any memory
    // scope (aspect::usm_atomic_host_allocations == 0 on the Data Center GPU
    // Max 1550). The GPU page-faults with AtomicAccessViolation and the
    // Level Zero driver aborts the process, so every runtime start on Aurora
    // died here. Constructing on the host needs no device atomic at all.
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
      HLOG(kError, "ServerInitGpuQueues (SYCL): queue construction failed "
           "(gpu_id={})", gpu_id);
      FinalizeGpuQueues();
      return false;
    }
    dev.gpu2cpu_queue.shm_.off_ = queue_off;
    dev.gpu2cpu_queue.shm_.alloc_id_ = ctp::ipc::AllocatorId{0, 0};
    dev.gpu2cpu_queue.ptr_ = reinterpret_cast<clio::run::GpuTaskQueue *>(
        dev.queue_backend + queue_off);

    HLOG(kInfo, "ServerInitGpuQueues (SYCL): gpu_id={} queue at {} ({}MB)",
         gpu_id, static_cast<void *>(dev.gpu2cpu_queue.ptr_),
         kQueueBackendBytes / (1024 * 1024));
  }

  // Device-aware memcpy / device-pointer detection are now plain header
  // functions (ctp::DeviceAwareMemcpy / ctp::IsDevicePointer in gpu_api.h) —
  // no runtime hook to install here.
  return true;
}

void gpu::IpcManager::FinalizeGpuQueues() {
  if (per_gpu_devices_.empty()) return;
  auto &q = ctp::GpuApi::SyclQueue();
  for (auto &dev : per_gpu_devices_) {
    if (dev.queue_backend) {
      sycl::free(dev.queue_backend, q);   // host or shared USM: same free
      dev.queue_backend = nullptr;
    }
    if (dev.ring.dev_ring) {
      sycl::free(dev.ring.dev_ring, q);
      dev.ring.dev_ring = nullptr;
    }
    if (dev.ring.host_entries) {
      sycl::free(dev.ring.host_entries, q);
      dev.ring.host_entries = nullptr;
    }
    if (dev.ring.host_ready) {
      sycl::free(dev.ring.host_ready, q);
      dev.ring.host_ready = nullptr;
    }
    dev.gpu2cpu_queue = ctp::ipc::FullPtr<clio::run::GpuTaskQueue>::GetNull();
    dev.client_backends.clear();
  }
  per_gpu_devices_.clear();
}

bool gpu::IpcManager::RegisterClientBackend(const ClientBackend &b) {
  if (b.gpu_id >= per_gpu_devices_.size()) return false;
  u64 key = (static_cast<u64>(b.alloc_id.major_) << 32) |
            static_cast<u64>(b.alloc_id.minor_);
  per_gpu_devices_[b.gpu_id].client_backends[key] = b;
  return true;
}

void gpu::IpcManager::UnregisterClientBackend(
    u32 gpu_id, const ctp::ipc::AllocatorId &alloc_id) {
  if (gpu_id >= per_gpu_devices_.size()) return;
  u64 key = (static_cast<u64>(alloc_id.major_) << 32) |
            static_cast<u64>(alloc_id.minor_);
  per_gpu_devices_[gpu_id].client_backends.erase(key);
}

// FindClientBackend is now inline in gpu_ipc_manager.h.

/**
 * The batched device-ring transport is not built on the SYCL path.
 *
 * ServerInitGpuQueues above leaves `ring.dev_ring` null, so GetGpuInfo hands
 * the kernel a null `gpu2cpu_ring` and IpcGpu2Cpu::SendIn takes the legacy
 * per-task GpuTaskQueue path instead. Nothing ever pushes to a ring here, so
 * there is nothing for the worker to drain -- but the symbol is referenced
 * unconditionally by the worker poll loop in libclio_run_cxx, so it has to
 * exist.
 *
 * This costs the SYCL path the ring's whole point: one D2H copy per BATCH of
 * submissions rather than per submission. Porting it is mechanical (SYCL's
 * malloc_host is directly device-addressable, so it needs no
 * cudaHostGetDevicePointer step at all) and worth doing before any SYCL
 * performance claim -- but a correctness bring-up does not need it.
 */
bool gpu::IpcManager::RingNext(u32 gpu_id, clio::run::GpuRingEntry *out) {
  // Ported from gpu2cpu_init_hip.cc; the only backend call on this path is
  // the per-batch tail publish at the bottom. See that file for the reasoning
  // behind the stamp protocol and the two acquire fences.
  if (gpu_id >= per_gpu_devices_.size()) return false;
  auto &m = per_gpu_devices_[gpu_id].ring;
  if (m.dev_ring == nullptr || m.host_ready == nullptr) return false;
  // The ring is single-consumer by protocol (`tail`, `pending`, `pending_pos`
  // are plain fields). A file-static lock, NOT a member: GpuRingMirror's
  // layout is ABI for every TU that inlines gpu_ipc_manager.h's accessors,
  // and putting the mutex in the struct broke all of them at once.
  static std::mutex drain_mu[64];
  std::lock_guard<std::mutex> lk(drain_mu[gpu_id % 64]);
  if (m.pending_pos < m.pending.size()) {
    *out = m.pending[m.pending_pos++];
    return true;
  }
  m.pending.clear();
  m.pending_pos = 0;
  // THE STAMP IS THE ARRIVAL SIGNAL: head_ lives in device memory and is
  // never probed. A stamp carrying this slot's generation proves a producer
  // both claimed the slot and finished writing it -- the producer's system
  // fence orders its entry write ahead of its stamp write.
  auto *rdy = static_cast<volatile unsigned int *>(
      static_cast<void *>(m.host_ready));
  u32 accepted = 0;
  while (accepted < clio::run::kGpuRingCapacity) {
    const unsigned long long slot = m.tail + accepted;
    const u32 idx = static_cast<u32>(slot) & clio::run::kGpuRingMask;
    const unsigned int want =
        static_cast<unsigned int>(slot / clio::run::kGpuRingCapacity) + 1u;
    if (rdy[idx] != want) break;
    // ACQUIRE between the stamp and the entry, so the compiler cannot hoist
    // the entry load above the stamp check and hand us a stale generation.
    std::atomic_thread_fence(std::memory_order_acquire);
    clio::run::GpuRingEntry e = m.host_entries[idx];
    // Seqlock-style recheck: a changed stamp means the bytes may be torn.
    std::atomic_thread_fence(std::memory_order_acquire);
    if (rdy[idx] != want) break;
    m.pending.push_back(e);
    ++accepted;
  }
  if (accepted == 0) return false;
  // Publish the new tail so producers blocked on a full ring advance: one
  // H2D copy per BATCH of real work, never per poll.
  m.tail += accepted;
  auto *stream = static_cast<sycl::queue *>(m.stream);
  stream->memcpy(&m.dev_ring->tail_, &m.tail, sizeof(m.tail)).wait();
  *out = m.pending[m.pending_pos++];
  return true;
}

CLIO_RUN_GPU_API bool ChiServerBootstrapSyclGpu(IpcManager *self,
                                                clio::run::u32 queue_depth,
                                                size_t backend_bytes) {
  (void)backend_bytes;
  if (!self) return false;
  if (!self->gpu_ipc_) {
    self->gpu_ipc_ = std::make_unique<gpu::IpcManager>();
  }
  return self->gpu_ipc_->ServerInitGpuQueues(queue_depth);
}

#endif  // CTP_IS_HOST

}  // namespace clio::run

#endif  // CTP_ENABLE_SYCL && !(CUDA||ROCM)
