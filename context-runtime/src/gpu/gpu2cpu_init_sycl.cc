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

#include <sycl/sycl.hpp>

#include <cstring>
#include <memory>
#include <new>

namespace clio::run {

namespace {
class clio_sycl_init_queue_kernel;
}

/**
 * Construct the GpuTaskQueue inside a pinned-host backend, on the device.
 *
 * OUTSIDE THE CTP_IS_HOST GUARD BELOW, and that is the whole point.
 *
 * In CUDA, a kernel is a __global__ function at namespace scope and the
 * launch is ordinary host code, so host-only regions may freely contain
 * launches. In SYCL the kernel BODY is a lambda inside the submitting
 * function -- so a `#if CTP_IS_HOST` around the submitter deletes the
 * kernel from the device pass, no device code is emitted for it, and the
 * host pass then submits a kernel the runtime has never heard of:
 *
 *   Assertion `It != m_DeviceKernelInfoMap.end()' failed.
 *   ProgramManager::getDeviceKernelInfo(...)
 *
 * which is what `clio_run start` did. Splitting the kernel out into a
 * function that touches no host-only state lets both passes see it, while
 * ServerInitGpuQueues -- which does reach host-only IpcManager members --
 * stays guarded.
 *
 * @return the queue's offset within `backend`, or size_t(-1) on failure.
 */
static size_t SyclBuildQueueInBackend(sycl::queue &q, char *backend,
                                      size_t backend_bytes, u32 queue_depth) {
  size_t *out_off = sycl::malloc_shared<size_t>(1, q);
  if (!out_off) return static_cast<size_t>(-1);
  *out_off = static_cast<size_t>(-1);
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<clio_sycl_init_queue_kernel>([=]() {
      ctp::ipc::MemoryBackend proxy;
      proxy.data_ = backend;
      proxy.data_capacity_ = backend_bytes;
      CLIO_QUEUE_ALLOC_T *alloc = proxy.MakeAlloc<CLIO_QUEUE_ALLOC_T>();
      if (!alloc) {
        *out_off = static_cast<size_t>(-1);
        return;
      }
      ctp::ipc::FullPtr<clio::run::GpuTaskQueue> queue =
          alloc->NewObj<clio::run::GpuTaskQueue>(
              alloc, /*num_lanes=*/1u, /*num_prio=*/2u, queue_depth);
      *out_off = queue.IsNull() ? static_cast<size_t>(-1)
                                : queue.shm_.off_.load();
    });
  }).wait_and_throw();
  const size_t off = *out_off;
  sycl::free(out_off, q);
  return off;
}

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

    dev.queue_backend = static_cast<char *>(
        sycl::malloc_host(kQueueBackendBytes, q));
    if (!dev.queue_backend) {
      HLOG(kError, "ServerInitGpuQueues (SYCL): malloc_host failed (gpu_id={})",
           gpu_id);
      FinalizeGpuQueues();
      return false;
    }
    dev.queue_backend_size = kQueueBackendBytes;
    std::memset(dev.queue_backend, 0, kQueueBackendBytes);

    const size_t queue_off = SyclBuildQueueInBackend(
        q, dev.queue_backend, kQueueBackendBytes, queue_depth);
    if (queue_off == static_cast<size_t>(-1)) {
      HLOG(kError, "ServerInitGpuQueues (SYCL): device queue construction "
           "failed (gpu_id={})", gpu_id);
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
      sycl::free(dev.queue_backend, q);
      dev.queue_backend = nullptr;
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
  (void)gpu_id;
  (void)out;
  return false;
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
