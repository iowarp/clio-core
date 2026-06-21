/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * GPU producer-only stress test (SYCL).
 *
 * SYCL twin of test_gpu_kernel_stress_gpu.cc. SYCL kernels run as
 * single_task by convention in this codebase (the CLIO_GPU_INIT
 * macro and IpcGpu2Cpu::ClientSend assume one work-item per kernel),
 * so we serialize submissions on the host: per slot, launch a
 * single_task that does Send + Wait, queue them all without waiting,
 * then wait_and_throw on the SYCL queue. The CPU GPU worker pops them
 * concurrently from gpu2cpu_queue (multi-MPSC ring buffer).
 */

#if CTP_ENABLE_SYCL && !(CTP_ENABLE_CUDA || CTP_ENABLE_ROCM)

#include "simple_test.h"
#include "test_gpu_kernel_stress_common.h"

#include <clio_ctp/util/gpu_api.h>
#include <sycl/sycl.hpp>

using TaskT = clio::run::MOD_NAME::GpuSubmitTask;

// SYCL kernel-name class. Namespace-scope (not anonymous) so DPC++'s
// host-side kernel-info table can resolve the mangled name.
class chi_sycl_stress_kernel;

TEST_CASE("GPU producer-only stress: kernel submits N tasks (SYCL)",
          "[gpu2cpu][stress][sycl]") {
  using namespace chi_test_gpu_stress;
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  const clio::run::u32 gpu_id = 0;

  char *base = nullptr;
  ctp::ipc::AllocatorId alloc_id = ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost, kBackendBytes,
      &base);
  REQUIRE(!alloc_id.IsNull());
  REQUIRE(base != nullptr);

  auto handles = PlaceTaskSlots(base, alloc_id, gpu_id);
  REQUIRE(handles.size() == kNumTasks);

  clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(gpu_id);
  REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

  // Stage gpu_info, kernel-scope IpcManager, and the FullPtr array in
  // shared USM so each kernel functor can capture pointers by value.
  //
  // Use a fresh GPU queue rather than ctp::GpuApi::SyclQueue() (the
  // singleton). The singleton was used for ServerInitGpuQueues during
  // CLIO_INIT; on the DPC++ CUDA backend, subsequent submissions
  // on the same singleton can silently no-op (observed: kernel runs
  // but malloc_shared writes don't propagate to host). A fresh queue
  // avoids that and inherits the same context, so the pinned-host
  // gpu2cpu_queue and any malloc_host backend are still reachable.
  sycl::queue q{sycl::gpu_selector_v};
  auto *info_storage = sycl::malloc_shared<clio::run::IpcManagerGpuInfo>(1, q);
  REQUIRE(info_storage != nullptr);
  *info_storage = gpu_info;

  auto *ipc_storage = sycl::malloc_shared<clio::run::gpu::IpcManager>(1, q);
  REQUIRE(ipc_storage != nullptr);
  new (ipc_storage) clio::run::gpu::IpcManager();

  auto *handle_storage =
      sycl::malloc_shared<ctp::ipc::FullPtr<TaskT>>(kNumTasks, q);
  REQUIRE(handle_storage != nullptr);
  for (clio::run::u32 i = 0; i < kNumTasks; ++i) handle_storage[i] = handles[i];

  // Slot index lives in shared USM so the kernel functor can read it
  // (lambda captures by value, but we want a fresh slot per submission).
  auto *slot_storage = sycl::malloc_shared<clio::run::u32>(1, q);
  REQUIRE(slot_storage != nullptr);

  std::fprintf(stderr, "[STRESS] queuing %u single_task submissions\n",
               kNumTasks);
  // Marker the kernel writes through to confirm the body actually executed.
  // DPC++ has been observed to elide kernel emission when the lambda body's
  // only host-pass content is (void)cast no-ops, so write through this
  // pointer unconditionally.
  auto *marker_storage = sycl::malloc_shared<clio::run::u32>(1, q);
  REQUIRE(marker_storage != nullptr);
  // First do a minimal sanity kernel to confirm the SYCL+CUDA backend
  // can run a basic kernel under our setup.
  *marker_storage = 0;
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class chi_sycl_stress_sanity>([=]() {
      *marker_storage = 0xBEEF;
    });
  }).wait_and_throw();
  std::fprintf(stderr, "[STRESS] sanity marker=%u (expected %u)\n",
               *marker_storage, 0xBEEF);
  REQUIRE(*marker_storage == 0xBEEF);

  for (clio::run::u32 i = 0; i < kNumTasks; ++i) {
    *slot_storage = i;
    *marker_storage = 0;
    try {
      q.submit([&](sycl::handler &cgh) {
        cgh.single_task<chi_sycl_stress_kernel>([=]() {
          // Both passes must touch every capture so DPC++'s host-side
          // capture analysis lays out the kernel argument tuple
          // identically to the device pass — otherwise DPC++ asserts
          // "Unexpected kernel lambda size".
          (void)info_storage; (void)ipc_storage;
          (void)handle_storage; (void)slot_storage;
          *marker_storage = 1;
#if CTP_IS_DEVICE_PASS
          CLIO_GPU_INIT(*info_storage, ipc_storage);
          *marker_storage = 2;
          clio::run::u32 slot = *slot_storage;
          auto fp = handle_storage[slot];
          if (!fp.IsNull()) {
            *marker_storage = 3;
            auto fut = CLIO_IPC->Send(fp);
            *marker_storage = 4;
            fut.Wait();
            *marker_storage = 5;
          } else {
            *marker_storage = 99;
          }
          (void)g_ipc_manager;
#endif
        });
      }).wait_and_throw();
    } catch (const sycl::exception &e) {
      std::fprintf(stderr, "[STRESS] slot=%u SYCL EXCEPTION: %s marker=%u\n",
                   i, e.what(), *marker_storage);
      std::fflush(stderr);
      _exit(1);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "[STRESS] slot=%u std EXCEPTION: %s marker=%u\n",
                   i, e.what(), *marker_storage);
      std::fflush(stderr);
      _exit(1);
    }
    if (*marker_storage != 5u) {
      std::fprintf(stderr, "[STRESS] slot=%u marker=%u (expected 5)\n", i,
                   *marker_storage);
      std::fflush(stderr);
      _exit(1);
    }
  }

  clio::run::u32 first_bad = VerifyResults(handles, gpu_id);
  REQUIRE(first_bad == kNumTasks);
  std::fprintf(stderr, "[STRESS] all %u tasks completed correctly\n",
               kNumTasks);

  ipc_storage->~IpcManager();
  sycl::free(static_cast<void *>(ipc_storage), q);
  sycl::free(static_cast<void *>(info_storage), q);
  sycl::free(static_cast<void *>(handle_storage), q);
  sycl::free(slot_storage, q);
  sycl::free(marker_storage, q);
  ipc->FreeGpuBackend(gpu_id, alloc_id);
}

SIMPLE_TEST_MAIN()

#else  // SYCL not enabled (or CUDA/ROCm is)

int main() { return 0; }

#endif
