/*
 * iowarp CTE GPU integration test — PutBlob + GetBlob round trip.
 *
 * Asserts the iowarp "producer-only" GPU CTE contract that kvhdf5 is built on:
 * the host pre-allocates the self-contained Pod Task (its completion record
 * lives in the embedded fut_) + blob buffer in registered kDeviceMem backends,
 * and the kernel only Send()s the task and
 * Wait()s on its future — no orchestrator pause/resume, no device-side
 * allocation. If a future iowarp bump breaks this contract, this test fails
 * and the failure is attributable to iowarp, not to kvhdf5.
 *
 * Faithfully mirrors iowarp-core's reference proof-of-concept
 *   context-transfer-engine/test/unit/gpu/test_cte_devmem_putget.cc
 * swapping only the harness (SimpleTest -> Catch2, the kvhdf5 standard) and
 * lifting the one-time bring-up into a shared singleton (SharedCteEnv) so that
 * additional GPU integration cases added later reuse the expensive server
 * start-up instead of re-paying it per case.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/future.h>
#include <clio_runtime/gpu/gpu_info.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

// Catch2 macros (REQUIRE) are used by host-only code below (the shared
// environment + the TEST_CASE). Only pull them in on the host compilation
// pass; the nvcc device pass never sees REQUIRE. SharedCteEnv (the one-time
// CLIO bring-up) lives in a shared header so other GPU integration cases
// reuse the expensive server start; it self-guards out of the device pass.
#if !CTP_IS_DEVICE_PASS
#include <catch2/catch_test_macros.hpp>
#endif
#include "cte_env.h"

using namespace std::chrono_literals;

namespace cte = clio::cte::core;

namespace {

constexpr clio::run::u32 kBlobBytes = 256;
constexpr clio::run::u32 kPatternSeed = 0xC3u;
constexpr const char *kBlobName = "kdev";  // SSO-friendly (<= 23 chars)

}  // namespace

/** Fill the blob_data buffer on device with the byte pattern. */
__global__ void CteFillKernel(char *buf, clio::run::u32 size, clio::run::u32 seed) {
  clio::run::u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  buf[i] = static_cast<char>((seed ^ i) & 0xFFu);
}

/** Submit one pre-built device-resident task and wait for completion. */
__global__ void CteSubmitPutKernel(clio::run::IpcManagerGpuInfo info,
                                   ctp::ipc::FullPtr<cte::PodPutBlobTask> task) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (threadIdx.x != 0) return;
  auto fut = g_ipc_manager_ptr->Send(task);
  fut.Wait();
  (void)g_ipc_manager;
}

__global__ void CteSubmitGetKernel(clio::run::IpcManagerGpuInfo info,
                                   ctp::ipc::FullPtr<cte::PodGetBlobTask> task) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (threadIdx.x != 0) return;
  auto fut = g_ipc_manager_ptr->Send(task);
  fut.Wait();
  (void)g_ipc_manager;
}

/** Verify the byte pattern on the device-resident GET buffer (host reads it
 *  back via cudaMemcpy after the kernel returns). */
clio::run::u32 VerifyDevicePattern(const char *device_buf, clio::run::u32 size,
                             clio::run::u32 seed) {
  std::vector<char> host(size);
  ctp::GpuApi::Memcpy(host.data(), device_buf, size);
  for (clio::run::u32 i = 0; i < size; ++i) {
    char want = static_cast<char>((seed ^ i) & 0xFFu);
    if (host[i] != want) return i;
  }
  return size;  // all match
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("iowarp CTE PutBlob+GetBlob round trip (device-memory task & data)",
          "[integration][gpu][cte][putget]") {
  auto &env = kvhdf5::itest::SharedCteEnv();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  REQUIRE(ipc->GetGpuQueueCount() >= 1u);

  clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
  REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

  // ---- 1) Allocate kDeviceMem backends ----
  // (a) Task slots: PodPutBlobTask + PodGetBlobTask (each self-contained — the
  //     Task carries its own completion record in fut_, no co-located FutureShm).
  const clio::run::u32 kPutSlot = sizeof(cte::PodPutBlobTask);
  const clio::run::u32 kGetSlot = sizeof(cte::PodGetBlobTask);
  const clio::run::u32 kTaskBackendBytes = kPutSlot + kGetSlot + 64;
  char *task_dev_base = nullptr;
  auto task_alloc_id = ipc->AllocateAndRegisterGpuBackend(
      /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
      kTaskBackendBytes, &task_dev_base);
  REQUIRE(!task_alloc_id.IsNull());
  REQUIRE(task_dev_base != nullptr);

  // (b) Blob data: a single 256-byte buffer used for both Put and Get.
  char *blob_dev = nullptr;
  auto blob_alloc_id = ipc->AllocateAndRegisterGpuBackend(
      /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem, kBlobBytes,
      &blob_dev);
  REQUIRE(!blob_alloc_id.IsNull());
  REQUIRE(blob_dev != nullptr);

  // ---- 2) Construct host prototypes via placement-new and stamp them onto
  //         the device task slots via cudaMemcpy. ----
  // PutBlob prototype:
  alignas(64) char put_proto[kPutSlot];
  std::memset(put_proto, 0, sizeof(put_proto));
  ctp::ipc::ShmPtr<> put_blob_shm;
  put_blob_shm.alloc_id_.SetNull();
  put_blob_shm.off_ = reinterpret_cast<clio::run::u64>(blob_dev);
  auto *put_proto_task = new (put_proto) cte::PodPutBlobTask(
      clio::run::CreateTaskId(), cte::kCtePoolId, clio::run::PoolQuery::ToLocalCpu(),
      env.tag_id, kBlobName, /*offset=*/clio::run::u64(0),
      static_cast<clio::run::u64>(kBlobBytes), put_blob_shm,
      /*score=*/-1.0f, cte::Context(), /*flags=*/clio::run::u32(0));
  put_proto_task->fut_.task_size_ = sizeof(cte::PodPutBlobTask);
  ctp::GpuApi::Memcpy(task_dev_base, put_proto, sizeof(put_proto));

  // GetBlob prototype:
  alignas(64) char get_proto[kGetSlot];
  std::memset(get_proto, 0, sizeof(get_proto));
  ctp::ipc::ShmPtr<> get_blob_shm;
  get_blob_shm.alloc_id_.SetNull();
  get_blob_shm.off_ = reinterpret_cast<clio::run::u64>(blob_dev);
  auto *get_proto_task = new (get_proto) cte::PodGetBlobTask(
      clio::run::CreateTaskId(), cte::kCtePoolId, clio::run::PoolQuery::ToLocalCpu(),
      env.tag_id, kBlobName, /*offset=*/clio::run::u64(0),
      static_cast<clio::run::u64>(kBlobBytes), /*flags=*/clio::run::u32(0), get_blob_shm);
  get_proto_task->fut_.task_size_ = sizeof(cte::PodGetBlobTask);
  ctp::GpuApi::Memcpy(task_dev_base + kPutSlot, get_proto, sizeof(get_proto));

  // ---- 3) Fill blob_data on device with the source pattern. ----
  clio::run::u32 fill_threads = 256;
  clio::run::u32 fill_blocks = (kBlobBytes + fill_threads - 1) / fill_threads;
  CteFillKernel<<<fill_blocks, fill_threads>>>(blob_dev, kBlobBytes,
                                               kPatternSeed);
  ctp::GpuApi::Synchronize();

  // ---- 4) Build kernel-visible FullPtrs (raw device addresses stashed in
  //         off_, null alloc_id). ----
  ctp::ipc::FullPtr<cte::PodPutBlobTask> put_fp;
  put_fp.shm_.alloc_id_.SetNull();
  put_fp.shm_.off_ = reinterpret_cast<clio::run::u64>(task_dev_base);
  put_fp.ptr_ = reinterpret_cast<cte::PodPutBlobTask *>(task_dev_base);
  ctp::ipc::FullPtr<cte::PodGetBlobTask> get_fp;
  get_fp.shm_.alloc_id_.SetNull();
  get_fp.shm_.off_ = reinterpret_cast<clio::run::u64>(task_dev_base + kPutSlot);
  get_fp.ptr_ = reinterpret_cast<cte::PodGetBlobTask *>(task_dev_base + kPutSlot);

  // ---- 5) Launch the PutBlob kernel and wait. ----
  std::fprintf(stderr, "[put] launching CteSubmitPutKernel\n");
  CteSubmitPutKernel<<<1, 32>>>(gpu_info, put_fp);
  ctp::GpuApi::Synchronize();

  // Pull return_code_ back from device.
  cte::PodPutBlobTask put_after{};
  ctp::GpuApi::Memcpy(reinterpret_cast<char *>(&put_after), task_dev_base,
                      sizeof(cte::PodPutBlobTask));
  std::fprintf(stderr, "[put] return_code=%u\n", put_after.return_code_.load());
  REQUIRE(put_after.return_code_.load() == 0u);

  // ---- 6) Zero out the blob_data buffer on device so the GetBlob readback
  //         is verifiable. ----
  std::vector<char> zeros(kBlobBytes, 0);
  ctp::GpuApi::Memcpy(blob_dev, zeros.data(), kBlobBytes);

  // ---- 7) Launch the GetBlob kernel and wait. ----
  std::fprintf(stderr, "[get] launching CteSubmitGetKernel\n");
  CteSubmitGetKernel<<<1, 32>>>(gpu_info, get_fp);
  ctp::GpuApi::Synchronize();

  cte::PodGetBlobTask get_after{};
  ctp::GpuApi::Memcpy(reinterpret_cast<char *>(&get_after),
                      task_dev_base + kPutSlot, sizeof(cte::PodGetBlobTask));
  std::fprintf(stderr, "[get] return_code=%u\n", get_after.return_code_.load());
  REQUIRE(get_after.return_code_.load() == 0u);

  // ---- 8) Verify the device buffer contains the original pattern. ----
  clio::run::u32 first_bad = VerifyDevicePattern(blob_dev, kBlobBytes, kPatternSeed);
  if (first_bad != kBlobBytes) {
    std::fprintf(stderr, "[verify] mismatch at index %u (out of %u)\n",
                 first_bad, kBlobBytes);
  }
  REQUIRE(first_bad == kBlobBytes);
  std::fprintf(stderr, "[ok] cte devmem put+get round trip ok (%u bytes)\n",
               kBlobBytes);

  // ---- 9) Free backends. ----
  ipc->FreeGpuBackend(/*gpu_id=*/0, blob_alloc_id);
  ipc->FreeGpuBackend(/*gpu_id=*/0, task_alloc_id);
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
