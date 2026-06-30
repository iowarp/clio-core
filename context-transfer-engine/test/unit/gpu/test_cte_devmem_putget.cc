/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Focused proof-of-concept test for the producer-only GPU2CPU path with
 * **everything in device memory**:
 *   - Task POD (PodPutBlobTask / PodGetBlobTask) lives in a registered
 *     kDeviceMem backend. The host pre-constructs a prototype via
 *     placement-new, then cudaMemcpy's it onto the device.
 *   - gpu::FutureShm sits co-located right after the task POD, also in
 *     kDeviceMem.
 *   - Blob payload bytes live in a separate kDeviceMem backend and the
 *     bdev runtime memcpy's them via DeviceAwareMemcpy on the bdev path.
 *
 * Worker flow exercised:
 *   1. Kernel mutates the device-side POD (writes pattern to blob_data,
 *      then calls Send) and waits on the device-side gpu::FutureShm.
 *   2. CPU GPU worker pops the task, D2H-copies the POD into a host
 *      scratch slot, calls Container::FixupAfterCopy to relocate the
 *      clio::run::priv::string SSO data_ pointer, runs the chimod handler,
 *      then H2D-copies the (mutated) POD back to the original device
 *      address and flips FUTURE_COMPLETE on the device fshm via
 *      cudaMemcpy.
 *   3. Kernel sees FUTURE_COMPLETE, reads task->return_code_ from
 *      device memory, and proceeds to GetBlob.
 *
 * The two-task round-trip gives us end-to-end coverage of:
 *   - Device→host POD copy + SSO fixup on the inbound path.
 *   - DeviceAwareMemcpy on the bdev side reading kDeviceMem blob_data.
 *   - Host→device POD writeback so the kernel sees output fields.
 *   - cudaMemcpy of the FUTURE_COMPLETE flag word as the device-fshm
 *     wakeup signal.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

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
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace cte = clio::cte::core;
using ClioInit = clio::run::RuntimeMode;

namespace {

constexpr clio::run::u32 kBlobBytes = 256;
constexpr clio::run::u32 kPatternSeed = 0xC3u;
constexpr const char *kBlobName = "kdev";  // SSO-friendly (≤ 23 chars)

bool g_initialized = false;
cte::TagId g_tag_id;

/** One-shot bring-up: server, CTE pool, kRam bdev target, tag. */
void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  std::fprintf(stderr, "[INIT] Bringing up Clio server\n");
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  REQUIRE(cte::CLIO_CTE_CLIENT_INIT());
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);
  cte_client->Init(cte::kCtePoolId);
  cte::CreateParams params;
  auto create_task = cte_client->AsyncCreate(
      clio::run::PoolQuery::Dynamic(), cte::kCtePoolName, cte::kCtePoolId, params);
  create_task.Wait();
  REQUIRE(create_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);

  // Register a kRam bdev target so PutBlob has somewhere to land.
  const clio::run::u64 kRamCapacity = 64ULL << 20;  // 64 MiB
  clio::run::PoolId bdev_pool_id(960, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("cte_devmem_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, kRamCapacity);
  bdev_create.Wait();
  REQUIRE(bdev_create->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "cte_devmem_ram", clio::run::bdev::BdevType::kRam, kRamCapacity,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);

  // Pre-create the tag the test will reference.
  auto tag_task = cte_client->AsyncGetOrCreateTag("cte_devmem_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  g_tag_id = tag_task->tag_id_;

  g_initialized = true;
  std::fprintf(stderr, "[INIT] Ready (tag=(%u,%u))\n",
               g_tag_id.major_, g_tag_id.minor_);
#endif
}

}  // namespace

/** Fill the blob_data buffer on device with the byte pattern. */
__global__ void DevMemFillKernel(char *buf, clio::run::u32 size, clio::run::u32 seed) {
  clio::run::u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  buf[i] = static_cast<char>((seed ^ i) & 0xFFu);
}

/** Submit one pre-built device-resident task and wait for completion. */
__global__ void DevMemSubmitPutKernel(clio::run::IpcManagerGpuInfo info,
                                       ctp::ipc::FullPtr<cte::PodPutBlobTask> task) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (threadIdx.x != 0) return;
  auto fut = g_ipc_manager_ptr->Send(task);
  fut.Wait();
  (void)g_ipc_manager;
}

__global__ void DevMemSubmitGetKernel(clio::run::IpcManagerGpuInfo info,
                                       ctp::ipc::FullPtr<cte::PodGetBlobTask> task) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (threadIdx.x != 0) return;
  auto fut = g_ipc_manager_ptr->Send(task);
  fut.Wait();
  (void)g_ipc_manager;
}

/** Verify the byte pattern on the device-resident GET buffer (host
 *  reads it back via cudaMemcpy after the kernel returns). */
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

TEST_CASE("CTE PutBlob+GetBlob round trip with device-memory task & data",
          "[cte][devmem][putblob][getblob]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  REQUIRE(ipc->GetGpuQueueCount() >= 1u);

  clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
  REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

  // ---- 1) Allocate kDeviceMem backends ----
  // (a) Task slots: PodPutBlobTask + PodGetBlobTask (each self-contained — the Task
  //     carries its own completion record in fut_, no co-located FutureShm).
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
      /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
      kBlobBytes, &blob_dev);
  REQUIRE(!blob_alloc_id.IsNull());
  REQUIRE(blob_dev != nullptr);

  // ---- 2) Construct host prototypes via placement-new and stamp them
  //         onto the device task slots via cudaMemcpy. ----
  // PutBlob prototype:
  alignas(64) char put_proto[kPutSlot];
  std::memset(put_proto, 0, sizeof(put_proto));
  ctp::ipc::ShmPtr<> put_blob_shm;
  put_blob_shm.alloc_id_.SetNull();
  put_blob_shm.off_ = reinterpret_cast<clio::run::u64>(blob_dev);
  auto *put_proto_task = new (put_proto) cte::PodPutBlobTask(
      clio::run::CreateTaskId(), cte::kCtePoolId, clio::run::PoolQuery::ToLocalCpu(),
      g_tag_id, kBlobName, /*offset=*/clio::run::u64(0),
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
      g_tag_id, kBlobName, /*offset=*/clio::run::u64(0),
      static_cast<clio::run::u64>(kBlobBytes), /*flags=*/clio::run::u32(0),
      get_blob_shm);
  get_proto_task->fut_.task_size_ = sizeof(cte::PodGetBlobTask);
  ctp::GpuApi::Memcpy(task_dev_base + kPutSlot, get_proto, sizeof(get_proto));

  // ---- 3) Fill blob_data on device with the source pattern. ----
  clio::run::u32 fill_threads = 256;
  clio::run::u32 fill_blocks = (kBlobBytes + fill_threads - 1) / fill_threads;
  DevMemFillKernel<<<fill_blocks, fill_threads>>>(blob_dev, kBlobBytes,
                                                    kPatternSeed);
  ctp::GpuApi::Synchronize();

  // ---- 4) Build kernel-visible FullPtrs (raw device addresses
  //         stashed in off_, null alloc_id). ----
  ctp::ipc::FullPtr<cte::PodPutBlobTask> put_fp;
  put_fp.shm_.alloc_id_.SetNull();
  put_fp.shm_.off_ = reinterpret_cast<clio::run::u64>(task_dev_base);
  put_fp.ptr_ = reinterpret_cast<cte::PodPutBlobTask *>(task_dev_base);
  ctp::ipc::FullPtr<cte::PodGetBlobTask> get_fp;
  get_fp.shm_.alloc_id_.SetNull();
  get_fp.shm_.off_ =
      reinterpret_cast<clio::run::u64>(task_dev_base + kPutSlot);
  get_fp.ptr_ = reinterpret_cast<cte::PodGetBlobTask *>(
      task_dev_base + kPutSlot);

  // ---- 5) Launch the PutBlob kernel and wait. ----
  std::fprintf(stderr, "[PUT] launching DevMemSubmitPutKernel\n");
  auto t0 = std::chrono::steady_clock::now();
  DevMemSubmitPutKernel<<<1, 32>>>(gpu_info, put_fp);
  ctp::GpuApi::Synchronize();
  auto t1 = std::chrono::steady_clock::now();

  // Pull return_code_ back from device.
  cte::PodPutBlobTask put_after{};
  ctp::GpuApi::Memcpy(reinterpret_cast<char *>(&put_after),
                        task_dev_base, sizeof(cte::PodPutBlobTask));
  std::fprintf(stderr, "[PUT] return_code=%u took=%lld ms\n",
               put_after.return_code_.load(),
               (long long)std::chrono::duration_cast<
                   std::chrono::milliseconds>(t1 - t0).count());
  REQUIRE(put_after.return_code_.load() == 0u);

  // ---- 6) Zero out the blob_data buffer on device so the GetBlob
  //         readback is verifiable. ----
  std::vector<char> zeros(kBlobBytes, 0);
  ctp::GpuApi::Memcpy(blob_dev, zeros.data(), kBlobBytes);

  // ---- 7) Launch the GetBlob kernel and wait. ----
  std::fprintf(stderr, "[GET] launching DevMemSubmitGetKernel\n");
  auto g0 = std::chrono::steady_clock::now();
  DevMemSubmitGetKernel<<<1, 32>>>(gpu_info, get_fp);
  ctp::GpuApi::Synchronize();
  auto g1 = std::chrono::steady_clock::now();

  cte::PodGetBlobTask get_after{};
  ctp::GpuApi::Memcpy(reinterpret_cast<char *>(&get_after),
                        task_dev_base + kPutSlot,
                        sizeof(cte::PodGetBlobTask));
  std::fprintf(stderr, "[GET] return_code=%u took=%lld ms\n",
               get_after.return_code_.load(),
               (long long)std::chrono::duration_cast<
                   std::chrono::milliseconds>(g1 - g0).count());
  REQUIRE(get_after.return_code_.load() == 0u);

  // ---- 8) Verify the device buffer contains the original pattern. ----
  clio::run::u32 first_bad = VerifyDevicePattern(blob_dev, kBlobBytes,
                                            kPatternSeed);
  if (first_bad != kBlobBytes) {
    std::fprintf(stderr,
                 "[VERIFY] mismatch at index %u (out of %u)\n",
                 first_bad, kBlobBytes);
  }
  REQUIRE(first_bad == kBlobBytes);
  std::fprintf(stderr, "[OK] devmem put+get round trip ok (%u bytes)\n",
               kBlobBytes);

  // ---- 9) Free backends. ----
  ipc->FreeGpuBackend(/*gpu_id=*/0, blob_alloc_id);
  ipc->FreeGpuBackend(/*gpu_id=*/0, task_alloc_id);
}

#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
