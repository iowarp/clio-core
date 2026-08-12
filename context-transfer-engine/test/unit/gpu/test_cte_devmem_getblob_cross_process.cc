/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Cross-process CTE GetBlob reading directly into a CUDA-IPC device-memory
 * destination buffer -- the symmetric read direction to
 * test_cte_devmem_putget_cross_process.cc's PutBlob-from-device case. Kept
 * in its own binary because two RuntimeServer+CLIO_INIT sequences in one
 * process hang (a test-infrastructure limitation unrelated to GPU IPC).
 *
 * PutBlob goes through an ordinary HOST buffer here, so only the
 * GetBlob-into-device path is under test. AsyncGetBlob's zero-IPC fast path
 * (TryReadBlobShm) used to std::memcpy straight from the RAM bdev's mapped
 * SHM segment into the caller's destination; for a device destination that
 * segfaults. CopyBlobBytesInto (core_client.cc) now uses
 * ctp::DeviceAwareMemcpy so it transparently handles either.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"
#include "runtime_server.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>

#include <cstring>
#include <vector>

namespace cte = clio::cte::core;

namespace {
constexpr size_t kBytes = 4096;
constexpr char kPatternSeed = 0x5A;
}  // namespace

TEST_CASE("CTE GetBlob reads directly into a cross-process CUDA-IPC "
          "device-memory destination buffer",
          "[cte][devmem][cross_process][gpu_ipc][693]") {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return;
  }

  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start());
  REQUIRE(server.WaitForReady());

  clio::run::test::SetEnvVar("CLIO_WITH_RUNTIME", "0");
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false));
  REQUIRE(cte::CLIO_CTE_CLIENT_INIT());

  auto *cte_client = CLIO_CTE_CLIENT;
  cte::CreateParams params;
  auto create_task = cte_client->AsyncCreate(
      clio::run::PoolQuery::Dynamic(), cte::kCtePoolName, cte::kCtePoolId,
      params);
  create_task.Wait();
  REQUIRE(create_task->GetReturnCode() == 0);

  clio::run::PoolId bdev_pool_id(963, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("cte_devmem_get_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, 64ULL << 20);
  bdev_create.Wait();
  REQUIRE(bdev_create->GetReturnCode() == 0);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "cte_devmem_get_ram", clio::run::bdev::BdevType::kRam, 64ULL << 20,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);

  auto tag_task = cte_client->AsyncGetOrCreateTag("cte_devmem_get_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  cte::TagId tag_id = tag_task->tag_id_;

  // ---- PutBlob a known pattern via an ordinary HOST buffer. ----
  std::vector<char> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i) {
    pattern[i] = static_cast<char>((kPatternSeed + i) & 0xFF);
  }
  auto put_buffer = CLIO_IPC->AllocateBuffer(kBytes);
  REQUIRE(!put_buffer.IsNull());
  std::memcpy(put_buffer.ptr_, pattern.data(), kBytes);
  ctp::ipc::ShmPtr<> put_blob_data = put_buffer.shm_.template Cast<void>();
  auto put_task = cte_client->AsyncPutBlob(tag_id, "devmem_get_blob", 0,
                                           kBytes, put_blob_data);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);
  CLIO_IPC->FreeBuffer(put_buffer);

  // ---- GetBlob directly into real GPU device memory in THIS process. ----
  auto *ipc = CLIO_IPC;
  char *device_base = nullptr;
  auto alloc_id = ipc->AllocateAndRegisterGpuBackend(
      /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem, kBytes,
      &device_base);
  REQUIRE(!alloc_id.IsNull());
  REQUIRE(device_base != nullptr);

  ctp::ipc::ShmPtr<> get_blob_data;
  get_blob_data.alloc_id_ = alloc_id;
  get_blob_data.off_ = reinterpret_cast<clio::run::u64>(device_base);
  auto get_task = cte_client->AsyncGetBlob(tag_id, "devmem_get_blob", 0,
                                           kBytes, 0, get_blob_data);
  get_task.Wait();
  REQUIRE(get_task->GetReturnCode() == 0);

  // ---- Copy the device buffer back to host and verify the pattern. ----
  std::vector<char> readback(kBytes);
  ctp::GpuApi::Memcpy(readback.data(), device_base, kBytes);
  REQUIRE(std::memcmp(readback.data(), pattern.data(), kBytes) == 0);

  ipc->FreeGpuBackend(/*gpu_id=*/0, alloc_id);
}

SIMPLE_TEST_MAIN()

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
