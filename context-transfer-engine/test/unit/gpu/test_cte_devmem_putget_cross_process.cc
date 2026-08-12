/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Cross-process CTE PutBlob/GetBlob with a CUDA-IPC device-memory buffer.
 *
 * test_cte_devmem_putget.cc already covers the single-process (IsRuntime()
 * == true) case, where AllocateAndRegisterGpuBackend short-circuits
 * straight to gpu_ipc_->RegisterClientBackend and no bytes ever cross a
 * transport. This test spawns a REAL separate clio_run daemon so the blob
 * data has to travel through the actual client/daemon boundary a genuine
 * external client uses.
 *
 * A device pointer isn't host-readable, so the network transport that
 * carries a PutBlob task's bulk payload can't read it directly (it would
 * segfault). CoreClient::AsyncPutBlob detects this (IsDevicePointer on the
 * resolved blob_data) and stages a D2H copy into a fresh SHM buffer before
 * submitting -- the same "daemon can't reach it, stage a copy" idiom the
 * private-memory AsyncPutBlob overload already uses for issue #830.
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

TEST_CASE("CTE PutBlob+GetBlob round trip with a cross-process CUDA-IPC "
          "device-memory buffer",
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

  clio::run::PoolId bdev_pool_id(962, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("cte_devmem_cp2_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, 64ULL << 20);
  bdev_create.Wait();
  REQUIRE(bdev_create->GetReturnCode() == 0);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "cte_devmem_cp2_ram", clio::run::bdev::BdevType::kRam, 64ULL << 20,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);

  auto tag_task = cte_client->AsyncGetOrCreateTag("cte_devmem_cp2_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  cte::TagId tag_id = tag_task->tag_id_;

  // ---- Allocate real GPU device memory in THIS process and fill it. ----
  auto *ipc = CLIO_IPC;
  char *device_base = nullptr;
  auto alloc_id = ipc->AllocateAndRegisterGpuBackend(
      /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem, kBytes,
      &device_base);
  REQUIRE(!alloc_id.IsNull());
  REQUIRE(device_base != nullptr);

  std::vector<char> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i) {
    pattern[i] = static_cast<char>((kPatternSeed + i) & 0xFF);
  }
  ctp::GpuApi::Memcpy(device_base, pattern.data(), kBytes);

  // ---- PutBlob with a ShmPtr into that device backend. ----
  ctp::ipc::ShmPtr<> blob_data;
  blob_data.alloc_id_ = alloc_id;
  blob_data.off_ = reinterpret_cast<clio::run::u64>(device_base);
  auto put_task = cte_client->AsyncPutBlob(tag_id, "devmem_cp2_blob", 0,
                                           kBytes, blob_data);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  // ---- GetBlob back into a fresh host buffer and verify the pattern. ----
  auto get_buffer = CLIO_IPC->AllocateBuffer(kBytes);
  REQUIRE(!get_buffer.IsNull());
  ctp::ipc::ShmPtr<> get_blob_data = get_buffer.shm_.template Cast<void>();
  auto get_task = cte_client->AsyncGetBlob(tag_id, "devmem_cp2_blob", 0,
                                           kBytes, 0, get_blob_data);
  get_task.Wait();
  REQUIRE(get_task->GetReturnCode() == 0);

  REQUIRE(std::memcmp(get_buffer.ptr_, pattern.data(), kBytes) == 0);

  CLIO_IPC->FreeBuffer(get_buffer);
  ipc->FreeGpuBackend(/*gpu_id=*/0, alloc_id);
}

SIMPLE_TEST_MAIN()

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
