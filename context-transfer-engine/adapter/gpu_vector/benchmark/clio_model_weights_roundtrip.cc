/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * End-to-end integration test for the model-weights CAE assimilation path.
 *
 *   1. Generate a deterministic "weights" file.
 *   2. Assimilate it into a single CTE tag via the CAE ModelWeightsAssimilator
 *      (protocol modelweights::), producing <tag>_b<block>_pi<page> blobs.
 *   3. Map that tag with clio::cte::gpu_vector::Vector<uint8_t> and stream every
 *      byte back through read_range (cold-miss faults pull pages from CTE).
 *   4. Verify the bytes read through the GPU vector match the original file.
 *
 * This proves the full clio-core round-trip: assimilate -> tag -> map -> read,
 * with weights served on demand by the GPU vector rather than a file re-read.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_info.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

// Defined in clio_mw_assimilate.cc (plain C++ TU — CAE headers don't compile
// under nvcc). Assimilates `path` into tag `tag` for a (page_size, nblocks)
// vector layout; returns 0 on success.
int mw_assimilate(const std::string& path, const std::string& tag,
                  unsigned long long page_size, unsigned nblocks);

/** Read the whole vector [0,total) through read_range into `out` (device). */
__global__ void ReadAllKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceView<uint8_t> view,
                              uint8_t *out, clio::run::u64 total) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<uint8_t> v(view, g_ipc_manager_ptr);
  if (blockIdx.x != 0) return;  // single-block (nblocks=1) layout
  v.read_range(0, total, [out](clio::run::u64 i, uint8_t b) { out[i] = b; });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

namespace {

void EnsureInit(clio::run::u64 bdev_capacity_bytes) {
  std::fprintf(stderr, "[INIT] Starting Clio server\n");
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer)) {
    std::fprintf(stderr, "[INIT] CLIO_INIT failed\n");
    std::exit(2);
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[INIT] CLIO_CTE_CLIENT_INIT failed\n");
    std::exit(2);
  }
  auto *cte_client = CLIO_CTE_CLIENT;
  cte_client->Init(clio::cte::core::kCtePoolId);
  clio::cte::core::CreateParams params;
  auto create_task = cte_client->AsyncCreate(
      clio::run::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
      clio::cte::core::kCtePoolId, params);
  create_task.Wait();
  std::this_thread::sleep_for(50ms);

  clio::run::PoolId bdev_pool_id(951, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("mw_roundtrip_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, bdev_capacity_bytes);
  bdev_create.Wait();
  std::this_thread::sleep_for(50ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "mw_roundtrip_ram", clio::run::bdev::BdevType::kRam, bdev_capacity_bytes,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  std::this_thread::sleep_for(50ms);
}

}  // namespace

int main(int argc, char *argv[]) {
  const clio::run::u32 gpu_id = 0;
  const clio::run::u64 page_size = 64 * 1024;  // 64 KiB
  const clio::run::u64 num_pages = 8;
  const clio::run::u64 total = page_size * num_pages;  // 512 KiB "weights"
  const std::string tag = "gemma-4-E2B";  // tag = model name
  (void)argc; (void)argv;

  // 1. Generate a deterministic weight file.
  std::string tmp = std::getenv("TEMP") ? std::getenv("TEMP") : ".";
  std::string path = tmp + "/mw_roundtrip_weights.bin";
  std::vector<uint8_t> src(total);
  for (clio::run::u64 i = 0; i < total; ++i) {
    src[i] = static_cast<uint8_t>((i * 131u + 7u) & 0xFFu);
  }
  {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(src.data()), total);
  }
  std::fprintf(stderr, "[MW] wrote %llu B weights to %s\n",
               (unsigned long long)total, path.c_str());

  EnsureInit(/*bdev_capacity_bytes=*/std::max<clio::run::u64>(64ULL << 20,
                                                              total * 8));

  // 2. Assimilate the file into tag via the CAE ModelWeightsAssimilator
  //    (isolated in a plain-C++ TU; see clio_mw_assimilate.cc).
  if (mw_assimilate(path, tag, page_size, /*nblocks=*/1) != 0) {
    std::fprintf(stderr, "[MW] FAIL: assimilation failed\n");
    return 1;
  }

  // 3. Map the tag with the GPU vector and read every byte back via read_range.
  //    gpu_pages_per_block >= num_pages so the stripe fits (no eviction);
  //    allow_cold_miss_fault=true so cold pages fault from CTE synchronously.
  gv::Vector<uint8_t> vec(tag, /*nblocks=*/1, gpu_id,
                          /*gpu_pages_per_block=*/(clio::run::u32)num_pages,
                          /*host_pages_per_block=*/0, page_size,
                          /*cache_period_us=*/0, gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true);

  cudaStream_t stream = nullptr;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  uint8_t *dout = nullptr;
  cudaMalloc(&dout, total);
  cudaMemset(dout, 0, total);

  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(gpu_id);
  ReadAllKernel<<<1, 32, 0, stream>>>(gpu_info, vec.Device(), dout, total);
  cudaStreamSynchronize(stream);

  std::vector<uint8_t> got(total, 0);
  cudaMemcpy(got.data(), dout, total, cudaMemcpyDeviceToHost);

  // 4. Verify.
  clio::run::u64 mismatches = 0, first_bad = total;
  for (clio::run::u64 i = 0; i < total; ++i) {
    if (got[i] != src[i]) {
      if (mismatches < 8) {
        std::fprintf(stderr, "[MW] mismatch @%llu: got %u want %u\n",
                     (unsigned long long)i, got[i], src[i]);
      }
      if (first_bad == total) first_bad = i;
      ++mismatches;
    }
  }

  cudaFree(dout);
  cudaStreamDestroy(stream);

  if (mismatches == 0) {
    std::fprintf(stderr,
                 "[MW] PASS: all %llu bytes read through the vector match the "
                 "assimilated tag '%s'\n",
                 (unsigned long long)total, tag.c_str());
    return 0;
  }
  std::fprintf(stderr, "[MW] FAIL: %llu/%llu bytes wrong (first @%llu)\n",
               (unsigned long long)mismatches, (unsigned long long)total,
               (unsigned long long)first_bad);
  return 1;
}

#endif  // !CTP_IS_DEVICE_PASS

#else  // no CUDA/ROCm

int main() { return 0; }

#endif
