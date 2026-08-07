/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * LARGER THAN GPU MEMORY: one long-running kernel reads an entire compressed
 * dataset bigger than the GPU's VRAM through read_range.
 *
 * This is the end-state claim of the gpu_vector design: the dataset does not
 * fit on the GPU, is not resident, is stored compressed, and a single kernel
 * still walks all of it, with the cache demand-faulting and evicting the whole
 * way. Default size is 20 GiB against a 16 GiB-class card; override with
 * CLIO_TEST_HUGE_GB.
 *
 * VERIFICATION BY CHECKSUM, NOT BY BUFFER. The obvious "write every byte to an
 * output array and memcmp" does not scale here: a 20 GiB pinned host buffer on
 * a 32 GiB machine is not viable, and allocating it would test the allocator
 * rather than the vector. Instead the kernel folds every byte it reads into a
 * position-weighted 64-bit sum, and the host computes the same sum
 * analytically. The weighting makes the sum sensitive to a byte landing at the
 * WRONG index (a plain sum would not catch transposed pages), and it is
 * commutative -- which matters because read_range consumes a page across 32
 * lanes in parallel, so the accumulation order is not deterministic.
 *
 * STORAGE TIER. 20 GiB will not fit in a RAM bdev on a 32 GiB host, so the
 * backing tier is a FILE bdev. That is also the honest configuration: a dataset
 * this size lives on storage, and the fault path is what pulls it up.
 *
 * COST. This test writes and then reads tens of GiB through the compressor. It
 * is minutes, not seconds, and is labelled "huge" so it can be excluded from
 * ordinary runs.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <clio_ctp/util/gpu_api.h>
#include <clio_ctp/introspect/system_info.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace gv  = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

namespace {

bool g_initialized = false;

// 1 MiB pages: at 20 GiB a 64 KiB page would mean ~330k blobs, which measures
// per-blob task overhead rather than the fault path.
constexpr clio::run::u64 kPageSizeBytes = 1024 * 1024;
// 64 MiB of HBM cache. Tiny next to the extent on purpose.
constexpr clio::run::u32 kCachePages = 64;

/** The compressor chimod's pool (compose: pool_id 600.0,
 *  next_pool_id 512.0). Page traffic routed here is compressed. */
inline clio::run::PoolId CompressorPool() {
  return clio::run::PoolId(600, 0);
}

clio::run::u64 HugeBytes() {
  const char *env = std::getenv("CLIO_TEST_HUGE_GB");
  double gb = env ? std::atof(env) : 20.0;
  if (gb <= 0) gb = 20.0;
  clio::run::u64 bytes = static_cast<clio::run::u64>(gb * 1024.0 * 1024.0 * 1024.0);
  // Whole pages only.
  return (bytes / kPageSizeBytes) * kPageSizeBytes;
}

void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10646;
  std::string cfg_path = "/tmp/gpu_vec_huge_" + std::to_string(port) + ".yaml";
  const char *dir_env = std::getenv("CLIO_TEST_HUGE_DIR");
  std::string spill = (dir_env ? std::string(dir_env) : std::string("/tmp")) +
                      "/clio_huge_tier.bin";
  {
    std::ofstream cfg(cfg_path);
    cfg << "networking:\n  port: " << port << "\n\n"
        << "runtime:\n  num_threads: 16\n  queue_depth: 65536\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"2048MB\"\n\n"
        << "  - mod_name: clio_cte_compressor\n"
        << "    pool_name: cte_compressor\n    pool_query: local\n"
        << "    pool_id: \"600.0\"\n    next_pool_id: \"512.0\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n    pool_id: \"512.0\"\n"
        // FILE tier: the extent is far larger than host RAM.
        << "    storage:\n      - path: \"" << spill << "\"\n"
        << "        bdev_type: \"file\"\n"
        << "        capacity_limit: \"131072MB\"\n"
        << "        score: 0.5\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", cfg_path.c_str(), 1);
  if (std::getenv("CLIO_CTE_COMPRESS_LIB") == nullptr) {
    ctp::SystemInfo::Setenv("CLIO_CTE_COMPRESS_LIB", "zstd", 1);
  }
  std::fprintf(stderr, "[INIT] huge: compose=%s tier=%s lib=%s\n",
               cfg_path.c_str(), spill.c_str(),
               std::getenv("CLIO_CTE_COMPRESS_LIB"));
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  std::this_thread::sleep_for(1s);
  g_initialized = true;
#endif
}

}  // namespace

/** Position-weighted, order-independent fold. Must match HostChecksum(). */
CTP_GPU_FUN clio::run::u64 FoldByte(clio::run::u64 i, unsigned char b) {
  return static_cast<clio::run::u64>(b) *
         (static_cast<clio::run::u64>(i) * 2654435761ULL + 1ULL);
}

/**
 * Walk the ENTIRE extent in one launch, folding every byte into `sum`.
 * Nothing is written back, so host memory is irrelevant to the dataset size.
 */
__global__ void HugeChecksumKernel(clio::run::IpcManagerGpuInfo info,
                                   gv::DeviceView<uint8_t> view,
                                   unsigned long long *sum,
                                   clio::run::u64 total) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<uint8_t> v(view, g_ipc_manager_ptr);
  if (blockIdx.x != 0) return;
  v.read_range(0, total, [sum](clio::run::u64 i, uint8_t b) {
    atomicAdd(sum, static_cast<unsigned long long>(FoldByte(i, b)));
  });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

namespace {

/** Same compressible-but-real payload shape as the compressed fault test. */
uint8_t PayloadByte(clio::run::u64 i) {
  return static_cast<uint8_t>(((i / 64) & 0x3F) + ((i * 7u) & 0x0F));
}

clio::run::u64 HostChecksum(clio::run::u64 total) {
  clio::run::u64 acc = 0;
  for (clio::run::u64 i = 0; i < total; ++i) {
    acc += static_cast<clio::run::u64>(PayloadByte(i)) *
           (i * 2654435761ULL + 1ULL);
  }
  return acc;
}

void PutHugePages(const std::string &tag, clio::run::u32 npages) {
  gv::Vector<uint8_t> vec(tag, /*nblocks=*/1, /*gpu_id=*/0, kCachePages,
                          /*host_pages_per_block=*/0, kPageSizeBytes,
                          /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/CompressorPool());
  auto tag_id = vec.TagId();
  // PutBlob through the COMPRESSOR pool (600), not the core: the compressor
  // compresses and forwards to its next_pool_id (the core, 512), which is
  // also where the tag lives. Going straight to the core -- as this test
  // originally did -- stores the pages verbatim and silently tests nothing.
  clio::cte::core::Client cte_c(CompressorPool());
  auto *cte = &cte_c;
  auto t0 = std::chrono::steady_clock::now();
  // Track the actual shrink: without this the test cannot distinguish real
  // compression from a silently pass-through compressor.
  clio::run::u64 orig_total = 0, comp_total = 0;

  for (clio::run::u32 p = 0; p < npages; ++p) {
    auto buf = CLIO_IPC->AllocateBuffer(kPageSizeBytes);
    auto *bytes = reinterpret_cast<uint8_t *>(buf.ptr_);
    const clio::run::u64 base = static_cast<clio::run::u64>(p) * kPageSizeBytes;
    for (clio::run::u64 j = 0; j < kPageSizeBytes; ++j) {
      bytes[j] = PayloadByte(base + j);
    }
    clio::cte::core::Context ctx;
    ctx.dynamic_compress_ = 1;  // static: use the pinned lossless lib
    ctx.compress_preset_ = 1;   // FAST: this is a lot of bytes
    std::string blob_name = tag + "_b0_pi" + std::to_string(p);
    auto task = cte->AsyncPutBlob(tag_id, blob_name, 0, kPageSizeBytes,
                                  buf.shm_.template Cast<void>(), 1.0f, ctx, 0);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
    orig_total += task->context_.actual_original_size_;
    comp_total += task->context_.actual_compressed_size_;
    if ((p % 1024) == 0) {
      auto dt = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr, "[HUGE] put %u/%u pages (%.1f s)\n", p, npages, dt);
    }
  }
  std::fprintf(stderr,
               "[HUGE] stored: original=%.2f GiB compressed=%.2f GiB (%.2fx)\n",
               orig_total / 1073741824.0, comp_total / 1073741824.0,
               comp_total ? (double) orig_total / (double) comp_total : 0.0);
  // Prove the compressor ran and shrank the data, else the read-back below
  // says nothing about compressed faulting.
  REQUIRE(orig_total > 0);
  REQUIRE(comp_total > 0);
  REQUIRE(comp_total < orig_total);
}

}  // namespace

TEST_CASE("gpu_vector: compressed dataset LARGER THAN VRAM read fully by one kernel",
          "[gpu_vector][cte][compress][fault][lossless][huge]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  auto gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  const clio::run::u64 total = HugeBytes();
  const clio::run::u32 npages =
      static_cast<clio::run::u32>(total / kPageSizeBytes);

  size_t vram_free = 0, vram_total = 0;
  cudaMemGetInfo(&vram_free, &vram_total);
  std::fprintf(stderr,
               "[HUGE] dataset=%.2f GiB, VRAM total=%.2f GiB, cache=%.0f MiB "
               "(%u pages of %llu KiB), %.1fx oversubscribed\n",
               total / 1073741824.0, vram_total / 1073741824.0,
               (kCachePages * kPageSizeBytes) / 1048576.0, npages,
               (unsigned long long)(kPageSizeBytes / 1024),
               static_cast<double>(npages) / kCachePages);
  // The whole point: the dataset must not fit in VRAM.
  REQUIRE(total > static_cast<clio::run::u64>(vram_total));

  const std::string tag = "gpu_vector_huge_compress";
  PutHugePages(tag, npages);

  unsigned long long *dsum = nullptr;
  REQUIRE(cudaMalloc(&dsum, sizeof(unsigned long long)) == cudaSuccess);
  REQUIRE(cudaMemset(dsum, 0, sizeof(unsigned long long)) == cudaSuccess);
  {
    gv::Vector<uint8_t> vec(tag, /*nblocks=*/1, /*gpu_id=*/0, kCachePages,
                            /*host_pages_per_block=*/0, kPageSizeBytes,
                            /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                            /*manager_threads_per_block=*/32,
                            /*allow_cold_miss_fault=*/true,
                            /*storage_pool_id=*/CompressorPool());
    auto t0 = std::chrono::steady_clock::now();
    HugeChecksumKernel<<<1, 32>>>(gpu_info, vec.Device(), dsum, total);

    // Progress watch: a dataset this size takes minutes, and a silent hang is
    // indistinguishable from slow progress without this.
    unsigned long long last = ~0ULL;
    int stagnant = 0;
    for (int t = 0; t < 2400 && stagnant < 40; ++t) {
      std::this_thread::sleep_for(1s);
      auto s = vec.StatsSnapshot();
      stagnant = (s.resolve_total == last) ? stagnant + 1 : 0;
      if (stagnant == 0 && (t % 15) == 0) {
        std::fprintf(stderr, "[HUGE] t=%4ds faulted %llu/%u pages (%.1f%%)\n",
                     t, s.resolve_fault_get, npages,
                     100.0 * s.resolve_fault_get / npages);
      }
      last = s.resolve_total;
    }
    ctp::GpuApi::Synchronize();
    auto dt = std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - t0).count();

    auto s = vec.StatsSnapshot();
    std::fprintf(stderr,
                 "[HUGE] done in %.1f s: resolve=%llu cold=%llu fault=%llu "
                 "hits=%llu (%.1f MiB/s effective)\n",
                 dt, s.resolve_total, s.resolve_cold_miss, s.resolve_fault_get,
                 s.resolve_hits, (total / 1048576.0) / (dt > 0 ? dt : 1));
    REQUIRE(s.resolve_fault_get == npages);
    REQUIRE(s.resolve_hits == 0);
  }

  unsigned long long got = 0;
  REQUIRE(cudaMemcpy(&got, dsum, sizeof(got), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  cudaFree(dsum);

  const clio::run::u64 want = HostChecksum(total);
  std::fprintf(stderr, "[HUGE] checksum got=%llu want=%llu\n",
               (unsigned long long) got, (unsigned long long) want);
  REQUIRE(got == want);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
