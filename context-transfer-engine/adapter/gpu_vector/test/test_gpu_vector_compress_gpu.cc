/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Compressed GPU-vector unit test (CUDA / ROCm).
 *
 * Proves the *compressed* gpu_vector: page traffic is routed through the
 * compressor chimod, which compresses in HBM (cuSZp / whatever
 * CLIO_CTE_COMPRESS_LIB pins) and forwards the compressed blob to the CTE
 * core, and decompresses on the fault path -- entirely transparently to the
 * Vector<T> device API.
 *
 * Topology (compose config, written below):
 *
 *     Vector page PutBlob/GetBlob  ->  compressor (pool 600)
 *                                        |  compress in HBM / decompress
 *                                        v
 *                                      CTE core   (pool 512 == kCtePoolId)
 *                                        |
 *                                        v
 *                                      RAM bdev   (pool 301)
 *
 * The Vector creates its CTE *tag* on kCtePoolId (the core) as always, but is
 * constructed with storage_pool_id = 600 so per-page evictions/faults are
 * routed through the compressor. The compressor's next_pool_id_ (from compose)
 * is 512, so it forwards compressed blobs to the same core that owns the tag.
 *
 * Flow:
 *   Phase A (writer):  Vector<float> v; kernel writes a smooth field;
 *                      FlushAllSync() -> every dirty page is PUT through the
 *                      compressor (compressed in HBM) -> stored in core.
 *                      Destroy the writer (drops all resident HBM pages).
 *   Phase B (reader):  a *fresh* Vector<float> on the SAME tag/geometry;
 *                      the read kernel cold-faults every page -> GetBlob
 *                      through the compressor -> decompressed into HBM.
 *   Verify: |readback - expected| <= 2e-3 (cuSZp abs error bound is 1e-3).
 *
 * The two-instance split guarantees the read path actually FAULTS (and thus
 * decompresses) instead of reading pages still resident from the write.
 *
 * Large pages (256 KiB) are required: cuSZp only round-trips correctly for
 * blobs >= ~256 KiB (small-blob correctness bug, see the putblob bench).
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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace {

bool g_initialized = false;

// Compressor entrypoint pool the Vector routes page traffic through. Distinct
// from the core (kCtePoolId == 512) so tag ops keep hitting the core.
inline clio::run::PoolId StoragePool() { return clio::run::PoolId(600, 0); }

/** Smooth, bounded field: good compression ratio, magnitude ~[0.1, 0.9] so a
 *  1e-3 absolute error bound is meaningful. */
__host__ __device__ inline float Expected(clio::run::u64 i) {
  return 0.5f + 0.4f * sinf(static_cast<float>(i) * 1.0e-4f);
}

/**
 * Bring up the CLIO Runtime with a compose config that places the compressor
 * (pool 600) in front of the CTE core (pool 512) + RAM bdev (pool 301).
 * Gated to the host pass.
 */
void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  // Distributed reuse: when CLIO_GV_EXTERNAL_CONF is set, the harness has already
  // provided CLIO_SERVER_CONF (a multi-node compose: compressor 600 -> a cte_core
  // 512 that spans the cluster, + hostfile). Use it as-is so this exact Vector<T>
  // write/evict/fault/read test runs over the DISTRIBUTED store. Single-node
  // ctest never sets it, so that path is unchanged.
  if (std::getenv("CLIO_GV_EXTERNAL_CONF")) {
    std::fprintf(stderr, "[INIT] gpu_vector compressed: external distributed "
                         "conf=%s\n",
                 std::getenv("CLIO_SERVER_CONF"));
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
    std::this_thread::sleep_for(2s);  // let compose pools + cluster settle
    g_initialized = true;
    return;
  }
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10520;
  std::string cfg_path = "/tmp/gpu_vec_compress_" + std::to_string(port) + ".yaml";
  {
    std::ofstream cfg(cfg_path);
    cfg << "networking:\n  port: " << port << "\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 65536\n\n"
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
        << "    storage:\n      - path: \"ram::cte_ram_tier1\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"2048MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", cfg_path.c_str(), 1);
  std::fprintf(stderr, "[INIT] gpu_vector compressed: compose=%s\n",
               cfg_path.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(1s);  // let compose pools initialize
  g_initialized = true;
#endif
}

}  // namespace

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

/** Write v[i] = Expected(i) across a page-aligned stripe per block. */
__global__ void GpuVecCompressWriteKernel(clio::run::IpcManagerGpuInfo info,
                                          gv::DeviceView<float> view,
                                          clio::run::u64 total) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  clio::run::u64 stripe = (total + gridDim.x - 1) / gridDim.x;
  clio::run::u64 lo = static_cast<clio::run::u64>(blockIdx.x) * stripe;
  clio::run::u64 hi = lo + stripe;
  if (hi > total) hi = total;
  v.write_range(lo, hi, [](clio::run::u64 i) { return Expected(i); });
  (void)g_ipc_manager;
}

/** Read v[i] back into result[i]. Cold-faults every page. */
__global__ void GpuVecCompressReadKernel(clio::run::IpcManagerGpuInfo info,
                                         gv::DeviceView<float> view,
                                         float *result, clio::run::u64 total) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  clio::run::u64 stripe = (total + gridDim.x - 1) / gridDim.x;
  clio::run::u64 lo = static_cast<clio::run::u64>(blockIdx.x) * stripe;
  clio::run::u64 hi = lo + stripe;
  if (hi > total) hi = total;
  v.read_range(lo, hi, [result](clio::run::u64 i, float val) {
    result[i] = val;
  });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: compressed write/evict then fault/read round-trip",
          "[gpu_vector][compress][cte][stress]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;

  const clio::run::u32 nblocks = 2;
  const clio::run::u32 pages_per_block = 2;
  // 256 KiB pages: cuSZp only round-trips correctly at this size or larger.
  const clio::run::u64 page_size_bytes = 256ULL * 1024;
  const clio::run::u64 elems_per_page = page_size_bytes / sizeof(float);
  const clio::run::u64 total =
      static_cast<clio::run::u64>(nblocks) * pages_per_block * elems_per_page;
  const char *tag = "gpu_vec_compress_tag";
  clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(0);

  std::fprintf(stderr,
               "[GPUVEC-C] nblocks=%u ppb=%u page=%lluB elems/page=%llu "
               "total=%llu (%.1f MiB)\n",
               nblocks, pages_per_block, (unsigned long long)page_size_bytes,
               (unsigned long long)elems_per_page, (unsigned long long)total,
               (double)(total * sizeof(float)) / (1024.0 * 1024.0));

  // ---- Phase A: writer vector -> flush (compress) -> destroy ----
  //
  // kAsync mode: the async cache-manager kernel flushes every dirty page on its
  // tick period (Phase 3), routing each page PutBlob to the compressor, which
  // compresses it IN HBM and stores the compressed blob in the core. The write
  // kernel runs on the default stream; the manager runs on its own non-blocking
  // stream and issues the puts. cuSZp's compression kernels then run on an idle
  // GPU during the host sleep below (the write kernel has finished, and the
  // manager's flush kernel EXITS after issuing the async puts -- it does not
  // spin-wait), so there is no manager/compressor GPU contention on the write
  // path. We sleep long enough (several tick periods) for all pages to flush
  // and the puts to complete before destroying the writer.
  {
    gv::Vector<float> vec(tag, nblocks, /*gpu_id=*/0, pages_per_block,
                          /*host_pages_per_block=*/0, page_size_bytes,
                          /*cache_period_us=*/20000, gv::CacheMode::kAsync,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/StoragePool());
    auto view = vec.Device();
    GpuVecCompressWriteKernel<<<nblocks, 32>>>(gpu_info, view, total);
    ctp::GpuApi::Synchronize();
    std::this_thread::sleep_for(3s);  // let the manager flush + compress all pages
    std::fprintf(stderr, "[GPUVEC-C] wrote + flushed (compressed) %llu elems\n",
                 (unsigned long long)total);
  }  // writer destroyed: all resident HBM pages dropped

  // ---- Phase B: DEVICE-SIDE read of the compressed vector ----
  //
  // A fresh reader Vector on the same tag/geometry. An on-device page fault
  // (read_range cold-miss) would DEADLOCK on a single GPU: the fault kernel
  // spin-waits on-device for its GetBlob, but the compressor services it by
  // launching cuSZp's decompression kernel -- which internally cudaMalloc/
  // cudaMemcpy-synchronizes the device -- so it cannot make progress while the
  // fault kernel monopolizes the GPU. (Uncompressed faults are fine: a CPU
  // memcpy services them, no GPU kernel.)
  //
  // FaultAllSync() sidesteps this: it decompresses every stored page into its
  // HBM slot from the HOST while the GPU is idle (no spin-waiting kernel to
  // starve the decompressor), then marks the slots resident. The device read
  // kernel below then reads resident HBM pages -- a genuine device-side read of
  // compressed-at-rest data, no on-device fault, no deadlock.
  auto *result = ctp::GpuApi::MallocHost<float>(total * sizeof(float));
  REQUIRE(result != nullptr);
  std::memset(result, 0, total * sizeof(float));
  // Two read modes. DEFAULT: the read kernel cold-faults every page ON-DEVICE.
  // This deadlocked under GPU compression until the compressor was moved to a
  // dedicated CUDA context (A100 compute preemption lets its decompress run
  // while the fault kernel spins; cross-context UVA writes into the HBM page) --
  // so the transparent on-access fault now works and is the default guard here.
  // CLIO_GV_ONDEVICE_FAULT=0 selects the host-orchestrated FaultAllSync path.
  const char *ondev = std::getenv("CLIO_GV_ONDEVICE_FAULT");
  const bool on_device_fault = !ondev || ondev[0] != '0';
  {
    gv::Vector<float> vec(tag, nblocks, /*gpu_id=*/0, pages_per_block,
                          /*host_pages_per_block=*/0, page_size_bytes,
                          /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/StoragePool());
    auto view = vec.Device();
    if (on_device_fault) {
      std::fprintf(stderr, "[GPUVEC-C] ON-DEVICE fault path (no FaultAllSync)\n");
      GpuVecCompressReadKernel<<<nblocks, 32>>>(gpu_info, view, result, total);
      ctp::GpuApi::Synchronize();
    } else {
      vec.FaultAllSync();  // host-decompress every page into HBM, mark resident
      std::fprintf(stderr, "[GPUVEC-C] FaultAllSync: materialized all pages\n");
      GpuVecCompressReadKernel<<<nblocks, 32>>>(gpu_info, view, result, total);
      ctp::GpuApi::Synchronize();
    }
  }

  // ---- Verify within cuSZp's absolute error bound ----
  double max_abs_err = 0.0, mean_abs_err = 0.0;
  clio::run::u64 first_bad = total;
  for (clio::run::u64 i = 0; i < total; ++i) {
    double e = std::fabs((double)result[i] - (double)Expected(i));
    mean_abs_err += e;
    if (e > max_abs_err) max_abs_err = e;
    if (e > 2.0e-3 && first_bad == total) first_bad = i;
  }
  mean_abs_err /= (double)total;
  std::fprintf(stderr,
               "[GPUVEC-C] device read %llu elems  max_abs_err=%.3e "
               "mean_abs_err=%.3e (eb=1e-3)\n",
               (unsigned long long)total, max_abs_err, mean_abs_err);
  if (first_bad != total) {
    std::fprintf(stderr, "[GPUVEC-C] first out-of-bound at %llu: got %.6f exp %.6f\n",
                 (unsigned long long)first_bad, result[first_bad],
                 Expected(first_bad));
  }
  REQUIRE(max_abs_err <= 2.0e-3);
  std::fprintf(stderr,
               "[GPUVEC-C] PASS: compressed GPU-vector device round-trips "
               "%llu elems within error bound\n",
               (unsigned long long)total);

  ctp::GpuApi::FreeHost(result);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
