/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The GPU-only paging path, asserted rather than assumed.
 *
 * Every other gpu_vector test runs against a host RAM tier with a CPU codec
 * and a blocking (spinning) fault path. That combination cannot fail the way
 * the GPU one does, so it proved nothing about the configuration that matters:
 *
 *   1. the CTE tier is kHbm, and kHbm is real device memory
 *   2. there is NO host tier at all, so data cannot quietly relocate
 *   3. the codec is nvcomp, not a CPU stand-in
 *   4. decompression takes a device pointer in and a device pointer out --
 *      nothing is staged through the host
 *   5. the fault path YIELDS instead of spinning, which is what makes 3 and 4
 *      reachable: a resident kernel blocks every later launch in its context,
 *      including the codec's own kernels
 *
 * This test is separate from test_gpu_vector_compress on purpose. CLIO_INIT
 * runs once per PROCESS and the storage tiers are fixed at that moment, so an
 * HBM-only configuration cannot be a second TEST_CASE in a binary that already
 * initialized a RAM-tier runtime -- it needs its own process.
 *
 * Decompression is launched by the CPU and batched by the compressor, so this
 * needs only the ordinary host nvcomp -- there is no device-API dependency to
 * skip on.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <chrono>
#include <fstream>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;

namespace {

constexpr clio::run::u64 kPageBytes = 64 * 1024;
constexpr clio::run::u64 kPageElems = kPageBytes / sizeof(clio::run::u32);
const clio::run::PoolId kCompressorPool(512, 0);
const clio::run::PoolId kCorePool(513, 0);

/** nvcomp-lz4's CTE wire id (CompressionFactory's registry). */
constexpr int kNvcompLz4WireId = 11;

/**
 * A weight that actually compresses, without pretending weights are trivial.
 * A repeating small value set is what a byte codec collapses; if this were
 * random the compressed image would not fit the tier and the test would be
 * measuring the spill it exists to forbid.
 */
CTP_INLINE_CROSS_FUN clio::run::u32 Weight(clio::run::u64 i) {
  return static_cast<clio::run::u32>((i / 64) % 17) * 0x01010101u;
}

CTP_INLINE_CROSS_FUN clio::run::u32 Activation(clio::run::u64 i) {
  return static_cast<clio::run::u32>((i % 7) + 1);
}

}  // namespace

/** Seed the weights through the YIELDING fault path. */
__global__ void HbmSeedKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<clio::run::u32> v,
                              clio::run::u64 per, gy::YieldableView<> yv,
                              gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(clio::run::u64, off, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, run, 0);
  const clio::run::u64 base = static_cast<clio::run::u64>(yv.Block()) * per;

  CLIO_YBEGIN();
  for (; off < per; off += kPageElems) {
    CLIO_YCALL(v.HoldPageYield(
        base + off, (off + kPageElems <= per) ? kPageElems : (per - off),
        &run));
    {
      const clio::run::u64 n =
          (off + kPageElems <= per) ? kPageElems : (per - off);
      for (clio::run::u64 i = threadIdx.x; i < n; i += blockDim.x) {
        v[base + off + i] = Weight(base + off + i);
      }
    }
  }
  // SubmitPut clears `dirty` as it submits, so a lane still writing the last
  // page when thread 0 flushes would lose its writes AND leave the page
  // looking clean.
  __syncthreads();
  if (threadIdx.x == 0) {
    v.BeginFlush(base, per);
  }
  __syncthreads();
  CLIO_YIELD_IF(v.AnyTransferInFlight());
  CLIO_YEND();
}

/** sum(w[i] * activation(i)) read back through the yielding fault path. */
__global__ void HbmDotKernel(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<clio::run::u32> v,
                             clio::run::u64 per, unsigned long long *sum,
                             gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(clio::run::u64, off, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, run, 0);
  const clio::run::u64 base = static_cast<clio::run::u64>(yv.Block()) * per;

  CLIO_YBEGIN();
  for (; off < per; off += kPageElems) {
    CLIO_YCALL(v.HoldPageYield(
        base + off, (off + kPageElems <= per) ? kPageElems : (per - off),
        &run));
    {
      const clio::run::u64 n =
          (off + kPageElems <= per) ? kPageElems : (per - off);
      unsigned long long acc = 0;
      for (clio::run::u64 i = threadIdx.x; i < n; i += blockDim.x) {
        acc += static_cast<unsigned long long>(v.at(base + off + i)) *
               Activation(base + off + i);
      }
      atomicAdd(sum, acc);
    }
  }
  CLIO_YEND();
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: nvcomp on a kHBM-only tier, decoded in-kernel",
          "[gpu_vector][hbm][nvcomp]") {
  const unsigned kBlocks = 4;
  const clio::run::u64 kPagesPerBlock = 16;   // pages each block walks
  const clio::run::u32 kSlots = 4;            // cache slots: forces eviction
  const clio::run::u64 per = kPagesPerBlock * kPageElems;
  const clio::run::u64 n = per * kBlocks;

  {
    std::ofstream cfg("gpu_vector_hbm_nvcomp.yaml");
    REQUIRE(cfg.is_open());
    // NOTE the storage section: ONE tier, and it is hbm. There is deliberately
    // no RAM tier beneath it, so "the data is on the GPU" is not something the
    // run has to achieve -- it is the only thing the configuration permits. A
    // page that cannot fit has nowhere to go and its put fails loudly.
    //
    // The tier is sized to hold the COMPRESSED image but NOT the raw one (see
    // capacity_limit below), so a run whose codec silently did not compress
    // cannot pass: its puts run out of tier and fail.
    cfg << "networking:\n  port: 9436\n\n"
        << "runtime:\n  num_threads: 4\n  queue_depth: 4096\n"
        << "  first_busy_wait: 2000000000\n\n"
        << "gpu:\n  queue_depth: 4096\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"512MB\"\n\n"
        << "  - mod_name: clio_cte_compressor\n"
        << "    pool_name: cte_compressor\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n    next_pool_id: \"513.0\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"513.0\"\n"
        << "    storage:\n"
        << "      - path: \"hbm::gv_hbm_only\"\n"
        << "        bdev_type: \"hbm\"\n"
        // 2MB against a 4MB raw image: the vector CANNOT fit here unless the
        // codec actually ran. With no tier beneath this one, a run that
        // quietly stored raw bytes fails on put_errors instead of passing.
        << "        capacity_limit: \"2MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_hbm_nvcomp.yaml",
                            1);
  }

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  gv::Vector<clio::run::u32> vec("gv_hbm_nvcomp", {0}, kPageBytes, kBlocks,
                                 kSlots, n, kCompressorPool, kNvcompLz4WireId);
  vec.EnableStats();

  // --- seed -----------------------------------------------------------
  {
    gy::Yieldable<> drv(kBlocks, 32);
    gy::YieldStack stack(kBlocks, 32, 256);
    const clio::run::u32 rounds = drv.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          HbmSeedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
              gpu_info, vec.GetDevice(0), per, view, stack.View());
        },
        [] {}, /*max_rounds=*/200000);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::fprintf(stderr, "[hbm-nvcomp] seed rounds=%u\n", rounds);
  }

  // A writeback that failed means the tier could not hold the data. With no
  // host tier configured that is the failure this test exists to catch, and it
  // used to be SILENT: the page was marked clean and its bytes dropped.
  REQUIRE(vec.ReadStats(0).put_errors == 0);

  // --- the stored-size table publishes for the encoded tag --------------
  // (The zero-copy device-tier map is gone: kernels never alias shared CTE
  // tier memory. The stored-size table is metadata -- the run-fetch's
  // existence proof -- and publishing it for every page doubles as the
  // assertion that all pages were actually stored.)
  REQUIRE(vec.PublishStoredSizes() == n / kPageElems);

  // The compressed image has to be SMALLER than the raw one, or the codec did
  // not run and everything above passed for the wrong reason.
  clio::run::u64 stored = 0;
  {
    clio::cte::core::Client core(kCorePool);
    for (clio::run::u64 p = 0; p < n / kPageElems; ++p) {
      char name[32];
      gv::PageBlobName(p, name);
      auto sz = core.AsyncGetBlobSize(vec.TagId(), name);
      sz.Wait();
      if (sz->GetReturnCode() == 0) stored += sz->size_;
    }
  }
  const clio::run::u64 logical = n * sizeof(clio::run::u32);
  std::fprintf(stderr, "[hbm-nvcomp] logical=%lluB stored=%lluB ratio=%.1fx\n",
               (unsigned long long) logical, (unsigned long long) stored,
               stored ? (double) logical / (double) stored : 0.0);
  REQUIRE(stored > 0);
  REQUIRE(stored < logical);

  // --- read back and check the arithmetic -------------------------------
  unsigned long long *d_sum = nullptr;
  REQUIRE(cudaMalloc(&d_sum, sizeof(unsigned long long)) == cudaSuccess);
  REQUIRE(cudaMemset(d_sum, 0, sizeof(unsigned long long)) == cudaSuccess);
  {
    gy::Yieldable<> drv(kBlocks, 32);
    gy::YieldStack stack(kBlocks, 32, 256);
    const clio::run::u32 rounds = drv.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          HbmDotKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
              gpu_info, vec.GetDevice(0), per, d_sum, view, stack.View());
        },
        [] {}, /*max_rounds=*/200000);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::fprintf(stderr, "[hbm-nvcomp] read rounds=%u\n", rounds);
  }

  unsigned long long got = 0;
  REQUIRE(cudaMemcpy(&got, d_sum, sizeof(got), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  unsigned long long want = 0;
  for (clio::run::u64 i = 0; i < n; ++i) {
    want += static_cast<unsigned long long>(Weight(i)) * Activation(i);
  }
  std::fprintf(stderr, "[hbm-nvcomp] got=%llu want=%llu\n", got, want);
  REQUIRE(got == want);
  REQUIRE(vec.ReadStats(0).put_errors == 0);

  cudaFree(d_sum);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS
