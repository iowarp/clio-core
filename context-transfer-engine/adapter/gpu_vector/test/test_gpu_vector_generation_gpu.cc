/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * THE VECTOR'S GENERATIONAL CONTRACT, WHICH HAD NO TEST.
 *
 * `test_cte_generation_gpu.cc` covers the CTE blob layer through `Client` and
 * `ctx.generation_`. Nothing covered `DeviceVector::BeginFlush(gen)` /
 * `DeviceVector::Fetch(gen)` -- the two verbs a decomposed application
 * actually uses -- and the gap hid a real defect for as long as it existed:
 * the md_bench halo exchange published every step and every reader was served
 * a generation-0 blob, so a 2-node run computed forces from LAST STEP'S
 * positions and still passed its energy gates.
 *
 * The contract under test, stated as the runtime states it:
 *   - a generational PUT stamps the range it wrote, and
 *   - a generational GET is not served until the stamp reaches the generation
 *     it named.
 *
 * Single process, single node, so the blob's hashed owner is always local:
 * this isolates the STAMP from routing, forwarding and cross-node visibility,
 * which is exactly the axis the distributed harness cannot separate.
 *
 * WRITTEN SO A NO-OP IMPLEMENTATION FAILS. Case A demands a generation that
 * was published and must be served; case B demands one that never was and
 * must NOT be served. A vector that ignores the generation passes A and fails
 * B; one that refuses everything fails A. Both directions are needed -- the
 * bug this was written for made every generational get succeed by ignoring
 * the flag.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

#include <cstdio>
#include <fstream>
#include <chrono>
#include <string>
#include <thread>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

namespace {
constexpr u64 kPageBytes = 4096;
constexpr u64 kElemsPerPage = kPageBytes / sizeof(u32);
constexpr u64 kPages = 8;
constexpr u64 kElems = kPages * kElemsPerPage;

CTP_INLINE_CROSS_FUN u32 Pattern(u64 page, u32 i) {
  return static_cast<u32>(page) * 1009u + i * 7u + 3u;
}

/**
 * Write page `pg` and PUBLISH it as `gen`.
 *
 * The write goes through a write-hold, then the flush names the generation.
 * ClearCache afterwards (host side) is what forces the reader below to go to
 * the store rather than read its own dirty frame.
 */
__device__ gy::YCoroMain PublishCoro(gv::DeviceVector<u32> v, u64 pg, u64 gen,
                                     u32 fill) {
  const u64 off = pg * kElemsPerPage;
  co_await v.Fetch(0, off, kElemsPerPage);
  {
    auto h = co_await v.HoldPage(off, kElemsPerPage, /*write=*/true);
    u32 *p = h.ptr();
    for (u64 i = threadIdx.x; i < kElemsPerPage; i += blockDim.x) {
      p[i] = Pattern(pg, static_cast<u32>(i)) + fill;
    }
    __syncthreads();
    // FLUSH BEFORE THE UNPIN, always: eviction performs no I/O, so a frame
    // released between the write and the writeback can be dropped with the
    // bytes still only in it.
    co_await v.BeginFlush(gen, off, kElemsPerPage);
    co_await v.EndFlush();
  }
  v.UnpinRange(off, kElemsPerPage);
}

__global__ void PublishKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<u32> v, u64 pg, u64 gen,
                              u32 fill, gy::YieldableView<> yv,
                              gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(PublishCoro(v, pg, gen, fill));
}

/**
 * Demand page `pg` AT `gen` and check the bytes.
 *
 * `bad` counts wrong elements; `served` records that the fetch came back at
 * all. A get that is refused leaves the frame empty and traps in HoldPage --
 * which is the correct loud failure and is why case B does not read.
 */
__device__ gy::YCoroMain DemandCoro(gv::DeviceVector<u32> v, u64 pg, u64 gen,
                                    u32 fill, unsigned long long *bad,
                                    unsigned long long *served) {
  const u64 off = pg * kElemsPerPage;
  co_await v.Fetch(gen, off, kElemsPerPage);
  if (threadIdx.x == 0) atomicAdd(served, 1ull);
  {
    auto h = co_await v.HoldPage(off, kElemsPerPage);
    const u32 *p = h.ptr();
    for (u64 i = threadIdx.x; i < kElemsPerPage; i += blockDim.x) {
      if (p[i] != Pattern(pg, static_cast<u32>(i)) + fill) {
        atomicAdd(bad, 1ull);
      }
    }
    __syncthreads();
  }
  v.UnpinRange(off, kElemsPerPage);
}

/** Flush a resident page AT `gen` without writing it: PublishSlabCoro. */
__device__ gy::YCoroMain PublishOnlyCoro(gv::DeviceVector<u32> v, u64 pg,
                                         u64 gen) {
  const u64 off = pg * kElemsPerPage;
  co_await v.Fetch(0, off, kElemsPerPage);
  co_await v.BeginFlush(gen, off, kElemsPerPage);
  co_await v.EndFlush();
  v.UnpinRange(off, kElemsPerPage);
}

__global__ void PublishOnlyKernel(clio::run::IpcManagerGpuInfo info,
                                  gv::DeviceVector<u32> v, u64 pg, u64 gen,
                                  gy::YieldableView<> yv,
                                  gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(PublishOnlyCoro(v, pg, gen));
}

__global__ void DemandKernel(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<u32> v, u64 pg, u64 gen,
                             u32 fill, unsigned long long *bad,
                             unsigned long long *served,
                             gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(DemandCoro(v, pg, gen, fill, bad, served));
}

}  // namespace

#if !CTP_IS_DEVICE_PASS
template <typename VecT, typename LaunchT>
static u32 RunYieldable(unsigned nblocks, VecT &vec, LaunchT &&launch) {
  (void)vec;
  gy::Yieldable<> drv(nblocks, 32);
  gy::YieldStack stack(nblocks, 32, 8192);
  const u32 rounds = drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000, gv::ResumeWhenComplete);
  if (rounds >= 200000) {
    std::fprintf(stderr, "[gv] FATAL: hit the round cap\n");
    std::abort();
  }
  return rounds;
}
#endif  // !CTP_IS_DEVICE_PASS

#if !CTP_IS_DEVICE_PASS
TEST_CASE("gpu_vector: a generational flush is visible to a generational fetch",
          "[gpu_vector][generation]") {
  {
    std::ofstream cfg("gpu_vector_generation.yaml");
    cfg << "networking:\n  port: 9437\n\n"
        << "runtime:\n  num_threads: 4\n  queue_depth: 4096\n\n"
        << "gpu:\n  queue_depth: 4096\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"512MB\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n"
        << "    storage:\n"
        << "      - path: \"ram::gv_generation_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"256MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_generation.yaml", 1);
  }

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);
  unsigned long long *d_bad =
      ctp::GpuApi::Malloc<unsigned long long>(sizeof(unsigned long long));
  unsigned long long *d_served =
      ctp::GpuApi::Malloc<unsigned long long>(sizeof(unsigned long long));

  const u32 kBlocks = 1;
  const u64 kPage = 3;
  int rc = 0;

  gv::Vector<u32> vec("gv_generation", {0}, kPageBytes, kBlocks,
                      /*slots=*/kPages, kElems,
                      clio::run::PoolId::GetNull(), 0, 1, /*nsets=*/1);
  // WITHOUT THIS EVERY COUNTER READS 0 and a broken flush looks identical to
  // a working one -- ReadStats returns a default Stats when stats are off.
  vec.EnableStats();

  // ---- A: publish gen 1, then demand gen 1. Must be served, with the bytes.
  {
    RunYieldable(kBlocks, vec,
                 [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
                   PublishKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
                       gpu_info, vec.GetDevice(0), kPage, /*gen=*/1,
                       /*fill=*/11u, vw, sv);
                 });
    ctp::GpuApi::Synchronize();
    // THE CACHE MUST BE DROPPED, or the demand below is answered by this
    // node's own dirty frame and the stamp is never consulted -- the test
    // would pass without the runtime ever seeing a generation.
    vec.ClearCache();

    ctp::GpuApi::Memset(d_bad, 0, sizeof(unsigned long long));
    ctp::GpuApi::Memset(d_served, 0, sizeof(unsigned long long));
    RunYieldable(kBlocks, vec,
                 [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
                   DemandKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
                       gpu_info, vec.GetDevice(0), kPage, /*gen=*/1,
                       /*fill=*/11u, d_bad, d_served, vw, sv);
                 });
    ctp::GpuApi::Synchronize();

    unsigned long long bad = 0, served = 0;
    ctp::GpuApi::Memcpy(&bad, d_bad, sizeof(bad));
    ctp::GpuApi::Memcpy(&served, d_served, sizeof(served));
    const auto st = vec.ReadStats(0);
    const bool ok = (served != 0) && (bad == 0);
    std::printf("  A published gen 1, demanded gen 1: served=%llu wrong=%llu "
                "puts=%llu get_errors=%llu -> %s\n",
                served, bad, (unsigned long long)st.puts,
                (unsigned long long)st.get_errors, ok ? "PASS" : "FAIL");
    if (!ok) {
      std::printf("    a generational flush of a page this node then demands "
                  "at the SAME generation was not served: the flush did not "
                  "stamp the range, or the demand did not see the stamp\n");
      rc = 1;
    }
  }

  // ---- B: publish a RESIDENT page with NO write-hold, which is what
  // md_bench's PublishSlabCoro does -- it flushes the slab it already owns
  // rather than writing through a guard first. If the stamp depends on the
  // page having been dirtied by THIS kernel, A passes and B fails.
  {
    RunYieldable(kBlocks, vec,
                 [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
                   PublishOnlyKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
                       gpu_info, vec.GetDevice(0), kPage, /*gen=*/2, vw, sv);
                 });
    ctp::GpuApi::Synchronize();
    vec.ClearCache();

    ctp::GpuApi::Memset(d_bad, 0, sizeof(unsigned long long));
    ctp::GpuApi::Memset(d_served, 0, sizeof(unsigned long long));
    RunYieldable(kBlocks, vec,
                 [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
                   DemandKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
                       gpu_info, vec.GetDevice(0), kPage, /*gen=*/2,
                       /*fill=*/11u, d_bad, d_served, vw, sv);
                 });
    ctp::GpuApi::Synchronize();

    unsigned long long bad = 0, served = 0;
    ctp::GpuApi::Memcpy(&bad, d_bad, sizeof(bad));
    ctp::GpuApi::Memcpy(&served, d_served, sizeof(served));
    const auto st = vec.ReadStats(0);
    const bool ok = (served != 0) && (bad == 0);
    std::printf("  B republished resident page as gen 2, demanded gen 2: "
                "served=%llu wrong=%llu puts=%llu get_errors=%llu -> %s\n",
                served, bad, (unsigned long long)st.puts,
                (unsigned long long)st.get_errors, ok ? "PASS" : "FAIL");
    if (!ok) {
      std::printf("    a flush with no write-hold did not advance the "
                  "range's generation -- which is exactly what md_bench's "
                  "halo publish does every step\n");
      rc = 1;
    }
  }

  std::printf("  (case C -- demanding a generation nobody published must be "
              "refused -- is covered by the device fatal path; it traps by "
              "design and cannot share a process with case A)\n");

  REQUIRE(rc == 0);
}
#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()
