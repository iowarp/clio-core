/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The gpu_vector smoke test, driven by SYCL kernels instead of CUDA ones.
 *
 * Deliberately the same workload as test_gpu_vector_smoke_gpu.cc -- write
 * every element through a held page, flush as it goes, read it all back and
 * count mismatches -- because the POINT is that only the launch changes.
 *
 * WHAT IS SHARED WITH THE CUDA BUILD, verbatim:
 *   - DeviceVector and the whole page cache (device_vector.h)
 *   - the coroutine machinery (yield_coro.h) and the lane stack
 *     (yield_stack.h)
 *   - the host relaunch driver (Yieldable in yieldable.h), including round
 *     compaction, the wait-tag readiness test and the round cap
 *   - the host Vector, the CTE client, and task submission
 *
 * WHAT IS NOT: the kernel bodies' outermost layer, which lives in
 * gv_sycl_kernels.cc for the reason documented in gv_sycl_kernels.h -- a
 * SYCL kernel is a lambda inside its launcher, so it cannot share a
 * translation unit with host code that the device pass will not parse.
 *
 * This file is ordinary C++: it is NOT compiled with -fsycl.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

#include <cstdio>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "gv_sycl_kernels.h"
#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
namespace gk = clio::cte::gpu_vector::sycl_test;

namespace {

constexpr clio::run::u64 kPageBytes = 4096;
constexpr clio::run::u64 kElems = 4096;   // 4 pages

/**
 * Run a yieldable kernel to completion -- the CUDA test's helper unchanged,
 * round cap included.
 *
 * CONVERGENCE IS PART OF THE RESULT. A servicer that never makes progress
 * still returns correct bytes for whatever pages happened to be resident, so
 * a data-only check can pass while livelocked.
 */
template <typename LaunchT>
clio::run::u32 RunYieldable(unsigned nblocks, LaunchT &&launch) {
  gy::Yieldable<> drv(nblocks, 32);
  // 8192, not the macro-era 256: coroutine frames are compiler-laid-out and
  // spill into the lane.
  gy::YieldStack stack(nblocks, 32, 8192);
  const clio::run::u32 rounds = drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000, gv::ResumeWhenComplete);
  if (rounds >= 200000) {
    std::fprintf(stderr, "[gv-sycl] FATAL: hit the round cap -- faults are "
                         "not being served\n");
    std::abort();
  }
  return rounds;
}

/** The vector's own counters. A mismatch count says WHAT went wrong; these
 *  say where -- a failed transfer shows up in get_errors/put_errors and
 *  nowhere else. */
template <typename VecT>
void DumpStats(const char *what, VecT &v) {
  const auto s = v.ReadStats(0);
  std::fprintf(stderr,
               "[%s] faults=%llu puts=%llu evicts=%llu get_err=%llu "
               "put_err=%llu flush_skipped=%llu occupied=%u\n",
               what, (unsigned long long)s.faults, (unsigned long long)s.puts,
               (unsigned long long)s.evicts,
               (unsigned long long)s.get_errors,
               (unsigned long long)s.put_errors,
               (unsigned long long)s.flush_skipped, v.Occupied(0));
}

unsigned long long *AllocCounter(clio::run::u64 npages = 0) {
  const size_t n = static_cast<size_t>(npages) + 1;
  auto *p = ctp::GpuApi::Malloc<unsigned long long>(
      n * sizeof(unsigned long long));
  ctp::GpuApi::Memset(p, 0, n * sizeof(unsigned long long));
  return p;
}

unsigned long long ReadTotalAndReportPages(unsigned long long *p,
                                           clio::run::u64 npages,
                                           const char *what) {
  std::vector<unsigned long long> h(static_cast<size_t>(npages) + 1, 0);
  ctp::GpuApi::Memcpy(h.data(), p,
                      (static_cast<size_t>(npages) + 1) *
                          sizeof(unsigned long long));
  std::string bad_pages;
  for (clio::run::u64 i = 0; i < npages; ++i) {
    if (h[i + 1] != 0) bad_pages += std::to_string(i) + " ";
  }
  std::fprintf(stderr, "[%s] wrong-pages=[%s]\n", what,
               bad_pages.empty() ? "none" : bad_pages.c_str());
  ctp::GpuApi::Free(p);
  return h[0];
}

unsigned long long ReadAndFree(unsigned long long *p) {
  unsigned long long v = 1;
  ctp::GpuApi::Memcpy(&v, p, sizeof(v));
  ctp::GpuApi::Free(p);
  return v;
}

}  // namespace

TEST_CASE("gpu_vector sycl: write, flush, read back",
          "[gpu_vector][sycl][smoke]") {
  {
    std::ofstream cfg("gpu_vector_sycl_test.yaml");
    REQUIRE(cfg.is_open());
    // The compose section is NOT optional: without it the CTE pool has no
    // storage targets, so the first page writeback has nowhere to land and
    // never completes.
    cfg << "networking:\n  port: 9433\n\n"
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
        << "      - path: \"ram::gv_sycl_smoke_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"256MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_sycl_test.yaml", 1);
  }

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  const clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  // The per-block IpcManager array. CUDA gets this from __shared__ storage
  // born fresh at each launch; SYCL needs it allocated and published once,
  // sized for the widest grid any launch below will use.
  constexpr clio::run::u32 kMaxBlocks = 64;
  gk::InitBlockIpc(kMaxBlocks, gpu_info);

  gv::Vector<clio::run::u32> vec("gv_sycl_smoke", {0}, kPageBytes,
                                 /*nblocks=*/1, /*pages_per_block=*/4, kElems);
  // Opt in to the device counters. A mismatch count says WHAT went wrong;
  // faults/puts/evicts/get_err/put_err say WHERE, and a transfer that
  // returned non-zero shows up nowhere else.
  vec.EnableStats();

  RunYieldable(1, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                      gy::YieldStackView sv) {
    gk::LaunchFill(g, b, vec.GetDevice(0), kElems, vw, sv);
  });

  // DROP THE CACHE BEFORE READING BACK. Without this the check reads the
  // frames the writer left resident and passes whether or not a single byte
  // reached the CTE -- it would verify the page cache against itself. With
  // it, every element below has made the full round trip through a flush and
  // a fetch.
  vec.ClearCache(0);

  unsigned long long *d_bad = AllocCounter();
  RunYieldable(1, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                      gy::YieldStackView sv) {
    gk::LaunchCheck(g, b, vec.GetDevice(0), kElems, d_bad, vw, sv);
  });
  const unsigned long long bad = ReadAndFree(d_bad);
  DumpStats("sycl-smoke", vec);
  std::fprintf(stderr, "[sycl-smoke] mismatches=%llu / %llu\n", bad,
               (unsigned long long)kElems);
  REQUIRE(bad == 0);

  // ---- oversubscribed: 16 pages through 4 slots ------------------------
  //
  // Every page past the fourth forces an eviction, and every victim is
  // DIRTY, so its bytes must be written back before the slot is reused.
  // Reading it all back therefore exercises writeback and slot reuse, not
  // just the cache.
  //
  // Same runtime deliberately: CLIO_INIT is once per PROCESS.
  {
    const clio::run::u64 n = 16 * 1024;
    gv::Vector<clio::run::u32> big("gv_sycl_evict", {0}, kPageBytes,
                                   /*nblocks=*/1, /*pages_per_block=*/4, n);
    big.EnableStats();

    RunYieldable(1, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                        gy::YieldStackView sv) {
      gk::LaunchFill(g, b, big.GetDevice(0), n, vw, sv);
    });

    big.ClearCache(0);   // same reason as above: read STORAGE, not the cache

    const clio::run::u64 npages_big =
        n * sizeof(clio::run::u32) / kPageBytes;
    unsigned long long *d2 = AllocCounter(npages_big);
    RunYieldable(1, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                        gy::YieldStackView sv) {
      gk::LaunchCheck(g, b, big.GetDevice(0), n, d2, vw, sv);
    });
    const unsigned long long b2 =
        ReadTotalAndReportPages(d2, npages_big, "sycl-evict");
    DumpStats("sycl-evict", big);
    std::fprintf(stderr, "[sycl-evict] mismatches=%llu / %llu\n", b2,
                 (unsigned long long)n);
    REQUIRE(b2 == 0);
  }

  // ---- many blocks ------------------------------------------------------
  // Each block drives its own page-aligned slice, so blocks never contend
  // for a PAGE -- but they do contend for a SET, which is why associativity
  // is 8 rather than the number of pages one block holds.
  //
  // STOPS AT 8, and the reason is known rather than arbitrary: at 32 blocks
  // this wedges, with the driver relaunching and no block ever completing.
  // The CUDA build carries the same workload to 128, but it submits through
  // the BATCHED DEVICE RING, and that transport is not ported --
  // ServerInitGpuQueues (SYCL) leaves ring.dev_ring null and every
  // submission falls back to the legacy per-task GpuTaskQueue, built with
  // ONE lane. Raising this bound is the same work item as porting the ring
  // (see gpu2cpu_init_sycl.cc), not a separate mystery.
  for (unsigned nb : {2u, 8u}) {
    const clio::run::u64 per = 2 * 1024;            // 2 pages per block
    const clio::run::u64 n = per * nb;
    gv::Vector<clio::run::u32> mv("gv_sycl_multi" + std::to_string(nb), {0},
                                  kPageBytes, /*nsets=*/nb,
                                  /*frames_per_set=*/8, n);

    RunYieldable(nb, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                         gy::YieldStackView sv) {
      gk::LaunchMultiFill(g, b, mv.GetDevice(0), per, vw, sv);
    });

    mv.ClearCache(0);    // same reason as above: read STORAGE, not the cache

    unsigned long long *d3 = AllocCounter();
    RunYieldable(nb, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                         gy::YieldStackView sv) {
      gk::LaunchMultiCheck(g, b, mv.GetDevice(0), per, d3, vw, sv);
    });
    const unsigned long long b3 = ReadAndFree(d3);
    std::fprintf(stderr, "[sycl-multi-%u] mismatches=%llu / %llu\n", nb, b3,
                 (unsigned long long)n);
    REQUIRE(b3 == 0);
  }
}

SIMPLE_TEST_MAIN()
