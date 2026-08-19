/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Race stress for the paging protocol: many blocks concurrently probe
 * (lock-free TryHoldRawConst), fault (FetchPagesBatched), async-prefetch
 * (FetchPagesBatchedAsync), and evict the SAME small page space, verifying
 * every read byte against its seeded pattern.
 *
 * Exists because the cross-block claim-visibility race (fetching stored
 * with only a BLOCK-scoped fence before page_num) served the EVICTED page's
 * bytes to lock-free probes in other blocks — which surfaced end-to-end as
 * intermittent garbage llama output after ~3 minutes of model load, and
 * took hours to attribute. This reproduces that class of bug in seconds:
 * a paging bug shows up as wrong bytes far more often than as a crash.
 *
 * Page->table affinity mirrors the MoE kernel: block_override_ = pn % nblocks,
 * so every block touching page pn uses the SAME table — probe/claim/evict
 * collide across blocks by construction.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <cstdlib>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;

/** Run a yieldable kernel to completion (see the other gpu_vector tests). */
template <typename LaunchT>
static clio::run::u32 RunYieldable(unsigned nblocks, LaunchT &&launch) {
  gy::Yieldable<> drv(nblocks, 32);
  // 8 KB lanes, not 256 B: the stress kernel is a C++20 COROUTINE now and
  // its frame lives on this lane -- a macro-era 256-byte lane overflows on
  // the first co_await and traps. The macro seed kernel shares the driver
  // and simply leaves the extra room unused.
  gy::YieldStack stack(nblocks, 32, 8192);
  return drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000);
}

namespace {
// 512 KiB pages — the MoE workload's granularity. The 4 KiB first cut
// PASSED on known-broken code: its DMA lands in microseconds, so the
// claim->land window the race needs never opened. Window length is the
// amplification lever, not iteration count.
constexpr clio::run::u64 kPageBytes = 128ull << 10;
constexpr clio::run::u64 kPageElems = kPageBytes / sizeof(clio::run::u32);

/** Seed value for element i. */
CTP_INLINE_CROSS_FUN clio::run::u32 Seed(clio::run::u64 i) {
  return static_cast<clio::run::u32>(i * 2654435761u + 17u);
}
}  // namespace

/**
 * Seeds a whole slice on the GPU, through the paging path.
 *
 * It used to flush after EVERY page, because "an evicted dirty page is
 * DROPPED, not written back (weights-path assumption)" -- so deferring the
 * flush seeded cross-page garbage. That assumption was a BUG (the slot claim
 * chose its victim without checking `dirty`), and it is fixed: an eviction
 * writes its victim back. One flush at the end is now correct, and the slice
 * is deliberately larger than the cache so the eviction writeback is what
 * stores most of it.
 */
__global__ void RaceSeedKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<clio::run::u32> v,
                               clio::run::u64 per, gy::YieldableView<> yv,
                               gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(clio::run::u64, i, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, run, 0);
  const clio::run::u64 base = static_cast<clio::run::u64>(yv.Block()) * per;

  CLIO_YBEGIN();
  for (; i < per; i += run) {
    CLIO_YCALL(v.HoldPageYield(base + i, per - i, &run, /*write=*/true));
    for (clio::run::u64 k = threadIdx.x; k < run; k += blockDim.x) {
      v[base + i + k] = Seed(base + i + k);
    }
  }
  // Collective, internally BATCHED (one multi-put per 64 pages).
  v.FlushAsync(base, per);
  CLIO_YIELD_IF(v.AnyTransferInFlight());
  CLIO_YEND();
}

/**
 * The concurrency stressor, as a COROUTINE. The original was a plain
 * kernel that camped idle warps on a shared flag to saturate the SMs and
 * then took BLOCKING in-kernel waits (HoldRawConst after a miss, the
 * synchronous batched fetch) -- a deliberate recreation of the llama
 * wedge. That design is the exact anti-pattern the coroutine machinery
 * abolishes: a resident kernel must never wait on work the host (or the
 * GPU's own copy/launch path) still has to do. What the test PROTECTS is
 * still fully exercised here -- concurrent demand faults, fire-and-forget
 * prefetches of unrelated ranges, evictions recycling slots mid-read, and
 * generation-validated long raw reads that catch a slot being re-tenanted
 * under a reader -- but every wait is a suspension.
 *
 * err[0] mismatches, err[1] first bad page, err[2] got, err[3] want.
 */
__device__ gy::YCoroMain RaceStressCoro(gv::DeviceVector<clio::run::u32> v,
                                        clio::run::u64 total_pages, int iters,
                                        unsigned long long *err,
                                        clio::run::u32 block) {
  unsigned int rng = 1234567u + 97u * block;
  for (int it = 0; it < iters; ++it) {
    rng = rng * 1664525u + 1013904223u;
    const clio::run::u64 pn = rng % total_pages;
    v.block_override_ = (clio::run::u32) (pn % v.h_->nblocks_);

    // Fire-and-forget prefetch of an unrelated range every iteration --
    // the long in-flight window is what widens claim races enough to
    // fire. Score 1.0 == make resident; block-collective.
    {
      rng = rng * 1664525u + 1013904223u;
      const clio::run::u64 pf = rng % total_pages;
      v.block_override_ = (clio::run::u32) (pf % v.h_->nblocks_);
      v.RescorePagesBatchedAsync(4u, [pf, total_pages](clio::run::u32 i) {
        return gv::PageScore{(pf + i) % total_pages, 1.0f};
      });
      v.block_override_ = (clio::run::u32) (pn % v.h_->nblocks_);
    }

    // Demand access: suspend until resident, then take a RAW pointer with
    // its claim generation and read across a long window. Nothing pins a
    // held page, so a sibling's claim can re-tenant the slot mid-read --
    // the generation check below is what must catch that.
    clio::run::u64 run = 0;
    co_await v.HoldPage(pn * kPageElems, kPageElems, &run);
    if (threadIdx.x == 0 && run != 0) {
      clio::run::u32 gen = 0;
      gv::Page *slot = nullptr;
      const clio::run::u32 *q = v.TryHoldRawConstG(pn * kPageElems, kPageElems,
                                                   &run, &gen, &slot);
      if (q != nullptr && run != 0) {
        // LONG READ: ~50 us of "compute" while holding the raw pointer.
        const long long t0 = clock64();
        while (clock64() - t0 < 100000) {
        }
        const clio::run::u64 idx[3] = {0, run / 2, run - 1};
        clio::run::u32 got[3];
        for (int k = 0; k < 3; ++k) got[k] = q[idx[k]];
        // Only a read that SURVIVED its claim generation may be judged:
        // a recycled slot is the expected hazard, not a data error.
        if (v.HoldStillValid(slot, gen)) {
          for (int k = 0; k < 3; ++k) {
            const clio::run::u64 el = pn * kPageElems + idx[k];
            const clio::run::u32 want = Seed(el);
            if (got[k] != want) {
              if (atomicAdd(err, 1ull) == 0) {
                err[1] = pn;
                err[2] = got[k];
                err[3] = want;
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ void RaceStressKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceVector<clio::run::u32> v,
                                 clio::run::u64 total_pages, int iters,
                                 unsigned long long *err,
                                 gy::YieldableView<> yv,
                                 gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(RaceStressCoro(v, total_pages, iters, err, yv.Block()));
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: concurrent probe/fault/prefetch/evict serves exact bytes",
          "[gpu_vector][race]") {
  {
    std::ofstream cfg("gpu_vector_race.yaml");
    REQUIRE(cfg.is_open());
    cfg << "networking:\n  port: 9433\n\n"
        << "runtime:\n  num_threads: 4\n  queue_depth: 4096\n\n"
        << "gpu:\n  queue_depth: 4096\n\n"
        // The seeded image is ~1.2 GB; the default core's target is far
        // smaller (host puts failed ENOMEM at page 800 without this).
        // Compose the DEFAULT core pool (name+id the client binds to) with
        // a big tier, so both the host client and the vector use it.
        << "compose:\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: clio_cte_core\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n"
        << "    storage:\n"
        // CLIO_RACE_TIER=hbm turns this into the sub-minute kHbm wedge
        // repro (managed-queue migration ping-pong under resident faulting
        // kernels): same machinery as the 4-minute llama runs, verdict in
        // the test's own runtime (~30 s; a 60 s timeout = wedge).
        << "      - path: \"" << (std::getenv("CLIO_RACE_TIER")
                                       ? std::getenv("CLIO_RACE_TIER")
                                       : "ram")
        << "::gv_race_tier\"\n"
        << "        bdev_type: \"" << (std::getenv("CLIO_RACE_TIER")
                                            ? std::getenv("CLIO_RACE_TIER")
                                            : "ram")
        << "\"\n        capacity_limit: \"3GB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_race.yaml", 1);
  }

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  // 24 seed blocks x 8 pages = 192 pages of truth; the stress phase runs 24
  // blocks with only 8 cache slots per table — heavy oversubscription, so
  // claim/evict runs constantly against the lock-free probes.
  constexpr unsigned kBlocks = 48;
  // pages_per_block must exceed 128 so multi_per_block >= 3 and the ASYNC
  // batch slots exist at all (ceil(ppb/64) - 1 of them); 384 pages walked
  // through 192 slots keeps claim/evict hot.
  constexpr clio::run::u64 kPagesPerSlice = 480;
  constexpr clio::run::u64 kTotalPages = kBlocks * kPagesPerSlice;
  // Pressure high enough to model llama's sustained fault storms — the
  // managed-queue migration wedge needs continuous device pushes + CPU
  // drains under resident kernels, not a short burst. CLIO_RACE_ITERS
  // overrides for a quick byte-exactness verdict (the full 8192 runs many
  // minutes; correctness bugs historically reproduced within hundreds).
  int kIters = 8192;
  if (const char *it_env = std::getenv("CLIO_RACE_ITERS")) {
    kIters = std::atoi(it_env);
    if (kIters <= 0) kIters = 8192;
  }
  const clio::run::u64 n = kTotalPages * kPageElems;

  gv::Vector<clio::run::u32> vec("gv_race", {0}, kPageBytes, kBlocks,
                                 /*pages_per_block=*/192, n);

  // GPU seed, through the paging path. This was 23040 sequential host
  // PutBlobs -- ~5 minutes, and the reason this test ran far over its budget
  // -- because "the GPU flush path is deliberately NOT used: a walk larger
  // than the cache DROPS dirty evictions by design". That drop was a bug in
  // the slot claim, not a design, and it is fixed, so the device path can
  // seed its own image again. It also makes the seed itself a test of
  // eviction writeback at scale: 480 pages per block through 192 slots.
  {
    const clio::run::u64 per = kPagesPerSlice * kPageElems;
    vec.EnableStats();
    const clio::run::u32 seed_rounds =
        RunYieldable(kBlocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                  gy::YieldStackView sv) {
          RaceSeedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
              gpu_info, vec.GetDevice(0), per, vw, sv);
        });
    ctp::GpuApi::Synchronize();
    const auto st = vec.ReadStats(0);
    std::fprintf(stderr,
                 "[race-seed] rounds=%u faults=%llu puts=%llu evicts=%llu "
                 "put_errors=%llu\n",
                 seed_rounds, (unsigned long long) st.faults,
                 (unsigned long long) st.puts, (unsigned long long) st.evicts,
                 (unsigned long long) st.put_errors);
    REQUIRE(st.put_errors == 0);
  }

  // Split write-path from read-path corruption BEFORE the stress phase:
  // verify the seeded blobs from the HOST. A wrong byte here means the
  // batched flush already stored another page's data; a clean pass here
  // followed by a stress failure indicts the device read path.
  {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    std::vector<clio::run::u32> buf(kPageElems, 0u);
    clio::run::u64 bad_pages = 0;
    for (clio::run::u64 pg = 0; pg < kTotalPages; pg += 7) {
      char name[32];
      gv::PageBlobName(pg, name);
      auto f = core.AsyncGetBlob(vec.TagId(), std::string(name), 0, kPageBytes,
                                 0, reinterpret_cast<char *>(buf.data()));
      f.Wait();
      if (f->GetReturnCode() != 0) {
        std::fprintf(stderr, "seed-verify: host read %s rc=%d\n", name,
                     f->GetReturnCode());
        ++bad_pages;
        continue;
      }
      for (clio::run::u64 i = 0; i < kPageElems; i += kPageElems / 4) {
        const clio::run::u64 el = pg * kPageElems + i;
        if (buf[i] != Seed(el)) {
          std::fprintf(stderr,
                       "seed-verify: page %llu el %llu got=0x%x want=0x%x\n",
                       (unsigned long long) pg, (unsigned long long) el,
                       buf[i], Seed(el));
          ++bad_pages;
          break;
        }
      }
    }
    if (bad_pages != 0) {
      std::fprintf(stderr, "seed-verify: %llu corrupt pages BEFORE stress\n",
                   (unsigned long long) bad_pages);
    }
    REQUIRE(bad_pages == 0);
  }

  unsigned long long *err = nullptr;
  err = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(err)>>(4 * sizeof(unsigned long long));
  ctp::GpuApi::Memset(err, 0, 4 * sizeof(unsigned long long));

  RunYieldable(kBlocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                            gy::YieldStackView sv) {
    RaceStressKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
        gpu_info, vec.GetDevice(0), kTotalPages, kIters, err, vw, sv);
  });
  // GpuApi::Synchronize error-checks fatally, so reaching the next
  // line IS the assertion.
  ctp::GpuApi::Synchronize();

  unsigned long long h[4] = {0};
  ctp::GpuApi::Memcpy(h, err, sizeof(h));
  if (h[0] != 0) {
    fprintf(stderr,
            "race: %llu mismatches; first bad page=%llu got=0x%llx "
            "want=0x%llx\n",
            h[0], h[1], h[2], h[3]);
  }
  REQUIRE(h[0] == 0);
  ctp::GpuApi::Free(err);
}

#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()
