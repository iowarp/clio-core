/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * A shared page cache: one copy of a page for the whole grid.
 *
 * Every CUDA block owns a private page table today, so a page that eight
 * blocks read is stored eight times. In the MD workload that is 100% of the
 * positions vector, held by 8.22 blocks on average -- 1536 frames for a
 * 96-page vector. A shared cache puts a page in ONE home set,
 * Hash(page_num) % nsets, and every block finds it there.
 *
 * WHAT IS ASSERTED, and why the byte check alone is not enough: a broken
 * shared cache that quietly kept per-block copies would still return the
 * right bytes. So the test pairs them:
 *
 *   1. bytes -- every block reads the same range and every element matches;
 *   2. COPIES -- MaxPageCopies() is 1 in shared mode. That is the claim.
 *      The private run is the control and is expected to exceed 1.
 *
 * The access pattern is the maximally-shared one on purpose: all blocks read
 * the whole vector. Private mode stores nblocks copies of every page; shared
 * mode stores one.
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
#include <thread>
#include <vector>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

namespace {
constexpr u64 kPageBytes = 4096;
constexpr u64 kEpp = kPageBytes / sizeof(u32);
constexpr u32 kBlocks = 8;               // CUDA blocks, and sets when shared
constexpr u64 kPages = 8;
constexpr u64 kElems = kPages * kEpp;
/** Private: frames per table. Shared: frames per SET, 2x over-provisioned so
 *  hash collisions have room -- the byte budget is held by cache_frames_. */
constexpr u32 kSlotsShared = 4;

CTP_INLINE_CROSS_FUN u32 Pattern(u64 i) {
  return static_cast<u32>(i) * 2654435761u + 12345u;
}
}  // namespace

#if !CTP_IS_DEVICE_PASS
template <typename LaunchT>
static u32 RunYieldable(unsigned nblocks, LaunchT &&launch) {
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

/** Block b seeds page b and publishes it. One writer per page. */
__device__ gy::YCoroMain SeedCoro(gv::DeviceVector<u32> v, u32 block) {
  const u64 off = static_cast<u64>(block) * kEpp;
  co_await v.Fetch(0, off, kEpp);
  {
    auto h = co_await v.HoldPage(off, kEpp, /*write=*/true);
    for (u64 i = threadIdx.x; i < kEpp; i += blockDim.x) {
      h[off + i] = Pattern(off + i);
    }
  }
  co_await v.Flush(0, off, kEpp);
  // Release the fetch's pins. A no-op in private mode, where the guard above
  // already dropped its own; required in shared mode, where Fetch is the
  // pinner and nothing else will.
  v.UnpinRange(off, kEpp);
}

/**
 * EVERY block reads EVERY page -- the pattern a shared cache exists for.
 * One page at a time so the pinned working set stays at one page per block,
 * well inside a set.
 */
__device__ gy::YCoroMain ReadAllCoro(gv::DeviceVector<u32> v,
                                     unsigned long long *bad) {
  for (u64 pg = 0; pg < kPages; ++pg) {
    const u64 off = pg * kEpp;
    co_await v.Fetch(0, off, kEpp);
    {
      auto h = co_await v.HoldPage(off, kEpp);
      for (u64 i = threadIdx.x; i < kEpp; i += blockDim.x) {
        if (h[off + i] != Pattern(off + i)) atomicAdd(bad, 1ull);
      }
    }
    __syncthreads();
    v.UnpinRange(off, kEpp);
  }
}

/**
 * SEVERAL BLOCKS WRITE ONE PAGE, each its own slice, and each publishes only
 * that slice.
 *
 * In a shared cache this is literally one frame being written by every block
 * at once. It is safe only because the slices are byte-disjoint and Flush is
 * byte-exact -- a whole-page flush here would have each block ship its view
 * of the other seven slices and clobber them.
 */
__device__ gy::YCoroMain SliceWriteCoro(gv::DeviceVector<u32> v, u32 block,
                                        u32 nblocks) {
  const u64 slice = kEpp / nblocks;
  const u64 off = static_cast<u64>(block) * slice;   // all inside page 0
  co_await v.Fetch(0, off, slice);
  {
    auto h = co_await v.HoldPage(off, slice, /*write=*/true);
    for (u64 i = threadIdx.x; i < slice; i += blockDim.x) {
      h[off + i] = Pattern(off + i) ^ 0xABCDu;
    }
  }
  __syncthreads();
  co_await v.Flush(0, off, slice);       // exactly this block's bytes
  v.UnpinRange(off, slice);
}

/** Read back the whole page every block just co-wrote. */
__device__ gy::YCoroMain SliceVerifyCoro(gv::DeviceVector<u32> v,
                                         unsigned long long *bad) {
  co_await v.Fetch(0, 0, kEpp);
  {
    auto h = co_await v.HoldPage(0, kEpp);
    for (u64 i = threadIdx.x; i < kEpp; i += blockDim.x) {
      if (h[i] != (Pattern(i) ^ 0xABCDu)) atomicAdd(bad, 1ull);
    }
  }
  __syncthreads();
  v.UnpinRange(0, kEpp);
}

/**
 * SEVERAL BLOCKS FETCH DIFFERENT SLICES OF ONE PAGE, concurrently.
 *
 * This is the case a shared cache creates and a private one cannot: two
 * blocks filling the SAME frame with different sub-ranges. Each block then
 * reads back only what it asked for, so any block whose slice was dropped
 * from the frame's valid range fails here rather than silently reading bytes
 * nobody fetched.
 */
__device__ gy::YCoroMain PartialFetchCoro(gv::DeviceVector<u32> v, u32 block,
                                          u32 nblocks,
                                          unsigned long long *bad) {
  const u64 slice = kEpp / nblocks;
  const u64 off = static_cast<u64>(block) * slice;   // all inside page 0
  co_await v.Fetch(0, off, slice);
  {
    auto h = co_await v.HoldPage(off, slice);
    for (u64 i = threadIdx.x; i < slice; i += blockDim.x) {
      if (h[off + i] != Pattern(off + i)) atomicAdd(bad, 1ull);
    }
  }
  __syncthreads();
  v.UnpinRange(off, slice);
}

/**
 * Every block streams every page with a cache far smaller than the vector,
 * so frames are evicted and refetched under contention from all sides.
 * Correct data here means eviction is picking victims nobody is using.
 */
__device__ gy::YCoroMain ChurnCoro(gv::DeviceVector<u32> v, u32 block,
                                   u32 nblocks, u64 pages,
                                   unsigned long long *bad) {
  for (u64 r = 0; r < 3; ++r) {
    for (u64 k = 0; k < pages; ++k) {
      // Start each block at a different page so they collide continuously
      // instead of marching in lockstep.
      const u64 pg = (k + block) % pages;
      const u64 off = pg * kEpp;
      co_await v.Fetch(0, off, kEpp);
      {
        auto h = co_await v.HoldPage(off, kEpp);
        for (u64 i = threadIdx.x; i < kEpp; i += blockDim.x) {
          if (h[off + i] != Pattern(off + i)) atomicAdd(bad, 1ull);
        }
      }
      __syncthreads();
      v.UnpinRange(off, kEpp);
    }
  }
  (void)nblocks;
}

#define GV_KERNEL(name, coro)                                                 \
  __global__ void name(clio::run::IpcManagerGpuInfo info,                     \
                       gv::DeviceVector<u32> v,                               \
                       unsigned long long *bad, gy::YieldableView<> yv,       \
                       gy::YieldStackView ys) {                               \
    CLIO_GPU_INIT(info, nullptr);                                             \
    v.Init(yv.Block());                                                       \
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());                              \
    __syncthreads();                                                          \
    CLIO_YCORO_RUN(coro);                                                     \
  }

GV_KERNEL(SeedKernel, SeedCoro(v, yv.Block()))
GV_KERNEL(ReadAllKernel, ReadAllCoro(v, bad))
GV_KERNEL(SliceWriteKernel, SliceWriteCoro(v, yv.Block(), kBlocks))
GV_KERNEL(SliceVerifyKernel, SliceVerifyCoro(v, bad))
GV_KERNEL(PartialFetchKernel, PartialFetchCoro(v, yv.Block(), kBlocks, bad))
GV_KERNEL(ChurnKernel, ChurnCoro(v, yv.Block(), kBlocks, kPages, bad))

#if !CTP_IS_DEVICE_PASS
TEST_CASE("gpu_vector: a shared cache stores a page once",
          "[gpu_vector][shared_cache]") {
  {
    std::ofstream cfg("gpu_vector_shared_cache.yaml");
    cfg << "networking:\n  port: 9431\n\n"
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
        << "      - path: \"ram::gv_shared_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"256MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_shared_cache.yaml",
                            1);
  }
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  clio::run::IpcManagerGpuInfo gpu =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);
  unsigned long long *d_bad =
      ctp::GpuApi::Malloc<unsigned long long>(sizeof(unsigned long long));

  int rc = 0;

  // Every block reads every page. THE CLAIM is that the cache holds each of
  // them once, no matter how many blocks are in it. There is no private-table
  // arm any more -- per-block tables are gone -- so the control is the
  // arithmetic: with kBlocks blocks and one table each this would have read
  // back kBlocks copies of every page.
  auto run_mode = [&](const char *what, u32 slots) -> u32 {
    gv::Vector<u32> vec("gv_shared_b", {0}, kPageBytes, kBlocks, slots,
                        kElems);
    {
      std::vector<u32> zero(kElems, 0u);
      vec.Preload(zero.data(), kElems);
      vec.ClearCache();
    }
    RunYieldable(kBlocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                              gy::YieldStackView sv) {
      SeedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, vec.GetDevice(0), d_bad,
                                                  vw, sv);
    });
    ctp::GpuApi::Synchronize();
    vec.ClearCache();

    ctp::GpuApi::Memset(d_bad, 0, sizeof(unsigned long long));
    RunYieldable(kBlocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                              gy::YieldStackView sv) {
      ReadAllKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, vec.GetDevice(0),
                                                     d_bad, vw, sv);
    });
    ctp::GpuApi::Synchronize();

    unsigned long long bad = 0;
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(&bad),
                        reinterpret_cast<const char *>(d_bad),
                        sizeof(unsigned long long));
    const u32 copies = vec.MaxPageCopies(0);
    std::printf("  %-8s bad=%llu  max copies of one page=%u\n", what,
                (unsigned long long)bad, copies);
    if (bad != 0) {
      std::printf("    FAIL: %llu elements wrong\n", (unsigned long long)bad);
      rc = 1;
    }
    return copies;
  };

  const u32 shared_copies = run_mode("shared", kSlotsShared);

  // ---- the cases a SHARED cache creates, which a private one cannot ----
  //
  // Above, every block reads whole pages, so a frame is either wholly there
  // or absent. These three put several blocks on ONE frame at once.
  auto scenario = [&](const char *what, u32 slots, u64 pages, auto &&body) {
    gv::Vector<u32> vec(what, {0}, kPageBytes, kBlocks, slots, pages * kEpp);
    {
      std::vector<u32> zero(static_cast<size_t>(pages * kEpp), 0u);
      vec.Preload(zero.data(), pages * kEpp);
      vec.ClearCache();
    }
    // Seed every page through the one-writer-per-page path.
    RunYieldable(kBlocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                              gy::YieldStackView sv) {
      SeedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, vec.GetDevice(0), d_bad,
                                                  vw, sv);
    });
    ctp::GpuApi::Synchronize();
    vec.ClearCache();
    ctp::GpuApi::Memset(d_bad, 0, sizeof(unsigned long long));
    body(vec);
    ctp::GpuApi::Synchronize();
    unsigned long long bad = 0;
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(&bad),
                        reinterpret_cast<const char *>(d_bad),
                        sizeof(unsigned long long));
    std::printf("  %-28s bad=%llu -> %s\n", what, (unsigned long long)bad,
                bad == 0 ? "PASS" : "FAIL");
    if (bad != 0) rc = 1;
  };

  // 1. Eight blocks each fetch a DIFFERENT SLICE of page 0, at the same time,
  //    then read back only their own slice. One frame, eight partial fills.
  scenario("shared: partial fetch x8", kSlotsShared, kPages,
           [&](gv::Vector<u32> &vec) {
             RunYieldable(kBlocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                       gy::YieldStackView sv) {
               PartialFetchKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
                   gpu, vec.GetDevice(0), d_bad, vw, sv);
             });
           });

  // 2. Eight blocks WRITE disjoint slices of page 0 -- one shared frame,
  //    eight concurrent writers -- each flushing only its own bytes. Then the
  //    page is re-read whole. A whole-page flush, or a lost slice, shows up
  //    as mismatches here.
  scenario("shared: disjoint writers x8", kSlotsShared, kPages,
           [&](gv::Vector<u32> &vec) {
             RunYieldable(kBlocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                       gy::YieldStackView sv) {
               SliceWriteKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
                   gpu, vec.GetDevice(0), d_bad, vw, sv);
             });
             ctp::GpuApi::Synchronize();
             vec.ClearCache();          // force the read to go to the store
             RunYieldable(kBlocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                       gy::YieldStackView sv) {
               SliceVerifyKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
                   gpu, vec.GetDevice(0), d_bad, vw, sv);
             });
           });

  // 3. Eviction under contention: 8 pages, 2 frames per set, every block
  //    streaming every page three times from a different starting offset.
  scenario("shared: eviction churn", /*slots=*/2, kPages,
           [&](gv::Vector<u32> &vec) {
             RunYieldable(kBlocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                                       gy::YieldStackView sv) {
               ChurnKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
                   gpu, vec.GetDevice(0), d_bad, vw, sv);
             });
           });

  // THE CLAIM. Shared mode stores each page exactly once no matter how many
  // blocks read it.
  if (shared_copies != 1u) {
    std::printf("  FAIL: shared cache holds a page in %u frames, expected 1\n",
                shared_copies);
    rc = 1;
  }
  std::printf("  replication: %u blocks reading every page -> %u copies\n",
              kBlocks, shared_copies);
  REQUIRE(rc == 0);
}
#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()
