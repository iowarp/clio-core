/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * UpdateRange: read a range another block already has, without fetching a
 * second copy of it.
 *
 * Every block owns a private page table, so a page two blocks both read is
 * stored twice -- measured at 7.81 copies of every page of the MD positions
 * vector across 16 blocks. UpdateRange borrows the owner's frame instead,
 * and the thing that proves it is the FAULT COUNT: a borrow must not fault,
 * where the same read through BeginFetch/HoldPage must.
 *
 * `from` is supplied by the caller, never discovered -- see the verb's
 * comment for why a directory is the wrong answer.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

#include <cstdio>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

namespace {
constexpr u64 kPageBytes = 4096;
constexpr u64 kEpp = kPageBytes / sizeof(u32);
constexpr u32 kBlocks = 8;
constexpr u64 kElems = kBlocks * kEpp;   // exactly one page per block
constexpr u32 kSlots = 4;
constexpr u32 kCheck = 256;              // elements verified per page

CTP_INLINE_CROSS_FUN u32 Pattern(u32 owner, u64 i) {
  return owner * 1000003u + static_cast<u32>(i) * 17u + 5u;
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

/** Block b writes and publishes page b, leaving it resident in b's table. */
__device__ gy::YCoroMain SeedCoro(gv::DeviceVector<u32> v, u32 block) {
  const u64 off = static_cast<u64>(block) * kEpp;
  co_await v.BeginFetch(v.PageLo(off), v.PageSpan(off, kEpp));
  co_await v.AwaitFetch();
  {
    auto h = co_await v.HoldPage(off, kEpp, /*write=*/true);
    for (u64 i = threadIdx.x; i < kEpp; i += blockDim.x) {
      h[off + i] = Pattern(block, i);
    }
  }
  co_await v.BeginFlush(off, kEpp);
  co_await v.EndFlush();
}

/** Block b reads the page OWNED BY ANOTHER BLOCK through its owner. */
__device__ gy::YCoroMain BorrowCoro(gv::DeviceVector<u32> v, u32 block,
                                    u32 nblocks, unsigned long long *bad) {
  const u32 owner = (block + 1u) % nblocks;
  const u64 off = static_cast<u64>(owner) * kEpp;
  auto h = co_await v.UpdateRange(off, kEpp, owner);
  for (u64 i = threadIdx.x; i < kCheck; i += blockDim.x) {
    if (h[off + i] != Pattern(owner, i)) atomicAdd(bad, 1ull);
  }
}

/** The control: the same read, the ordinary way, into this block's table. */
__device__ gy::YCoroMain CopyCoro(gv::DeviceVector<u32> v, u32 block,
                                  u32 nblocks, unsigned long long *bad) {
  const u32 owner = (block + 2u) % nblocks;
  const u64 off = static_cast<u64>(owner) * kEpp;
  co_await v.BeginFetch(v.PageLo(off), v.PageSpan(off, kEpp));
  co_await v.AwaitFetch();
  auto h = co_await v.HoldPage(off, kEpp);
  for (u64 i = threadIdx.x; i < kCheck; i += blockDim.x) {
    if (h[off + i] != Pattern(owner, i)) atomicAdd(bad, 1ull);
  }
}

/**
 * WRITE THROUGH THE COPY, then publish it. The result of UpdateRange is this
 * block's OWN frame, so dirty tracking and flush work exactly as they do for
 * a faulted page -- that is what copying buys over pointing at the owner's
 * frame, which could not be written at all.
 */
__device__ gy::YCoroMain WriteCoro(gv::DeviceVector<u32> v, u32 block,
                                   u32 nblocks, unsigned long long *bad) {
  const u32 owner = (block + 4u) % nblocks;
  const u64 off = static_cast<u64>(owner) * kEpp;
  {
    auto h = co_await v.UpdateRange(off, kEpp, owner, /*write=*/true);
    for (u64 i = threadIdx.x; i < kEpp; i += blockDim.x) {
      h[off + i] = Pattern(owner, i) + 77u;
    }
  }
  co_await v.BeginFlush(off, kEpp);
  co_await v.EndFlush();
  (void)bad;
}

/** A `from` that does NOT have the page: must fall back, not fail. */
__device__ gy::YCoroMain FallbackCoro(gv::DeviceVector<u32> v, u32 block,
                                      u32 nblocks, unsigned long long *bad) {
  const u32 owner = (block + 3u) % nblocks;
  const u64 off = static_cast<u64>(owner) * kEpp;
  // Name a block that owns nothing of this range; the probe misses and the
  // verb must fetch instead of trapping or returning garbage.
  auto h = co_await v.UpdateRange(off, kEpp, (block + 5u) % nblocks);
  for (u64 i = threadIdx.x; i < kCheck; i += blockDim.x) {
    if (h[off + i] != Pattern(owner, i)) atomicAdd(bad, 1ull);
  }
}

#define GV_KERNEL(name, coro)                                                 \
  __global__ void name(clio::run::IpcManagerGpuInfo info,                     \
                       gv::DeviceVector<u32> v, u32 nblocks,                  \
                       unsigned long long *bad, gy::YieldableView<> yv,       \
                       gy::YieldStackView ys) {                               \
    CLIO_GPU_INIT(info, nullptr);                                             \
    v.Init(yv.Block());                                                       \
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());                              \
    __syncthreads();                                                          \
    CLIO_YCORO_RUN(coro);                                                     \
  }

GV_KERNEL(SeedKernel, SeedCoro(v, yv.Block()))
GV_KERNEL(BorrowKernel, BorrowCoro(v, yv.Block(), nblocks, bad))
GV_KERNEL(CopyKernel, CopyCoro(v, yv.Block(), nblocks, bad))
GV_KERNEL(FallbackKernel, FallbackCoro(v, yv.Block(), nblocks, bad))
GV_KERNEL(WriteKernel, WriteCoro(v, yv.Block(), nblocks, bad))

#if !CTP_IS_DEVICE_PASS
TEST_CASE("gpu_vector: UpdateRange borrows a peer's frame",
          "[gpu_vector][update_range]") {
  {
    std::ofstream cfg("gpu_vector_update.yaml");
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
        << "      - path: \"ram::gv_upd_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"256MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_update.yaml", 1);
  }
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  clio::run::IpcManagerGpuInfo gpu =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);
  unsigned long long *d_bad =
      ctp::GpuApi::Malloc<unsigned long long>(sizeof(unsigned long long));

  gv::Vector<u32> vec("gv_upd", {0}, kPageBytes, kBlocks, kSlots, kElems);
  vec.EnableStats();
  // PRELOAD FIRST. A page that has never been written has no blob; the fetch
  // path creates a zero-length one, and a whole-page put into that is
  // refused -- every flush came back an error and the backing store stayed
  // zeros, with only the vector's put_errors counter to say so.
  {
    std::vector<u32> zero(kElems, 0u);
    vec.Preload(zero.data(), kElems);
    vec.ClearCache();
  }

  auto run = [&](const char *what, auto kern) {
    ctp::GpuApi::Memset(d_bad, 0, sizeof(unsigned long long));
    const auto before = vec.ReadStats(0);
    RunYieldable(kBlocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                              gy::YieldStackView sv) {
      kern<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, vec.GetDevice(0), kBlocks,
                                            d_bad, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    const auto after = vec.ReadStats(0);
    unsigned long long bad = 0;
    ctp::GpuApi::Memcpy(&bad, d_bad, sizeof(bad));
    const unsigned long long faults =
        static_cast<unsigned long long>(after.faults - before.faults);
    std::printf("  %-28s faults=%-4llu wrong=%llu  puts=%llu put_err=%llu "
                "get_err=%llu evicts=%llu\n", what, faults, bad,
                (unsigned long long)(after.puts - before.puts),
                (unsigned long long)(after.put_errors - before.put_errors),
                (unsigned long long)(after.get_errors - before.get_errors),
                (unsigned long long)(after.evicts - before.evicts));
    return std::pair<unsigned long long, unsigned long long>(faults, bad);
  };

  run("seed (each block 1 page)", SeedKernel);
  {
    // Did the seed actually reach the backing store? The borrow reads the
    // owner's CACHE, so it would look right even if the flush never landed.
    std::vector<u32> host(kElems, 0u);
    vec.Download(host.data(), kElems);
    u64 storebad = 0;
    for (u32 b = 0; b < kBlocks; ++b) {
      for (u64 i = 0; i < kCheck; ++i) {
        if (host[b * kEpp + i] != Pattern(b, i)) ++storebad;
      }
    }
    std::printf("    host[0]=%u want=%u | host[kEpp]=%u want=%u\n",
                host[0], Pattern(0, 0), host[kEpp], Pattern(1, 0));
    std::printf("  backing store after seed: %llu of %llu wrong\n",
                (unsigned long long)storebad,
                (unsigned long long)(kBlocks * kCheck));
  }
  const auto borrow = run("UpdateRange from owner", BorrowKernel);
  const auto copy = run("control: BeginFetch+Hold", CopyKernel);
  const auto fb = run("UpdateRange, wrong owner", FallbackKernel);

  const auto wr = run("write through the copy", WriteKernel);
  {
    std::vector<u32> host(kElems, 0u);
    vec.Download(host.data(), kElems);
    u64 wrong = 0;
    for (u32 b = 0; b < kBlocks; ++b) {
      // Block b wrote the page owned by (b+4)%kBlocks.
      const u32 owner = (b + 4u) % kBlocks;
      for (u64 i = 0; i < kCheck; ++i) {
        if (host[owner * kEpp + i] != Pattern(owner, i) + 77u) ++wrong;
      }
    }
    std::printf("  write reached the store:     %llu of %llu wrong\n",
                (unsigned long long)wrong,
                (unsigned long long)(kBlocks * kCheck));
    REQUIRE(wrong == 0);
  }
  REQUIRE(wr.second == 0);

  // Data has to be right in all three.
  REQUIRE(borrow.second == 0);
  REQUIRE(copy.second == 0);
  REQUIRE(fb.second == 0);
  // THE POINT: borrowing does not pull a second copy into this block's
  // table, so it costs no fault, while the ordinary path costs one per
  // block. Without that gap the verb is just a slower HoldPage.
  REQUIRE(borrow.first == 0);
  REQUIRE(copy.first >= kBlocks);
  // A wrong `from` must degrade to a fetch, not fail.
  REQUIRE(fb.first >= 1);
}
#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()
