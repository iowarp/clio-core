/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Smoke test for the device-paged vector: one block, one warp, a working
 * set that fits, write-then-read through operator[] with an explicit
 * flush. This is the minimum that exercises every moving part -- page
 * fault, dirty tracking, FlushAsync/AwaitFlush -- so a failure here means
 * the mechanism is broken rather than a policy being wrong.
 *
 * The multi-page hang this file used to carry as a KNOWN LIMITATION ("raising
 * kElems past one page hangs: the kernel waits on a flush that never
 * completes") was the blocking fault path deadlocking against itself: the
 * kernel sat on the SM waiting for a writeback, and a resident kernel blocks
 * every later launch in its context, including the one servicing that
 * writeback. The kernels here yield now, so more than one page works and the
 * constant below is 4 pages.
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
#include <vector>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;

/**
 * Run a yieldable kernel to completion.
 *
 * Every launch below goes through this: a yielding kernel is not launched
 * once but re-launched until no block is left suspended, with a fresh
 * continuation stack backing the blocks' saved state.
 */
template <typename VecT, typename LaunchT>
static clio::run::u32 RunYieldable(unsigned nblocks, VecT &vec,
                                   LaunchT &&launch) {
  gy::Yieldable<> drv(nblocks, 32);
  // 8192, not the macro-era 256: coroutine frames are compiler-laid-out and
  // spill into the lane, overflowing anything page-thin.
  gy::YieldStack stack(nblocks, 32, 8192);
  const clio::run::u32 rounds = drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      // No service callback: the block resolves its own fault (design 5.1).
      // The get it submits is completed by the runtime's CPU workers, not
      // here. The round cap still matters -- see below.
      [] {}, /*max_rounds=*/200000,
      gv::ResumeWhenComplete);
  // CONVERGENCE IS PART OF THE RESULT, not just the data. A servicer that
  // never makes progress still returns correct bytes for pages that happened
  // to be resident, so a data-only check can pass while livelocked -- observed
  // exactly that at 200,000 rounds.
  if (rounds >= 200000) {
    std::fprintf(stderr, "[gv] FATAL: hit the round cap -- faults are not "
                         "being served\n");
    std::abort();
  }
  return rounds;
}

namespace {
constexpr clio::run::u64 kPageBytes = 4096;
constexpr clio::run::u64 kElems = 4096;          // 4 pages
}  // namespace

/** Fill every element with a known function of its index. */
/**
 * All four kernels are device COROUTINES faulting through the co_await
 * HoldPage verb, not the blocking hold.
 *
 * The blocking form waits for the fault INSIDE the kernel, and a resident
 * kernel blocks every later launch in its context -- including the work that
 * would service the fault. This test wedged in exactly that way: it reached
 * the first kernel and never came back, with the host parked in
 * cudaDeviceSynchronize forever. Suspending instead lets the device drain.
 *
 * They are also no longer single-threaded. A hold is block-collective (it
 * synchronizes internally), so `if (threadIdx.x != 0) return` would deadlock
 * the barrier; every lane runs the loop and they split each page between them.
 */
__device__ gy::YCoroMain FillCoro(gv::DeviceVector<clio::run::u32> v,
                                  clio::run::u64 n) {
  for (clio::run::u64 i = 0; i < n;) {
    clio::run::u64 run = 0;
    {
      co_await v.Fetch(0, v.PageLo(i), v.PageSpan(i, 1));
      auto h = co_await v.HoldPage(i, n - i, /*write=*/true);
      run = h.run();
      for (clio::run::u64 k = threadIdx.x; k < run; k += blockDim.x) {
        h[i + k] = static_cast<clio::run::u32>((i + k) * 7 + 1);
      }
    }   // guard dies here: the page is unpinned but still DIRTY
    // FLUSH AS WE GO. The vector never writes back on its own, so a dirty page
    // is unevictable until the caller flushes it. Writing more pages than the
    // table holds without flushing is a caller error, and the servicer says so
    // rather than silently writing data out. Flushing per page keeps the dirty
    // set at one.
    co_await v.Flush(0, i, run);
    // GIVE THE PIN BACK. Fetch is the pinner; a range fetched and never
    // unpinned is a frame no block can reclaim, and the set fills up.
    v.UnpinRange(v.PageLo(i), v.PageSpan(i, 1));
    i += run;
  }
  // Every page was flushed as it was written; nothing is left dirty.
}

__global__ void FillKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<clio::run::u32> v,
                           clio::run::u64 n, gy::YieldableView<> yv,
                           gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(FillCoro(v, n));
}

/** Read everything back and count mismatches. */
__device__ gy::YCoroMain CheckCoro(gv::DeviceVector<clio::run::u32> v,
                                   clio::run::u64 n,
                                   unsigned long long *bad) {
  for (clio::run::u64 i = 0; i < n;) {
    co_await v.Fetch(0, v.PageLo(i), v.PageSpan(i, 1));
    auto h = co_await v.HoldPage(i, n - i);
    for (clio::run::u64 k = threadIdx.x; k < h.run(); k += blockDim.x) {
      if (h[i + k] != static_cast<clio::run::u32>((i + k) * 7 + 1)) {
        atomicAdd(bad, 1ull);
      }
    }
    const clio::run::u64 run = h.run();
    v.UnpinRange(v.PageLo(i), v.PageSpan(i, 1));
    i += run;
  }
}

__global__ void CheckKernel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<clio::run::u32> v,
                            clio::run::u64 n, unsigned long long *bad,
                            gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(CheckCoro(v, n, bad));
}

/**
 * Multi-block: block b owns elements [b*per, (b+1)*per). `per` is a whole
 * number of pages, so no two blocks ever touch the same page -- the page
 * table is partitioned per block, and a page resident in two blocks at
 * once would give two independent copies of the same bytes.
 *
 * The slice is keyed off yv.Block(), NOT blockIdx.x: the driver relaunches a
 * COMPACTED grid of whichever blocks are still pending, so blockIdx.x is not
 * stable across resumes and a block would come back owning someone else's
 * slice.
 */
__device__ gy::YCoroMain MultiFillCoro(gv::DeviceVector<clio::run::u32> v,
                                       clio::run::u64 per,
                                       clio::run::u32 block) {
  const clio::run::u64 base = static_cast<clio::run::u64>(block) * per;
  for (clio::run::u64 i = 0; i < per;) {
    clio::run::u64 run = 0;
    {
      co_await v.Fetch(0, v.PageLo(base + i), v.PageSpan(base + i, 1));
      auto h = co_await v.HoldPage(base + i, per - i, /*write=*/true);
      run = h.run();
      for (clio::run::u64 k = threadIdx.x; k < run; k += blockDim.x) {
        h[base + i + k] = static_cast<clio::run::u32>((base + i + k) * 7 + 1);
      }
    }   // guard dies: unpinned, still dirty
    // Flush as we go -- see FillCoro. With 2 slots per block this vector is
    // oversubscribed, so leaving pages dirty would fill the table and the
    // fault could not be served without a writeback nobody asked for.
    co_await v.Flush(0, base + i, run);
    v.UnpinRange(v.PageLo(base + i), v.PageSpan(base + i, 1));
    i += run;
  }
}

__global__ void MultiFillKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<clio::run::u32> v,
                                clio::run::u64 per, gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(MultiFillCoro(v, per, yv.Block()));
}

__device__ gy::YCoroMain MultiCheckCoro(gv::DeviceVector<clio::run::u32> v,
                                        clio::run::u64 per,
                                        unsigned long long *bad,
                                        clio::run::u32 block) {
  const clio::run::u64 base = static_cast<clio::run::u64>(block) * per;
  for (clio::run::u64 i = 0; i < per;) {
    co_await v.Fetch(0, v.PageLo(base + i), v.PageSpan(base + i, 1));
    auto h = co_await v.HoldPage(base + i, per - i);
    for (clio::run::u64 k = threadIdx.x; k < h.run(); k += blockDim.x) {
      if (h[base + i + k] !=
          static_cast<clio::run::u32>((base + i + k) * 7 + 1)) {
        atomicAdd(bad, 1ull);
      }
    }
    const clio::run::u64 run = h.run();
    v.UnpinRange(v.PageLo(base + i), v.PageSpan(base + i, 1));
    i += run;
  }
}

__global__ void MultiCheckKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceVector<clio::run::u32> v,
                                 clio::run::u64 per, unsigned long long *bad,
                                 gy::YieldableView<> yv,
                                 gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(MultiCheckCoro(v, per, bad, yv.Block()));
}

#if !CTP_IS_DEVICE_PASS
TEST_CASE("gpu_vector: write, flush, read back", "[gpu_vector][smoke]") {
  // The gpu2cpu queue defaults to SIXTEEN entries (gpu_queue_depth_ = 16).
  // Every faulting block has a task in flight, so 128 blocks overflow it and
  // the kernel waits on work that was never queued -- measured: 64 blocks
  // pass, 128 hangs. Size it for the block counts this test drives.
  {
    std::ofstream cfg("gpu_vector_test.yaml");
    REQUIRE(cfg.is_open());
    // The compose section is NOT optional. Without it the CTE pool is built
    // from default CreateParams and has NO storage targets, so the first page
    // writeback has nowhere to land and never completes -- and back when this
    // test faulted through the BLOCKING hold, its kernel spun on that
    // writeback forever and the host hung in cudaDeviceSynchronize. The test
    // did not fail, it wedged, which is why it read as a flake rather than a
    // missing config.
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
        << "      - path: \"ram::gv_smoke_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"256MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_test.yaml", 1);
  }

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  clio::run::IpcManagerGpuInfo gpu_info{};
  gpu_info = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  // 4 pages of capacity for a 4-page vector: everything fits, so no
  // eviction should be needed.
  gv::Vector<clio::run::u32> probe("gv_probe", {0}, kPageBytes,
                                   /*nblocks=*/1, /*pages_per_block=*/2,
                                   kElems);
  std::fprintf(stderr, "[probe] second vector constructed OK\n");

  gv::Vector<clio::run::u32> vec("gv_smoke", {0}, kPageBytes,
                                 /*nblocks=*/1, /*pages_per_block=*/4, kElems);

  RunYieldable(1, vec, [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
    FillKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu_info, vec.GetDevice(0), kElems, vw, sv); });
  ctp::GpuApi::Synchronize();

  unsigned long long *d_bad = nullptr;
  d_bad = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_bad)>>(sizeof(unsigned long long));
  ctp::GpuApi::Memset(d_bad, 0, sizeof(unsigned long long));

  RunYieldable(1, vec, [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
    CheckKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu_info, vec.GetDevice(0), kElems, d_bad, vw, sv); });
  ctp::GpuApi::Synchronize();

  unsigned long long bad = 1;
  ctp::GpuApi::Memcpy(&bad, d_bad, sizeof(bad));
  ctp::GpuApi::Free(d_bad);
  std::fprintf(stderr, "[smoke] mismatches=%llu / %llu\n",
               (unsigned long long) bad, (unsigned long long) kElems);
  REQUIRE(bad == 0);

  // ---- oversubscribed: 16 pages through 4 slots ------------------------
  //
  // Every page past the fourth forces EvictPages, and every victim is
  // DIRTY, so its bytes must be written back before the slot is reused.
  // Reading the whole thing back therefore exercises eviction write-back
  // and slot reuse, not just the cache.
  //
  // Same runtime deliberately: CLIO_INIT is once per PROCESS, and a second
  // TEST_CASE that re-inits simply hangs.
  {
    const clio::run::u64 n = 16 * 1024;
    gv::Vector<clio::run::u32> big("gv_evict", {0}, kPageBytes,
                                   /*nblocks=*/1, /*pages_per_block=*/4, n);

    RunYieldable(1, big, [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
      FillKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu_info, big.GetDevice(0), n, vw, sv); });
    ctp::GpuApi::Synchronize();

    unsigned long long *d2 = nullptr;
    d2 = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d2)>>(sizeof(unsigned long long));
    ctp::GpuApi::Memset(d2, 0, sizeof(unsigned long long));
    RunYieldable(1, big, [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
      CheckKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu_info, big.GetDevice(0), n, d2, vw, sv); });
    ctp::GpuApi::Synchronize();

    unsigned long long b2 = 1;
    ctp::GpuApi::Memcpy(&b2, d2, sizeof(b2));
    ctp::GpuApi::Free(d2);
    std::fprintf(stderr, "[evict] mismatches=%llu / %llu\n",
                 (unsigned long long) b2, (unsigned long long) n);
    REQUIRE(b2 == 0);
  }

  // ---- many blocks ------------------------------------------------------
  // Each block drives its own slice. Slices are page-aligned so blocks never
  // contend for a PAGE -- but they do contend for a SET, which is the new
  // sizing rule: distinct pages hash into shared sets, so associativity must
  // cover however many blocks can hold a page of one set at the same time.
  // At 2 frames per set a third block finds them both pinned and the cache
  // says so ("set=0 full for page 11: 2 of 2 frames pinned"). 8 is the
  // measured floor here.
  //
  // Runs to 128 blocks, the spec's maximum. Each block gets as many cache
  // slots as pages, so this axis measures BLOCK COUNT alone.
  //
  // Oversubscription combined with HIGH block counts reaches 64 (covered
  // below); 128 still hangs and is not diagnosed. The abort that used to
  // appear here from ~32 blocks up is fixed: the host task copy is now keyed
  // to its device slot instead of freed in SendOut, and the device completion
  // flag is written LAST so a re-submitted slot cannot race SendOut's tail.
  for (unsigned nb : {8u, 32u, 64u, 128u}) {
    const clio::run::u64 per = 2 * 1024;            // 2 pages per block
    const clio::run::u64 n = per * nb;
    gv::Vector<clio::run::u32> mv("gv_multi" + std::to_string(nb), {0},
                                  kPageBytes, /*nsets=*/nb,
                                  /*frames_per_set=*/8, n);

    RunYieldable(nb, mv, [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
      MultiFillKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu_info, mv.GetDevice(0), per, vw, sv); });
    ctp::GpuApi::Synchronize();

    unsigned long long *d3 = nullptr;
    d3 = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d3)>>(sizeof(unsigned long long));
    ctp::GpuApi::Memset(d3, 0, sizeof(unsigned long long));
    RunYieldable(nb, mv, [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
      MultiCheckKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu_info, mv.GetDevice(0), per, d3, vw, sv); });
    ctp::GpuApi::Synchronize();

    unsigned long long b3 = 1;
    ctp::GpuApi::Memcpy(&b3, d3, sizeof(b3));
    ctp::GpuApi::Free(d3);
    std::fprintf(stderr, "[multi-%u] mismatches=%llu / %llu\n", nb,
                 (unsigned long long) b3, (unsigned long long) n);
    REQUIRE(b3 == 0);
  }

  // ---- many blocks AND oversubscribed ----------------------------------
  // Both pressures at once: 64 blocks, each walking 4 pages through 2 slots,
  // so every block evicts and writes back mid-kernel while 63 others do the
  // same. 128 here still hangs, and the runtime's own heartbeat says why it
  // is NOT a producer-side problem: processed= freezes while outstanding=91
  // sits queued with blocked=0, i.e. the CPU workers stall with work waiting.
  // That points at a worker blocking in a CUDA call while 128 resident blocks
  // spin, not at anything the vector does. Switching the CTE tier to pinned
  // memory (to make the transfers pure DMA) was tried and made it worse --
  // it also broke the non-evicting 128 case -- so that is not the cause
  // either. Raising num_threads 4 -> 16, on the theory that every
  // RecvIn/SendOut blocks its worker in a CUDA memcpy and 128 concurrent
  // faults simply outnumber the workers, also did not help. Left at 64
  // pending a real diagnosis.
  //
  // Thread states during the hang (from /proc, since ptrace is restricted
  // here): 31 in ep_poll, FOUR in poll_schedule_timeout, 2 in
  // hrtimer_nanosleep, 2 futex, 2 running. Nothing is blocked in an
  // uninterruptible driver ioctl, so the workers are not deadlocked on the
  // GPU -- they are sitting in CUDA's polling wait for an operation that
  // never completes, while 128 resident blocks spin. That is the thread to
  // pull next: which GPU operation is outstanding, and why the device never
  // retires it.
  //
  // Also eliminated: launching the faulting kernel on a NON-BLOCKING stream
  // instead of the legacy default one. Every stream the runtime creates is
  // already cudaStreamNonBlocking, so nothing waits on the legacy stream, and
  // the hang is unchanged either way.
  {
    const unsigned nb = 64u;
    const clio::run::u64 per = 4 * 1024;
    const clio::run::u64 n = per * nb;
    // Associativity 8 for the same reason as above; oversubscription now
    // comes from the BYTE BUDGET (half the frame array) against 4 pages per
    // block, not from a two-slot table.
    gv::Vector<clio::run::u32> ov("gv_multi_ov", {0}, kPageBytes, nb,
                                  /*frames_per_set=*/8, n);
    RunYieldable(nb, ov, [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
      MultiFillKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu_info, ov.GetDevice(0), per, vw, sv); });
    ctp::GpuApi::Synchronize();

    unsigned long long *d4 = nullptr;
    d4 = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d4)>>(sizeof(unsigned long long));
    ctp::GpuApi::Memset(d4, 0, sizeof(unsigned long long));
    RunYieldable(nb, ov, [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
      MultiCheckKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu_info, ov.GetDevice(0), per, d4, vw, sv); });
    ctp::GpuApi::Synchronize();
    unsigned long long b4 = 1;
    ctp::GpuApi::Memcpy(&b4, d4, sizeof(b4));
    ctp::GpuApi::Free(d4);
    std::fprintf(stderr, "[multi-oversub-64] mismatches=%llu / %llu\n",
                 (unsigned long long) b4, (unsigned long long) n);
    REQUIRE(b4 == 0);
  }

}

#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()
