/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Does an asynchronous flush overlap with compute? The simplest possible test.
 *
 * Each iteration a block spins for a fixed time, writes a REGION, and flushes
 * that region. The flush is block level: one BeginFlush over the whole region,
 * which issues a put per page and leaves them all in flight, so a large region
 * amortises the per-put round trip instead of paying it once per 64 KB.
 *
 * Nothing is read, nothing is faulted, nothing is evicted in the timed loop --
 * the cache is sized to hold every region at once and everything is made
 * resident before timing starts. The ONLY I/O measured is the write-back.
 *
 *   sync    spin, write, BeginFlush, WaitFlush        -> spin + flush
 *   async   spin, collect PREVIOUS flush, write,      -> max(spin, flush)
 *           BeginFlush, continue
 *
 * Both baselines are taken the same way as the real runs -- spin with no I/O,
 * and I/O with no spin -- so "sum" and "max" are measured quantities and the
 * verdict is arithmetic. Written bytes are verified from the host afterwards;
 * a flush that loses data fails instead of posting a speedup.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

/** Per-lane yield frame; coroutine frames are compiler-laid-out and far
 *  larger than the hand-packed macro ones. */
#if defined(CLIO_YIELD_CORO)
static constexpr u32 kYieldLaneBytes = 4096;
#else
static constexpr u32 kYieldLaneBytes = 256;
#endif

/** Cap on a single host-side verification read. Generous next to a normal
 *  read (sub-millisecond even from the NVMe tier) so it only ever fires on a
 *  genuinely stuck task, never on a slow one. */
static constexpr float kVerifyTimeoutSec = 120.0f;

/** True when the kernels below are the yieldable coroutine forms. */
#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
#define GV_FLUSH_CORO 1
#endif

/** Value written at index i on pass `pass`. */
CTP_INLINE_CROSS_FUN u32 Val(u64 i, u32 pass) {
  return static_cast<u32>(i * 2654435761u + pass * 2246822519u + 1u);
}

/** Busy-wait `us` microseconds of SM time. Identical in both kernels. */
CTP_GPU_FUN void Spin(u32 us, u64 clock_khz) {
  if (us == 0) return;
  const long long ticks =
      static_cast<long long>(us) * static_cast<long long>(clock_khz) / 1000ll;
  const long long t0 = clock64();
  while (clock64() - t0 < ticks) {
    // Pure spin, no __nanosleep: sleeping lets the clock drop, which makes the
    // same "microseconds" cost different wall time depending on what else the
    // kernel is doing -- exactly the confound this benchmark exists to avoid.
  }
}

#if defined(GV_FLUSH_CORO)
/**
 * Wait for this block's outstanding transfers by PARKING, not spinning.
 *
 * DeviceVector::WaitFlush spins on the device (AwaitPut in a loop). That
 * deadlocks: the kernel stays resident spinning while the completion it is
 * waiting for needs the host side to make progress, and the host is inside
 * cuCtxSynchronize waiting for the kernel. Every kernel in this benchmark
 * used to do exactly that, and every configuration hung in the warm pass --
 * including the stock defaults -- with the main thread parked in
 * cuCtxSynchronize forever.
 *
 * The yieldable form suspends the block back to the host driver instead, so
 * the runtime gets to run, complete the put, and resume us. Waiting on
 * AnyTransferInFlight rather than a range is deliberate: it is the same
 * condition HoldPageCoro's writeback step uses, and it cannot miss a put
 * that a concurrent eviction started.
 */
__device__ gy::YCoroTask FlushWaitCoro(gv::DeviceVector<u32> &v) {
  CLIO_CO_YIELD_WHEN((v.ReapFlushed(), v.ReapFetched()),
                     v.AnyTransferInFlight(), v.FlushWaitTag());
}

/** Write one page of a region. Shared by the warm and timed coroutines. */
__device__ void WritePage(gv::DeviceVector<u32> &v, u64 poff, u32 pass) {
  for (u64 i = threadIdx.x; i < v.h_->elems_per_page_; i += blockDim.x) {
    v[poff + i] = Val(poff + i, pass);
  }
  __syncthreads();
}

/** Touch and flush every page, so the timed loop starts from a known state. */
__device__ gy::YCoroMain WarmCoro(gv::DeviceVector<u32> v, u64 iters,
                                  u64 pages_per_region, u32 block) {
  const u64 region_elems = pages_per_region * v.h_->elems_per_page_;
  const u64 block_base = static_cast<u64>(block) * iters * region_elems;
  for (u64 it = 0; it < iters; ++it) {
    const u64 off = block_base + it * region_elems;
    for (u64 pg = 0; pg < pages_per_region; ++pg) {
      const u64 poff = off + pg * v.h_->elems_per_page_;
      u64 run = 0;
      co_await v.HoldPage(poff, v.h_->elems_per_page_, &run);
      WritePage(v, poff, 0u);
    }
    if (threadIdx.x == 0) v.BeginFlush(off, region_elems);
    __syncthreads();
    co_await FlushWaitCoro(v);
  }
}

__global__ void WarmKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<u32> v, u64 iters,
                           u64 pages_per_region, gy::YieldableView<> yv,
                           gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(WarmCoro(v, iters, pages_per_region, yv.Block()));
}
#else
/** Touch and flush every page, so the timed loop faults nothing. */
__global__ void WarmKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<u32> v, u64 iters,
                           u64 pages_per_region, gy::YieldableView<> yv,
                           gy::YieldStackView ys) {
  (void) ys;
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  const u64 region_elems = pages_per_region * v.h_->elems_per_page_;
  const u64 block_base = static_cast<u64>(yv.Block()) * iters * region_elems;
  for (u64 it = 0; it < iters; ++it) {
    {
      const u64 off = block_base + it * region_elems;
      for (u64 pg = 0; pg < pages_per_region; ++pg) {
        const u64 poff = off + pg * v.h_->elems_per_page_;
        v.HoldPage(poff, v.h_->elems_per_page_);
        for (u64 i = threadIdx.x; i < v.h_->elems_per_page_; i += blockDim.x) {
          v[poff + i] = Val(poff + i, 0u);
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        v.BeginFlush(off, region_elems);
        v.WaitFlush(off, region_elems);
      }
      __syncthreads();
      }
  }
}
#endif  // GV_FLUSH_CORO

/**
 * spin -> write region -> flush region, `iters` times.
 *
 * @param async 0: BeginFlush then WaitFlush immediately.
 *              1: BeginFlush and collect it on the NEXT iteration, after that
 *                 iteration's spin, so the transfer runs under the spin.
 */
#if defined(GV_FLUSH_CORO)
__device__ gy::YCoroMain SpinWriteFlushCoro(gv::DeviceVector<u32> v, u64 iters,
                                            u64 pages_per_region, u32 spin_us,
                                            u64 clock_khz, int do_write,
                                            u32 pass, int async, u32 block) {
  const u64 region_elems = pages_per_region * v.h_->elems_per_page_;
  const u64 block_base = static_cast<u64>(block) * iters * region_elems;
  bool pending = false;

  for (u64 it = 0; it < iters; ++it) {
    // ---- compute ----
    Spin(spin_us, clock_khz);
    __syncthreads();

    // ---- async: collect the previous flush, which has been running under
    //      the spin above. Sync collects it below, before the next spin --
    //      that placement is the entire difference the benchmark measures.
    if (async && pending) {
      co_await FlushWaitCoro(v);
      pending = false;
    }

    if (do_write) {
      const u64 off = block_base + it * region_elems;
      // Page at a time, so the whole block is inside one page at any moment --
      // the granularity the vector's paging contract assumes.
      for (u64 pg = 0; pg < pages_per_region; ++pg) {
        const u64 poff = off + pg * v.h_->elems_per_page_;
        // Required: operator[] indexes the HELD page and does no resolution,
        // so without this last_page_ is null and the write dereferences it.
        u64 run = 0;
        co_await v.HoldPage(poff, v.h_->elems_per_page_, &run);
        WritePage(v, poff, pass);
      }
      // ---- block-level flush: ONE call covering the whole region ----
      if (threadIdx.x == 0) v.BeginFlush(off, region_elems);
      __syncthreads();
      if (async) {
        pending = true;
      } else {
        co_await FlushWaitCoro(v);
      }
    }
  }
  if (pending) co_await FlushWaitCoro(v);
}

__global__ void SpinWriteFlushKernel(clio::run::IpcManagerGpuInfo info,
                                     gv::DeviceVector<u32> v, u64 iters,
                                     u64 pages_per_region, u32 spin_us,
                                     u64 clock_khz, int do_write, u32 pass,
                                     int async, gy::YieldableView<> yv,
                                     gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SpinWriteFlushCoro(v, iters, pages_per_region, spin_us,
                                    clock_khz, do_write, pass, async,
                                    yv.Block()));
}
#else
__global__ void SpinWriteFlushKernel(clio::run::IpcManagerGpuInfo info,
                                     gv::DeviceVector<u32> v, u64 iters,
                                     u64 pages_per_region, u32 spin_us,
                                     u64 clock_khz, int do_write, u32 pass,
                                     int async, gy::YieldableView<> yv,
                                     gy::YieldStackView ys) {
  (void) ys;
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  const u64 region_elems = pages_per_region * v.h_->elems_per_page_;
  const u64 block_base = static_cast<u64>(yv.Block()) * iters * region_elems;
  long long prev = -1;

  for (u64 it = 0; it < iters; ++it) {
    {
      // ---- compute ----
      Spin(spin_us, clock_khz);
      __syncthreads();

      // ---- async: collect the previous flush, which has been running under
      //      the spin above ----
      if (async && do_write && threadIdx.x == 0 && prev >= 0) {
        v.WaitFlush(block_base + static_cast<u64>(prev) * region_elems,
                    region_elems);
      }
      __syncthreads();

      if (do_write) {
        const u64 off = block_base + it * region_elems;
        // Page at a time, so the whole block is inside one page at any moment --
        // the granularity the vector's paging contract assumes.
        for (u64 pg = 0; pg < pages_per_region; ++pg) {
          const u64 poff = off + pg * v.h_->elems_per_page_;
          // Required: operator[] indexes the HELD page and does no resolution,
          // so without this last_page_ is null and the write dereferences it.
          v.HoldPage(poff, v.h_->elems_per_page_);
          for (u64 i = threadIdx.x; i < v.h_->elems_per_page_; i += blockDim.x) {
            v[poff + i] = Val(poff + i, pass);
          }
          __syncthreads();
        }
        // ---- block-level flush: ONE call covering the whole region ----
        if (threadIdx.x == 0) {
          v.BeginFlush(off, region_elems);
          if (!async) {
            v.WaitFlush(off, region_elems);
          }
        }
        __syncthreads();
        prev = static_cast<long long>(it);
      }
      }
  }

  if (async == 1 && do_write && threadIdx.x == 0 && prev >= 0) {
    v.WaitFlush(block_base + static_cast<u64>(prev) * region_elems,
                region_elems);
  }
}
#endif  // GV_FLUSH_CORO

/**
 * The READ mirror of SpinWriteFlushKernel: spin, then read a region.
 *
 * sync  (async=0)  touch the region and let it demand-fault, page by page.
 * async (async=1)  BeginFetch every page of the NEXT region before reading
 *                  this one, so those transfers run under this region's spin
 *                  and read. RescorePage pins what is being fetched and
 *                  releases what has been consumed, so the prefetch's slot
 *                  claim evicts a finished page rather than one still in use.
 *
 * The cache holds exactly two regions, so a prefetched region can be resident
 * alongside the one being read and nothing else fits -- prefetching either
 * lands in time or it does not.
 */
__global__ void SpinReadPrefetchKernel(clio::run::IpcManagerGpuInfo info,
                                       gv::DeviceVector<u32> v, u64 iters,
                                       u64 pages_per_region, u32 spin_us,
                                       u64 clock_khz, int async,
                                       unsigned long long *sum,
                                       unsigned long long *bad,
                                       unsigned long long *first_bad,
                                       unsigned long long *bad_off,
                                       u32 *bad_got) {
  CLIO_GPU_INIT(info, nullptr);
  const u64 region_elems = pages_per_region * v.h_->elems_per_page_;
  const u64 block_base = static_cast<u64>(blockIdx.x) * iters * region_elems;
  unsigned long long acc = 0;

  for (u64 it = 0; it < iters; ++it) {
    Spin(spin_us, clock_khz);
    __syncthreads();

    // Ask for the next region BEFORE consuming this one.
    if (async && threadIdx.x == 0 && it + 1 < iters) {
      const u64 nxt = block_base + (it + 1) * region_elems;
      for (u64 pg = 0; pg < pages_per_region; ++pg) {
        const u64 pn = v.PageOf(nxt + pg * v.h_->elems_per_page_);
        v.RescorePage(pn, 1000.0f);
        v.BeginFetch(pn);
      }
    }
    __syncthreads();

    const u64 off = block_base + it * region_elems;
    for (u64 pg = 0; pg < pages_per_region; ++pg) {
      const u64 poff = off + pg * v.h_->elems_per_page_;
      v.HoldPage(poff, v.h_->elems_per_page_);
      unsigned long long local = 0;
      unsigned long long wrong = 0;
      for (u64 i = threadIdx.x; i < v.h_->elems_per_page_; i += blockDim.x) {
        const u32 got = v.at(poff + i);
        local += got;
        // Compare element by element: a single global sum says only THAT
        // something is wrong, never which page or offset.
        if (got != Val(poff + i, 0u)) {
          ++wrong;
          atomicMin(first_bad, static_cast<unsigned long long>(v.PageOf(poff)));
          // Capture the actual bytes. Val is invertible mod 2^32, so the host
          // can recover WHICH index produced this value -- that turns "wrong"
          // into "came from offset X", which is the difference between
          // guessing and knowing.
          const unsigned long long prev =
              atomicMin(bad_off, static_cast<unsigned long long>(poff + i));
          if (poff + i < prev) *bad_got = got;
        }
      }
      acc += local;
      if (wrong) atomicAdd(bad, wrong);
      __syncthreads();
      // Consumed: lowest score makes it the next victim, freeing a slot for
      // the prefetch that follows.
      if (async && threadIdx.x == 0) {
        v.RescorePage(v.PageOf(poff), -1000.0f);
      }
      __syncthreads();
    }
  }
  atomicAdd(sum, acc);
}

/** Diagnostic: each slot's page_num and the first word of its buffer. */
__global__ void DumpSlotsKernel(gv::DeviceVector<u32> v,
                                unsigned long long *out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  // Block 0's slice sits at the front of the shared page table.
  const gv::Page *tbl = v.h_->pages_;
  for (clio::run::u32 i = 0; i < v.h_->pages_per_block_; ++i) {
    out[2 * i] = tbl[i].page_num;
    out[2 * i + 1] =
        (tbl[i].page_num == gv::kNoPage || tbl[i].data == nullptr)
            ? 0ull
            : static_cast<const u32 *>(tbl[i].data)[0];
  }
}

#if !CTP_IS_DEVICE_PASS

namespace {

struct Args {
  u32 blocks = 1;
  u64 iters = 4;         // regions written per block
  u32 spin_us = 2000;    // compute per iteration
  u64 page_kb = 1024;    // page granularity
  u64 flush_mb = 128;    // region flushed per iteration
  u32 threads = 256;
  u32 repeat = 3;
  bool read = false;     // read+prefetch benchmark instead of write+flush
  // Per-block page-cache slots. 0 = the historical behaviour: write mode sizes
  // the cache to the WHOLE working set so nothing is ever evicted and the
  // explicit flush is the only I/O. Setting it smaller makes the cache a real
  // cache -- the block must evict to make room, which is the only way the
  // lower tiers (host DRAM, NVMe) ever get exercised.
  u32 pages_per_block = 0;
};

/**
 * Runs a yieldable kernel to completion, relaunching it as blocks suspend.
 *
 * Both Reset() calls are required. RunToCompletion does NOT reset -- only the
 * constructor does -- so a reused runner whose driver still reads "done" from
 * the previous call skips the launch entirely and reports an instant, empty
 * success.
 */
class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [] {}, /*max_rounds=*/2000000);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char **argv) {
  Args a;
  // GV_TIER_TYPE=ram|pinned. Pageable ("ram") makes cudaMemcpyAsync
  // host-synchronous and forces the driver to stage every device transfer
  // through an internal pinned buffer -- a D2H followed by an H2H.
  const char *tier_type_env = getenv("GV_TIER_TYPE");
  const std::string tier_type = tier_type_env ? tier_type_env : "ram";
  // Worker doze experiment: sequential single-block ops measured ~30ms each
  // while flooded ops cost ~35us. If that gap is workers sleeping between
  // ops, making them busy-wait the whole run should erase it.
  long busy_us = 10000;
  if (const char *e = getenv("GV_BUSY_US")) busy_us = atol(e);
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    auto next = [&]() -> const char * {
      return (i + 1 < argc) ? argv[++i] : "0";
    };
    if (f == "--blocks") a.blocks = std::atoi(next());
    else if (f == "--iters") a.iters = std::atoll(next());
    else if (f == "--spin-us") a.spin_us = std::atoi(next());
    else if (f == "--page-kb") a.page_kb = std::atoll(next());
    else if (f == "--flush-mb") a.flush_mb = std::atoll(next());
    else if (f == "--threads") a.threads = std::atoi(next());
    else if (f == "--repeat") a.repeat = std::atoi(next());
    else if (f == "--pages-per-block") a.pages_per_block = std::atoi(next());
    else if (f == "--read") a.read = true;
    else if (f == "--help") {
      std::printf(
          "usage: %s [--blocks N] [--iters N] [--spin-us N] [--page-kb N] "
          "[--flush-mb N] [--threads N] [--repeat N] "
          "[--pages-per-block N]\n"
          "tiers (MB, 0 = omit): GV_HBM_MB GV_DRAM_MB GV_NVME_MB "
          "GV_NVME_PATH\n", argv[0]);
      return 0;
    }
  }

  // The read+prefetch path has NOT been converted to the yieldable form: its
  // kernel still calls the blocking HoldPage/AwaitFetch, which spins on the
  // device while the completion it waits for needs the host to run -- the
  // deadlock that hung every configuration of this benchmark, including the
  // stock defaults. Refusing is better than reproducing that hang; the fix is
  // the same conversion the write path just got.
  if (a.read) {
    std::fprintf(stderr,
                 "--read is unavailable: SpinReadPrefetchKernel still uses the "
                 "blocking HoldPage path and deadlocks a resident kernel. Only "
                 "the write+flush path has been converted to the yieldable "
                 "coroutine form.\n");
    return 1;
  }

  const u64 page_bytes = a.page_kb * 1024;
  const u64 page_elems = page_bytes / sizeof(u32);
  const u64 flush_bytes = a.flush_mb * 1024ull * 1024ull;
  const u64 pages_per_region = flush_bytes / page_bytes;
  if (pages_per_region == 0) {
    std::fprintf(stderr, "flush region smaller than one page\n");
    return 1;
  }
  // The vector's LOGICAL size is the whole working set in both modes -- it is
  // what the kernel walks. The CACHE size is what differs: write mode holds
  // every region so the flush is the only I/O; read mode holds just two, so
  // the working set exceeds it and every region must be faulted in.
  //
  // These were conflated (n was derived from the cache size), which left the
  // vector claiming a size of 2 regions while the kernel read 8.
  const u64 pages_per_block_total = pages_per_region * a.iters;
  u64 slots = a.read ? (pages_per_region * 2) : pages_per_block_total;
  // An explicit cache size wins over both defaults. Capping at the working set
  // keeps a too-large request from allocating slots that can never be filled.
  if (a.pages_per_block > 0) {
    slots = a.pages_per_block;
    if (slots > pages_per_block_total) slots = pages_per_block_total;
  }
  const u64 n = page_elems * pages_per_block_total * a.blocks;
  const double resident_mb =
      static_cast<double>(slots * page_bytes * a.blocks) / (1024.0 * 1024.0);
  const double working_mb = static_cast<double>(pages_per_block_total *
                                                page_bytes * a.blocks) /
                            (1024.0 * 1024.0);

  // ---- tier ladder -------------------------------------------------------
  // VRAM / host DRAM / NVMe, each optional so the historical single-host-tier
  // invocation still works (GV_DRAM_MB alone reproduces it).
  //
  // SCORE DIRECTION DEPENDS ON THE BLOB SCORE, AND THIS PATH IS NOT THE GNN
  // PATH. MaxBwDpe splits targets into preferred (target_score_ <= blob_score)
  // and fallback, then sorts the PREFERRED group by target_score_ DESCENDING
  // -- highest score wins -- with bandwidth breaking ties only.
  //
  // The gpu_vector flushes a touched page at score 1.0 (SubmitPut clamps the
  // page's accumulated access score into [0,1], and any written page has
  // accumulated at least 1.0). At blob_score 1.0 EVERY tier is preferred, so
  // the ladder is decided purely by descending score. A 0.0/0.6/1.0 ladder --
  // correct for the GNN path, which puts at 0.5 and relies on kHBM being the
  // only target at or below it -- is therefore exactly BACKWARDS here: it was
  // measured putting all 128MB on NVMe with a 512MB VRAM tier sitting empty.
  //
  // Hence VRAM 1.0 / DRAM 0.6 / NVMe 0.0: descending order walks the ladder
  // top-down and spills to the next tier only when one fills.
  const double kVramScore = 1.0, kDramScore = 0.6, kNvmeScore = 0.0;
  auto env_mb = [](const char *k, long dflt) -> long {
    const char *e = getenv(k);
    return e ? std::atol(e) : dflt;
  };
  const long hbm_mb = env_mb("GV_HBM_MB", 0);
  const long dram_mb = env_mb("GV_DRAM_MB", 12288);
  const long nvme_mb = env_mb("GV_NVME_MB", 0);
  const char *nvme_path_env = getenv("GV_NVME_PATH");
  const std::string nvme_path =
      nvme_path_env ? nvme_path_env : std::string("/tmp/gv_flush_nvme.dat");
  if (hbm_mb <= 0 && dram_mb <= 0 && nvme_mb <= 0) {
    std::fprintf(stderr, "no tiers configured (GV_HBM_MB/GV_DRAM_MB/GV_NVME_MB)\n");
    return 1;
  }
  {
    std::ofstream cfg("gpu_vector_flush.yaml");
    cfg << "networking:\n  port: 9437\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n"
        << "  first_busy_wait: " << busy_us << "\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n"
        << "    storage:\n";
    if (hbm_mb > 0) {
      cfg << "      - path: \"hbm::gv_flush_hbm\"\n"
          << "        bdev_type: \"hbm\"\n"
          << "        capacity_limit: \"" << hbm_mb << "MB\"\n"
          << "        score: " << kVramScore << "\n";
    }
    if (dram_mb > 0) {
      cfg << "      - path: \"" << tier_type << "::gv_flush_tier\"\n"
          << "        bdev_type: \"" << tier_type << "\"\n"
          << "        capacity_limit: \"" << dram_mb << "MB\"\n"
          << "        score: " << kDramScore << "\n";
    }
    if (nvme_mb > 0) {
      cfg << "      - path: \"" << nvme_path << "\"\n"
          << "        bdev_type: \"file\"\n"
          << "        capacity_limit: \"" << nvme_mb << "MB\"\n"
          << "        score: " << kNvmeScore << "\n";
    }
    cfg << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_flush.yaml", 1);
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  // Which tier actually holds the bytes. Storage targets take PoolId(512+i, 1)
  // in DECLARATION order, so the index depends on which tiers were enabled.
  // Declaring a tier is not evidence it received anything -- the DPE ranks by
  // a predicted bandwidth model and has been measured leaving a correctly
  // sized HBM tier empty -- so the split is read from the bdevs themselves.
  struct TierProbe {
    const char *name;
    clio::run::bdev::Client cli;
    clio::run::u64 free0 = 0;
  };
  std::vector<TierProbe> tiers;
  {
    clio::run::u32 idx = 512;
    if (hbm_mb > 0) tiers.push_back({"VRAM", clio::run::bdev::Client(clio::run::PoolId(idx++, 1))});
    if (dram_mb > 0) tiers.push_back({"DRAM", clio::run::bdev::Client(clio::run::PoolId(idx++, 1))});
    if (nvme_mb > 0) tiers.push_back({"NVMe", clio::run::bdev::Client(clio::run::PoolId(idx++, 1))});
    for (auto &t : tiers) {
      auto s = t.cli.AsyncGetStats();
      s.Wait();
      t.free0 = s->remaining_size_;
    }
  }
  // STREAM POOL ACCOUNTING. The runtime deadlocks when DeviceAwareMemcpy's
  // 64-stream pool empties: BorrowStream then returns null forever (it must
  // not create a stream while a kernel is resident) and the worker spins,
  // stops draining its lane, and the faulting kernel waits on tasks that
  // never run. Printing the accounting at the END of a HEALTHY run is what
  // separates the two possible causes: outstanding must return to 0 when the
  // workload is idle. A non-zero value here means streams LEAK, and the pool
  // is simply running down until it hits zero -- which would make the hang a
  // matter of run length, not of concurrency.
  auto report_streams = []() {
    const long b = ctp::GpuApi::StreamBorrows().load();
    const long r = ctp::GpuApi::StreamReturns().load();
    const long w = ctp::GpuApi::StreamWarmed().load();
    std::printf("  STREAM POOL: warmed=%ld borrows=%ld returns=%ld "
                "outstanding=%ld%s\n", w, b, r, b - r,
                (b - r) > 0 ? "   <-- LEAK: not returned at idle" : "");
  };
  auto report_tiers = [&]() {
    std::printf("  TIER SPLIT:");
    u64 total_used = 0;
    for (auto &t : tiers) {
      auto s = t.cli.AsyncGetStats();
      s.Wait();
      const u64 used = t.free0 >= s->remaining_size_ ? t.free0 - s->remaining_size_ : 0;
      total_used += used;
      std::printf("  %s %lluMB", t.name, (unsigned long long) (used >> 20));
    }
    std::printf("   (total %lluMB)\n", (unsigned long long) (total_used >> 20));
  };

  // cudaDeviceProp::clockRate was removed in CUDA 13; the attribute query is
  // the supported way to ask and works on every version.
  int clock_khz_i = 0;
  cudaDeviceGetAttribute(&clock_khz_i, cudaDevAttrClockRate, 0);
  const u64 clock_khz = static_cast<u64>(clock_khz_i);

  std::printf(
      "gpu_vector block-level flush benchmark\n"
      "  blocks=%u threads=%u iters=%llu spin=%u us/iter\n"
      "  page=%lluKB  flush region=%lluMB (%llu pages/flush)\n"
      // Spell out BOTH scopes. "cache=1 pages/block (64MB)" was read as a
      // 64MB page; it is 1MB per block across 64 blocks. Same for the working
      // set, which is per-block x blocks.
      "  cache=%llu pages/block = %.0fMB per block, %.0fMB over %u blocks%s\n"
      "  each block walks %llu pages, so a 1-page cache faults %llu times "
      "per block (%llu total)\n"
      "  working set=%.0fMB\n"
      "  tiers: VRAM %ldMB / DRAM %ldMB / NVMe %ldMB\n",
      a.blocks, a.threads, (unsigned long long) a.iters, a.spin_us,
      (unsigned long long) a.page_kb, (unsigned long long) a.flush_mb,
      (unsigned long long) pages_per_region, (unsigned long long) slots,
      static_cast<double>(slots * page_bytes) / (1024.0 * 1024.0), resident_mb,
      a.blocks,
      (slots < pages_per_block_total)
          ? "  (cache < working set: blocks must EVICT)"
          : "  (fully resident: flush is the only I/O)",
      (unsigned long long) pages_per_block_total,
      (unsigned long long) pages_per_block_total,
      (unsigned long long) (pages_per_block_total * a.blocks), working_mb,
      hbm_mb, dram_mb, nvme_mb);

  gv::Vector<u32> vec("gv_flush", {0}, page_bytes, a.blocks,
                      static_cast<u32>(slots), n);
  vec.EnableStats();
  auto dev = vec.GetDevice(0);

  YieldRunner runner(a.blocks, a.threads);
  auto warm = [&]() {
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      WarmKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dev, a.iters,
                                                  pages_per_region, vw, sv);
    });
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::fprintf(stderr, "warm failed: %s\n",
                   cudaGetErrorString(cudaGetLastError()));
      std::exit(1);
    }
  };
  warm();

  // GV_PROBE_GET: time HOST-side GetBlob directly -- no GPU queue, no page
  // table, no kernel. Separates "GetBlob is slow" from "the GPU round trip or
  // the vector's per-fault table scans are slow".
  if (getenv("GV_PROBE_GET")) {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    std::vector<char> buf(page_bytes);
    auto probe = [&](const char *name, int n) {
      double best = 1e30, sum = 0;
      for (int i = 0; i < n; ++i) {
        const double t0 = NowMs();
        auto f = core.AsyncGetBlob(vec.TagId(), std::string(name), 0,
                                   page_bytes, 0, buf.data());
        f.Wait();
        (void) f->GetReturnCode();
        const double ms = NowMs() - t0;
        sum += ms;
        if (ms < best) best = ms;
      }
      std::printf("  host GetBlob %-12s x%d: avg %8.1f us  best %8.1f us\n",
                  name, n, sum * 1000.0 / n, best * 1000.0);
    };
    probe("p0", 100);            // exists (warm flushed it)
    probe("nonexistent", 100);   // missing
    clio::run::CLIO_RUNTIME_FINALIZE();
    return 0;
  }

  // ---------------- read + prefetch mode -----------------------------
  if (a.read) {
    unsigned long long *d_sum = nullptr;
    cudaMalloc(&d_sum, sizeof(unsigned long long));
    unsigned long long *d_bad = nullptr, *d_first = nullptr, *d_boff = nullptr;
    u32 *d_bgot = nullptr;
    cudaMalloc(&d_bad, sizeof(unsigned long long));
    cudaMalloc(&d_first, sizeof(unsigned long long));
    cudaMalloc(&d_boff, sizeof(unsigned long long));
    cudaMalloc(&d_bgot, sizeof(u32));
    auto rrun = [&](u32 spin, int async, unsigned long long *out) {
      double best = 1e30;
      for (u32 r = 0; r < a.repeat; ++r) {
        cudaMemset(d_sum, 0, sizeof(unsigned long long));
        cudaMemset(d_bad, 0, sizeof(unsigned long long));
        const unsigned long long big = ~0ull;
        cudaMemcpy(d_first, &big, sizeof(big), cudaMemcpyHostToDevice);
        cudaMemcpy(d_boff, &big, sizeof(big), cudaMemcpyHostToDevice);
        cudaMemset(d_bgot, 0, sizeof(u32));
        vec.ResetStats();
        cudaDeviceSynchronize();
        const double t0 = NowMs();
        SpinReadPrefetchKernel<<<a.blocks, a.threads>>>(
            gpu, dev, a.iters, pages_per_region, spin, clock_khz, async, d_sum,
            d_bad, d_first, d_boff, d_bgot);
        if (cudaDeviceSynchronize() != cudaSuccess) {
          std::fprintf(stderr, "read kernel failed: %s\n",
                       cudaGetErrorString(cudaGetLastError()));
          std::exit(1);
        }
        const double ms = NowMs() - t0;
        if (ms < best) best = ms;
      }
      cudaMemcpy(out, d_sum, sizeof(*out), cudaMemcpyDeviceToHost);
      unsigned long long nb = 0, fb = 0;
      cudaMemcpy(&nb, d_bad, sizeof(nb), cudaMemcpyDeviceToHost);
      cudaMemcpy(&fb, d_first, sizeof(fb), cudaMemcpyDeviceToHost);
      unsigned long long bo = 0; u32 bg = 0;
      cudaMemcpy(&bo, d_boff, sizeof(bo), cudaMemcpyDeviceToHost);
      cudaMemcpy(&bg, d_bgot, sizeof(bg), cudaMemcpyDeviceToHost);
      const auto st_now = vec.ReadStats(0);
      std::printf("    [%s spin=%u] wrong_elems=%llu first_bad_page=%lld "
                  "faults=%llu GET_ERRORS=%llu\n",
                  async ? "async" : "sync", spin, nb,
                  (fb == ~0ull) ? -1LL : (long long) fb,
                  (unsigned long long) st_now.faults,
                  (unsigned long long) st_now.get_errors);
      if (nb && bo != ~0ull) {
        // Invert Val: got = j*A + 1 (mod 2^32) => j = (got-1) * A^-1.
        const u32 A = 2654435761u;
        u32 inv = 1u;
        for (int k = 0; k < 6; ++k) inv *= (2u - A * inv);
        const u32 src = static_cast<u32>((bg - 1u) * inv);
        std::printf("      first bad offset=%llu got=%u -> that value is "
                    "Val(%u), i.e. offset %lld away (%.3f pages)\n",
                    bo, bg, src, (long long) src - (long long) bo,
                    ((double) src - (double) bo) / (double) page_elems);
      }
      return best;
    };

    // ---- PROBE: does fetched data ever reach device memory? -----------
    // Fault ONE region synchronously (pages 0..15 land in slots that held
    // pages 96..111 from the warm pass), wait 300ms after the kernel exits so
    // any straggling copy has long finished, then dump what is PHYSICALLY in
    // each slot. Data from pages 0..15 => copies land, readers raced them.
    // Data still from 96..111 => the copies never arrived at all.
    {
      cudaMemset(d_sum, 0, sizeof(unsigned long long));
      cudaMemset(d_bad, 0, sizeof(unsigned long long));
      const unsigned long long big2 = ~0ull;
      cudaMemcpy(d_first, &big2, sizeof(big2), cudaMemcpyHostToDevice);
      cudaMemcpy(d_boff, &big2, sizeof(big2), cudaMemcpyHostToDevice);
      cudaMemset(d_bgot, 0, sizeof(u32));
      SpinReadPrefetchKernel<<<a.blocks, a.threads>>>(
          gpu, dev, 1, pages_per_region, 0, clock_khz, 0, d_sum, d_bad,
          d_first, d_boff, d_bgot);
      if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "probe kernel failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return 1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      unsigned long long *d_dump = nullptr;
      cudaMalloc(&d_dump, 2 * slots * sizeof(unsigned long long));
      DumpSlotsKernel<<<1, 32>>>(dev, d_dump);
      cudaDeviceSynchronize();
      std::vector<unsigned long long> hd(2 * slots);
      cudaMemcpy(hd.data(), d_dump, 2 * slots * sizeof(unsigned long long),
                 cudaMemcpyDeviceToHost);
      const u32 A2 = 2654435761u;
      u32 inv2 = 1u;
      for (int k = 0; k < 6; ++k) inv2 *= (2u - A2 * inv2);
      unsigned long long nbad = 0;
      cudaMemcpy(&nbad, d_bad, sizeof(nbad), cudaMemcpyDeviceToHost);
      std::printf("  PROBE (1 region, sync, wrong_elems=%llu):\n", nbad);
      for (u64 i2 = 0; i2 < slots; ++i2) {
        const unsigned long long pn = hd[2 * i2];
        if (pn == ~0ull) continue;
        const u32 got0 = static_cast<u32>(hd[2 * i2 + 1]);
        const u32 src = (got0 - 1u) * inv2;
        const u64 src_page = src / page_elems;
        std::printf("    slot %2llu claims page %3llu -- buffer holds page "
                    "%llu%s\n",
                    (unsigned long long) i2, pn, (unsigned long long) src_page,
                    (pn == src_page) ? "" : "   <-- STALE");
      }
      cudaFree(d_dump);
      // Probe consumed region 0's warm state; restore it for the real runs.
      warm();
    }

    // The warm pass wrote Val(idx, 0) everywhere; that is what must come back.
    unsigned long long want = 0;
    const u64 total = static_cast<u64>(a.blocks) * a.iters * pages_per_region *
                      page_elems;
    for (u64 i = 0; i < total; ++i) want += Val(i, 0u);

    unsigned long long g_spin = 0, g_io = 0, g_sync = 0, g_async = 0;
    const double spin_only = rrun(a.spin_us, 0, &g_spin);
    const double io_only = rrun(0, 0, &g_io);
    const auto io_st = vec.ReadStats(0);
    const double sync_ms = rrun(a.spin_us, 0, &g_sync);
    const auto sync_st = vec.ReadStats(0);
    const double async_ms = rrun(a.spin_us, 1, &g_async);
    const auto async_st = vec.ReadStats(0);

    const bool ok = (g_sync == want && g_async == want && g_io == want);
    if (!ok) {
      std::fprintf(stderr,
                   "  want=%llu  read-only=%llu  sync=%llu  async=%llu\n"
                   "  (all three should equal want; which ones differ says "
                   "whether this is a prefetch bug or a warm/seed bug)\n",
                   want, g_io, g_sync, g_async);
    }
    const double mx = (spin_only > io_only) ? spin_only : io_only;
    std::printf(
        "\n  READ + PREFETCH  (cache holds 2 of %llu regions -- every region "
        "faults)\n"
        "  spin only        %8.2f ms\n"
        "  read only        %8.2f ms   (faults=%llu, %.0f MB/s)\n"
        "  ---------------------------------------------\n"
        "  sync  measured   %8.2f ms   (faults=%llu)\n"
        "  async measured   %8.2f ms   (faults=%llu prefetch=%llu landed=%llu "
        "late=%llu)  %s\n"
        "  ---------------------------------------------\n"
        "  sum  spin+read   %8.2f ms   <- what NO overlap costs\n"
        "  max  spin,read   %8.2f ms   <- what FULL overlap costs\n"
        "  speedup async    %8.2fx\n"
        "  read hidden      %8.1f %%\n",
        (unsigned long long) a.iters, spin_only, io_only,
        (unsigned long long) io_st.faults,
        (io_only > 0.0) ? working_mb / (io_only / 1000.0) : 0.0,
        sync_ms, (unsigned long long) sync_st.faults,
        async_ms, (unsigned long long) async_st.faults,
        (unsigned long long) async_st.prefetches,
        (unsigned long long) async_st.prefetch_hits,
        (unsigned long long) async_st.prefetch_late,
        ok ? "data=OK" : "data=MISMATCH",
        spin_only + io_only, mx, sync_ms / async_ms,
        (sync_ms > mx) ? (sync_ms - async_ms) / (sync_ms - mx) * 100.0 : 0.0);
    report_tiers();
    report_streams();
    if (!ok) {
      std::fprintf(stderr, "\nDATA MISMATCH -- timings above are void\n");
      return 1;
    }
    clio::run::CLIO_RUNTIME_FINALIZE();
    return 0;
  }

  u32 pass = 1;
  auto run = [&](u32 spin, int do_write, int async) {
    double best = 1e30;
    for (u32 r = 0; r < a.repeat; ++r) {
      vec.ResetStats();
      cudaDeviceSynchronize();
      const double t0 = NowMs();
      const u32 this_pass = pass++;
      runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        SpinWriteFlushKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dev, a.iters, pages_per_region, spin, clock_khz, do_write,
            this_pass, async, vw, sv);
      });
      if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "kernel failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
      }
      const double ms = NowMs() - t0;
      if (ms < best) best = ms;
    }
    return best;
  };

  // TOTAL MEASURED TIME, EXCLUDING INITIALIZATION. The only end-to-end number
  // available until now was the pipeline's process wall clock, and 44-55% of
  // that is startup, tier creation, the vector's page-cache allocation, the
  // warm pass, and the 2048 verification reads -- none of which is the thing
  // under test. Reporting it as "the time" would have compressed the
  // full-residency win from 18.4% to about 9% and buried the cache-axis trend
  // under a ~6 s constant. This clock spans every timed phase (all `repeat`
  // executions, not just the best of each) and nothing else.
  const double measured_t0 = NowMs();
  const double spin_only = run(a.spin_us, 0, 0);   // compute, no I/O
  const double io_only = run(0, 1, 0);             // I/O, no compute
  const auto io_stats = vec.ReadStats(0);
  const double sync_ms = run(a.spin_us, 1, 0);
  const auto sync_stats = vec.ReadStats(0);
  const double async_ms = run(a.spin_us, 1, 1);
  const auto async_stats = vec.ReadStats(0);
  const double measured_total = NowMs() - measured_t0;
  const u32 last_pass = pass - 1;

  // Verify the async pass reached the CTE. Sampled: first and last page of
  // every region, which is enough to catch a lost or mis-ordered flush without
  // reading hundreds of megabytes back through the client.
  bool ok = true;
  {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    std::vector<u32> buf(static_cast<size_t>(page_elems));
    for (u32 b = 0; b < a.blocks && ok; ++b) {
      for (u64 it = 0; it < a.iters && ok; ++it) {
        const u64 first = (b * a.iters + it) * pages_per_region;
        const u64 probe[2] = {first, first + pages_per_region - 1};
        for (int k = 0; k < 2 && ok; ++k) {
          char name[32];
          gv::PageBlobName(probe[k], name);
          auto f = core.AsyncGetBlob(vec.TagId(), std::string(name), 0,
                                     page_bytes, 0,
                                     reinterpret_cast<char *>(buf.data()));
          // BOUNDED. An unbounded Wait() here is how this benchmark hangs:
          // IpcCpu2Self::RecvOut polls task->IsComplete() forever when
          // max_sec is 0, so a GetBlobTask whose completion is never set
          // parks the process indefinitely -- observed with every worker
          // idle in SuspendMe, no worker executing, and the GPU at 0%.
          // A missed lane wakeup cannot explain it (the worker's epoll is
          // capped at max_sleep and re-polls unconditionally), so the task
          // is either lost at submission or completed without IsComplete()
          // being set. Either way, report WHICH blob and stop rather than
          // stalling a whole sweep on one unresolvable read.
          if (!f.Wait(kVerifyTimeoutSec)) {
            std::fprintf(stderr,
                         "\n  *** HANG: GetBlob(%s) did not complete within "
                         "%.0fs (block %u, iter %llu, page %llu). The task "
                         "never signalled completion.\n",
                         name, (double) kVerifyTimeoutSec, b,
                         (unsigned long long) it,
                         (unsigned long long) probe[k]);
            ok = false;
            break;
          }
          if (f->GetReturnCode() != 0) {
            std::fprintf(stderr, "  read of %s failed rc=%d\n", name,
                         f->GetReturnCode());
            ok = false;
            break;
          }
          const u64 base = probe[k] * page_elems;
          for (u64 i = 0; i < page_elems; i += 97) {
            if (buf[static_cast<size_t>(i)] != Val(base + i, last_pass)) {
              std::fprintf(stderr,
                           "  MISMATCH %s elem %llu: got %u want %u\n", name,
                           (unsigned long long) i, buf[static_cast<size_t>(i)],
                           Val(base + i, last_pass));
              ok = false;
              break;
            }
          }
        }
      }
    }
  }

  const double per_iter = static_cast<double>(a.iters);
  const double mx = (spin_only > io_only) ? spin_only : io_only;
  // Bytes moved per pass are the WORKING SET (every iteration writes a full
  // region), not the cache size. Those were the same number until the cache
  // became independently sizeable; using resident_mb here would silently
  // divide the reported bandwidth by the cache ratio.
  const double flush_mbps =
      (io_only > 0.0) ? (working_mb / (io_only / 1000.0)) : 0.0;
  std::printf(
      "\n  spin only        %8.2f ms   (%.0f us/iter)\n"
      "  flush only       %8.2f ms   (%.0f us/iter, puts=%llu, %.0f MB/s)\n"
      "  ---------------------------------------------\n"
      "  sync  measured   %8.2f ms   (puts=%llu)\n"
      "  async measured   %8.2f ms   (puts=%llu)  %s\n"
      "  ---------------------------------------------\n"
      "  sum  spin+flush  %8.2f ms   <- what NO overlap costs\n"
      "  max  spin,flush  %8.2f ms   <- what FULL overlap costs\n"
      "  speedup async    %8.2fx\n"
      "  flush hidden     %8.1f %%\n"
      // WHY cache size may or may not matter. A page cache can only pay for
      // itself through REUSE: a hit avoids a fault. This kernel writes each
      // page once and flushes it, so `faults` is the number of reads the cache
      // could have served. If faults is 0 at every cache size, then the bytes
      // moved are identical (the same `puts`) no matter how big the cache is,
      // and a flat curve is the CORRECT answer rather than a suspicious one --
      // the cache has nothing to hit. Evicts vs puts says whether writebacks
      // came from reclaiming slots or from the explicit flush.
      "  measured total   %8.2f ms   (all timed phases; EXCLUDES init, warm, "
      "and verification)\n"
      "  faults           %8llu     (cache hits are impossible when this is 0)\n"
      "  evicts / puts    %8llu / %llu\n",
      spin_only, spin_only * 1000.0 / per_iter,
      io_only, io_only * 1000.0 / per_iter,
      (unsigned long long) io_stats.puts, flush_mbps,
      sync_ms, (unsigned long long) sync_stats.puts,
      async_ms, (unsigned long long) async_stats.puts,
      ok ? "data=OK" : "data=MISMATCH",
      spin_only + io_only, mx, sync_ms / async_ms,
      (sync_ms > mx) ? (sync_ms - async_ms) / (sync_ms - mx) * 100.0 : 0.0,
      measured_total, (unsigned long long) io_stats.faults,
      (unsigned long long) io_stats.evicts, (unsigned long long) io_stats.puts);
  report_tiers();
  report_streams();

  if (!ok) {
    std::fprintf(stderr, "\nDATA MISMATCH -- timings above are void\n");
    return 1;
  }
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}

#endif  // !CTP_IS_DEVICE_PASS
