/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Model-weights benchmark for the device-paged vector, comparing RAW and
 * COMPRESSED storage as the working set outgrows GPU memory.
 *
 * The variable that matters is the CTE's GPU (kHbm) tier. Pages live there in
 * their STORED form, so compression changes how much of the model fits in
 * device memory; whatever does not fit spills to the host RAM tier and is
 * fetched over PCIe on every touch. The GPU page cache itself always holds
 * plain bytes -- compression buys tier residency, not cache capacity.
 *
 * That gives three regimes, selected by --hbm-mb:
 *
 *   1. Both fit.        Compression is pure overhead: the same pages are
 *                       device-resident either way and the compressed run
 *                       additionally decompresses each fetch. Expect raw to
 *                       win on time and compressed to win on utilization.
 *   2. Raw spills,      The compressed image is device-resident while the raw
 *      compressed fits. one goes to host RAM. Expect compressed to win, by
 *                       the margin between a device read and a PCIe read.
 *   3. Neither fits.    Compressed still moves fewer bytes over PCIe per page,
 *                       so it should stay ahead, by less.
 *
 * The workload is an inference-shaped weighted sum over the whole weight
 * vector. Each block walks its own slice and, with --prefetch, raises the
 * score of the page it is about to need so the organizer keeps it in the top
 * tier -- a metadata hint, not a data copy.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

/**
 * Frame space per lane. Macro frames are hand-packed (a few u64s); coroutine
 * frames are compiler-laid-out and carry the whole live state plus resume /
 * destroy pointers, so they get an order of magnitude more headroom.
 */
static constexpr clio::run::u32 kYieldLaneBytes = 4096;

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include "bench_memcpy_probe.h"

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;

namespace {

// The DATA's flat-block granularity, deliberately FIXED and independent of
// the vector's page size. PageIsFlat/Weight decide compressibility on this
// granule, so tying it to the runtime page size would change the generated
// bytes whenever the page size changed -- and a page-size sweep would then be
// comparing different datasets, not different paging.
constexpr clio::run::u64 kFlatGranuleBytes = 64 * 1024;
constexpr clio::run::u64 kFlatGranuleElems =
    kFlatGranuleBytes / sizeof(clio::run::u32);
const clio::run::PoolId kCompressorPool(512, 0);
const clio::run::PoolId kCorePool(513, 0);
constexpr int kLz4WireId = 4;      // registry: {"lz4", 4, ...}   CPU codec
// GPU codec. This is the one that matters here: decompressing on the device
// is impossible while a kernel spins on its fault, because the codec is itself
// a kernel and cannot launch behind a resident one. A yielding fault path is
// what makes it reachable at all.
constexpr int kNvcompLz4WireId = 11;  // registry: {"nvcomp-lz4", 11, ...}

/**
 * A weight that compresses like real model data, not like a test pattern.
 *
 * The obvious "small repeating value set" compresses ~34x, which makes every
 * tier look big enough and collapses the three regimes into one. Real
 * quantized weights are close to random at byte level (measured elsewhere:
 * lz4 saves ~4% on Q4_K). This mixes a pseudo-random low nibble with a
 * slowly-varying high nibble, landing around 2x -- compressible enough that
 * residency actually changes, without pretending weights are trivially
 * compressible.
 */
CTP_INLINE_CROSS_FUN clio::run::u32 Weight(clio::run::u64 i);

/**
 * Is this page one of the highly compressible ones?
 *
 * Hashed rather than striped so the compressible pages are scattered through
 * the model: a contiguous compressible half would let the tier hold a solid
 * run of it and flatter residency for reasons that have nothing to do with the
 * ratio.
 */
CTP_INLINE_CROSS_FUN bool PageIsFlat(clio::run::u64 page, clio::run::u32 pct) {
  clio::run::u32 h = static_cast<clio::run::u32>(page * 2654435761u);
  h ^= h >> 15;
  return (h % 100u) < pct;
}

CTP_INLINE_CROSS_FUN clio::run::u32 Weight(clio::run::u64 i,
                                           clio::run::u32 flat_pct) {
  // A flat page is a single repeated value: what a byte codec collapses.
  if (PageIsFlat(i / kFlatGranuleElems, flat_pct)) {
    return 0x01010101u;
  }
  return Weight(i);
}

CTP_INLINE_CROSS_FUN clio::run::u32 Weight(clio::run::u64 i) {
  // Runs of kRun identical values: structured weight data repeats locally,
  // and it is that repetition -- not the value distribution -- that a byte
  // codec exploits. kRun sets the compression ratio; 8 lands near 2x.
  constexpr clio::run::u64 kRun = 8;
  clio::run::u32 r = static_cast<clio::run::u32>((i / kRun) * 2654435761u);
  r ^= r >> 13;
  return (r & 0x3F3F3F3Fu) |
         (static_cast<clio::run::u32>((i / 4096) % 13) * 0x40404040u);
}

CTP_INLINE_CROSS_FUN clio::run::u32 Activation(clio::run::u64 i) {
  return static_cast<clio::run::u32>((i % 7) + 1);
}

/**
 * BASELINE KERNEL -- the textbook out-of-core GPU model, for comparison.
 *
 * The GPU vector's whole claim is that a kernel can fault on data it does not
 * have and keep running. The honest control for that claim is how everyone
 * does it WITHOUT such a mechanism:
 *
 *   1. storage I/O is SYNCHRONOUS   -- the host blocks on a CTE GetBlob
 *   2. HBM<->DRAM copy is SYNCHRONOUS -- a blocking cudaMemcpy per tile
 *   3. THE KERNEL TERMINATES for I/O -- one launch per tile, and the grid is
 *      torn down and rebuilt between every transfer
 *
 * Those three are one design, not three: once a kernel must exit to get data,
 * the transfer is necessarily host-driven and serialised behind it. Nothing
 * overlaps -- read, copy, compute and teardown all run end to end.
 *
 * It computes EXACTLY the contributions the paged kernel does (same integer
 * arithmetic, same per-page accounting), so the existing got-vs-want checksum
 * validates the baseline rather than a second implementation being trusted on
 * inspection. Integer accumulation means order does not change the total.
 */
__global__ void WeightsBaselineKernel(const clio::run::u32 *tile,
                                      clio::run::u64 gbase, clio::run::u64 n,
                                      unsigned long long *sum,
                                      unsigned long long *page_sum,
                                      unsigned *page_visits,
                                      clio::run::u64 page_elems) {
  unsigned long long r = 0;
  for (clio::run::u64 i = threadIdx.x; i < n; i += blockDim.x) {
    r += static_cast<unsigned long long>(tile[i]) * Activation(gbase + i);
  }
  atomicAdd(sum, r);
  atomicAdd(&page_sum[gbase / page_elems], r);
  if (threadIdx.x == 0) {
    atomicAdd(&page_visits[gbase / page_elems], 1u);
  }
}

clio::run::u64 NowMs() {
  return static_cast<clio::run::u64>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

}  // namespace

/**
 * Seeding is the phase that actually wedges, so it yields too.
 *
 * Every page it writes is dirty, so making room for the next one flushes the
 * previous one -- and on the kHbm tier that writeback is a device copy that
 * cannot schedule while this kernel sits on the SM waiting for it. Suspending
 * lets the copy run.
 */
#if defined(CLIO_YIELD_CORO) && defined(__clang__)
/**
 * The seed pass as a per-lane C++20 coroutine (clang-CUDA builds). Ordinary
 * locals -- `off` and the guard cross suspends and the COMPILER puts them in
 * the frame; compare the CLIO_YLOCAL_INIT bookkeeping in the macro version
 * below. Suspends stay block-collective: every yield site votes.
 */
__device__ gy::YCoroMain SeedLaneCoro(gv::DeviceVector<clio::run::u32> v,
                                      clio::run::u64 per,
                                      clio::run::u64 page_elems,
                                      clio::run::u32 flat_pct,
                                      clio::run::u32 block) {
  const clio::run::u64 base = static_cast<clio::run::u64>(block) * per;
  for (clio::run::u64 off = 0; off < per; off += page_elems) {
    auto h = co_await v.HoldPage(
        base + off, (off + page_elems <= per) ? page_elems : (per - off),
        /*write=*/true);
    const clio::run::u64 n =
        (off + page_elems <= per) ? page_elems : (per - off);
    for (clio::run::u64 i = threadIdx.x; i < n; i += blockDim.x) {
      h[base + off + i] = Weight(base + off + i, flat_pct);
    }
  }
  // Final writeback -- explicit, because drops and eviction never write a
  // page back on their own now. Collective, internally BATCHED.
  v.FlushAsync(base, per);
  co_await v.AwaitFlush();
}

__global__ void SeedKernelYield(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<clio::run::u32> v,
                                clio::run::u64 per,
                                clio::run::u64 page_elems,
                                clio::run::u32 flat_pct,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SeedLaneCoro(v, per, page_elems, flat_pct, yv.Block()));
}
#endif  // CLIO_YIELD_CORO

/**
 * The measured pass: a weighted sum over this block's slice of the model,
 * SUSPENDING on a miss. Every lane votes on whether its page is resident,
 * the whole block suspends if ANY lane is waiting, and the kernel
 * exits so the fetch can land. On resume the page is resident for everyone, so
 * the read is an ordinary parallel loop with no fault path in it at all.
 */
#if defined(CLIO_YIELD_CORO) && defined(__clang__)
/** The measured pass as a per-lane coroutine; same shape as SeedLaneCoro. */
__device__ gy::YCoroMain WeightsLaneCoro(gv::DeviceVector<clio::run::u32> v,
                                         clio::run::u64 per,
                                         clio::run::u64 page_elems,
                                         unsigned long long *sum,
                                         unsigned long long *page_sum,
                                         unsigned *page_visits,
                                         clio::run::u32 block) {
  const clio::run::u64 base = static_cast<clio::run::u64>(block) * per;
  unsigned long long acc = 0;
  for (clio::run::u64 off = 0; off < per; off += page_elems) {
    auto h = co_await v.HoldPage(
        base + off, (off + page_elems <= per) ? page_elems : (per - off));
    const clio::run::u64 n =
        (off + page_elems <= per) ? page_elems : (per - off);
    unsigned long long r = 0;                     // register, not the frame
    for (clio::run::u64 i = threadIdx.x; i < n; i += blockDim.x) {
      r += static_cast<unsigned long long>(h[base + off + i]) *
           Activation(base + off + i);
    }
    acc += r;
    atomicAdd(&page_sum[(base + off) / page_elems], r);
    if (threadIdx.x == 0) {
      atomicAdd(&page_visits[(base + off) / page_elems], 1u);
    }
  }
  atomicAdd(sum, acc);
}

__global__ void WeightsKernelYield(clio::run::IpcManagerGpuInfo info,
                                   gv::DeviceVector<clio::run::u32> v,
                                   clio::run::u64 per,
                                   clio::run::u64 page_elems,
                                   unsigned long long *sum,
                                   unsigned long long *page_sum,
                                   unsigned *page_visits,
                                   gy::YieldableView<> yv,
                                   gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(WeightsLaneCoro(v, per, page_elems, sum, page_sum,
                                 page_visits, yv.Block()));
}
#endif  // CLIO_YIELD_CORO

#if !CTP_IS_DEVICE_PASS

int main(int argc, char **argv) {
  unsigned blocks = 16;
  clio::run::u64 hbm_mb = 64;
  // Storage tier: without it no workload ever touches a disk.
  unsigned long long nvme_mb = 0;
  std::string nvme_path = "/tmp/gv_storage_tier.dat";
  bool hbm_only = false;   // omit the host spill tier entirely
  clio::run::u64 page_kb = 64;  // vector page size; the DATA granule is fixed
  // Threads per block. 32 = ONE warp, which means one warp is all there is to
  // decode with; the in-kernel nvcomp decoder is warp-cooperative, so a wider
  // block is what lets several pages decode at once.
  clio::run::u32 nthreads = 32;
  clio::run::u32 rt_threads = 8;   // runtime worker threads
  clio::run::u64 pages_per_block = 16;
  clio::run::u32 slots = 4;
  bool compressed = false;
  bool gpu_codec = true;   // nvcomp by default; --cpu-codec for lz4
  int prefetch = 1;
  int repeat = 3;
  // The out-of-core model WITHOUT in-kernel faulting: sync storage I/O,
  // sync HBM<->DRAM copy, and the kernel torn down for every transfer.
  bool baseline = false;
  clio::run::u32 flat_pct = 0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? std::atoll(argv[++i]) : 0; };
    if (a == "--blocks") blocks = static_cast<unsigned>(next());
    else if (a == "--hbm-mb") hbm_mb = static_cast<clio::run::u64>(next());
    else if (a == "--hbm-only") hbm_only = true;
    else if (a == "--nvme-mb") nvme_mb = next();
    // next() parses a number; the path needs the raw argv token.
    else if (a == "--nvme-path" && i + 1 < argc) nvme_path = argv[++i];
    else if (a == "--threads") nthreads = static_cast<clio::run::u32>(next());
    else if (a == "--rt-threads") rt_threads = static_cast<clio::run::u32>(next());
    else if (a == "--pages") pages_per_block = static_cast<clio::run::u64>(next());
    else if (a == "--slots") slots = static_cast<clio::run::u32>(next());
    else if (a == "--compressed") compressed = true;
    else if (a == "--cpu-codec") gpu_codec = false;
    else if (a == "--no-prefetch") prefetch = 0;
    // --baseline REPLACES THE TIMED LOOP ONLY; the seed still runs through
    // the vector's coroutine kernel.
    else if (a == "--baseline") { baseline = true; }
    else if (a == "--flat-pct") flat_pct = static_cast<clio::run::u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--repeat") repeat = static_cast<int>(next());
    else if (a == "--help") {
      std::fprintf(stderr,
                   "usage: %s [--blocks N] [--hbm-mb M] [--hbm-only] [--threads T] [--pages P] "
                   "[--slots S] [--compressed] [--no-prefetch] [--repeat R]\n"
                 "       [--baseline]   NO in-kernel faulting: one kernel launch\n"
                 "                      per tile, blocking GetBlob + blocking memcpy\n",
                   argv[0]);
      return 0;
    }
  }

  // The GPU tier is the whole point: pages live there in their STORED form, so
  // its capacity decides how much of the model is device-resident. Everything
  // beyond it spills to the host RAM tier below.
  const clio::run::u64 page_bytes = page_kb * 1024;
  // REFERENCE CEILING for the paging path: a bare H2D cudaMemcpy at exactly
  // this page size, on an idle device before the runtime starts. The gap
  // between this and the achieved rate is the vector's overhead.
  const MemcpyProbe mcp = ProbeMemcpyBandwidth(static_cast<size_t>(page_bytes));
  const clio::run::u64 page_elems = page_bytes / sizeof(clio::run::u32);

  {
    std::ofstream cfg("gv_weights_bench.yaml");
    cfg << "networking:\n  port: 9435\n\n"
        // first_busy_wait: workers that fall back to sleeping add their sleep to
        // every device fault, since a fault is a synchronous round trip. With
        // the 1000us default a fault costs ~5ms regardless of where the page
        // lives, which buries the tier/codec difference this benchmark exists
        // to measure. Keep them spinning for the duration of a run.
        << "runtime:\n  num_threads: " << rt_threads << "\n  queue_depth: 8192\n"
        << "  first_busy_wait: 2000000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"2GB\"\n\n"
        << "  - mod_name: clio_cte_compressor\n"
        << "    pool_name: cte_compressor\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n    next_pool_id: \"513.0\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"513.0\"\n"
        << "    storage:\n"
        << "      - path: \"hbm::gv_bench_hbm\"\n"
        << "        bdev_type: \"hbm\"\n"
        << "        capacity_limit: \"" << hbm_mb << "MB\"\n"
        << "        score: 1.0\n";
    // The host RAM tier is the SPILL tier, and it is the only place data can
    // land that is not the GPU. --hbm-only omits it, so "every byte is in
    // HBM" stops being something the run happens to achieve and becomes
    // something the configuration cannot violate: with no tier below it, a
    // page that does not fit in HBM has nowhere to go and the put fails
    // loudly instead of quietly relocating to the host.
    if (!hbm_only) {
      // SIZED FROM THE WORKLOAD, not a constant. This was hardcoded at 2GB,
      // which silently capped how large a model could be benchmarked: a
      // dataset bigger than the kHBM tier plus 2GB had nowhere to put the
      // remainder and the writebacks failed. An out-of-core run at 2x VRAM
      // (16GB of weights against a 4GB tier) needs 12GB of spill, so the host
      // tier is derived from the actual logical size with a 1GB margin.
      const clio::run::u64 logical_mb_cfg =
          (pages_per_block * blocks * page_bytes) / (1024ull * 1024ull);
      cfg << "      - path: \"ram::gv_bench_ram\"\n"
          << "        bdev_type: \"ram\"\n"
          << "        capacity_limit: \"" << (logical_mb_cfg + 1024)
          << "MB\"\n"
          << "        score: 0.2\n";
    }
      // OPTIONAL STORAGE TIER. Without it the whole dataset lives in host
      // DRAM and NOTHING EVER TOUCHES STORAGE -- what such a run measures is
      // DRAM over PCIe, not I/O. On this machine that spill is nearly free,
      // which is exactly why a cache-size sweep over a DRAM-only hierarchy
      // comes back flat: there is no penalty for the cache to save.
      //
      // score BELOW the host tier. MaxBwDpe splits on target_score <=
      // blob_score and sorts the preferred group DESCENDING, and the vector
      // puts pages at blob score 1.0, so HIGHER score = preferred. This is the
      // REVERSE of the GNN trainer's hierarchy, whose put path uses blob score
      // 0.5 -- copying its numbers here would silently make storage the
      // FIRST-choice tier.
      if (nvme_mb > 0) {
        cfg << "      - path: \"" << nvme_path << "\"\n"
            << "        bdev_type: \"file\"\n"
            << "        capacity_limit: \"" << nvme_mb << "MB\"\n"
            << "        score: 0.0\n";
      }

    cfg << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_weights_bench.yaml", 1);

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "bench: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "bench: cte client init failed\n");
    return 1;
  }
  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  const clio::run::u64 per = pages_per_block * page_elems;
  const clio::run::u64 n = per * blocks;
  const clio::run::u64 logical = n * sizeof(clio::run::u32);
  const std::string tag =
      std::string("gvw_") + (compressed ? "lz4_" : "raw_") +
      std::to_string(blocks) + "_" + std::to_string(hbm_mb) + "_" +
      std::to_string(pages_per_block);

  gv::Vector<clio::run::u32> vec(tag, {0}, page_bytes, blocks, slots, n,
                                 kCompressorPool,
                                 compressed ? (gpu_codec ? kNvcompLz4WireId
                                                        : kLz4WireId)
                                            : 0);
  // Writeback failures are the one error that used to be invisible: a put that
  // the CTE rejected still marked its page clean, so the bytes were dropped
  // and only the checksum noticed. Count them and report.
  vec.EnableStats();

  clio::run::u32 seed_checks = 0;
  {
    gy::Yieldable<> sdrv(blocks, nthreads);
    gy::YieldStack sstack(blocks, nthreads, kYieldLaneBytes);
    const clio::run::u32 seed_rounds = sdrv.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          SeedKernelYield<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
              gpu_info, vec.GetDevice(0), per, page_elems, flat_pct, view,
              sstack.View());
        },
        // Abort the moment a writeback FAILS, rather than waiting out a
        // round cap. A failed put means the tier is full and the page stays
        // dirty, so no slot can ever be cleaned and the fault path would spin
        // forever -- the caller knows that, the driver cannot. Checked every
        // 64 rounds because it costs a device read.
        [&]() -> bool {
          if (std::getenv("CLIO_YCORO_TRACE") != nullptr &&
              (seed_checks & 255u) == 0u && sdrv.NumPending() > 0) {
            const clio::run::u32 b = sdrv.PendingBlock(0);
            std::fprintf(stderr,
                         "[ytrace] round=%u pending=%u b=%u status=%u tag=%llx\n",
                         seed_checks, sdrv.NumPending(), b,
                         (unsigned) sdrv.BlockState(b).status_,
                         (unsigned long long) sdrv.BlockState(b).wait_tag_);
            // Dump block 0's page table: which slots are wedged, and how.
            gv::DeviceVector<clio::run::u32> dv = vec.GetDevice(0);
            gv::VecHeader hh;
            ctp::GpuApi::Memcpy(&hh, dv.h_, sizeof(hh));
            std::vector<gv::Page> pg(hh.pages_per_block_);
            ctp::GpuApi::Memcpy(pg.data(), hh.pages_ + (size_t)b * hh.pages_per_block_,
                       pg.size() * sizeof(gv::Page));
            for (clio::run::u32 i = 0; i < hh.pages_per_block_; ++i) {
              if (pg[i].page_num == gv::kNoPage && !pg[i].fetching &&
                  !pg[i].flushing) continue;
              std::fprintf(stderr,
                           "  [pt] slot=%u page=%lld dirty=%u flushing=%u "
                           "fetching=%u evicting=%u\n",
                           i, (long long) pg[i].page_num, pg[i].dirty,
                           pg[i].flushing, pg[i].fetching, pg[i].evicting);
            }
          }
          if ((++seed_checks & 63u) != 0u) return true;
          return vec.ReadStats(0).put_errors == 0;
        },
        // Backstop for a stall with no put error. Must SCALE with the image:
        // a healthy seed costs a few rounds per page (measured 1734 rounds
        // for 256 pages, ~6.8/page), so a fixed cap that suits a 16MB run
        // false-trips a 2GB one. 200/page is ~30x headroom.
        /*max_rounds=*/static_cast<clio::run::u32>(
            (n / page_elems) * 200ull + 50000ull));
    const bool sdrv_aborted = sdrv.Aborted();
    std::fprintf(stderr, "[seed] rounds=%u%s\n", seed_rounds,
                 sdrv_aborted ? " ABORTED (writeback failed)" : "");
    {
      const auto seed_stats = vec.ReadStats(0);
      const clio::run::u32 seed_cap =
          static_cast<clio::run::u32>((n / page_elems) * 200ull + 50000ull);
      if (sdrv_aborted || seed_rounds >= seed_cap ||
          seed_stats.put_errors != 0) {
        std::fprintf(stderr,
                     "bench: SEED DID NOT CONVERGE (rounds=%u, "
                     "put_errors=%llu). The tier cannot hold this image -- "
                     "raise --hbm-mb, lower --pages, or compress harder. "
                     "Refusing to report a measurement.\n",
                     seed_rounds,
                     (unsigned long long) seed_stats.put_errors);
        return 1;
      }
    }
  }
  ctp::GpuApi::Synchronize();

  // Stored footprint: what the model actually occupies in the CTE, which is
  // what decides whether it fits the GPU tier.
  clio::run::u64 stored = 0;
  {
    clio::cte::core::Client core(kCorePool);
    for (clio::run::u64 p = 0; p < n / page_elems; ++p) {
      char name[32];
      gv::PageBlobName(p, name);
      auto sz = core.AsyncGetBlobSize(vec.TagId(), name);
      sz.Wait();
      if (sz->GetReturnCode() == 0) stored += sz->size_;
    }
  }

  // Publish the stored-size table (encoded tags only). It is the
  // run-fetch's existence proof; the compressor's puts land asynchronously,
  // so retry until every page is covered. The zero-copy device-tier map
  // that used to be built here is gone: kernels no longer alias shared CTE
  // tier memory, and the fits-in-VRAM regime is expressed as a PRIVATE
  // cache that covers the model (--slots >= --pages), seeded by the seed
  // pass and never faulting afterwards.
  if (compressed) {
    const clio::run::u64 npages_all = n / page_elems;
    clio::run::u64 found = 0;
    for (int attempt = 0; attempt < 200; ++attempt) {
      found = vec.PublishStoredSizes();
      if (found == npages_all) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::fprintf(stderr, "[ssz] stored-size table: %llu/%llu pages\n",
                 (unsigned long long) found,
                 (unsigned long long) npages_all);
  }

  unsigned long long *d_sum = nullptr;
  d_sum = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_sum)>>(sizeof(unsigned long long));
  const clio::run::u64 total_pages = n / page_elems;
  unsigned long long *d_page_sum = nullptr;
  unsigned *d_page_visits = nullptr;
  d_page_sum = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_page_sum)>>(total_pages * sizeof(unsigned long long));
  d_page_visits = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_page_visits)>>(total_pages * sizeof(unsigned));
  unsigned long long want = 0;
  for (clio::run::u64 i = 0; i < n; ++i) {
    want += static_cast<unsigned long long>(Weight(i, flat_pct)) * Activation(i);
  }

  // ---- BASELINE DRIVER ------------------------------------------------
  // One tile at a time, nothing overlapped: block on the CTE read, block on
  // the H2D copy, launch, tear the grid down, repeat. This is the control the
  // vector's in-kernel faulting is measured against.
  ctp::ipc::FullPtr<char> bl_host;
  clio::run::u32 *bl_dev = nullptr;
  clio::cte::core::Client *bl_core = nullptr;
  if (baseline) {
    bl_host = CLIO_IPC->AllocateBuffer(static_cast<size_t>(page_bytes));
    if (bl_host.IsNull()) {
      std::fprintf(stderr, "bench: baseline staging alloc failed\n");
      return 1;
    }
    // GpuApi::Malloc fails fatally, which is the same abort with less code.
    bl_dev = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(bl_dev)>>(
        static_cast<size_t>(page_bytes));
    bl_core = new clio::cte::core::Client(kCorePool);
    std::fprintf(stderr, "[baseline] staging ready, %llu tiles of %lluB\n",
                 (unsigned long long)total_pages,
                 (unsigned long long)page_bytes);
  }
  auto run_baseline = [&]() {
    for (clio::run::u64 p = 0; p < total_pages; ++p) {
      if ((p & 7) == 0) {
        std::fprintf(stderr, "[baseline] tile %llu/%llu\n",
                     (unsigned long long)p, (unsigned long long)total_pages);
      }
      char nm[32];
      gv::PageBlobName(p, nm);
      // (1) SYNCHRONOUS storage read -- the host waits for the tier.
      auto gf = bl_core->AsyncGetBlob(vec.TagId(), nm, 0, page_bytes, 0,
                                      bl_host.shm_.template Cast<void>(),
                                      clio::run::PoolQuery::Local());
      gf.Wait();
      if (gf->GetReturnCode() != 0) continue;   // never stored
      // (2) SYNCHRONOUS HBM<->DRAM copy.
      ctp::GpuApi::Memcpy(reinterpret_cast<char *>(bl_dev), bl_host.ptr_,
                          static_cast<size_t>(page_bytes));
      // (3) The kernel runs ONLY while its tile is resident, then exits.
      WeightsBaselineKernel<<<1, nthreads>>>(
          bl_dev, p * page_elems, page_elems, d_sum, d_page_sum, d_page_visits,
          page_elems);
      ctp::GpuApi::Synchronize();
    }
    return true;
  };

  double best_gbps = 0.0;
  clio::run::u32 rounds = 0;
  clio::run::u64 best_ms = 0;
  bool ok = true;
  for (int r = 0; r < repeat; ++r) {
    ctp::GpuApi::Memset(d_sum, 0, sizeof(unsigned long long));
    ctp::GpuApi::Memset(d_page_sum, 0, total_pages * sizeof(unsigned long long));
    ctp::GpuApi::Memset(d_page_visits, 0, total_pages * sizeof(unsigned));
    const clio::run::u64 t0 = NowMs();
    if (baseline) {
      if (!run_baseline()) {
        std::fprintf(stderr, "bench: baseline tile loop failed\n");
        return 1;
      }
    } else {
      gy::Yieldable<> drv(blocks, nthreads);
      gy::YieldStack ystack(blocks, nthreads, kYieldLaneBytes);
      rounds = drv.RunToCompletion(
          [&](dim3 g, dim3 b, gy::YieldableView<> view) {
            WeightsKernelYield<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
                gpu_info, vec.GetDevice(0), per, page_elems, d_sum, d_page_sum,
                d_page_visits, view, ystack.View());
          },
          []{},
          /*max_rounds=*/200000,
          [](clio::run::u32, clio::run::u64 tag) {
            // 4 bytes over PCIe (~5us) instead of relaunching the grid (~50us+)
            // just to have the block discover its page has not landed.
            unsigned flag = 0;
            ctp::GpuApi::Memcpy(&flag,
                                reinterpret_cast<const unsigned *>(tag),
                                sizeof(flag));
            return flag != 0;
          });
    }
    ctp::GpuApi::Synchronize();
    const clio::run::u64 ms = NowMs() - t0;
    unsigned long long got = 0;
    ctp::GpuApi::Memcpy(&got, d_sum, sizeof(got));
    if (got != want) {
      ok = false;
      std::fprintf(stderr, "[cmp] got=%llu want=%llu ratio=%.4f\n",
                   (unsigned long long) got, (unsigned long long) want,
                   want ? (double) got / (double) want : 0.0);
      std::vector<unsigned long long> ps(total_pages);
      std::vector<unsigned> pv(total_pages);
      ctp::GpuApi::Memcpy(ps.data(), d_page_sum,
                 total_pages * sizeof(unsigned long long));
      ctp::GpuApi::Memcpy(pv.data(), d_page_visits, total_pages * sizeof(unsigned));
      int shown = 0;
      for (clio::run::u64 p = 0; p < total_pages && shown < 6; ++p) {
        unsigned long long w = 0;
        for (clio::run::u64 e = 0; e < page_elems; ++e) {
          const clio::run::u64 i = p * page_elems + e;
          w += (unsigned long long) Weight(i, flat_pct) * Activation(i);
        }
        if (ps[p] != w) {
          std::fprintf(stderr,
                       "  page %llu: visits=%u got=%llu want=%llu %s\n",
                       (unsigned long long) p, pv[p], ps[p], w,
                       (pv[p] != 1) ? "<- visited != once" : "<- wrong data");
          ++shown;
        }
      }
    }
    const double gbps =
        (ms == 0) ? 0.0
                  : (static_cast<double>(logical) / (1024.0 * 1024.0 * 1024.0)) /
                        (static_cast<double>(ms) / 1000.0);
    if (gbps > best_gbps) {
      best_gbps = gbps;
      best_ms = ms;
    }
  }
  ctp::GpuApi::Free(d_sum);

  // A run that could not write its data back is not a valid measurement,
  // however good its throughput looks. Reported SEPARATELY from the checksum:
  // they are different failures, and a run can lose data while still summing
  // to the right number over the pages that survived.
  //
  // Only WRITEBACK failures count here. A failed GET is normal and expected:
  // the first touch of a never-written page misses in the CTE by design, and
  // the claimed slot is what turns that miss into a write-allocate.
  const auto stats = vec.ReadStats(0);
  if (stats.put_errors != 0) {
    std::fprintf(stderr,
                 "GVW ERROR: %llu writeback(s) FAILED -- the tier could not "
                 "hold the data, so those pages were never stored. The "
                 "numbers below are not a valid measurement.\n",
                 (unsigned long long) stats.put_errors);
  }

  // ---- TIER PLACEMENT CHECK -------------------------------------------
  // Data must land in the FASTEST tier that has capacity, spilling only once
  // that tier is full. MEASURED, not assumed: MaxBwDpe splits tiers on
  // target_score <= blob_score and ranks within the group, so a tier scored on
  // the wrong side of the blob's own score silently drops out of the preferred
  // set. That has produced "kHBM 0MiB" with a healthy, correctly sized HBM
  // tier sitting empty -- no error, just a run that never touched the GPU tier
  // and lost the device-to-device fault path with it.
  //
  // Tier bdevs are numbered from major 512 with minor 1, in CONFIG ORDER,
  // independent of cte_core's own major: (512,1) is the first storage
  // entry (kHBM) and (513,1) the second (host). Deriving them from
  // cte_core's major instead gave remaining > capacity -- an impossible
  // reading that would have been reported as a placement violation.
  {
    clio::run::bdev::Client t_fast(clio::run::PoolId(512, 1));
    clio::run::bdev::Client t_host(clio::run::PoolId(513, 1));
    auto fa = t_fast.AsyncGetStats(); fa.Wait();
    auto ha = t_host.AsyncGetStats(); ha.Wait();
    const clio::run::u64 fast_cap = (clio::run::u64)hbm_mb * 1024ull * 1024ull;
    const clio::run::u64 host_cap = (logical / (1024ull * 1024ull) + 1024ull) * 1024ull * 1024ull;
    const clio::run::u64 fast_used =
        fast_cap > fa->remaining_size_ ? fast_cap - fa->remaining_size_ : 0;
    const clio::run::u64 host_used =
        host_cap > ha->remaining_size_ ? host_cap - ha->remaining_size_ : 0;
    // RAW remaining is printed alongside the derived used, because the
    // derived number alone is not interpretable: if a queried pool does not
    // exist or the stat fails, remaining reads 0 and "used" then equals the
    // full capacity -- which looks like a completely full tier rather than a
    // failed query. Both were indistinguishable in the first version of this
    // report and it nearly produced a false VIOLATION.
    std::fprintf(stderr,
                 "TIER SPLIT: kHBM used=%lluMiB cap=%lluMiB remain=%lluMiB | "
                 "host used=%lluMiB cap=%lluMiB remain=%lluMiB%s\n",
                 (unsigned long long)(fast_used >> 20),
                 (unsigned long long)(fast_cap >> 20),
                 (unsigned long long)(fa->remaining_size_ >> 20),
                 (unsigned long long)(host_used >> 20),
                 (unsigned long long)(host_cap >> 20),
                 (unsigned long long)(ha->remaining_size_ >> 20),
                 (fast_used == 0 && fa->remaining_size_ == fast_cap)
                     ? "   <-- nothing landed in the fastest tier"
                     : "");
  }

  std::fprintf(stderr,
               "GVW mode=%s%s blocks=%u thr=%u hbm=%lluMB slots=%u pages=%llu "
               "flat=%u%% logical=%.1fMB stored=%.1fMB fits=%s ms=%llu GB/s=%.2f "
               "checksum=%s put_errors=%llu faults=%llu get_errors=%llu "
               "evicts=%llu rounds=%u memcpy_pin_gbps=%.2f memcpy_page_gbps=%.2f\n",
               baseline ? "baseline" : (compressed ? (gpu_codec ? "nvcomp" : "lz4") : "raw"),
               baseline ? "" : "+yield", blocks, nthreads,
               (unsigned long long) hbm_mb, slots,
               (unsigned long long) (n / page_elems), flat_pct,
               logical / (1024.0 * 1024.0), stored / (1024.0 * 1024.0),
               (stored <= hbm_mb * 1024ull * 1024ull) ? "yes" : "no",
               (unsigned long long) best_ms, best_gbps, ok ? "OK" : "MISMATCH",
               (unsigned long long) stats.put_errors,
               (unsigned long long) stats.faults,
               (unsigned long long) stats.get_errors,
               (unsigned long long) stats.evicts, rounds,
               mcp.pinned_gbps, mcp.pageable_gbps);
  return ok ? 0 : 1;
}

#endif  // !CTP_IS_DEVICE_PASS
