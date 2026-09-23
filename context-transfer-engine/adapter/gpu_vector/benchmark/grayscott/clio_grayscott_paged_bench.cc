/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Gray-Scott reaction-diffusion over a GPU vector whose grid does not fit on
 * the device.
 *
 * This is the STENCIL access pattern, and it is the one the other three
 * benchmarks do not have:
 *
 *   - flush   writes a region and flushes it       (write, no reuse)
 *   - weights re-reads a matrix every pass         (read, full reuse)
 *   - kmeans  streams the point set once per pass  (read, no reuse)
 *   - THIS    reads a 3-plane window that SLIDES   (read+write, partial reuse)
 *
 * The sliding window is the point. Computing plane z needs planes z-1, z and
 * z+1, and computing z+1 then needs z, z+1, z+2 -- so two of the three planes
 * are immediately reused. A cache of >= 4 pages therefore turns 3 reads per
 * plane into 1, and a smaller cache cannot. That is a real reuse distance,
 * unlike kmeans where nothing is revisited within a pass.
 *
 * LAYOUT. One page is exactly one XY plane, so the page size chosen on the
 * command line sets the plane dimensions (16 KB -> 64x64, 8 MB -> 2048x1024).
 * That makes --page-kb a first-class axis instead of an arbitrary chunking of
 * a fixed grid. The vector holds four regions -- u, v, u_next, v_next -- and
 * the step writes into the next pair and then swaps, because a stencil cannot
 * be computed in place.
 *
 * CACHE REQUIREMENT. The kernel holds three input planes at once (z-1, z, z+1)
 * plus writes an output plane, so slots must be >= 4. A smaller cache is
 * REJECTED rather than run: with slots < 4 a plane the kernel is still reading
 * can be evicted under it, which would not crash but would silently read
 * whatever replaced it.
 *
 * Correctness is a checksum of the final field, compared across configurations
 * with a RELATIVE TOLERANCE and only WITHIN one page size (page size sets the
 * grid geometry here, so a different page size solves a different problem).
 *
 * KNOWN OPEN ISSUE: THE RESULT IS NOT REPRODUCIBLE RUN TO RUN.
 *
 * The SAME configuration, run three times, gives checksums spread over
 * 3.37e-04 (1239598.10 / 1239180.61 / 1239424.31 at 16 blocks, 64KB pages).
 * A deterministic stencil should be bit-identical, and the double-precision
 * reduction over 1.34e8 values only reassociates at ~1e-12, so this is a
 * genuine data race or stale read that remains in this benchmark.
 *
 * This supersedes an earlier reading of the same evidence. The difference
 * BETWEEN block counts (8.39e-04) is the same order as the run-to-run spread,
 * so it was never established as a decomposition effect -- it was mostly
 * nondeterminism, and a fixed-configuration control should have been run
 * before attributing it to the block count.
 *
 * Two real defects were found and fixed along the way and did reduce it
 * (7.71e-03 -> 1.70e-03 -> 8.39e-04): missing cross-step flush waits, and a
 * per-block cache that carried stale copies of neighbouring blocks' planes
 * across a region swap. Neither closed it.
 *
 * TREAT THE TIMINGS AS INDICATIVE ONLY until this is resolved.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include "../bench_flush_data.h"
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "../bench_memcpy_probe.h"

/** CTE core pool id, matching pool_id "512.0" in the server config the
 *  benchmark writes. The baseline talks to the core directly. */
const clio::run::PoolId kCorePool(512, 0);

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

#if defined(CLIO_YIELD_CORO)
static constexpr u32 kYieldLaneBytes = 4096;
#else
static constexpr u32 kYieldLaneBytes = 256;
#endif


/*
 * THE DEVICE CODE IS NOT HERE ANY MORE.
 *
 * The workload -- the initial condition, the reaction term, the three
 * coroutines and the staged-plane baseline -- lives in grayscott_kernels.h
 * (over grayscott_math.h), in ONE copy compiled by both backends. The
 * launches live in cuda/ and sycl/ and differ only in how a grid is
 * submitted. This file is the host driver and is now ordinary C++.
 *
 * See grayscott_launch.h for why the seam is at the launch.
 */
#include "grayscott_kernels.h"
#include "grayscott_launch.h"

namespace gs = clio::gv_bench::grayscott;

// HOST DRIVER ONLY BELOW THIS LINE. Under CUDA this file is still compiled
// BY the CUDA compiler, whose device pass member-checks host bodies -- and
// gv::Vector is #if !CTP_IS_DEVICE_PASS. Under SYCL the guard is
// transparent: the driver is a plain C++ TU.

#if !CTP_IS_DEVICE_PASS

// Cross-node reduction. Included INSIDE the device-pass guard: it uses the
// CTE client, whose members are compiled out of the CUDA device pass.
#include "../bench_dist.h"
#include "../bench_ckpt.h"
// Host-only for the same reason: the prefetcher is pure host policy that
// speaks to the CTE, and never appears in a kernel.
#include "grayscott_prefetch.h"

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}

  /**
   * Route this driver's post-round states into the vector's prefetchers.
   *
   * The seam is the runtime's (Yieldable knows nothing about the CTE) and the
   * policy is the vector's (which owns the tag and the client); this call is
   * the one line that joins them.
   */
  void BindPrefetch(gv::Vector<float> &vec) {
    drv_.SetYieldObserver(vec.YieldObserver());
  }

  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    // Both resets are required: RunToCompletion does not reset, so a reused
    // runner whose driver still reads "done" skips the launch entirely.
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [] {}, /*max_rounds=*/2000000,
      gv::ResumeWhenComplete);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

}  // namespace

int main(int argc, char **argv) {
  u32 blocks = 64, threads = 256, slots = 12, steps = 4;
  // Z-SLAB DECOMPOSITION, like the MPI edition. --nodes N --node i gives
  // this process a contiguous slab of the SAME global field; nz stays
  // global so the fixed boundary (z == 0, z == nz-1) is honoured however
  // the domain is cut. One node is the whole field, so a single-process run
  // is unchanged and is the reference a distributed run must reproduce.
  u32 nodes = 1, node = 0;
  // NEGATIVE CONTROL. With the exchange off a multi-node run must produce
  // a DIFFERENT checksum; if it does not, the exchange is not doing
  // anything and the passing run proves nothing.
  // NON-EMPTY, not merely present. docker-compose passes an unset variable
  // through as GS_NO_HALO= (empty), and getenv returns a non-null empty
  // string for that -- so a presence test silently enables the control in
  // every harness run, which is exactly how a correct build reported the
  // control's wrong answer.
  const char *const gs_no_halo = getenv("GS_NO_HALO");
  const bool halo_off = gs_no_halo != nullptr && gs_no_halo[0] != '\0';
  u64 page_kb = 1024, data_mb = 16384, hbm_mb = 4096;
  int repeat = 3;
  // Out-of-core WITHOUT in-kernel faulting: synchronous CTE reads AND
  // writes, synchronous memcpy both ways, kernel torn down per z.
  bool baseline = false;
  // Storage tier: without it no workload ever touches a disk.
  unsigned long long nvme_mb = 0;
  std::string nvme_path = "/tmp/gv_storage_tier.dat";
  bool hbm_only = false;
  float Du = 0.2f, Dv = 0.1f, F = 0.02f, K = 0.048f, dt = 0.5f;
  // PREFETCH: tier hints issued from the driver's gap between rounds. "none"
  // is the baseline and the default -- registering a prefetcher changes where
  // pages live, so every existing number stays comparable unless asked.
  // "stride" is the generic control that knows nothing about Gray-Scott; the
  // gap between it and "gs" is what "workload-specific" is worth.
  std::string prefetch = "none";
  // 0 = SIZE IT FROM THE FAST TIER; see the derivation below.
  u64 lookahead = 4;
  float pf_hot = 1.0f, pf_warm = 0.2f, pf_cold = 0.0f;
  bool pf_demote = true;
  // 0 = no gate. See PrefetchPolicy::stall_every_.
  u64 pf_stall_every = 0;
  // Tier scores and the host tier's capacity. Defaults reproduce the historic
  // configuration exactly; see the config writer for what they mean.
  float hbm_score = 1.0f, ram_score = 0.2f, nvme_score = 0.0f;
  // CHECKPOINTING. 0 = off. Every N steps the whole vector is snapshotted
  // with Vector::Copy (the lazy copy-on-write primitive) and then
  // MATERIALISED, because an untouched Copy is zero bytes and would produce
  // no checkpoint at all. `ckpt_drain` demotes each finished checkpoint out
  // of the fast tier: a materialised checkpoint lands at blob score 1.0 and
  // otherwise sits in VRAM holding cold history that nobody reads again.
  u64 ckpt_every = 0;
  bool ckpt_drain = false;
  bool ckpt_final = true;   // --no-ckpt: skip the end-of-run checkpoint
  float ckpt_cold = 0.0f;
  unsigned long long ram_mb = 0;   // 0 = data_mb + 1024, the historic value

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    auto nextf = [&]() -> float {
      return (i + 1 < argc) ? std::strtof(argv[++i], nullptr) : 0.0f;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--slots") slots = static_cast<u32>(next());
    else if (a == "--steps") steps = static_cast<u32>(next());
    else if (a == "--nodes") nodes = static_cast<u32>(next());
    else if (a == "--node") node = static_cast<u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--data-mb") data_mb = next();
    else if (a == "--hbm-mb") hbm_mb = next();
    else if (a == "--repeat") repeat = static_cast<int>(next());
    else if (a == "--hbm-only") hbm_only = true;
    else if (a == "--nvme-mb") nvme_mb = next();
    // next() parses a number; the path needs the raw argv token.
    else if (a == "--nvme-path" && i + 1 < argc) nvme_path = argv[++i];
    else if (a == "--baseline") baseline = true;
    else if (a == "--prefetch" && i + 1 < argc) prefetch = argv[++i];
    else if (a == "--lookahead") lookahead = next();
    else if (a == "--pf-hot") pf_hot = nextf();
    else if (a == "--pf-warm") pf_warm = nextf();
    else if (a == "--pf-cold") pf_cold = nextf();
    else if (a == "--pf-no-demote") pf_demote = false;
    else if (a == "--pf-stall-every") pf_stall_every = next();
    else if (a == "--hbm-score") hbm_score = nextf();
    else if (a == "--ram-score") ram_score = nextf();
    else if (a == "--nvme-score") nvme_score = nextf();
    else if (a == "--ram-mb") ram_mb = next();
    else if (a == "--ckpt-every") ckpt_every = next();
    else if (a == "--ckpt-drain") ckpt_drain = true;
    else if (a == "--no-ckpt") ckpt_final = false;
    else if (a == "--ckpt-cold") ckpt_cold = nextf();
    else if (a == "--Du") Du = nextf();
    else if (a == "--Dv") Dv = nextf();
    else if (a == "--F") F = nextf();
    else if (a == "--K") K = nextf();
    else if (a == "--dt") dt = nextf();
    else if (a == "--help") {
      std::printf("usage: %s [--blocks N] [--threads N] [--slots N] "
                  "[--steps N] [--page-kb N] [--data-mb N] [--hbm-mb N] "
                  "[--repeat N] [--hbm-only] [--Du f] [--Dv f] [--F f] "
                  "[--K f] [--dt f]\n"
                  "       [--prefetch none|stride|gs] [--lookahead N] "
                  "(0 = size it to the fast tier)\n"
                  "       [--pf-hot f] [--pf-warm f] [--pf-cold f] "
                  "[--pf-no-demote] [--pf-stall-every N]\n"
                  "       [--nvme-mb N] [--nvme-path P] [--ram-mb N]\n"
                  "       [--ckpt-every N] [--ckpt-drain] [--ckpt-cold f]\n"
                  "       [--hbm-score f] [--ram-score f] [--nvme-score f]\n"
                  "         tier scores: a tier scored ABOVE the vector's put "
                  "score (0.5) is\n"
                  "         never a first choice. For capacity-ordered "
                  "placement across all\n"
                  "         three tiers pass e.g. 0.5 / 0.3 / 0.1.\n",
                  argv[0]);
      return 0;
    }
  }

  if (prefetch != "none" && prefetch != "stride" && prefetch != "gs") {
    std::fprintf(stderr, "GRAYSCOTT ERROR: --prefetch must be one of "
                 "none|stride|gs (got '%s')\n", prefetch.c_str());
    return 2;
  }
  // A prefetch run against the baseline driver would measure nothing: the
  // baseline tears the kernel down per z and never yields, so no cursor is
  // ever published and no hint is ever derived. Refused rather than run,
  // because it would report prefetch=gs and sent=0 and look like a result.
  if (baseline && prefetch != "none") {
    std::fprintf(stderr, "GRAYSCOTT ERROR: --baseline does not yield, so no "
                 "cursor is published and --prefetch %s would emit nothing.\n",
                 prefetch.c_str());
    return 2;
  }

  // The coroutine-mode refusal that used to be here moved with the kernels:
  // a build that cannot compile them does not produce this target at all now
  // (see the CMake guards, and adapter/CMakeLists.txt).
  // The kernel holds 6 input planes + 2 output planes at once. A smaller cache
  // could evict a plane the kernel is still reading -- that would not crash,
  // it would silently read whatever replaced it, so it is refused.
  const u32 kNeededSlots = 10;
  if (slots < kNeededSlots) {
    std::fprintf(stderr,
                 "GRAYSCOTT ERROR: slots=%u but the stencil holds %u planes at "
                 "once (z-1,z,z+1 of u and v, plus both outputs). A smaller "
                 "cache would let a plane still being read be evicted.\n",
                 slots, kNeededSlots);
    return 2;
  }

  const u64 page_bytes = page_kb * 1024;

  // REFERENCE CEILING for the paging path: a bare H2D cudaMemcpy at exactly
  // this page size, on an idle device before the runtime starts. The gap
  // between this and the achieved rate is the vector's overhead.
  const MemcpyProbe mcp = ProbeMemcpyBandwidth(static_cast<size_t>(page_bytes));
  const u64 plane = page_bytes / sizeof(float);   // one page == one XY plane
  // Square-ish plane: nx*ny == plane, both powers of two.
  u64 nx = 1, ny = plane;
  while (nx * 2 <= ny) { nx *= 2; ny /= 2; }
  // Four regions (u, v, u_next, v_next) share the dataset budget.
  const u64 total_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
  u64 nz = total_elems / (4 * plane);
  if (nz < 3) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: %lluMB at %lluKB pages leaves "
                 "nz=%llu planes; need at least 3 for a stencil.\n",
                 (unsigned long long)data_mb, (unsigned long long)page_kb,
                 (unsigned long long)nz);
    return 2;
  }
  if (node >= nodes) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: --node %u out of range for "
                 "--nodes %u\n", node, nodes);
    return 2;
  }
  // This node's slab of the global z range. The halo planes it reads are
  // its neighbours' pages, which is why the tag namespace below is SHARED:
  // a vector's pages are CTE blobs and blobs hash across the cluster, so
  // the paging path IS the halo exchange. Per-node namespaces would give
  // each node a private field and silently decouple the physics.
  // HOW THE HALO EXCHANGE WORKS, now that it does.
  //
  // Every Fetch used to pass generation 0, and generation 0 means ANY
  // VERSION IS ACCEPTABLE -- so a node never demanded a current halo
  // plane and was content with a stale one. Steps 0-2 happened to come
  // out right; the divergence started at step 3 and was nondeterministic.
  //
  // The fix is to demand a generation, but ONLY of a neighbour's plane.
  // A page's generation is stamped by the FETCH that delivers it, not by
  // the flush that publishes it, so a plane this node wrote locally and
  // never re-fetched sits at generation 0 forever -- demanding one on it
  // hangs the block with "gen stall: page N at gen 0 want G". Own planes
  // are current by construction and pass 0; the halo passes the step's
  // generation. The seed publishes AS generation 1, which is what step 0
  // demands; step s reads at s+1 and publishes as s+2.
  //
  // MEASURED, 2 nodes vs the single-node reference of 36410.579344:
  //   demand on  -> 36410.579344, bit-exact, three consecutive runs
  //   demand off -> 36104.119147 (GS_NO_HALO=1, the negative control)
  // The control disables the DEMAND, not the barrier, because an earlier
  // version disabled only the barrier and still came out exact -- which
  // is its own result: the generation IS the barrier, as md's harness
  // says. The explicit flush/barrier/ClearCache below may now be
  // redundant; that is an optimisation, not a correctness question, and
  // has not been tested on its own.
  const u64 nz_local = (nz + nodes - 1) / nodes;
  const u64 zbase = static_cast<u64>(node) * nz_local;
  const u64 zend = (zbase + nz_local < nz) ? (zbase + nz_local) : nz;
  if (nodes > 1 && zbase >= nz) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: --nodes %u leaves node %u with "
                 "no planes of nz=%llu\n", nodes, node,
                 (unsigned long long)nz);
    return 2;
  }
  const u64 zper = ((zend - zbase) + blocks - 1) / blocks;
  const u64 region = plane * nz;
  const u64 n = 4 * region;
  const double logical_mb =
      static_cast<double>(n * sizeof(float)) / (1024.0 * 1024.0);
  // Historic default: enough host tier to hold the whole grid with room over,
  // i.e. a hierarchy with no real pressure on the host tier.
  const unsigned long long ram_cap_mb =
      (ram_mb != 0) ? ram_mb : (data_mb + 1024);

  // ---- "OPTIMAL" LOOKAHEAD ---------------------------------------------
  //
  // --lookahead 0 sizes the promotion window to the FAST TIER rather than
  // guessing a constant, which is what makes "optimal prefetching" a defined
  // thing on this workload rather than a vibe.
  //
  // Gray-Scott's access order is a deterministic sweep: block b reads planes
  // z0..z1 in order, four regions per z. So the future is fully known, and
  // the Belady-optimal placement is simply "the fast tier holds the next
  // |fast tier| accesses". A prefetcher that promotes exactly that far ahead
  // and demotes everything behind it IS optimal for this pattern -- there is
  // no cleverer policy available, only a better-sized window.
  //
  // At any instant the hot set is, across the grid,
  //     blocks * 4 regions * (L + 2)          [+2: the z-1 and z still held]
  // planes. Setting that equal to the fast tier's capacity in planes:
  //     L = hbm_planes / (4 * blocks) - 2
  //
  // Undersized, the tier sits part empty and misses are served from a slower
  // one. Oversized, promotions evict each other before they are used and the
  // migrations are wasted work -- which is why this is a real optimum with a
  // curve either side of it, not a "bigger is better" knob.
  const u64 hbm_planes =
      static_cast<u64>(hbm_mb) * 1024ull * 1024ull / page_bytes;
  const u64 auto_lookahead =
      (hbm_planes > 4ull * blocks * 2ull)
          ? (hbm_planes / (4ull * blocks) - 2ull)
          : 1ull;
  if (lookahead == 0) {
    lookahead = auto_lookahead;
    std::printf("  lookahead: auto = %llu planes (fast tier holds %llu "
                "planes; %u blocks x 4 regions x (L+2))\n",
                (unsigned long long)lookahead,
                (unsigned long long)hbm_planes, blocks);
  }

  // THE BENCH OWNS ITS CONFIG ONLY WHEN NOBODY ELSE SUPPLIED ONE. Writing
  // one and Setenv-ing it with overwrite=1 unconditionally makes it
  // impossible to point this bench at a cluster: any CLIO_SERVER_CONF the
  // caller exported is clobbered a line later, so every node stands up its
  // own single-host runtime on the same port and they collide. A distributed
  // harness needs exactly that config -- one naming a hostfile and the other
  // nodes -- so an already-set CLIO_SERVER_CONF is left alone.
  if (getenv("CLIO_SERVER_CONF") != nullptr) {
    std::printf("  runtime: using CLIO_SERVER_CONF=%s (not writing one)\n",
                getenv("CLIO_SERVER_CONF"));
  } else {
    std::ofstream cfg("gv_grayscott_bench.yaml");
    cfg << "networking:\n  port: 9441\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n"
        << "  first_busy_wait: 10000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
        << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
        << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n"
        // TIER SCORES ARE A COMMAND-LINE AXIS, and getting them wrong is the
        // difference between a three-tier run and a two-tier run that says it
        // has three.
        //
        // MaxBwDpe partitions targets into `target_score <= blob_score`
        // (PREFERRED, sorted by score DESCENDING) and the rest (FALLBACK,
        // taken only when no preferred tier has room). The vector writes every
        // page at kVectorBlobScore = 0.5, so A TIER SCORED ABOVE 0.5 IS NEVER
        // A FIRST CHOICE FOR ANYTHING THIS VECTOR WRITES.
        //
        // The old defaults -- HBM 1.0, RAM 0.2 -- put HBM in the fallback
        // group, which is why every run of this benchmark reported
        // "kHBM used=0MiB ... nothing landed in the fastest tier" while the
        // comment here asserted the opposite. Preserved as the DEFAULTS
        // anyway, so existing sweeps reproduce exactly; a run that wants
        // capacity-ordered placement across all three tiers passes scores at
        // or below 0.5, descending (e.g. 0.5 / 0.3 / 0.1).
        << "      - path: \"hbm::gv_gs_hbm\"\n        bdev_type: \"hbm\"\n"
        << "        capacity_limit: \"" << hbm_mb << "MB\"\n"
        << "        score: " << hbm_score << "\n";
    if (!hbm_only) {
      cfg << "      - path: \"ram::gv_gs_ram\"\n        bdev_type: \"ram\"\n"
          << "        capacity_limit: \"" << ram_cap_mb << "MB\"\n"
          << "        score: " << ram_score << "\n";
    }
      // OPTIONAL STORAGE TIER. Without it the whole dataset lives in host
      // DRAM and NOTHING EVER TOUCHES STORAGE -- what such a run measures is
      // DRAM over PCIe, not I/O. On this machine that spill is nearly free,
      // which is exactly why a cache-size sweep over a DRAM-only hierarchy
      // comes back flat: there is no penalty for the cache to save.
      //
      // score BELOW the host tier, so the preferred group ranks it last.
      //
      // IT MUST STILL BE <= THE BLOB SCORE TO BE PREFERRED AT ALL. At the
      // default 0.0 it is, for any blob score -- but a prefetcher demoting to
      // "cold" has to name a score at or above this one, or storage lands in
      // the FALLBACK group and the ordering there is by the bandwidth model
      // rather than by the operator's declared tiering. That model is not
      // trustworthy for this (see the note in MaxBwDpe::SelectTargets, which
      // rated an HBM tier at 118 MB/s against host RAM at 1600), so a cold
      // score should equal this number rather than merely be below it.
      if (nvme_mb > 0) {
        cfg << "      - path: \"" << nvme_path << "\"\n"
            << "        bdev_type: \"file\"\n"
            << "        persistence_level: \"temporary\"\n"
            << "        capacity_limit: \"" << nvme_mb << "MB\"\n"
            << "        score: " << nvme_score << "\n";
      }

    cfg << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_grayscott_bench.yaml", 1);
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "GRAYSCOTT ERROR: cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);
  // Per-block device state the SYCL backend allocates once; no-op on CUDA.
  gs::InitBackend(blocks, gpu);

  std::printf("Gray-Scott over a GPU vector\n"
              "  grid=%llux%llux%llu (page = one %lluKB plane)\n"
              "  blocks=%u (%llu planes each) threads=%u cache=%u pages/block\n"
              "  fields=4 (u,v,u_next,v_next)  total=%.0fMB  kHBM=%lluMB%s\n"
              "  steps=%u Du=%.3f Dv=%.3f F=%.3f K=%.3f dt=%.2f\n",
              (unsigned long long)nx, (unsigned long long)ny,
              (unsigned long long)nz, (unsigned long long)page_kb, blocks,
              (unsigned long long)zper, threads, slots, logical_mb,
              (unsigned long long)hbm_mb, hbm_only ? " (HBM ONLY)" : "",
              steps, Du, Dv, F, K, dt);

  // ASSOCIATIVITY, NOT FULL ASSOCIATIVITY. A lookup scans its set, so one
  // giant set costs O(frames) per probe -- kmeans went 4.7 s -> 42 s as a
  // 512-way cache. Keep a set per block and floor the width at 8, which is
  // what covers several blocks colliding on one set.
  // This kernel holds eight planes per step per block, so the set must be
  // wide enough for several blocks' worth of those.
  // ---- cross-node reduction client -------------------------------------
  // After the runtime is up, and only when there is a reduction to do.
  std::unique_ptr<clio::cte::core::Client> cte_red;
  clio::cte::core::TagId red_tag{};
  u64 red_round = 0;
  if (nodes > 1) {
    cte_red = std::make_unique<clio::cte::core::Client>(
        clio::cte::core::kCtePoolId);
    auto t = cte_red->AsyncGetOrCreateTag("gv_grayscott_red");
    t.Wait();
    if (t->GetReturnCode() != 0) {
      std::fprintf(stderr, "GRAYSCOTT ERROR: could not create reduction "
                   "tag\n");
      return 1;
    }
    red_tag = t->tag_id_;
  }

  gv::Vector<float> vec("gv_grayscott", {0}, page_bytes, blocks,
                        slots < 24u ? 24u : slots, n);
  vec.EnableStats();
  auto dev = vec.GetDevice(0);
  YieldRunner runner(blocks, threads);

  const u64 ubase = 0, vbase = region, unext = 2 * region, vnext = 3 * region;

  // ---- PREFETCHERS ------------------------------------------------------
  // Registered on the VECTOR (which owns the tag and the CTE client) and
  // fired from the DRIVER's post-round gap (which is the only host time with
  // no kernel resident). BindPrefetch joins the two.
  gv::PrefetchPolicy pf_policy;
  pf_policy.hot_ = pf_hot;
  pf_policy.warm_ = pf_warm;
  pf_policy.cold_ = pf_cold;
  pf_policy.lookahead_ = lookahead;
  pf_policy.stall_every_ = static_cast<u32>(pf_stall_every);
  vec.SetPrefetchPolicy(pf_policy);
  std::shared_ptr<gs::GrayScottPrefetcher> gs_pf;
  if (prefetch == "gs") {
    // zbase/zend/zper are the SAME locals LaunchStep is given below, which is
    // what keeps the host's idea of block b's slab from drifting from the
    // kernel's.
    gs_pf = std::make_shared<gs::GrayScottPrefetcher>(
        plane, nz, zbase, zend, zper, pf_policy, pf_demote);
    vec.RegisterPrefetcher(gs_pf);
  } else if (prefetch == "stride") {
    // Blind to three of the four regions and to demotion entirely -- that is
    // the point of it. It is pointed at u's current region because the cursor
    // it sees counts z, and u is the region whose plane index the cursor
    // equals. It cannot learn that v, unext and vnext exist.
    vec.RegisterPrefetcher(std::make_shared<gv::StridePrefetcher>(
        ubase / plane, nz, pf_policy));
  }
  if (vec.HasPrefetchers()) runner.BindPrefetch(vec);

  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
    gs::LaunchSeed(g, b, gpu, dev, plane, nx, ny, nz, zper, ubase, vbase,
                   zbase, zend, vw, sv);
  });
  ctp::GpuApi::Synchronize();
  // NO seed-side flush/barrier/invalidate. SeedCoro publishes its own
  // planes AS generation 1 (BeginFlush(1) + EndFlush), and step 0's halo
  // fetch DEMANDS generation 1 -- the demand polls until the peer's
  // publish is served, so the generation is the barrier. A whole-table
  // flush here republishes nothing new, and an invalidate drops frames
  // that were never stale (the seed touches only this node's slab).

  double *d_sum = nullptr;
  d_sum = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_sum)>>(sizeof(double));

  // ---- BASELINE DRIVER ------------------------------------------------
  // Per z: blocking reads of the 6 input planes, blocking H2D, one kernel
  // launch, blocking D2H, then BLOCKING PUTS of the two output planes. Every
  // transfer is on the critical path and the grid is rebuilt for each z.
  //
  // The 6 reads are issued every iteration rather than kept as a sliding
  // window. That IS the model being measured: with the kernel torn down at
  // every step there is no in-kernel state to carry the window in, and a host
  // that wanted to keep one would be reimplementing the cache this benchmark
  // exists to compare against.
  ctp::ipc::FullPtr<char> bl_h[6];
  ctp::ipc::FullPtr<char> bl_out[2];
  float *bl_d[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  float *bl_dout[2] = {nullptr, nullptr};
  clio::cte::core::Client *bl_core = nullptr;
  const u64 page_bytes_gs = plane * sizeof(float);
  if (baseline) {
    for (int i = 0; i < 6; ++i) {
      bl_h[i] = CLIO_IPC->AllocateBuffer((size_t)page_bytes_gs);
      if (bl_h[i].IsNull()) {
        std::fprintf(stderr, "GRAYSCOTT ERROR: baseline alloc failed\n");
        return 1;
      }
      // GpuApi::Malloc fails fatally, which is the same abort with less code.
      bl_d[i] = ctp::GpuApi::Malloc<float>((size_t)page_bytes_gs);
    }
    for (int i = 0; i < 2; ++i) {
      bl_out[i] = CLIO_IPC->AllocateBuffer((size_t)page_bytes_gs);
      if (bl_out[i].IsNull()) {
        std::fprintf(stderr, "GRAYSCOTT ERROR: baseline alloc failed\n");
        return 1;
      }
      bl_dout[i] = ctp::GpuApi::Malloc<float>((size_t)page_bytes_gs);
    }
    bl_core = new clio::cte::core::Client(kCorePool);
  }
  // Blocking read of one plane into staging slot `i`.
  auto bl_read = [&](int i, u64 region_base, u64 z) {
    const std::string nm = std::to_string((region_base + z * plane) / plane);
    auto gf = bl_core->AsyncGetBlob(vec.TagId(), nm, 0, page_bytes_gs, 0,
                                    bl_h[i].shm_.template Cast<void>(),
                                    clio::run::PoolQuery::Local());
    gf.Wait();
    if (gf->GetReturnCode() != 0) {
      std::memset(bl_h[i].ptr_, 0, (size_t)page_bytes_gs);
    }
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(bl_d[i]), bl_h[i].ptr_,
                        (size_t)page_bytes_gs);
    return true;
  };
  // SYNCHRONOUS WRITE of one output plane: D2H, then a blocking PutBlob.
  auto bl_write = [&](int i, u64 region_base, u64 z) {
    ctp::GpuApi::Memcpy(bl_out[i].ptr_,
                        reinterpret_cast<const char *>(bl_dout[i]),
                        (size_t)page_bytes_gs);
    const std::string nm = std::to_string((region_base + z * plane) / plane);
    auto pf = bl_core->AsyncPutBlob(vec.TagId(), nm, 0, page_bytes_gs,
                                    bl_out[i].shm_.template Cast<void>(), 1.0f);
    pf.Wait();
    return pf->GetReturnCode() == 0;
  };
  auto run_baseline_step = [&](u64 cu_, u64 cv_, u64 nu_, u64 nv_) {
    for (u64 z = 0; z < nz; ++z) {
      const bool interior = (z > 0 && z + 1 < nz);
      const u64 zm = interior ? (z - 1) : z;
      const u64 zp = interior ? (z + 1) : z;
      if (!bl_read(0, cu_, zm) || !bl_read(1, cu_, z) || !bl_read(2, cu_, zp) ||
          !bl_read(3, cv_, zm) || !bl_read(4, cv_, z) || !bl_read(5, cv_, zp)) {
        return false;
      }
      gs::LaunchBaseline(threads, bl_d[0], bl_d[1], bl_d[2], bl_d[3], bl_d[4],
                         bl_d[5], bl_dout[0], bl_dout[1], plane, nx, ny,
                         interior ? 1 : 0, Du, Dv, F, K, dt);
      ctp::GpuApi::Synchronize();
      if (!bl_write(0, nu_, z) || !bl_write(1, nv_, z)) return false;
    }
    return true;
  };

  double best_ms = 1e30, checksum = 0.0;
  u64 ckpt_id = 0, ckpt_pages = 0;
  double ckpt_ms = 0.0;
  for (int r = 0; r < repeat; ++r) {
    // RE-SEED between repeats. Without this, repeat 2 continues evolving the
    // field left by repeat 1, so each timed run measures a different physical
    // state and the reported checksum depends on `repeat` -- which makes it
    // useless as a correctness check and makes the repeats non-comparable.
    if (r > 0) {
      runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        gs::LaunchSeed(g, b, gpu, dev, plane, nx, ny, nz, zper, ubase, vbase,
                       zbase, zend, vw, sv);
      });
      ctp::GpuApi::Synchronize();
    }
    vec.ResetStats();
    vec.ResetPrefetchStats();
    vec.ResetPrefetchHistory();
    vec.PrefetchRunBegin();
    ctp::GpuApi::Synchronize();
    const double t0 = NowMs();
    u64 cu = ubase, cv = vbase, nu = unext, nv = vnext;
    // Checkpoints are KEPT ALIVE for the whole run: dropping the handle would
    // let the copy tag go away and with it the bytes we just produced.
    std::vector<std::unique_ptr<gv::Vector<float>>> ckpts;
    for (u32 s = 0; s < steps; ++s) {
      // RE-ARM EVERY STEP. The swap below makes last step's outputs this
      // step's inputs, so a prefetcher still holding the old assignment
      // promotes the wrong four regions.
      //
      // NO HISTORY RESET HERE. The dedup map models the score the CTE holds
      // for a page, which a change of MEANING does not invalidate -- see
      // Vector::ResetPrefetchHistory. Clearing it re-sent every page its own
      // current score at each step boundary, which the CTE no-ops, and made
      // the promote/demote tally meaningless.
      if (gs_pf) gs_pf->SetRegions(cu, cv, nu, nv);
      if (baseline) {
        if (!run_baseline_step(cu, cv, nu, nv)) {
          std::fprintf(stderr, "GRAYSCOTT ERROR: baseline step failed\n");
          return 1;
        }
      } else {
        runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          gs::LaunchStep(g, b, gpu, dev, plane, nx, ny, nz, zper, cu, cv, nu,
                         nv, Du, Dv, F, K, dt, zbase, zend,
                         // GS_NO_HALO=1 forces generation 0 -- "any
                         // version" -- which is the control: it disables
                         // the DEMAND itself, not merely the barrier.
                         halo_off ? 0 : static_cast<u64>(s) + 1, vw, sv);
        });
      }
      // NO per-step flush/barrier/invalidate: the generational demand IS
      // the halo exchange. A halo fetch names generation s+1 and a stale
      // RESIDENT peer frame takes the gen-stale refetch path, so neither
      // an explicit barrier nor a cache clear is needed -- and the clear
      // was actively harmful: SettleAndInvalidate dropped EVERY frame
      // each step, so the eviction path never engaged in a distributed
      // run (the OOC gate measured evicts=0 against 844 single-node,
      // which is what exposed this). The "removing the flush measured
      // 5.98% worse" note that used to justify this block was measured
      // BEFORE generations were threaded through, on single runs of a
      // quantity later shown to vary run-to-run.
      //
      // GS_NO_HALO=1 remains the negative control, applied where it now
      // belongs: it forces the step's halo demand back to generation 0
      // at the launch site.
      std::swap(cu, nu);
      std::swap(cv, nv);

      // ---- CHECKPOINT ------------------------------------------------
      // Vector::Copy is COPY-ON-WRITE: it registers the checkpoint chimod as
      // the copy tag's fault handler and duplicates nothing. So the snapshot
      // must be MATERIALISED, and materialised SOON: the handler pulls from
      // the source at fault time, and the region swap above overwrites each
      // pair every second step, so a checkpoint left lazy for two more steps
      // would silently capture later bytes instead of these.
      //
      // Materialisation is server-side (a size-0 put takes the put-side
      // materialise-only fault), so the bytes never cross to the host -- this
      // is a checkpoint being made durable, not a read-back.
      if (ckpt_every != 0 && ((s + 1) % ckpt_every) == 0) {
        const double ck0 = NowMs();
        auto snap = vec.Copy("gv_gs_ck" + std::to_string(ckpt_id));
        const u64 mat = snap->MaterializeAll();
        if (mat == 0) {
          std::fprintf(stderr, "GRAYSCOTT ERROR: checkpoint %llu failed to "
                       "materialize\n", (unsigned long long)ckpt_id);
          return 1;
        }
        ckpt_pages += mat;
        // DRAIN IT. A materialised checkpoint is written at blob score 1.0
        // (the handler puts at -1.0, which PutBlobImpl resolves to 1.0 for a
        // new blob), so it outranks the LIVE grid on every tier and parks
        // cold history in VRAM. Nothing ever reads it again in this run, so
        // demoting it is not speculative -- it is the drain that has to
        // happen anyway, done early instead of by eviction pressure later.
        if (ckpt_drain) {
          snap->SetPrefetchPolicy(pf_policy);
          snap->ScoreAllPages(ckpt_cold);
          snap->DrainPrefetch();
        }
        ckpts.push_back(std::move(snap));
        ckpt_ms += NowMs() - ck0;
        ++ckpt_id;
      }
    }
    ctp::GpuApi::Synchronize();
    const double ms = NowMs() - t0;
    if (ms < best_ms) best_ms = ms;
    // AFTER the timer, deliberately. The SUBMISSIONS are inside the timed
    // region -- they are host time in the round gaps and the run really pays
    // for them -- but the run never waits on a rescore to complete, so
    // charging it for that wait would be charging it for work it did not do.
    // The drain is here only so the tier occupancy read below reflects the
    // migrations this run asked for rather than a race with them.
    vec.DrainPrefetch();

    // GS_PLANE_DUMP=1 prints a per-plane sum of v. Diffing that between a
    // 1-node and a 2-node run says WHICH planes are wrong, which is the
    // question -- a whole-field checksum only says that something is. Uses
    // the existing sum kernel one plane at a time, so it measures the same
    // bytes the gate does rather than a second path that could differ.
    const char *const gs_dump = getenv("GS_PLANE_DUMP");
    if (gs_dump != nullptr && gs_dump[0] != '\0') {
      for (u64 z = zbase; z < zend; ++z) {
        ctp::GpuApi::Memset(d_sum, 0, sizeof(double));
        runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          gs::LaunchSum(g, b, gpu, dev, plane, nz, 1, cv, d_sum, z, z + 1,
                        vw, sv);
        });
        ctp::GpuApi::Synchronize();
        double pz = 0.0;
        ctp::GpuApi::Memcpy(&pz, d_sum, sizeof(double));
        std::printf("  PLANE %llu %.6f\n", (unsigned long long)z, pz);
      }
      std::fflush(stdout);
    }
    ctp::GpuApi::Memset(d_sum, 0, sizeof(double));
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      gs::LaunchSum(g, b, gpu, dev, plane, nz, zper, cv, d_sum, zbase, zend,
                    vw, sv);
    });
    ctp::GpuApi::Synchronize();
    ctp::GpuApi::Memcpy(&checksum, d_sum, sizeof(double));
    // Each node summed only its own slab; the gate compares a checksum of
    // the WHOLE field against a single-node reference, so the partials
    // have to be combined before it is reported.
    if (nodes > 1 &&
        !clio_bench_dist::ReduceSum(*cte_red, red_tag, node, nodes,
                                    red_round++, &checksum, 1, "gsred")) {
      std::fprintf(stderr, "GRAYSCOTT ERROR: checksum reduction failed\n");
      return 1;
    }
  }

  const auto st = vec.ReadStats(0);
  const auto pf = vec.ReadPrefetchStats();
  // Bytes touched per step: 6 input planes + 2 output planes per z.
  const double moved_gb =
      static_cast<double>(nz) * plane * sizeof(float) * 8.0 * steps /
      (1024.0 * 1024.0 * 1024.0);
  const double gbps = (best_ms > 0.0) ? moved_gb / (best_ms / 1000.0) : 0.0;

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
  //
  // ALL THREE TIERS, and the storage one is not optional to report: a
  // three-tier run whose report covers two of them cannot answer "what
  // fraction of the data is where", which on a tiered evaluation IS the
  // result. The third bdev is (514,1) by the same config-order rule.
  {
    const clio::run::u64 fast_cap = (clio::run::u64)hbm_mb * 1024ull * 1024ull;
    const clio::run::u64 host_cap =
        hbm_only ? 0ull : (clio::run::u64)ram_cap_mb * 1024ull * 1024ull;
    const clio::run::u64 stor_cap = (clio::run::u64)nvme_mb * 1024ull * 1024ull;
    // Bdev minors follow CONFIG ORDER, and the host tier is skipped entirely
    // under --hbm-only -- so storage is the SECOND entry there, not the
    // third. Indexing it unconditionally as 514 read a pool that does not
    // exist and reported the storage tier as completely full.
    clio::run::bdev::Client t_fast(clio::run::PoolId(512, 1));
    auto fa = t_fast.AsyncGetStats(); fa.Wait();
    const clio::run::u64 fast_rem = fa->remaining_size_;
    clio::run::u64 host_rem = 0, stor_rem = 0;
    if (!hbm_only) {
      clio::run::bdev::Client t_host(clio::run::PoolId(513, 1));
      auto ha = t_host.AsyncGetStats(); ha.Wait();
      host_rem = ha->remaining_size_;
    }
    if (nvme_mb > 0) {
      clio::run::bdev::Client t_stor(
          clio::run::PoolId(hbm_only ? 513 : 514, 1));
      auto sa = t_stor.AsyncGetStats(); sa.Wait();
      stor_rem = sa->remaining_size_;
    }
    auto used = [](clio::run::u64 cap, clio::run::u64 rem) {
      return cap > rem ? cap - rem : 0ull;
    };
    const clio::run::u64 fast_used = used(fast_cap, fast_rem);
    const clio::run::u64 host_used = used(host_cap, host_rem);
    const clio::run::u64 stor_used = used(stor_cap, stor_rem);
    const double total_mb =
        static_cast<double>((fast_used + host_used + stor_used) >> 20);
    auto pct = [&](clio::run::u64 u) {
      return total_mb > 0.0 ? 100.0 * static_cast<double>(u >> 20) / total_mb
                            : 0.0;
    };
    // RAW remaining is printed alongside the derived used, because the
    // derived number alone is not interpretable: if a queried pool does not
    // exist or the stat fails, remaining reads 0 and "used" then equals the
    // full capacity -- which looks like a completely full tier rather than a
    // failed query. Both were indistinguishable in the first version of this
    // report and it nearly produced a false VIOLATION.
    std::fprintf(stderr,
                 "TIER SPLIT: kHBM used=%lluMiB cap=%lluMiB remain=%lluMiB | "
                 "host used=%lluMiB cap=%lluMiB remain=%lluMiB | "
                 "stor used=%lluMiB cap=%lluMiB remain=%lluMiB%s\n",
                 (unsigned long long)(fast_used >> 20),
                 (unsigned long long)(fast_cap >> 20),
                 (unsigned long long)(fast_rem >> 20),
                 (unsigned long long)(host_used >> 20),
                 (unsigned long long)(host_cap >> 20),
                 (unsigned long long)(host_rem >> 20),
                 (unsigned long long)(stor_used >> 20),
                 (unsigned long long)(stor_cap >> 20),
                 (unsigned long long)(stor_rem >> 20),
                 (fast_used == 0 && fast_rem == fast_cap)
                     ? "   <-- nothing landed in the fastest tier"
                     : "");
    // The line the tiered evaluation actually reads: where the data ENDED UP,
    // as a fraction. A capacity split is what was ASKED for; this is what was
    // achieved, and on this workload the two came apart silently for as long
    // as the fast tier was unreachable.
    std::fprintf(stderr,
                 "TIER PCT: vram=%.1f%% dram=%.1f%% nvme=%.1f%% "
                 "(placed %.0fMiB of %.0fMiB logical)\n",
                 pct(fast_used), pct(host_used), pct(stor_used), total_mb,
                 logical_mb);
  }

  std::fprintf(stderr,
               "GRAYSCOTT mode=%s blocks=%u thr=%u nx=%llu ny=%llu nz=%llu "
               "page_kb=%llu slots=%u steps=%u data_mb=%.0f hbm_mb=%llu "
               "ms=%.1f GB/s=%.2f v_checksum=%.6f faults=%llu evicts=%llu "
               "puts=%llu get_errors=%llu put_errors=%llu memcpy_pin_gbps=%.2f memcpy_page_gbps=%.2f "
               // PREFETCH ACCOUNTING. Reported unconditionally, including for
               // a --prefetch none run where every field is 0: a run whose
               // pf_sent is 0 is NOT a prefetching run whatever its timings
               // say, and that has to be readable from the log rather than
               // inferred from the command line.
               "prefetch=%s lookahead=%llu stall_every=%llu "
               "pf_hints=%llu pf_sent=%llu "
               "pf_deduped=%llu pf_batches=%llu pf_promote=%llu "
               "pf_demote=%llu pf_errors=%llu pf_notfound=%llu "
               "pf_dropped=%llu ckpt_every=%llu ckpt_n=%llu "
               "ckpt_gb=%.2f ckpt_ms=%.0f ckpt_drain=%d\n",
               baseline ? "baseline" : "paged",
               blocks, threads, (unsigned long long)nx, (unsigned long long)ny,
               (unsigned long long)nz, (unsigned long long)page_kb, slots,
               steps, logical_mb, (unsigned long long)hbm_mb, best_ms, gbps,
               checksum, (unsigned long long)st.faults,
               (unsigned long long)st.evicts, (unsigned long long)st.puts,
               (unsigned long long)st.get_errors,
               (unsigned long long)st.put_errors,
               mcp.pinned_gbps, mcp.pageable_gbps,
               vec.PrefetcherNames().c_str(), (unsigned long long)lookahead,
               (unsigned long long)pf_stall_every,
               (unsigned long long)pf.hints_, (unsigned long long)pf.sent_,
               (unsigned long long)pf.deduped_,
               (unsigned long long)pf.batches_,
               (unsigned long long)pf.promotions_,
               (unsigned long long)pf.demotions_,
               (unsigned long long)pf.errors_,
               (unsigned long long)pf.not_found_,
               (unsigned long long)pf.dropped_,
               (unsigned long long)ckpt_every, (unsigned long long)ckpt_id,
               static_cast<double>(ckpt_pages) * page_bytes /
                   (1024.0 * 1024.0 * 1024.0),
               ckpt_ms, ckpt_drain ? 1 : 0);

  ctp::GpuApi::Free(d_sum);
  // FINAL-STATE CHECKPOINT: vector.Copy of the whole field, on
  // by default (--no-ckpt skips it). After every gate, because the
  // multi-node path drops the cache first -- see bench_ckpt.h.
  // Unlike the periodic --ckpt-every snapshots this one needs no
  // MaterializeAll: nothing writes the field after it, so the lazy copy
  // cannot be overtaken by a later step.
  std::unique_ptr<gv::Vector<float>> final_ck;
  if (ckpt_final) {
    final_ck = clio_bench_ckpt::FinalCheckpoint(vec, "gv_gs_ckpt_final", nodes);
  }
  BenchFlushData();
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}

#endif  // !CTP_IS_DEVICE_PASS
