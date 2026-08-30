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
  const bool halo_off = getenv("GS_NO_HALO") != nullptr;
  u64 page_kb = 1024, data_mb = 16384, hbm_mb = 4096;
  int repeat = 3;
  // Out-of-core WITHOUT in-kernel faulting: synchronous CTE reads AND
  // writes, synchronous memcpy both ways, kernel torn down per z.
  bool baseline = false;
  // Storage tier: without it no workload ever touches a disk.
  unsigned long long nvme_mb = 0;
  std::string nvme_path = "/tmp/gv_storage_tier.dat";
  bool hbm_only = false;
  float Du = 0.2f, Dv = 0.1f, F = 0.02f, K = 0.048f, dt = 1.0f;

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
    else if (a == "--Du") Du = nextf();
    else if (a == "--Dv") Dv = nextf();
    else if (a == "--F") F = nextf();
    else if (a == "--K") K = nextf();
    else if (a == "--dt") dt = nextf();
    else if (a == "--help") {
      std::printf("usage: %s [--blocks N] [--threads N] [--slots N] "
                  "[--steps N] [--page-kb N] [--data-mb N] [--hbm-mb N] "
                  "[--repeat N] [--hbm-only] [--Du f] [--Dv f] [--F f] "
                  "[--K f] [--dt f]\n", argv[0]);
      return 0;
    }
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
  // MULTI-NODE IS STILL REFUSED, and the exchange below is why it is close
  // rather than done. Slab bounds, shared namespace, per-step flush +
  // barrier + ClearCache and the checksum reduction are all in place, and
  // a 2-node run is BIT-EXACT against the single-node reference for steps
  // 0, 1 and 2. It diverges at step 3 (rel 4.6e-3, growing to 8.4e-3 by
  // step 4).
  //
  // WHERE IT GOES WRONG. ClearCache does not refuse a page whose flush is
  // still in flight -- it zeroes `flushing` and drops `data` outright, so
  // an outstanding device-side BeginFlush is discarded, and a task that
  // completes afterwards can land in a frame already reassigned to another
  // page. The first two steps survive because nothing is still in flight
  // when the clear runs; step 3 is where they start to overlap. The fix is
  // to settle outstanding flushes before invalidating, which needs a
  // primitive the vector does not currently expose.
  //
  // This bench's checksum has a TOLERANCE, so unlike gmx it cannot fail
  // loudly on a stale plane -- which is exactly why it must not run
  // multi-node until it is right.
  if (nodes > 1) {
    std::fprintf(stderr,
                 "GRAYSCOTT ERROR: --nodes %u requested, but the halo "
                 "exchange diverges from step 3 (ClearCache discards "
                 "in-flight flushes). Refusing rather than reporting a "
                 "plausible, wrong checksum.\n", nodes);
    return 2;
  }
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
        // MaxBwDpe sorts the preferred group DESCENDING and the vector puts at
        // blob score 1.0, so the HIGHER score is the preferred tier: HBM must
        // sit above the host tier.
        << "      - path: \"hbm::gv_gs_hbm\"\n        bdev_type: \"hbm\"\n"
        << "        capacity_limit: \"" << hbm_mb << "MB\"\n"
        << "        score: 1.0\n";
    if (!hbm_only) {
      cfg << "      - path: \"ram::gv_gs_ram\"\n        bdev_type: \"ram\"\n"
          << "        capacity_limit: \"" << (data_mb + 1024) << "MB\"\n"
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
            << "        persistence_level: \"temporary\"\n"
            << "        capacity_limit: \"" << nvme_mb << "MB\"\n"
            << "        score: 0.0\n";
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

  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
    gs::LaunchSeed(g, b, gpu, dev, plane, nx, ny, nz, zper, ubase, vbase,
                   zbase, zend, vw, sv);
  });
  ctp::GpuApi::Synchronize();
  // THE SEED IS SHARDED TOO, so the first step's halo planes belong to a
  // peer and must be published before anyone reads them.
  if (nodes > 1) {
    vec.FlushResidentToCte();
    if (!clio_bench_dist::Barrier(*cte_red, red_tag, node, nodes,
                                  red_round++, "gsseed")) {
      std::fprintf(stderr, "GRAYSCOTT ERROR: seed barrier failed\n");
      return 1;
    }
    vec.ClearCache();
  }

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
    ctp::GpuApi::Synchronize();
    const double t0 = NowMs();
    u64 cu = ubase, cv = vbase, nu = unext, nv = vnext;
    for (u32 s = 0; s < steps; ++s) {
      if (baseline) {
        if (!run_baseline_step(cu, cv, nu, nv)) {
          std::fprintf(stderr, "GRAYSCOTT ERROR: baseline step failed\n");
          return 1;
        }
      } else {
        runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          gs::LaunchStep(g, b, gpu, dev, plane, nx, ny, nz, zper, cu, cv, nu,
                         nv, Du, Dv, F, K, dt, zbase, zend, vw, sv);
        });
      }
      // THIS IS THE HALO EXCHANGE. Computing plane z needs z-1 and z+1, so
      // at a slab edge a node reads a plane its NEIGHBOUR just wrote. A
      // page the neighbour still holds resident is invisible to everyone
      // else, so without this the next step reads a stale boundary plane
      // and the field quietly diverges -- and unlike gmx, whose gates are
      // bit-exact, this bench's checksum has a tolerance that would absorb
      // the error rather than fail. The flush publishes this slab; the
      // barrier stops any node starting the next step before every peer
      // has published. GS_NO_HALO=1 skips both, as the negative control.
      if (nodes > 1 && !halo_off) {
        ctp::GpuApi::Synchronize();
        // StepCoro BeginFlush-es the planes it writes and EndFlush-es at
        // the end, but dropping this whole-table publish measured WORSE
        // (5.98% vs 0.84%), so some writes are not reaching the CTE by
        // kernel exit alone. Keep it until that is understood.
        vec.FlushResidentToCte();
        if (!clio_bench_dist::Barrier(*cte_red, red_tag, node, nodes,
                                      red_round++, "gsbar")) {
          std::fprintf(stderr, "GRAYSCOTT ERROR: halo barrier failed\n");
          return 1;
        }
        // FLUSHING IS ONLY HALF OF IT. A node that faulted in a peer's
        // boundary plane KEEPS that frame resident, so next step it reads
        // its own stale copy and never notices the peer rewrote the plane.
        // Publishing without invalidating measured 34251.995658 against a
        // single-node 36410.579344 -- 5.9% off, silently. Drop every frame
        // so the next step refaults from the CTE. Ordering matters: flush
        // (publish mine), barrier (everyone has published), THEN clear.
        vec.ClearCache();
      }
      std::swap(cu, nu);
      std::swap(cv, nv);
    }
    ctp::GpuApi::Synchronize();
    const double ms = NowMs() - t0;
    if (ms < best_ms) best_ms = ms;

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
  {
    clio::run::bdev::Client t_fast(clio::run::PoolId(512, 1));
    clio::run::bdev::Client t_host(clio::run::PoolId(513, 1));
    auto fa = t_fast.AsyncGetStats(); fa.Wait();
    auto ha = t_host.AsyncGetStats(); ha.Wait();
    const clio::run::u64 fast_cap = (clio::run::u64)hbm_mb * 1024ull * 1024ull;
    const clio::run::u64 host_cap = (clio::run::u64)(data_mb + 1024) * 1024ull * 1024ull;
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
               "GRAYSCOTT mode=%s blocks=%u thr=%u nx=%llu ny=%llu nz=%llu "
               "page_kb=%llu slots=%u steps=%u data_mb=%.0f hbm_mb=%llu "
               "ms=%.1f GB/s=%.2f v_checksum=%.6f faults=%llu evicts=%llu "
               "puts=%llu get_errors=%llu put_errors=%llu memcpy_pin_gbps=%.2f memcpy_page_gbps=%.2f\n",
               baseline ? "baseline" : "paged",
               blocks, threads, (unsigned long long)nx, (unsigned long long)ny,
               (unsigned long long)nz, (unsigned long long)page_kb, slots,
               steps, logical_mb, (unsigned long long)hbm_mb, best_ms, gbps,
               checksum, (unsigned long long)st.faults,
               (unsigned long long)st.evicts, (unsigned long long)st.puts,
               (unsigned long long)st.get_errors,
               (unsigned long long)st.put_errors,
               mcp.pinned_gbps, mcp.pageable_gbps);

  ctp::GpuApi::Free(d_sum);
  BenchFlushData();
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}

#endif  // !CTP_IS_DEVICE_PASS
