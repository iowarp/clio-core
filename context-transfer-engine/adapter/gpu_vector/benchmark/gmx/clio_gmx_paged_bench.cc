/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * GROMACS science kernel -- PME charge spreading and force-stage gathering --
 * over a GPU vector whose GRID does not fit on the device.
 *
 * WHY THIS KERNEL. The md benchmark already covers the short-range pair
 * loop; what GROMACS adds that no other benchmark here has is PME: N atoms
 * SCATTER onto a K^3 mesh through an order-4 B-spline (each atom touches a
 * 4x4x4 neighbourhood), and the mesh is then read back at atom positions.
 * The mesh is the thing that outgrows VRAM, and spreading is the one PME
 * stage with a local, streamable access pattern -- the FFT that follows in
 * real PME needs the whole mesh at once, which is exactly the consumer that
 * cannot page (measured on GROMACS itself; see the eternia notes), so this
 * benchmark ends where the FFT would begin.
 *
 * DECOMPOSITION: ONE WRITER PER PAGE, BY CONSTRUCTION. A page is one XY
 * plane; a CUDA block owns a contiguous z-range of planes and PULLS into
 * each: the atoms whose spline base lands in bins z-3..z are exactly the
 * ones that touch plane z, so the block visits plane z once, accumulates
 * every contribution, publishes, and moves on. Two blocks never write one
 * page, so the write-site flush is sound under eviction -- the same rule the
 * md resort's gather learned (two blocks sharing a page silently clobber
 * each other under page-granular writeback).
 *
 * FIXED-POINT, FOR A DIGIT-EXACT GATE. Atoms within one plane still collide
 * on grid points, and float atomics make the sum order-dependent -- a
 * tolerance gate would then hide real staleness bugs inside "atomics
 * noise". Charge is therefore accumulated in Q40.24 fixed point on an
 * unsigned-64 mesh: integer addition commutes, so the paged run must match
 * the dense in-VRAM reference BIT FOR BIT, total charge must equal the sum
 * of input charges EXACTLY, and any lost or stale page shows as a hard
 * mismatch rather than a plausible wobble.
 *
 * GATES (all exact, no tolerances):
 *   CONSERVATION  sum over mesh == sum of input charges, in fixed point
 *   MESH          64-bit checksum of the paged mesh == dense reference mesh
 *   GATHER        fixed-point interpolation energy == dense reference
 *
 * OUT OF CORE: --cap M caps the mesh cache at M pages. The spread's window
 * is self-limiting (a block holds ONE plane at a time; the gather holds
 * four), so the floor is small and pressure means eviction of published
 * planes and refaults on the gather pass -- Fetch/Flush consistency, not
 * luck, is what the exact gates certify.
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
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

#if defined(CLIO_YIELD_CORO)
static constexpr u32 kYieldLaneBytes = 4096;
#endif


/*
 * THE DEVICE CODE IS NOT HERE ANY MORE.
 *
 * The workload -- the atom generator, the spline weights, the four
 * coroutines and the dense reference bodies -- lives in gmx_kernels.h (over
 * gmx_math.h), in ONE copy compiled by both backends. The launches live in
 * cuda/ and sycl/ and differ only in how a grid is submitted. This file is
 * the host driver and is now ordinary C++.
 *
 * See gmx_launch.h for why the seam is at the launch.
 */
#include "gmx_kernels.h"
#include "gmx_launch.h"

namespace gx = clio::gv_bench::gmx;
using gx::Frac01;
using gx::kFxScale;
using gx::Lcg;
using gx::Spline4;

#if !CTP_IS_DEVICE_PASS

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

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
        [] {}, /*max_rounds=*/2000000, gv::ResumeWhenComplete);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

}  // namespace

int main(int argc, char **argv) {
  u32 blocks = 16, threads = 256, cap = 0;
  // Skip the dense in-VRAM reference: a 6 GB-class mesh cannot hold a
  // second full copy beside the paged one, and the reference exists only
  // for the two bit-equality gates. CONSERVATION stays enforced -- it is
  // self-contained (mesh total == input charge, exact).
  bool no_dense = false;
  // Optional file tier (full CTE stack: hbm-resident cache + RAM + file).
  u64 nvme_mb = 0;
  std::string nvme_path = "/tmp/gv_gmx_tier.dat";
  u64 page_kb = 64, atoms = 200000;
  int repeat = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--cap") cap = static_cast<u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--atoms") atoms = next();
    else if (a == "--repeat") repeat = static_cast<int>(next());
    else if (a == "--no-dense") no_dense = true;
    else if (a == "--nvme-mb") nvme_mb = next();
    else if (a == "--nvme-path" && i + 1 < argc) nvme_path = argv[++i];
    else if (a == "--help") {
      std::printf("usage: %s [--blocks N] [--threads N] [--cap PAGES] "
                  "[--page-kb N] [--atoms N] [--repeat N]\n", argv[0]);
      return 0;
    }
  }

  // The coroutine-mode refusal moved with the kernels: a build that cannot
  // compile them does not produce this target at all now.
  // One page = one XY plane of u64 mesh points: K^2 * 8 bytes. ANY K whose
  // plane is exactly the page works, not just powers of two -- the MPI
  // edition takes arbitrary K, and matching its 6 GB-class meshes (e.g.
  // K=912 -> page-kb 6498) needs the same freedom here. The exactness
  // check stays: a page that is not exactly one plane breaks the
  // one-writer-per-plane decomposition the digit-exact gates rely on.
  const u64 page_bytes = page_kb * 1024;
  u64 K = static_cast<u64>(std::sqrt(
      static_cast<double>(page_bytes / sizeof(unsigned long long))));
  while (K * K * sizeof(unsigned long long) < page_bytes) ++K;
  if (K * K * sizeof(unsigned long long) != page_bytes) {
    std::fprintf(stderr, "GMX ERROR: --page-kb %llu is not a square u64 "
                 "plane (K^2*8 must equal the page exactly); e.g. 8, 32, "
                 "128, 512, 6498 (K=912)...\n",
                 (unsigned long long)page_kb);
    return 2;
  }
  const u64 plane = K * K;
  const u64 nmesh = plane * K;   // K^3
  const u64 zper = (K + blocks - 1) / blocks;
  // The gather holds 4 planes at once per block; refuse a cache that could
  // evict a plane mid-read (same rule as grayscott's slots >= window).
  if (cap != 0 && cap < 4 * blocks + 2) {
    std::fprintf(stderr, "GMX ERROR: --cap %u < %u (4 planes held per block "
                 "x %u blocks + slack). The gather window would evict pages "
                 "it is reading.\n", cap, 4 * blocks + 2, blocks);
    return 2;
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
    std::ofstream cfg("gv_gmx_bench.yaml");
    // The RAM tier must hold the WHOLE mesh: a fixed capacity under
    // K^3*8B makes writebacks fail (put_errors > 0) and the conservation
    // gate then reports the lost planes. Derived with a 1 GB margin.
    const u64 ram_mb =
        K * K * K * sizeof(unsigned long long) / (1024ull * 1024ull) +
        1024ull;
    cfg << "networking:\n  port: 9447\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n"
        << "  first_busy_wait: 10000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
        << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
        << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n"
        << "      - path: \"ram::gv_gmx_ram\"\n        bdev_type: \"ram\"\n"
        << "        capacity_limit: \"" << ram_mb << "MB\"\n"
        << "        score: 1.0\n";
    if (nvme_mb > 0) {
      cfg << "      - path: \"" << nvme_path << "\"\n"
          << "        bdev_type: \"file\"\n"
          << "        capacity_limit: \"" << nvme_mb << "MB\"\n"
          << "        score: 0.0\n"
          << "        persistence_level: \"temporary\"\n";
    }
    cfg << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_gmx_bench.yaml", 1);
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "GMX ERROR: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "GMX ERROR: cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);
  // Per-block device state the SYCL backend allocates once; no-op on CUDA.
  // WITHOUT THIS the first paged launch reads a null IpcManager and faults
  // inside BeginFetch -- memcheck: "Invalid __global__ read ... Access to 0x0".
  gx::InitBackend(blocks, gpu);

  std::printf("PME spread+gather over a GPU vector\n"
              "  mesh=%llu^3 (page = one %lluKB plane, %llu planes)\n"
              "  atoms=%llu  blocks=%u threads=%u  cache=%s\n",
              (unsigned long long)K, (unsigned long long)page_kb,
              (unsigned long long)K, (unsigned long long)atoms, blocks,
              threads,
              cap == 0 ? "resident" : (std::to_string(cap) + " pages").c_str());

  // ---- Atoms: deterministic cloud, z-binned CSR on the host. -------------
  std::vector<float> hx(atoms), hy(atoms), hz(atoms);
  std::vector<long long> hq(atoms);
  std::vector<u32> bin_count(K + 1, 0);
  {
    u64 s = 0x9E3779B97F4A7C15ull;
    for (u64 a = 0; a < atoms; ++a) {
      s = Lcg(s); hx[a] = Frac01(s) * static_cast<float>(K);
      s = Lcg(s); hy[a] = Frac01(s) * static_cast<float>(K);
      s = Lcg(s); hz[a] = Frac01(s) * static_cast<float>(K);
      // Alternating unit charges in Q40.24: the mesh total must come back
      // as EXACTLY zero for even atom counts.
      hq[a] = (a & 1) ? -(1ll << 24) : (1ll << 24);
    }
  }
  // Sort by base bin iz0 = floor(z) - 1 (mod K): CSR over bins.
  std::vector<u32> order(atoms);
  {
    std::vector<u32> bin(atoms);
    for (u64 a = 0; a < atoms; ++a) {
      const int iz0 = static_cast<int>(std::floor(hz[a])) - 1;
      bin[a] = static_cast<u32>((iz0 + static_cast<int>(K)) %
                                static_cast<int>(K));
      bin_count[bin[a] + 1]++;
    }
    for (u64 b = 0; b < K; ++b) bin_count[b + 1] += bin_count[b];
    std::vector<u32> cur(bin_count.begin(), bin_count.end() - 1);
    for (u64 a = 0; a < atoms; ++a) order[cur[bin[a]]++] = static_cast<u32>(a);
  }
  auto permute_f = [&](std::vector<float> &v) {
    std::vector<float> t(atoms);
    for (u64 a = 0; a < atoms; ++a) t[a] = v[order[a]];
    v.swap(t);
  };
  permute_f(hx); permute_f(hy); permute_f(hz);
  {
    std::vector<long long> t(atoms);
    for (u64 a = 0; a < atoms; ++a) t[a] = hq[order[a]];
    hq.swap(t);
  }
  long long q_total = 0;
  for (u64 a = 0; a < atoms; ++a) q_total += hq[a];

  auto *d_ax = ctp::GpuApi::Malloc<float>(atoms * sizeof(float));
  auto *d_ay = ctp::GpuApi::Malloc<float>(atoms * sizeof(float));
  auto *d_az = ctp::GpuApi::Malloc<float>(atoms * sizeof(float));
  auto *d_aq = ctp::GpuApi::Malloc<long long>(atoms * sizeof(long long));
  auto *d_bs = ctp::GpuApi::Malloc<u32>((K + 1) * sizeof(u32));
  ctp::GpuApi::Memcpy(d_ax, hx.data(), atoms * sizeof(float));
  ctp::GpuApi::Memcpy(d_ay, hy.data(), atoms * sizeof(float));
  ctp::GpuApi::Memcpy(d_az, hz.data(), atoms * sizeof(float));
  ctp::GpuApi::Memcpy(d_aq, hq.data(), atoms * sizeof(long long));
  ctp::GpuApi::Memcpy(d_bs, bin_count.data(), (K + 1) * sizeof(u32));

  // ---- Dense reference: same kernels, plain memory. ----------------------
  double t_ref_spread = 0.0;
  unsigned long long ref[4] = {0, 0, 0, 0};
  auto *d_out = ctp::GpuApi::Malloc<unsigned long long>(
      4 * sizeof(unsigned long long));
  ctp::GpuApi::Memset(d_out, 0, 4 * sizeof(unsigned long long));
  const u64 bper = (K + blocks - 1) / blocks;
  if (!no_dense) {
    auto *d_mesh = ctp::GpuApi::Malloc<unsigned long long>(
        nmesh * sizeof(unsigned long long));
    ctp::GpuApi::Memset(d_mesh, 0, nmesh * sizeof(unsigned long long));
    const double t_ref0 = NowMs();
    gx::LaunchDenseSpread(blocks, threads, d_mesh, d_ax, d_ay, d_az, d_aq,
                          d_bs, K, plane, zper);
    ctp::GpuApi::Synchronize();
    t_ref_spread = NowMs() - t_ref0;
    gx::LaunchDenseSum(d_mesh, nmesh, d_out);
    gx::LaunchDenseGather(blocks, threads, d_mesh, d_ax, d_ay, d_az, d_aq,
                          d_bs, K, plane, bper, &d_out[2]);
    ctp::GpuApi::Synchronize();
    ctp::GpuApi::Memcpy(ref, d_out, sizeof(ref));
    ctp::GpuApi::Free(d_mesh);
  }

  // ---- Paged path. -------------------------------------------------------
  const u32 tags = 24;
  gv::Vector<unsigned long long> mesh(
      "gv_gmx_mesh", {0}, page_bytes, blocks, tags, nmesh,
      clio::run::PoolId::GetNull(), 0, 1, 0,
      cap == 0 ? static_cast<u32>(K + 2) : cap);
  mesh.EnableStats();
  auto dmesh = mesh.GetDevice(0);
  YieldRunner runner(blocks, threads);

  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                 gy::YieldStackView sv) {
    gx::LaunchZero(g, b, gpu, dmesh, K, plane, zper, vw,
                                                sv);
  });
  ctp::GpuApi::Synchronize();

  double t_spread = 0.0, t_gather = 0.0;
  unsigned long long got[4] = {0, 0, 0, 0};
  for (int r = 0; r < repeat; ++r) {
    if (r != 0) {
      runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        gx::LaunchZero(g, b, gpu, dmesh, K, plane, zper,
                                                    vw, sv);
      });
      ctp::GpuApi::Synchronize();
    }
    const double t0 = NowMs();
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      gx::LaunchSpread(g, b, gpu, dmesh, d_ax, d_ay, d_az, d_aq, d_bs, K,
                       plane, zper, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    const double t1 = NowMs();
    t_spread += t1 - t0;
    ctp::GpuApi::Memset(d_out, 0, 4 * sizeof(unsigned long long));
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      gx::LaunchSum(g, b, gpu, dmesh, K, plane, zper, d_out, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      gx::LaunchGather(g, b, gpu, dmesh, d_ax, d_ay, d_az, d_aq, d_bs, K,
                       plane, bper, &d_out[2], vw, sv);
    });
    ctp::GpuApi::Synchronize();
    t_gather += NowMs() - t1;
    ctp::GpuApi::Memcpy(got, d_out, sizeof(got));
  }

  const auto st = mesh.ReadStats(0);
  std::printf("  paging: faults=%llu evicts=%llu puts=%llu get_errors=%llu "
              "put_errors=%llu\n",
              (unsigned long long)st.faults, (unsigned long long)st.evicts,
              (unsigned long long)st.puts, (unsigned long long)st.get_errors,
              (unsigned long long)st.put_errors);
  std::printf("  spread %.1f ms (dense %.1f)  gather+sum %.1f ms\n",
              t_spread / repeat, t_ref_spread, t_gather / repeat);

  // ---- The three exact gates. --------------------------------------------
  int rc = 0;
  const unsigned long long want_q =
      static_cast<unsigned long long>(q_total);
  if (got[0] != want_q) {
    std::printf("  CONSERVATION GATE: FAIL (mesh total %llu != input %llu)\n",
                got[0], want_q);
    rc = 1;
  } else {
    std::printf("  CONSERVATION GATE: PASS (mesh total == input charge, "
                "exact)\n");
  }
  if (no_dense) {
    std::printf("  MESH/GATHER GATES: SKIPPED (--no-dense; conservation "
                "only)\n");
  } else {
    if (got[0] != ref[0] || got[1] != ref[1]) {
      std::printf("  MESH GATE: FAIL (paged q=%llu ck=%llu vs dense q=%llu "
                  "ck=%llu)\n", got[0], got[1], ref[0], ref[1]);
      rc = 1;
    } else {
      std::printf("  MESH GATE: PASS (checksum bit-equal to dense "
                  "reference)\n");
    }
    if (got[2] != ref[2]) {
      std::printf("  GATHER GATE: FAIL (paged %llu vs dense %llu)\n", got[2],
                  ref[2]);
      rc = 1;
    } else {
      std::printf("  GATHER GATE: PASS (interpolation energy bit-equal)\n");
    }
  }
  std::printf("%s\n", rc == 0 ? "GMX BENCH: ALL GATES PASS"
                              : "GMX BENCH: GATE FAILURE");
  BenchFlushData();
  return rc;
}
#endif  // !CTP_IS_DEVICE_PASS
