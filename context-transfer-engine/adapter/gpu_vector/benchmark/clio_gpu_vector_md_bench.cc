/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * eternia-MD: a from-scratch reimplementation of the LAMMPS melt benchmark
 * (lj/cut, NVE, periodic box) with ALL simulation state device-canonical on
 * paged gv::Vectors. Design: core/eternia.md. LAMMPS itself is only the
 * correctness oracle.
 *
 * STAGE 1 (this file today): the layout and the integrator.
 *   - bin-major, padded-bin atom layout on two vectors, x (xyz + type in .w)
 *     and v (vxyz + spare), pages containing whole bins;
 *   - K1 (half-kick + drift), K4 (second half-kick), K5 (thermo reduction);
 *   - the BALLISTIC GATE: under a constant acceleration, velocity Verlet is
 *     algebraically exact, and per-atom updates are order-identical to a
 *     host float replica, so the gate demands BITWISE equality against the
 *     replica plus closed-form agreement in double. Any mismatch is a bug
 *     in the layout, the holds, or the kernels -- there is nowhere to hide.
 *   - the RESIDENT contract: stage-1 caches are sized to fit, and the run
 *     asserts zero faults / zero evictions in the timed kernels.
 *
 * Stages 2-5 (binning/resort, forces, the device-built Verlet list on a
 * paged neigh vector, out-of-core configs, stock-LAMMPS golden validation)
 * layer onto this file without changing the structures introduced here.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

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

/** Coroutine frames live in the lane; integrator coroutines are small. */
static constexpr u32 kYieldLaneBytes = 4096;
/** Elements per atom in x and v (float4 packing). */
static constexpr u32 kStride = 4;

#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
#define GV_MD_CORO 1
#endif

namespace {

/**
 * The keystone layout (eternia.md section 2): atoms live bin-major in
 * padded bins; an atom's index is bin * cap + slot; empty slots carry
 * type = -1 in x's .w lane. Pages contain WHOLE bins (cap * kStride divides
 * the page), so every kernel's working set is a set of whole pages known
 * from geometry. Stage 1 only needs the layout to exist and to survive the
 * integrator; binning DYNAMICS (resort, stencils) arrive in stage 2.
 */
struct Geometry {
  double box = 0.0;        // cubic box edge
  double bin_edge = 0.0;   // >= cutoff + skin
  u32 nb = 0;              // bins per dimension
  u64 nbins = 0;           // nb^3
  u32 cap = 0;             // atom slots per bin (padded, fixed)
  u64 nslots = 0;          // nbins * cap == total atom slots
  u64 nelems = 0;          // nslots * kStride == vector elements
  u64 natoms = 0;          // real atoms
};

struct Args {
  u32 lattice = 20;        // FCC unit cells per dimension -> 4*n^3 atoms
  double rho = 0.8442;     // reduced density (the melt deck's)
  double cutoff = 2.5;
  double skin = 0.3;
  u32 cap = 32;
  u64 page_kb = 64;
  u32 blocks = 64;
  u32 threads = 256;
  u32 slots = 0;           // x/v cache slots per block; 0 = resident (auto)
  u64 steps = 100;
  double dt = 0.005;
  int gate = 1;            // stage 1: the ballistic gate IS the run
  double g[3] = {0.1, -0.05, 0.02};   // constant acceleration for the gate
};

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

/** Deterministic per-atom initial velocity: any fixed function works for the
 *  gate (the replica and the closed form both use it); keep it simple and
 *  well-spread. */
void InitVelocity(u64 atom_id, float out[3]) {
  const double a = static_cast<double>(atom_id);
  out[0] = static_cast<float>(0.20 * std::sin(0.37 * a + 0.1));
  out[1] = static_cast<float>(0.20 * std::cos(0.53 * a + 0.7));
  out[2] = static_cast<float>(0.20 * std::sin(0.71 * a + 1.3));
}

/**
 * Build the bin-major initial state on the host: FCC lattice positions
 * binned into the padded layout. Refuses loudly if any bin overflows cap --
 * a silent overflow would corrupt a neighbour bin, which is the exact class
 * of quiet failure this project is done tolerating.
 * @return false on overflow.
 */
bool BuildInitialState(const Args &a, const Geometry &g,
                       std::vector<float> &hx, std::vector<float> &hv,
                       std::vector<u32> &bin_count) {
  hx.assign(g.nelems, 0.0f);
  hv.assign(g.nelems, 0.0f);
  // Sentinel every slot first.
  for (u64 s = 0; s < g.nslots; ++s) hx[s * kStride + 3] = -1.0f;
  bin_count.assign(g.nbins, 0u);

  const double cell = g.box / a.lattice;   // FCC unit-cell edge
  static const double kBasis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  u64 atom_id = 0;
  for (u32 i = 0; i < a.lattice; ++i) {
    for (u32 j = 0; j < a.lattice; ++j) {
      for (u32 k = 0; k < a.lattice; ++k) {
        for (int b = 0; b < 4; ++b) {
          const double px = (i + kBasis[b][0]) * cell;
          const double py = (j + kBasis[b][1]) * cell;
          const double pz = (k + kBasis[b][2]) * cell;
          u32 bx = static_cast<u32>(px / g.bin_edge);
          u32 by = static_cast<u32>(py / g.bin_edge);
          u32 bz = static_cast<u32>(pz / g.bin_edge);
          if (bx >= g.nb) bx = g.nb - 1;   // boundary atom, edge bin
          if (by >= g.nb) by = g.nb - 1;
          if (bz >= g.nb) bz = g.nb - 1;
          const u64 bin = (static_cast<u64>(bz) * g.nb + by) * g.nb + bx;
          const u32 slot = bin_count[bin]++;
          if (slot >= g.cap) {
            std::fprintf(stderr,
                         "bin %llu overflows cap=%u -- raise --cap\n",
                         (unsigned long long)bin, g.cap);
            return false;
          }
          const u64 e = (bin * g.cap + slot) * kStride;
          hx[e + 0] = static_cast<float>(px);
          hx[e + 1] = static_cast<float>(py);
          hx[e + 2] = static_cast<float>(pz);
          hx[e + 3] = 1.0f;   // type 1
          float vv[3];
          InitVelocity(atom_id, vv);
          hv[e + 0] = vv[0];
          hv[e + 1] = vv[1];
          hv[e + 2] = vv[2];
          hv[e + 3] = 0.0f;
          ++atom_id;
        }
      }
    }
  }
  return atom_id == g.natoms;
}

}  // namespace

#if defined(GV_MD_CORO)

/**
 * K1: first velocity-Verlet half -- v += (dt/2)*accel, x += dt*v.
 * A block owns pages block, block+nblocks, ...; per page it write-holds the
 * SAME page of x and v and streams the slots. Sentinel slots are skipped.
 * No periodic wrap here BY DESIGN: positions wrap at the resort (stage 2),
 * exactly as LAMMPS wraps at reneighbour -- and it keeps the ballistic gate
 * algebraically exact.
 *
 * Stage 2 note: `accel` becomes a read-hold on the f vector; the hold
 * structure of this kernel does not change.
 */
/** Flush every dirty page of the SHARED x/v tables to the backing store, so
 *  host Download (which reads the store, not the frames) sees the truth.
 *  One block does it; the others exit immediately. Validation-path only --
 *  steady-state MD never flushes x/v (they are device-canonical). */
__device__ gy::YCoroMain FlushAllCoro(gv::DeviceVector<float> x,
                                      gv::DeviceVector<float> v, u32 block) {
  if (block == 0) {
    x.FlushAsync(0, x.h_->size_);
    co_await x.AwaitFlush();
    v.FlushAsync(0, v.h_->size_);
    co_await v.AwaitFlush();
  }
}

__device__ gy::YCoroMain IntegrateCoro(gv::DeviceVector<float> x,
                                       gv::DeviceVector<float> v,
                                       float dt, float gx, float gy_,
                                       float gz, int drift, u32 nblocks,
                                       u32 block) {
  const u64 epp = x.h_->elems_per_page_;
  const u64 npages = (x.h_->size_ + epp - 1) / epp;
  const float half = 0.5f * dt;
  for (u64 pg = block; pg < npages; pg += nblocks) {
    auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
    auto hv = co_await v.HoldPage(pg * epp, epp, /*write=*/true);
    float *const px = hx.ptr();
    float *const pv = hv.ptr();
    const u64 nslots = epp / kStride;
    for (u64 s = threadIdx.x; s < nslots; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) continue;   // padded slot
      // EXPLICIT fma on every update, mirrored by std::fmaf in the host
      // replica: the bitwise gate must not depend on which contractions a
      // compiler happens to choose.
      const float vx0 = __fmaf_rn(half, gx, pv[e + 0]);
      const float vy0 = __fmaf_rn(half, gy_, pv[e + 1]);
      const float vz0 = __fmaf_rn(half, gz, pv[e + 2]);
      pv[e + 0] = vx0;
      pv[e + 1] = vy0;
      pv[e + 2] = vz0;
      if (drift) {
        px[e + 0] = __fmaf_rn(dt, vx0, px[e + 0]);
        px[e + 1] = __fmaf_rn(dt, vy0, px[e + 1]);
        px[e + 2] = __fmaf_rn(dt, vz0, px[e + 2]);
      }
    }
    __syncthreads();
    // Guards die here: pages unpin. Nothing flushes -- x/v are device-
    // canonical; the backing store is only written on eviction (none in the
    // resident regime) or by an explicit host Download for validation.
  }
}

/** K5: thermo -- KE and net momentum, block-reduced then atomically merged.
 *  Read-holds only. Reduction scratch is dynamic shared memory after the
 *  yield header. */
__device__ gy::YCoroMain ThermoCoro(gv::DeviceVector<float> x,
                                    gv::DeviceVector<float> v,
                                    double *out, u32 nblocks, u32 block) {
  extern __shared__ char smem_raw[];
  double *red = reinterpret_cast<double *>(smem_raw + CLIO_YIELD_SMEM_BYTES);
  const u64 epp = x.h_->elems_per_page_;
  const u64 npages = (x.h_->size_ + epp - 1) / epp;
  double ke = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
  for (u64 pg = block; pg < npages; pg += nblocks) {
    auto hx = co_await x.HoldPage(pg * epp, epp);
    auto hv = co_await v.HoldPage(pg * epp, epp);
    const float *const px = hx.ptr();
    const float *const pv = hv.ptr();
    const u64 nslots = epp / kStride;
    for (u64 s = threadIdx.x; s < nslots; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) continue;
      const double vx = pv[e + 0], vy = pv[e + 1], vz = pv[e + 2];
      ke += 0.5 * (vx * vx + vy * vy + vz * vz);   // m = 1
      mx += vx; my += vy; mz += vz;
    }
    __syncthreads();
  }
  // Tree-reduce the four sums through shared, one at a time.
  const double vals[4] = {ke, mx, my, mz};
  for (int q = 0; q < 4; ++q) {
    red[threadIdx.x] = vals[q];
    __syncthreads();
    for (u32 w = blockDim.x / 2; w > 0; w >>= 1) {
      if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w];
      __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&out[q], red[0]);
    __syncthreads();
  }
}

/* x and v are created with ONE shared table (vector nblocks = 1): a single
 * copy of every page, held by all GPU blocks. block_override_ = 0 points
 * every block at that table; blocks write DISJOINT pages, so the only
 * sharing is the lock-free probes. */

__global__ void IntegrateKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<float> v, float dt,
                                float gx, float gy_, float gz, int drift,
                                u32 nblocks, gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.block_override_ = 0;
  v.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(IntegrateCoro(x, v, dt, gx, gy_, gz, drift, nblocks,
                               yv.Block()));
}

__global__ void ThermoKernel(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<float> x,
                             gv::DeviceVector<float> v, double *out,
                             u32 nblocks, gy::YieldableView<> yv,
                             gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.block_override_ = 0;
  v.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ThermoCoro(x, v, out, nblocks, yv.Block()));
}

__global__ void FlushAllKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<float> x,
                               gv::DeviceVector<float> v,
                               gy::YieldableView<> yv,
                               gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.block_override_ = 0;
  v.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(FlushAllCoro(x, v, yv.Block()));
}

#if !CTP_IS_DEVICE_PASS
namespace {

class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    drv_.ResetTimers();
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [] {}, /*max_rounds=*/2000000,
        [](u32, u64 tag) -> bool {
          unsigned int f = 0;
          ctp::GpuApi::Memcpy(&f, reinterpret_cast<const unsigned int *>(tag),
                              sizeof(f));
          return (f & 1u) != 0u;
        });
  }
  double KernelMs() const { return drv_.KernelMs(); }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

}  // namespace
#endif  // !CTP_IS_DEVICE_PASS
#endif  // GV_MD_CORO

#if !CTP_IS_DEVICE_PASS
int main(int argc, char **argv) {
#if !defined(GV_MD_CORO)
  (void)argc; (void)argv;
  std::fprintf(stderr,
               "eternia-MD requires the clang device-coroutine build "
               "(CLIO_GPU_YIELD_CORO=ON)\n");
  return 2;
#else
  Args a;
  for (int i = 1; i < argc; ++i) {
    auto want = [&](const char *k) {
      return std::strcmp(argv[i], k) == 0 && i + 1 < argc;
    };
    if (want("--lattice")) a.lattice = static_cast<u32>(atoi(argv[++i]));
    else if (want("--rho")) a.rho = atof(argv[++i]);
    else if (want("--cutoff")) a.cutoff = atof(argv[++i]);
    else if (want("--skin")) a.skin = atof(argv[++i]);
    else if (want("--cap")) a.cap = static_cast<u32>(atoi(argv[++i]));
    else if (want("--page-kb")) a.page_kb = static_cast<u64>(atol(argv[++i]));
    else if (want("--blocks")) a.blocks = static_cast<u32>(atoi(argv[++i]));
    else if (want("--threads")) a.threads = static_cast<u32>(atoi(argv[++i]));
    else if (want("--slots")) a.slots = static_cast<u32>(atoi(argv[++i]));
    else if (want("--steps")) a.steps = static_cast<u64>(atol(argv[++i]));
    else if (want("--dt")) a.dt = atof(argv[++i]);
    else if (std::strcmp(argv[i], "--no-gate") == 0) a.gate = 0;
    else {
      std::fprintf(stderr, "unknown arg %s\n", argv[i]);
      return 1;
    }
  }

  // ---- geometry -----------------------------------------------------------
  Geometry g;
  g.natoms = 4ull * a.lattice * a.lattice * a.lattice;
  g.box = a.lattice * std::cbrt(4.0 / a.rho);
  const double min_edge = a.cutoff + a.skin;
  g.nb = static_cast<u32>(g.box / min_edge);
  if (g.nb == 0) g.nb = 1;
  g.bin_edge = g.box / g.nb;
  g.nbins = static_cast<u64>(g.nb) * g.nb * g.nb;
  g.cap = a.cap;
  g.nslots = g.nbins * g.cap;
  g.nelems = g.nslots * kStride;

  const u64 page_bytes = a.page_kb * 1024;
  const u64 page_elems = page_bytes / sizeof(float);
  const u64 bin_elems = static_cast<u64>(g.cap) * kStride;
  if (page_elems % bin_elems != 0) {
    std::fprintf(stderr,
                 "page (%llu elems) must hold WHOLE bins (%llu elems/bin): "
                 "adjust --page-kb or --cap\n",
                 (unsigned long long)page_elems,
                 (unsigned long long)bin_elems);
    return 1;
  }
  const u64 npages = (g.nelems + page_elems - 1) / page_elems;
  // THE VECTOR OWNS WHOLE PAGES. A partial tail page would leave slots the
  // kernels can reach but the host never initialized: Preload zero-pads
  // them, zero is not the sentinel, and 2464 phantom atoms integrated the
  // constant field -- caught by the ballistic gate as a momentum excess of
  // exactly 2464*dt*g. Rounding the size up puts every reachable slot under
  // the layout's contract: a real atom or an explicit sentinel.
  g.nelems = npages * page_elems;
  g.nslots = g.nelems / kStride;
  // x and v use ONE SHARED table (vector nblocks = 1): one copy of every
  // page, held by all GPU blocks, which write disjoint pages. Stage 1 is
  // the resident regime: the default cache holds everything. An explicit
  // --slots below that is a configuration for later stages, not this gate.
  const u32 slots =
      (a.slots != 0) ? a.slots : static_cast<u32>(npages + 2);

  std::printf(
      "eternia-MD stage 1 (integrator + ballistic gate)\n"
      "  atoms=%llu box=%.4f bins=%u^3 cap=%u slots/bin-pad=%.0f%%\n"
      "  page=%lluKB (%llu bins/page) pages=%llu blocks=%u threads=%u "
      "cache=%u slots (one shared table)\n"
      "  steps=%llu dt=%g g=(%g,%g,%g)\n",
      (unsigned long long)g.natoms, g.box, g.nb, g.cap,
      100.0 * (1.0 - static_cast<double>(g.natoms) / g.nslots),
      (unsigned long long)a.page_kb, (unsigned long long)(page_elems / bin_elems),
      (unsigned long long)npages, a.blocks, a.threads, slots,
      (unsigned long long)a.steps, a.dt, a.g[0], a.g[1], a.g[2]);

  // ---- host initial state -------------------------------------------------
  std::vector<float> hx, hv;
  std::vector<u32> bin_count;
  if (!BuildInitialState(a, g, hx, hv, bin_count)) return 1;

  // ---- runtime ------------------------------------------------------------
  {
    std::ofstream cfg("gpu_vector_md.yaml");
    const char *thr = getenv("GV_THREADS");
    auto env_mb = [](const char *k, long d) {
      const char *e = getenv(k);
      return e ? std::atol(e) : d;
    };
    const long hbm_mb = env_mb("GV_HBM_MB", 0);
    const long dram_mb = env_mb("GV_DRAM_MB", 4096);
    cfg << "networking:\n  port: 9438\n\n"
        << "runtime:\n  num_threads: " << (thr ? atoi(thr) : 8)
        << "\n  queue_depth: 8192\n  first_busy_wait: 10000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
        << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
        << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n";
    if (hbm_mb > 0) {
      cfg << "      - path: \"hbm::gv_md_hbm\"\n        bdev_type: \"hbm\"\n"
          << "        capacity_limit: \"" << hbm_mb << "MB\"\n"
          << "        score: 1.0\n";
    }
    cfg << "      - path: \"ram::gv_md_ram\"\n        bdev_type: \"ram\"\n"
        << "        capacity_limit: \"" << dram_mb << "MB\"\n"
        << "        score: " << (hbm_mb > 0 ? "0.6" : "1.0") << "\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_md.yaml", 1);
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

  // ---- vectors ------------------------------------------------------------
  gv::Vector<float> vx("md_x", {0}, page_bytes, /*nblocks=*/1, slots,
                       g.nelems);
  gv::Vector<float> vv("md_v", {0}, page_bytes, /*nblocks=*/1, slots,
                       g.nelems);
  vx.EnableStats();
  vv.EnableStats();
  vx.Preload(hx.data(), g.nelems);
  vv.Preload(hv.data(), g.nelems);
  vx.ClearCache();
  vv.ClearCache();
  vx.Prefetch(0, npages, 0, 1);
  vv.Prefetch(0, npages, 0, 1);
  auto dx = vx.GetDevice(0);
  auto dv = vv.GetDevice(0);

  double *d_thermo = ctp::GpuApi::Malloc<double>(4 * sizeof(double));

  YieldRunner runner(a.blocks, a.threads);
  const u32 smem_thermo =
      CLIO_YIELD_SMEM_BYTES + a.threads * sizeof(double);

  // ---- the run ------------------------------------------------------------
  const float fdt = static_cast<float>(a.dt);
  const float fgx = static_cast<float>(a.g[0]);
  const float fgy = static_cast<float>(a.g[1]);
  const float fgz = static_cast<float>(a.g[2]);
  const double t0 = NowMs();
  for (u64 step = 0; step < a.steps; ++step) {
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      IntegrateKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu, dx, dv, fdt, fgx, fgy, fgz, /*drift=*/1, a.blocks, vw, sv);
    });
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      IntegrateKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu, dx, dv, fdt, fgx, fgy, fgz, /*drift=*/0, a.blocks, vw, sv);
    });
  }
  ctp::GpuApi::Synchronize();
  const double run_ms = NowMs() - t0;

  double thermo[4] = {0, 0, 0, 0};
  double thermo2[4] = {0, 0, 0, 0};
  for (int rep = 0; rep < 2; ++rep) {
    ctp::GpuApi::Memset(d_thermo, 0, 4 * sizeof(double));
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      ThermoKernel<<<gr, b, smem_thermo>>>(gpu, dx, dv, d_thermo, a.blocks,
                                           vw, sv);
    });
    ctp::GpuApi::Synchronize();
    ctp::GpuApi::Memcpy(rep == 0 ? thermo : thermo2, d_thermo,
                        4 * sizeof(double));
  }
  std::printf("  thermo rep0: KE=%.9g p=(%.9g,%.9g,%.9g)\n"
              "  thermo rep1: KE=%.9g p=(%.9g,%.9g,%.9g)  %s\n",
              thermo[0], thermo[1], thermo[2], thermo[3], thermo2[0],
              thermo2[1], thermo2[2], thermo2[3],
              (thermo[0] == thermo2[0] && thermo[1] == thermo2[1] &&
               thermo[2] == thermo2[2] && thermo[3] == thermo2[3])
                  ? "[deterministic]" : "[VARIES: race]");

  // ---- resident-regime assertion -----------------------------------------
  const auto sx = vx.ReadStats(0);
  const auto sv2 = vv.ReadStats(0);
  const bool resident_ok = (sx.faults == 0 && sx.evicts == 0 &&
                            sv2.faults == 0 && sv2.evicts == 0);
  std::printf("  paging: x faults=%llu evicts=%llu | v faults=%llu "
              "evicts=%llu  %s\n",
              (unsigned long long)sx.faults, (unsigned long long)sx.evicts,
              (unsigned long long)sv2.faults, (unsigned long long)sv2.evicts,
              resident_ok ? "[resident contract HELD]"
                          : "[RESIDENT CONTRACT VIOLATED]");

  // ---- the ballistic gate -------------------------------------------------
  // (a) BITWISE: replay the identical float recurrence on the host. The
  //     device updates each atom independently with the same operation
  //     order, so any bit of divergence is a real defect.
  // (b) CLOSED FORM: velocity Verlet under constant g is exact;
  //     x_n = x0 + n dt v0 + n^2 dt^2/2 g in double bounds float drift.
  int rc = 0;
  if (a.gate) {
    // Download reads the BACKING STORE; the kernels wrote the cache frames.
    // Flush first (validation-path only -- steady-state MD never flushes
    // x/v; they are device-canonical).
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      FlushAllKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dx, dv, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    std::vector<float> gx_out(g.nelems), gv_out(g.nelems);
    vx.Download(gx_out.data(), g.nelems);
    vv.Download(gv_out.data(), g.nelems);
    u64 bit_x = 0, bit_v = 0;
    double max_cf = 0.0;
    double ke_ref = 0.0, mom_ref[3] = {0, 0, 0};
    const double n = static_cast<double>(a.steps);
    for (u64 s = 0; s < g.nslots; ++s) {
      const u64 e = s * kStride;
      if (hx[e + 3] < 0.0f) continue;
      // (a) float replica, same op order as the kernel
      float rx[3] = {hx[e], hx[e + 1], hx[e + 2]};
      float rv[3] = {hv[e], hv[e + 1], hv[e + 2]};
      const float gg[3] = {fgx, fgy, fgz};
      const float fhalf = 0.5f * fdt;
      for (u64 st = 0; st < a.steps; ++st) {
        for (int d = 0; d < 3; ++d) {
          rv[d] = std::fmaf(fhalf, gg[d], rv[d]);
          rx[d] = std::fmaf(fdt, rv[d], rx[d]);
        }
        for (int d = 0; d < 3; ++d) rv[d] = std::fmaf(fhalf, gg[d], rv[d]);
      }
      for (int d = 0; d < 3; ++d) {
        if (rx[d] != gx_out[e + d]) {
          if (bit_x < 4) {
            std::printf("    x mismatch slot=%llu d=%d page=%llu got=%.9g "
                        "want=%.9g diff=%.3e\n",
                        (unsigned long long)s, d,
                        (unsigned long long)(e / (page_bytes / sizeof(float))),
                        gx_out[e + d], rx[d],
                        std::fabs(static_cast<double>(gx_out[e + d]) - rx[d]));
          }
          ++bit_x;
        }
        if (rv[d] != gv_out[e + d]) {
          if (bit_v < 4) {
            std::printf("    v mismatch slot=%llu d=%d got=%.9g want=%.9g "
                        "diff=%.3e\n",
                        (unsigned long long)s, d, gv_out[e + d], rv[d],
                        std::fabs(static_cast<double>(gv_out[e + d]) - rv[d]));
          }
          ++bit_v;
        }
        // (b) closed form in double
        const double cf = static_cast<double>(hx[e + d]) +
                          n * a.dt * static_cast<double>(hv[e + d]) +
                          0.5 * n * n * a.dt * a.dt * a.g[d];
        const double err = std::fabs(cf - gx_out[e + d]);
        if (err > max_cf) max_cf = err;
        const double vd = gv_out[e + d];
        mom_ref[d] += vd;
        ke_ref += 0.5 * vd * vd;
      }
    }
    const double ke_err = std::fabs(thermo[0] - ke_ref) /
                          (ke_ref != 0.0 ? ke_ref : 1.0);
    double mom_err = 0.0;
    for (int d = 0; d < 3; ++d) {
      mom_err = std::max(mom_err, std::fabs(thermo[1 + d] - mom_ref[d]));
    }
    std::printf("    mom dev=(%.9g,%.9g,%.9g) ref=(%.9g,%.9g,%.9g)\n",
                thermo[1], thermo[2], thermo[3], mom_ref[0], mom_ref[1],
                mom_ref[2]);
    std::printf("  gate: bitwise mismatches x=%llu v=%llu | closed-form max "
                "|dx|=%.3e | thermo KE dev=%.6f host=%.6f rel_err=%.2e "
                "mom_err=%.2e\n",
                (unsigned long long)bit_x, (unsigned long long)bit_v, max_cf,
                thermo[0], ke_ref, ke_err, mom_err);
    const bool gate_ok = (bit_x == 0 && bit_v == 0) && (max_cf < 1e-2) &&
                         (ke_err < 1e-9) && (mom_err < 1e-6) && resident_ok;
    std::printf("  BALLISTIC GATE: %s\n", gate_ok ? "PASS" : "FAIL");
    if (!gate_ok) rc = 1;
  }

  std::printf("  %llu steps in %.1f ms (%.3f ms/step, %.1f Matom-steps/s)\n",
              (unsigned long long)a.steps, run_ms, run_ms / a.steps,
              static_cast<double>(g.natoms) * a.steps / run_ms / 1000.0);
  ctp::GpuApi::Free(d_thermo);
  return rc;
#endif  // GV_MD_CORO
}
#endif  // !CTP_IS_DEVICE_PASS
