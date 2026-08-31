/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * LBANN science kernel -- MLP training with the WEIGHTS out of core -- over
 * a GPU vector.
 *
 * WHY THIS SHAPE. LBANN's paged-vector story was always the weights: the
 * model outgrows VRAM while each training step still touches all of it
 * (forward reads W, backward reads W again and then updates it in place).
 * That is the "re-read everything every pass, then WRITE it" pattern -- the
 * weights benchmark covers the read half; this adds the update half, which
 * is what makes eviction dangerous: an unpublished SGD update undone by a
 * refault is exactly the frozen-physics failure the md workload had, in
 * model-training clothes (loss keeps falling, weights quietly revert).
 *
 * STRUCTURE. Two layers, x -> ReLU(W1 x + b1) -> W2 a1 + b2, MSE loss, SGD.
 * One paged vector holds [W1 | b1 | W2 | b2]; activations, gradients and
 * the batch are small and stay resident. Each step runs five phases, each a
 * kernel launch (the inter-layer barrier), all reading or writing the paged
 * weights through Fetch/HoldPage and publishing every update at the write
 * site:
 *
 *   fwd1  reads W1 rows      (block owns h-rows; pages slide)
 *   fwd2  reads W2 rows      (block owns o-rows)
 *   bwd1  reads W2 rows      (PRE-update, classic backprop ordering)
 *   upd2  W2 -= lr dW2       (block owns o-rows: ONE WRITER PER PAGE)
 *   upd1  W1 -= lr dW1       (block owns h-rows: ONE WRITER PER PAGE)
 *
 * DETERMINISM BY CONSTRUCTION. Every output element is computed by exactly
 * one thread with a fixed-order sequential sum -- no atomics anywhere on
 * the training path -- so the paged run must match a dense in-VRAM
 * reference BIT FOR BIT: the per-step losses are compared as doubles for
 * equality, and the final weights as an order-independent integer sum of
 * their float bit patterns. Any stale page, lost update or wrong-order
 * flush is a hard gate failure, not a slightly-different loss curve.
 *
 * OUT OF CORE: --cap M caps the weight cache at M pages. fwd/bwd sweep the
 * whole model every step, so cap < model pages means every step refaults
 * and re-publishes -- Fetch before every hold, Flush before every unpin,
 * the contract this benchmark exists to certify under pressure.
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
 * The workload -- the weight initialiser, the seven training-phase
 * coroutines and the dense reference kernels -- lives in lbann_kernels.h
 * (over lbann_math.h), in ONE copy compiled by both backends. The launches
 * live in cuda/ and sycl/ and differ only in how a grid is submitted.
 *
 * See lbann_launch.h for why the seam is at the launch.
 */
#include "lbann_kernels.h"
#include "lbann_launch.h"

namespace lb = clio::gv_bench::lbann;
using lb::Lcg;
using lb::Sym01;


#if !CTP_IS_DEVICE_PASS

// Cross-node collectives. Included INSIDE the device-pass guard: it uses
// the CTE client, whose members are compiled out of the CUDA device pass.
#include "../bench_dist.h"

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
  u32 blocks = 8, threads = 256, cap = 0;
  // MODEL-PARALLEL, like the MPI edition: a node owns a contiguous band of
  // h-rows of W1 and o-rows of W2, and updates only those in place. What
  // crosses the wire is the activations and deltas, not the weights.
  u32 nodes = 1, node = 0;
  u64 page_kb = 64, I = 256, H = 4096, O = 64, B = 64, steps = 5;
  float lr = 0.01f;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--nodes") nodes = static_cast<u32>(next());
    else if (a == "--node") node = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--cap") cap = static_cast<u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--in") I = next();
    else if (a == "--hidden") H = next();
    else if (a == "--out") O = next();
    else if (a == "--batch") B = next();
    else if (a == "--steps") steps = next();
    else if (a == "--lr" && i + 1 < argc) lr = std::strtof(argv[++i], nullptr);
    else if (a == "--help") {
      std::printf("usage: %s [--blocks N] [--threads N] [--cap PAGES] "
                  "[--page-kb N] [--in N] [--hidden N] [--out N] [--batch N] "
                  "[--steps N] [--lr F]\n", argv[0]);
      return 0;
    }
  }

  // The coroutine-mode refusal moved with the kernels: a build that cannot
  // compile them does not produce this target at all now.
  const u64 page_bytes = page_kb * 1024;
  const u64 elems_per_page = page_bytes / sizeof(float);
  // Rows must tile pages exactly so an h-range is a whole-page range and
  // the one-writer-per-page rule holds by construction.
  if (elems_per_page % I != 0 || elems_per_page % H != 0) {
    std::fprintf(stderr, "LBANN ERROR: a %lluKB page must hold whole rows of "
                 "W1 (I=%llu) and W2 (H=%llu).\n",
                 (unsigned long long)page_kb, (unsigned long long)I,
                 (unsigned long long)H);
    return 2;
  }
  const u64 w1_n = H * I, w2_n = O * H;
  const u64 w1_off = 0;
  const u64 b1_off = w1_n;                       // b1 lives page-aligned after W1
  const u64 b1_pad = elems_per_page;             // one page for b1
  const u64 w2_off = w1_n + b1_pad;
  const u64 b2_off = w2_off + w2_n;
  const u64 b2_pad = elems_per_page;
  const u64 n = w2_off + w2_n + b2_pad;
  const u64 npages = (n + elems_per_page - 1) / elems_per_page;
  const u64 rpp1 = elems_per_page / I;           // W1 rows per page
  const u64 rpp2 = elems_per_page / H;           // W2 rows per page
  if (H % blocks != 0 || O % blocks != 0 ||
      (H / blocks) % rpp1 != 0 || (O / blocks) % (rpp2 ? rpp2 : 1) != 0) {
    std::fprintf(stderr, "LBANN ERROR: blocks must evenly split H and O into "
                 "whole pages (H=%llu O=%llu rpp1=%llu rpp2=%llu blocks=%u).\n",
                 (unsigned long long)H, (unsigned long long)O,
                 (unsigned long long)rpp1, (unsigned long long)rpp2, blocks);
    return 2;
  }
  if (node >= nodes) {
    std::fprintf(stderr, "LBANN ERROR: --node %u out of range for "
                 "--nodes %u\n", node, nodes);
    return 2;
  }
  // The node's band. H and O must split evenly across nodes AND blocks, or
  // a row belongs to nobody and the layer is silently smaller than it says.
  if (nodes > 1 && (H % nodes != 0 || O % nodes != 0)) {
    std::fprintf(stderr, "LBANN ERROR: H=%llu and O=%llu must divide "
                 "--nodes %u\n", (unsigned long long)H,
                 (unsigned long long)O, nodes);
    return 2;
  }
  // MODEL-PARALLEL BAND SPLIT. A node owns h-rows of W1 and o-rows of W2.
  // a1 and d2 are laid out [feature][batch] (a1[h*B+b], d2[o*B+b]), so a
  // band is a CONTIGUOUS slice and the exchange is a slice gather.
  //
  //   a1  all-gathered after Fwd1 -- Fwd2 sums over every h, Upd2 reads
  //       every h.
  //   d2  all-gathered after Fwd2 -- Bwd1 sums over every o.
  //   d1  needs nothing: Bwd1 computes this node's own h-rows and Upd1
  //       consumes exactly those.
  //
  // Bwd1 also reads ALL of W2, including o-rows a PEER updates, which is
  // a cross-node read of the shared paged vector. That is the same shape
  // as grayscott's halo and is handled the same way -- see the note there
  // on demanding a generation only of a peer's page.
  // THE RESIDUAL DISTRIBUTED ERROR IS THE BIASES, AND IT IS PAGE-GRANULAR
  // FALSE SHARING. W1 and W2 split cleanly: a node's rows are contiguous
  // and, for the default geometry, land on page boundaries (W1 row = I =
  // 256 elems, 64 rows/page, H/2 = 2048 rows -- aligned; W2 row = H, 4
  // rows/page, O/2 = 32 -- aligned). The BIASES do not: b1 is 4096 floats
  // and b2 is 64, so each fits ENTIRELY INSIDE ONE PAGE. Node 0 writes the
  // lower half of that page and node 1 the upper, and writeback is
  // page-granular -- so each node writes the whole page and clobbers the
  // other's half. Nothing reports an error; the weights simply end up
  // wrong by roughly one accumulated bias update, which is the measured
  // max |paged - dense| = 4.7e-4.
  //
  // This is the same hazard as splitting any paged work off a page
  // boundary, and the fix is not another collective: either the biases are
  // padded so each node's slice owns whole pages, or their update is
  // replicated so every node writes identical bytes and the clobber is
  // harmless (which needs d1 gathered, as a1 and d2 already are).
  const u64 h0 = (H / nodes) * node, h1 = h0 + H / nodes;
  const u64 o0 = (O / nodes) * node, o1 = o0 + O / nodes;
  // Blocks subdivide this node's band, not the whole layer.
  const u64 hper = (h1 - h0) / blocks, oper = (o1 - o0) / blocks;
  // THE DENSE REFERENCE IS NOT SHARDED. It is a plain in-VRAM copy of the
  // whole network and takes no row base, so a node-local hper makes every
  // node compute rows [0, H/nodes) -- the same wrong reference on each,
  // which is why both nodes agreed on a dense loss while disagreeing on
  // the paged one. It must span the full layer.
  const u64 hper_all = H / blocks, oper_all = O / blocks;

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
    std::ofstream cfg("gv_lbann_bench.yaml");
    cfg << "networking:\n  port: 9449\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n"
        << "  first_busy_wait: 10000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
        << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
        << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n"
        << "      - path: \"ram::gv_lb_ram\"\n        bdev_type: \"ram\"\n"
        << "        capacity_limit: \"4GB\"\n        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_lbann_bench.yaml", 1);
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "LBANN ERROR: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "LBANN ERROR: cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);
  // Per-block device state the SYCL backend allocates once; no-op on CUDA.
  lb::InitBackend(blocks, gpu);

  std::printf("MLP training with paged weights\n"
              "  layers %llu -> %llu -> %llu  batch=%llu  steps=%llu lr=%.3f\n"
              "  weights %.1f MB in %llu pages of %lluKB  cache=%s\n"
              "  blocks=%u threads=%u\n",
              (unsigned long long)I, (unsigned long long)H,
              (unsigned long long)O, (unsigned long long)B,
              (unsigned long long)steps, lr,
              static_cast<double>(n * sizeof(float)) / 1048576.0,
              (unsigned long long)npages, (unsigned long long)page_kb,
              cap == 0 ? "resident" : (std::to_string(cap) + " pages").c_str(),
              blocks, threads);

  // ---- Batch and targets: deterministic, resident. -----------------------
  std::vector<float> hxv(B * I), hyv(B * O);
  for (u64 i = 0; i < B * I; ++i) hxv[i] = Sym01(Lcg(0xA02BDBF7BB3C0A7ull + i));
  for (u64 i = 0; i < B * O; ++i) hyv[i] = Sym01(Lcg(0x6C62272E07BB0142ull + i));
  auto *d_x = ctp::GpuApi::Malloc<float>(B * I * sizeof(float));
  auto *d_y = ctp::GpuApi::Malloc<float>(B * O * sizeof(float));
  ctp::GpuApi::Memcpy(d_x, hxv.data(), B * I * sizeof(float));
  ctp::GpuApi::Memcpy(d_y, hyv.data(), B * O * sizeof(float));
  auto *d_a1 = ctp::GpuApi::Malloc<float>(H * B * sizeof(float));
  auto *d_d1 = ctp::GpuApi::Malloc<float>(H * B * sizeof(float));
  auto *d_d2 = ctp::GpuApi::Malloc<float>(O * B * sizeof(float));
  auto *d_lp = ctp::GpuApi::Malloc<double>(O * B * sizeof(double));
  auto *d_dg = ctp::GpuApi::Malloc<unsigned long long>(
      2 * sizeof(unsigned long long));

  std::vector<double> loss_ref(steps), loss_got(steps);
  // `lo`/`hi` bound which of d_lp this caller actually wrote. The dense
  // reference spans the whole array; a paged node writes only its own
  // o-band, and summing the rest would fold in whatever the dense pass
  // left there -- which is exactly how the distributed loss came out
  // node-dependent and too low.
  auto host_loss_range = [&](u64 lo, u64 hi) {
    std::vector<double> lp(O * B);
    ctp::GpuApi::Memcpy(lp.data(), d_lp, O * B * sizeof(double));
    double s = 0.0;                 // fixed order: one deterministic sum
    for (u64 i = lo; i < hi; ++i) s += lp[i];
    return s;
  };
  auto host_loss = [&]() {
    return host_loss_range(0, O * B) / static_cast<double>(B * O);
  };

  // ---- Dense reference training. -----------------------------------------
  auto *d_wref = ctp::GpuApi::Malloc<float>(n * sizeof(float));
  lb::LaunchDenseSeed(d_wref, n);
  ctp::GpuApi::Synchronize();
  const double t_ref0 = NowMs();
  for (u64 s = 0; s < steps; ++s) {
    lb::LaunchDenseFwd1(blocks, threads, d_wref, w1_off, b1_off, I, H, B, d_x,
                        d_a1, hper_all);
    lb::LaunchDenseFwd2(blocks, threads, d_wref, w2_off, b2_off, H, O, B, d_a1,
                        d_y, d_d2, d_lp, oper_all);
    lb::LaunchDenseBwd1(blocks, threads, d_wref, w2_off, H, O, B, d_a1, d_d2,
                        d_d1, hper_all, rpp2 ? rpp2 : 1);
    lb::LaunchDenseUpd2(blocks, threads, d_wref, w2_off, b2_off, H, O, B, d_a1,
                        d_d2, lr, oper_all);
    lb::LaunchDenseUpd1(blocks, threads, d_wref, w1_off, b1_off, I, H, B, d_x,
                        d_d1, lr, hper_all);
    ctp::GpuApi::Synchronize();
    loss_ref[s] = host_loss();
  }
  const double t_ref = NowMs() - t_ref0;
  ctp::GpuApi::Memset(d_dg, 0, 2 * sizeof(unsigned long long));
  lb::LaunchDenseDigest(d_wref, n, &d_dg[0]);
  ctp::GpuApi::Synchronize();

  // ---- Paged training. ---------------------------------------------------
  const u64 eper = ((npages + blocks - 1) / blocks) * elems_per_page;
  gv::Vector<float> w("gv_lbann_w", {0}, page_bytes, blocks, 24, n,
                      clio::run::PoolId::GetNull(), 0, 1, 0,
                      cap == 0 ? static_cast<u32>(npages + 2) : cap);
  w.EnableStats();
  auto dw = w.GetDevice(0);
  YieldRunner runner(blocks, threads);
  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                 gy::YieldStackView sv) {
    lb::LaunchSeed(g, b, gpu, dw, n, eper,
                                                elems_per_page, vw, sv);
  });
  ctp::GpuApi::Synchronize();

  // ---- cross-node collectives ------------------------------------------
  // Built after the runtime is up and only when there is an exchange to do.
  // The tag is NOT per-node: the slices have to meet.
  std::unique_ptr<clio::cte::core::Client> cte_x;
  clio::cte::core::TagId x_tag{};
  u64 x_round = 0;
  // clio::run::u64 is unsigned long; bench_dist speaks unsigned long long.
  using dist_u64 = clio_bench_dist::u64;
  std::vector<dist_u64> a1_lo(nodes), a1_hi(nodes), d2_lo(nodes),
      d2_hi(nodes);
  for (u32 nd = 0; nd < nodes; ++nd) {
    a1_lo[nd] = (H / nodes) * nd * B;  a1_hi[nd] = a1_lo[nd] + (H / nodes) * B;
    d2_lo[nd] = (O / nodes) * nd * B;  d2_hi[nd] = d2_lo[nd] + (O / nodes) * B;
  }
  if (nodes > 1) {
    cte_x = std::make_unique<clio::cte::core::Client>(
        clio::cte::core::kCtePoolId);
    auto t = cte_x->AsyncGetOrCreateTag("gv_lbann_x");
    t.Wait();
    if (t->GetReturnCode() != 0) {
      std::fprintf(stderr, "LBANN ERROR: could not create exchange tag\n");
      return 1;
    }
    x_tag = t->tag_id_;
  }
  // Staging for the gathers: the arrays live on the device.
  std::vector<float> h_a1(static_cast<size_t>(H) * B);
  std::vector<float> h_d2(static_cast<size_t>(O) * B);
  // d1 is only needed whole for the REPLICATED b1 update; the weight
  // update still uses this node's own rows.
  std::vector<float> h_d1(static_cast<size_t>(H) * B);
  const auto gather = [&](float *dev, std::vector<float> &host,
                          const std::vector<dist_u64> &los,
                          const std::vector<dist_u64> &his,
                          const char *what) -> bool {
    if (nodes <= 1) return true;
    ctp::GpuApi::Memcpy(host.data(), dev, host.size() * sizeof(float));
    if (!clio_bench_dist::AllGatherF32(*cte_x, x_tag, node, nodes,
                                       x_round++, host.data(), los[node],
                                       his[node], los.data(), his.data(),
                                       what)) {
      return false;
    }
    ctp::GpuApi::Memcpy(dev, host.data(), host.size() * sizeof(float));
    return true;
  };

  const double t0 = NowMs();
  for (u64 s = 0; s < steps; ++s) {
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      lb::LaunchFwd1(g, b, gpu, dw, w1_off, b1_off, I,
                                                  H, B, d_x, d_a1, hper, rpp1,
                                                  h0, h1, vw, sv);
    });
    // Fwd2 sums over EVERY h, so it needs the whole a1, not this node's
    // band. Without this each node forward-propagates a fraction of the
    // hidden layer and the loss is quietly wrong rather than failing.
    ctp::GpuApi::Synchronize();
    if (!gather(d_a1, h_a1, a1_lo, a1_hi, "lba1")) {
      std::fprintf(stderr, "LBANN ERROR: a1 gather failed\n");
      return 1;
    }
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      lb::LaunchFwd2(g, b, gpu, dw, w2_off, b2_off, H,
                                                  O, B, d_a1, d_y, d_d2, d_lp,
                                                  oper, rpp2 ? rpp2 : 1, o0, o1,
                                                  vw, sv);
    });
    // Bwd1 sums over EVERY o, so d2 has to be whole before it runs.
    ctp::GpuApi::Synchronize();
    if (!gather(d_d2, h_d2, d2_lo, d2_hi, "lbd2")) {
      std::fprintf(stderr, "LBANN ERROR: d2 gather failed\n");
      return 1;
    }
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      lb::LaunchBwd1(g, b, gpu, dw, w2_off, H, O, B,
                                                  d_a1, d_d2, d_d1, hper,
                                                  rpp2 ? rpp2 : 1, h0, h1,
                                                  // Bwd1 reads W2 PRE-update, i.e. as the
                                                  // peers left it at the end of the last
                                                  // step. Seed publishes 1, Upd2 at step s
                                                  // publishes s+2, so step s demands s+1.
                                                  o0, o1,
                                                  static_cast<u64>(s) + 1, vw, sv);
    });
    // b1 is computed from the WHOLE d1 on every node, so gather it before
    // the updates run.
    ctp::GpuApi::Synchronize();
    if (!gather(d_d1, h_d1, a1_lo, a1_hi, "lbd1")) {
      std::fprintf(stderr, "LBANN ERROR: d1 gather failed\n");
      return 1;
    }
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      lb::LaunchUpd2(g, b, gpu, dw, w2_off, b2_off, H,
                                                  O, B, d_a1, d_d2, lr, oper,
                                                  rpp2 ? rpp2 : 1, o0, o1,
                                                  static_cast<u64>(s) + 2, vw, sv);
    });
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      lb::LaunchUpd1(g, b, gpu, dw, w1_off, b1_off, I,
                                                  H, B, d_x, d_d1, lr, hper,
                                                  rpp1, h0, h1, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    // Each node summed only its own o-band; the loss is over the whole
    // output layer, so the partials are combined before the divide.
    double lpart = host_loss_range(o0 * B, o1 * B);
    if (nodes > 1 &&
        !clio_bench_dist::ReduceSum(*cte_x, x_tag, node, nodes, x_round++,
                                    &lpart, 1, "lbloss")) {
      std::fprintf(stderr, "LBANN ERROR: loss reduction failed\n");
      return 1;
    }
    loss_got[s] = lpart / static_cast<double>(B * O);
    // WHY THERE IS NO PER-STEP WEIGHT EXCHANGE HERE. Bwd1 reads ALL of W2,
    // including o-rows a peer just updated, so next step it sums against a
    // stale cached copy -- that is the residual 1e-3 the weight gate
    // reports. The obvious patch, FlushResidentToCte + barrier +
    // invalidate every step, was tried and made it far WORSE (loss
    // diverging from step 1, 1.08 vs 0.47): a whole-table flush also
    // republishes the peer's rows this node merely READ, clobbering their
    // newer values with a stale cached copy. The correct fix is the one
    // that worked for grayscott -- demand a generation on a PEER's page
    // and leave your own alone -- which needs generations threaded
    // through Bwd1's weight fetches, distinguishing own o-rows from a
    // peer's. That is real work, not a line here.
  }
  const double t_paged = NowMs() - t0;
  // THE DIGEST READS EVERY WEIGHT, including the bands this node's peers
  // updated. A page a peer still holds resident is invisible here, and a
  // page this node cached earlier is stale -- measured as the two nodes
  // reporting DIFFERENT digests, each correct only on its own band.
  // Publish, wait for every peer to publish, then drop the cache so the
  // digest pass refaults the whole vector from the CTE.
  if (nodes > 1) {
    ctp::GpuApi::Synchronize();
    w.FlushResidentToCte();
    if (!clio_bench_dist::Barrier(*cte_x, x_tag, node, nodes, x_round++,
                                  "lbdg")) {
      std::fprintf(stderr, "LBANN ERROR: digest barrier failed\n");
      return 1;
    }
    if (!clio_bench_dist::SettleAndInvalidate(w)) {
      std::fprintf(stderr, "LBANN ERROR: cache never settled\n");
      return 1;
    }
  }
  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                 gy::YieldStackView sv) {
    lb::LaunchDigest(g, b, gpu, dw, n, eper,
                                                  elems_per_page, &d_dg[1],
                                                  vw, sv);
  });
  ctp::GpuApi::Synchronize();
  unsigned long long dg[2] = {0, 0};
  ctp::GpuApi::Memcpy(dg, d_dg, sizeof(dg));

  const auto st = w.ReadStats(0);
  std::printf("  paging: faults=%llu evicts=%llu puts=%llu get_errors=%llu "
              "put_errors=%llu\n",
              (unsigned long long)st.faults, (unsigned long long)st.evicts,
              (unsigned long long)st.puts, (unsigned long long)st.get_errors,
              (unsigned long long)st.put_errors);
  std::printf("  %llu steps: paged %.1f ms/step, dense %.1f ms/step\n",
              (unsigned long long)steps, t_paged / steps, t_ref / steps);

  int rc = 0;
  bool loss_ok = true;
  // ACROSS NODES THE COMPARISON IS BOUNDED, NOT BIT-EQUAL. Combining
  // partials changes the summation order and float addition is not
  // associative, so a distributed run cannot reproduce the single-node
  // bits however correct it is. Measured drift over 5 steps: 6e-15 at step
  // 0 growing to 2.4e-8 at step 4. The single-node path keeps the exact
  // comparison -- there is no reordering there to excuse a difference.
  const double loss_tol = (nodes > 1) ? 1e-6 : 0.0;
  for (u64 s = 0; s < steps; ++s) {
    const double diff = loss_got[s] - loss_ref[s];
    const double adiff = diff < 0 ? -diff : diff;
    const double scale = loss_ref[s] != 0.0 ?
        (loss_ref[s] < 0 ? -loss_ref[s] : loss_ref[s]) : 1.0;
    if (adiff > loss_tol * scale) {
      std::printf("  LOSS GATE: step %llu paged %.17g != dense %.17g\n",
                  (unsigned long long)s, loss_got[s], loss_ref[s]);
      loss_ok = false;
    }
  }
  if (loss_ok) {
    std::printf("  LOSS GATE: PASS (all %llu steps within tolerance; final loss "
                "%.6f -> %.6f)\n",
                (unsigned long long)steps, loss_ref[0],
                loss_ref[steps - 1]);
  } else {
    rc = 1;
  }
  if (nodes > 1) {
    // ELEMENTWISE, not the digest. The digest is a bit-exact hash, so one
    // differing low bit rehashes to something completely unrelated and it
    // cannot express "close". This walks the same weights and takes the
    // largest absolute difference, so it still fails on a SINGLE wrong
    // element -- a sum or a looser hash would average one away.
    auto *d_md = ctp::GpuApi::Malloc<unsigned long long>(
        sizeof(unsigned long long));
    ctp::GpuApi::Memset(d_md, 0, sizeof(unsigned long long));
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      lb::LaunchMaxDiff(g, b, gpu, dw, n, eper, elems_per_page, d_wref,
                        d_md, 0, n, static_cast<u64>(steps) + 2, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    unsigned long long md = 0;
    ctp::GpuApi::Memcpy(&md, d_md, sizeof(md));
    ctp::GpuApi::Free(d_md);
    const double maxdiff = static_cast<double>(md) / 1e9;
    // WHICH region drifts. A whole-vector maximum says only that one does.
    if (const char *e = getenv("LB_REGION_DIFF")) {
      if (e[0] != '\0') {
        const struct { const char *nm; u64 lo, hi; } regs[] = {
            {"W1", w1_off, w1_off + I * H}, {"b1", b1_off, b1_off + H},
            {"W2", w2_off, w2_off + H * O}, {"b2", b2_off, b2_off + O},
            // W2 split by ownership. If the drift is in the PEER band the
            // generational demand is not landing; if it is in this node's
            // OWN band then Upd2 is subtracting from a stale W2. The two
            // have different fixes, so the probe has to tell them apart.
            {"W2-own", w2_off + o0 * H, w2_off + o1 * H},
            {"W2-peer-lo", w2_off, w2_off + o0 * H},
            {"W2-peer-hi", w2_off + o1 * H, w2_off + H * O}};
        for (const auto &r : regs) {
          auto *d_r = ctp::GpuApi::Malloc<unsigned long long>(
              sizeof(unsigned long long));
          ctp::GpuApi::Memset(d_r, 0, sizeof(unsigned long long));
          runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                         gy::YieldStackView sv) {
            lb::LaunchMaxDiff(g, b, gpu, dw, n, eper, elems_per_page,
                              d_wref, d_r, r.lo, r.hi,
                              static_cast<u64>(steps) + 2, vw, sv);
          });
          ctp::GpuApi::Synchronize();
          unsigned long long rv = 0;
          ctp::GpuApi::Memcpy(&rv, d_r, sizeof(rv));
          ctp::GpuApi::Free(d_r);
          std::printf("  REGION %s max|diff| = %.6g\n", r.nm,
                      static_cast<double>(rv) / 1e9);
        }
      }
    }
    // 1e-5 absolute on weights that start at O(1) and are stepped by an
    // lr-scaled gradient: loose enough for a reordered sum over 5 steps
    // (measured drift 2.4e-8 in the loss), tight enough that a stale or
    // dropped page -- which moves a weight by its whole update -- fails.
    if (maxdiff > 1e-5) {
      std::printf("  WEIGHT GATE: FAIL (max |paged - dense| = %.3g > "
                  "1e-5)\n", maxdiff);
      rc = 1;
    } else {
      std::printf("  WEIGHT GATE: PASS (max |paged - dense| = %.3g, "
                  "%u nodes)\n", maxdiff, nodes);
    }
  } else if (dg[0] != dg[1]) {
    std::printf("  WEIGHT GATE: FAIL (digest paged %llu != dense %llu)\n",
                dg[1], dg[0]);
    rc = 1;
  } else {
    std::printf("  WEIGHT GATE: PASS (final weights bit-equal to dense "
                "reference)\n");
  }
  std::printf("%s\n", rc == 0 ? "LBANN BENCH: ALL GATES PASS"
                              : "LBANN BENCH: GATE FAILURE");
  BenchFlushData();
  return rc;
}
#endif  // !CTP_IS_DEVICE_PASS
