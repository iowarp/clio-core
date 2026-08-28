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
  u64 page_kb = 64, I = 256, H = 4096, O = 64, B = 64, steps = 5;
  float lr = 0.01f;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
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
  const u64 hper = H / blocks, oper = O / blocks;

  {
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
  auto host_loss = [&]() {
    std::vector<double> lp(O * B);
    ctp::GpuApi::Memcpy(lp.data(), d_lp, O * B * sizeof(double));
    double s = 0.0;                 // fixed order: one deterministic sum
    for (u64 i = 0; i < O * B; ++i) s += lp[i];
    return s / static_cast<double>(B * O);
  };

  // ---- Dense reference training. -----------------------------------------
  auto *d_wref = ctp::GpuApi::Malloc<float>(n * sizeof(float));
  lb::LaunchDenseSeed(d_wref, n);
  ctp::GpuApi::Synchronize();
  const double t_ref0 = NowMs();
  for (u64 s = 0; s < steps; ++s) {
    lb::LaunchDenseFwd1(blocks, threads, d_wref, w1_off, b1_off, I, H, B, d_x,
                        d_a1, hper);
    lb::LaunchDenseFwd2(blocks, threads, d_wref, w2_off, b2_off, H, O, B, d_a1,
                        d_y, d_d2, d_lp, oper);
    lb::LaunchDenseBwd1(blocks, threads, d_wref, w2_off, H, O, B, d_a1, d_d2,
                        d_d1, hper, rpp2 ? rpp2 : 1);
    lb::LaunchDenseUpd2(blocks, threads, d_wref, w2_off, b2_off, H, O, B, d_a1,
                        d_d2, lr, oper);
    lb::LaunchDenseUpd1(blocks, threads, d_wref, w1_off, b1_off, I, H, B, d_x,
                        d_d1, lr, hper);
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

  const double t0 = NowMs();
  for (u64 s = 0; s < steps; ++s) {
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      lb::LaunchFwd1(g, b, gpu, dw, w1_off, b1_off, I,
                                                  H, B, d_x, d_a1, hper, rpp1,
                                                  vw, sv);
    });
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      lb::LaunchFwd2(g, b, gpu, dw, w2_off, b2_off, H,
                                                  O, B, d_a1, d_y, d_d2, d_lp,
                                                  oper, rpp2 ? rpp2 : 1, vw,
                                                  sv);
    });
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      lb::LaunchBwd1(g, b, gpu, dw, w2_off, H, O, B,
                                                  d_a1, d_d2, d_d1, hper,
                                                  rpp2 ? rpp2 : 1, vw, sv);
    });
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      lb::LaunchUpd2(g, b, gpu, dw, w2_off, b2_off, H,
                                                  O, B, d_a1, d_d2, lr, oper,
                                                  rpp2 ? rpp2 : 1, vw, sv);
    });
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      lb::LaunchUpd1(g, b, gpu, dw, w1_off, b1_off, I,
                                                  H, B, d_x, d_d1, lr, hper,
                                                  rpp1, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    loss_got[s] = host_loss();
  }
  const double t_paged = NowMs() - t0;
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
  for (u64 s = 0; s < steps; ++s) {
    if (loss_got[s] != loss_ref[s]) {
      std::printf("  LOSS GATE: step %llu paged %.17g != dense %.17g\n",
                  (unsigned long long)s, loss_got[s], loss_ref[s]);
      loss_ok = false;
    }
  }
  if (loss_ok) {
    std::printf("  LOSS GATE: PASS (all %llu steps bit-equal; final loss "
                "%.6f -> %.6f)\n",
                (unsigned long long)steps, loss_ref[0],
                loss_ref[steps - 1]);
  } else {
    rc = 1;
  }
  if (dg[0] != dg[1]) {
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
