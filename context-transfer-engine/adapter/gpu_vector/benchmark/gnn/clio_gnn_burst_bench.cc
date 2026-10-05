/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * DATA ORGANIZATION for bursty arrivals: cluster-batched GNN over a tiered,
 * paged feature matrix.
 *
 * THE QUESTION. Not "does prefetching pay per page" -- that framing was
 * measured to death on Gray-Scott and loses, because a sliding stencil reads
 * each page once per step and its write order already IS its access order, so
 * a migration is a pure extra copy. The question here is capacity management
 * for a BURST: when one batch's working set arrives, is there room for it in
 * the fast tier, and is it already there?
 *
 * THE WORKLOAD. Cluster-GCN: each batch's seeds come from one METIS partition,
 * so it needs that partition's nodes plus their 1-hop halo. Feature rows are
 * renumbered partition-contiguous (gnn_burst_prep.py), which is what turns a
 * node set into a small PAGE set -- without that renumbering a batch touches
 * ~100% of pages at every page size and NOTHING can help. Measured on
 * ogbn-products, 2048 partitions, 100 KiB pages: a burst is ~1247 of 9567
 * pages (122 MiB of 934 MiB), and consecutive bursts in greedy order share
 * about half their pages.
 *
 * THE TWO ARMS.
 *   none      run the bursts. Pages sit wherever the DPE placed them at
 *             ingest, which is write order -- and write order here is NOT
 *             access order, unlike Gray-Scott. That is the whole point.
 *   organize  one batch ahead: FREE the pages the departing burst no longer
 *             needs (demote), and FETCH the arriving burst's pages (promote),
 *             both as one CTE bulk rescore. Freeing is not overhead here, it
 *             is what makes room for the next burst to land fast.
 *
 * WHAT IS REPORTED. Wall clock, fault service time, and the tier split -- but
 * the number this exists to produce is SLOW-TIER BYTES PER BURST, because a
 * full fast tier does not stall in this CTE (MaxBwDpe drops a target with no
 * room and silently ranks the next one), so the cost of having no headroom is
 * misplacement, which is invisible in any latency counter.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>

#include <chrono>
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
#else
static constexpr u32 kYieldLaneBytes = 256;
#endif

#include "gnn_burst_kernels.h"
#include "gnn_burst_launch.h"

namespace gb = clio::gv_bench::gnn_burst;

#if !CTP_IS_DEVICE_PASS

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

class YieldRunner {
 public:
  YieldRunner(unsigned nb, unsigned nt)
      : drv_(nb, nt), stack_(nb, nt, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> v) {
          launch(g, b, v, stack_.View());
        },
        [] {}, /*max_rounds=*/2000000, gv::ResumeWhenComplete);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

/** One burst: the page ids it needs, in ascending order. */
struct Burst {
  std::vector<u64> pages;
};

/** bursts.bin: header [nbursts, npages, rows_per_page, N, F] then per burst
 *  [count, page ids...]. Written by gnn_burst_prep.py. */
bool LoadBursts(const std::string &path, std::vector<Burst> &out, u64 &npages,
                u64 &rows_per_page, u64 &N, u64 &F) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  long long hdr[5];
  f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
  if (!f) return false;
  const long long nb = hdr[0];
  npages = static_cast<u64>(hdr[1]);
  rows_per_page = static_cast<u64>(hdr[2]);
  N = static_cast<u64>(hdr[3]);
  F = static_cast<u64>(hdr[4]);
  out.resize(static_cast<size_t>(nb));
  for (long long i = 0; i < nb; ++i) {
    long long cnt = 0;
    f.read(reinterpret_cast<char *>(&cnt), sizeof(cnt));
    if (!f || cnt < 0) return false;
    out[i].pages.resize(static_cast<size_t>(cnt));
    f.read(reinterpret_cast<char *>(out[i].pages.data()),
           static_cast<std::streamsize>(cnt * sizeof(long long)));
    if (!f) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  std::string data = "gnndata/burst2048";
  std::string mode = "none";          // none | organize
  std::string nvme_path = "./gnn_tier.dat";
  u32 blocks = 8, threads = 256, slots = 16;
  int repeat = 1;
  // Tier split as a PERCENTAGE OF THE DATASET, so the hierarchy scales with
  // whatever feature matrix is loaded.
  double vram_pct = 15.0, dram_pct = 15.0, nvme_pct = 80.0;
  float hbm_score = 0.5f, ram_score = 0.3f, nvme_score = 0.1f;
  float hot = 0.6f, cold = 0.1f;
  int lead = 1;                       // organize this many bursts ahead
  // WHEN THE WORKING SET FITS THE FAST TIER, FREEING IS PURE COST. The frees
  // exist to make room; with WS < tier there is already room, and every free
  // is a migration (read+write) spent to create headroom nobody needed.
  bool do_free = true;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() -> std::string {
      return (i + 1 < argc) ? argv[++i] : std::string();
    };
    if (a == "--data") data = nx();
    else if (a == "--mode") mode = nx();
    else if (a == "--nvme-path") nvme_path = nx();
    else if (a == "--blocks") blocks = std::stoul(nx());
    else if (a == "--threads") threads = std::stoul(nx());
    else if (a == "--slots") slots = std::stoul(nx());
    else if (a == "--repeat") repeat = std::stoi(nx());
    else if (a == "--vram-pct") vram_pct = std::stod(nx());
    else if (a == "--dram-pct") dram_pct = std::stod(nx());
    else if (a == "--nvme-pct") nvme_pct = std::stod(nx());
    else if (a == "--hot") hot = std::stof(nx());
    else if (a == "--cold") cold = std::stof(nx());
    else if (a == "--lead") lead = std::stoi(nx());
    else if (a == "--no-free") do_free = false;
    else if (a == "--help") {
      std::printf("usage: %s [--data DIR] [--mode none|organize] "
                  "[--blocks N] [--slots N] [--repeat N]\n"
                  "       [--vram-pct f] [--dram-pct f] [--nvme-pct f] "
                  "[--hot f] [--cold f] [--lead N] [--no-free]\n", argv[0]);
      return 0;
    }
  }
  if (mode != "none" && mode != "organize") {
    std::fprintf(stderr, "GNNBURST ERROR: --mode must be none|organize\n");
    return 2;
  }

  std::vector<Burst> bursts;
  u64 npages = 0, rows_per_page = 0, N = 0, F = 0;
  if (!LoadBursts(data + "/bursts.bin", bursts, npages, rows_per_page, N, F)) {
    std::fprintf(stderr, "GNNBURST ERROR: cannot read %s/bursts.bin -- run "
                 "gnn_burst_prep.py first\n", data.c_str());
    return 2;
  }
  const u64 page_bytes = rows_per_page * F * sizeof(float);
  const u64 epp = page_bytes / sizeof(float);
  const u64 n_elems = npages * epp;
  const double data_mb =
      static_cast<double>(npages * page_bytes) / (1024.0 * 1024.0);
  double ws_pages = 0;
  for (const auto &b : bursts) ws_pages += static_cast<double>(b.pages.size());
  ws_pages /= static_cast<double>(bursts.size());

  const u64 vram_mb = static_cast<u64>(data_mb * vram_pct / 100.0);
  const u64 dram_mb = static_cast<u64>(data_mb * dram_pct / 100.0);
  const u64 nvme_mb = static_cast<u64>(data_mb * nvme_pct / 100.0);

  std::printf("GNN burst: N=%llu F=%llu pages=%llu page=%lluKiB data=%.0fMiB\n"
              "  bursts=%zu  WS=%.0f pages (%.0fMiB, %.1f%% of data)\n"
              "  tiers: vram=%lluMB dram=%lluMB nvme=%lluMB  mode=%s lead=%d\n",
              (unsigned long long)N, (unsigned long long)F,
              (unsigned long long)npages, (unsigned long long)(page_bytes >> 10),
              data_mb, bursts.size(), ws_pages,
              ws_pages * page_bytes / (1024.0 * 1024.0),
              100.0 * ws_pages / static_cast<double>(npages),
              (unsigned long long)vram_mb, (unsigned long long)dram_mb,
              (unsigned long long)nvme_mb, mode.c_str(), lead);

  if (getenv("CLIO_SERVER_CONF") == nullptr) {
    std::ofstream cfg("gnn_burst_bench.yaml");
    cfg << "networking:\n  port: 9443\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n"
        << "  first_busy_wait: 10000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
        << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
        << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n"
        // Every tier at or below the vector's put score (0.5) so the DPE fills
        // them in CAPACITY order; a tier scored above it is never a first
        // choice and silently sits empty.
        << "      - path: \"hbm::gnn_burst_hbm\"\n        bdev_type: \"hbm\"\n"
        << "        capacity_limit: \"" << vram_mb << "MB\"\n"
        << "        score: " << hbm_score << "\n"
        << "      - path: \"ram::gnn_burst_ram\"\n        bdev_type: \"ram\"\n"
        << "        capacity_limit: \"" << dram_mb << "MB\"\n"
        << "        score: " << ram_score << "\n"
        << "      - path: \"" << nvme_path << "\"\n"
        << "        bdev_type: \"file\"\n"
        << "        persistence_level: \"temporary\"\n"
        << "        capacity_limit: \"" << nvme_mb << "MB\"\n"
        << "        score: " << nvme_score << "\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gnn_burst_bench.yaml", 1);
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "GNNBURST ERROR: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "GNNBURST ERROR: cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  gv::Vector<float> vec("gnn_burst", {0}, page_bytes, blocks,
                        slots < 24u ? 24u : slots, n_elems);
  vec.EnableStats();
  auto dev = vec.GetDevice(0);

  // ---- INGEST -----------------------------------------------------------
  // Straight from the reordered file, page by page. This is what establishes
  // the BASELINE PLACEMENT: the DPE fills vram, then dram, then storage in
  // write order, so low page ids land fast and high ones land slow. Access
  // order (the burst sequence) is unrelated to it -- which is exactly the
  // condition Gray-Scott lacked.
  {
    const std::string fp = data + "/feat_reordered.f32";
    std::ifstream in(fp, std::ios::binary);
    if (!in) {
      std::fprintf(stderr, "GNNBURST ERROR: cannot open %s\n", fp.c_str());
      return 1;
    }
    std::vector<char> buf(static_cast<size_t>(page_bytes));
    const double t0 = NowMs();
    vec.PreloadPages(0, npages, [&](u64 pg, char *dstb) {
      in.read(buf.data(), static_cast<std::streamsize>(page_bytes));
      const std::streamsize got = in.gcount();
      std::memcpy(dstb, buf.data(), static_cast<size_t>(got));
      if (static_cast<u64>(got) < page_bytes) {
        std::memset(dstb + got, 0, static_cast<size_t>(page_bytes - got));
      }
      (void)pg;
    });
    std::printf("  ingest: %.0f MiB in %.1f s\n", data_mb,
                (NowMs() - t0) / 1000.0);
  }

  // Device page list, sized for the largest burst.
  size_t max_pages = 0;
  for (const auto &b : bursts) max_pages = std::max(max_pages, b.pages.size());
  u64 *d_pages = ctp::GpuApi::Malloc<u64>(max_pages * sizeof(u64));
  double *d_out = ctp::GpuApi::Malloc<double>(F * sizeof(double));
  YieldRunner runner(blocks, threads);

  gv::PrefetchPolicy pol;
  pol.hot_ = hot;
  pol.cold_ = cold;
  vec.SetPrefetchPolicy(pol);

  double best_ms = 1e30, checksum = 0.0;
  for (int r = 0; r < repeat; ++r) {
    vec.ResetStats();
    vec.ResetPrefetchStats();
    vec.ResetPrefetchHistory();
    ctp::GpuApi::Memset(d_out, 0, F * sizeof(double));
    ctp::GpuApi::Synchronize();
    const double t0 = NowMs();

    for (size_t b = 0; b < bursts.size(); ++b) {
      // ---- THE ORGANIZER, one batch ahead ----------------------------
      // FREE then FETCH, as one batch of hints. Freeing is not overhead: a
      // full fast tier does not stall, it silently sends the arriving burst
      // somewhere slow, so making room IS the mechanism.
      if (mode == "organize") {
        const size_t nb = b + static_cast<size_t>(lead);
        if (nb < bursts.size()) {
          const auto &cur = bursts[b].pages;
          const auto &nxt = bursts[nb].pages;
          // FREE FIRST, AS ITS OWN SUBMISSION, AND ONLY THEN FETCH.
          //
          // This ordering is the whole mechanism, not a detail. MaxBwDpe has
          // NO EVICTION: it filters targets by remaining_space_ >= data_size
          // and silently ranks the next tier down. So a promotion into a FULL
          // fast tier does not displace anything -- it lands in DRAM or
          // storage, and the organizer's own demotions are the only thing
          // that can make room for it.
          //
          // Submitting both in one call merged them into a single page-ordered
          // batch, so frees and fetches interleaved and raced: measured
          // vram=14.9% (unchanged, i.e. still full) with dram collapsing
          // 14.9%->5.4% -- the organizer pushed data DOWN and got nothing UP.
          // Two ordered submissions keep the frees ahead of the fetches in the
          // queue.
          std::vector<gv::PrefetchHint> frees, fetches;
          frees.reserve(cur.size());
          fetches.reserve(nxt.size());
          if (do_free) {
            size_t j = 0;
            for (u64 p : cur) {
              while (j < nxt.size() && nxt[j] < p) ++j;
              if (j >= nxt.size() || nxt[j] != p) {
                frees.push_back(gv::PrefetchHint{p, cold});
              }
            }
          }
          for (u64 p : nxt) fetches.push_back(gv::PrefetchHint{p, hot});
          vec.PrefetchNow(frees);
          vec.PrefetchNow(fetches);
        }
      }

      const auto &pg = bursts[b].pages;
      ctp::GpuApi::Memcpy(reinterpret_cast<char *>(d_pages),
                          reinterpret_cast<const char *>(pg.data()),
                          pg.size() * sizeof(u64));
      const u64 np = pg.size();
      runner.Run([&](dim3 g, dim3 t, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        gb::LaunchBurst(g, t, gpu, dev, d_pages, np, epp, blocks, d_out,
                        static_cast<u32>(F), vw, sv);
      });
    }
    ctp::GpuApi::Synchronize();
    const double ms = NowMs() - t0;
    if (ms < best_ms) best_ms = ms;
    vec.DrainPrefetch();

    std::vector<double> h_out(static_cast<size_t>(F), 0.0);
    ctp::GpuApi::Memcpy(h_out.data(), d_out, F * sizeof(double));
    checksum = 0.0;
    for (double v : h_out) checksum += v;
  }

  const auto st = vec.ReadStats(0);
  const auto pf = vec.ReadPrefetchStats();

  // ---- TIER SPLIT -------------------------------------------------------
  {
    clio::run::bdev::Client t_fast(clio::run::PoolId(512, 1));
    clio::run::bdev::Client t_host(clio::run::PoolId(513, 1));
    clio::run::bdev::Client t_stor(clio::run::PoolId(514, 1));
    auto fa = t_fast.AsyncGetStats(); fa.Wait();
    auto ha = t_host.AsyncGetStats(); ha.Wait();
    auto sa = t_stor.AsyncGetStats(); sa.Wait();
    auto used = [](u64 cap, u64 rem) { return cap > rem ? cap - rem : 0ull; };
    const u64 fc = vram_mb * 1024ull * 1024ull, hc = dram_mb * 1024ull * 1024ull,
              sc = nvme_mb * 1024ull * 1024ull;
    const u64 fu = used(fc, fa->remaining_size_), hu = used(hc, ha->remaining_size_),
              su = used(sc, sa->remaining_size_);
    const double tot = static_cast<double>((fu + hu + su) >> 20);
    std::fprintf(stderr,
                 "TIER PCT: vram=%.1f%% dram=%.1f%% nvme=%.1f%% "
                 "(placed %.0fMiB of %.0fMiB)\n",
                 tot > 0 ? 100.0 * (fu >> 20) / tot : 0.0,
                 tot > 0 ? 100.0 * (hu >> 20) / tot : 0.0,
                 tot > 0 ? 100.0 * (su >> 20) / tot : 0.0, tot, data_mb);
  }

  std::fprintf(stderr,
               "GNNBURST mode=%s blocks=%u slots=%u pages=%llu page_kb=%llu "
               "bursts=%zu ws_pages=%.0f data_mb=%.0f vram_mb=%llu "
               "ms=%.1f checksum=%.6f faults=%llu evicts=%llu "
               "get_errors=%llu pf_sent=%llu pf_promote=%llu pf_demote=%llu "
               "pf_deduped=%llu pf_errors=%llu pf_notfound=%llu lead=%d\n",
               mode.c_str(), blocks, slots, (unsigned long long)npages,
               (unsigned long long)(page_bytes >> 10), bursts.size(), ws_pages,
               data_mb, (unsigned long long)vram_mb, best_ms, checksum,
               (unsigned long long)st.faults, (unsigned long long)st.evicts,
               (unsigned long long)st.get_errors,
               (unsigned long long)pf.sent_, (unsigned long long)pf.promotions_,
               (unsigned long long)pf.demotions_,
               (unsigned long long)pf.deduped_, (unsigned long long)pf.errors_,
               (unsigned long long)pf.not_found_, lead);

  ctp::GpuApi::Free(d_pages);
  ctp::GpuApi::Free(d_out);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}

#endif  // !CTP_IS_DEVICE_PASS
