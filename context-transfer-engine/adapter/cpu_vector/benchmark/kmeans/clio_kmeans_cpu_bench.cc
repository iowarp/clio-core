/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Out-of-core k-means (Lloyd's algorithm) on cpu_vector.
 *
 * The CPU port of the gpu_vector agent_eval k-means task
 * (gpu_vector/agent_eval/tasks/kmeans/TASK.md): same workload, same flags,
 * same RESULT line, so its output is directly comparable to that task's
 * independent reference (ref.cc, built here as clio_kmeans_cpu_ref).
 *
 *   - The point set lives in a cpu_vector<float> whose page cache is bounded
 *     by --cache-mb; the CTE backing store is a RAM tier sized for the data.
 *   - Data is generated INTO the vector by the worker threads (fetch, write
 *     hold, flush), so later iterations read it back through eviction.
 *   - Each iteration: threads take pages from a shared cursor, Fetch + hold
 *     each page, assign its points, accumulate per-thread double sums, and
 *     unpin. Only the iterations are timed.
 *
 * Extra flag: --threads N (default: hardware concurrency).
 */

#include <clio_cte/core/core_client.h>
#include <clio_cte/cpu_vector/cpu_vector.h>
#include <clio_runtime/clio_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace cv = clio::cte::cpu_vector;
using clio::run::u32;
using clio::run::u64;

namespace {

float PointVal(unsigned long long idx, unsigned dims, unsigned k) {
  const unsigned long long point = idx / dims;
  const unsigned long long dim = idx % dims;
  const unsigned long long cluster = point % k;
  const float centre = static_cast<float>(cluster) * 8.0f;
  const unsigned long long h =
      (point * 6364136223846793005ull + dim * 1442695040888963407ull);
  const float jitter =
      static_cast<float>(static_cast<unsigned>(h >> 40)) *
          (2.0f / 16777216.0f) -
      1.0f;
  return centre + jitter;
}

struct Args {
  u64 data_mb = 256;
  unsigned dims = 32;
  unsigned k = 16;
  int iters = 4;
  u64 cache_mb = 512;
  unsigned threads = 0;
};

Args Parse(int argc, char **argv) {
  Args a;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string f = argv[i];
    const char *v = argv[i + 1];
    if (f == "--data-mb") a.data_mb = std::strtoull(v, nullptr, 10);
    else if (f == "--dims") a.dims = static_cast<unsigned>(std::atoi(v));
    else if (f == "--clusters") a.k = static_cast<unsigned>(std::atoi(v));
    else if (f == "--iters") a.iters = std::atoi(v);
    else if (f == "--cache-mb") a.cache_mb = std::strtoull(v, nullptr, 10);
    else if (f == "--threads") a.threads = static_cast<unsigned>(std::atoi(v));
    else {
      std::fprintf(stderr, "unknown flag %s\n", f.c_str());
      std::exit(2);
    }
  }
  if (a.threads == 0) a.threads = std::max(1u, std::thread::hardware_concurrency());
  return a;
}

/** Runtime config: one RAM tier sized for the whole data set (+ slack). */
std::string WriteConfig(u64 data_mb) {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  const std::string path =
      std::string((d && *d) ? d : "/tmp") + "/kmeans_cpu_config.yaml";
  std::ofstream f(path);
  f << "networking:\n  port: 9462\n"
       "runtime:\n  num_threads: 4\n  queue_depth: 1024\n"
       "compose:\n  - mod_name: clio_cte_core\n    pool_name: clio_cte\n"
       "    pool_query: local\n    pool_id: 512.0\n    storage:\n"
       "      - path: \"ram::kmeans_cpu_dram\"\n        bdev_type: \"ram\"\n"
       "        capacity_limit: \""
    << (data_mb * 2 + 64) << "MB\"\n        score: 1.0\n"
       "    dpe:\n      dpe_type: \"max_bw\"\n";
  return path;
}

/** Run fn(thread, page) over pages [0, np) with a shared cursor. */
template <typename Fn>
void ParallelPages(unsigned threads, u64 np, Fn fn) {
  std::atomic<u64> next{0};
  std::atomic<int> failed{0};
  std::vector<std::thread> ts;
  for (unsigned t = 0; t < threads; ++t) {
    ts.emplace_back([&, t] {
      try {
        for (u64 pg; (pg = next.fetch_add(1)) < np;) fn(t, pg);
      } catch (const std::exception &e) {
        std::fprintf(stderr, "thread %u: %s\n", t, e.what());
        failed.store(1);
      }
    });
  }
  for (auto &th : ts) th.join();
  if (failed.load()) std::exit(1);
}

}  // namespace

int main(int argc, char **argv) {
  const Args a = Parse(argc, argv);
  const u64 npts = a.data_mb * 1048576ull / (4ull * a.dims);
  const u64 n = npts * a.dims;

  // A page holds a whole number of points: 64 KiB rounded down to a
  // multiple of one point's bytes.
  const u64 pt_bytes = 4ull * a.dims;
  const u64 page_bytes = std::max<u64>(pt_bytes, (65536 / pt_bytes) * pt_bytes);
  const u64 frames = std::max<u64>(a.cache_mb * 1048576ull / page_bytes,
                                   2ull * a.threads);
  const u32 set_size = 8;
  const u32 nsets = static_cast<u32>((frames + set_size - 1) / set_size);
  const u32 capacity = static_cast<u32>(frames);

  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", WriteConfig(a.data_mb).c_str(), 1);
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true) ||
      !clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "runtime init failed\n");
    return 1;
  }

  cv::Vector<float> vec("kmeans_cpu_points", page_bytes, n, nsets, set_size,
                        clio::run::PoolId::GetNull(), 0, 1, capacity);
  vec.EnableStats();
  const u64 epp = vec.ElemsPerPage();
  const u64 np = vec.NumPages();
  std::fprintf(stderr,
               "kmeans_cpu: %llu points x %u dims, %llu pages of %llu B, "
               "cache %u frames (%llu MB), %u threads\n",
               (unsigned long long)npts, a.dims, (unsigned long long)np,
               (unsigned long long)page_bytes, capacity,
               (unsigned long long)(capacity * page_bytes >> 20), a.threads);

  // ---- generate into the vector (untimed) ----
  ParallelPages(a.threads, np, [&](unsigned, u64 pg) {
    const u64 lo = pg * epp;
    const u64 cnt = std::min(epp, n - lo);
    vec.Fetch(0, lo, cnt);
    auto h = vec.HoldPage(lo, cnt, /*write=*/true);
    for (u64 i = lo; i < lo + cnt; ++i) h[i] = PointVal(i, a.dims, a.k);
    vec.Flush(0, lo, cnt);
    vec.UnpinRange(lo, cnt);
  });

  // ---- initial centroids: the first k points, read back from the vector ----
  std::vector<float> cent(static_cast<size_t>(a.k) * a.dims);
  vec.Fetch(0, 0, static_cast<u64>(a.k) * a.dims);
  for (u64 i = 0; i < cent.size();) {
    auto h = vec.HoldPage(i, cent.size() - i);
    for (u64 j = 0; j < h.run(); ++j) cent[i + j] = h[i + j];
    i += h.run();
  }
  vec.UnpinRange(0, static_cast<u64>(a.k) * a.dims);
  vec.ResetStats();

  // ---- timed iterations ----
  std::vector<u64> counts(a.k);
  std::vector<std::vector<double>> tsum(a.threads);
  std::vector<std::vector<u64>> tcnt(a.threads);
  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < a.iters; ++it) {
    for (unsigned t = 0; t < a.threads; ++t) {
      tsum[t].assign(static_cast<size_t>(a.k) * a.dims, 0.0);
      tcnt[t].assign(a.k, 0);
    }
    ParallelPages(a.threads, np, [&](unsigned t, u64 pg) {
      const u64 lo = pg * epp;
      const u64 cnt = std::min(epp, n - lo);
      vec.Fetch(0, lo, cnt);
      auto h = vec.HoldPage(lo, cnt);
      const float *pts = h.ptr();
      double *ls = tsum[t].data();
      u64 *lc = tcnt[t].data();
      for (u64 p = 0; p < cnt / a.dims; ++p) {
        const float *pt = pts + p * a.dims;
        float best = 3.4e38f;
        unsigned bk = 0;
        for (unsigned c = 0; c < a.k; ++c) {
          const float *cc = cent.data() + static_cast<size_t>(c) * a.dims;
          float d = 0.0f;
          for (unsigned i = 0; i < a.dims; ++i) {
            const float x = pt[i] - cc[i];
            d += x * x;
          }
          if (d < best) { best = d; bk = c; }
        }
        for (unsigned i = 0; i < a.dims; ++i) ls[bk * a.dims + i] += pt[i];
        ++lc[bk];
      }
      vec.UnpinRange(lo, cnt);
    });
    std::vector<double> sums(static_cast<size_t>(a.k) * a.dims, 0.0);
    std::fill(counts.begin(), counts.end(), 0);
    for (unsigned t = 0; t < a.threads; ++t) {
      for (size_t i = 0; i < sums.size(); ++i) sums[i] += tsum[t][i];
      for (unsigned c = 0; c < a.k; ++c) counts[c] += tcnt[t][c];
    }
    for (unsigned c = 0; c < a.k; ++c) {
      if (counts[c] == 0) continue;
      for (unsigned i = 0; i < a.dims; ++i) {
        cent[c * a.dims + i] =
            static_cast<float>(sums[c * a.dims + i] / counts[c]);
      }
    }
  }
  const double ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count() /
      std::max(1, a.iters);

  double csum = 0.0;
  for (float v : cent) csum += v;
  const auto s = vec.ReadStats();
  std::printf("RESULT counts=");
  for (unsigned c = 0; c < a.k; ++c) {
    std::printf(c ? ",%llu" : "%llu", (unsigned long long)counts[c]);
  }
  std::printf(" csum=%.6f ms_per_iter=%.3f faults=%llu evicts=%llu\n", csum,
              ms, (unsigned long long)s.faults, (unsigned long long)s.evicts);
  std::fflush(stdout);
  return 0;
}
