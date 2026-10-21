/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * cpu_vector against an in-process runtime with one RAM tier.
 *
 * Covers the contract the GPU tests check for gpu_vector, minus the device:
 * preload/fetch/hold round trip, write + flush + download, HoldPage on an
 * unfetched page, eviction under a cache much smaller than the vector, many
 * threads sharing one cache, and a generational flush/fetch.
 */

#include <clio_cte/core/core_client.h>
#include <clio_cte/cpu_vector/cpu_vector.h>
#include <clio_runtime/clio_runtime.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;
namespace cv = clio::cte::cpu_vector;
using clio::run::u32;
using clio::run::u64;

namespace {

/** Where the runtime config goes: CLIO_TEST_DATA_DIR if it names a real
 *  directory (trimmed), else the system temp directory. */
std::string TestDataDir() {
  std::string d;
  if (const char *e = clio::run::env::GetCompat("TEST_DATA_DIR")) d = e;
  while (!d.empty() && (d.back() == ' ' || d.back() == '\t')) d.pop_back();
  std::error_code ec;
  if (!d.empty() && fs::is_directory(d, ec)) return d;
  return fs::temp_directory_path(ec).string();
}

class Fixture {
 public:
  std::string config_path_;
  Fixture() {
    config_path_ = TestDataDir() + "/cpu_vector_config.yaml";
    if (fs::exists(config_path_)) fs::remove(config_path_);
    {
      std::ofstream f(config_path_);
      REQUIRE(f.is_open());
      // A dedicated port: the runtime default (9413) is shared with every
      // other test and any running daemon.
      f << R"(
networking:
  port: 9461
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_cte_core
    pool_name: clio_cte
    pool_query: local
    pool_id: 512.0
    storage:
      - path: "ram::cpu_vector_dram"
        bdev_type: "ram"
        capacity_limit: "256MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
)";
    }
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  ~Fixture() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
  }
};

Fixture &Env() {
  static Fixture f;
  return f;
}

constexpr u64 kPageBytes = 4096;
constexpr u64 kEpp = kPageBytes / sizeof(u64);

std::vector<u64> Iota(u64 n, u64 base) {
  std::vector<u64> v(n);
  std::iota(v.begin(), v.end(), base);
  return v;
}

}  // namespace

TEST_CASE("cpu_vector: preload, fetch, hold reads what was written",
          "[cpu_vector]") {
  Env();
  const u64 n = 10 * kEpp + 17;  // a partial last page
  cv::Vector<u64> v("cpuvec_roundtrip", kPageBytes, n, 4, 4);
  const auto src = Iota(n, 1000);
  v.Preload(src.data(), n);

  v.Fetch(0, 0, n);
  for (u64 off = 0; off < n; off += kEpp) {
    auto h = v.HoldPage(off, kEpp);
    REQUIRE(h);
    for (u64 i = off; i < off + h.run(); ++i) REQUIRE(h[i] == src[i]);
  }
  v.UnpinRange(0, n);
  REQUIRE(v.MaxPageCopies() == 1);
}

TEST_CASE("cpu_vector: write, flush, download", "[cpu_vector]") {
  Env();
  const u64 n = 6 * kEpp;
  cv::Vector<u64> v("cpuvec_flush", kPageBytes, n, 2, 4);
  v.Fetch(0, 0, n);  // nothing written yet: pages read as zeros
  for (u64 off = 0; off < n; off += kEpp) {
    auto h = v.HoldPage(off, kEpp, /*write=*/true);
    for (u64 i = off; i < off + kEpp; ++i) h[i] = i * 3;
  }
  v.Flush(0, 0, n);
  v.UnpinRange(0, n);

  std::vector<u64> out(n, 0);
  REQUIRE(v.Download(out.data(), n) == n);
  for (u64 i = 0; i < n; ++i) REQUIRE(out[i] == i * 3);
}

TEST_CASE("cpu_vector: HoldPage without Fetch throws", "[cpu_vector]") {
  Env();
  cv::Vector<u64> v("cpuvec_nofetch", kPageBytes, 4 * kEpp, 2, 2);
  bool threw = false;
  try {
    (void)v.HoldPage(kEpp, 1);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  REQUIRE(threw);

  // A partial fetch covers only what it named.
  v.Fetch(0, 0, 8);
  threw = false;
  try {
    (void)v.HoldPage(16, 1);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  REQUIRE(threw);
  v.UnpinRange(0, 8);
}

TEST_CASE("cpu_vector: a cache far smaller than the vector evicts",
          "[cpu_vector]") {
  Env();
  const u64 pages = 64;
  const u64 n = pages * kEpp;
  // 2 sets x 2 frames = 4 resident pages for 64.
  cv::Vector<u64> v("cpuvec_evict", kPageBytes, n, 2, 2);
  v.EnableStats();
  const auto src = Iota(n, 7);
  v.Preload(src.data(), n);
  for (int pass = 0; pass < 2; ++pass) {
    for (u64 pg = 0; pg < pages; ++pg) {
      v.Fetch(0, pg * kEpp, kEpp);
      auto h = v.HoldPage(pg * kEpp, kEpp);
      REQUIRE(h[pg * kEpp] == src[pg * kEpp]);
      REQUIRE(h[pg * kEpp + kEpp - 1] == src[pg * kEpp + kEpp - 1]);
      v.UnpinRange(pg * kEpp, kEpp);
    }
  }
  const auto s = v.ReadStats();
  REQUIRE(s.evicts > 0);
  REQUIRE(s.get_errors == 0);
  REQUIRE(v.Occupied() <= 4);
}

TEST_CASE("cpu_vector: many threads share one cache", "[cpu_vector]") {
  Env();
  const u64 pages = 32;
  const u64 n = pages * kEpp;
  cv::Vector<u64> v("cpuvec_threads", kPageBytes, n, 4, 4);
  const auto src = Iota(n, 1);
  v.Preload(src.data(), n);

  constexpr int kThreads = 8;
  std::atomic<u64> bad{0}, errors{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&, t] {
      try {
        for (int it = 0; it < 200; ++it) {
          const u64 pg = (static_cast<u64>(t) * 7 + it * 13) % pages;
          v.Fetch(0, pg * kEpp, kEpp);
          auto h = v.HoldPage(pg * kEpp, kEpp);
          for (u64 i = pg * kEpp; i < (pg + 1) * kEpp; i += 61) {
            if (h[i] != src[i]) bad.fetch_add(1);
          }
          v.UnpinRange(pg * kEpp, kEpp);
        }
      } catch (...) {
        errors.fetch_add(1);
      }
    });
  }
  for (auto &th : ts) th.join();
  REQUIRE(errors.load() == 0);
  REQUIRE(bad.load() == 0);
  REQUIRE(v.MaxPageCopies() == 1);
}

TEST_CASE("cpu_vector: generational flush releases a generational fetch",
          "[cpu_vector]") {
  Env();
  const u64 n = 2 * kEpp;
  cv::Vector<u64> w("cpuvec_gen", kPageBytes, n, 2, 2);
  cv::Vector<u64> r("cpuvec_gen", kPageBytes, n, 2, 2);

  w.Fetch(0, 0, n);
  for (u64 off = 0; off < n; off += kEpp) {
    auto h = w.HoldPage(off, kEpp, true);
    for (u64 i = off; i < off + kEpp; ++i) h[i] = 100 + i;
  }
  w.Flush(1, 0, n);
  w.UnpinRange(0, n);

  r.Fetch(1, 0, n);
  auto h = r.HoldPage(kEpp, kEpp);
  REQUIRE(h[kEpp] == 100 + kEpp);
  r.UnpinRange(0, n);
}

SIMPLE_TEST_MAIN()
