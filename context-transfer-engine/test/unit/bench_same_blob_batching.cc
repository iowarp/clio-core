/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

// Benchmark for worker-local task batching (issue #820).
//
// Measures the exact contention batching exists to remove: many small writes
// aimed at ONE blob. The filesystem stores each 1 MiB page as one blob, so a
// sequential 4 KiB workload sends 256 tasks at a single page-blob, and each has
// to hold that blob's #680 write token across its whole body -- allocation AND
// the bdev copy. They drain single-file, and the contender re-polls on a
// periodic cadence, which is where the measured ~1 s tail comes from.
//
// The A/B is on the SAME BINARY via CLIO_TASK_BATCHING, which is the point:
// no rebuild, no branch switch, no host drift between arms. Whatever differs is
// the batching phase.
//
// Reports mean AND the tail (p50/p99/p99.9/max), because the tail is the thing
// being attacked -- a mean-only comparison would hide it.
//
// Env knobs:
//   BATCH_BENCH_THREADS (8)      concurrent writers (all at the same blob)
//   BATCH_BENCH_ITERS   (2000)   writes per thread
//   BATCH_BENCH_CHUNK   (4096)   bytes per write

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace {

const char *kTargetName = "batch_bench_target";
constexpr clio::run::u64 kTargetSize = 2ULL * 1024 * 1024 * 1024;  // 2 GiB

int FromEnv(const char *name, int dflt) {
  if (const char *e = std::getenv(name)) {
    int n = std::atoi(e);
    if (n > 0) return n;
  }
  return dflt;
}

double NowUs() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

class Fixture {
 public:
  bool initialized_ = false;
  Fixture() {
    bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    ok = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto *cte = CLIO_CTE_CLIENT;
    clio::run::PoolId bdev_pool_id(921, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto create = bdev_client.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), kTargetName, bdev_pool_id,
        clio::run::bdev::BdevType::kRam, kTargetSize);
    create.Wait();
    auto reg = cte->AsyncRegisterTarget(kTargetName,
                                        clio::run::bdev::BdevType::kRam,
                                        kTargetSize, clio::run::PoolQuery::Local(),
                                        bdev_pool_id);
    reg.Wait();
    REQUIRE(reg->GetReturnCode() == 0);
    initialized_ = true;
  }
};

Fixture *g_fixture = nullptr;

double Pct(std::vector<double> &v, double p) {
  if (v.empty()) return 0.0;
  size_t i = static_cast<size_t>(p * (v.size() - 1));
  return v[i];
}

}  // namespace

TEST_CASE("Batching bench: many small writes at ONE blob", "[cte][bench][820]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  const int kThreads = FromEnv("BATCH_BENCH_THREADS", 8);
  const int kIters = FromEnv("BATCH_BENCH_ITERS", 2000);
  const size_t kChunk = static_cast<size_t>(FromEnv("BATCH_BENCH_CHUNK", 4096));

  clio::cte::core::Tag tag("batch_bench_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string blob = "hot_page";

  std::vector<std::vector<double>> lat(kThreads);
  std::atomic<int> errors{0};

  double t0 = NowUs();
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      auto *ipc = CLIO_IPC;
      auto *cte = CLIO_CTE_CLIENT;
      ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kChunk);
      if (buf.IsNull()) { errors++; return; }
      std::memset(buf.ptr_, 'a' + (t % 26), kChunk);
      lat[t].reserve(kIters);
      for (int i = 0; i < kIters; ++i) {
        // Distinct, disjoint offsets -- all landing on the SAME blob, which is
        // exactly the page-blob contention the fs produces.
        clio::run::u64 off =
            (static_cast<clio::run::u64>(i) * kThreads + t) * kChunk;
        double s = NowUs();
        auto p = cte->AsyncPutBlob(tag_id, blob, off, kChunk,
                                   ctp::ipc::ShmPtr<>(buf.shm_));
        p.Wait();
        lat[t].push_back(NowUs() - s);
        if (p->GetReturnCode() != 0) errors++;
      }
      ipc->FreeBuffer(buf);
    });
  }
  for (auto &th : threads) th.join();
  double elapsed_us = NowUs() - t0;

  std::vector<double> all;
  for (auto &v : lat) all.insert(all.end(), v.begin(), v.end());
  std::sort(all.begin(), all.end());
  double sum = 0;
  for (double x : all) sum += x;

  const char *mode = std::getenv("CLIO_TASK_BATCHING");
  std::printf(
      "[#820 BENCH] batching=%s threads=%d iters=%d chunk=%zu\n"
      "[#820 BENCH]   ops=%zu elapsed=%.1f ms  throughput=%.0f ops/s\n"
      "[#820 BENCH]   mean=%.2f us  p50=%.2f  p99=%.2f  p99.9=%.2f  max=%.2f us\n"
      "[#820 BENCH]   errors=%d\n",
      (mode == nullptr ? "on(default)" : mode), kThreads, kIters, kChunk,
      all.size(), elapsed_us / 1000.0,
      all.empty() ? 0.0 : all.size() / (elapsed_us / 1e6),
      all.empty() ? 0.0 : sum / all.size(), Pct(all, 0.50), Pct(all, 0.99),
      Pct(all, 0.999), all.empty() ? 0.0 : all.back(), errors.load());

  REQUIRE(errors.load() == 0);
}

TEST_CASE("Async burst at ONE blob (batching A/B)", "[cte][bench][820]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *ipc = CLIO_IPC;
  auto *cte = CLIO_CTE_CLIENT;

  // The workload batching actually needs: MANY OUTSTANDING requests to one
  // blob. A synchronous client has exactly one in flight per thread, so no
  // worker ever sees two same-blob requests in a single shard drain and every
  // batch group is a group of one. Here the submitter fires a whole burst
  // before waiting on any of it, which is what #817's async writes produce
  // (write(2) returns as soon as the bytes are staged).
  const size_t kChunk = static_cast<size_t>(FromEnv("BATCH_BENCH_CHUNK", 4096));
  const int kBurst = FromEnv("BATCH_BENCH_BURST", 256);
  const int kRounds = FromEnv("BATCH_BENCH_ROUNDS", 10);

  clio::cte::core::Tag tag("async_burst_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  std::vector<ctp::ipc::FullPtr<char>> bufs;
  for (int i = 0; i < kBurst; ++i) {
    auto b = ipc->AllocateBuffer(kChunk);
    REQUIRE(!b.IsNull());
    std::memset(b.ptr_, 'a' + (i % 26), kChunk);
    bufs.push_back(b);
  }

  std::vector<double> round_ms;
  for (int r = 0; r < kRounds + 1; ++r) {  // round 0 warms the blob
    const std::string blob = "burst_" + std::to_string(r);
    // Warm: lay the blob down once so the timed pass is an overwrite, not a
    // first-touch allocation.
    for (int i = 0; i < kBurst; ++i) {
      auto p = cte->AsyncPutBlob(tag_id, blob,
                                 static_cast<clio::run::u64>(i) * kChunk, kChunk,
                                 ctp::ipc::ShmPtr<>(bufs[i].shm_));
      p.Wait();
    }
    double t0 = NowUs();
    std::vector<clio::run::Future<clio::cte::core::PutBlobTask>> futs;
    futs.reserve(kBurst);
    for (int i = 0; i < kBurst; ++i) {
      futs.push_back(cte->AsyncPutBlob(tag_id, blob,
                                       static_cast<clio::run::u64>(i) * kChunk,
                                       kChunk,
                                       ctp::ipc::ShmPtr<>(bufs[i].shm_)));
    }
    for (auto &f : futs) f.Wait();
    double el = (NowUs() - t0) / 1000.0;
    for (auto &f : futs) REQUIRE(f->GetReturnCode() == 0);
    if (r > 0) round_ms.push_back(el);
  }
  for (auto &b : bufs) ipc->FreeBuffer(b);

  std::sort(round_ms.begin(), round_ms.end());
  const char *mode = std::getenv("CLIO_CTE_BATCHING");
  std::printf(
      "\n[#820 BURST] cte_batching=%s burst=%d x %zu B to ONE blob, %d rounds\n"
      "[#820 BURST]   median=%.3f ms  min=%.3f  max=%.3f\n",
      (mode == nullptr ? "off(default)" : mode), kBurst, kChunk, kRounds,
      round_ms[round_ms.size() / 2], round_ms.front(), round_ms.back());
}

TEST_CASE("Vectored vs N single puts at ONE blob", "[cte][bench][820]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *ipc = CLIO_IPC;
  auto *cte = CLIO_CTE_CLIENT;

  // The measurement that isolates the mechanism this whole issue rests on:
  // the SAME bytes written to the SAME blob, as N separate single-region tasks
  // (N write-token acquires, N metadata mutations, N bdev passes) versus ONE
  // vectored task (one of each). No batching phase involved, no concurrency, no
  // host-drift between arms -- just the task shape.
  const size_t kChunk = static_cast<size_t>(FromEnv("BATCH_BENCH_CHUNK", 4096));
  const int kSegs = FromEnv("BATCH_BENCH_SEGS", 256);   // 256 x 4 KiB = a page
  const int kRounds = FromEnv("BATCH_BENCH_ROUNDS", 20);

  clio::cte::core::Tag tag("vec_vs_single_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  std::vector<ctp::ipc::FullPtr<char>> bufs;
  for (int i = 0; i < kSegs; ++i) {
    auto b = ipc->AllocateBuffer(kChunk);
    REQUIRE(!b.IsNull());
    std::memset(b.ptr_, 'a' + (i % 26), kChunk);
    bufs.push_back(b);
  }

  std::vector<double> single_ms, vec_ms;
  for (int r = 0; r < kRounds; ++r) {
    // Warm both blobs first so neither arm pays first-touch allocation that the
    // other does not -- timing a fresh blob against a warm one is the classic
    // way to manufacture a difference that is not there.
    const std::string blob_s = "single_" + std::to_string(r);
    const std::string blob_v = "vector_" + std::to_string(r);
    for (int pass = 0; pass < 2; ++pass) {
      double t0 = NowUs();
      for (int i = 0; i < kSegs; ++i) {
        auto p = cte->AsyncPutBlob(tag_id, blob_s,
                                   static_cast<clio::run::u64>(i) * kChunk,
                                   kChunk, ctp::ipc::ShmPtr<>(bufs[i].shm_));
        p.Wait();
      }
      double t1 = NowUs();
      std::vector<clio::cte::core::BlobSegment> segs;
      for (int i = 0; i < kSegs; ++i) {
        segs.push_back(clio::cte::core::BlobSegment(
            static_cast<clio::run::u64>(i) * kChunk, kChunk,
            ctp::ipc::ShmPtr<>(bufs[i].shm_)));
      }
      auto pv = cte->AsyncPutBlobVectored(tag_id, blob_v, segs);
      pv.Wait();
      double t2 = NowUs();
      REQUIRE(pv->GetReturnCode() == 0);
      if (pass == 1) {  // pass 0 is the warm-up
        single_ms.push_back((t1 - t0) / 1000.0);
        vec_ms.push_back((t2 - t1) / 1000.0);
      }
    }
  }
  for (auto &b : bufs) ipc->FreeBuffer(b);

  std::sort(single_ms.begin(), single_ms.end());
  std::sort(vec_ms.begin(), vec_ms.end());
  double smed = single_ms[single_ms.size() / 2];
  double vmed = vec_ms[vec_ms.size() / 2];
  std::printf(
      "\n[#820 BENCH] %d x %zu B to ONE blob, %d rounds (medians):\n"
      "[#820 BENCH]   %d single PutBlobs : %.3f ms  (range %.3f-%.3f)\n"
      "[#820 BENCH]   1 vectored PutBlob : %.3f ms  (range %.3f-%.3f)\n"
      "[#820 BENCH]   speedup: %.2fx\n",
      kSegs, kChunk, kRounds, kSegs, smed, single_ms.front(), single_ms.back(),
      vmed, vec_ms.front(), vec_ms.back(), smed / vmed);
}

int main(int argc, char **argv) {
  g_fixture = new Fixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int rc = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  return rc;
}
