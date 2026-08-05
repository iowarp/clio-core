/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * IOR-like file-per-process chain benchmark (issue #886 performance).
 *
 * Every node runs this binary (rank = NODE_ID-1). Each rank writes its OWN
 * tag's blobs (file-per-process), then reads them back (IOR read-verify),
 * then reads its neighbor's (cross-node phase). Blob-based barriers between
 * phases. The pool under test comes from CLIO_CTE_POOL (the chain top or
 * the core directly), so one binary measures both configurations.
 *
 * Output (stderr, machine-parsable):
 *   BENCH <phase> rank=<r> mb=<total> secs=<s> mbps=<rate>
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kNumNodes = 4;

int EnvInt(const char *name, int def) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::atoi(v) : def;
}

// Knobs: BENCH_BLOBS (count/rank), BENCH_KB (blob size), BENCH_DEPTH
// (outstanding gets; writes pipeline through the defer registry itself).
const int kBlobsPerRank = EnvInt("BENCH_BLOBS", 64);
const clio::run::u64 kBlobSize = 1024ULL * EnvInt("BENCH_KB", 1024);
const int kDepth = EnvInt("BENCH_DEPTH", 8);

int Rank() {
  const char *env = std::getenv("NODE_ID");
  return env ? std::atoi(env) - 1 : 0;
}

clio::cte::core::Client *Cte() { return CLIO_CTE_CLIENT; }

int g_barrier_epoch = 0;
bool Barrier(const clio::cte::core::TagId &tag_id) {
  const int epoch = g_barrier_epoch++;
  char one = 1;
  {
    std::string mine = "bar_" + std::to_string(epoch) + "_" +
                       std::to_string(Rank());
    auto put = Cte()->AsyncPutBlob(tag_id, mine, 0, 1, &one);
    put.Wait();
    if (put->GetReturnCode() != 0) return false;
  }
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
  for (int r = 0; r < kNumNodes; ++r) {
    std::string name = "bar_" + std::to_string(epoch) + "_" + std::to_string(r);
    while (true) {
      auto sz = Cte()->AsyncGetBlobSize(tag_id, name);
      sz.Wait();
      if (sz->GetReturnCode() == 0 && sz->size_ == 1) break;
      if (std::chrono::steady_clock::now() > deadline) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  return true;
}

double Mbps(clio::run::u64 bytes, double secs) {
  return secs > 0 ? (bytes / (1024.0 * 1024.0)) / secs : 0.0;
}

}  // namespace

int main(int, char **) {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    fprintf(stderr, "CLIO_INIT failed\n");
    return 2;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    fprintf(stderr, "CTE client init failed\n");
    return 2;
  }
  const int rank = Rank();
  fprintf(stderr, "bench: rank %d ready\n", rank);

  clio::cte::core::Tag bar_tag("bench_barrier");
  const clio::cte::core::TagId bar_id = bar_tag.GetTagId();

  // File-per-process: one tag per rank, kBlobsPerRank 1MiB page blobs.
  clio::cte::core::Tag my_tag("bench_file_" + std::to_string(rank));
  const clio::cte::core::TagId my_id = my_tag.GetTagId();
  const int peer = (rank + 1) % kNumNodes;
  clio::cte::core::Tag peer_tag("bench_file_" + std::to_string(peer));
  const clio::cte::core::TagId peer_id = peer_tag.GetTagId();

  std::vector<char> buf(kBlobSize);
  const clio::run::u64 total = kBlobsPerRank * kBlobSize;

  if (!Barrier(bar_id)) return 3;

  // Client-side staging micro-probe: AllocateBuffer + 1MiB memcpy + Free —
  // the fixed per-submit cost of the client-mode defer staging path.
  {
    auto *ipc = CLIO_IPC;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 32; ++i) {
      auto b = ipc->AllocateBuffer(kBlobSize);
      if (b.IsNull()) break;
      std::memcpy(b.ptr_, buf.data(), kBlobSize);
      ipc->FreeBuffer(b);
    }
    double s = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "BENCH staging rank=%d us_per_op=%.0f\n", rank,
            s * 1e6 / 32);
  }

  // Raw private-put submit probe: how long does ONE AsyncPutBlob submit
  // (stage + task + ring send, no wait) take, with nothing else in flight?
  {
    std::vector<clio::run::Future<clio::cte::core::PutBlobTask>> futs;
    std::vector<double> us(16, 0);
    for (int i = 0; i < 16; ++i) {
      auto s0 = std::chrono::steady_clock::now();
      futs.push_back(Cte()->AsyncPutBlob(my_id, "probe_" + std::to_string(i),
                                         0, kBlobSize, buf.data()));
      us[i] = std::chrono::duration<double, std::micro>(
                  std::chrono::steady_clock::now() - s0).count();
    }
    for (auto &f : futs) f.Wait();
    std::sort(us.begin(), us.end());
    fprintf(stderr, "BENCH rawsubmit rank=%d min=%.0f p50=%.0f max=%.0f\n",
            rank, us.front(), us[8], us.back());
  }

  // ---- WRITE own file via the DEFER pipeline ----
  // AsyncPutBlobDefer is THE intended write path (clio-fs, YCSB, lmcache
  // all write through it): it owns a copy of the bytes at submit, batches
  // and pipelines internally, and flow-controls on SHM staging. The timed
  // region includes the full drain — these are DURABLE-acked bytes/sec.
  {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<double> submit_us(kBlobsPerRank, 0);
    for (int i = 0; i < kBlobsPerRank; ++i) {
      std::memset(buf.data(), 'a' + ((rank + i) % 26), kBlobSize);
      auto s0 = std::chrono::steady_clock::now();
      int rc = Cte()->AsyncPutBlobDefer(my_id, "page_" + std::to_string(i), 0,
                                        kBlobSize, buf.data());
      submit_us[i] = std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - s0).count();
      if (rc != 0) {
        fprintf(stderr, "BENCH write submit FAILED rank=%d blob=%d rc=%d\n",
                rank, i, rc);
        return 4;
      }
    }
    {
      std::vector<double> s = submit_us;
      std::sort(s.begin(), s.end());
      fprintf(stderr,
              "BENCH submit_us rank=%d min=%.0f p50=%.0f p90=%.0f max=%.0f\n",
              rank, s.front(), s[s.size() / 2], s[s.size() * 9 / 10],
              s.back());
    }
    double submit_s = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count();
    clio::cte::core::Client::AwaitPutsUntilSpace(0);
    if (clio::cte::core::Client::DeferErrorCount() != 0) {
      fprintf(stderr, "BENCH write FAILED rank=%d defer_errors=%llu\n", rank,
              (unsigned long long)clio::cte::core::Client::DeferErrorCount());
      return 4;
    }
    double s = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr,
            "BENCH write rank=%d mb=%llu secs=%.3f mbps=%.1f kb=%llu "
            "submit_secs=%.3f\n",
            rank, (unsigned long long)(total >> 20), s, Mbps(total, s),
            (unsigned long long)(kBlobSize >> 10), submit_s);
  }
  if (!Barrier(bar_id)) return 3;

  // Windowed DEFER reads (read-your-writes-consistent AsyncGetBlobDefer —
  // the intended read path): kDepth outstanding gets, each with its own
  // destination buffer, verified on completion.
  auto read_phase = [&](const char *label, const clio::cte::core::TagId &tid,
                        int seed_rank, bool verify) -> bool {
    std::vector<std::vector<char>> bufs(kDepth,
                                        std::vector<char>(kBlobSize));
    std::vector<std::pair<clio::run::Future<clio::cte::core::GetBlobTask>,
                          int>> win;
    win.reserve(kDepth);
    auto complete_front = [&]() -> bool {
      auto &f = win.front().first;
      const int idx = win.front().second;
      f.Wait();
      if (f->GetReturnCode() != 0) {
        fprintf(stderr, "BENCH %s FAILED rank=%d blob=%d rc=%u\n", label,
                rank, idx, f->GetReturnCode());
        return false;
      }
      if (verify && bufs[idx % kDepth][0] !=
                        static_cast<char>('a' + ((seed_rank + idx) % 26))) {
        fprintf(stderr, "BENCH %s VERIFY FAILED rank=%d blob=%d\n", label,
                rank, idx);
        return false;
      }
      win.erase(win.begin());
      return true;
    };
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kBlobsPerRank; ++i) {
      if ((int)win.size() == kDepth && !complete_front()) return false;
      win.emplace_back(
          Cte()->AsyncGetBlobDefer(tid, "page_" + std::to_string(i), 0,
                                   kBlobSize, bufs[i % kDepth].data()),
          i);
    }
    while (!win.empty()) {
      if (!complete_front()) return false;
    }
    double s = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr,
            "BENCH %s rank=%d mb=%llu secs=%.3f mbps=%.1f depth=%d\n", label,
            rank, (unsigned long long)(total >> 20), s, Mbps(total, s),
            kDepth);
    return true;
  };

  // ---- READ own file (IOR read-verify) ----
  if (!read_phase("read_own", my_id, rank, /*verify=*/true)) return 5;
  if (!Barrier(bar_id)) return 3;

  // ---- READ own file AGAIN (steady-state cached reads) ----
  if (!read_phase("read_own2", my_id, rank, /*verify=*/false)) return 5;
  if (!Barrier(bar_id)) return 3;

  // ---- READ peer's file (cross-node) ----
  if (!read_phase("read_peer", peer_id, peer, /*verify=*/true)) return 6;
  if (!Barrier(bar_id)) return 3;

  fprintf(stderr, "BENCH done rank=%d\n", rank);
  return 0;
}
