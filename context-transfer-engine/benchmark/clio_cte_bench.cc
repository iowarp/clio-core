/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * CTE Core throughput benchmark (Put / Get / PutGet).
 *
 * Shares its CLI + metrics with clio_redis_bench via bench_common.h so the
 * two are apples-to-apples. See bench_common.h for the full flag list;
 * notable additions: --max-total-blobs (global bounded keyspace split
 * evenly across threads, keys cycle) and --time-limit SECONDS (run for
 * a duration instead of a fixed count).
 */

#include "bench_common.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/introspect/system_info.h>
#include <clio_ctp/util/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono;
using clio_bench::BenchArgs;

namespace {

/** True once the time limit (if any) has elapsed since `start`. */
inline bool TimeUp(const steady_clock::time_point &start, double limit_s) {
  if (limit_s <= 0.0) return false;
  return duration<double>(steady_clock::now() - start).count() >= limit_s;
}

/** blob index for op n: cycle within [0, keyspace) when bounded. */
inline long KeyIndex(long n, clio_bench::u64 keyspace) {
  return keyspace > 0 ? static_cast<long>(n % static_cast<long>(keyspace))
                      : n;
}

}  // namespace

class CTEBenchmark {
 public:
  CTEBenchmark(const BenchArgs &a, std::string node_id)
      : a_(a),
        node_id_(std::move(node_id)),
        per_thread_blobs_(a.PerThreadBlobs()) {}

  bool Run() {
    PrintInfo();
    if (a_.test_case == "Put") return RunGeneric(Mode::kPut);
    if (a_.test_case == "Get") return RunGeneric(Mode::kGet);
    if (a_.test_case == "PutGet") return RunGeneric(Mode::kPutGet);
    HLOG(kError, "Unknown test case: {} (Put|Get|PutGet)", a_.test_case);
    return false;
  }

 private:
  enum class Mode { kPut, kGet, kPutGet };

  void PrintInfo() {
    HLOG(kInfo, "=== CTE Core Benchmark ===");
    HLOG(kInfo, "Node ID: {}  Test: {}  Threads: {}  Depth: {}", node_id_,
         a_.test_case, a_.threads, a_.depth);
    HLOG(kInfo, "I/O size: {}  io-count/thread: {}  max-total-blobs: {} "
                "({}/thread)  time-limit: {}s  query: {}",
         clio_bench::FormatSize(a_.io_size), a_.io_count, a_.max_total_blobs,
         per_thread_blobs_, a_.time_limit_s, a_.query_type);
    HLOG(kInfo, "===========================");
  }

  // PoolQuery flavor each AsyncPutBlob/AsyncGetBlob is issued with.
  // Read once per worker thread; constant for the duration of the run.
  clio::run::PoolQuery MakeQuery() const {
    if (a_.query_type == "dynamic") {
      return clio::run::PoolQuery::Dynamic();
    }
    if (a_.query_type == "direct0") {
      // DirectHash(0) — routing mode is non-Local, so under CLIO_FORCE_NET=1
      // every op takes the loopback ZMQ path even on single-node.
      return clio::run::PoolQuery::DirectHash(0);
    }
    return clio::run::PoolQuery::Local();
  }

  // Number of distinct keys a thread uses (finite pool for Get to read).
  long KeyspaceSize() const {
    if (per_thread_blobs_ > 0) return static_cast<long>(per_thread_blobs_);
    return a_.io_count > 0 ? a_.io_count : 1;
  }

  void Worker(Mode mode, size_t tid, std::atomic<bool> &err,
              std::vector<long long> &times, std::vector<clio_bench::u64> &ops) {
    auto *cte = CLIO_CTE_CLIENT;
    auto put_shm = CLIO_IPC->AllocateBuffer(a_.io_size);
    auto get_shm = CLIO_IPC->AllocateBuffer(a_.io_size);
    std::memset(put_shm.ptr_, static_cast<int>(tid & 0xFF), a_.io_size);
    std::memset(get_shm.ptr_, 0, a_.io_size);  // pre-fault dest pages
    ctp::ipc::ShmPtr<> put_ptr = put_shm.shm_.template Cast<void>();
    ctp::ipc::ShmPtr<> get_ptr = get_shm.shm_.template Cast<void>();

    std::string tag_name = "tag_n" + node_id_ + "_t" + std::to_string(tid);
    auto tag_task = cte->AsyncGetOrCreateTag(tag_name);
    tag_task.Wait();
    clio::cte::core::TagId tag_id = tag_task->tag_id_;
    auto blob_name = [&](long k) {
      return "blob_t" + std::to_string(tid) + "_" + std::to_string(k);
    };
    const clio::run::PoolQuery pq = MakeQuery();

    // Get needs the keyspace populated first (untimed).
    if (mode == Mode::kGet) {
      for (long k = 0; k < KeyspaceSize(); ++k) {
        auto t = cte->AsyncPutBlob(tag_id, blob_name(k), 0, a_.io_size,
                                   put_ptr, 0.8f,
                                   clio::cte::core::Context(), 0, pq);
        t.Wait();
        if (t->return_code_.load() != 0) {
          err.store(true, std::memory_order_relaxed);
          CLIO_IPC->FreeBuffer(put_shm);
          CLIO_IPC->FreeBuffer(get_shm);
          return;
        }
      }
    }

    const bool timed = a_.time_limit_s > 0.0;
    const long target = timed ? std::numeric_limits<long>::max() : a_.io_count;
    clio_bench::u64 done = 0;
    auto start = steady_clock::now();

    for (long i = 0; i < target; i += a_.depth) {
      if (err.load(std::memory_order_relaxed)) break;
      if (timed && TimeUp(start, a_.time_limit_s)) break;
      long batch = timed ? a_.depth : std::min<long>(a_.depth, target - i);

      if (mode == Mode::kPut || mode == Mode::kPutGet) {
        std::vector<clio::run::Future<clio::cte::core::PutBlobTask>> pts;
        pts.reserve(batch);
        for (long j = 0; j < batch; ++j) {
          pts.push_back(cte->AsyncPutBlob(
              tag_id, blob_name(KeyIndex(i + j, per_thread_blobs_)), 0,
              a_.io_size, put_ptr, 0.8f,
              clio::cte::core::Context(), 0, pq));
        }
        for (auto &t : pts) {
          t.Wait();
          if (t->return_code_.load() != 0) {
            HLOG(kError, "[t{}] PutBlob rc={}", tid, t->return_code_.load());
            err.store(true, std::memory_order_relaxed);
          }
        }
      }
      if (mode == Mode::kGet || mode == Mode::kPutGet) {
        for (long j = 0; j < batch; ++j) {
          auto t = cte->AsyncGetBlob(
              tag_id, blob_name(KeyIndex(i + j, per_thread_blobs_)), 0,
              a_.io_size, 0, get_ptr, pq);
          t.Wait();
          if (t->return_code_.load() != 0) {
            HLOG(kError, "[t{}] GetBlob rc={}", tid, t->return_code_.load());
            err.store(true, std::memory_order_relaxed);
          }
        }
      }
      done += static_cast<clio_bench::u64>(batch);
    }

    times[tid] =
        duration_cast<microseconds>(steady_clock::now() - start).count();
    ops[tid] = done;
    CLIO_IPC->FreeBuffer(put_shm);
    CLIO_IPC->FreeBuffer(get_shm);
  }

  bool RunGeneric(Mode mode) {
    if (mode == Mode::kGet) {
      HLOG(kInfo, "Populating {} keys/thread for Get...", KeyspaceSize());
    }
    std::vector<std::thread> threads;
    std::vector<long long> times(a_.threads);
    std::vector<clio_bench::u64> ops(a_.threads);
    std::atomic<bool> err{false};
    for (size_t i = 0; i < a_.threads; ++i) {
      threads.emplace_back(&CTEBenchmark::Worker, this, mode, i,
                           std::ref(err), std::ref(times), std::ref(ops));
    }
    for (auto &t : threads) t.join();
    clio_bench::PrintResults(a_.test_case, a_, times, ops);
    return !err.load();
  }

  BenchArgs a_;
  std::string node_id_;
  clio_bench::u64 per_thread_blobs_;  // a_.max_total_blobs / threads
};

int main(int argc, char **argv) {
  BenchArgs args = clio_bench::ParseBenchArgs(argc, argv);
  if (!args.ok) return 1;

  HLOG(kInfo, "Initializing Clio runtime...");
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Clio runtime");
    return 1;
  }
  struct ClientFinalizeGuard {
    ~ClientFinalizeGuard() {
      auto *mgr = CLIO_RUNTIME_MANAGER;
      if (mgr) mgr->ClientFinalize();
    }
  } finalize_guard;
  std::this_thread::sleep_for(milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    HLOG(kError, "Failed to initialize CTE client");
    return 1;
  }
  std::this_thread::sleep_for(milliseconds(200));

  const char *node_id_env = std::getenv("NODE_ID");
  std::string node_id;
  if (node_id_env && node_id_env[0] != '\0') {
    node_id = node_id_env;
  } else {
    node_id = ctp::SystemInfo::GetHostname();
  }

  CTEBenchmark bench(args, node_id);
  if (!bench.Run()) {
    HLOG(kError, "Benchmark failed: a PutBlob/GetBlob returned non-zero rc");
    return 1;
  }
  return 0;
}
