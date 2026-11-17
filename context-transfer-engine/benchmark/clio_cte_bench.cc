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
    if (a_.test_case == "PutSync") return RunGeneric(Mode::kPutSync);
    if (a_.test_case == "GetSync") return RunGeneric(Mode::kGetSync);
    if (a_.test_case == "PutGetSync") return RunGeneric(Mode::kPutGetSync);
    if (a_.test_case == "PutStreamSync") {
      return RunGeneric(Mode::kPutStreamSync);
    }
    HLOG(kError,
         "Unknown test case: {} "
         "(Put|Get|PutGet|PutSync|GetSync|PutGetSync|PutStreamSync)",
         a_.test_case);
    return false;
  }

 private:
  enum class Mode {
    kPut,
    kGet,
    kPutGet,
    kPutSync,
    kGetSync,
    kPutGetSync,
    kPutStreamSync
  };

  // PutStreamSync (issue #1007): each thread STREAMS io_size writes at
  // sequentially growing offsets through 16 MiB blob segments -- the small-
  // sustained-write shape -- instead of whole-value puts to rotating keys.
  static constexpr clio::run::u64 kStreamBlobBytes = 16 * 1024 * 1024;

  /**
   * Whether a mode runs the synchronous client API (Client::PutBlob /
   * GetBlob: the shm cache path when connected, else the task path).
   * @param m the mode
   * @return true for the *Sync modes
   */
  static bool IsSyncMode(Mode m) {
    return m == Mode::kPutSync || m == Mode::kGetSync ||
           m == Mode::kPutGetSync || m == Mode::kPutStreamSync;
  }

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

  /**
   * Placement score every task-path (non-Sync) bench put carries. CLIO_BENCH_PUT_SCORE
   * overrides the historical 0.8; -1 (auto) is what the shm write-through
   * cache absorbs, an explicit score takes its task path.
   * @return the score, read once per process
   */
  static float PutScore() {
    static const float v = [] {
      const char *e = std::getenv("CLIO_BENCH_PUT_SCORE");
      return (e != nullptr && *e) ? std::strtof(e, nullptr) : 0.8f;
    }();
    return v;
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
    // One destination REGION PER IN-FLIGHT GET. With a single shared buffer
    // the Get loop had to Wait() each op before issuing the next (concurrent
    // reads would race on the destination), which silently serialized Gets and
    // made --depth a no-op for reads: Get d64 measured ~= Get d1 while the
    // properly-pipelined Puts scaled 5x. Per-slot regions let Gets batch
    // exactly like Puts.
    const size_t get_slots = a_.depth > 0 ? static_cast<size_t>(a_.depth) : 1;
    auto get_shm = CLIO_IPC->AllocateBuffer(a_.io_size * get_slots);
    std::memset(put_shm.ptr_, static_cast<int>(tid & 0xFF), a_.io_size);
    std::memset(get_shm.ptr_, 0, a_.io_size * get_slots);  // pre-fault dest pages
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
                                   put_ptr, PutScore(),
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
              a_.io_size, put_ptr, PutScore(),
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
        // Pipeline exactly like the Put arm: submit `batch` async Gets (each
        // into its OWN destination slot), then drain. This is what makes
        // --depth meaningful for reads.
        std::vector<clio::run::Future<clio::cte::core::GetBlobTask>> gts;
        gts.reserve(batch);
        for (long j = 0; j < batch; ++j) {
          ctp::ipc::ShmPtr<> slot =
              get_ptr + static_cast<size_t>(j) * a_.io_size;
          gts.push_back(cte->AsyncGetBlob(
              tag_id, blob_name(KeyIndex(i + j, per_thread_blobs_)), 0,
              a_.io_size, 0, slot, pq));
        }
        for (auto &t : gts) {
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

  /**
   * Synchronous-API worker: PutSync/GetSync/PutGetSync/PutStreamSync issue
   * Client::PutBlob / GetBlob from PRIVATE buffers. With the shm cache
   * connected (CLIO_CTE_CLIENT_INIT connects it when the compose has one) a
   * put is a copy into the cache segment with no task; otherwise every op is
   * a task the call waits on. Puts carry no score and no context, which is
   * what the cache path requires. PutGetSync alternates put/get per key.
   */
  void WorkerSync(Mode mode, size_t tid, std::atomic<bool> &err,
                  std::vector<long long> &times,
                  std::vector<clio_bench::u64> &ops) {
    auto *cte = CLIO_CTE_CLIENT;
    std::vector<char> put_buf(a_.io_size, static_cast<char>(tid & 0xFF));
    std::vector<char> get_buf(a_.io_size, 0);
    std::string tag_name = "tag_n" + node_id_ + "_t" + std::to_string(tid);
    clio::cte::core::TagId tag_id;
    if (cte->GetOrCreateTag(tag_name, &tag_id) != 0) {
      err.store(true, std::memory_order_relaxed);
      return;
    }
    auto blob_name = [&](long k) {
      return "blob_t" + std::to_string(tid) + "_" + std::to_string(k);
    };
    const clio::run::PoolQuery pq = MakeQuery();
    auto check = [&](clio::run::u32 rc, const char *what) {
      if (rc != 0) {
        HLOG(kError, "[t{}] {} rc={}", tid, what, rc);
        err.store(true, std::memory_order_relaxed);
      }
    };
    if (mode == Mode::kGetSync) {  // populate the keyspace (untimed)
      for (long k = 0; k < KeyspaceSize(); ++k) {
        check(cte->PutBlob(tag_id, blob_name(k), 0, a_.io_size,
                           put_buf.data(), -1.0f, nullptr, 0, pq),
              "PutBlob");
      }
    }
    const bool timed = a_.time_limit_s > 0.0;
    const long target = timed ? std::numeric_limits<long>::max() : a_.io_count;
    clio_bench::u64 done = 0;
    auto start = steady_clock::now();
    for (long i = 0; i < target; ++i) {
      if (err.load(std::memory_order_relaxed)) break;
      if (timed && TimeUp(start, a_.time_limit_s)) break;
      const std::string name = blob_name(KeyIndex(i, per_thread_blobs_));
      if (mode == Mode::kPutSync || mode == Mode::kPutGetSync) {
        check(cte->PutBlob(tag_id, name, 0, a_.io_size, put_buf.data(), -1.0f,
                           nullptr, 0, pq),
              "PutBlob");
      }
      if (mode == Mode::kPutStreamSync) {
        const clio::run::u64 pos = static_cast<clio::run::u64>(i) * a_.io_size;
        check(cte->PutBlob(tag_id,
                           blob_name(static_cast<long>(pos / kStreamBlobBytes)) +
                               "_stream",
                           pos % kStreamBlobBytes, a_.io_size, put_buf.data(),
                           -1.0f, nullptr, 0, pq),
              "PutBlob(stream)");
      }
      if (mode == Mode::kGetSync || mode == Mode::kPutGetSync) {
        check(cte->GetBlob(tag_id, name, 0, a_.io_size, get_buf.data(), 0,
                           nullptr, pq),
              "GetBlob");
      }
      ++done;
    }
    times[tid] =
        duration_cast<microseconds>(steady_clock::now() - start).count();
    ops[tid] = done;
  }

  bool RunGeneric(Mode mode) {
    if (mode == Mode::kGet || mode == Mode::kGetSync) {
      HLOG(kInfo, "Populating {} keys/thread for Get...", KeyspaceSize());
    }
    std::vector<std::thread> threads;
    std::vector<long long> times(a_.threads);
    std::vector<clio_bench::u64> ops(a_.threads);
    std::atomic<bool> err{false};
    for (size_t i = 0; i < a_.threads; ++i) {
      if (IsSyncMode(mode)) {
        threads.emplace_back(&CTEBenchmark::WorkerSync, this, mode, i,
                             std::ref(err), std::ref(times), std::ref(ops));
      } else {
        threads.emplace_back(&CTEBenchmark::Worker, this, mode, i,
                             std::ref(err), std::ref(times), std::ref(ops));
      }
    }
    for (auto &t : threads) t.join();
    clio_bench::PrintResults(a_.test_case, a_, times, ops);
    PrintCacheStats();
    return !err.load();
  }

  /** Node-wide shm cache counters, when this client is connected to it. */
  static void PrintCacheStats() {
    auto *cte = CLIO_CTE_CLIENT;
    const auto *s = cte != nullptr ? cte->CacheStats() : nullptr;
    if (s == nullptr) return;
    std::printf("WTC stats: puts=%llu hits=%llu misses=%llu fills=%llu "
                "evictions=%llu waits=%llu flush_batches=%llu "
                "flush_ranges=%llu flush_skipped=%llu miss_absent=%llu "
                "miss_nonres=%llu miss_range=%llu miss_busy=%llu "
                "invalidations=%llu\n",
                (unsigned long long)s->puts_.load(),
                (unsigned long long)s->get_hits_.load(),
                (unsigned long long)s->get_misses_.load(),
                (unsigned long long)s->fills_.load(),
                (unsigned long long)s->evictions_.load(),
                (unsigned long long)s->waits_.load(),
                (unsigned long long)s->flush_batches_.load(),
                (unsigned long long)s->flush_ranges_.load(),
                (unsigned long long)s->flush_skipped_.load(),
                (unsigned long long)s->miss_absent_.load(),
                (unsigned long long)s->miss_nonres_.load(),
                (unsigned long long)s->miss_range_.load(),
                (unsigned long long)s->miss_busy_.load(),
                (unsigned long long)s->invalidations_.load());
  }

  BenchArgs a_;
  std::string node_id_;
  clio_bench::u64 per_thread_blobs_;  // a_.max_total_blobs / threads
};

int main(int argc, char **argv) {
  BenchArgs args = clio_bench::ParseBenchArgs(argc, argv);
  if (!args.ok) return 1;

  HLOG(kInfo, "Initializing Clio runtime...");
  // CLIO_BENCH_SELF_RUN=1 self-launches the runtime IN-PROCESS (embedded
  // mode) instead of attaching to an external clio_run — so both runtime
  // modes can be measured from the same binary.
  const char *self_run = std::getenv("CLIO_BENCH_SELF_RUN");
  const bool embed = self_run != nullptr && self_run[0] == '1';
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, embed)) {
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
