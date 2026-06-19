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
 * CTE SemanticSearch benchmark.
 *
 * 1. Writes a configurable number of blobs, each a configurable size,
 *    under a single tag. Every blob is filled with the same keyword so a
 *    BM25 search for that keyword matches all of them.
 * 2. Issues ONE broadcast SemanticSearch for that keyword and asks for a
 *    configurable number of top-k results. With a broadcast query every
 *    tag-owning container scores its slice and returns its local top-k;
 *    SemanticSearchTask::AggregateOut then merges those partial sets and keeps
 *    the global top-k by descending BM25 score.
 *
 * On completion it prints an overall retrieval-performance summary — the
 * average / min / max latency to query the top-k blobs — plus a single
 * machine-parseable `[SEM_BENCH] key=value ...` record (consumed by the
 * jarvis package's _get_stat).
 *
 * Flags (all optional):
 *   --blobs N         number of blobs to write   (default 1000)
 *   --size  BYTES     size of each blob in bytes  (default 4096)
 *   --results K       number of results to return (default 10; 0 = all)
 *   --keyword W       keyword stored in / searched for (default "needle")
 *   --query-iters N   repeat the search N times for a stable latency (default 5)
 *
 * Example:
 *   clio_cte_semantic_bench --blobs 5000 --size 8192 --results 20 --query-iters 10
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_ctp/util/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono;

namespace {

/** Format a double with fixed precision — the HLOG formatter only handles
 *  plain "{}" placeholders, not "{:.Nf}" specs. */
std::string F(double v, int prec) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
  return std::string(buf);
}

struct Args {
  long blobs = 1000;
  long size = 4096;
  unsigned results = 10;
  std::string keyword = "needle";
  int query_iters = 5;
  bool ok = true;
};

Args ParseArgs(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    auto next = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        HLOG(kError, "Missing value for {}", name);
        a.ok = false;
        return nullptr;
      }
      return argv[++i];
    };
    if (flag == "--blobs") {
      const char *v = next("--blobs");
      if (v) a.blobs = std::atol(v);
    } else if (flag == "--size") {
      const char *v = next("--size");
      if (v) a.size = std::atol(v);
    } else if (flag == "--results") {
      const char *v = next("--results");
      if (v) a.results = static_cast<unsigned>(std::atol(v));
    } else if (flag == "--keyword") {
      const char *v = next("--keyword");
      if (v) a.keyword = v;
    } else if (flag == "--query-iters") {
      const char *v = next("--query-iters");
      if (v) a.query_iters = std::atoi(v);
    } else {
      HLOG(kError, "Unknown flag: {}", flag);
      a.ok = false;
    }
  }
  if (a.blobs <= 0 || a.size <= 0) {
    HLOG(kError, "--blobs and --size must be > 0");
    a.ok = false;
  }
  if (a.query_iters <= 0) a.query_iters = 1;
  return a;
}

/** Fill `buf` (size bytes) with "<keyword> " repeated so the blob tokenizes
 *  into `keyword` terms that BM25 will match. */
void FillWithKeyword(char *buf, size_t size, const std::string &keyword) {
  std::string token = keyword + " ";
  for (size_t off = 0; off < size;) {
    size_t n = std::min(token.size(), size - off);
    std::memcpy(buf + off, token.data(), n);
    off += n;
  }
}

}  // namespace

int main(int argc, char **argv) {
  Args a = ParseArgs(argc, argv);
  if (!a.ok) return 1;

  HLOG(kInfo, "Initializing Chimaera runtime...");
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Chimaera runtime");
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

  auto *cte = CLIO_CTE_CLIENT;

  // --- 1. One tag, N blobs of the keyword ------------------------------
  const std::string tag_name = "semantic_bench";
  auto tag_task = cte->AsyncGetOrCreateTag(tag_name);
  tag_task.Wait();
  clio::cte::core::TagId tag_id = tag_task->tag_id_;

  // Single reusable SHM payload — every blob has identical keyword content.
  auto put_shm = CLIO_IPC->AllocateBuffer(a.size);
  if (put_shm.IsNull()) {
    HLOG(kError, "AllocateBuffer({}) failed", a.size);
    return 1;
  }
  FillWithKeyword(reinterpret_cast<char *>(put_shm.ptr_),
                  static_cast<size_t>(a.size), a.keyword);
  ctp::ipc::ShmPtr<> put_ptr = put_shm.shm_.template Cast<void>();

  HLOG(kInfo, "Writing {} blobs x {} bytes under tag '{}' (keyword='{}')...",
       a.blobs, a.size, tag_name, a.keyword);
  auto put_t0 = steady_clock::now();
  bool put_err = false;
  for (long k = 0; k < a.blobs; ++k) {
    std::string blob_name = "blob_" + std::to_string(k);
    auto t = cte->AsyncPutBlob(tag_id, blob_name.c_str(), 0,
                               static_cast<chi::u64>(a.size), put_ptr, 1.0f,
                               clio::cte::core::Context(), 0);
    t.Wait();
    if (t->return_code_.load() != 0) {
      HLOG(kError, "PutBlob '{}' rc={}", blob_name, t->return_code_.load());
      put_err = true;
      break;
    }
  }
  double put_s = duration<double>(steady_clock::now() - put_t0).count();
  CLIO_IPC->FreeBuffer(put_shm);
  if (put_err) return 1;
  HLOG(kInfo, "Ingest: {} blobs in {}s ({} blobs/s)", a.blobs, F(put_s, 3),
       F(a.blobs / (put_s > 0 ? put_s : 1), 0));

  // --- 2. Broadcast semantic search, repeated for a stable latency -----
  HLOG(kInfo, "SemanticSearch keyword='{}' k={} (broadcast), {} iters...",
       a.keyword, a.results, a.query_iters);
  double q_sum = 0.0, q_min = 1e30, q_max = 0.0;
  size_t last_count = 0;
  std::vector<clio::cte::core::SemanticSearchResult> last_results;
  for (int it = 0; it < a.query_iters; ++it) {
    auto q_t0 = steady_clock::now();
    auto search = cte->AsyncSemanticSearch(
        tag_name, ".*", a.keyword, a.results, chi::PoolQuery::Broadcast());
    search.Wait();
    double q_s = duration<double>(steady_clock::now() - q_t0).count();
    if (search->return_code_.load() != 0) {
      HLOG(kError, "SemanticSearch rc={}", search->return_code_.load());
      return 1;
    }
    q_sum += q_s;
    q_min = std::min(q_min, q_s);
    q_max = std::max(q_max, q_s);
    last_count = search->results_.size();
    last_results.assign(search->results_.begin(), search->results_.end());
  }
  double q_avg_us = (q_sum / a.query_iters) * 1e6;
  double q_min_us = q_min * 1e6;
  double q_max_us = q_max * 1e6;

  // Show a few top hits from the last query.
  size_t show = std::min<size_t>(last_results.size(), 5);
  for (size_t i = 0; i < show; ++i) {
    HLOG(kInfo, "  [{}] tag='{}' blob='{}' score={}", i,
         last_results[i].tag_name_, last_results[i].blob_name_,
         F(last_results[i].score_, 4));
  }

  // Overall retrieval-performance summary. The [SEM_BENCH] line is the
  // machine-parseable record consumed by jarvis _get_stat.
  HLOG(kInfo,
       "Retrieval: top-{} query over {} blobs -> {} results; "
       "latency avg={} us, min={} us, max={} us ({} iters)",
       a.results, a.blobs, last_count, F(q_avg_us, 1), F(q_min_us, 1),
       F(q_max_us, 1), a.query_iters);
  HLOG(kInfo,
       "[SEM_BENCH] query_avg_us={} query_min_us={} query_max_us={} "
       "results={} k={} blobs={} blob_size={} query_iters={} "
       "ingest_s={} ingest_blobs_per_s={}",
       F(q_avg_us, 1), F(q_min_us, 1), F(q_max_us, 1), last_count, a.results,
       a.blobs, a.size, a.query_iters, F(put_s, 3),
       F(a.blobs / (put_s > 0 ? put_s : 1), 0));
  return 0;
}
