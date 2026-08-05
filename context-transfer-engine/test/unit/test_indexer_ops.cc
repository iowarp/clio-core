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
 * Indexer chimod behavioral coverage (issue #905): scope enforcement,
 * asynchronous drain + search barrier, overwrite coalescing, DelBlob /
 * DelTag / RenameTag / TruncateBlob index maintenance, the kReindexScan
 * verb, and the per-tag first-insertion backfill. Composes core(512) +
 * indexer(564, scope tag_re "idx_.*") in an embedded runtime; one core
 * client is bound THROUGH the indexer (564) and one directly at the core
 * (512) to stage pre-existing data the indexer has never seen.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/indexer/indexer_client.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;

static std::string chi_test_data_dir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

class IndexerOpsFixture {
 public:
  static inline bool g_initialized = false;
  std::string config_path_;
  std::string restart_log_path_;

  IndexerOpsFixture() {
    if (g_initialized) return;
    config_path_ = chi_test_data_dir() + "/indexer_ops_config.yaml";
    restart_log_path_ = chi_test_data_dir() + "/indexer_ops_restart.bin";
    // Stale persisted index state from an earlier run would RESTORE into
    // this boot and pollute the assertions — start clean.
    fs::remove(IndexLogPath());
    fs::remove(IndexLogPath() + ".wal");
    CreateConfigFile();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    // Hermetic pool set (same discipline as the cache interpose test).
    ctp::SystemInfo::Setenv("CLIO_RESTART_LOG", restart_log_path_.c_str(), 1);

    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    g_initialized = true;
  }

  static std::string IndexLogPath() {
    return chi_test_data_dir() + "/indexer_ops_index.log";
  }

  void CreateConfigFile() {
    std::ofstream f(config_path_);
    REQUIRE(f.is_open());
    // index_log_path exercises the WAL appends on every mutation; the tiny
    // compact threshold makes the periodic sweep fold the WAL into a
    // snapshot mid-test (SnapshotIndex coverage).
    f << R"(
runtime:
  num_threads: 4
  queue_depth: 1024

compose:
  - mod_name: clio_cte_core
    pool_name: clio_cte
    pool_query: local
    pool_id: 512.0
    storage:
      - path: "ram::indexer_ops_ram"
        bdev_type: "ram"
        capacity_limit: "256MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"

  - mod_name: clio_cte_indexer
    pool_name: clio_cte_indexer
    pool_query: local
    pool_id: 564.0
    next_pool_id: "512.0"
    tag_re: "idx_.*"
    index_sweep_period_ms: 50
    index_log_path: ")" << IndexLogPath() << R"("
    index_wal_compact_bytes: "2KB"
)";
  }
};

namespace {

/** Put `body` as blob `name` under `tag_id` through `cte`. */
void PutText(clio::cte::core::Client &cte, const clio::cte::core::TagId &tag,
             const std::string &name, const std::string &body) {
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(body.size());
  REQUIRE(!buf.IsNull());
  memcpy(buf.ptr_, body.data(), body.size());
  auto put = cte.AsyncPutBlob(tag, name, 0, body.size(),
                              ctp::ipc::ShmPtr<>(buf.shm_), 1.0f);
  put.Wait();
  REQUIRE(put->GetReturnCode() == 0);
  ipc->FreeBuffer(buf);
}

/** Search through the indexer pool; returns blob names in rank order. */
std::vector<std::string> Search(clio::cte::core::Client &cte,
                                const std::string &query,
                                const std::string &tag_re = ".*",
                                const std::string &blob_re = ".*") {
  auto task = cte.AsyncSemanticSearch(tag_re, blob_re, query, 0,
                                      clio::run::PoolQuery::Local());
  task.Wait();
  REQUIRE(task->GetReturnCode() == 0);
  std::vector<std::string> names;
  for (const auto &r : task->results_) names.push_back(r.blob_name_);
  return names;
}

bool Contains(const std::vector<std::string> &v, const std::string &s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}

}  // namespace

TEST_CASE("Indexer - scope, drain, and index maintenance verbs",
          "[cte][indexer]") {
  IndexerOpsFixture fixture;

  // Clients: THROUGH the indexer (564) and DIRECT to the core (512).
  clio::cte::core::Client via_idx(clio::run::PoolId(564, 0));
  clio::cte::core::Client direct(clio::run::PoolId(512, 0));
  clio::cte::indexer::Client idx;
  idx.indexer_pool_id_ = clio::run::PoolId(564, 0);

  auto tag_of = [&](clio::cte::core::Client &c, const std::string &name) {
    auto t = c.AsyncGetOrCreateTag(name);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    return t->tag_id_;
  };

  // --- A SemanticSearch reaching the BARE core (no indexer above it) must
  // fail loudly (rc=2), not silently return empty — the miscomposed-
  // deployment guard in core_lib_exec.
  {
    auto task = direct.AsyncSemanticSearch(".*", ".*", "anything", 0,
                                           clio::run::PoolQuery::Local());
    task.Wait();
    REQUIRE(task->GetReturnCode() == 2);
    REQUIRE(task->results_.empty());
  }

  // --- Scope: only tags matching "idx_.*" are tokenized -------------------
  auto tag1 = tag_of(via_idx, "idx_tag1");
  auto tag_out = tag_of(via_idx, "other_tag");
  PutText(via_idx, tag1, "doc_a", "apollo lunar module descended to the moon");
  PutText(via_idx, tag_out, "doc_x", "apollo apollo apollo out of scope text");

  auto hits = Search(via_idx, "apollo lunar");
  REQUIRE(Contains(hits, "doc_a"));
  REQUIRE(!Contains(hits, "doc_x"));  // out-of-scope tag never indexed

  // --- Overwrite: index converges to the LATEST content (coalesced) -------
  // Matched docs are always returned (score 0 when no term hits — the
  // census semantics), so assert on SCORES: the old vocabulary must stop
  // scoring and the new one must score.
  PutText(via_idx, tag1, "doc_a", "gardening tomatoes need full sunshine");
  {
    auto task = via_idx.AsyncSemanticSearch(".*", "^doc_a$", "apollo lunar",
                                            0, clio::run::PoolQuery::Local());
    task.Wait();
    REQUIRE(task->GetReturnCode() == 0);
    REQUIRE(!task->results_.empty());
    REQUIRE(task->results_[0].score_ == 0.0);

    auto task2 = via_idx.AsyncSemanticSearch(".*", "^doc_a$",
                                             "tomatoes sunshine", 0,
                                             clio::run::PoolQuery::Local());
    task2.Wait();
    REQUIRE(task2->GetReturnCode() == 0);
    REQUIRE(!task2->results_.empty());
    REQUIRE(task2->results_[0].score_ > 0.0);
  }

  // --- RenameTag: result rows carry the new tag name ----------------------
  {
    auto rn = via_idx.AsyncRenameTag("idx_tag1", "idx_tag1_renamed", tag1);
    rn.Wait();
    REQUIRE(rn->GetReturnCode() == 0);
    auto task = via_idx.AsyncSemanticSearch("idx_tag1_renamed", ".*",
                                            "tomatoes", 0,
                                            clio::run::PoolQuery::Local());
    task.Wait();
    REQUIRE(task->GetReturnCode() == 0);
    REQUIRE(!task->results_.empty());
    REQUIRE(task->results_[0].tag_name_ == "idx_tag1_renamed");
  }

  // --- TruncateBlob to 0: the doc leaves the index ------------------------
  {
    auto tr = via_idx.AsyncTruncateBlob(tag1, "doc_a", 0);
    tr.Wait();
    REQUIRE(tr->GetReturnCode() == 0);
    hits = Search(via_idx, "tomatoes sunshine");
    REQUIRE(!Contains(hits, "doc_a"));
  }

  // --- DelBlob: removed from results --------------------------------------
  PutText(via_idx, tag1, "doc_b", "steam locomotives crossed the prairie");
  hits = Search(via_idx, "locomotives prairie");
  REQUIRE(Contains(hits, "doc_b"));
  {
    auto del = via_idx.AsyncDelBlob(tag1, "doc_b");
    del.Wait();
    REQUIRE(del->GetReturnCode() == 0);
    hits = Search(via_idx, "locomotives prairie");
    REQUIRE(!Contains(hits, "doc_b"));
  }

  // --- kReindexScan: pre-existing data the indexer never saw --------------
  // Stage blobs DIRECTLY at the core (bypassing the indexer entirely).
  auto tag_scan = tag_of(direct, "idx_scan");
  PutText(direct, tag_scan, "scan_1", "volcanic basalt columns by the sea");
  PutText(direct, tag_scan, "scan_2", "glacier carved fjords in the north");
  hits = Search(via_idx, "basalt fjords glacier");
  REQUIRE(!Contains(hits, "scan_1"));  // never flowed through the indexer
  {
    auto scan = idx.AsyncReindexScan("^idx_scan$", ".*",
                                     clio::run::PoolQuery::Local());
    scan.Wait();
    REQUIRE(scan->GetReturnCode() == 0);
    REQUIRE(scan->blobs_enqueued_ >= 2);
  }
  hits = Search(via_idx, "basalt columns");
  REQUIRE(Contains(hits, "scan_1"));
  hits = Search(via_idx, "glacier fjords");
  REQUIRE(Contains(hits, "scan_2"));

  // --- First-insertion backfill: one write indexes the tag's past ---------
  auto tag_bf = tag_of(direct, "idx_backfill");
  PutText(direct, tag_bf, "bf_old", "ancient manuscripts in the archive");
  // First write THROUGH the indexer for this tag triggers the backfill.
  PutText(via_idx, tag_bf, "bf_new", "fresh newspaper headlines today");
  hits = Search(via_idx, "ancient manuscripts archive");
  REQUIRE(Contains(hits, "bf_old"));  // backfilled without any explicit scan
  hits = Search(via_idx, "newspaper headlines");
  REQUIRE(Contains(hits, "bf_new"));

  // --- DelTag: every doc of the tag leaves the index ----------------------
  {
    auto del = via_idx.AsyncDelTag("idx_backfill");
    del.Wait();
    REQUIRE(del->GetReturnCode() == 0);
    hits = Search(via_idx, "ancient manuscripts newspaper");
    REQUIRE(!Contains(hits, "bf_old"));
    REQUIRE(!Contains(hits, "bf_new"));
  }

  // --- Persistence restore: a SECOND indexer pool over the same log files
  // must serve searches it never saw written — its Create() runs
  // RestoreIndex over 564's snapshot + WAL. (In-process stand-in for a
  // daemon reboot; the restart integration test covers the real thing.)
  {
    clio::cte::indexer::IndexerConfig params;
    params.next_pool_id_ = clio::run::PoolId(512, 0);
    params.tag_re_ = "idx_.*";
    params.index_log_path_ = IndexerOpsFixture::IndexLogPath();
    auto create = idx.AsyncCreateIndexer(clio::run::PoolQuery::Local(),
                                         "indexer_restore",
                                         clio::run::PoolId(565, 0), params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);

    clio::cte::core::Client via_restored(clio::run::PoolId(565, 0));
    auto task = via_restored.AsyncSemanticSearch(
        ".*", "^scan_1$", "volcanic basalt columns", 0,
        clio::run::PoolQuery::Local());
    task.Wait();
    REQUIRE(task->GetReturnCode() == 0);
    REQUIRE(!task->results_.empty());
    REQUIRE(task->results_[0].blob_name_ == "scan_1");
    REQUIRE(task->results_[0].score_ > 0.0);
  }
}

SIMPLE_TEST_MAIN()
