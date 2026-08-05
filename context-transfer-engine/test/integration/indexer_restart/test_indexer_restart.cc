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
 * Indexer restart test client (issue #905), two-mode:
 *
 *   --put     Phase 1: put four topically-distinct documents through the
 *             indexer pool (564.0), flush the core's metadata/data, and
 *             sanity-check that SemanticSearch ranks them correctly while
 *             the inline-maintained index is live.
 *   --verify  Phase 2 (after a daemon restart + re-compose): run the SAME
 *             searches WITHOUT any puts. Results can only come from the
 *             index the indexer rebuilt from the restored storage below —
 *             an empty or wrong ranking means the rebuild failed.
 *
 * Orchestrated by test_indexer_restart.sh; binds the client to the indexer
 * pool via CLIO_CTE_POOL=564.0 (exported by the script).
 */

#include <cstring>
#include <string>
#include <vector>

#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_runtime/clio_runtime.h>

namespace {

constexpr const char *kTagName = "idx_restart_tag";

struct Doc {
  const char *name;
  const char *body;
};

// Distinct vocabularies so BM25 rankings are unambiguous.
const Doc kDocs[] = {
    {"doc_space",
     "Apollo 11 landed on the Moon in 1969. Astronauts Armstrong and Aldrin "
     "walked the lunar surface while Collins orbited overhead."},
    {"doc_cook",
     "A roux is equal parts butter and flour cooked into a paste, the base "
     "for classic French sauces like bechamel and espagnole."},
    {"doc_garden",
     "Tomato plants need full sun, regular watering, and well-drained soil "
     "with slightly acidic pH for a healthy fruit set."},
    {"doc_rail",
     "The transcontinental railroad joined the Central Pacific and Union "
     "Pacific lines at Promontory Summit in 1869 with a golden spike."},
};
constexpr size_t kNumDocs = sizeof(kDocs) / sizeof(kDocs[0]);

bool InitClient() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Failed to init Clio client");
    return false;
  }
  // CLIO_CTE_POOL=564.0 (set by the driver script) binds the singleton to
  // the indexer pool without creating anything.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    HLOG(kError, "Failed to init CTE client");
    return false;
  }
  return true;
}

/** Run one search and dump the results for the log. */
std::vector<clio::cte::core::SemanticSearchResult> Search(
    const std::string &query, clio::run::u32 k) {
  auto *cte = CLIO_CTE_CLIENT;
  auto task = cte->AsyncSemanticSearch(".*", "^doc_.*$", query, k,
                                       clio::run::PoolQuery::Broadcast());
  task.Wait();
  if (task->GetReturnCode() != 0) {
    HLOG(kError, "SemanticSearch('{}') rc={}", query, task->GetReturnCode());
    return {};
  }
  for (size_t i = 0; i < task->results_.size(); ++i) {
    HLOG(kInfo, "  #{} {} (score {})", i, task->results_[i].blob_name_,
         task->results_[i].score_);
  }
  return task->results_;
}

/** The assertions both phases share: the index (fresh or rebuilt) must
 *  contain exactly the four docs and rank them by topic. */
int VerifySearches(const char *phase) {
  // Empty query: BM25 scores are all 0 but every regex-matched indexed doc
  // is returned — a direct census of the index.
  HLOG(kInfo, "{}: census query", phase);
  auto all = Search("", 0);
  if (all.size() != kNumDocs) {
    HLOG(kError, "{}: expected {} indexed docs, found {}", phase, kNumDocs,
         all.size());
    return 1;
  }

  HLOG(kInfo, "{}: space query", phase);
  auto space = Search("apollo moon lunar surface astronauts", 4);
  if (space.empty() || space[0].blob_name_ != "doc_space" ||
      space[0].score_ <= 0.0) {
    HLOG(kError, "{}: space query did not rank doc_space first", phase);
    return 1;
  }
  for (size_t i = 1; i < space.size(); ++i) {
    if (space[i].score_ > space[i - 1].score_) {
      HLOG(kError, "{}: results not sorted by descending score", phase);
      return 1;
    }
  }

  HLOG(kInfo, "{}: cooking query", phase);
  auto cook = Search("butter flour roux sauce", 2);
  if (cook.empty() || cook[0].blob_name_ != "doc_cook") {
    HLOG(kError, "{}: cooking query did not rank doc_cook first", phase);
    return 1;
  }

  HLOG(kSuccess, "{}: searches verified", phase);
  return 0;
}

int PutPhase() {
  if (!InitClient()) return 1;
  auto *cte = CLIO_CTE_CLIENT;
  auto *ipc = CLIO_IPC;
  if (ipc == nullptr) {
    HLOG(kError, "Phase 1: no IPC manager");
    return 1;
  }

  auto tag = cte->AsyncGetOrCreateTag(kTagName);
  tag.Wait();
  if (tag->GetReturnCode() != 0) {
    HLOG(kError, "Phase 1: GetOrCreateTag rc={}", tag->GetReturnCode());
    return 1;
  }

  for (size_t i = 0; i < kNumDocs; ++i) {
    const size_t len = strlen(kDocs[i].body);
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(len);
    if (buf.IsNull()) {
      HLOG(kError, "Phase 1: AllocateBuffer failed");
      return 1;
    }
    memcpy(buf.ptr_, kDocs[i].body, len);
    ctp::ipc::ShmPtr<> shm(buf.shm_);
    auto put = cte->AsyncPutBlob(tag->tag_id_, kDocs[i].name, 0, len, shm,
                                 1.0f);
    put.Wait();
    ipc->FreeBuffer(buf);
    if (put->GetReturnCode() != 0) {
      HLOG(kError, "Phase 1: PutBlob '{}' rc={}", kDocs[i].name,
           put->GetReturnCode());
      return 1;
    }
    HLOG(kInfo, "Phase 1: put '{}' ({} bytes)", kDocs[i].name, len);
  }

  // Pre-restart sanity: the inline-maintained index must already serve
  // correct rankings; otherwise phase 2 would diagnose the wrong thing.
  if (VerifySearches("Phase 1") != 0) return 1;

  // Persist the core's metadata snapshot and data so the restart has
  // something to rebuild from (routed through the indexer, forwarded down).
  auto flush_meta = cte->AsyncFlushMetadata(clio::run::PoolQuery::Local(), 0);
  flush_meta.Wait();
  auto flush_data = cte->AsyncFlushData(clio::run::PoolQuery::Local(), 0, 0);
  flush_data.Wait();
  HLOG(kSuccess, "Phase 1: {} docs stored, index verified, state flushed",
       kNumDocs);
  return 0;
}

int VerifyPhase() {
  if (!InitClient()) return 1;
  // NO puts here: everything the searches return was rebuilt from storage.
  return VerifySearches("Phase 2");
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    HLOG(kError, "Usage: {} [--put|--verify]", argv[0]);
    return 1;
  }
  const std::string mode = argv[1];
  if (mode == "--put") return PutPhase();
  if (mode == "--verify") return VerifyPhase();
  HLOG(kError, "Usage: {} [--put|--verify]", argv[0]);
  return 1;
}
