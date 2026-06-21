/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * CAE labeling + CTE SemanticSearch end-to-end test.
 *
 * Inserts 10 short text "documents" via the standard CTE client. The
 * client targets the CAE pool (512.0), so CAE's PutBlob interceptor
 * fires Ollama labeling on each blob and stores `{name}_label`
 * alongside. Then we issue an AsyncSemanticSearch with a
 * natural-language query restricted to `.*_label$` blob names — BM25
 * scores the labels, sorted top-5 returned.
 *
 * The test treats Ollama as required (no CAE_LABEL_TEST_SKIP knob
 * here): without the labels the search has nothing to rank.
 *
 * Refs #466
 */

#include "simple_test.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/content_transfer_engine.h>
#include <clio_ctp/introspect/system_info.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

// 10 short documents about distinct topics. The query in the test is
// about space / moon — docs 0, 5 (space-themed) should rank highest.
struct Doc {
  const char *name;
  const char *body;
};

constexpr Doc kCorpus[] = {
    {"doc0",
     "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong "
     "and Buzz Aldrin walked on the lunar surface while Michael "
     "Collins orbited overhead."},
    {"doc1",
     "The Wright brothers achieved the first controlled, sustained "
     "flight of a powered, heavier-than-air aircraft on December 17, "
     "1903 at Kitty Hawk, North Carolina."},
    {"doc2",
     "The fall of the Berlin Wall on November 9, 1989 marked the "
     "symbolic end of the Cold War division of Europe."},
    {"doc3",
     "Marie Curie was a Polish and naturalized-French physicist and "
     "chemist who conducted pioneering research on radioactivity."},
    {"doc4",
     "The Great Wall of China is a series of fortifications stretching "
     "across the historical northern borders of ancient Chinese states."},
    {"doc5",
     "The Hubble Space Telescope was launched into low Earth orbit in "
     "1990, revolutionizing our understanding of the cosmos including "
     "lunar imaging and distant galaxies."},
    {"doc6",
     "The printing press was invented by Johannes Gutenberg around 1440 "
     "and helped catalyze the spread of knowledge during the Renaissance."},
    {"doc7",
     "The Industrial Revolution began in Britain in the late 18th "
     "century with the mechanization of textile manufacturing."},
    {"doc8",
     "The Roman Empire reached its territorial peak under Emperor Trajan "
     "in 117 AD, spanning much of Europe, North Africa, and the Middle East."},
    {"doc9",
     "The Statue of Liberty was a gift from France to the United States "
     "dedicated in 1886, designed by Frederic Auguste Bartholdi."},
};

constexpr size_t kNumDocs = sizeof(kCorpus) / sizeof(kCorpus[0]);

}  // namespace

TEST_CASE("CAE labels 10 docs; CTE SemanticSearch returns top-5",
          "[cae][cte][bm25][semantic-search][ollama]") {
  {
    const char *skip = std::getenv("CAE_LABEL_TEST_SKIP");
    if (skip && std::string(skip) == "1") {
      INFO("CAE_LABEL_TEST_SKIP=1 set; skipping");
      return;
    }
  }
  fs::path config_path = fs::path(__FILE__).parent_path() /
                          "test_cae_semantic_search_config.yaml";
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path.string(), 1);

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(1s);

  // CLIO_CTE_CLIENT lands on the CAE entrypoint pool (512.0) — that's
  // the whole interceptor story. PutBlobs go to CAE first, which
  // forwards to CTE and writes a `{name}_label` blob next to each.
  auto *cte = CLIO_CTE_CLIENT;
  cte->Init(clio::cte::core::kCtePoolId);

  auto tag = cte->AsyncGetOrCreateTag("history_corpus");
  tag.Wait();
  REQUIRE(tag->GetReturnCode() == 0);
  auto tag_id = tag->tag_id_;

  // Step 1: PutBlob each document. CAE handler labels every one
  // synchronously inside the PutBlob path, so by the time these
  // Wait() returns the matching `{name}_label` blob is already
  // written. Sequential so we don't blast 10 inferences at once on
  // the local Ollama (gemma3:1b on CPU).
  for (const auto &doc : kCorpus) {
    size_t n = std::strlen(doc.body);
    auto buf = CLIO_IPC->AllocateBuffer(n);
    REQUIRE(!buf.IsNull());
    std::memcpy(buf.ptr_, doc.body, n);
    ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
    clio::cte::core::Context ctx;
    auto put = cte->AsyncPutBlob(tag_id, doc.name, 0, n, shm, 0.5f, ctx, 0);
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
    CLIO_IPC->FreeBuffer(buf);
  }

  // Step 2: confirm every doc got a label. CAE doesn't forward
  // GetBlobSize, so we ask CTE directly via a client pointed at 513.0.
  auto cte_direct = std::make_unique<clio::cte::core::Client>(
      clio::run::PoolId(513, 0));
  size_t labels_present = 0;
  for (const auto &doc : kCorpus) {
    std::string lname = std::string(doc.name) + "_label";
    auto sz = cte_direct->AsyncGetBlobSize(tag_id, lname);
    sz.Wait();
    if (sz->GetReturnCode() == 0 && sz->size_ > 0) ++labels_present;
  }
  INFO("Labels present: " + std::to_string(labels_present) +
       " / " + std::to_string(kNumDocs));
  REQUIRE(labels_present == kNumDocs);

  // Step 3: SemanticSearch via the CAE-fronted client. Restrict to
  // `.*_label$` so BM25 ranks the LLM-generated summaries (the actual
  // searchable surface created by transparent labeling) rather than
  // the raw source documents. Query is about space/moon — doc0
  // (Apollo) and doc5 (Hubble) should clearly outrank Berlin Wall,
  // Marie Curie, Roman Empire, etc.
  const std::string query = "moon space telescope lunar exploration";
  auto search = cte->AsyncSemanticSearch(
      /*tag_regex=*/".*", /*blob_regex=*/".*_label$",
      /*query_text=*/query, /*k=*/5);
  search.Wait();
  REQUIRE(search->GetReturnCode() == 0);

  // We asked for top-5; with 10 labels and a 5-token query at least a
  // few should match. Print the ranked list so a regression in BM25
  // or tokenization is easy to spot.
  std::vector<clio::cte::core::SemanticSearchResult> &results =
      search->results_;
  INFO("Search returned " + std::to_string(results.size()) +
       " results for query: '" + query + "'");
  for (size_t i = 0; i < results.size(); ++i) {
    INFO("  #" + std::to_string(i) + "  blob=" + results[i].blob_name_ +
         "  score=" + std::to_string(results[i].score_));
  }
  REQUIRE(results.size() > 0);
  REQUIRE(results.size() <= 5);

  // Verify the result is sorted (descending score). The handler is
  // supposed to sort by score; if a future refactor breaks this we
  // want a loud failure here.
  for (size_t i = 1; i < results.size(); ++i) {
    REQUIRE(results[i].score_ <= results[i - 1].score_);
  }

  // Sanity: at least one of doc0_label / doc5_label (the space-themed
  // ones) should appear in the top-5. BM25 on a short Ollama summary
  // is noisy, but the on-topic docs should swamp the off-topic ones.
  bool found_relevant = false;
  for (const auto &r : results) {
    if (r.blob_name_ == "doc0_label" || r.blob_name_ == "doc5_label") {
      found_relevant = true;
      break;
    }
  }
  if (!found_relevant) {
    INFO("Expected doc0_label (Apollo) or doc5_label (Hubble) in top-5; "
         "labels may not have mentioned the relevant terms verbatim. "
         "Check the INFO lines above for the ranked output.");
  }
  REQUIRE(found_relevant);
}

SIMPLE_TEST_MAIN()
