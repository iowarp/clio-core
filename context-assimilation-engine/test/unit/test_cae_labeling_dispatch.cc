/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * CAE labeling-dispatch + semantic-search integration test (no Ollama).
 *
 * The Ollama-gated tests (test_cae_labeling / test_cae_semantic_search) need a
 * live LLM to assert on real label text. This test instead brings up CAE with
 * label rules whose endpoint points at a dead port, so it exercises the
 * CAE-side dispatch logic that runs BEFORE any HTTP call and never needs a
 * model:
 *
 *   - Runtime::PutBlob transparent-labeling branch: FindLabelMatch (including
 *     its std::regex_error catch, a matching rule, and the no-match path),
 *     prompt resolution, chunking, and the graceful OllamaGenerate failure
 *     (label simply isn't produced; the blob PutBlob still succeeds).
 *   - Runtime::SemanticSearch forwarding to the CTE BM25 backend (CAE adds no
 *     LLM logic here — it just forwards), exercising the kSemanticSearch
 *     dispatch path.
 *
 * Labeling failures must never flip the PutBlob return code, so every PutBlob
 * here is expected to return 0.
 */

#include "simple_test.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/content_transfer_engine.h>
#include <clio_ctp/introspect/system_info.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

// PutBlob a small patterned buffer under `blob_name` and return the rc.
int PutNamedBlob(clio::cte::core::Client *cte_client,
                 const clio::cte::core::TagId &tag_id,
                 const std::string &blob_name) {
  const size_t data_size = 2048;
  auto buffer = CLIO_IPC->AllocateBuffer(data_size);
  REQUIRE(!buffer.IsNull());
  for (size_t i = 0; i < data_size; ++i) {
    buffer.ptr_[i] = static_cast<char>('a' + (i % 26));
  }
  ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();
  clio::cte::core::Context ctx;
  auto put_task = cte_client->AsyncPutBlob(tag_id, blob_name.c_str(), 0,
                                           data_size, blob_data, 1.0f, ctx, 0);
  put_task.Wait();
  int rc = put_task->GetReturnCode();
  CLIO_IPC->FreeBuffer(buffer);
  return rc;
}

}  // namespace

TEST_CASE("CAE labeling dispatch + semantic search forward",
          "[cae][labeling][dispatch][semanticsearch]") {
  fs::path config_path = fs::path(__FILE__).parent_path() /
                         "test_cae_labeling_dispatch_config.yaml";
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path.string(), 1);

  bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer);
  REQUIRE(success);
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;

  // Let compose pools materialize.
  std::this_thread::sleep_for(1s);

  // Point the CTE client at the entrypoint pool (512.0 = CAE interceptor).
  auto *cte_client = CLIO_CTE_CLIENT;
  cte_client->Init(clio::cte::core::kCtePoolId);

  // GetOrCreateTag first so CAE caches tag_id -> "label_tag" for the PutBlob
  // labeling lookup. The tag name matches the second label rule's tag_re.
  auto tag_task = cte_client->AsyncGetOrCreateTag("label_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  REQUIRE(!tag_task->tag_id_.IsNull());
  auto tag_id = tag_task->tag_id_;

  // 1. Blob name "doc_alpha" matches the (valid) second rule's blob_re
  //    "doc_.*". FindLabelMatch walks past the invalid-regex first rule (its
  //    regex_error is caught), matches the second, resolves the "summarize"
  //    prompt, chunks the payload, and calls OllamaGenerate against the dead
  //    endpoint — which fails, so no label blob is written. The original blob
  //    PutBlob must still succeed.
  REQUIRE(PutNamedBlob(cte_client, tag_id, "doc_alpha") == 0);

  // 2. Blob name "misc_beta" does not match "doc_.*", so FindLabelMatch
  //    returns nullptr and PutBlob returns immediately after forwarding.
  REQUIRE(PutNamedBlob(cte_client, tag_id, "misc_beta") == 0);

  // 3. The original blob is intact despite the labeling attempt.
  {
    const size_t data_size = 2048;
    auto get_buffer = CLIO_IPC->AllocateBuffer(data_size);
    REQUIRE(!get_buffer.IsNull());
    std::memset(get_buffer.ptr_, 0, data_size);
    ctp::ipc::ShmPtr<> gd = get_buffer.shm_.template Cast<void>();
    auto get_task =
        cte_client->AsyncGetBlob(tag_id, "doc_alpha", 0, data_size, 0, gd);
    get_task.Wait();
    REQUIRE(get_task->GetReturnCode() == 0);
    REQUIRE(get_buffer.ptr_[0] == 'a');
    CLIO_IPC->FreeBuffer(get_buffer);
  }

  // 4. SemanticSearch through CAE forwards to the CTE BM25 backend (no LLM).
  //    Run locally against the entrypoint pool; a well-formed query must
  //    return rc 0 regardless of how many blobs match.
  auto search_task = cte_client->AsyncSemanticSearch(
      "label_tag", "doc_.*", "alpha document", 5, clio::run::PoolQuery::Local());
  search_task.Wait();
  REQUIRE(search_task->GetReturnCode() == 0);
}

SIMPLE_TEST_MAIN()
