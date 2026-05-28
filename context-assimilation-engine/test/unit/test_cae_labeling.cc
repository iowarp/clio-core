/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * CAE Transparent Labeling Integration Test
 *
 * Brings up CAE at pool 512.0 with labeling rules pointing at a local
 * Ollama instance (assumed running at http://127.0.0.1:11434 with the
 * "gemma3:1b" model pulled). Puts a text blob whose tag name matches
 * the labeling rule, then verifies that a `{blob_name}_label` blob
 * appeared in the same tag with non-empty contents.
 *
 * The labeling is best-effort — if Ollama is unreachable the underlying
 * PutBlob still succeeds, but the label blob will be missing and this
 * test will SKIP (we treat missing-Ollama as "not the bug we're
 * testing"). To skip explicitly set CAE_LABEL_TEST_SKIP=1.
 *
 * See GitHub issue #466.
 */

#include "simple_test.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/content_transfer_engine.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

TEST_CASE("CAE transparent labeling via Ollama writes {name}_label",
          "[cae][labeling][ollama]") {
  if (const char *skip = std::getenv("CAE_LABEL_TEST_SKIP");
      skip && std::string(skip) == "1") {
    INFO("CAE_LABEL_TEST_SKIP=1 set; skipping");
    return;
  }

  fs::path config_path = fs::path(__FILE__).parent_path() /
                          "test_cae_labeling_config.yaml";
  setenv("CLIO_SERVER_CONF", config_path.c_str(), 1);

  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer);
  REQUIRE(success);
  SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
  std::this_thread::sleep_for(1s);

  // Point the global CTE client at the CAE entrypoint pool (512.0). The
  // CAE container is what intercepts PutBlob and fires the LLM call.
  auto *cte_client = CLIO_CTE_CLIENT;
  cte_client->Init(clio::cte::core::kCtePoolId);

  // Tag name matches `.*\.txt$` in the YAML rule; blob name matches
  // `.*` so labeling fires.
  auto tag_task = cte_client->AsyncGetOrCreateTag("notes.txt");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  REQUIRE(!tag_task->tag_id_.IsNull());
  auto tag_id = tag_task->tag_id_;

  // Body of the blob — small text payload so gemma3:1b summarizes
  // quickly even on CPU-only inference.
  const std::string body =
      "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong "
      "became the first human to step onto the lunar surface, followed "
      "by Buzz Aldrin. The mission marked the culmination of the "
      "United States' efforts to fulfill President Kennedy's pledge to "
      "reach the Moon before the end of the decade.";

  auto put_buf = CLIO_IPC->AllocateBuffer(body.size());
  REQUIRE(!put_buf.IsNull());
  std::memcpy(put_buf.ptr_, body.data(), body.size());
  ctp::ipc::ShmPtr<> put_shm = put_buf.shm_.template Cast<void>();

  clio::cte::core::Context ctx;
  auto put_task = cte_client->AsyncPutBlob(
      tag_id, "apollo", 0, body.size(), put_shm, 0.5f, ctx, 0);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);
  CLIO_IPC->FreeBuffer(put_buf);

  // Labeling runs synchronously inside the CAE PutBlob handler, so by
  // the time PutBlob's future resolves the label blob is already
  // stored. Still, poll a few times in case the labeling worker is
  // slightly behind on a busy CI box.
  const std::string label_name = "apollo_label";
  const size_t label_capacity = 8 * 1024;
  bool got_label = false;
  std::string label_text;

  for (int attempt = 0; attempt < 30 && !got_label; ++attempt) {
    auto get_buf = CLIO_IPC->AllocateBuffer(label_capacity);
    REQUIRE(!get_buf.IsNull());
    std::memset(get_buf.ptr_, 0, label_capacity);
    ctp::ipc::ShmPtr<> get_shm = get_buf.shm_.template Cast<void>();

    auto get_task = cte_client->AsyncGetBlob(
        tag_id, label_name, 0, label_capacity, 0, get_shm);
    get_task.Wait();
    if (get_task->GetReturnCode() == 0) {
      // Trim trailing zeros (we requested label_capacity but the actual
      // label is shorter).
      size_t end = label_capacity;
      while (end > 0 && get_buf.ptr_[end - 1] == '\0') --end;
      if (end > 0) {
        label_text.assign(get_buf.ptr_, end);
        got_label = true;
      }
    }
    CLIO_IPC->FreeBuffer(get_buf);
    if (!got_label) std::this_thread::sleep_for(2s);
  }

  if (!got_label) {
    INFO("Label blob never appeared; is ollama running at 127.0.0.1:11434 "
         "with gemma3:1b? Set CAE_LABEL_TEST_SKIP=1 to skip.");
  }
  REQUIRE(got_label);
  REQUIRE(label_text.size() > 0);
  INFO("Generated label: " + label_text);
}

SIMPLE_TEST_MAIN()
