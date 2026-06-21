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
 * "gemma3:1b" model pulled). Two scenarios:
 *
 *   1. Small blob, large context_length — single LLM call, label stored.
 *   2. Larger blob, tiny context_length (512 tokens) — CAE chunks the
 *      blob, summarizes every chunk, concatenates the per-chunk
 *      responses, and stores the result as one label blob.
 *
 * The labeling is best-effort — if Ollama is unreachable the underlying
 * PutBlob still succeeds, but the label blob will be missing and the
 * tests fail. Set CAE_LABEL_TEST_SKIP=1 to skip entirely.
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
#include <clio_ctp/introspect/system_info.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

bool ShouldSkip() {
  const char *skip = std::getenv("CAE_LABEL_TEST_SKIP");
  return skip && std::string(skip) == "1";
}

/**
 * Lazy clio bring-up. Multiple TEST_CASEs in the same binary share
 * one runtime — CLIO_INIT is internally guarded but we also avoid
 * the post-init sleep on subsequent calls.
 */
void EnsureClio() {
  static bool s_initialized = false;
  if (s_initialized) return;
  fs::path config_path = fs::path(__FILE__).parent_path() /
                          "test_cae_labeling_config.yaml";
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path.string(), 1);
  bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer);
  REQUIRE(success);
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(1s);
  auto *cte_client = CLIO_CTE_CLIENT;
  cte_client->Init(clio::cte::core::kCtePoolId);
  s_initialized = true;
}

/**
 * Poll for `{blob_name}_label` until it appears (up to `attempts`
 * 2-second poll cycles) and return its contents. The label is written
 * synchronously inside the CAE PutBlob handler, but we still poll to
 * tolerate slightly-delayed scheduling on busy CI boxes.
 */
std::string WaitForLabel(clio::cte::core::Client *cte_client,
                         const clio::cte::core::TagId &tag_id,
                         const std::string &label_name,
                         size_t label_capacity, int attempts) {
  for (int attempt = 0; attempt < attempts; ++attempt) {
    auto get_buf = CLIO_IPC->AllocateBuffer(label_capacity);
    REQUIRE(!get_buf.IsNull());
    std::memset(get_buf.ptr_, 0, label_capacity);
    ctp::ipc::ShmPtr<> get_shm = get_buf.shm_.template Cast<void>();

    auto get_task = cte_client->AsyncGetBlob(
        tag_id, label_name, 0, label_capacity, 0, get_shm);
    get_task.Wait();
    std::string out;
    if (get_task->GetReturnCode() == 0) {
      size_t end = label_capacity;
      while (end > 0 && get_buf.ptr_[end - 1] == '\0') --end;
      if (end > 0) out.assign(get_buf.ptr_, end);
    }
    CLIO_IPC->FreeBuffer(get_buf);
    if (!out.empty()) return out;
    std::this_thread::sleep_for(2s);
  }
  return {};
}

}  // namespace

TEST_CASE("CAE transparent labeling via Ollama writes {name}_label",
          "[cae][labeling][ollama]") {
  if (ShouldSkip()) {
    INFO("CAE_LABEL_TEST_SKIP=1 set; skipping");
    return;
  }
  EnsureClio();
  auto *cte_client = CLIO_CTE_CLIENT;

  // ".txt" tag → first YAML rule with context_length 4096 (single shot).
  auto tag_task = cte_client->AsyncGetOrCreateTag("notes.txt");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  REQUIRE(!tag_task->tag_id_.IsNull());
  auto tag_id = tag_task->tag_id_;

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

  std::string label = WaitForLabel(cte_client, tag_id, "apollo_label",
                                   8 * 1024, 30);
  if (label.empty()) {
    INFO("Label blob never appeared; is ollama running at 127.0.0.1:11434 "
         "with gemma3:1b? Set CAE_LABEL_TEST_SKIP=1 to skip.");
  }
  REQUIRE(!label.empty());
  INFO("Generated label: " + label);
}

TEST_CASE("CAE transparent labeling chunks oversized blobs",
          "[cae][labeling][ollama][chunking]") {
  if (ShouldSkip()) {
    INFO("CAE_LABEL_TEST_SKIP=1 set; skipping");
    return;
  }
  EnsureClio();
  auto *cte_client = CLIO_CTE_CLIENT;

  // ".chunked" tag → YAML rule with context_length=512. Chunking math
  // in CAE::PutBlob reserves ~25% of context for the prompt+response
  // and assumes ~3 bytes/token, so the effective per-chunk budget is
  // ~512 * 0.75 * 3 ≈ 1150 bytes minus prompt size (~80 bytes) ≈ 1070.
  // Build a ~5 KB body of three distinct paragraphs so we get 4+
  // chunks and the concatenated label has clearly more content than a
  // single-shot summary would produce.
  auto tag_task = cte_client->AsyncGetOrCreateTag("history.chunked");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  auto tag_id = tag_task->tag_id_;

  // Three distinct historical events, each on its own paragraph. Padded
  // with filler so the byte budget per chunk is exceeded and the
  // chunker has to split the blob into multiple pieces.
  const std::string paragraph_filler(900, ' ');
  std::string body;
  body +=
      "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong and "
      "Buzz Aldrin walked on the lunar surface while Michael Collins "
      "orbited overhead.";
  body += paragraph_filler;
  body +=
      "The Wright brothers achieved the first controlled, sustained flight "
      "of a powered, heavier-than-air aircraft on December 17, 1903 at "
      "Kitty Hawk, North Carolina.";
  body += paragraph_filler;
  body +=
      "The fall of the Berlin Wall on November 9, 1989 marked the symbolic "
      "end of the Cold War division of Europe and led to German "
      "reunification the following year.";
  body += paragraph_filler;
  REQUIRE(body.size() > 2 * 1024);  // sanity — definitely larger than one chunk

  auto put_buf = CLIO_IPC->AllocateBuffer(body.size());
  REQUIRE(!put_buf.IsNull());
  std::memcpy(put_buf.ptr_, body.data(), body.size());
  ctp::ipc::ShmPtr<> put_shm = put_buf.shm_.template Cast<void>();

  clio::cte::core::Context ctx;
  auto put_task = cte_client->AsyncPutBlob(
      tag_id, "history", 0, body.size(), put_shm, 0.5f, ctx, 0);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);
  CLIO_IPC->FreeBuffer(put_buf);

  // Allow more total polling time — multiple chunks → multiple LLM
  // calls. 60 attempts × 2 s = 2 minutes worst case.
  std::string label = WaitForLabel(cte_client, tag_id, "history_label",
                                   32 * 1024, 60);
  REQUIRE(!label.empty());
  // The chunked label concatenates per-chunk summaries separated by
  // "\n\n" (see core_runtime.cc::PutBlob step 6). With our 3-paragraph
  // body sliced across multiple chunks, expect at least one such
  // separator to appear — that's how we confirm the chunking path
  // actually fired instead of a single-shot summary.
  INFO("Chunked label (" + std::to_string(label.size()) + " bytes): " +
       label.substr(0, 400));
  REQUIRE(label.find("\n\n") != std::string::npos);
}

SIMPLE_TEST_MAIN()
