/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Summarizer summarize-and-store integration test (no Ollama).
 *
 * test_summarizer_dispatch covers the path where inference FAILS. The success
 * path — prompt the model, then write the response back as `{name}_label` —
 * was only ever covered by the Ollama-gated tests, so it went untested on any
 * machine without a model installed. This test closes that gap by standing up
 * an in-process HTTP stub that answers like Ollama, pointing the summarizer's
 * label_endpoint at it, and asserting the summary blob actually lands in CTE
 * with the stub's text.
 *
 * The stub binds a FIXED port because the compose YAML has to name the
 * endpoint before the runtime starts.
 */

#include "simple_test.h"
#include "summarizer_http_stub.h"

#include <chrono>
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
using clio_cae_test::OneShotHttpServer;

namespace {

// Must match `label_endpoint` in test_summarizer_store_config.yaml.
constexpr unsigned short kStubPort = 21434;
// The `response` field the stub returns, i.e. the summary the summarizer
// must store verbatim.
constexpr const char *kStubSummary = "A stub summary of the document.";

/** PutBlob a small patterned buffer under `blob_name` and return the rc. */
int PutNamedBlob(clio::cte::core::Client *cte_client,
                 const clio::cte::core::TagId &tag_id,
                 const std::string &blob_name, size_t data_size) {
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

/** Read a blob back as a std::string, or "" when it doesn't exist. */
std::string ReadBlob(clio::cte::core::Client *cte,
                     const clio::cte::core::TagId &tag_id,
                     const std::string &blob_name) {
  auto sz = cte->AsyncGetBlobSize(tag_id, blob_name);
  sz.Wait();
  if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
    return std::string();
  }
  size_t n = sz->size_;
  auto buf = CLIO_IPC->AllocateBuffer(n);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, 0, n);
  ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
  auto get = cte->AsyncGetBlob(tag_id, blob_name, 0, n, 0, shm);
  get.Wait();
  std::string out;
  if (get->GetReturnCode() == 0) {
    out.assign(buf.ptr_, n);
  }
  CLIO_IPC->FreeBuffer(buf);
  return out;
}

}  // namespace

TEST_CASE("Summarizer stores the model response as {name}_label",
          "[cae][summarizer][store]") {
  // Stand the stub up BEFORE the runtime: the first PutBlob below prompts it
  // synchronously inside the summarizer's handler.
  OneShotHttpServer stub("HTTP/1.1 200 OK\r\n",
                         std::string("{\"response\":\"") + kStubSummary + "\"}",
                         kStubPort);
  REQUIRE(stub.Port() == kStubPort);

  fs::path config_path =
      fs::path(__FILE__).parent_path() / "test_summarizer_store_config.yaml";
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path.string(), 1);

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(1s);

  // The entrypoint (512.0) is the CAE core, which forwards to the summarizer
  // at 401.0 and on to the CTE core at 513.0.
  auto *cte = CLIO_CTE_CLIENT;
  cte->Init(clio::cte::core::kCtePoolId);

  auto tag = cte->AsyncGetOrCreateTag("notes.txt");
  tag.Wait();
  REQUIRE(tag->GetReturnCode() == 0);
  auto tag_id = tag->tag_id_;

  // 1. A matching blob is summarized. Summarization is synchronous inside the
  //    PutBlob handler, so the summary blob exists by the time Wait() returns.
  REQUIRE(PutNamedBlob(cte, tag_id, "doc_alpha", 2048) == 0);

  // The summarizer writes below itself, so read the summary through a client
  // pointed straight at the CTE core (the CAE core forwards GetBlobSize
  // nowhere -- it only implements four verbs).
  auto cte_direct =
      std::make_unique<clio::cte::core::Client>(clio::run::PoolId(513, 0));
  REQUIRE(ReadBlob(cte_direct.get(), tag_id, "doc_alpha_label") ==
          std::string(kStubSummary));
  REQUIRE(stub.RequestCount() >= 1);

  // The prompt must carry the blob's ACTUAL bytes. The handler snapshots the
  // payload out of shared memory AFTER forwarding the task down the chain, so
  // this is what catches the forward having consumed or invalidated the
  // inbound buffer -- a corrupted payload would still produce a summary blob
  // (the stub answers regardless), and every other assertion here would pass.
  std::vector<std::string> bodies = stub.RequestBodies();
  REQUIRE(!bodies.empty());
  REQUIRE(bodies[0].find("Summarize this document:") != std::string::npos);
  // 2048 bytes of the 'a'..'z' cycle: check a stretch long enough that a
  // zeroed or recycled buffer cannot pass by luck.
  REQUIRE(bodies[0].find("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz") !=
          std::string::npos);

  // 2. The original payload is untouched by the summarization.
  std::string original = ReadBlob(cte_direct.get(), tag_id, "doc_alpha");
  REQUIRE(original.size() == 2048);
  REQUIRE(original[0] == 'a');

  // 3. A blob whose name does not match the rule's blob_re is stored with no
  //    summary beside it.
  REQUIRE(PutNamedBlob(cte, tag_id, "misc_beta", 512) == 0);
  REQUIRE(ReadBlob(cte_direct.get(), tag_id, "misc_beta_label").empty());

  stub.Stop();
}

SIMPLE_TEST_MAIN()
