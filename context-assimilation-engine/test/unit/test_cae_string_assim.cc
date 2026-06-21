/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * StringDataAssimilator end-to-end test.
 *
 * Feeds a payload directly through AssimilationCtx::src_data with
 * src="string::<blob_name>" and dst="iowarp::<tag_name>", drives it
 * through CAE's AsyncParseOmni, then reads the bytes back via the CTE
 * client and asserts the round-trip. Covers:
 *   - The factory routes "string::" to StringDataAssimilator.
 *   - The AssimilationCtx::src_data field survives serialization.
 *   - The resulting tag matches dst's path; the resulting single blob
 *     is named after src's path; the body is src_data verbatim.
 *   - A binary payload (full 0..255 byte range) round-trips intact —
 *     proves src_data is byte-clean despite riding std::string.
 *   - Malformed sources error rather than silently no-op.
 *
 * NOTE: zero-byte payloads aren't tested — CTE's PutBlob rejects
 * size=0, which surfaces here as a PutBlob rc=2. The agentic-loop
 * use case always produces some bytes, so we treat empty payloads
 * as out-of-scope rather than papering over the underlying behavior.
 */

#include "simple_test.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/constants.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_ctp/introspect/system_info.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

TEST_CASE("StringDataAssimilator stores ctx.src_data as one named blob",
          "[cae][string-assim][parseomni]") {
  fs::path config_path = fs::path(__FILE__).parent_path() /
                          "test_cae_string_assim_config.yaml";
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path.string(), 1);

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(1s);

  // CAE peer pool at 400.0; CTE peer pool at 512.0 (kCtePoolId default).
  clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);
  auto *cte_client = CLIO_CTE_CLIENT;
  cte_client->Init(clio::cte::core::kCtePoolId);

  // -------- 1. Happy path: a single small payload --------------------
  const std::string payload = "agent loop emitted: {\"step\":7,\"ok\":true}";
  clio::cae::core::AssimilationCtx ctx;
  ctx.src = "string::agent_step_7";
  ctx.dst = "iowarp::agent_outputs";
  ctx.format = "string";
  ctx.src_data = payload;

  std::vector<clio::cae::core::AssimilationCtx> bundle{ctx};
  auto parse_task = cae_client.AsyncParseOmni(bundle);
  parse_task.Wait();
  REQUIRE(parse_task->result_code_ == 0);
  REQUIRE(parse_task->num_tasks_scheduled_ == 1);

  // Verify via direct CTE: tag derived from dst, blob name from src.
  auto tag_task = cte_client->AsyncGetOrCreateTag("agent_outputs");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  auto tag_id = tag_task->tag_id_;
  REQUIRE(!tag_id.IsNull());

  auto size_task = cte_client->AsyncGetBlobSize(tag_id, "agent_step_7");
  size_task.Wait();
  REQUIRE(size_task->GetReturnCode() == 0);
  REQUIRE(size_task->size_ == payload.size());

  auto get_buf = CLIO_IPC->AllocateBuffer(payload.size());
  REQUIRE(!get_buf.IsNull());
  std::memset(get_buf.ptr_, 0, payload.size());
  auto get_task = cte_client->AsyncGetBlob(
      tag_id, "agent_step_7", /*offset=*/0, payload.size(),
      /*flags=*/0, get_buf.shm_.template Cast<void>());
  get_task.Wait();
  REQUIRE(get_task->GetReturnCode() == 0);
  REQUIRE(std::memcmp(get_buf.ptr_, payload.data(), payload.size()) == 0);
  CLIO_IPC->FreeBuffer(get_buf);

  // -------- 2. Binary payload (non-printable bytes) ------------------
  // Confirms src_data is byte-clean — the field is std::string but we
  // pack arbitrary bytes through it.
  std::string blob_bytes;
  blob_bytes.reserve(256);
  for (int i = 0; i < 256; ++i) {
    blob_bytes.push_back(static_cast<char>(i));
  }
  clio::cae::core::AssimilationCtx bin_ctx;
  bin_ctx.src = "string::binary_dump";
  bin_ctx.dst = "iowarp::agent_outputs";
  bin_ctx.format = "string";
  bin_ctx.src_data = blob_bytes;
  std::vector<clio::cae::core::AssimilationCtx> bin_bundle{bin_ctx};
  auto bin_task = cae_client.AsyncParseOmni(bin_bundle);
  bin_task.Wait();
  REQUIRE(bin_task->result_code_ == 0);

  auto bin_size = cte_client->AsyncGetBlobSize(tag_id, "binary_dump");
  bin_size.Wait();
  REQUIRE(bin_size->GetReturnCode() == 0);
  REQUIRE(bin_size->size_ == blob_bytes.size());

  auto bin_buf = CLIO_IPC->AllocateBuffer(blob_bytes.size());
  std::memset(bin_buf.ptr_, 0, blob_bytes.size());
  auto bin_get = cte_client->AsyncGetBlob(
      tag_id, "binary_dump", 0, blob_bytes.size(), 0,
      bin_buf.shm_.template Cast<void>());
  bin_get.Wait();
  REQUIRE(bin_get->GetReturnCode() == 0);
  REQUIRE(std::memcmp(bin_buf.ptr_, blob_bytes.data(),
                       blob_bytes.size()) == 0);
  CLIO_IPC->FreeBuffer(bin_buf);

  // -------- 3. Malformed src — wrong protocol -------------------------
  // The factory should refuse a "foo::..." src; ParseOmni reports the
  // error rather than silently dropping the assimilation.
  clio::cae::core::AssimilationCtx bad_ctx;
  bad_ctx.src = "foo::nope";
  bad_ctx.dst = "iowarp::agent_outputs";
  bad_ctx.format = "string";
  bad_ctx.src_data = "ignored";
  std::vector<clio::cae::core::AssimilationCtx> bad_bundle{bad_ctx};
  auto bad_task = cae_client.AsyncParseOmni(bad_bundle);
  bad_task.Wait();
  REQUIRE(bad_task->result_code_ != 0);
}

SIMPLE_TEST_MAIN()
