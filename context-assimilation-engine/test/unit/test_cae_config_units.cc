/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * test_cae_config_units.cc
 *
 * Pure unit tests (no CLIO runtime) for the header-only CAE core config/task
 * plumbing that the runtime-integration tests never exercise directly:
 *
 *   - CreateParams::LoadConfig — YAML parsing of next_pool_id, plus the
 *     empty-config early-return and the malformed-YAML best-effort catch.
 *   - CreateParams serialize round-trip.
 *   - CreateParams copy and compose-pool-id constructors.
 *
 * The transparent-summarization config (label_endpoint / label_prompts /
 * label_matches) moved to the summarizer chimod — its coverage lives in
 * test_summarizer_config_units.cc.
 *
 * These paths live in core_tasks.h and compile straight into this TU, so no
 * runtime, CTE client, or network is required.
 */

#include "simple_test.h"

#include <clio_cae/core/core_tasks.h>
#include <clio_runtime/config_manager.h>

#include "clio_ctp/data_structures/serialization/global_serialize.h"

#include <string>
#include <vector>

using clio::cae::core::CreateParams;

namespace {

// Build a PoolConfig carrying `yaml` as its remaining-config blob.
clio::run::PoolConfig MakePoolConfig(const std::string &yaml) {
  clio::run::PoolConfig pc;
  pc.config_ = yaml;
  return pc;
}

}  // namespace

TEST_CASE("CreateParams - LoadConfig empty config is a no-op",
          "[cae][config][loadconfig]") {
  CreateParams params;
  // Empty config_ hits the early `return` before the YAML parse.
  params.LoadConfig(MakePoolConfig(""));
  REQUIRE(params.next_pool_id_.IsNull());
}

TEST_CASE("CreateParams - LoadConfig parses next_pool_id",
          "[cae][config][loadconfig]") {
  CreateParams params;
  params.LoadConfig(MakePoolConfig("next_pool_id: \"513.7\"\n"));
  REQUIRE(params.next_pool_id_.major_ == 513);
  REQUIRE(params.next_pool_id_.minor_ == 7);
}

TEST_CASE("CreateParams - LoadConfig ignores next_pool_id without a dot",
          "[cae][config][loadconfig]") {
  CreateParams params;
  // No '.' => the inner parse branch is skipped, id stays null.
  params.LoadConfig(MakePoolConfig("next_pool_id: \"nominor\"\n"));
  REQUIRE(params.next_pool_id_.IsNull());
}

TEST_CASE("CreateParams - LoadConfig ignores summarizer keys",
          "[cae][config][loadconfig]") {
  CreateParams params;
  // label_* moved to the summarizer chimod. A stale config carrying them is
  // parsed for next_pool_id and the rest is simply not this module's schema.
  params.LoadConfig(MakePoolConfig(
      "next_pool_id: \"513.0\"\n"
      "label_endpoint: \"http://host:11434\"\n"));
  REQUIRE(params.next_pool_id_.major_ == 513);
  REQUIRE(params.next_pool_id_.minor_ == 0);
}

TEST_CASE("CreateParams - LoadConfig swallows malformed YAML",
          "[cae][config][loadconfig]") {
  CreateParams params;
  // Unbalanced brackets => YAML::Load throws => caught by the best-effort
  // catch(...) so LoadConfig must not propagate.
  REQUIRE_NOTHROW(params.LoadConfig(MakePoolConfig("next_pool_id: [ {a: ")));
  // Nothing was successfully parsed.
  REQUIRE(params.next_pool_id_.IsNull());
}

TEST_CASE("CreateParams - copy constructor copies all fields",
          "[cae][config][ctor]") {
  CreateParams src;
  src.LoadConfig(MakePoolConfig("next_pool_id: \"321.4\"\n"));

  CreateParams copy(src);
  REQUIRE(copy.next_pool_id_.major_ == 321);
  REQUIRE(copy.next_pool_id_.minor_ == 4);
}

TEST_CASE("CreateParams - compose pool-id constructor copies config",
          "[cae][config][ctor]") {
  CreateParams src;
  src.LoadConfig(MakePoolConfig("next_pool_id: \"555.6\"\n"));

  // The pool_id argument is intentionally ignored by the ctor; it exists to
  // match the compressor compose pattern.
  clio::run::PoolId pid(42, 0);
  CreateParams composed(pid, src);
  REQUIRE(composed.next_pool_id_.major_ == 555);
  REQUIRE(composed.next_pool_id_.minor_ == 6);
}

TEST_CASE("CreateParams - serialize round-trips config",
          "[cae][config][serialize]") {
  CreateParams in;
  in.LoadConfig(MakePoolConfig("next_pool_id: \"700.2\"\n"));

  std::vector<char> buf;
  {
    ctp::ipc::GlobalSerialize<std::vector<char>> oa(buf);
    in.serialize(oa);
    oa.Finalize();
  }

  CreateParams out;
  {
    ctp::ipc::GlobalDeserialize<std::vector<char>> ia(buf);
    out.serialize(ia);
  }

  REQUIRE(out.next_pool_id_.major_ == 700);
  REQUIRE(out.next_pool_id_.minor_ == 2);
}

SIMPLE_TEST_MAIN()
