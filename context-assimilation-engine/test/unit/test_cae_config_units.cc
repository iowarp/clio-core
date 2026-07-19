/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * test_cae_config_units.cc
 *
 * Pure unit tests (no CLIO runtime) for the header-only CAE config/task
 * plumbing that the runtime-integration tests never exercise directly:
 *
 *   - CreateParams::LoadConfig — YAML parsing of next_pool_id, label_endpoint,
 *     label_prompts and label_matches (all sub-fields), plus the empty-config
 *     early-return and the malformed-YAML best-effort catch.
 *   - CreateParams / LabelMatch serialize round-trips.
 *   - CreateParams copy and compose-pool-id constructors.
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
using clio::cae::core::LabelMatch;

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
  REQUIRE(params.label_endpoint_.empty());
  REQUIRE(params.label_matches_.empty());
  REQUIRE(params.label_prompts_.empty());
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

TEST_CASE("CreateParams - LoadConfig parses label_endpoint",
          "[cae][config][loadconfig]") {
  CreateParams params;
  params.LoadConfig(MakePoolConfig("label_endpoint: \"http://host:11434\"\n"));
  REQUIRE(params.label_endpoint_ == "http://host:11434");
}

TEST_CASE("CreateParams - LoadConfig parses label_prompts map",
          "[cae][config][loadconfig]") {
  CreateParams params;
  params.LoadConfig(MakePoolConfig(
      "label_prompts:\n"
      "  summarize: \"Summarize this:\"\n"
      "  classify: \"Classify this:\"\n"));
  REQUIRE(params.label_prompts_.size() == 2);
  REQUIRE(params.label_prompts_["summarize"] == "Summarize this:");
  REQUIRE(params.label_prompts_["classify"] == "Classify this:");
}

TEST_CASE("CreateParams - LoadConfig parses label_matches with all fields",
          "[cae][config][loadconfig]") {
  CreateParams params;
  params.LoadConfig(MakePoolConfig(
      "label_matches:\n"
      "  - tag_re: \"docs/.*\"\n"
      "    blob_re: \".*\\\\.txt\"\n"
      "    model: \"gemma3:1b\"\n"
      "    prompt: \"summarize\"\n"
      "    context_length: 8192\n"
      "    num_predict: 64\n"));
  REQUIRE(params.label_matches_.size() == 1);
  const LabelMatch &m = params.label_matches_[0];
  REQUIRE(m.tag_re_ == "docs/.*");
  REQUIRE(m.blob_re_ == ".*\\.txt");
  REQUIRE(m.model_ == "gemma3:1b");
  REQUIRE(m.prompt_ == "summarize");
  REQUIRE(m.context_length_ == 8192);
  REQUIRE(m.num_predict_ == 64);
}

TEST_CASE("CreateParams - LoadConfig label_matches uses field defaults",
          "[cae][config][loadconfig]") {
  CreateParams params;
  // Only tag_re/blob_re given; context_length_/num_predict_ keep struct
  // defaults, and the optional string fields stay empty.
  params.LoadConfig(MakePoolConfig(
      "label_matches:\n"
      "  - tag_re: \"a\"\n"
      "    blob_re: \"b\"\n"));
  REQUIRE(params.label_matches_.size() == 1);
  const LabelMatch &m = params.label_matches_[0];
  REQUIRE(m.tag_re_ == "a");
  REQUIRE(m.blob_re_ == "b");
  REQUIRE(m.model_.empty());
  REQUIRE(m.prompt_.empty());
  REQUIRE(m.context_length_ == 4096);  // struct default
  REQUIRE(m.num_predict_ == 0);        // struct default
}

TEST_CASE("CreateParams - LoadConfig parses a full combined config",
          "[cae][config][loadconfig]") {
  CreateParams params;
  params.LoadConfig(MakePoolConfig(
      "next_pool_id: \"600.1\"\n"
      "label_endpoint: \"http://127.0.0.1:11434\"\n"
      "label_prompts:\n"
      "  p: \"Prompt:\"\n"
      "label_matches:\n"
      "  - tag_re: \"t\"\n"
      "    blob_re: \"b\"\n"
      "    model: \"m\"\n"
      "    prompt: \"p\"\n"));
  REQUIRE(params.next_pool_id_.major_ == 600);
  REQUIRE(params.next_pool_id_.minor_ == 1);
  REQUIRE(params.label_endpoint_ == "http://127.0.0.1:11434");
  REQUIRE(params.label_prompts_.size() == 1);
  REQUIRE(params.label_matches_.size() == 1);
}

TEST_CASE("CreateParams - LoadConfig swallows malformed YAML",
          "[cae][config][loadconfig]") {
  CreateParams params;
  // Unbalanced brackets => YAML::Load throws => caught by the best-effort
  // catch(...) so LoadConfig must not propagate.
  REQUIRE_NOTHROW(params.LoadConfig(MakePoolConfig("label_matches: [ {a: ")));
  // Nothing was successfully parsed.
  REQUIRE(params.label_matches_.empty());
}

TEST_CASE("CreateParams - copy constructor copies all fields",
          "[cae][config][ctor]") {
  CreateParams src;
  src.LoadConfig(MakePoolConfig(
      "next_pool_id: \"321.4\"\n"
      "label_endpoint: \"http://e\"\n"
      "label_prompts:\n"
      "  p: \"P\"\n"
      "label_matches:\n"
      "  - tag_re: \"t\"\n"
      "    blob_re: \"b\"\n"));

  CreateParams copy(src);
  REQUIRE(copy.next_pool_id_.major_ == 321);
  REQUIRE(copy.next_pool_id_.minor_ == 4);
  REQUIRE(copy.label_endpoint_ == "http://e");
  REQUIRE(copy.label_prompts_.size() == 1);
  REQUIRE(copy.label_matches_.size() == 1);
}

TEST_CASE("CreateParams - compose pool-id constructor copies config",
          "[cae][config][ctor]") {
  CreateParams src;
  src.LoadConfig(MakePoolConfig("label_endpoint: \"http://compose\"\n"));

  // The pool_id argument is intentionally ignored by the ctor; it exists to
  // match the compressor compose pattern.
  clio::run::PoolId pid(42, 0);
  CreateParams composed(pid, src);
  REQUIRE(composed.label_endpoint_ == "http://compose");
}

TEST_CASE("LabelMatch - serialize round-trips every field",
          "[cae][config][serialize]") {
  LabelMatch m;
  m.tag_re_ = "tag.*";
  m.blob_re_ = "blob.*";
  m.model_ = "model-x";
  m.prompt_ = "prompt-key";
  m.context_length_ = 12345;
  m.num_predict_ = 99;

  std::vector<char> buf;
  {
    ctp::ipc::GlobalSerialize<std::vector<char>> oa(buf);
    m.serialize(oa);
    oa.Finalize();
  }

  LabelMatch out;
  {
    ctp::ipc::GlobalDeserialize<std::vector<char>> ia(buf);
    out.serialize(ia);
  }

  REQUIRE(out.tag_re_ == "tag.*");
  REQUIRE(out.blob_re_ == "blob.*");
  REQUIRE(out.model_ == "model-x");
  REQUIRE(out.prompt_ == "prompt-key");
  REQUIRE(out.context_length_ == 12345);
  REQUIRE(out.num_predict_ == 99);
}

TEST_CASE("CreateParams - serialize round-trips config", "[cae][config][serialize]") {
  CreateParams in;
  in.LoadConfig(MakePoolConfig(
      "next_pool_id: \"700.2\"\n"
      "label_endpoint: \"http://ser\"\n"
      "label_prompts:\n"
      "  k: \"V\"\n"
      "label_matches:\n"
      "  - tag_re: \"tr\"\n"
      "    blob_re: \"br\"\n"
      "    model: \"mo\"\n"
      "    prompt: \"k\"\n"
      "    context_length: 2048\n"));

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
  REQUIRE(out.label_endpoint_ == "http://ser");
  REQUIRE(out.label_prompts_.size() == 1);
  REQUIRE(out.label_prompts_["k"] == "V");
  REQUIRE(out.label_matches_.size() == 1);
  REQUIRE(out.label_matches_[0].tag_re_ == "tr");
  REQUIRE(out.label_matches_[0].context_length_ == 2048);
}

SIMPLE_TEST_MAIN()
