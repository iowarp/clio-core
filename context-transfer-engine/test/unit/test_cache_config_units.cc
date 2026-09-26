/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * test_cache_config_units.cc
 *
 * Pure unit tests (no CLIO runtime) for the header-only cache chimod config
 * that the interpose/stack suites never reach: they compose the cache pool
 * from a C++ CacheConfig, so the `cache:` YAML a deployment writes -- the
 * next_pool_id that points this layer at the rest of the chain, and the score
 * floor that keeps hot raw bytes resident -- was only ever parsed in
 * production.
 *
 * Coverage:
 *   - CacheConfig::LoadConfig: next_pool_id, min_score, the empty-config early
 *     return, the dotless next_pool_id that is ignored rather than guessed at,
 *     and the best-effort catch that keeps malformed YAML from taking the
 *     compose down with it.
 *   - CacheConfig serialize round-trip and the compose pool-id constructor,
 *     both of which have to carry the parsed config to the container.
 *
 * These paths live in cache_tasks.h and compile straight into this TU, so no
 * runtime, CTE client, or network is required.
 */

#include "simple_test.h"

#include <clio_cte/cache/cache_tasks.h>
#include <clio_runtime/config_manager.h>

#include "clio_ctp/data_structures/serialization/global_serialize.h"

#include <string>
#include <vector>

using clio::cte::cache::CacheConfig;

namespace {

// Build a PoolConfig carrying `yaml` as its remaining-config blob.
clio::run::PoolConfig MakePoolConfig(const std::string &yaml) {
  clio::run::PoolConfig pc;
  pc.mod_name_ = "clio_cte_cache";
  pc.pool_name_ = "cache_config_test";
  pc.config_ = yaml;
  return pc;
}

}  // namespace

TEST_CASE("CacheConfig - LoadConfig empty config is a no-op",
          "[cache][config][loadconfig]") {
  CacheConfig cfg;
  // Empty config_ hits the early `return` before the YAML parse.
  cfg.LoadConfig(MakePoolConfig(""));
  REQUIRE(cfg.next_pool_id_.IsNull());
  REQUIRE(cfg.min_score_ == 0.5f);
}

TEST_CASE("CacheConfig - LoadConfig parses next_pool_id",
          "[cache][config][loadconfig]") {
  CacheConfig cfg;
  cfg.LoadConfig(MakePoolConfig("next_pool_id: \"562.0\"\n"));
  REQUIRE(cfg.next_pool_id_.major_ == 562);
  REQUIRE(cfg.next_pool_id_.minor_ == 0);
}

TEST_CASE("CacheConfig - LoadConfig ignores next_pool_id without a dot",
          "[cache][config][loadconfig]") {
  CacheConfig cfg;
  // No minor => no id. Defaulting the minor here would point the chain at a
  // container that may not be the one the deployment meant.
  cfg.LoadConfig(MakePoolConfig("next_pool_id: \"nominor\"\n"));
  REQUIRE(cfg.next_pool_id_.IsNull());
}

TEST_CASE("CacheConfig - LoadConfig parses min_score",
          "[cache][config][loadconfig]") {
  CacheConfig cfg;
  cfg.LoadConfig(MakePoolConfig("min_score: 0.25\n"));
  REQUIRE(cfg.min_score_ > 0.24f);
  REQUIRE(cfg.min_score_ < 0.26f);
  // A config that names only min_score leaves the chain link alone.
  REQUIRE(cfg.next_pool_id_.IsNull());
}

TEST_CASE("CacheConfig - LoadConfig parses both keys together",
          "[cache][config][loadconfig]") {
  CacheConfig cfg;
  cfg.LoadConfig(MakePoolConfig(
      "next_pool_id: \"512.3\"\n"
      "min_score: 0.75\n"));
  REQUIRE(cfg.next_pool_id_.major_ == 512);
  REQUIRE(cfg.next_pool_id_.minor_ == 3);
  REQUIRE(cfg.min_score_ > 0.74f);
  REQUIRE(cfg.min_score_ < 0.76f);
}

TEST_CASE("CacheConfig - LoadConfig swallows malformed YAML",
          "[cache][config][loadconfig]") {
  CacheConfig cfg;
  // Config parsing is best-effort: a broken blob must not escape the
  // catch(...) and abort the compose that carried it.
  REQUIRE_NOTHROW(cfg.LoadConfig(MakePoolConfig("next_pool_id: [ {a: ")));
  REQUIRE(cfg.next_pool_id_.IsNull());
}

TEST_CASE("CacheConfig - compose pool-id constructor copies config",
          "[cache][config][ctor]") {
  CacheConfig src;
  src.LoadConfig(MakePoolConfig(
      "next_pool_id: \"562.1\"\n"
      "min_score: 0.9\n"));

  // The pool_id argument is intentionally ignored by the ctor; it exists to
  // match the compose pattern the other chimods use.
  clio::run::PoolId pid(563, 0);
  CacheConfig composed(pid, src);
  REQUIRE(composed.next_pool_id_.major_ == 562);
  REQUIRE(composed.next_pool_id_.minor_ == 1);
  REQUIRE(composed.min_score_ > 0.89f);
  REQUIRE(composed.min_score_ < 0.91f);
}

TEST_CASE("CacheConfig - serialize round-trips every field",
          "[cache][config][serialize]") {
  CacheConfig in;
  in.LoadConfig(MakePoolConfig(
      "next_pool_id: \"700.2\"\n"
      "min_score: 0.125\n"));

  std::vector<char> buf;
  {
    ctp::ipc::GlobalSerialize<std::vector<char>> oa(buf);
    in.serialize(oa);
    oa.Finalize();
  }

  CacheConfig out;
  {
    ctp::ipc::GlobalDeserialize<std::vector<char>> ia(buf);
    out.serialize(ia);
  }

  REQUIRE(out.next_pool_id_.major_ == 700);
  REQUIRE(out.next_pool_id_.minor_ == 2);
  // min_score_ is the field a serialize() audit would most easily drop: it has
  // a sane default, so a lost value looks like a working cache that quietly
  // ignores the deployment's floor.
  REQUIRE(out.min_score_ > 0.124f);
  REQUIRE(out.min_score_ < 0.126f);
}

SIMPLE_TEST_MAIN()
