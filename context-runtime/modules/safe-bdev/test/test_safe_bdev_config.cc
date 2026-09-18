/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

/**
 * Unit tests for safe_bdev::CreateParams -- the compose-time config, which no
 * other suite reaches: the chimod tests build their CreateParams in C++ and
 * hand it straight to GetOrCreatePool, so the `safe_bdev:` YAML a deployment
 * actually writes was only ever parsed in production.
 *
 * Coverage:
 *   A. LoadConfig: max_failures, alloc_log, and the `members` sequence in both
 *      of its spellings (by pool name, and by an already-created pool id).
 *   B. The keys' absence: a config that names none of them leaves the
 *      constructor's defaults alone.
 *   C. members replaces, rather than appends to, whatever was already there.
 *   D. CreateParams serialization round-trip -- the parsed config has to cross
 *      IPC intact to reach the container.
 *
 * Pure host test: CreateParams is header-only, so there is no server, no
 * shared memory and no member bdev in sight.
 */

#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

#include "clio_ctp/data_structures/serialization/local_serialize.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/safe_bdev/safe_bdev_tasks.h"
#include "simple_test.h"

using clio::run::safe_bdev::CreateParams;
using clio::run::safe_bdev::MemberBdevDesc;

namespace {

/** Build a PoolConfig whose YAML body is `body`. */
clio::run::PoolConfig MakePoolConfig(const std::string &body) {
  clio::run::PoolConfig cfg;
  cfg.mod_name_ = "clio_safe_bdev";
  cfg.pool_name_ = "safe_bdev_config_test";
  cfg.config_ = body;
  return cfg;
}

}  // namespace

TEST_CASE("SafeBdevConfig - LoadConfig parses max_failures and alloc_log",
          "[safe_bdev][config]") {
  CreateParams p;
  REQUIRE(p.max_failures_ == 1);
  REQUIRE(p.alloc_log_path_.empty());

  p.LoadConfig(MakePoolConfig(
      "max_failures: 2\n"
      "alloc_log: /tmp/safe_bdev_alloc.log\n"));

  REQUIRE(p.max_failures_ == 2);
  REQUIRE(p.alloc_log_path_ == "/tmp/safe_bdev_alloc.log");
}

TEST_CASE("SafeBdevConfig - LoadConfig parses members by pool name",
          "[safe_bdev][config]") {
  CreateParams p;
  p.LoadConfig(MakePoolConfig(
      "max_failures: 1\n"
      "members:\n"
      "  - pool_name: bdev_0\n"
      "    node_id: 0\n"
      "  - pool_name: bdev_1\n"
      "    node_id: 3\n"));

  REQUIRE(p.members_.size() == 2);
  REQUIRE(p.members_[0].pool_name_ == "bdev_0");
  REQUIRE(p.members_[0].node_id_ == 0);
  // Named members carry no pool id: the container resolves the name at create.
  REQUIRE(p.members_[0].pool_id_.IsNull());
  REQUIRE(p.members_[1].pool_name_ == "bdev_1");
  REQUIRE(p.members_[1].node_id_ == 3);
}

TEST_CASE("SafeBdevConfig - LoadConfig parses members by pool id",
          "[safe_bdev][config]") {
  CreateParams p;
  p.LoadConfig(MakePoolConfig(
      "members:\n"
      "  - pool_id_major: 4100\n"
      "    pool_id_minor: 7\n"
      // A member given without a minor is the pool's first container, so the
      // minor defaults to 0 rather than making the entry unusable.
      "  - pool_id_major: 4101\n"));

  REQUIRE(p.members_.size() == 2);
  REQUIRE(p.members_[0].pool_id_.major_ == 4100);
  REQUIRE(p.members_[0].pool_id_.minor_ == 7);
  REQUIRE(p.members_[1].pool_id_.major_ == 4101);
  REQUIRE(p.members_[1].pool_id_.minor_ == 0);
}

TEST_CASE("SafeBdevConfig - LoadConfig leaves unnamed keys at their defaults",
          "[safe_bdev][config]") {
  CreateParams p(3, {MemberBdevDesc("bdev_preset", 1)}, "/tmp/preset.log");

  // A config that mentions none of this module's keys must not blank out what
  // the caller already set -- compose passes the remaining-config blob through
  // whether or not it has a safe-bdev section in it.
  p.LoadConfig(MakePoolConfig("unrelated_key: 5\n"));

  REQUIRE(p.max_failures_ == 3);
  REQUIRE(p.alloc_log_path_ == "/tmp/preset.log");
  REQUIRE(p.members_.size() == 1);
  REQUIRE(p.members_[0].pool_name_ == "bdev_preset");
}

TEST_CASE("SafeBdevConfig - LoadConfig replaces the member list",
          "[safe_bdev][config]") {
  CreateParams p(1, {MemberBdevDesc("stale_0", 0), MemberBdevDesc("stale_1", 0)},
                 "");

  p.LoadConfig(MakePoolConfig(
      "members:\n"
      "  - pool_name: fresh_0\n"));

  // Not appended: a config's members ARE the membership, and an appended list
  // would silently widen the EC group past what the deployment asked for.
  REQUIRE(p.members_.size() == 1);
  REQUIRE(p.members_[0].pool_name_ == "fresh_0");
}

TEST_CASE("SafeBdevConfig - a scalar members key is ignored",
          "[safe_bdev][config]") {
  CreateParams p(2, {MemberBdevDesc("kept", 0)}, "");

  // `members: none` is a map entry but not a sequence; parsing it as one would
  // throw out of a compose that is otherwise valid.
  p.LoadConfig(MakePoolConfig("members: none\n"));

  REQUIRE(p.members_.size() == 1);
  REQUIRE(p.members_[0].pool_name_ == "kept");
}

TEST_CASE("SafeBdevConfig - CreateParams round-trips through cereal",
          "[safe_bdev][config][serialize]") {
  CreateParams src;
  src.LoadConfig(MakePoolConfig(
      "max_failures: 2\n"
      "alloc_log: /tmp/rt.log\n"
      "members:\n"
      "  - pool_name: bdev_a\n"
      "    node_id: 1\n"
      "  - pool_name: bdev_b\n"
      "    node_id: 2\n"));

  std::string buf;
  {
    ctp::ipc::LocalSerialize<std::string> save(buf);
    save(src);
    save.Finalize();
  }
  REQUIRE(!buf.empty());

  CreateParams dst;
  {
    ctp::ipc::LocalDeserialize<std::string> load(buf);
    load(dst);
  }

  REQUIRE(dst.max_failures_ == 2);
  REQUIRE(dst.alloc_log_path_ == "/tmp/rt.log");
  REQUIRE(dst.members_.size() == 2);
  REQUIRE(dst.members_[0].pool_name_ == "bdev_a");
  REQUIRE(dst.members_[0].node_id_ == 1);
  REQUIRE(dst.members_[1].pool_name_ == "bdev_b");
  REQUIRE(dst.members_[1].node_id_ == 2);
}

SIMPLE_TEST_MAIN()
