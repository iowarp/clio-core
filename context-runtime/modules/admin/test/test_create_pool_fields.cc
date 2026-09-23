/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

/**
 * CreatePoolFields — the base every ChiMod's create task shares.
 *
 * Its AggregateOut is what a broadcast pool creation runs once per surviving
 * node, and two of its branches decide whether a create SUCCEEDS:
 *
 *   - the first replica to report a pool id wins, and a later empty answer
 *     must not erase it;
 *   - a replica that could not be delivered because its node DIED is forgiven
 *     (issue #856) -- without that, one unreachable node turns an otherwise
 *     successful create into an error, which is what the leader-election suite
 *     asserts on right after it kills a node.
 *
 * Neither branch had a test. They only fire with a foreign replica in hand,
 * which no single-node suite produces, so the nearest coverage was a
 * multi-node cluster run -- far from the three lines that encode the rule.
 * Aggregating is pure field arithmetic on two tasks, though, so a replica can
 * simply be built here: no runtime, no cluster, no sockets.
 */

#include "simple_test.h"

#include <clio_runtime/admin/admin_tasks.h>

#include <string>

using clio::run::admin::CreateTask;
using clio::run::admin::CreatePoolFields;

namespace {

/** A create task addressed at pool @p pool_major (0 => a null pool id). */
CreateTask MakeCreateTask(clio::run::u32 pool_major) {
  return CreateTask(clio::run::TaskId(1, 2, 3), clio::run::kAdminPoolId,
                    clio::run::PoolQuery::Local(), "clio_admin",
                    std::string("test_pool"),
                    clio::run::PoolId(pool_major, 0), nullptr);
}

/** Aggregate @p replica into @p origin the way the generated dispatch does. */
void Aggregate(CreateTask *origin, CreateTask *replica) {
  origin->AggregateOut(ctp::ipc::FullPtr<clio::run::Task>(replica));
}

}  // namespace

TEST_CASE("CreatePoolFields - the emplace constructor fills the shared fields",
          "[admin][create_pool_fields]") {
  CreateTask task = MakeCreateTask(42);

  REQUIRE(task.chimod_name_.str() == "clio_admin");
  REQUIRE(task.pool_name_.str() == "test_pool");
  REQUIRE(task.new_pool_id_.major_ == 42);
  REQUIRE(task.error_message_.size() == 0);
  // The flags come from CreateTask's template arguments, which the base cannot
  // see -- it takes them as constructor parameters instead.
  REQUIRE(task.is_admin_);
  REQUIRE_FALSE(task.do_compose_);

  // The whole point of the base: a create task IS one, so this conversion is
  // ordinary rather than a reinterpretation. The runtime depends on it.
  CreatePoolFields *fields = &task;
  REQUIRE(fields->new_pool_id_.major_ == 42);
}

TEST_CASE("CreatePoolFields - the first replica to report a pool id wins",
          "[admin][create_pool_fields][aggregate]") {
  CreateTask origin = MakeCreateTask(0);  // nothing reported yet
  REQUIRE(origin.new_pool_id_.IsNull());

  CreateTask replica = MakeCreateTask(301);
  Aggregate(&origin, &replica);
  REQUIRE(origin.new_pool_id_.major_ == 301);

  // Every replica creates the same pool, so a second answer changes nothing --
  // and an EMPTY second answer must not erase what the first reported.
  CreateTask empty = MakeCreateTask(0);
  Aggregate(&origin, &empty);
  REQUIRE(origin.new_pool_id_.major_ == 301);
}

TEST_CASE("CreatePoolFields - a replica's error message reaches the origin",
          "[admin][create_pool_fields][aggregate]") {
  CreateTask origin = MakeCreateTask(301);
  CreateTask replica = MakeCreateTask(301);
  replica.error_message_ = "target node refused the create";

  Aggregate(&origin, &replica);
  REQUIRE(origin.error_message_.str() == "target node refused the create");

  // First message wins, for the same reason the first pool id does: a later
  // silent replica must not blank out the one explanation anybody has.
  CreateTask quiet = MakeCreateTask(301);
  Aggregate(&origin, &quiet);
  REQUIRE(origin.error_message_.str() == "target node refused the create");
}

TEST_CASE("CreatePoolFields - an undeliverable replica does not fail the create",
          "[admin][create_pool_fields][aggregate]") {
  // Issue #856. The pool genuinely exists once a node created it and reported
  // its id; a replica aimed at a node that died is not evidence otherwise.
  CreateTask origin = MakeCreateTask(301);
  CreateTask dead_node = MakeCreateTask(301);
  dead_node.SetReturnCode(clio::run::kRun2RunNetworkTimeoutRC);

  Aggregate(&origin, &dead_node);
  REQUIRE(origin.new_pool_id_.major_ == 301);
  REQUIRE(origin.GetReturnCode() == 0);
}

TEST_CASE("CreatePoolFields - only the network timeout is forgiven",
          "[admin][create_pool_fields][aggregate]") {
  // A create that genuinely failed still propagates: forgiving every non-zero
  // code would turn a real error into a silent half-built pool.
  CreateTask origin = MakeCreateTask(301);
  CreateTask failed = MakeCreateTask(301);
  const int kRealFailure = 7;
  failed.SetReturnCode(kRealFailure);

  Aggregate(&origin, &failed);
  REQUIRE(origin.GetReturnCode() == kRealFailure);
}

TEST_CASE("CreatePoolFields - a timeout with no pool id still fails",
          "[admin][create_pool_fields][aggregate]") {
  // Forgiveness is conditional on SOMEBODY having created the pool. With no id
  // reported, an undeliverable replica is the only news there is.
  CreateTask origin = MakeCreateTask(0);
  CreateTask dead_node = MakeCreateTask(0);
  dead_node.SetReturnCode(clio::run::kRun2RunNetworkTimeoutRC);

  Aggregate(&origin, &dead_node);
  REQUIRE(origin.new_pool_id_.IsNull());
  REQUIRE(origin.GetReturnCode() == clio::run::kRun2RunNetworkTimeoutRC);
}

SIMPLE_TEST_MAIN()
