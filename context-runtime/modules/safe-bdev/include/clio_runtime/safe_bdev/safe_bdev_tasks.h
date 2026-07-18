/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SAFE_BDEV_TASKS_H_
#define SAFE_BDEV_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/config_manager.h>
#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

#include "autogen/safe_bdev_methods.h"
// Include admin tasks for BaseCreateTask / GetOrCreatePoolTask
#include <clio_runtime/admin/admin_tasks.h>
// Reuse the bdev task types (Block, AllocateBlocksTask, etc.)
#include <clio_runtime/bdev/bdev_tasks.h>

/**
 * Task struct definitions for safe_bdev
 *
 * safe_bdev is a declustered, erasure-coded block device that aggregates one
 * or more (potentially remote) member bdevs. It REUSES the bdev module's task
 * types for the data-plane operations (AllocateBlocks/FreeBlocks/Write/Read/
 * GetStats) and adds management tasks for membership and parity maintenance.
 */

namespace clio::run::safe_bdev {

// ---------------------------------------------------------------------------
// Reused task types from the bdev module. They are brought into this namespace
// with `using` so the autogen dispatcher can reference them unqualified, and so
// their method_ ids match this module's method ids (10..14).
// ---------------------------------------------------------------------------
using clio::run::bdev::Block;
using clio::run::bdev::PerfMetrics;
using clio::run::bdev::AllocateBlocksTask;
using clio::run::bdev::FreeBlocksTask;
using clio::run::bdev::WriteTask;
using clio::run::bdev::ReadTask;
using clio::run::bdev::GetStatsTask;

using MonitorTask = clio::run::admin::MonitorTask;
using DestroyTask = clio::run::admin::DestroyTask;

/**
 * Role of a member bdev within the declustered stripe layout.
 */
enum class MemberRole : clio::run::u32 {
  kData = 0,    // Holds data shards
  kParity = 1,  // Holds parity shards
  kSpare = 2    // Idle spare, available for recovery targets
};

/**
 * State of a member bdev.
 */
enum class MemberState : clio::run::u32 {
  kActive = 0,      // Healthy and serving I/O
  kFaulty = 1,      // Failed / unreachable
  kRecovering = 2   // Being reconstructed onto a spare
};

/**
 * MemberBdev — runtime POD describing one member bdev (may be remote).
 *
 * Kept cereal-serializable (no in-shm pointers) so it can live in plain
 * std::vector member state inside the Runtime and be embedded in non-Task
 * structures. Strings use std::string here because this struct does NOT travel
 * through the SHM Task path directly.
 */
struct MemberBdev {
  clio::run::PoolId pool_id_;             // Pool id of the member bdev
  std::string pool_name_;           // Human/file name of the member bdev
  clio::run::u32 node_id_;                // Node hosting the member (for remote members)
  clio::run::u32 role_;                   // MemberRole (0=data,1=parity,2=spare)
  clio::run::u32 state_;                  // MemberState (0=active,1=faulty,2=recovering)

  MemberBdev()
      : pool_id_(),
        pool_name_(),
        node_id_(0),
        role_(static_cast<clio::run::u32>(MemberRole::kData)),
        state_(static_cast<clio::run::u32>(MemberState::kActive)) {}

  MemberBdev(const clio::run::PoolId &pool_id, const std::string &pool_name,
             clio::run::u32 node_id, clio::run::u32 role, clio::run::u32 state)
      : pool_id_(pool_id),
        pool_name_(pool_name),
        node_id_(node_id),
        role_(role),
        state_(state) {}

  // Cereal serialization (mirror Block::serialize idiom).
  template <class Archive>
  void serialize(Archive &ar) {
    ar(pool_id_, pool_name_, node_id_, role_, state_);
  }
};

/**
 * MemberBdevDesc — user-facing descriptor of a member bdev passed at create
 * time. Plain serializable struct (goes through cereal in CreateParams).
 */
struct MemberBdevDesc {
  std::string pool_name_;  // Name/path of the member bdev pool
  clio::run::u32 node_id_;       // Node id hosting the member (0 = local)
  clio::run::PoolId pool_id_;    // Pool id of the member bdev (caller created it)

  MemberBdevDesc() : pool_name_(), node_id_(0), pool_id_() {}
  MemberBdevDesc(const std::string &pool_name, clio::run::u32 node_id)
      : pool_name_(pool_name), node_id_(node_id), pool_id_() {}
  MemberBdevDesc(const std::string &pool_name, clio::run::u32 node_id,
                 const clio::run::PoolId &pool_id)
      : pool_name_(pool_name), node_id_(node_id), pool_id_(pool_id) {}

  template <class Archive>
  void serialize(Archive &ar) {
    ar(pool_name_, node_id_, pool_id_);
  }
};

/**
 * CreateParams for safe_bdev chimod.
 * This is its OWN CreateParams (NOT bdev's). It carries the EC fault tolerance
 * target and the initial member-bdev descriptor list. Serialized through cereal
 * (lives in chimod_params_), so std::vector / std::string are fine here.
 */
struct CreateParams {
  clio::run::u32 max_failures_;                  // Max simultaneous member failures to tolerate
  std::vector<MemberBdevDesc> members_;    // Initial member bdev descriptors
  // Path to the persistent allocator-state log (WAL). Empty => logging disabled
  // (no file created, no behavior change). When set, the per-group allocators
  // and the append-only group structure are persisted here and recovered on a
  // subsequent create over the same members + same path.
  std::string alloc_log_path_;

  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "clio_safe_bdev";

  CreateParams() : max_failures_(1), members_(), alloc_log_path_() {}

  CreateParams(clio::run::u32 max_failures,
               const std::vector<MemberBdevDesc> &members,
               const std::string &alloc_log_path = "")
      : max_failures_(max_failures),
        members_(members),
        alloc_log_path_(alloc_log_path) {}

  // Serialization support for cereal
  template <class Archive>
  void serialize(Archive &ar) {
    ar(max_failures_, members_, alloc_log_path_);
  }

  /**
   * Load configuration from PoolConfig (for compose mode).
   * Mirrors bdev's LoadConfig style; parses `max_failures` and a `members`
   * YAML sequence of {pool_name, node_id}.
   */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    YAML::Node config = YAML::Load(pool_config.config_);

    if (config["max_failures"]) {
      max_failures_ = config["max_failures"].as<clio::run::u32>();
    }

    if (config["alloc_log"]) {
      alloc_log_path_ = config["alloc_log"].as<std::string>();
    }

    if (config["members"] && config["members"].IsSequence()) {
      members_.clear();
      for (const auto &m : config["members"]) {
        MemberBdevDesc desc;
        if (m["pool_name"]) {
          desc.pool_name_ = m["pool_name"].as<std::string>();
        }
        if (m["node_id"]) {
          desc.node_id_ = m["node_id"].as<clio::run::u32>();
        }
        if (m["pool_id_major"]) {
          desc.pool_id_ = clio::run::PoolId(
              m["pool_id_major"].as<clio::run::u32>(),
              m["pool_id_minor"] ? m["pool_id_minor"].as<clio::run::u32>() : 0);
        }
        members_.push_back(desc);
      }
    }
  }
};

/**
 * CreateTask - Initialize the safe_bdev container.
 * Type alias for GetOrCreatePoolTask with this module's CreateParams.
 */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * AddBdevTask - Add a new member bdev to the device (Method::kAddBdev).
 */
struct AddBdevTask : public clio::run::Task {
  IN clio::run::priv::string pool_name_;  // Name/path of the member bdev to add
  IN clio::run::u32 node_id_;             // Node id hosting the member (0 = local)
  IN clio::run::PoolId member_pool_id_;   // Pool id of the member bdev to add
  IN clio::run::u32 as_parity_;           // Non-zero => add as a parity drive

  /** SHM default constructor */
  CTP_CROSS_FUN AddBdevTask()
      : clio::run::Task(),
        pool_name_(CLIO_PRIV_ALLOC),
        node_id_(0),
        member_pool_id_(),
        as_parity_(0) {}

  /** Emplace constructor */
  explicit AddBdevTask(const clio::run::TaskId &task_node, const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query,
                       const std::string &pool_name, clio::run::u32 node_id,
                       const clio::run::PoolId &member_pool_id, clio::run::u32 as_parity)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kAddBdev),
        pool_name_(CLIO_PRIV_ALLOC, pool_name),
        node_id_(node_id),
        member_pool_id_(member_pool_id),
        as_parity_(as_parity) {}

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(pool_name_, node_id_, member_pool_id_, as_parity_);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }

  /** Copy from another AddBdevTask */
  void Copy(const ctp::ipc::FullPtr<AddBdevTask> &other) {
    Task::Copy(other.template Cast<Task>());
    pool_name_ = other->pool_name_;
    node_id_ = other->node_id_;
    member_pool_id_ = other->member_pool_id_;
    as_parity_ = other->as_parity_;
  }

  /** Aggregate replica results into this task */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<AddBdevTask>());
  }
};

/**
 * RemoveBdevTask - Remove a member bdev (Method::kRemoveBdev).
 */
struct RemoveBdevTask : public clio::run::Task {
  IN clio::run::PoolId target_pool_id_;  // Member pool to remove
  IN clio::run::u32 was_faulty_;         // Non-zero if the member failed (vs. clean removal)

  /** SHM default constructor */
  CTP_CROSS_FUN RemoveBdevTask()
      : clio::run::Task(), target_pool_id_(), was_faulty_(0) {}

  /** Emplace constructor */
  explicit RemoveBdevTask(const clio::run::TaskId &task_node,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query,
                          const clio::run::PoolId &target_pool_id,
                          clio::run::u32 was_faulty)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kRemoveBdev),
        target_pool_id_(target_pool_id),
        was_faulty_(was_faulty) {}

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(target_pool_id_, was_faulty_);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }

  /** Copy from another RemoveBdevTask */
  void Copy(const ctp::ipc::FullPtr<RemoveBdevTask> &other) {
    Task::Copy(other.template Cast<Task>());
    target_pool_id_ = other->target_pool_id_;
    was_faulty_ = other->was_faulty_;
  }

  /** Aggregate replica results into this task */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<RemoveBdevTask>());
  }
};

/**
 * RecoverBdevTask - Reconstruct a failed member onto a new member
 * (Method::kRecoverBdev).
 */
struct RecoverBdevTask : public clio::run::Task {
  IN clio::run::PoolId old_bdev_id_;      // Failed member to reconstruct
  IN clio::run::priv::string pool_name_;  // New member bdev name/path
  IN clio::run::u32 node_id_;             // Node id hosting the new member
  IN clio::run::PoolId new_pool_id_;      // Pool id of the fresh member bdev

  /** SHM default constructor */
  CTP_CROSS_FUN RecoverBdevTask()
      : clio::run::Task(),
        old_bdev_id_(),
        pool_name_(CLIO_PRIV_ALLOC),
        node_id_(0),
        new_pool_id_() {}

  /** Emplace constructor */
  explicit RecoverBdevTask(const clio::run::TaskId &task_node,
                           const clio::run::PoolId &pool_id,
                           const clio::run::PoolQuery &pool_query,
                           const clio::run::PoolId &old_bdev_id,
                           const std::string &pool_name, clio::run::u32 node_id,
                           const clio::run::PoolId &new_pool_id)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kRecoverBdev),
        old_bdev_id_(old_bdev_id),
        pool_name_(CLIO_PRIV_ALLOC, pool_name),
        node_id_(node_id),
        new_pool_id_(new_pool_id) {}

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(old_bdev_id_, pool_name_, node_id_, new_pool_id_);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }

  /** Copy from another RecoverBdevTask */
  void Copy(const ctp::ipc::FullPtr<RecoverBdevTask> &other) {
    Task::Copy(other.template Cast<Task>());
    old_bdev_id_ = other->old_bdev_id_;
    pool_name_ = other->pool_name_;
    node_id_ = other->node_id_;
    new_pool_id_ = other->new_pool_id_;
  }

  /** Aggregate replica results into this task */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<RecoverBdevTask>());
  }
};

/**
 * BuildParityTask - Periodic background task that drains the dirty-stripe log
 * and raises stripes to the current target parity level (Method::kBuildParity).
 * Submitted with SetPeriod + TASK_PERIODIC.
 */
struct BuildParityTask : public clio::run::Task {
  IN clio::run::u32 max_batch_;  // Max number of stripes to (re)build per pass

  /** SHM default constructor */
  CTP_CROSS_FUN BuildParityTask() : clio::run::Task(), max_batch_(0) {}

  /** Emplace constructor */
  explicit BuildParityTask(const clio::run::TaskId &task_node,
                           const clio::run::PoolId &pool_id,
                           const clio::run::PoolQuery &pool_query,
                           clio::run::u32 max_batch = 0)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kBuildParity),
        max_batch_(max_batch) {}

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(max_batch_);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }

  /** Copy from another BuildParityTask */
  void Copy(const ctp::ipc::FullPtr<BuildParityTask> &other) {
    Task::Copy(other.template Cast<Task>());
    max_batch_ = other->max_batch_;
  }

  /** Aggregate replica results into this task */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<BuildParityTask>());
  }
};

/**
 * FlushAllocLogTask - Periodic task that flushes (and compacts) the persistent
 * allocator-state log (WAL). Registered as TASK_PERIODIC from Create when an
 * alloc_log_path is configured. Carries no I/O parameters. Mirrors bdev's
 * FlushAllocLogTask.
 */
struct FlushAllocLogTask : public clio::run::Task {
  /** SHM default constructor */
  CTP_CROSS_FUN FlushAllocLogTask() : clio::run::Task() {}

  /** Emplace constructor */
  CTP_CROSS_FUN explicit FlushAllocLogTask(const clio::run::TaskId &task_node,
                                           const clio::run::PoolId &pool_id,
                                           const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kFlushAllocLog) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kFlushAllocLog;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }

  /** Copy from another FlushAllocLogTask */
  void Copy(const ctp::ipc::FullPtr<FlushAllocLogTask> &other) {
    Task::Copy(other.template Cast<Task>());
  }

  /** Aggregate replica results into this task */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<FlushAllocLogTask>());
  }
};

}  // namespace clio::run::safe_bdev

#endif  // SAFE_BDEV_TASKS_H_
