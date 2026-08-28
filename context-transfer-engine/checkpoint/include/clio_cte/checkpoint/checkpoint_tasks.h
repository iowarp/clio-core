/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_CHECKPOINT_CHECKPOINT_TASKS_H_
#define CLIO_CTE_CHECKPOINT_CHECKPOINT_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/checkpoint/autogen/checkpoint_methods.h>

#include <string>

namespace clio::cte::checkpoint {

/** The checkpoint chimod speaks the CTE core's vocabulary. */
using Context = clio::cte::core::Context;
using TagId = clio::cte::core::TagId;

/** Well-known default pool id/name for the fault handler. */
static constexpr clio::run::PoolId kCheckpointPoolId(565, 0);
static constexpr const char *kCheckpointPoolName = "clio_cte_checkpoint";

/**
 * Container creation params. next_pool_id_ is the pool the handler reads
 * sources from and materialises copies into (usually the core, or the top
 * of an interposition chain).
 */
struct CheckpointConfig {
  static constexpr const char* chimod_lib_name = "clio_cte_checkpoint";

  clio::run::PoolId next_pool_id_;  ///< Core (or chain top) this resolves via

  CheckpointConfig() : next_pool_id_(clio::run::PoolId::GetNull()) {}
  CheckpointConfig(const clio::run::PoolId &pool_id,
                   const CheckpointConfig &other)
      : next_pool_id_(other.next_pool_id_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    // EVERY field round-trips (the dropped-field audit rule).
    ar(next_pool_id_);
  }

  /** Load configuration from compose YAML. */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (!pool_config.config_.empty()) {
      try {
        YAML::Node node = YAML::Load(pool_config.config_);
        if (node["next_pool_id"]) {
          std::string next_str = node["next_pool_id"].as<std::string>();
          auto dot = next_str.find('.');
          if (dot != std::string::npos) {
            clio::run::u32 major = std::stoul(next_str.substr(0, dot));
            clio::run::u32 minor = std::stoul(next_str.substr(dot + 1));
            next_pool_id_ = clio::run::PoolId(major, minor);
          }
        }
      } catch (...) {
        // Config parsing is best-effort
      }
    }
  }
};

/** Standard pool creation. */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<CheckpointConfig>;

/** Cleanup the checkpoint container. */
struct DestroyTask : public clio::run::Task {
  DestroyTask() : clio::run::Task() {}

  explicit DestroyTask(const clio::run::TaskId &task_id,
                       const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDestroy) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    // OUT fields ONLY -- never Copy() (issue #915). No OUT fields here.
  }

  void Copy(const ctp::ipc::FullPtr<DestroyTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
  }

  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

using MonitorTask = clio::run::admin::MonitorTask;

}  // namespace clio::cte::checkpoint

#endif  // CLIO_CTE_CHECKPOINT_CHECKPOINT_TASKS_H_
