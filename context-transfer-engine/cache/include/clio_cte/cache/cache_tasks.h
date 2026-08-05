/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_CACHE_CACHE_TASKS_H_
#define CLIO_CTE_CACHE_CACHE_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/cache/autogen/cache_methods.h>

#include <string>

namespace clio::cte::cache {

/** The cache chimod speaks the CTE core's vocabulary. */
using Context = clio::cte::core::Context;
using TagId = clio::cte::core::TagId;

/**
 * Well-known default pool id/name — the TOP of the standard clio-fs chain
 * (cache 563 -> compressor 562 -> replication 561 -> core 512).
 */
static constexpr clio::run::PoolId kCachePoolId(563, 0);
static constexpr const char *kCachePoolName = "clio_cte_cache";

/**
 * Container creation params. next_pool_id_ is the pool this cache pushes
 * authoritative bytes to (usually the compressor). Semantics are
 * ASYNCHRONOUS WRITE-THROUGH: puts land below before the ack; there is no
 * dirty state and no flush period at this layer.
 */
struct CacheConfig {
  static constexpr const char* chimod_lib_name = "clio_cte_cache";

  clio::run::PoolId next_pool_id_;  ///< Next pool in the chain (e.g. 562.0)
  /// Score floor for the uncompressed cache copies (propagated as
  /// Context::replica_min_score_): the organizer never rescores a cache
  /// replica below it; only capacity eviction under genuine tier pressure
  /// reclaims one. Modest default keeps hot raw bytes resident without
  /// starving primaries of the fast tier.
  float min_score_ = 0.5f;

  CacheConfig() : next_pool_id_(clio::run::PoolId::GetNull()) {}
  CacheConfig(const clio::run::PoolId &pool_id, const CacheConfig &other)
      : next_pool_id_(other.next_pool_id_),
        min_score_(other.min_score_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    // EVERY field round-trips — the audit rule three silently-dropped-field
    // bugs on this branch bought us (replication copy-ctor, ReplicationConfig
    // period, CompressorConfig next_pool_id_).
    ar(next_pool_id_, min_score_);
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
        if (node["min_score"]) {
          min_score_ = node["min_score"].as<float>();
        }
      } catch (...) {
        // Config parsing is best-effort
      }
    }
  }
};

/** Standard pool creation. */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<CacheConfig>;

/** Cleanup the cache container. */
struct DestroyTask : public clio::run::Task {
  DestroyTask() : clio::run::Task() {}

  explicit DestroyTask(const clio::run::TaskId &task_id,
                       const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDestroy) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<DestroyTask>());
  }

  void Copy(const ctp::ipc::FullPtr<DestroyTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
  }

  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

using MonitorTask = clio::run::admin::MonitorTask;

}  // namespace clio::cte::cache

#endif  // CLIO_CTE_CACHE_CACHE_TASKS_H_
