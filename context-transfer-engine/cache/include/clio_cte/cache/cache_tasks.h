/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_CACHE_CACHE_TASKS_H_
#define CLIO_CTE_CACHE_CACHE_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/msan.h>
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
  /// Tag-scoped caching of the GPU pager path (PodMultiGetBlob). false: only
  /// tags enrolled with EnrollTag are cached there. true: every tag a pager
  /// fetches through this pool is enrolled on first use (`auto_enroll: true`).
  bool auto_enroll_ = false;

  CacheConfig() : next_pool_id_(clio::run::PoolId::GetNull()) {}
  CacheConfig(const clio::run::PoolId &pool_id, const CacheConfig &other)
      : next_pool_id_(other.next_pool_id_),
        min_score_(other.min_score_),
        auto_enroll_(other.auto_enroll_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    // EVERY field round-trips — the audit rule three silently-dropped-field
    // bugs on this branch bought us (replication copy-ctor, ReplicationConfig
    // period, CompressorConfig next_pool_id_).
    ar(next_pool_id_, min_score_, auto_enroll_);
  }

  /** Load configuration from compose YAML. */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (!pool_config.config_.empty()) {
      try {
        YAML::Node node = YAML::Load(pool_config.config_);
        // yaml-cpp is a prebuilt .so: the scalars its scanner just
        // produced carry no MSan shadow, so every key lookup and
        // .as<>() below reads memory it has no record of. One walk
        // here covers the whole tree.
        ctp::MsanUnpoisonYaml(node);
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
        if (node["auto_enroll"]) {
          auto_enroll_ = node["auto_enroll"].as<bool>();
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
    // OUT fields ONLY -- never Copy() (issue #915): a whole-task assignment
    // destroys this ORIGIN's identity and re-assigns IN shm members across
    // allocator segments. See Task::AggregateOut for the full contract.
    // This task declares no OUT fields, so the base call above (return code +
    // completer) is the entire merge.
  }

  void Copy(const ctp::ipc::FullPtr<DestroyTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
  }

  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

using MonitorTask = clio::run::admin::MonitorTask;

/**
 * Tag-scoped cache control (kEnrollTag / kInvalidateTag / kForgetTag). Sent
 * as a Broadcast so every node's cache container applies it; dropped_ sums
 * the cached copies released across the replies.
 */
struct TagCacheTask : public clio::run::Task {
  IN TagId tag_id_;
  OUT clio::run::u64 dropped_;

  TagCacheTask() : clio::run::Task(), tag_id_(TagId::GetNull()), dropped_(0) {}
  explicit TagCacheTask(const clio::run::TaskId &task_id,
                        const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query,
                        clio::run::u32 method, const TagId &tag_id)
      : clio::run::Task(task_id, pool_id, pool_query, method),
        tag_id_(tag_id),
        dropped_(0) {
    task_flags_.Clear();
  }
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_);
  }
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(dropped_);
  }
  void Copy(const ctp::ipc::FullPtr<TagCacheTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_id_ = other->tag_id_;
    dropped_ = other->dropped_;
  }
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    dropped_ += other_base.template Cast<TagCacheTask>()->dropped_;
  }
};

/**
 * Tag-metadata cache update (kTagCachePut / kTagCacheDrop), broadcast by the
 * node that completed a tag write so every node's cache agrees before the
 * write returns. Put carries the tag's static metadata: name, id, and its
 * fault handler (null pool = none).
 */
struct TagCacheUpdateTask : public clio::run::Task {
  IN clio::run::priv::string tag_name_;
  IN TagId tag_id_;
  IN clio::run::PoolId fault_pool_id_;
  IN clio::run::priv::string fault_pool_name_;
  IN clio::run::priv::string fault_params_;

  TagCacheUpdateTask()
      : clio::run::Task(),
        tag_name_(CLIO_PRIV_ALLOC),
        tag_id_(TagId::GetNull()),
        fault_pool_id_(clio::run::PoolId::GetNull()),
        fault_pool_name_(CLIO_PRIV_ALLOC),
        fault_params_(CLIO_PRIV_ALLOC) {}
  explicit TagCacheUpdateTask(const clio::run::TaskId &task_id,
                              const clio::run::PoolId &pool_id,
                              const clio::run::PoolQuery &pool_query,
                              clio::run::u32 method,
                              const std::string &tag_name,
                              const TagId &tag_id,
                              const clio::run::PoolId &fault_pool_id,
                              const std::string &fault_pool_name,
                              const std::string &fault_params)
      : clio::run::Task(task_id, pool_id, pool_query, method),
        tag_name_(CLIO_PRIV_ALLOC, tag_name),
        tag_id_(tag_id),
        fault_pool_id_(fault_pool_id),
        fault_pool_name_(CLIO_PRIV_ALLOC, fault_pool_name),
        fault_params_(CLIO_PRIV_ALLOC, fault_params) {
    task_flags_.Clear();
  }
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_name_, tag_id_, fault_pool_id_, fault_pool_name_, fault_params_);
  }
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }
  void Copy(const ctp::ipc::FullPtr<TagCacheUpdateTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_name_ = other->tag_name_;
    tag_id_ = other->tag_id_;
    fault_pool_id_ = other->fault_pool_id_;
    fault_pool_name_ = other->fault_pool_name_;
    fault_params_ = other->fault_params_;
  }
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
  }
};

}  // namespace clio::cte::cache

#endif  // CLIO_CTE_CACHE_CACHE_TASKS_H_
