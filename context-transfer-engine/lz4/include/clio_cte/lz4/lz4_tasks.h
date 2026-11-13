/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_LZ4_CACHE_TASKS_H_
#define CLIO_CTE_LZ4_CACHE_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/msan.h>
#include <clio_runtime/task.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/lz4/autogen/lz4_methods.h>

#include <string>

namespace clio::cte::lz4 {

/** The LZ4 chimod speaks the CTE core's vocabulary. */
using Context = clio::cte::core::Context;
using TagId = clio::cte::core::TagId;

/** Well-known default pool id/name. */
static constexpr clio::run::PoolId kLz4PoolId(565, 0);
static constexpr const char *kLz4PoolName = "clio_cte_lz4";

/** Context::compress_lib_ value that asks this module to compress a put
 *  (clio-infer: --codec 25). Any other value forwards untouched. */
static constexpr int kGpuLz4Lib = 25;
/** Context::compress_lib_ for interleaved rANS pages ("gpu-ans", ~1.48x on
 *  BF16 esm pages against LZ4's ~1.28x). Same container, same GPU decode. */
static constexpr int kGpuAnsLib = 26;

/**
 * Container creation params.
 *   next_pool_id   pool this module forwards to (usually the CTE core)
 *   depth          encoder hash-chain depth: ratio vs encode speed (32:
 *                  ~1.28x on BF16 esm pages, ~26 MiB/s per encoding thread)
 *   staging_mb     device staging for pages that are not in a device tier
 *                  (their stored bytes are copied up, then decoded)
 */
struct Lz4Config {
  static constexpr const char* chimod_lib_name = "clio_cte_lz4";

  clio::run::PoolId next_pool_id_;
  clio::run::u32 depth_ = 32;
  clio::run::u32 staging_mb_ = 256;

  Lz4Config() : next_pool_id_(clio::run::PoolId::GetNull()) {}
  Lz4Config(const clio::run::PoolId &pool_id, const Lz4Config &other)
      : next_pool_id_(other.next_pool_id_),
        depth_(other.depth_),
        staging_mb_(other.staging_mb_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    // EVERY field round-trips (a dropped field silently resets on restart).
    ar(next_pool_id_, depth_, staging_mb_);
  }

  /** Load configuration from compose YAML. */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (pool_config.config_.empty()) return;
    try {
      YAML::Node node = YAML::Load(pool_config.config_);
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
      if (node["depth"]) depth_ = node["depth"].as<clio::run::u32>();
      if (node["staging_mb"]) {
        staging_mb_ = node["staging_mb"].as<clio::run::u32>();
      }
    } catch (...) {
      // Config parsing is best-effort; defaults stand.
    }
  }
};

/** Standard pool creation. */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<Lz4Config>;

/** Cleanup the LZ4 container. */
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

}  // namespace clio::cte::lz4

#endif  // CLIO_CTE_LZ4_CACHE_TASKS_H_
