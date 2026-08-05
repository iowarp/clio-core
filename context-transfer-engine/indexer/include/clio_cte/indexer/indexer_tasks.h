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

#ifndef CLIO_CTE_INDEXER_INDEXER_TASKS_H_
#define CLIO_CTE_INDEXER_INDEXER_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_ctp/util/config_parse.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/indexer/autogen/indexer_methods.h>

#include <string>

namespace clio::cte::indexer {

/** The indexer chimod speaks the CTE core's vocabulary. */
using Context = clio::cte::core::Context;
using TagId = clio::cte::core::TagId;

/**
 * Well-known default pool id/name. In the standard chain the indexer sits
 * directly ABOVE the core (indexer 564 -> core 512): the mutating data
 * verbs must flow through it for the index to stay current, and search
 * clients address it (or any pool that forwards down to it).
 */
static constexpr clio::run::PoolId kIndexerPoolId(564, 0);
static constexpr const char *kIndexerPoolName = "clio_cte_indexer";

/**
 * Container creation params. next_pool_id_ is the pool this indexer
 * forwards non-search verbs to (usually the CTE core). The index itself is
 * DERIVED state: nothing here configures persistence because the index is
 * never persisted — on restart it is rebuilt from the storage below.
 */
struct IndexerConfig {
  static constexpr const char* chimod_lib_name = "clio_cte_indexer";

  clio::run::PoolId next_pool_id_;  ///< Next pool in the chain (e.g. 512.0)
  /// Period of the asynchronous index drain (kIndexSweep). Indexing is OFF
  /// the put ack path: a put only enqueues its (tag, blob) key — coalesced,
  /// so N overwrites of a hot blob cost ONE re-tokenize — and the sweep (or
  /// a SemanticSearch, which drains first for read-your-writes) does the
  /// work. 0 disables the sweep: the index is then updated only when a
  /// search runs (lazy indexing).
  clio::run::u32 index_sweep_period_ms_ = 100;
  /// Index SCOPE: only blobs whose full tag name matches tag_re_ AND whose
  /// blob name matches blob_re_ are tokenized (std::regex_match, same
  /// full-string semantics as the search filters). Out-of-scope content
  /// costs nothing — no scan, no index memory, no WAL. Defaults index
  /// everything.
  std::string tag_re_ = ".*";
  std::string blob_re_ = ".*";
  /// Index persistence (issue #905): when non-empty, the module keeps its
  /// state in <path> (snapshot) + <path>.wal (append-only deltas) and a
  /// restart RESTORES instead of rescanning storage. Empty = in-memory
  /// only: a restart comes up EMPTY and re-indexes incrementally (first
  /// insertion per tag backfills that tag; kReindexScan backfills on
  /// demand).
  std::string index_log_path_;
  /// WAL size that triggers snapshot+truncate compaction (checked by the
  /// periodic sweep).
  clio::run::u64 index_wal_compact_bytes_ = 8ULL * 1024 * 1024;

  IndexerConfig() : next_pool_id_(clio::run::PoolId::GetNull()) {}
  IndexerConfig(const clio::run::PoolId &pool_id, const IndexerConfig &other)
      : next_pool_id_(other.next_pool_id_),
        index_sweep_period_ms_(other.index_sweep_period_ms_),
        tag_re_(other.tag_re_),
        blob_re_(other.blob_re_),
        index_log_path_(other.index_log_path_),
        index_wal_compact_bytes_(other.index_wal_compact_bytes_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    // EVERY field round-trips (the cache/replication silently-dropped-field
    // audit rule).
    ar(next_pool_id_, index_sweep_period_ms_, tag_re_, blob_re_,
       index_log_path_, index_wal_compact_bytes_);
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
        if (node["index_sweep_period_ms"]) {
          index_sweep_period_ms_ =
              node["index_sweep_period_ms"].as<clio::run::u32>();
        }
        if (node["tag_re"]) {
          tag_re_ = node["tag_re"].as<std::string>();
        }
        if (node["blob_re"]) {
          blob_re_ = node["blob_re"].as<std::string>();
        }
        if (node["index_log_path"]) {
          index_log_path_ = ctp::ConfigParse::ExpandPath(
              node["index_log_path"].as<std::string>());
        }
        if (node["index_wal_compact_bytes"]) {
          index_wal_compact_bytes_ = ctp::ConfigParse::ParseSize(
              node["index_wal_compact_bytes"].as<std::string>());
        }
      } catch (...) {
        // Config parsing is best-effort
      }
    }
  }
};

/** Standard pool creation. */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<IndexerConfig>;

/** Cleanup the indexer container. */
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

/**
 * Explicit on-demand index scan (Method::kReindexScan): enumerate existing
 * blobs matching tag_re_ AND blob_re_ (intersected with the module's
 * configured scope) via BlobQuery on the chain below, and ENQUEUE them for
 * the asynchronous drain. The only way pre-existing cold data enters the
 * index besides a tag's first-insertion backfill — restarts restore
 * persisted state and never rescan.
 */
struct ReindexScanTask : public clio::run::Task {
  IN clio::run::priv::string tag_regex_;
  IN clio::run::priv::string blob_regex_;
  OUT clio::run::u64 blobs_enqueued_;

  ReindexScanTask()
      : clio::run::Task(),
        tag_regex_(CLIO_PRIV_ALLOC),
        blob_regex_(CLIO_PRIV_ALLOC),
        blobs_enqueued_(0) {}

  explicit ReindexScanTask(const clio::run::TaskId &task_id,
                           const clio::run::PoolId &pool_id,
                           const clio::run::PoolQuery &pool_query,
                           const std::string &tag_regex,
                           const std::string &blob_regex)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kReindexScan),
        tag_regex_(CLIO_PRIV_ALLOC, tag_regex),
        blob_regex_(CLIO_PRIV_ALLOC, blob_regex),
        blobs_enqueued_(0) {}

  template <typename Ar>
  void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar);
    ar(tag_regex_, blob_regex_);
  }

  template <typename Ar>
  void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);
    ar(blobs_enqueued_);
  }

  void Copy(const ctp::ipc::FullPtr<ReindexScanTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_regex_ = other->tag_regex_;
    blob_regex_ = other->blob_regex_;
    blobs_enqueued_ = other->blobs_enqueued_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<ReindexScanTask>();
    blobs_enqueued_ += other->blobs_enqueued_;
  }
};

/** Periodic driver of the asynchronous index drain (Method::kIndexSweep). */
struct IndexSweepTask : public clio::run::Task {
  IndexSweepTask() : clio::run::Task() {}

  explicit IndexSweepTask(const clio::run::TaskId &task_id,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kIndexSweep) {}

  void Copy(const ctp::ipc::FullPtr<IndexSweepTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
  }

  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

}  // namespace clio::cte::indexer

#endif  // CLIO_CTE_INDEXER_INDEXER_TASKS_H_
