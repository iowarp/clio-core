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

#ifndef CLIO_CTE_STREAM_STREAM_TASKS_H_
#define CLIO_CTE_STREAM_STREAM_TASKS_H_

#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/stream/autogen/stream_methods.h>

namespace clio::cte::stream {

/** Canonical stream pool id and name. */
static constexpr clio::run::PoolId kStreamPoolId(565, 0);
inline constexpr const char *kStreamPoolName = "clio_cte_stream";

/** Page size of a stream: data lives in page blobs of this size. */
GLOBAL_CROSS_CONST clio::run::u64 kStreamPageSize = 1024ULL * 1024ULL;

/**
 * Blob name of the page holding byte `off` of a stream.
 * @param off byte offset in the stream
 * @return decimal page index
 */
inline std::string StreamPageName(clio::run::u64 off) {
  return std::to_string(off / kStreamPageSize);
}

/** Operations on a stream's logical size (SizeOpTask::op_). */
enum class StreamSizeOp : clio::run::u32 {
  kGet = 0,      ///< read the size
  kMax = 1,      ///< size = max(size, value): a write or close pushes its end
  kReserve = 2,  ///< old = size; size += value: synchronous append
  kSet = 3,      ///< size = value: truncate or extend
  kDrop = 4,     ///< forget the stream and discard its pending appends
};

/**
 * One staged append. The staged blob's NAME encodes everything needed to
 * merge it (see MakeStagedName), so staged appends survive a daemon restart.
 */
struct AppendEntry {
  std::string staged_name_;     ///< blob name under the staging tag
  clio::run::u64 size_ = 0;     ///< staged bytes
  clio::run::u64 clock_ = 0;    ///< origin's monotonic clock (primary key)
  clio::run::u32 origin_ = 0;   ///< node that accepted the append
  clio::run::u64 counter_ = 0;  ///< origin's append counter (tiebreak)
  /** Offset of this append's bytes in the carrying PlanTask::payload_, or
   *  kNoPayload: the home then reads the staged blob (restart recovery). */
  clio::run::u64 payload_off_ = ~0ULL;

  /** Serialize for the wire. */
  template <class Ar>
  void serialize(Ar &ar) {
    ar(staged_name_, size_, clock_, origin_, counter_, payload_off_);
  }
};

/** AppendEntry::payload_off_ value: the bytes are not carried inline. */
GLOBAL_CROSS_CONST clio::run::u64 kNoPayload = ~0ULL;

/** Container creation params. */
struct StreamConfig {
  static constexpr const char *chimod_lib_name = "clio_cte_stream";

  /** Pool data is read from and written to (the CTE chain; default core). */
  clio::run::PoolId next_pool_id_;
  /** Base path of each container's durable log ("" = no persistence). The
   *  container appends "." + its container id. */
  std::string log_path_;
  /** Pool staged appends are written to (default: next_pool_id_). Staged
   *  bytes are written once and read at most once (after a restart), so a
   *  deployment points this below any cache layer. */
  clio::run::PoolId staging_pool_id_;

  StreamConfig()
      : next_pool_id_(clio::run::PoolId::GetNull()),
        staging_pool_id_(clio::run::PoolId::GetNull()) {}
  StreamConfig(const clio::run::PoolId &pool_id, const StreamConfig &other)
      : next_pool_id_(other.next_pool_id_), log_path_(other.log_path_),
        staging_pool_id_(other.staging_pool_id_) {
    (void)pool_id;
  }

  /** Serialize every field. */
  template <class Archive>
  void serialize(Archive &ar) {
    ar(next_pool_id_, log_path_, staging_pool_id_);
  }

  /**
   * Load configuration from compose YAML.
   * @param pool_config compose entry for this pool
   */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (pool_config.config_.empty()) return;
    try {
      YAML::Node node = YAML::Load(pool_config.config_);
      if (node["next_pool_id"]) {
        std::string s = node["next_pool_id"].as<std::string>();
        auto dot = s.find('.');
        if (dot != std::string::npos) {
          next_pool_id_ = clio::run::PoolId(std::stoul(s.substr(0, dot)),
                                            std::stoul(s.substr(dot + 1)));
        }
      }
      if (node["log_path"]) log_path_ = node["log_path"].as<std::string>();
      if (node["staging_pool_id"]) {
        staging_pool_id_ = clio::run::PoolId::FromString(
            node["staging_pool_id"].as<std::string>());
      }
    } catch (...) {
      // Config parsing is best-effort: defaults apply.
    }
  }
};

/** Standard pool creation. */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<StreamConfig>;
using MonitorTask = clio::run::admin::MonitorTask;

/** Destroy the stream container. */
struct DestroyTask : public clio::run::Task {
  DestroyTask() : clio::run::Task() {}
  explicit DestroyTask(const clio::run::TaskId &task_id,
                       const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDestroy) {}
  /** No OUT fields beyond the base. */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &o) {
    Task::AggregateOut(o);
  }
  void Copy(const ctp::ipc::FullPtr<DestroyTask> &o) {
    Task::Copy(o.template Cast<clio::run::Task>());
  }
  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/** SizeOp: read or change a stream's logical size at its home. */
struct SizeOpTask : public clio::run::Task {
  IN clio::cte::core::TagId tag_id_;
  IN clio::run::u32 op_;      ///< StreamSizeOp
  IN clio::run::u64 value_;   ///< operand (unused by kGet / kDrop)
  OUT clio::run::u64 old_size_;
  OUT clio::run::u64 new_size_;

  SizeOpTask()
      : clio::run::Task(), tag_id_(clio::cte::core::TagId::GetNull()), op_(0),
        value_(0), old_size_(0), new_size_(0) {}
  explicit SizeOpTask(const clio::run::TaskId &task_id,
                      const clio::run::PoolId &pool_id,
                      const clio::run::PoolQuery &pool_query,
                      const clio::cte::core::TagId &tag_id, StreamSizeOp op,
                      clio::run::u64 value)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kSizeOp),
        tag_id_(tag_id), op_(static_cast<clio::run::u32>(op)), value_(value),
        old_size_(0), new_size_(0) {}
  void Copy(const ctp::ipc::FullPtr<SizeOpTask> &o) {
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    tag_id_ = o->tag_id_; op_ = o->op_; value_ = o->value_;
    old_size_ = o->old_size_; new_size_ = o->new_size_;
  }
  /** Merge OUT fields only. */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &b) {
    Task::AggregateOut(b);
    auto o = b.template Cast<SizeOpTask>();
    old_size_ = o->old_size_; new_size_ = o->new_size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(tag_id_, op_, value_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(old_size_, new_size_);
  }
};

/** Append: stage bytes for a deferred append (caller's node). */
struct AppendTask : public clio::run::Task {
  IN clio::cte::core::TagId tag_id_;
  IN clio::run::u32 home_;  ///< container that owns the stream's size
  IN clio::run::u64 size_;
  IN ctp::ipc::ShmPtr<> data_;
  OUT clio::run::u64 bytes_written_;

  AppendTask()
      : clio::run::Task(), tag_id_(clio::cte::core::TagId::GetNull()),
        home_(0), size_(0), data_(ctp::ipc::ShmPtr<>::GetNull()),
        bytes_written_(0) {}
  explicit AppendTask(const clio::run::TaskId &task_id,
                      const clio::run::PoolId &pool_id,
                      const clio::run::PoolQuery &pool_query,
                      const clio::cte::core::TagId &tag_id,
                      clio::run::u32 home, clio::run::u64 size,
                      ctp::ipc::ShmPtr<> data)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kAppend),
        tag_id_(tag_id), home_(home), size_(size), data_(data),
        bytes_written_(0) {}
  void Copy(const ctp::ipc::FullPtr<AppendTask> &o) {
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    tag_id_ = o->tag_id_; home_ = o->home_; size_ = o->size_;
    data_ = o->data_; bytes_written_ = o->bytes_written_;
  }
  /** Merge OUT fields only. */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &b) {
    Task::AggregateOut(b);
    bytes_written_ = b.template Cast<AppendTask>()->bytes_written_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(tag_id_, home_, size_, data_);
    ar.bulk(data_, size_, BULK_XFER);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(bytes_written_);
  }
};

/** Flush: wait until this node's appends to a stream are merged. */
struct FlushTask : public clio::run::Task {
  IN clio::cte::core::TagId tag_id_;
  IN clio::run::u32 home_;
  OUT clio::run::u64 size_;  ///< stream size after the merge

  FlushTask()
      : clio::run::Task(), tag_id_(clio::cte::core::TagId::GetNull()),
        home_(0), size_(0) {}
  explicit FlushTask(const clio::run::TaskId &task_id,
                     const clio::run::PoolId &pool_id,
                     const clio::run::PoolQuery &pool_query,
                     const clio::cte::core::TagId &tag_id,
                     clio::run::u32 home)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kFlush),
        tag_id_(tag_id), home_(home), size_(0) {}
  void Copy(const ctp::ipc::FullPtr<FlushTask> &o) {
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    tag_id_ = o->tag_id_; home_ = o->home_; size_ = o->size_;
  }
  /** Merge OUT fields only. */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &b) {
    Task::AggregateOut(b);
    size_ = b.template Cast<FlushTask>()->size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(tag_id_, home_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(size_);
  }
};

/** Sequence: periodic drain of this node's pending appends. */
struct SequenceTask : public clio::run::Task {
  SequenceTask() : clio::run::Task() {}
  explicit SequenceTask(const clio::run::TaskId &task_id,
                        const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kSequence) {}
  void Copy(const ctp::ipc::FullPtr<SequenceTask> &o) {
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
  }
  /** No OUT fields beyond the base. */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &o) {
    Task::AggregateOut(o);
  }
  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/**
 * Plan: one node's batch of appends to a stream, sent to the stream's home,
 * which orders it, reserves the tail, persists the plan, copies, finishes.
 */
struct PlanTask : public clio::run::Task {
  IN clio::cte::core::TagId tag_id_;
  IN std::vector<AppendEntry> entries_;
  IN std::string payload_;  ///< the entries' bytes (see payload_off_)
  OUT clio::run::u64 new_size_;

  PlanTask()
      : clio::run::Task(), tag_id_(clio::cte::core::TagId::GetNull()),
        new_size_(0) {}
  explicit PlanTask(const clio::run::TaskId &task_id,
                    const clio::run::PoolId &pool_id,
                    const clio::run::PoolQuery &pool_query,
                    const clio::cte::core::TagId &tag_id,
                    const std::vector<AppendEntry> &entries,
                    const std::string &payload)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kPlan),
        tag_id_(tag_id), entries_(entries), payload_(payload), new_size_(0) {}
  void Copy(const ctp::ipc::FullPtr<PlanTask> &o) {
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    tag_id_ = o->tag_id_; entries_ = o->entries_; payload_ = o->payload_;
    new_size_ = o->new_size_;
  }
  /** Merge OUT fields only. */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &b) {
    Task::AggregateOut(b);
    new_size_ = b.template Cast<PlanTask>()->new_size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(tag_id_, entries_, payload_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(new_size_);
  }
};

}  // namespace clio::cte::stream

#endif  // CLIO_CTE_STREAM_STREAM_TASKS_H_
