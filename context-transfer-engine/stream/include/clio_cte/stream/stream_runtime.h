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

/**
 * Stream runtime: a byte-stream layer over CTE tags.
 *
 * Two roles per container:
 *  - HOME of the streams routed to it: owns each stream's logical size and
 *    merges its deferred appends (Plan), persisting both in a RecordLog.
 *  - ORIGIN of the appends submitted on its node: stages the bytes as a
 *    self-describing blob, queues the entry, and ships batches to each
 *    stream's home every kSequencePeriodUs (Sequence -> Collect -> Plan).
 *
 * Staged blob name: "sa.<major>.<minor>.<home>.<clock>.<origin>.<counter>"
 * (hex). It carries everything needed to merge the append, so a restarted
 * container can find and re-queue staged appends it is responsible for.
 */
#ifndef CLIO_CTE_STREAM_STREAM_RUNTIME_H_
#define CLIO_CTE_STREAM_STREAM_RUNTIME_H_

#include <atomic>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/record_log.h>
#include <clio_cte/stream/stream_client.h>
#include <clio_cte/stream/stream_tasks.h>

namespace clio::cte::stream {

/** Period of each node's append drain (microseconds). */
GLOBAL_CROSS_CONST double kSequencePeriodUs = 1000.0;
/** Largest merge slice: staged appends read into one buffer at a time. */
GLOBAL_CROSS_CONST clio::run::u64 kMaxSliceBytes = 16ULL << 20;
/** Staging tag holding appended bytes until they are merged. */
inline constexpr const char *kStagingTagName = "_clio_stream_staging";

/** A staged append queued on its origin node. */
struct PendingAppend {
  clio::cte::core::TagId tag_;
  clio::run::u32 home_ = 0;
  AppendEntry entry_;
  clio::run::u64 not_before_ns_ = 0;  ///< retry backoff after a failed ship
  clio::run::u64 local_seq_ = 0;      ///< this node's enqueue order
};

/** One persisted merge plan (in flight until its kDone record). */
struct StreamPlan {
  clio::run::u64 id_ = 0;
  clio::cte::core::TagId tag_;
  clio::run::u64 base_ = 0;  ///< stream offset of the first entry
  std::vector<AppendEntry> entries_;  ///< in merge order
};

/** Stream container. */
class Runtime : public clio::run::Container {
 public:
  using CreateParams = StreamConfig;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override = default;

  // ---- Method handlers ----
  /** Bind the data client, open/replay the log, recover staged appends. */
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  /** No-op teardown. */
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  /** No-op monitor. */
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  /** Get / max / reserve / set / drop a stream's size (home). */
  clio::run::TaskResume SizeOp(clio::run::shared_ptr<SizeOpTask> &task);
  /** Stage one deferred append and queue it (origin). */
  clio::run::TaskResume Append(clio::run::shared_ptr<AppendTask> &task);
  /** Wait for this node's appends to a stream to be merged (origin). */
  clio::run::TaskResume Flush(clio::run::shared_ptr<FlushTask> &task);
  /** Periodic: ship queued appends to their homes (origin). */
  clio::run::TaskResume Sequence(clio::run::shared_ptr<SequenceTask> &task);
  /** ManyToOne aggregate: merge the combined batch via Plan (home). */
  clio::run::TaskResume Collect(clio::run::shared_ptr<CollectTask> &task);
  /** Order, reserve, persist and copy one batch (home). */
  clio::run::TaskResume Plan(clio::run::shared_ptr<PlanTask> &task);

  // ---- Container virtuals (autogen/stream_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  /** Recovery start (`clio_run restart`): Create replays the log. */
  void Restart(const clio::run::PoolId &pool_id, const std::string &pool_name,
               clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(
      clio::run::u32 method,
      clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void LocalLoadTask(clio::run::u32 method,
                     clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method,
                     clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  void AggregateOut(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
      const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
      const clio::run::shared_ptr<clio::run::Task> &member_task) override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(
      clio::run::u32 method, clio::run::LoadTaskArchive &archive) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig,
      bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(
      clio::run::u32 method) override;

  /**
   * In-process size read for a co-located caller (clio-fs on the same node
   * as the stream's home). Never blocks on the network.
   * @param tag stream tag
   * @param size receives the size (0 if unknown)
   * @return true if this container knows the stream
   */
  bool LocalSize(const clio::cte::core::TagId &tag, clio::run::u64 *size);

 private:
  /** Per-stream home state. */
  struct StreamState {
    clio::run::u64 size_ = 0;
    bool busy_ = false;  ///< a Plan is merging into this stream
  };

  // ---- home side (stream_home.cc) ----
  /**
   * Apply one size operation under mu_ and persist it.
   * @param tag stream tag
   * @param op operation
   * @param value operand
   * @param old_size receives the size before
   * @return size after
   */
  clio::run::u64 ApplySizeOpLocked(const clio::cte::core::TagId &tag,
                                   StreamSizeOp op, clio::run::u64 value,
                                   clio::run::u64 *old_size);
  /**
   * Remove entries already merged (or being merged) by this incarnation
   * and remember the rest.
   * @param entries batch, filtered in place
   */
  void DedupeLocked(std::vector<AppendEntry> *entries);
  /**
   * Copy a plan's staged bytes into the stream's pages, then delete the
   * staged blobs. Idempotent: a missing staged blob was already merged.
   * @param plan the plan to execute
   * @param ok receives false if any copy failed
   */
  clio::run::TaskResume ExecutePlan(StreamPlan plan, bool *ok);
  /**
   * Copy one slice [first, last) of a plan starting at stream offset `off`.
   * @param plan the plan
   * @param first first entry index
   * @param last one past the last entry index
   * @param off stream offset of entry `first`
   * @param ok receives false if any copy failed
   */
  clio::run::TaskResume CopySlice(const StreamPlan *plan, size_t first,
                                  size_t last, clio::run::u64 off, bool *ok);
  /** Delete every staged blob of `entries` (best effort). */
  clio::run::TaskResume DeleteStaged(std::vector<AppendEntry> entries);

  // ---- persistence (stream_log.cc) ----
  /** Log record types. */
  enum LogRec : clio::run::u32 {
    kRecSize = 1,  ///< tag, size
    kRecDrop = 2,  ///< tag, wall time
    kRecPlan = 3,  ///< plan id, tag, base, entries
    kRecDone = 4,  ///< plan id
    kRecNames = 5,  ///< recently merged staged names (compaction snapshot)
  };
  /** Open the log; replay it on restart, discard it on a fresh start. */
  void OpenLog();
  /** Apply one replayed record. */
  void ReplayRecord(clio::run::u32 type, const std::string &payload);
  /** Rewrite the log as a snapshot of the live state (mu_ held). */
  void CompactLocked();
  /** Append a size record (mu_ held). */
  void LogSizeLocked(const clio::cte::core::TagId &tag, clio::run::u64 size);
  /** Append a plan record (mu_ held). */
  void LogPlanLocked(const StreamPlan &plan);
  /** Append a done record (mu_ held). */
  void LogDoneLocked(clio::run::u64 plan_id);
  /** Append a drop record (mu_ held). */
  void LogDropLocked(const clio::cte::core::TagId &tag);

  // ---- origin side / recovery (stream_runtime.cc) ----
  /**
   * Build the staged blob name of an append.
   * @param tag stream tag
   * @param home stream home
   * @param e entry (clock, origin, counter)
   * @return staged blob name
   */
  static std::string MakeStagedName(const clio::cte::core::TagId &tag,
                                    clio::run::u32 home,
                                    const AppendEntry &e);
  /**
   * Parse a staged blob name.
   * @param name staged blob name
   * @param tag receives the stream tag
   * @param home receives the stream home
   * @param e receives clock / origin / counter / name
   * @return false if `name` is not a staged append
   */
  static bool ParseStagedName(const std::string &name,
                              clio::cte::core::TagId *tag,
                              clio::run::u32 *home, AppendEntry *e);
  /** Queue an entry for shipping (starts the drain on first use). */
  void Enqueue(PendingAppend p);
  /** Start the periodic drain if it is not running. */
  void EnsureSequence();
  /** Restart: re-queue staged appends this node originated or homes. */
  clio::run::TaskResume RecoverStaged();
  /** Restart: finish plans that were in flight at the crash. */
  clio::run::TaskResume FinishOpenPlans();

  StreamConfig config_;
  clio::cte::core::Client cte_;
  Client self_;
  clio::cte::core::TagId staging_tag_;
  clio::run::u32 node_ = 0;
  bool is_restart_ = false;

  // home side, guarded by mu_
  std::mutex mu_;
  std::unordered_map<clio::cte::core::TagId, StreamState> streams_;
  std::unordered_map<clio::cte::core::TagId, clio::run::u64> dropped_;
  std::unordered_map<std::string, clio::run::u64> merged_names_;
  std::unordered_map<clio::run::u64, StreamPlan> open_plans_;
  clio::run::u64 next_plan_id_ = 1;
  clio::cte::core::RecordLog log_;

  // origin side, guarded by q_mu_
  std::mutex q_mu_;
  std::vector<PendingAppend> queue_;
  /** Per stream: local_seq_ of this node's queued appends not yet merged. */
  std::unordered_map<clio::cte::core::TagId, std::set<clio::run::u64>>
      local_pending_;
  clio::run::u64 last_clock_ = 0;
  clio::run::u64 counter_ = 0;
  clio::run::u64 enq_seq_ = 0;  ///< last local_seq_ handed out
  bool seq_started_ = false;
  std::atomic<clio::run::u64> pending_count_{0};
};

}  // namespace clio::cte::stream

#endif  // CLIO_CTE_STREAM_STREAM_RUNTIME_H_
