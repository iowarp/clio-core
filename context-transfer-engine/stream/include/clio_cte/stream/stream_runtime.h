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
 *    self-describing blob, queues the entry, and ships its batch for each
 *    stream straight to that stream's home every kSequencePeriodUs
 *    (Sequence -> Plan).
 *
 * Staged blob name: "sa.<major>.<minor>.<home>.<clock>.<origin>.<counter>"
 * (hex). It carries everything needed to merge the append, so a restarted
 * container can find and re-queue the staged appends it had accepted.
 */
#ifndef CLIO_CTE_STREAM_STREAM_RUNTIME_H_
#define CLIO_CTE_STREAM_STREAM_RUNTIME_H_

#include <atomic>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <chrono>
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
  clio::run::u64 local_seq_ = 0;      ///< this node's enqueue order
  /** The appended bytes, shipped inline to the home (empty for an append
   *  re-queued after a restart: the home reads its staged blob). */
  std::string data_;
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
  /** Order, reserve, persist and copy one batch (home). */
  clio::run::TaskResume Plan(clio::run::shared_ptr<PlanTask> &task);

  // ---- Container virtuals (autogen/stream_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  /** Recovering start (a plain `clio_run start`): Create replays the log. */
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

  /**
   * Streams restored from this container's log that the owner of their
   * sizes (clio-fs) has not reconciled yet. After a restart a stream's size
   * may be older than the truth: its file was served elsewhere while this
   * node was down. Size ops and merges on such a stream wait until it is
   * reconciled or released (at most kRestoreGateS).
   * @return the unverified stream tags
   */
  std::vector<clio::cte::core::TagId> UnverifiedStreams();
  /**
   * Apply the authoritative size to a restored stream and release it.
   * @param tag stream tag
   * @param op kSet (the size was changed elsewhere) or kMax (a floor)
   * @param value the size
   */
  void ReconcileRestored(const clio::cte::core::TagId &tag, StreamSizeOp op,
                         clio::run::u64 value);
  /** Release every restored stream still waiting (nothing to reconcile). */
  void ReleaseRestored();

 private:
  /**
   * Whether ops on `tag` must still wait for reconciliation (mu_ held).
   * @param tag stream tag
   * @return true while restored-but-unreconciled and within the gate time
   */
  bool GatedLocked(const clio::cte::core::TagId &tag);
  /**
   * Whether any restored stream still waits for reconciliation (mu_ held).
   * @return true while any is held and the gate time has not passed
   */
  bool GatedAny();
  /**
   * The container serving a stream homed on `home` right now: `home` while
   * its node is alive, else its failover successor.
   * @param home the stream's home container
   * @return the live home
   */
  clio::run::u32 LiveHome(clio::run::u32 home) const;
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
   * Copy a plan's bytes into the stream's pages, then queue its staged blobs
   * for deletion. Idempotent: a missing staged blob was already merged.
   * @param plan the plan to execute
   * @param payload the bytes carried with the plan (entries' payload_off_),
   *        or null when replaying a plan after a restart
   * @param ok receives false if any copy failed
   */
  clio::run::TaskResume ExecutePlan(StreamPlan plan, const std::string *payload,
                                    bool *ok);
  /**
   * Copy one slice [first, last) of a plan starting at stream offset `off`.
   * @param plan the plan
   * @param payload inline bytes (may be null)
   * @param first first entry index
   * @param last one past the last entry index
   * @param off stream offset of entry `first`
   * @param ok receives false if any copy failed
   */
  clio::run::TaskResume CopySlice(const StreamPlan *plan,
                                  const std::string *payload, size_t first,
                                  size_t last, clio::run::u64 off, bool *ok);
  /** Queue staged blobs for background deletion (merged or discarded). */
  void QueueStagedDelete(const std::vector<AppendEntry> &entries);
  /** Delete a bounded batch of queued staged blobs (Sequence tick). */
  clio::run::TaskResume ReapStaged();

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
  /**
   * Ship one chunk of a stream's queued appends (bytes inline) to its home.
   * @param chunk appends of one stream, in acceptance order
   * @param ok receives true if the home merged them
   */
  clio::run::TaskResume ShipChunk(const std::vector<PendingAppend> *chunk,
                                  bool *ok);
  /** Queue an entry for shipping (starts the drain on first use). */
  void Enqueue(PendingAppend p);
  /** Start the periodic drain if it is not running. */
  void EnsureSequence();
  /** Restart: re-queue the staged appends this node originated. */
  clio::run::TaskResume RecoverStaged();
  /**
   * Hold a task until Create has bound the CTE clients and reopened the
   * stream log. The container takes tasks as soon as it is registered,
   * before Create runs: after a restart a Plan merged appends from offset
   * 0 over the file's first records, and its copies went to pool 0.0.
   * @param ok OUT false if Create still had not got that far after
   *        kReadyWaitMs
   */
  clio::run::TaskResume AwaitReady(bool &ok);
  /** Set once Create has bound its clients and opened the log. */
  std::atomic<bool> ready_{false};
  /** Longest a task waits for Create (ms). */
  static constexpr clio::run::u64 kReadyWaitMs = 120000;
  /** Poll period while waiting for Create (us). */
  static constexpr double kReadyPollUs = 1000.0;
  /** Restart: finish plans that were in flight at the crash. */
  clio::run::TaskResume FinishOpenPlans();
  /** fsync the size log if kLogSyncPeriodMs passed and it has new records
   *  (called from the Sequence tick). */
  void SyncLogPeriodically();

  StreamConfig config_;
  clio::cte::core::Client cte_;      ///< pages
  clio::cte::core::Client staging_;  ///< staged appends
  Client self_;
  clio::cte::core::TagId staging_tag_;
  clio::run::u32 node_ = 0;
  bool is_restart_ = false;

  // home side, guarded by mu_
  std::mutex mu_;
  std::unordered_map<clio::cte::core::TagId, StreamState> streams_;
  /** Merge plans interrupted by a restart wait to be finished (mu_). */
  bool plans_pending_ = false;
  /** How long restored streams wait for reconciliation at most (s). */
  static constexpr int kRestoreGateS = 120;
  /** Restored streams not yet reconciled (see UnverifiedStreams; mu_). */
  std::unordered_set<clio::cte::core::TagId> unverified_;
  /** When the restore gate gives up and releases everything. */
  std::chrono::steady_clock::time_point gate_deadline_{};
  std::unordered_map<clio::cte::core::TagId, clio::run::u64> dropped_;
  std::unordered_map<std::string, clio::run::u64> merged_names_;
  std::unordered_map<clio::run::u64, StreamPlan> open_plans_;
  clio::run::u64 next_plan_id_ = 1;
  clio::cte::core::RecordLog log_;
  /** Unforced size-log records are fsynced at least this often (ms), so a
   *  power loss loses at most this much of them (like ext4's commit). */
  static constexpr int kLogSyncPeriodMs = 5000;
  std::chrono::steady_clock::time_point last_log_sync_ =
      std::chrono::steady_clock::now();

  // origin side, guarded by q_mu_
  std::mutex q_mu_;
  std::vector<PendingAppend> queue_;  ///< in acceptance order
  /** Streams whose last batch failed to reach the home: nothing of theirs
   *  ships before this time, so a retry is never overtaken by newer
   *  appends from this node. */
  std::unordered_map<clio::cte::core::TagId, clio::run::u64> hold_until_;
  /** Per stream: local_seq_ of this node's queued appends not yet merged. */
  std::unordered_map<clio::cte::core::TagId, std::set<clio::run::u64>>
      local_pending_;
  clio::run::u64 last_clock_ = 0;
  clio::run::u64 counter_ = 0;
  clio::run::u64 enq_seq_ = 0;  ///< last local_seq_ handed out
  bool seq_started_ = false;
  /** Staged blobs whose appends are merged (or discarded), to delete. */
  std::vector<std::string> to_delete_;
  std::atomic<clio::run::u64> pending_count_{0};
};

}  // namespace clio::cte::stream

#endif  // CLIO_CTE_STREAM_STREAM_RUNTIME_H_
