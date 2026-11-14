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

#ifndef CLIO_CTE_SHM_WRITETHRU_CACHE_RUNTIME_H_
#define CLIO_CTE_SHM_WRITETHRU_CACHE_RUNTIME_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>
#include <clio_cte/core/shm_wtc.h>
#include <clio_cte/shm_writethru_cache/shm_writethru_cache_client.h>
#include <clio_cte/shm_writethru_cache/shm_writethru_cache_tasks.h>

namespace clio::cte::shm_wtc {

using clio::cte::core::TagId;

/** One coalesced, flush-ready range. */
struct FlushRecord {
  clio::run::u32 slot_ = 0;    ///< blob table slot
  clio::run::u64 off_ = 0;     ///< offset inside the blob
  clio::run::u64 size_ = 0;    ///< bytes (clamped by ReadRange)
  clio::run::u32 ranges_ = 0;  ///< raw log ranges merged into this record
  BlobKey key_;                ///< blob identity
  clio::run::u32 home_ = 0;    ///< container of the blob's home cache
  clio::run::u64 payload_off_ = 0;  ///< offset in its home group's staging
  clio::run::u32 put_flags_ = 0;    ///< put flags of the blob
};

/** Nodes holding a blob, as tracked by the blob's home. */
struct SharerSet {
  std::set<clio::run::u32> nodes_;    ///< containers that may cache it
  std::set<clio::run::u32> writers_;  ///< containers whose writes landed
};

/**
 * shm_writethru_cache runtime: owns the segment, drains its log, answers
 * every core verb for task callers, and keeps cached copies coherent across
 * nodes (blob homes track sharers and invalidate them when a write lands).
 */
class Runtime : public clio::cte::core::CoreInterposer {
 public:
  using CreateParams = WtcConfig;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override = default;

  // ---- Method handlers ----
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume GetOrCreateTag(
      clio::run::shared_ptr<clio::cte::core::GetOrCreateTagTask<
          clio::cte::core::CreateParams>> &task);
  clio::run::TaskResume PutBlob(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  clio::run::TaskResume GetBlob(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);
  clio::run::TaskResume DelBlob(
      clio::run::shared_ptr<clio::cte::core::DelBlobTask> &task);
  clio::run::TaskResume DelTag(
      clio::run::shared_ptr<clio::cte::core::DelTagTask> &task);
  clio::run::TaskResume GetBlobSize(
      clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task);
  clio::run::TaskResume MultiPutBlob(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task);
  clio::run::TaskResume SyncTag(
      clio::run::shared_ptr<clio::cte::core::SyncTagTask> &task);
  clio::run::TaskResume FlushLog(clio::run::shared_ptr<FlushLogTask> &task);
  clio::run::TaskResume Invalidate(clio::run::shared_ptr<InvalidateTask> &task);
  clio::run::TaskResume HomeFlush(clio::run::shared_ptr<HomeFlushTask> &task);
  /** Any other core method: drain the local log, then forward verbatim. */
  clio::run::TaskResume BarrierForward(clio::run::u32 method,
                                       clio::run::shared_ptr<clio::run::Task> task);

  /** Route data verbs submitter-local (the segment is per node). */
  clio::run::PoolQuery ScheduleTask(
      const clio::run::shared_ptr<clio::run::Task> &task) override;

  /** Segment view (tests). */
  clio::cte::core::wtc::View &GetView() { return view_; }

  // ---- Container virtuals (autogen/shm_writethru_cache_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                            clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  void AggregateOut(clio::run::u32 method,
                    clio::run::shared_ptr<clio::run::Task> &orig_task,
                    const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(clio::run::u32 method,
                   clio::run::shared_ptr<clio::run::Task> &agg_task,
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
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;

 private:
  /** Create and publish the segment. */
  bool OpenSegment();
  /** Core client bound to the next pool in the chain. */
  clio::cte::core::Client *Next();
  /** This node's id. */
  clio::run::u32 SelfNode() const;
  /** Container (of this pool) that is a blob's home. */
  clio::run::u32 HomeOf(const TagId &tag, const std::string &name) const;
  /** Whether a put/get context can be served by the segment. */
  static bool PlainContext(const clio::cte::core::Context &ctx) {
    return clio::cte::core::Client::WtcPlainContext(ctx);
  }
  /** Pop up to one batch of log ranges and coalesce them. */
  std::vector<FlushRecord> PopAndCoalesce(clio::run::u64 *popped);
  /** Resolve keys/homes and stage payloads per home node. */
  void StageRecords(std::vector<FlushRecord> &recs,
                    std::map<clio::run::u32, ctp::ipc::FullPtr<char>> *staging,
                    std::map<clio::run::u32, clio::run::u64> *staged);
  /** Descriptors of one home group. */
  static std::vector<clio::cte::core::MultiPutDesc> GroupDescs(
      const std::vector<FlushRecord> &recs, clio::run::u32 home);
  /** Re-send a failed group record by record; record sticky errors. */
  clio::run::TaskResume RetryGroup(const std::vector<FlushRecord> &recs,
                                   clio::run::u32 home,
                                   ctp::ipc::FullPtr<char> staging);
  /** Home side: record the writer, collect who to invalidate. */
  std::map<clio::run::u32, std::vector<BlobKey>> HomeApply(
      const std::vector<BlobKey> &keys, clio::run::u32 writer);
  /** Send invalidations and wait for every ack. */
  clio::run::TaskResume SendInvalidations(
      std::map<clio::run::u32, std::vector<BlobKey>> by_node);
  /** Reader side: register as a sharer with the blob's home. */
  clio::run::TaskResume RegisterSharer(const TagId &tag,
                                       const std::string &name);
  /** Ask the blobs' homes to drop every other container's copy. */
  clio::run::TaskResume FanoutInvalidate(const TagId &tag,
                                         const std::string &name);
  /** Wait until a blob has no unflushed ranges. */
  clio::run::TaskResume WaitBlobFlushed(const TagId &tag,
                                        const std::string &name);
  /** Fetch the whole blob from the chain into the segment. */
  clio::run::TaskResume FillFromChain(const TagId &tag, const std::string &name,
                                      bool *filled);

  WtcConfig config_;
  clio::cte::core::wtc::View view_;
  std::unique_ptr<clio::cte::core::Client> next_client_;
  std::atomic<bool> flushing_{false};
  std::mutex sharers_mtx_;
  std::map<std::string, SharerSet> sharers_;
};

/** Sharer-map key of a blob. */
inline std::string SharerKey(clio::run::u32 major, clio::run::u32 minor,
                             const std::string &name) {
  return std::to_string(major) + "." + std::to_string(minor) + "/" + name;
}

}  // namespace clio::cte::shm_wtc

#endif  // CLIO_CTE_SHM_WRITETHRU_CACHE_RUNTIME_H_
