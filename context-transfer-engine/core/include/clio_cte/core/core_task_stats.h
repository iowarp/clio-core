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

#ifndef CLIO_CTE_CORE_CORE_TASK_STATS_H_
#define CLIO_CTE_CORE_CORE_TASK_STATS_H_

/**
 * @file core_task_stats.h
 * Cost estimates (Container::GetTaskStats) for EVERY CTE core method.
 *
 * Shared by the core runtime and by CoreInterposer, so a task costs the same
 * whether it reaches the core directly or through an interposer chain
 * (cache, compressor, replication, indexer, write-back / write-through).
 *
 * Data verbs scale with their payload: io_size_ = bytes (this also picks the
 * bulk network lane at >= 4 KiB), compute_ = bytes / ~10 KB per CPU us + 2,
 * wall_time_ = bytes / ~500 MB/s (clio_runtime/task_cost.h). Batched verbs
 * sum their records. Control and metadata verbs are explicit constants seeded
 * from the measured cost of their nearest relative (TagQuery ~150-200 us,
 * a single metadata lookup ~10 us); the per-method coefficient learned from
 * real completions corrects the scale, so only the shape matters.
 */

#include <clio_runtime/task_cost.h>
#include <clio_cte/core/core_tasks.h>

namespace clio::cte::core {

/**
 * Payload bytes of a scalar put/get: size_, or the sum of segments_ when the
 * task is vectored (then size_ is ignored by the handler).
 * @tparam TaskT PutBlobTask, GetBlobTask, PodPutBlobTask or PodGetBlobTask
 * @param t the task
 * @return bytes the task moves
 */
template <typename TaskT>
inline clio::run::u64 CoreBlobBytes(const TaskT *t) {
  if constexpr (TaskT::kSupportsVectored) {
    if (!t->segments_.empty()) {
      clio::run::u64 total = 0;
      for (size_t i = 0; i < t->segments_.size(); ++i) {
        total += t->segments_[i].size_;
      }
      return total;
    }
  }
  return t->size_;
}

/**
 * Payload bytes and record count of a batched POD task (PodMultiGet/Put).
 * @tparam TaskT PodMultiGetBlobTask or PodMultiPutBlobTask
 * @param t the task
 * @param records output: number of live records
 * @return bytes summed over the live records
 */
template <typename TaskT>
inline clio::run::u64 CorePodMultiBytes(const TaskT *t,
                                        clio::run::u64 *records) {
  const clio::run::u32 n = t->count_ < kPodMultiMax ? t->count_ : kPodMultiMax;
  clio::run::u64 total = 0;
  for (clio::run::u32 i = 0; i < n; ++i) {
    total += t->reqs_[i].size_;
  }
  *records = n;
  return total;
}

/**
 * Stat for a batch of `records` puts/gets moving `bytes` in total: the data
 * term plus ~2 us of per-record dispatch (each record is a nested put/get).
 * @param bytes total payload bytes
 * @param records number of records in the batch
 * @return the populated TaskStat
 */
inline clio::run::TaskStat CoreBatchDataStat(clio::run::u64 bytes,
                                             clio::run::u64 records) {
  namespace cost = clio::run::cost;
  return cost::Data(bytes, cost::kCopyBytesPerCpuUs,
                    cost::kStorageBytesPerWallUs,
                    cost::kDataBaseComputeUs + 2 * records,
                    3.0f * static_cast<float>(records));
}

/**
 * Cost of the core's data-carrying verbs.
 * @param task the task (method_ selects the case)
 * @param stat output; set when the method moves payload bytes
 * @return true if the method was handled
 */
inline bool CoreDataTaskStats(const clio::run::Task *task,
                              clio::run::TaskStat *stat) {
  namespace cost = clio::run::cost;
  clio::run::u64 records = 0;
  switch (task->method_) {
    case Method::kPutBlob:
      *stat = cost::Data(CoreBlobBytes(static_cast<const PutBlobTask *>(task)));
      return true;
    case Method::kGetBlob:
      *stat = cost::Data(CoreBlobBytes(static_cast<const GetBlobTask *>(task)));
      return true;
    case Method::kPodPutBlob:
      *stat = cost::Data(
          CoreBlobBytes(static_cast<const PodPutBlobTask *>(task)));
      return true;
    case Method::kPodGetBlob:
      *stat = cost::Data(
          CoreBlobBytes(static_cast<const PodGetBlobTask *>(task)));
      return true;
    case Method::kMultiPutBlob: {
      // descs_ is serialized; the record count is not known without decoding
      // it, so the batch is charged as one put of data_len_ bytes.
      const auto *t = static_cast<const MultiPutBlobTask *>(task);
      *stat = cost::Data(t->data_len_);
      return true;
    }
    case Method::kPodMultiPutBlob: {
      const clio::run::u64 bytes = CorePodMultiBytes(
          static_cast<const PodMultiPutBlobTask *>(task), &records);
      *stat = CoreBatchDataStat(bytes, records);
      return true;
    }
    case Method::kPodMultiGetBlob: {
      const clio::run::u64 bytes = CorePodMultiBytes(
          static_cast<const PodMultiGetBlobTask *>(task), &records);
      *stat = CoreBatchDataStat(bytes, records);
      return true;
    }
    case Method::kEvict: {
      // Demotes (copies down) or drops up to bytes_ of data from a tier.
      const auto *t = static_cast<const EvictTask *>(task);
      *stat = cost::Data(t->bytes_, cost::kCopyBytesPerCpuUs,
                         cost::kStorageBytesPerWallUs, 20, 50.0f);
      return true;
    }
    default:
      return false;
  }
}

/**
 * Cost of the core's batched metadata verbs, scaled by their record count.
 * Serialized record lists are not decoded here; the count is estimated from
 * their byte length (a TagId is 8 bytes, a name ~32).
 * @param task the task (method_ selects the case)
 * @param stat output; set when the method is a batched metadata verb
 * @return true if the method was handled
 */
inline bool CoreBatchMetaTaskStats(const clio::run::Task *task,
                                   clio::run::TaskStat *stat) {
  namespace cost = clio::run::cost;
  constexpr clio::run::u64 kKeyBytes = 40;  // TagId + typical blob name
  constexpr clio::run::u64 kTagBytes = 8;   // serialized TagId
  switch (task->method_) {
    case Method::kPodMultiScore: {
      const auto *t = static_cast<const PodMultiScoreTask *>(task);
      const clio::run::u64 n =
          t->count_ < kPodMultiMax ? t->count_ : kPodMultiMax;
      *stat = cost::PerItem(n, 5, 2, 10.0f, 3.0f);  // one score write each
      return true;
    }
    case Method::kMultiDelBlob: {
      const auto *t = static_cast<const MultiDelBlobTask *>(task);
      *stat = cost::PerItem(t->keys_.size() / kKeyBytes + 1, 10, 10, 15.0f,
                            15.0f);
      return true;
    }
    case Method::kMultiDelTag: {
      const auto *t = static_cast<const MultiDelTagTask *>(task);
      *stat = cost::PerItem(t->tags_.size() / kTagBytes + 1, 10, 10, 15.0f,
                            15.0f);
      return true;
    }
    case Method::kUpdateTagNames: {
      const auto *t = static_cast<const UpdateTagNamesTask *>(task);
      *stat = cost::PerItem(t->ops_.size() / kKeyBytes + 1, 10, 10, 15.0f,
                            15.0f);
      return true;
    }
    default:
      return false;
  }
}

/**
 * Cost of the core's scans, searches, flushes and target management.
 * @param method the core method id
 * @param stat output; set when the method is one of these verbs
 * @return true if the method was handled
 */
inline bool CoreControlTaskStats(clio::run::u32 method,
                                 clio::run::TaskStat *stat) {
  namespace cost = clio::run::cost;
  switch (method) {
    case Method::kTagQuery:
      // Trigram-prefiltered regex search + a std::regex compile per call
      // (~150-200 us measured); clio-fs readdir and rename are built on it.
      *stat = cost::Scan();
      return true;
    case Method::kBlobQuery:
    case Method::kListLocalBlobs:
      *stat = cost::Fixed(300, 400.0f);  // regex over tags AND their blobs
      return true;
    case Method::kSemanticSearch:
      *stat = cost::Fixed(500, 700.0f);  // BM25 scoring over the text index
      return true;
    case Method::kTemporalSearch:
    case Method::kGetContainedBlobs:
      *stat = cost::Fixed(100, 150.0f);  // walk one tag's / every blob's mtime
      return true;
    case Method::kRenameTag:
      // A lookup plus three name steps on up to three containers.
      *stat = cost::Fixed(400, 500.0f);
      return true;
    case Method::kRegisterTarget:
      *stat = cost::ContainerCreate();  // creates the backing bdev pool
      return true;
    case Method::kUnregisterTarget:
      *stat = cost::ContainerDestroy();
      return true;
    case Method::kStatTargets:
    case Method::kGetCapacity:
      *stat = cost::Fixed(20, 100.0f);  // one GetStats per registered target
      return true;
    case Method::kFlushMetadata:
      *stat = cost::Fixed(100, 1000.0f);  // snapshot/compact the metadata log
      return true;
    case Method::kFlushData:
    case Method::kSyncTag:
      // Walks dirty blobs and issues their copies / bdev syncs as subtasks;
      // the volume is unknown up front, so this seeds the orchestration.
      *stat = cost::Fixed(100, 1000.0f);
      return true;
    case Method::kReorganizeBlob:
    case Method::kPodReorganizeBlob:
      // Re-place one blob; the copy itself runs as bdev subtasks.
      *stat = cost::Fixed(30, 100.0f);
      return true;
    case Method::kDynamicReorganize:
      *stat = cost::Fixed(100, 200.0f);  // periodic organizer pass
      return true;
    case Method::kPollTelemetryLog:
      *stat = cost::Fixed(20, 30.0f);  // drain the telemetry ring
      return true;
    case Method::kCreate:
      *stat = cost::ContainerCreate();
      return true;
    case Method::kDestroy:
      *stat = cost::ContainerDestroy();
      return true;
    default:
      return false;
  }
}

/**
 * Cost of the core's single-object metadata verbs: one lookup or mutation
 * against the tag / blob / target maps.
 * @param method the core method id
 * @param stat output; set when the method is one of these verbs
 * @return true if the method was handled
 */
inline bool CoreMetadataTaskStats(clio::run::u32 method,
                                  clio::run::TaskStat *stat) {
  namespace cost = clio::run::cost;
  switch (method) {
    case Method::kGetBlobSize:
    case Method::kGetResidency:
    case Method::kGetOrCreateTag:
    case Method::kGetTagSize:
    case Method::kGetTagName:
    case Method::kGetBlobScore:
    case Method::kGetBlobInfo:
    case Method::kDelBlob:
    case Method::kDelTag:
    case Method::kTruncateBlob:
    case Method::kGetOrCreateTagAlias:
    case Method::kGetNumAliases:
    case Method::kRegisterReplicaContainer:
    case Method::kTagXattr:
    case Method::kGetTargetInfo:
    case Method::kCowLink:
      *stat = cost::Metadata();
      return true;
    case Method::kListTargets:
    case Method::kSetDurability:
    case Method::kReorganizeHint:
    case Method::kMonitor:
      *stat = cost::Fixed(5, 10.0f);  // read or flip one in-memory field
      return true;
    default:
      return false;
  }
}

/**
 * Cost estimate for any CTE core method (the core's GetTaskStats).
 * @param task the task to estimate (may be null)
 * @return the task's cost features; an unknown method id gets the metadata
 *         constant rather than zeros
 */
inline clio::run::TaskStat CoreTaskStats(const clio::run::Task *task) {
  clio::run::TaskStat stat;
  if (task == nullptr) {
    return stat;
  }
  if (CoreDataTaskStats(task, &stat) || CoreBatchMetaTaskStats(task, &stat) ||
      CoreControlTaskStats(task->method_, &stat) ||
      CoreMetadataTaskStats(task->method_, &stat)) {
    return stat;
  }
  return clio::run::cost::Metadata();
}

}  // namespace clio::cte::core

#endif  // CLIO_CTE_CORE_CORE_TASK_STATS_H_
