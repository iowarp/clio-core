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

#ifndef WRPCTE_CORE_CLIENT_H_
#define WRPCTE_CORE_CLIENT_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/singleton.h>
#include <clio_cte/api.h>
#include <clio_cte/core/core_tasks.h>

namespace clio::cte::core {

class Client : public chi::ContainerClient {
 public:
  CTP_CROSS_FUN Client() = default;
  CTP_CROSS_FUN explicit Client(const chi::PoolId &pool_id) { Init(pool_id); }

#if CTP_IS_HOST
  /**
   * Asynchronous container creation - returns immediately
   * @param pool_query Pool query for task routing
   * @param pool_name Name of the pool
   * @param custom_pool_id Explicit pool ID
   * @param params Create parameters
   */
  chi::Future<CreateTask> AsyncCreate(
      const chi::PoolQuery &pool_query, const std::string &pool_name,
      const chi::PoolId &custom_pool_id,
      const CreateParams &params = CreateParams()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    // CRITICAL: CreateTask MUST use admin pool for GetOrCreatePool processing
    // Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Always use admin pool for CreateTask
        pool_query,
        CreateParams::chimod_lib_name,  // ChiMod name from CreateParams
        pool_name,                      // Pool name from parameter
        custom_pool_id,                 // Explicit pool ID from parameter
        this,                           // Client pointer for PostWait
        params);                        // CreateParams with configuration

    // Submit to runtime
    return ipc_manager->Send(task);
  }

  /**
   * GPU-callable AsyncCreate: takes const char* names for GPU kernel use.
   * Routes to CPU admin worker via PoolQuery::ToLocalCpu().
   * @param pool_query Pool query for task routing
   * @param pool_name Name of the pool (const char*, GPU-safe)
   * @param custom_pool_id Explicit pool ID
   */
  chi::Future<CreateTask> AsyncCreate(
      const chi::PoolQuery &pool_query, const char *pool_name,
      const chi::PoolId &custom_pool_id) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,
        pool_query,
        CreateParams::chimod_lib_name,
        pool_name,
        custom_pool_id,
        static_cast<chi::ContainerClient *>(nullptr));
    return ipc_manager->Send(task);
  }

  /**
   * Monitor container state - asynchronous
   */
  chi::Future<MonitorTask> AsyncMonitor(const chi::PoolQuery &pool_query,
                                        const std::string &query) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<MonitorTask>(
        chi::CreateTaskId(), pool_id_, pool_query, query);
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target registration - returns immediately
   * @param target_name Name of the target to register
   * @param bdev_type Block device type
   * @param total_size Total size of the target
   * @param target_query Pool query for target routing
   * @param bdev_id Block device ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<RegisterTargetTask> AsyncRegisterTarget(
      const std::string &target_name, clio::run::bdev::BdevType bdev_type,
      chi::u64 total_size,
      const chi::PoolQuery &target_query = chi::PoolQuery::Local(),
      const chi::PoolId &bdev_id = chi::PoolId::GetNull(),
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<RegisterTargetTask>(
        chi::CreateTaskId(), pool_id_, pool_query, target_name,
        bdev_type, total_size, target_query, bdev_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target unregistration - returns immediately
   * @param target_name Name of the target to unregister
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<UnregisterTargetTask> AsyncUnregisterTarget(
      const std::string &target_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<UnregisterTargetTask>(
        chi::CreateTaskId(), pool_id_, pool_query, target_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target listing - returns immediately
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<ListTargetsTask> AsyncListTargets(
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<ListTargetsTask>(
        chi::CreateTaskId(), pool_id_, pool_query);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous target stats update - returns immediately
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @param period_ms Period for periodic execution in milliseconds (0 = one-shot)
   */
  chi::Future<StatTargetsTask> AsyncStatTargets(
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic(),
      chi::u32 period_ms = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<StatTargetsTask>(
        chi::CreateTaskId(), pool_id_, pool_query);

    // Set task as periodic if period is specified
    if (period_ms > 0) {
      task->SetPeriod(static_cast<double>(period_ms), chi::kMilli);
      task->SetFlags(TASK_PERIODIC);
    }

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get target info - returns target score, capacity, and stats
   * @param target_name Name of the target
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetTargetInfoTask> AsyncGetTargetInfo(
      const std::string &target_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetTargetInfoTask>(
        chi::CreateTaskId(), pool_id_, pool_query, target_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get or create tag - returns immediately
   * @param tag_name Name of the tag
   * @param tag_id Optional tag ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetOrCreateTagTask<CreateParams>> AsyncGetOrCreateTag(
      const std::string &tag_name,
      const TagId &tag_id = TagId::GetNull(),
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetOrCreateTagTask<CreateParams>>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_name,
        tag_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous put blob with optional compression context - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param offset Offset within blob
   * @param size Size of data
   * @param blob_data Shared memory pointer to data
   * @param score Blob score for placement: -1.0=unknown (auto), 0.0-1.0=explicit tier
   * @param context Compression context
   * @param flags Operation flags
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<PutBlobTask> AsyncPutBlob(
      const TagId &tag_id,
      const char *blob_name,
      chi::u64 offset, chi::u64 size,
      ctp::ipc::ShmPtr<> blob_data, float score = -1.0f,
      const Context &context = Context(),
      chi::u32 flags = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<PutBlobTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, offset, size, blob_data, score, context, flags);

    // Stamp submit time so the receiver can compute end-to-end
    // submit→recv latency. steady_clock is monotonic on each node;
    // cross-node comparisons assume NTP-synced wall clocks (ares is
    // ~ms-synced via the cluster's chrony). Set after NewTask so it
    // overwrites the ctor's 0.
    task.ptr_->submit_ts_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();

    return ipc_manager->Send(task);
  }

  /** std::string overload */
  chi::Future<PutBlobTask> AsyncPutBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      chi::u64 offset, chi::u64 size,
      ctp::ipc::ShmPtr<> blob_data, float score = -1.0f,
      const Context &context = Context(),
      chi::u32 flags = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    return AsyncPutBlob(tag_id, blob_name.c_str(), offset, size,
                        blob_data, score, context, flags, pool_query);
  }

  /**
   * Asynchronous get blob - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param offset Offset within blob
   * @param size Size of data
   * @param flags Operation flags
   * @param blob_data Shared memory pointer for output
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetBlobTask> AsyncGetBlob(
      const TagId &tag_id,
      const char *blob_name,
      chi::u64 offset, chi::u64 size,
      chi::u32 flags,
      ctp::ipc::ShmPtr<> blob_data,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetBlobTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, offset, size, flags, blob_data);

    return ipc_manager->Send(task);
  }

  /** std::string overload */
  chi::Future<GetBlobTask> AsyncGetBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      chi::u64 offset, chi::u64 size,
      chi::u32 flags,
      ctp::ipc::ShmPtr<> blob_data,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    return AsyncGetBlob(tag_id, blob_name.c_str(), offset, size,
                        flags, blob_data, pool_query);
  }

  /**
   * Asynchronous reorganize blob - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param new_score New placement score
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<ReorganizeBlobTask> AsyncReorganizeBlob(
      const TagId &tag_id, const std::string &blob_name, float new_score,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<ReorganizeBlobTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name, new_score);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous delete blob - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<DelBlobTask> AsyncDelBlob(
      const TagId &tag_id,
      const std::string &blob_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<DelBlobTask>(chi::CreateTaskId(), pool_id_,
                                                  pool_query,
                                                  tag_id, blob_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous delete tag by tag ID - returns immediately
   * @param tag_id Tag ID to delete
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<DelTagTask> AsyncDelTag(
      const TagId &tag_id,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<DelTagTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous delete tag by tag name - returns immediately
   * @param tag_name Tag name to delete
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<DelTagTask> AsyncDelTag(
      const std::string &tag_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<DelTagTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get tag size - returns immediately
   * @param tag_id Tag ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetTagSizeTask> AsyncGetTagSize(
      const TagId &tag_id,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetTagSizeTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous poll telemetry log - returns immediately
   * @param minimum_logical_time Minimum logical time filter
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<PollTelemetryLogTask> AsyncPollTelemetryLog(
      std::uint64_t minimum_logical_time,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<PollTelemetryLogTask>(
        chi::CreateTaskId(), pool_id_, pool_query,
        minimum_logical_time);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get blob score - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetBlobScoreTask> AsyncGetBlobScore(
      const TagId &tag_id, const std::string &blob_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetBlobScoreTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get blob size - returns immediately
   * @param tag_id Tag ID
   * @param blob_name Name of the blob
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetBlobSizeTask> AsyncGetBlobSize(
      const TagId &tag_id,
      const std::string &blob_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetBlobSizeTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get blob info - returns immediately
   * Gets comprehensive blob metadata including score and block placement
   * @param tag_id Tag ID for blob lookup
   * @param blob_name Name of the blob
   * @param pool_query Pool query for task routing (default: Dynamic)
   * @return Future containing score, size, and block placement info
   */
  chi::Future<GetBlobInfoTask> AsyncGetBlobInfo(
      const TagId &tag_id,
      const std::string &blob_name,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetBlobInfoTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id,
        blob_name);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous get contained blobs - returns immediately
   * @param tag_id Tag ID
   * @param pool_query Pool query for task routing (default: Dynamic)
   */
  chi::Future<GetContainedBlobsTask> AsyncGetContainedBlobs(
      const TagId &tag_id,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<GetContainedBlobsTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous tag query - returns immediately
   * @param tag_regex Tag regex pattern to match
   * @param max_tags Maximum number of tags to return (0 = no limit)
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Future for async operation
   */
  chi::Future<TagQueryTask> AsyncTagQuery(
      const std::string &tag_regex, chi::u32 max_tags = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<TagQueryTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_regex, max_tags);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous blob query - returns immediately
   * @param tag_regex Tag regex pattern to match
   * @param blob_regex Blob regex pattern to match
   * @param max_blobs Maximum number of blobs to return (0 = no limit)
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Future for async operation
   */
  chi::Future<BlobQueryTask> AsyncBlobQuery(
      const std::string &tag_regex, const std::string &blob_regex,
      chi::u32 max_blobs = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<BlobQueryTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_regex, blob_regex,
        max_blobs);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous semantic search — BM25 keyword scoring over blob bytes.
   * @param tag_regex   Full-string match against tag names (std::regex_match)
   * @param blob_regex  Full-string match against blob names within matching tags
   * @param query_text  Natural-language query; tokenized and scored vs blobs
   * @param k           Maximum number of results returned, ordered by
   *                    descending BM25 score. 0 means "no cap".
   * @param pool_query  Default Broadcast — same as BlobQuery — so the
   *                    search runs across every tag-owning container.
   */
  chi::Future<SemanticSearchTask> AsyncSemanticSearch(
      const std::string &tag_regex, const std::string &blob_regex,
      const std::string &query_text, chi::u32 k = 10,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<SemanticSearchTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_regex, blob_regex,
        query_text, k);
    return ipc_manager->Send(task);
  }
  /**
   * Asynchronous temporal search — filter blobs by last-modified timestamp.
   * @param tag_regex    Full-string match against tag names (std::regex_match)
   * @param blob_regex   Full-string match against blob names within matching tags
   * @param time_begin   Inclusive lower bound, epoch nanoseconds (0 = no lower bound)
   * @param time_end     Inclusive upper bound, epoch nanoseconds (0 = no upper bound)
   * @param max_entries  Cap on returned results, sorted ascending by timestamp (0 = unlimited)
   * @param pool_query   Default Broadcast — search across every tag-owning container.
   */
  chi::Future<TemporalSearchTask> AsyncTemporalSearch(
      const std::string &tag_regex, const std::string &blob_regex,
      Timestamp time_begin = 0, Timestamp time_end = 0,
      chi::u32 max_entries = 0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast()) {
    auto *ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<TemporalSearchTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_regex, blob_regex,
        time_begin, time_end, max_entries);
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous flush metadata - returns immediately
   * @param pool_query Pool query for task routing (default: Local)
   * @param period_us Period in microseconds (0 = one-shot)
   */
  chi::Future<FlushMetadataTask> AsyncFlushMetadata(
      const chi::PoolQuery &pool_query = chi::PoolQuery::Local(),
      double period_us = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<FlushMetadataTask>(
        chi::CreateTaskId(), pool_id_, pool_query);

    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous flush data - returns immediately
   * @param pool_query Pool query for task routing (default: Local)
   * @param target_persistence_level Minimum persistence level for flush target
   * @param period_us Period in microseconds (0 = one-shot)
   */
  chi::Future<FlushDataTask> AsyncFlushData(
      const chi::PoolQuery &pool_query = chi::PoolQuery::Local(),
      int target_persistence_level = 1,
      double period_us = 0) {
    auto *ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<FlushDataTask>(
        chi::CreateTaskId(), pool_id_, pool_query, target_persistence_level);

    if (period_us > 0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }

    return ipc_manager->Send(task);
  }
#endif  // CTP_IS_HOST
};

// Global pointer-based singleton for CTE client with lazy initialization
CLIO_CTE_DEFINE_GLOBAL_PTR_VAR_H(clio::cte::core::Client, g_cte_client);

/**
 * Initialize CTE client and configuration subsystem
 * @param config_path Optional path to configuration file
 * @param pool_query Pool query type for CTE container creation (default:
 * Dynamic)
 * @return true if initialization succeeded, false otherwise
 */
bool CLIO_CTE_CLIENT_INIT(
    const std::string &config_path = "",
    const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic());

/**
 * Tag wrapper class - provides convenient API for tag operations
 */
class Tag {
 private:
  TagId tag_id_;
  std::string tag_name_;

 public:
  /**
   * Constructor - Call the CLIO_CTE client GetOrCreateTag function
   * @param tag_name Tag name to get or create
   */
  explicit Tag(const std::string &tag_name);

  /**
   * Constructor - Does not call CLIO_CTE client function, just sets the TagId
   * variable
   * @param tag_id Tag ID to use directly
   */
  explicit Tag(const TagId &tag_id);

  /**
   * PutBlob - Allocates a SHM pointer and then calls PutBlob (SHM)
   * @param blob_name Name of the blob
   * @param data Raw data pointer
   * @param data_size Size of data
   * @param off Offset within blob (default 0)
   * @param score Blob score for placement decisions (default 1.0)
   * @param context Compression context for workflow-aware decisions (default empty)
   */
  void PutBlob(const std::string &blob_name, const char *data, size_t data_size,
               size_t off = 0, float score = 1.0f, const Context &context = Context());

  /**
   * PutBlob (SHM) - Direct shared memory version
   * @param blob_name Name of the blob
   * @param data Shared memory pointer to data
   * @param data_size Size of data
   * @param off Offset within blob (default 0)
   * @param score Blob score for placement: -1.0=unknown (auto), 0.0-1.0=explicit tier
   * @param context Compression context for workflow-aware decisions (default empty)
   */
  void PutBlob(const std::string &blob_name, const ctp::ipc::ShmPtr<> &data,
               size_t data_size, size_t off = 0, float score = -1.0f,
               const Context &context = Context());

  /**
   * Asynchronous PutBlob (SHM) - Caller must manage shared memory lifecycle
   * @param blob_name Name of the blob
   * @param data Shared memory pointer to data (must remain valid until task
   * completes)
   * @param data_size Size of data
   * @param off Offset within blob (default 0)
   * @param score Blob score for placement: -1.0=unknown (auto), 0.0-1.0=explicit tier
   * @param context Compression context for workflow-aware decisions (default empty)
   * @return Task pointer for async operation
   * @note For raw data, caller must allocate shared memory using
   * CLIO_IPC->AllocateBuffer<void>() and keep the FullPtr alive until the async
   * task completes
   */
  chi::Future<PutBlobTask> AsyncPutBlob(const std::string &blob_name,
                                        const ctp::ipc::ShmPtr<> &data,
                                        size_t data_size, size_t off = 0,
                                        float score = -1.0f,
                                        const Context &context = Context());

  /**
   * GetBlob - Allocates shared memory, retrieves blob data, copies to output
   * buffer
   * @param blob_name Name of the blob to retrieve
   * @param data Output buffer to copy blob data into (must be pre-allocated by
   * caller)
   * @param data_size Size of data to retrieve (must be > 0)
   * @param off Offset within blob (default 0)
   * @note Automatically handles shared memory allocation/deallocation
   */
  void GetBlob(const std::string &blob_name, char *data, size_t data_size,
               size_t off = 0);

  /**
   * GetBlob (SHM) - Retrieves blob data into pre-allocated shared memory buffer
   * @param blob_name Name of the blob to retrieve
   * @param data Pre-allocated shared memory pointer for output data (must not
   * be null)
   * @param data_size Size of data to retrieve (must be > 0)
   * @param off Offset within blob (default 0)
   * @note Caller must pre-allocate shared memory using
   * CLIO_IPC->AllocateBuffer<void>(data_size)
   */
  void GetBlob(const std::string &blob_name, ctp::ipc::ShmPtr<> data,
               size_t data_size, size_t off = 0);

  /**
   * Get blob score
   * @param blob_name Name of the blob
   * @return Blob score (0.0-1.0)
   */
  float GetBlobScore(const std::string &blob_name);

  /**
   * Get blob size
   * @param blob_name Name of the blob
   * @return Blob size in bytes
   */
  chi::u64 GetBlobSize(const std::string &blob_name);

  /**
   * Get all blob names contained in this tag
   * @return Vector of blob names in this tag
   */
  std::vector<std::string> GetContainedBlobs();

  /**
   * Reorganize blob with new score for data placement optimization
   * @param blob_name Name of the blob to reorganize
   * @param new_score New placement score (0.0-1.0, higher = faster tier)
   */
  void ReorganizeBlob(const std::string &blob_name, float new_score);

  /**
   * Get the TagId for this tag
   * @return TagId of this tag
   */
  const TagId &GetTagId() const { return tag_id_; }
};

// Flush + reset the global Tag::PutBlob(const char*) timing accumulator
// and print a per-call breakdown of alloc / memcpy / RPC / free.
// Intended to be called by the POSIX (and other) adapters at the end of
// a `Filesystem::Write` so a single line summarises one user-visible
// write operation. Pass a short label that identifies the caller
// (e.g. "Write off=0 size=4G").
void FlushPutBlobTiming(const char *label);

}  // namespace clio::cte::core

// Global singleton macro for CTE client access (returns pointer, not reference)
#define CLIO_CTE_CLIENT                               \
  (&(*CTP_GET_GLOBAL_PTR_VAR(clio::cte::core::Client, \
                              clio::cte::core::g_cte_client)))

// Backward-compat aliases for the WRP_ -> CLIO_ rename. External code that
// still uses wrp_cte::core::* (e.g. wrp_cte::core::g_cte_client) resolves
// transparently to clio::cte::core::*. Paired with the wrp_cte/ forwarder
// shim header tree, this gives source-level compat for downstream projects
// that haven't migrated their identifiers yet. See rebranding.md.
// Pre-`clio::cte`-rename intermediate spelling.  In-tree code now uses
// `clio::cte::core::*`; downstream that already migrated off `wrp_cte::*`
// to the `clio_cte::*` waypoint keeps compiling via this alias.  Safe to
// use the simple `namespace X = Y;` form because no external chimod opens
// `namespace clio_cte::xxx {}`.
namespace clio_cte = clio::cte;
namespace wrp_cte = clio::cte;

// Client singleton accessor macro: legacy name.
#define WRP_CTE_CLIENT CLIO_CTE_CLIENT

// Client init function: legacy name aliased to canonical via #define so the
// call `wrp_cte::core::WRP_CTE_CLIENT_INIT(...)` expands to
// `wrp_cte::core::CLIO_CTE_CLIENT_INIT(...)` and resolves through the
// `wrp_cte = clio_cte` namespace alias.
#define WRP_CTE_CLIENT_INIT CLIO_CTE_CLIENT_INIT

#endif  // WRPCTE_CORE_CLIENT_H_
