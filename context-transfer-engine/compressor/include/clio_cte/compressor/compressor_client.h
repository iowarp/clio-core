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

#ifndef CLIO_CTE_COMPRESSOR_CLIENT_H_
#define CLIO_CTE_COMPRESSOR_CLIENT_H_

#include <clio_cte/core/core_client.h>
#include <clio_cte/compressor/compressor_tasks.h>

namespace clio::cte::compressor {

/**
 * Transparent compression client.
 *
 * Inherits the full CTE core client API so that user code can swap
 * between clio::cte::core::Client and clio::cte::compressor::Client
 * without any changes. AsyncPutBlob is intercepted and routed through
 * the compressor chimod (DynamicSchedule), and AsyncGetBlob is routed
 * through decompression. All other methods (target management, tag
 * management, metadata, etc.) pass through to the CTE core directly.
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;

  /**
   * Construct a compressor client (runtime-internal use).
   * Only the compressor pool is known; core pool is resolved per-task.
   */
  explicit Client(const clio::run::PoolId &compressor_pool_id)
      : compressor_pool_id_(compressor_pool_id) {
    clio::cte::core::Client::Init(compressor_pool_id);
  }

  /**
   * Construct a compressor client (user-facing).
   * @param compressor_pool_id Pool ID of the compressor chimod
   * @param core_pool_id Pool ID of the CTE core chimod
   */
  Client(const clio::run::PoolId &compressor_pool_id,
         const clio::run::PoolId &core_pool_id)
      : compressor_pool_id_(compressor_pool_id) {
    clio::cte::core::Client::Init(core_pool_id);
  }

  /**
   * Create the compressor container.
   */
  clio::run::Future<CreateTask> AsyncCreateCompressor(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &custom_pool_id) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        "clio_cte_compressor", pool_name, custom_pool_id, this);
    return ipc_manager->Send(task);
  }

  // ------------------------------------------------------------------
  // Overridden data-path methods: route through compressor
  // ------------------------------------------------------------------

  /**
   * AsyncPutBlob override: routes data through compression before storage.
   * The compressor runtime will analyze, compress, then call core PutBlob.
   */
  clio::run::Future<DynamicScheduleTask> AsyncPutBlob(
      const clio::cte::core::TagId &tag_id, const char *blob_name,
      clio::run::u64 offset, clio::run::u64 size, ctp::ipc::ShmPtr<> blob_data,
      float score = -1.0f,
      const clio::cte::core::Context &context = clio::cte::core::Context(),
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_IPC;
    // core_pool_id is a fallback — the runtime uses next_pool_id from
    // compose config when available.
    auto task = ipc_manager->NewTask<DynamicScheduleTask>(
        clio::run::CreateTaskId(), compressor_pool_id_, pool_query, tag_id,
        blob_name, offset, size, blob_data, score, context, flags,
        clio::run::PoolId::GetNull());
    return ipc_manager->Send(task);
  }

  /** std::string overload */
  clio::run::Future<DynamicScheduleTask> AsyncPutBlob(
      const clio::cte::core::TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size, ctp::ipc::ShmPtr<> blob_data,
      float score = -1.0f,
      const clio::cte::core::Context &context = clio::cte::core::Context(),
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncPutBlob(tag_id, blob_name.c_str(), offset, size, blob_data,
                        score, context, flags, pool_query);
  }

  /**
   * AsyncGetBlob override: retrieves and decompresses data transparently.
   * The compressor runtime will call core GetBlob then decompress.
   */
  clio::run::Future<DecompressTask> AsyncGetBlob(
      const clio::cte::core::TagId &tag_id, const char *blob_name,
      clio::run::u64 offset, clio::run::u64 size, clio::run::u32 flags,
      ctp::ipc::ShmPtr<> blob_data,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<DecompressTask>(
        clio::run::CreateTaskId(), compressor_pool_id_, pool_query, tag_id,
        blob_name, offset, size, flags, blob_data,
        clio::run::PoolId::GetNull());
    return ipc_manager->Send(task);
  }

  /** std::string overload */
  clio::run::Future<DecompressTask> AsyncGetBlob(
      const clio::cte::core::TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size, clio::run::u32 flags,
      ctp::ipc::ShmPtr<> blob_data,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncGetBlob(tag_id, blob_name.c_str(), offset, size, flags,
                        blob_data, pool_query);
  }

  // ------------------------------------------------------------------
  // Compressor-specific methods (used internally by compressor runtime)
  // ------------------------------------------------------------------

  /**
   * Explicit dynamic scheduling — analyzes data, picks best compressor,
   * compresses, and stores via core PutBlob.
   */
  clio::run::Future<DynamicScheduleTask> AsyncDynamicSchedule(
      const clio::run::PoolQuery &pool_query,
      const clio::cte::core::TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size, ctp::ipc::ShmPtr<> blob_data,
      float score, const clio::cte::core::Context &context,
      clio::run::u32 flags, const clio::run::PoolId &core_pool_id) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<DynamicScheduleTask>(
        clio::run::CreateTaskId(), compressor_pool_id_, pool_query, tag_id,
        blob_name, offset, size, blob_data, score, context, flags,
        core_pool_id);
    return ipc_manager->Send(task);
  }

  /**
   * Explicit compression — compresses data and stores via core PutBlob.
   */
  clio::run::Future<CompressTask> AsyncCompress(
      const clio::run::PoolQuery &pool_query,
      const clio::cte::core::TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size, ctp::ipc::ShmPtr<> blob_data,
      float score, const clio::cte::core::Context &context,
      clio::run::u32 flags, const clio::run::PoolId &core_pool_id) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<CompressTask>(
        clio::run::CreateTaskId(), compressor_pool_id_, pool_query, tag_id,
        blob_name, offset, size, blob_data, score, context, flags,
        core_pool_id);
    return ipc_manager->Send(task);
  }

  /**
   * Explicit decompression — retrieves via core GetBlob and decompresses.
   */
  clio::run::Future<DecompressTask> AsyncDecompressExplicit(
      const clio::run::PoolQuery &pool_query,
      const clio::cte::core::TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size, clio::run::u32 flags,
      ctp::ipc::ShmPtr<> blob_data, const clio::run::PoolId &core_pool_id) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<DecompressTask>(
        clio::run::CreateTaskId(), compressor_pool_id_, pool_query, tag_id,
        blob_name, offset, size, flags, blob_data, core_pool_id);
    return ipc_manager->Send(task);
  }

  /**
   * Poll a node's CPU utilization and worker load.
   * Use PoolQuery::Physical(node_id) to target a specific consumer node.
   * @param pool_query Pool routing — use Physical(node_id) for a remote node
   *                   or Local() for the current node.
   * @return Future for the PollNodeLoadTask whose OUT sample_ field carries
   *         the node's CPU% and aggregated worker load.
   */
  clio::run::Future<PollNodeLoadTask> AsyncPollNodeLoad(
      const clio::run::PoolQuery &pool_query) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<PollNodeLoadTask>(
        clio::run::CreateTaskId(), compressor_pool_id_, pool_query);
    return ipc_manager->Send(task);
  }

  /**
   * Spawn the periodic PollConsumers task that iterates this container's
   * tracked consumer list and calls PollNodeLoad on each consumer node.
   * @param pool_query Pool routing — typically Local().
   * @param period_us Period in microseconds (default 5s).
   * @return Future for the PollConsumersTask.
   */
  clio::run::Future<PollConsumersTask> AsyncPollConsumers(
      const clio::run::PoolQuery &pool_query, double period_us = 5000000) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<PollConsumersTask>(
        clio::run::CreateTaskId(), compressor_pool_id_, pool_query);
    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }
    return ipc_manager->Send(task);
  }

 private:
  clio::run::PoolId compressor_pool_id_;
};

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_CLIENT_H_
