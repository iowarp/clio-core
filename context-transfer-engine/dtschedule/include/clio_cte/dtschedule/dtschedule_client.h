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

#ifndef CLIO_CTE_DTSCHEDULE_DTSCHEDULE_CLIENT_H_
#define CLIO_CTE_DTSCHEDULE_DTSCHEDULE_CLIENT_H_

#include <clio_cte/core/core_client.h>
#include <clio_cte/dtschedule/dtschedule_tasks.h>

#include <string>

namespace clio::cte::dtschedule {

/**
 * DTSchedule client. Inherits the full CTE core client API (put/get/tag
 * operations work unchanged) and adds dtschedule-specific verbs:
 * AsyncCreate (CreateParams for initialization), AsyncPollNodeLoad,
 * AsyncRegisterConsumer, AsyncGetDecisionStats, AsyncSetKnobs.
 *
 * NOTE: The pool speaks the CTE core's own task interface, so a caller can
 * use a clio::cte::core::Client pointed at kDtschedulePoolId (or set
 * CLIO_CTE_POOL=565.0) and all core verbs (PutBlob, GetBlob, etc.) work
 * transparently with compression/placement decisions.
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;

  /**
   * Initialize client to communicate with dtschedule pool.
   *
   * @param dtschedule_pool_id Pool ID of the dtschedule chimod (default 565.0)
   * @param core_pool_id Pool ID of the CTE core chimod the module sits over
   */
  Client(const clio::run::PoolId &dtschedule_pool_id,
         const clio::run::PoolId &core_pool_id)
      : dtschedule_pool_id_(dtschedule_pool_id) {
    clio::cte::core::Client::Init(core_pool_id);
  }

  /**
   * Create/initialize the dtschedule container.
   *
   * Uses PoolQuery::Dynamic() to leverage caching; kAdminPoolId for CreateTask
   * routing. Parses configuration from compose YAML and initializes tier tables.
   *
   * @param pool_query Router for pool creation (typically Dynamic())
   * @param pool_name Name for the pool (e.g., "clio_cte_dtschedule")
   * @param custom_pool_id Custom pool ID (default 565.0 if null)
   * @param params Configuration (DtscheduleConfig)
   * @return Future<CreateTask>
   */
  clio::run::Future<CreateTask> AsyncCreate(
      const clio::run::PoolQuery &pool_query,
      const std::string &pool_name,
      const clio::run::PoolId &custom_pool_id,
      const DtscheduleConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId,
        clio::run::PoolQuery::Dynamic(),  // CreateTask always uses Dynamic()
        DtscheduleConfig::chimod_lib_name, pool_name, custom_pool_id, this,
        params);
    auto fut = ipc->Send(task);
    dtschedule_pool_id_ = custom_pool_id;
    return fut;
  }

  /**
   * Poll local node CPU% and worker queue stats.
   *
   * Phase 1: real implementation. Phase 3+ uses this for load-aware decisions.
   *
   * @param pool_query Router for reaching the dtschedule pool
   * @return Future<PollNodeLoadTask> with NodeLoadSample result
   */
  clio::run::Future<PollNodeLoadTask> AsyncPollNodeLoad(
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<PollNodeLoadTask>(
        clio::run::CreateTaskId(), dtschedule_pool_id_, pool_query);
    return ipc->Send(task);
  }

  /**
   * Spawn a periodic load sampling task (phase 3+).
   *
   * Runs SampleLoad periodically on the local pool to maintain a ring buffer
   * of recent CPU% samples. The task is set to repeat with the given period.
   *
   * @param pool_query Router for reaching the dtschedule pool (typically Local)
   * @param period_us Period in microseconds between samples
   * @return Future<SampleLoadTask> (fire-and-forget; periodic spawning handled internally)
   */
  clio::run::Future<SampleLoadTask> AsyncSampleLoad(
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic(),
      int period_us = 1000000) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<SampleLoadTask>(
        clio::run::CreateTaskId(), dtschedule_pool_id_, pool_query);
    if (period_us > 0) {
      task->SetPeriod(period_us, clio::run::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }
    return ipc_manager->Send(task);
  }

  /**
   * Register a consumer node for a tag (phase 4+).
   *
   * Phase 1: stub (no-op). Phase 4 uses this to pick scenario 2/3.
   *
   * @param tag_id Tag identifier
   * @param consumer_node Node ID of the consumer
   * @param pool_query Router for reaching the dtschedule pool
   * @return Future<RegisterConsumerTask>
   */
  clio::run::Future<RegisterConsumerTask> AsyncRegisterConsumer(
      const TagId &tag_id, uint32_t consumer_node,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RegisterConsumerTask>(
        clio::run::CreateTaskId(), dtschedule_pool_id_, pool_query);
    if (!task.IsNull()) {
      task->tag_id_ = tag_id;
      task->consumer_node_ = consumer_node;
    }
    return ipc->Send(task);
  }

  /**
   * Compress a blob at a remote consumer node (phase 4c - scenario 3).
   *
   * Sends raw bytes to the consumer C for compression. C writes a local cache
   * copy, compresses, stores at the owner, and registers itself as the copy
   * holder. Returns the owner's result and OUT context.
   *
   * @param tag_id Tag identifier
   * @param blob_name Blob name
   * @param blob_data ShmPtr to raw bytes
   * @param size Original (uncompressed) size
   * @param score Tier score hint
   * @param owner_node Hash owner of the blob
   * @param lib Codec library name
   * @param preset Codec preset
   * @param context CTE context (version, etc.)
   * @param pool_query Router (typically PoolQuery::Physical(C))
   * @return Future<CompressAtTask> with OUT fields: comp_size_, ctime_ms_, store_ms_
   */
  clio::run::Future<CompressAtTask> AsyncCompressAt(
      const TagId &tag_id, const std::string &blob_name,
      ctp::ipc::ShmPtr<> blob_data, size_t size, float score,
      uint32_t owner_node, const std::string &lib, const std::string &preset,
      const Context &context,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CompressAtTask>(
        clio::run::CreateTaskId(), dtschedule_pool_id_, pool_query);
    if (!task.IsNull()) {
      task->tag_id_ = tag_id;
      task->blob_name_ = blob_name;
      task->blob_data_ = blob_data;
      task->size_ = size;
      task->score_ = score;
      task->owner_node_ = owner_node;
      task->lib_ = lib;
      task->preset_ = preset;
      task->context_ = context;
    }
    return ipc->Send(task);
  }

  /**
   * Get aggregated decision statistics.
   *
   * Returns counters: total puts, compressed, skipped, bytes in/out,
   * per-codec and per-scenario breakdowns. Used by tests and trace analysis.
   *
   * @param pool_query Router for reaching the dtschedule pool
   * @return Future<GetDecisionStatsTask> with DecisionStats result
   */
  clio::run::Future<GetDecisionStatsTask> AsyncGetDecisionStats(
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<GetDecisionStatsTask>(
        clio::run::CreateTaskId(), dtschedule_pool_id_, pool_query);
    return ipc->Send(task);
  }

  /**
   * Set ablation knobs at runtime (no restart needed).
   *
   * Allows changing CCM strategy, load awareness, workflow awareness,
   * scenario forcing, etc. during benchmarks. Used by load-response experiments.
   *
   * @param knobs New knobs (Knobs struct with changed fields)
   * @param pool_query Router for reaching the dtschedule pool
   * @return Future<SetKnobsTask>
   */
  clio::run::Future<SetKnobsTask> AsyncSetKnobs(
      const Knobs &knobs,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<SetKnobsTask>(
        clio::run::CreateTaskId(), dtschedule_pool_id_, pool_query);
    if (!task.IsNull()) {
      task->knobs_ = knobs;
    }
    return ipc->Send(task);
  }

  clio::run::PoolId dtschedule_pool_id_;
};

}  // namespace clio::cte::dtschedule

#endif  // CLIO_CTE_DTSCHEDULE_DTSCHEDULE_CLIENT_H_
