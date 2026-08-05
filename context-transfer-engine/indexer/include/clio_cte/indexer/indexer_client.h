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

#ifndef CLIO_CTE_INDEXER_INDEXER_CLIENT_H_
#define CLIO_CTE_INDEXER_INDEXER_CLIENT_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/indexer/indexer_tasks.h>

#include <string>

namespace clio::cte::indexer {

/**
 * Indexer chimod client (issue #905). The pool speaks the CTE core's task
 * interface — point a clio::cte::core::Client at kIndexerPoolId (or any
 * chain that forwards into it) and AsyncSemanticSearch works unchanged;
 * this class only exists to create the pool with its chain configuration.
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;

  Client(const clio::run::PoolId &indexer_pool_id,
         const clio::run::PoolId &core_pool_id)
      : indexer_pool_id_(indexer_pool_id) {
    clio::cte::core::Client::Init(core_pool_id);
  }

#if CTP_IS_HOST
  /**
   * On-demand backfill: enumerate existing blobs matching the given regexes
   * (intersected with the module's configured scope) and enqueue them for
   * asynchronous indexing. The ONLY way to index pre-existing data besides
   * a tag's first-insertion backfill — restarts restore persisted state and
   * never rescan.
   */
  clio::run::Future<ReindexScanTask> AsyncReindexScan(
      const std::string &tag_regex = ".*",
      const std::string &blob_regex = ".*",
      const clio::run::PoolQuery &pool_query =
          clio::run::PoolQuery::Broadcast()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReindexScanTask>(
        clio::run::CreateTaskId(), indexer_pool_id_, pool_query, tag_regex,
        blob_regex);
    return ipc->Send(task);
  }

  /** Create the indexer container. Set params.next_pool_id_ to the pool it
   *  forwards to (usually the CTE core). */
  clio::run::Future<CreateTask> AsyncCreateIndexer(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &custom_pool_id, const IndexerConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        IndexerConfig::chimod_lib_name, pool_name, custom_pool_id, this,
        params);
    auto fut = ipc->Send(task);
    indexer_pool_id_ = custom_pool_id;
    return fut;
  }
#endif  // CTP_IS_HOST

  clio::run::PoolId indexer_pool_id_;
};

}  // namespace clio::cte::indexer

#endif  // CLIO_CTE_INDEXER_INDEXER_CLIENT_H_
