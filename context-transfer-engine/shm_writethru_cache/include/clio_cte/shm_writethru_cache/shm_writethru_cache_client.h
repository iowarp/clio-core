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

#ifndef CLIO_CTE_SHM_WRITETHRU_CACHE_CLIENT_H_
#define CLIO_CTE_SHM_WRITETHRU_CACHE_CLIENT_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/shm_writethru_cache/shm_writethru_cache_tasks.h>

#include <string>

namespace clio::cte::shm_wtc {

/**
 * Admin-facing client of the shm write-through cache pool. Data traffic does
 * NOT go through this class: bind a clio::cte::core::Client to the pool and
 * it uses the segment directly (see core_client.h, shm write-through cache).
 */
class Client : public clio::run::ContainerClient {
 public:
  Client() = default;
  explicit Client(const clio::run::PoolId &pool_id) { Init(pool_id); }

#if CTP_IS_HOST
  /**
   * Create (or get) the cache pool.
   * @param pool_query where to create it (Dynamic/Local/Broadcast)
   * @param pool_name pool name
   * @param custom_pool_id pool id to use
   * @param params configuration (next_pool_id_ is required)
   * @return future over the create task
   */
  clio::run::Future<CreateTask> AsyncCreate(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &custom_pool_id, const WtcConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        WtcConfig::chimod_lib_name, pool_name, custom_pool_id, this, params);
    auto fut = ipc->Send(task);
    pool_id_ = custom_pool_id;
    return fut;
  }

  /**
   * Run one flush pass now (tests; the periodic task normally does this).
   * @param pool_query target container
   * @return future over the flush task
   */
  clio::run::Future<FlushLogTask> AsyncFlushLog(
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<FlushLogTask>(clio::run::CreateTaskId(), pool_id_,
                                           pool_query);
    return ipc->Send(task);
  }
#endif  // CTP_IS_HOST
};

}  // namespace clio::cte::shm_wtc

#endif  // CLIO_CTE_SHM_WRITETHRU_CACHE_CLIENT_H_
