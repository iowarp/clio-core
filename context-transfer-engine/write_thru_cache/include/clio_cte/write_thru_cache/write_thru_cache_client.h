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

#ifndef CLIO_CTE_WRITE_THRU_CACHE_WRITE_THRU_CACHE_CLIENT_H_
#define CLIO_CTE_WRITE_THRU_CACHE_WRITE_THRU_CACHE_CLIENT_H_

#include <string>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/write_thru_cache/write_thru_cache_tasks.h>

namespace clio::cte::write_thru_cache {

/**
 * Write-through cache client. The pool speaks the CTE core's
 * task interface, so this is a clio::cte::core::Client bound to the cache
 * pool plus the call that creates it.
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;
  explicit Client(const clio::run::PoolId &pool_id)
      : clio::cte::core::Client(pool_id) {}

#if CTP_IS_HOST
  /**
   * Create the cache pool and bind this client to it.
   * @param pool_query where to create it (Dynamic/Local)
   * @param pool_name pool name
   * @param pool_id pool id to create
   * @param params next_volatile / next_persistent and policy keys
   * @return future for the creation task
   */
  clio::run::Future<CreateTask> AsyncCreateCache(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &pool_id, const WriteThruCacheConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        WriteThruCacheConfig::chimod_lib_name, pool_name, pool_id, this,
        params);
    Init(pool_id);
    return ipc->Send(task);
  }
#endif  // CTP_IS_HOST
};

}  // namespace clio::cte::write_thru_cache

#endif  // CLIO_CTE_WRITE_THRU_CACHE_WRITE_THRU_CACHE_CLIENT_H_
