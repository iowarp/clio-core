/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_CACHE_CACHE_CLIENT_H_
#define CLIO_CTE_CACHE_CACHE_CLIENT_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/cache/cache_tasks.h>

#include <string>

namespace clio::cte::cache {

/**
 * Cache chimod client (issue #886). The pool speaks the CTE core's task
 * interface — point a clio::cte::core::Client at kCachePoolId (or set
 * CLIO_CTE_POOL=563.0) for the data path; this class only exists to create
 * the pool with its chain configuration.
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;

  Client(const clio::run::PoolId &cache_pool_id,
         const clio::run::PoolId &core_pool_id)
      : cache_pool_id_(cache_pool_id) {
    clio::cte::core::Client::Init(core_pool_id);
  }

#if CTP_IS_HOST
  /** Create the cache container. Set params.next_pool_id_ to the pool it
   *  pushes authoritative bytes to (usually the compressor). */
  clio::run::Future<CreateTask> AsyncCreateCache(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &custom_pool_id, const CacheConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        CacheConfig::chimod_lib_name, pool_name, custom_pool_id, this,
        params);
    auto fut = ipc->Send(task);
    cache_pool_id_ = custom_pool_id;
    return fut;
  }
#endif  // CTP_IS_HOST

  clio::run::PoolId cache_pool_id_;
};

}  // namespace clio::cte::cache

#endif  // CLIO_CTE_CACHE_CACHE_CLIENT_H_
