/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_LZ4_CACHE_CLIENT_H_
#define CLIO_CTE_LZ4_CACHE_CLIENT_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/lz4/lz4_tasks.h>

#include <string>

namespace clio::cte::lz4 {

/**
 * LZ4 chimod client. The pool speaks the CTE core's task interface -- point a
 * clio::cte::core::Client at kLz4PoolId for the data path; this class only
 * creates the pool with its chain configuration.
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;

  Client(const clio::run::PoolId &lz4_pool_id,
         const clio::run::PoolId &core_pool_id)
      : lz4_pool_id_(lz4_pool_id) {
    clio::cte::core::Client::Init(core_pool_id);
  }

#if CTP_IS_HOST
  /** Create the LZ4 container. Set params.next_pool_id_ to the pool it
   *  forwards to (usually the CTE core). */
  clio::run::Future<CreateTask> AsyncCreateLz4(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &custom_pool_id, const Lz4Config &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        Lz4Config::chimod_lib_name, pool_name, custom_pool_id, this, params);
    auto fut = ipc->Send(task);
    lz4_pool_id_ = custom_pool_id;
    return fut;
  }
#endif  // CTP_IS_HOST

  clio::run::PoolId lz4_pool_id_;
};

}  // namespace clio::cte::lz4

#endif  // CLIO_CTE_LZ4_CACHE_CLIENT_H_
