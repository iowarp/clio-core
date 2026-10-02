/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_CHECKPOINT_CHECKPOINT_CLIENT_H_
#define CLIO_CTE_CHECKPOINT_CHECKPOINT_CLIENT_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/checkpoint/checkpoint_tasks.h>

#include <string>

namespace clio::cte::checkpoint {

/**
 * Checkpoint chimod client. The pool speaks the CTE core's task interface;
 * this class only exists to create the pool. Callers never address it for
 * data -- the CORE dispatches faults to it by itself.
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;

  Client(const clio::run::PoolId &checkpoint_pool_id,
         const clio::run::PoolId &core_pool_id)
      : checkpoint_pool_id_(checkpoint_pool_id) {
    clio::cte::core::Client::Init(core_pool_id);
  }

#if CTP_IS_HOST
  /** Create the checkpoint container (idempotent). params.next_pool_id_ is
   *  the pool it resolves sources through (usually the core). */
  clio::run::Future<CreateTask> AsyncCreateCheckpoint(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &custom_pool_id, const CheckpointConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        CheckpointConfig::chimod_lib_name, pool_name, custom_pool_id, this,
        params);
    auto fut = ipc->Send(task);
    checkpoint_pool_id_ = custom_pool_id;
    return fut;
  }
#endif  // CTP_IS_HOST

  clio::run::PoolId checkpoint_pool_id_;
};

}  // namespace clio::cte::checkpoint

#endif  // CLIO_CTE_CHECKPOINT_CHECKPOINT_CLIENT_H_
