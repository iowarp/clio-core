/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_REPLICATION_REPLICATION_CLIENT_H_
#define CLIO_CTE_REPLICATION_REPLICATION_CLIENT_H_

#include <clio_cte/core/core_client.h>
#include <clio_cte/replication/replication_tasks.h>

#include <string>

namespace clio::cte::replication {

/**
 * Replication client. Inherits the full CTE core client API (so a caller can
 * keep tagging/putting/getting through one object, including replica-targeted
 * puts/gets via Context::replica_) and adds the module's policy verbs:
 * ReplicateBlob (one blob → one replica) and FlushTag (a whole tag's
 * qualifying blobs → one replica).
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;

  /**
   * @param replication_pool_id Pool ID of the replication chimod
   * @param core_pool_id Pool ID of the CTE core chimod the module sits over
   */
  Client(const clio::run::PoolId &replication_pool_id,
         const clio::run::PoolId &core_pool_id)
      : replication_pool_id_(replication_pool_id) {
    clio::cte::core::Client::Init(core_pool_id);
  }

#if CTP_IS_HOST
  /** Create/initialize the replication container over a CTE core pool. */
  clio::run::Future<CreateTask> AsyncCreateReplication(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &custom_pool_id,
      const ReplicationConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        ReplicationConfig::chimod_lib_name, pool_name, custom_pool_id, this,
        params);
    auto fut = ipc->Send(task);
    replication_pool_id_ = custom_pool_id;
    return fut;
  }
#endif  // CTP_IS_HOST

  /** Bring replica `replica` of one blob up to date with the primary. */
  clio::run::Future<ReplicateBlobTask> AsyncReplicateBlob(
      const TagId &tag_id, const std::string &blob_name,
      int replica, const Context &context = Context(),
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReplicateBlobTask>(
        clio::run::CreateTaskId(), replication_pool_id_, pool_query, tag_id,
        blob_name, replica, context);
    return ipc->Send(task);
  }

  // NOTE (issue #886 interposition): there are deliberately NO cached
  // put/get wrappers here. The pool speaks the CTE core's own task
  // interface — point a clio::cte::core::Client at kReplicationPoolId (or
  // set CLIO_CTE_POOL=561.0 before CLIO_CTE_CLIENT_INIT) and plain
  // PutBlob/GetBlob get write-through durability and replica-served reads.

  /** ReplicateBlob every blob in the tag with score >= min_score. */
  clio::run::Future<FlushTagTask> AsyncFlushTag(
      const TagId &tag_id, int replica, float min_score = 0.0f,
      const Context &context = Context(),
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<FlushTagTask>(
        clio::run::CreateTaskId(), replication_pool_id_, pool_query, tag_id,
        replica, min_score, context);
    return ipc->Send(task);
  }

  clio::run::PoolId replication_pool_id_;
};

}  // namespace clio::cte::replication

#endif  // CLIO_CTE_REPLICATION_REPLICATION_CLIENT_H_
