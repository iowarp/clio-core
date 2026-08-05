/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CTE_CORE_CORE_INTERPOSER_H_
#define CLIO_CTE_CORE_CORE_INTERPOSER_H_

/**
 * Base class for chimods that INTERPOSE on the CTE core's task interface
 * (issue #886): pools that speak the core's method ids and task structs,
 * override a handful of data verbs (PutBlob/GetBlob/GetBlobSize/
 * MultiPutBlob), and forward every other core method to the next pool
 * verbatim — so a clio::cte::core::Client pointed at the interposer keeps
 * working unchanged (CLIO_CTE_POOL=<pool>, or a chimod's next_pool_id).
 *
 * The base centralizes the interposition MACHINERY once, shared by the
 * replication and compressor chimods (and any future interposer):
 *   - the forwarding primitives (CorePoolId/CoreContainer/ForwardToCore),
 *   - the dispatch DEFAULTS every Container virtual must delegate for
 *     methods the module does not override (wire Save/Load, NewTask/
 *     NewCopy, aggregation) — call these from the lib_exec default cases,
 *   - together with blob_batch.h, the record/region iteration that makes
 *     vectored and MultiPutBlob traffic uniform across modules.
 *
 * Module POLICY (what to do around a forwarded put/get) stays in the
 * derived runtime — this base has no opinions, only plumbing.
 *
 * Interposer rules the base assumes (see the replication chimod, the
 * reference implementation):
 *   - Compose the interposer AFTER the pool it forwards to.
 *   - Module-specific verbs must be numbered ABOVE the core's method-id
 *     space (>= 100): a collision would shadow a forwarded core verb.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/blob_batch.h>

namespace clio::cte::core {

class CoreInterposer : public clio::run::Container {
 protected:
  /** The pool this interposer forwards to. Derived Create() must set it
   *  (from its config's next_pool_id_); null falls back to the canonical
   *  core pool. */
  clio::run::PoolId interposer_next_pool_;

  clio::run::PoolId CorePoolId() const {
    return !interposer_next_pool_.IsNull() ? interposer_next_pool_
                                           : kCtePoolId;
  }

  /** The local next-pool container for direct forwarding (null before that
   *  pool is composed — compose order requires it BEFORE this module). */
  clio::run::ContainerHold CoreContainer() {
    return CLIO_POOL_MANAGER->GetRealOrStaticContainer(CorePoolId()).get();
  }

 public:
  /**
   * Route interposed tasks EXACTLY as the core routes its own (issue #886
   * multi-node): the core's ScheduleTask hash-routes blob verbs to their
   * owner container (HashBlobToContainer) and tag creation by tag-name
   * hash — the mechanism that makes the blob namespace CLUSTER-GLOBAL.
   * Without this override an interposer pool keeps the base default (the
   * task's own query), every chain-bound op resolves on the submitting
   * node, and cross-node readers simply never see each other's blobs.
   * Delegating (rather than copying the switch) keeps one routing
   * authority; the container counts of the chain pools match the core's
   * (all compose one-per-node), so the returned query is valid for this
   * pool too. Methods the core does not know (module verbs, 100+) fall
   * through its default and keep their query.
   */
  clio::run::PoolQuery ScheduleTask(
      const clio::run::shared_ptr<clio::run::Task> &task) override {
    clio::run::ContainerHold core = CoreContainer();
    if (core != nullptr) {
      return core->ScheduleTask(task);
    }
    return task->pool_query_;
  }

 protected:

  /**
   * Execute a task on the next pool's container verbatim — the generic
   * interposition escape: any core method this module does not override
   * behaves exactly as if the task had been addressed to that pool.
   */
  clio::run::TaskResume ForwardToCore(clio::run::u32 method,
                                      clio::run::shared_ptr<clio::run::Task> task) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
    clio::run::shared_ptr<clio::run::Task> cur_task =
        clio::run::GetCurrentTask();
#endif
    CLIO_TASK_BODY_BEGIN
    {
      clio::run::ContainerHold core = CoreContainer();
      if (core == nullptr) {
        // Compose order guarantees the next pool exists before this module;
        // null means a task raced pool teardown or a miscomposed deployment.
        task->SetReturnCode(1);
        CLIO_CO_RETURN;
      }
      CLIO_CO_AWAIT(core->Run(method, task));
    }
    CLIO_CO_RETURN;
    CLIO_TASK_BODY_END
  }

  // ---- Dispatch defaults ---------------------------------------------------
  // Call these from the lib_exec default cases so forwarded core methods
  // keep the core's wire serialization, allocation and aggregation.

  void ForwardSaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
    clio::run::ContainerHold core = CoreContainer();
    if (core != nullptr) {
      core->SaveTask(method, archive, task_ptr);
    }
  }

  void ForwardLoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
    clio::run::ContainerHold core = CoreContainer();
    if (core != nullptr) {
      core->LoadTask(method, archive, task_ptr);
    }
  }

  void ForwardLocalLoadTask(clio::run::u32 method,
                            clio::run::DefaultLoadArchive &archive,
                            clio::run::shared_ptr<clio::run::Task> &task_ptr) {
    clio::run::ContainerHold core = CoreContainer();
    if (core != nullptr) {
      core->LocalLoadTask(method, archive, task_ptr);
    }
  }

  void ForwardLocalSaveTask(clio::run::u32 method,
                            clio::run::DefaultSaveArchive &archive,
                            clio::run::shared_ptr<clio::run::Task> &task_ptr) {
    clio::run::ContainerHold core = CoreContainer();
    if (core != nullptr) {
      core->LocalSaveTask(method, archive, task_ptr);
    }
  }

  clio::run::shared_ptr<clio::run::Task> ForwardNewTask(clio::run::u32 method) {
    clio::run::ContainerHold core = CoreContainer();
    if (core != nullptr) {
      return core->NewTask(method);
    }
    return clio::run::shared_ptr<clio::run::Task>();
  }

  clio::run::shared_ptr<clio::run::Task> ForwardNewCopyTask(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig,
      bool deep) {
    clio::run::ContainerHold core = CoreContainer();
    if (core != nullptr) {
      return core->NewCopyTask(method, orig, deep);
    }
    return clio::run::shared_ptr<clio::run::Task>();
  }

  void ForwardAggregateOut(clio::run::u32 method,
                           clio::run::shared_ptr<clio::run::Task> &orig_task,
                           const clio::run::shared_ptr<clio::run::Task> &replica_task) {
    clio::run::ContainerHold core = CoreContainer();
    if (core != nullptr) {
      core->AggregateOut(method, orig_task, replica_task);
    } else {
      orig_task->AggregateOut(
          ctp::ipc::FullPtr<clio::run::Task>(replica_task.get()));
    }
  }

  void ForwardAggregateIn(clio::run::u32 method,
                          clio::run::shared_ptr<clio::run::Task> &agg_task,
                          const clio::run::shared_ptr<clio::run::Task> &member_task) {
    clio::run::ContainerHold core = CoreContainer();
    if (core != nullptr) {
      core->AggregateIn(method, agg_task, member_task);
    }
  }
};

}  // namespace clio::cte::core

#endif  // CLIO_CTE_CORE_CORE_INTERPOSER_H_
