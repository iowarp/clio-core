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

/**
 * Container virtual API dispatch for the indexer ChiMod (Run / Save / Load /
 * NewTask / Aggregate), switch-case over Method ids. Hand-maintained to
 * match clio_mod.yaml + indexer_methods.h (mirrors the cache chimod's
 * autogen/cache_lib_exec.cc).
 *
 * INTERPOSITION (issues #886/#905): this pool speaks the CTE core's task
 * interface. The X-macro lists the methods this module implements — the
 * core's mutating data verbs plus kSemanticSearch, with the core's task
 * structs — and every DEFAULT case delegates to the core container, so any
 * core method this module does not override behaves exactly as if the task
 * had been addressed to the core pool (including wire serialization:
 * Save/Load/NewTask for forwarded ids are the core's).
 */
#include "clio_cte/indexer/indexer_runtime.h"
#include "clio_cte/indexer/autogen/indexer_methods.h"
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>

namespace clio::cte::indexer {

// One case body per (function, method); declared once to keep the dispatch
// functions in sync.
#define CLIO_INDEXER_FOR_EACH_METHOD(X)                                   \
  X(kCreate, CreateTask, Create)                                          \
  X(kDestroy, DestroyTask, Destroy)                                       \
  X(kMonitor, MonitorTask, Monitor)                                       \
  X(kPutBlob, clio::cte::core::PutBlobTask, PutBlob)                      \
  X(kDelBlob, clio::cte::core::DelBlobTask, DelBlob)                      \
  X(kDelTag, clio::cte::core::DelTagTask, DelTag)                         \
  X(kSemanticSearch, clio::cte::core::SemanticSearchTask, SemanticSearch) \
  X(kTruncateBlob, clio::cte::core::TruncateBlobTask, TruncateBlob)       \
  X(kRenameTag, clio::cte::core::RenameTagTask, RenameTag)                \
  X(kMultiPutBlob, clio::cte::core::MultiPutBlobTask, MultiPutBlob)       \
  X(kIndexSweep, IndexSweepTask, IndexSweep)                              \
  X(kReindexScan, ReindexScanTask, ReindexScan)

void Runtime::Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
                   clio::run::u32 container_id) {
  clio::run::Container::Init(pool_id, pool_name, container_id);
  DefineModel(Method::kMaxMethodId);
  SetMethodNames(Method::GetMethodNames());
}

void Runtime::Restart(const clio::run::PoolId &pool_id, const std::string &pool_name,
                      clio::run::u32 container_id) {
  // Same protocol as the core: flag the warm start BEFORE Init/Create so
  // Create() rebuilds the index slice from the storage below.
  is_restart_ = true;
  Init(pool_id, pool_name, container_id);
}

clio::run::u64 Runtime::GetWorkRemaining() const { return 0; }

clio::run::TaskResume Runtime::Run(clio::run::u32 method,
                             clio::run::shared_ptr<clio::run::Task> task_ptr) {
  CLIO_TASK_BODY_BEGIN
  switch (method) {
#define X(MID, TASK, HANDLER)                                       \
    case Method::MID: {                                             \
      auto &typed = task_ptr.template Cast<TASK>();                 \
      CLIO_CO_AWAIT(HANDLER(typed));                                \
      break;                                                        \
    }
    CLIO_INDEXER_FOR_EACH_METHOD(X)
#undef X
    default:
      // Interposition escape: execute any other (core) method on the core
      // container, verbatim.
      CLIO_CO_AWAIT(ForwardToCore(method, task_ptr));
      break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive << *task_ptr.template Cast<TASK>();              \
      break;
    CLIO_INDEXER_FOR_EACH_METHOD(X)
#undef X
    default:
      ForwardSaveTask(method, archive, task_ptr);
      break;
  }
}

void Runtime::LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive >> *task_ptr.template Cast<TASK>();              \
      break;
    CLIO_INDEXER_FOR_EACH_METHOD(X)
#undef X
    default:
      ForwardLoadTask(method, archive, task_ptr);
      break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::AllocLoadTask(
    clio::run::u32 method, clio::run::LoadTaskArchive &archive) {
  clio::run::shared_ptr<clio::run::Task> task_ptr = NewTask(method);
  if (!task_ptr.IsNull()) {
    LoadTask(method, archive, task_ptr);
  }
  return task_ptr;
}

void Runtime::LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                            clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive >> *task_ptr.template Cast<TASK>();              \
      break;
    CLIO_INDEXER_FOR_EACH_METHOD(X)
#undef X
    default:
      ForwardLocalLoadTask(method, archive, task_ptr);
      break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::LocalAllocLoadTask(
    clio::run::u32 method, clio::run::DefaultLoadArchive &archive) {
  clio::run::shared_ptr<clio::run::Task> task_ptr = NewTask(method);
  if (!task_ptr.IsNull()) {
    LocalLoadTask(method, archive, task_ptr);
  }
  return task_ptr;
}

void Runtime::LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                            clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive << *task_ptr.template Cast<TASK>();              \
      break;
    CLIO_INDEXER_FOR_EACH_METHOD(X)
#undef X
    default:
      ForwardLocalSaveTask(method, archive, task_ptr);
      break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::NewCopyTask(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task_ptr, bool deep) {
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager) {
    return clio::run::shared_ptr<clio::run::Task>();
  }
  switch (method) {
#define X(MID, TASK, HANDLER)                                            \
    case Method::MID: {                                                  \
      auto new_task = ipc_manager->NewTask<TASK>();                      \
      if (!new_task.IsNull()) {                                          \
        new_task->Copy(ctp::ipc::FullPtr<TASK>(                          \
            orig_task_ptr.template Cast<TASK>().get()));                 \
        return new_task.template Cast<clio::run::Task>();                \
      }                                                                  \
      break;                                                             \
    }
    CLIO_INDEXER_FOR_EACH_METHOD(X)
#undef X
    default:
      return ForwardNewCopyTask(method, orig_task_ptr, deep);
  }
  return clio::run::shared_ptr<clio::run::Task>();
}

clio::run::shared_ptr<clio::run::Task> Runtime::NewTask(clio::run::u32 method) {
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager) {
    return clio::run::shared_ptr<clio::run::Task>();
  }
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      return ipc_manager->NewTask<TASK>().template Cast<clio::run::Task>();
    CLIO_INDEXER_FOR_EACH_METHOD(X)
#undef X
    default:
      return ForwardNewTask(method);
  }
}

void Runtime::AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                           const clio::run::shared_ptr<clio::run::Task> &replica_task) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                            \
    case Method::MID:                                                    \
      orig_task.template Cast<TASK>()->AggregateOut(                     \
          ctp::ipc::FullPtr<clio::run::Task>(replica_task.get()));       \
      break;
    CLIO_INDEXER_FOR_EACH_METHOD(X)
#undef X
    default:
      ForwardAggregateOut(method, orig_task, replica_task);
      break;
  }
}

void Runtime::AggregateIn(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
                          const clio::run::shared_ptr<clio::run::Task> &member_task) {
  // No ManyToOne methods of our own; forwarded core methods delegate.
  ForwardAggregateIn(method, agg_task, member_task);
}

#undef CLIO_INDEXER_FOR_EACH_METHOD

}  // namespace clio::cte::indexer
