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

#include <clio_cte/stream/stream_runtime.h>
#include <clio_cte/stream/autogen/stream_methods.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>

namespace clio::cte::stream {

// One case body per (function, method), declared once so the dispatch
// functions stay in sync.
#define CLIO_STREAM_FOR_EACH_METHOD(X)     \
  X(kCreate, CreateTask, Create)           \
  X(kDestroy, DestroyTask, Destroy)        \
  X(kMonitor, MonitorTask, Monitor)        \
  X(kSizeOp, SizeOpTask, SizeOp)           \
  X(kAppend, AppendTask, Append)           \
  X(kFlush, FlushTask, Flush)              \
  X(kSequence, SequenceTask, Sequence)     \
  X(kCollect, CollectTask, Collect)        \
  X(kPlan, PlanTask, Plan)

void Runtime::Init(const clio::run::PoolId &pool_id,
                   const std::string &pool_name,
                   clio::run::u32 container_id) {
  clio::run::Container::Init(pool_id, pool_name, container_id);
  DefineModel(Method::kMaxMethodId);
  SetMethodNames(Method::GetMethodNames());
}

void Runtime::Restart(const clio::run::PoolId &pool_id,
                      const std::string &pool_name,
                      clio::run::u32 container_id) {
  is_restart_ = true;
  Init(pool_id, pool_name, container_id);
}

clio::run::u64 Runtime::GetWorkRemaining() const {
  return pending_count_.load();
}

clio::run::TaskResume Runtime::Run(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> task_ptr) {
  CLIO_TASK_BODY_BEGIN
  switch (method) {
#define X(MID, TASK, HANDLER)                       \
    case Method::MID: {                             \
      auto &typed = task_ptr.template Cast<TASK>(); \
      CLIO_CO_AWAIT(HANDLER(typed));                \
      break;                                        \
    }
    CLIO_STREAM_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::SaveTask(clio::run::u32 method,
                       clio::run::SaveTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                     \
    case Method::MID:                             \
      archive << *task_ptr.template Cast<TASK>(); \
      break;
    CLIO_STREAM_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

void Runtime::LoadTask(clio::run::u32 method,
                       clio::run::LoadTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                     \
    case Method::MID:                             \
      archive >> *task_ptr.template Cast<TASK>(); \
      break;
    CLIO_STREAM_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::AllocLoadTask(
    clio::run::u32 method, clio::run::LoadTaskArchive &archive) {
  clio::run::shared_ptr<clio::run::Task> task_ptr = NewTask(method);
  if (!task_ptr.IsNull()) LoadTask(method, archive, task_ptr);
  return task_ptr;
}

void Runtime::LocalLoadTask(clio::run::u32 method,
                            clio::run::DefaultLoadArchive &archive,
                            clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                     \
    case Method::MID:                             \
      archive >> *task_ptr.template Cast<TASK>(); \
      break;
    CLIO_STREAM_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::LocalAllocLoadTask(
    clio::run::u32 method, clio::run::DefaultLoadArchive &archive) {
  clio::run::shared_ptr<clio::run::Task> task_ptr = NewTask(method);
  if (!task_ptr.IsNull()) LocalLoadTask(method, archive, task_ptr);
  return task_ptr;
}

void Runtime::LocalSaveTask(clio::run::u32 method,
                            clio::run::DefaultSaveArchive &archive,
                            clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                     \
    case Method::MID:                             \
      archive << *task_ptr.template Cast<TASK>(); \
      break;
    CLIO_STREAM_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::NewCopyTask(
    clio::run::u32 method,
    clio::run::shared_ptr<clio::run::Task> &orig_task_ptr, bool deep) {
  (void)deep;
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager) return clio::run::shared_ptr<clio::run::Task>();
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID: {                                        \
      auto new_task = ipc_manager->NewTask<TASK>();            \
      if (!new_task.IsNull()) {                                \
        new_task->Copy(ctp::ipc::FullPtr<TASK>(                \
            orig_task_ptr.template Cast<TASK>().get()));       \
        return new_task.template Cast<clio::run::Task>();      \
      }                                                        \
      break;                                                   \
    }
    CLIO_STREAM_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
  return clio::run::shared_ptr<clio::run::Task>();
}

clio::run::shared_ptr<clio::run::Task> Runtime::NewTask(
    clio::run::u32 method) {
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager) return clio::run::shared_ptr<clio::run::Task>();
  switch (method) {
#define X(MID, TASK, HANDLER) \
    case Method::MID:         \
      return ipc_manager->NewTask<TASK>().template Cast<clio::run::Task>();
    CLIO_STREAM_FOR_EACH_METHOD(X)
#undef X
    default:
      return clio::run::shared_ptr<clio::run::Task>();
  }
}

void Runtime::AggregateOut(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
    const clio::run::shared_ptr<clio::run::Task> &replica_task) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                    \
    case Method::MID:                                            \
      orig_task.template Cast<TASK>()->AggregateOut(             \
          ctp::ipc::FullPtr<clio::run::Task>(replica_task.get())); \
      break;
    CLIO_STREAM_FOR_EACH_METHOD(X)
#undef X
    default:
      orig_task->AggregateOut(
          ctp::ipc::FullPtr<clio::run::Task>(replica_task.get()));
      break;
  }
}

void Runtime::AggregateIn(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
    const clio::run::shared_ptr<clio::run::Task> &member_task) {
  // Only Collect combines member inputs (ManyToOne); every other method keeps
  // the default (the aggregate is a copy of the first member).
  if (method == Method::kCollect) {
    agg_task.template Cast<CollectTask>()->AggregateIn(
        ctp::ipc::FullPtr<clio::run::Task>(member_task.get()));
  }
}

#undef CLIO_STREAM_FOR_EACH_METHOD

}  // namespace clio::cte::stream
