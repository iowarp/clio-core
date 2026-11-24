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
 * Container dispatch shared by the policy cache modules (hand-maintained,
 * like the other interposers' autogen/<module>_lib_exec.cc). Include it inside the
 * module's namespace, after declaring `Runtime` (derived from
 * clio::cte::cache_base::CacheBase) and `CreateTask`.
 *
 * The X-list names every verb the cache implements, with the CORE's ids and
 * task structs; every default case delegates to the persistent chain
 * (ForwardCoherent for Run, the base's Forward* helpers otherwise), so any
 * core verb the cache does not override behaves as if sent to that pool.
 */

#define CLIO_POLICY_CACHE_FOR_EACH_METHOD(X)                                  \
  X(clio::cte::core::Method::kCreate, CreateTask, Create)                     \
  X(clio::cte::core::Method::kDestroy, clio::run::admin::DestroyTask,         \
    Destroy)                                                                  \
  X(clio::cte::core::Method::kMonitor, clio::run::admin::MonitorTask,         \
    Monitor)                                                                  \
  X(clio::cte::core::Method::kPutBlob, clio::cte::core::PutBlobTask, PutBlob) \
  X(clio::cte::core::Method::kGetBlob, clio::cte::core::GetBlobTask, GetBlob) \
  X(clio::cte::core::Method::kGetBlobSize, clio::cte::core::GetBlobSizeTask,  \
    GetBlobSize)                                                              \
  X(clio::cte::core::Method::kDelBlob, clio::cte::core::DelBlobTask, DelBlob) \
  X(clio::cte::core::Method::kDelTag, clio::cte::core::DelTagTask, DelTag)    \
  X(clio::cte::core::Method::kTruncateBlob,                                   \
    clio::cte::core::TruncateBlobTask, TruncateBlob)                          \
  X(clio::cte::core::Method::kMultiPutBlob,                                   \
    clio::cte::core::MultiPutBlobTask, MultiPutBlob)                          \
  X(clio::cte::core::Method::kSyncTag, clio::cte::core::SyncTagTask, SyncTag) \
  X(clio::cte::core::Method::kFlushData, clio::cte::core::FlushDataTask,      \
    FlushData)                                                                \
  X(clio::cte::cache_base::kDropTagCopies, clio::cte::core::DelTagTask,       \
    DropTagCopies)

/** Method names: the core's, plus the module verb. */
static const std::vector<std::string> &PolicyCacheMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(clio::cte::cache_base::kCacheMaxMethodId);
    const auto &core = clio::cte::core::Method::GetMethodNames();
    for (size_t i = 0; i < core.size() && i < v.size(); ++i) v[i] = core[i];
    v[clio::cte::cache_base::kDropTagCopies] = "DropTagCopies";
    return v;
  }();
  return names;
}

void Runtime::Init(const clio::run::PoolId &pool_id,
                   const std::string &pool_name,
                   clio::run::u32 container_id) {
  clio::run::Container::Init(pool_id, pool_name, container_id);
  DefineModel(clio::cte::cache_base::kCacheMaxMethodId);
  SetMethodNames(PolicyCacheMethodNames());
}

clio::run::u64 Runtime::GetWorkRemaining() const { return 0; }

clio::run::TaskResume Runtime::Run(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> task_ptr) {
  CLIO_TASK_BODY_BEGIN
  switch (method) {
#define X(MID, TASK, HANDLER)                        \
    case MID: {                                      \
      auto &typed = task_ptr.template Cast<TASK>();  \
      CLIO_CO_AWAIT(HANDLER(typed));                 \
      break;                                         \
    }
    CLIO_POLICY_CACHE_FOR_EACH_METHOD(X)
#undef X
    default:
      CLIO_CO_AWAIT(ForwardCoherent(method, task_ptr));
      break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::SaveTask(clio::run::u32 method,
                       clio::run::SaveTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                        \
    case MID:                                        \
      archive << *task_ptr.template Cast<TASK>();    \
      break;
    CLIO_POLICY_CACHE_FOR_EACH_METHOD(X)
#undef X
    default:
      ForwardSaveTask(method, archive, task_ptr);
      break;
  }
}

void Runtime::LoadTask(clio::run::u32 method,
                       clio::run::LoadTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                        \
    case MID:                                        \
      archive >> *task_ptr.template Cast<TASK>();    \
      break;
    CLIO_POLICY_CACHE_FOR_EACH_METHOD(X)
#undef X
    default:
      ForwardLoadTask(method, archive, task_ptr);
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
#define X(MID, TASK, HANDLER)                        \
    case MID:                                        \
      archive >> *task_ptr.template Cast<TASK>();    \
      break;
    CLIO_POLICY_CACHE_FOR_EACH_METHOD(X)
#undef X
    default:
      ForwardLocalLoadTask(method, archive, task_ptr);
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
#define X(MID, TASK, HANDLER)                        \
    case MID:                                        \
      archive << *task_ptr.template Cast<TASK>();    \
      break;
    CLIO_POLICY_CACHE_FOR_EACH_METHOD(X)
#undef X
    default:
      ForwardLocalSaveTask(method, archive, task_ptr);
      break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::NewCopyTask(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig,
    bool deep) {
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager) return clio::run::shared_ptr<clio::run::Task>();
  switch (method) {
#define X(MID, TASK, HANDLER)                                          \
    case MID: {                                                        \
      auto new_task = ipc_manager->NewTask<TASK>();                    \
      if (!new_task.IsNull()) {                                        \
        new_task->Copy(                                                \
            ctp::ipc::FullPtr<TASK>(orig.template Cast<TASK>().get())); \
        return new_task.template Cast<clio::run::Task>();              \
      }                                                                \
      break;                                                           \
    }
    CLIO_POLICY_CACHE_FOR_EACH_METHOD(X)
#undef X
    default:
      return ForwardNewCopyTask(method, orig, deep);
  }
  return clio::run::shared_ptr<clio::run::Task>();
}

clio::run::shared_ptr<clio::run::Task> Runtime::NewTask(clio::run::u32 method) {
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager) return clio::run::shared_ptr<clio::run::Task>();
  switch (method) {
#define X(MID, TASK, HANDLER) \
    case MID:                 \
      return ipc_manager->NewTask<TASK>().template Cast<clio::run::Task>();
    CLIO_POLICY_CACHE_FOR_EACH_METHOD(X)
#undef X
    default:
      return ForwardNewTask(method);
  }
}

void Runtime::AggregateOut(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
    const clio::run::shared_ptr<clio::run::Task> &replica_task) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                    \
    case MID:                                                    \
      orig_task.template Cast<TASK>()->AggregateOut(             \
          ctp::ipc::FullPtr<clio::run::Task>(replica_task.get())); \
      break;
    CLIO_POLICY_CACHE_FOR_EACH_METHOD(X)
#undef X
    default:
      ForwardAggregateOut(method, orig_task, replica_task);
      break;
  }
}

void Runtime::AggregateIn(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
    const clio::run::shared_ptr<clio::run::Task> &member_task) {
  ForwardAggregateIn(method, agg_task, member_task);
}

#undef CLIO_POLICY_CACHE_FOR_EACH_METHOD
