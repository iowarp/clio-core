/**
 * Container virtual API dispatch for the filesystem ChiMod (Run / Save / Load /
 * NewTask / Aggregate / DelTask), switch-case over Method ids. Hand-maintained
 * to match clio_mod.yaml + filesystem_methods.h (mirrors compressor's
 * autogen/compressor_lib_exec.cc — no clio_repo.yaml in-tree).
 */
#include "clio_cte/filesystem/filesystem_runtime.h"
#include "clio_cte/filesystem/autogen/filesystem_methods.h"
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>

namespace clio::cte::filesystem {

// One case body per (function, method). FN is the archive/op expression using
// the typed_task; declared once to keep the 12 dispatch functions in sync.
#define CLIO_FS_FOR_EACH_METHOD(X)        \
  X(kCreate, CreateTask, Create)          \
  X(kDestroy, DestroyTask, Destroy)       \
  X(kMonitor, MonitorTask, Monitor)       \
  X(kOpen, OpenTask, Open)                \
  X(kClose, CloseTask, Close)             \
  X(kRead, ReadTask, Read)                \
  X(kWrite, WriteTask, Write)             \
  X(kGetattr, GetattrTask, Getattr)       \
  X(kTruncate, TruncateTask, Truncate)    \
  X(kAppend, AppendTask, Append)          \
  X(kReaddir, ReaddirTask, Readdir)       \
  X(kMkdir, MkdirTask, Mkdir)             \
  X(kRmdir, RmdirTask, Rmdir)             \
  X(kUnlink, UnlinkTask, Unlink)          \
  X(kRename, RenameTask, Rename)          \
  X(kLink, LinkTask, Link)                \
  X(kStatSize, StatSizeTask, StatSize)    \
  X(kAppendSequence, AppendSequenceTask, AppendSequence)    \
  X(kAppendCollect, AppendCollectTask, AppendCollect)       \
  X(kAppendExecution, AppendExecutionTask, AppendExecution) \
  X(kAppendPlan, AppendPlanTask, AppendPlan)

void Runtime::Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
                   clio::run::u32 container_id) {
  clio::run::Container::Init(pool_id, pool_name, container_id);
  DefineModel(Method::kMaxMethodId);
  SetMethodNames(Method::GetMethodNames());
}

clio::run::u64 Runtime::GetWorkRemaining() const { return 0; }

clio::run::TaskResume Runtime::Run(clio::run::u32 method,
                             ctp::ipc::FullPtr<clio::run::Task> task_ptr,
                             clio::run::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  switch (method) {
#define X(MID, TASK, HANDLER)                                       \
    case Method::MID: {                                             \
      ctp::ipc::FullPtr<TASK> typed = task_ptr.template Cast<TASK>(); \
      CLIO_CO_AWAIT(HANDLER(typed, rctx));                          \
      break;                                                        \
    }
    CLIO_FS_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                       ctp::ipc::FullPtr<clio::run::Task> task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive << *task_ptr.template Cast<TASK>().ptr_;         \
      break;
    CLIO_FS_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

void Runtime::LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                       ctp::ipc::FullPtr<clio::run::Task> task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive >> *task_ptr.template Cast<TASK>().ptr_;         \
      break;
    CLIO_FS_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

ctp::ipc::FullPtr<clio::run::Task> Runtime::AllocLoadTask(
    clio::run::u32 method, clio::run::LoadTaskArchive &archive) {
  ctp::ipc::FullPtr<clio::run::Task> task_ptr = NewTask(method);
  if (!task_ptr.IsNull()) {
    LoadTask(method, archive, task_ptr);
  }
  return task_ptr;
}

void Runtime::LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                            ctp::ipc::FullPtr<clio::run::Task> task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive >> *task_ptr.template Cast<TASK>().ptr_;         \
      break;
    CLIO_FS_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

ctp::ipc::FullPtr<clio::run::Task> Runtime::LocalAllocLoadTask(
    clio::run::u32 method, clio::run::DefaultLoadArchive &archive) {
  ctp::ipc::FullPtr<clio::run::Task> task_ptr = NewTask(method);
  if (!task_ptr.IsNull()) {
    LocalLoadTask(method, archive, task_ptr);
  }
  return task_ptr;
}

void Runtime::LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                            ctp::ipc::FullPtr<clio::run::Task> task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      archive << *task_ptr.template Cast<TASK>().ptr_;         \
      break;
    CLIO_FS_FOR_EACH_METHOD(X)
#undef X
    default:
      break;
  }
}

ctp::ipc::FullPtr<clio::run::Task> Runtime::NewCopyTask(
    clio::run::u32 method, ctp::ipc::FullPtr<clio::run::Task> orig_task_ptr, bool deep) {
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager) {
    return ctp::ipc::FullPtr<clio::run::Task>();
  }
  (void)deep;
  switch (method) {
#define X(MID, TASK, HANDLER)                                            \
    case Method::MID: {                                                  \
      auto new_task = ipc_manager->NewTask<TASK>();                      \
      if (!new_task.IsNull()) {                                          \
        new_task->Copy(orig_task_ptr.template Cast<TASK>());             \
        return new_task.template Cast<clio::run::Task>();                      \
      }                                                                  \
      break;                                                             \
    }
    CLIO_FS_FOR_EACH_METHOD(X)
#undef X
    default: {
      auto new_task = ipc_manager->NewTask<clio::run::Task>();
      if (!new_task.IsNull()) {
        new_task->Copy(orig_task_ptr);
        return new_task;
      }
      break;
    }
  }
  return ctp::ipc::FullPtr<clio::run::Task>();
}

ctp::ipc::FullPtr<clio::run::Task> Runtime::NewTask(clio::run::u32 method) {
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager) {
    return ctp::ipc::FullPtr<clio::run::Task>();
  }
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      return ipc_manager->NewTask<TASK>().template Cast<clio::run::Task>();
    CLIO_FS_FOR_EACH_METHOD(X)
#undef X
    default:
      return ctp::ipc::FullPtr<clio::run::Task>();
  }
}

void Runtime::AggregateOut(clio::run::u32 method, ctp::ipc::FullPtr<clio::run::Task> orig_task,
                           const ctp::ipc::FullPtr<clio::run::Task> &replica_task) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                      \
    case Method::MID:                                              \
      orig_task.template Cast<TASK>()->AggregateOut(replica_task); \
      break;
    CLIO_FS_FOR_EACH_METHOD(X)
#undef X
    default:
      orig_task->AggregateOut(replica_task);
      break;
  }
}

void Runtime::AggregateIn(clio::run::u32 method, ctp::ipc::FullPtr<clio::run::Task> agg_task,
                          const ctp::ipc::FullPtr<clio::run::Task> &member_task) {
  // Only AppendCollect combines member inputs (ManyToOne). All other methods
  // keep the default no-op (the aggregate is a copy of the first member).
  if (method == Method::kAppendCollect) {
    agg_task.template Cast<AppendCollectTask>()->AggregateIn(member_task);
  }
}

void Runtime::DelTask(clio::run::u32 method, ctp::ipc::FullPtr<clio::run::Task> task_ptr) {
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager) {
    return;
  }
  switch (method) {
#define X(MID, TASK, HANDLER)                                  \
    case Method::MID:                                          \
      ipc_manager->DelTask(task_ptr.template Cast<TASK>());    \
      break;
    CLIO_FS_FOR_EACH_METHOD(X)
#undef X
    default:
      ipc_manager->DelTask(task_ptr);
      break;
  }
}

#undef CLIO_FS_FOR_EACH_METHOD

}  // namespace clio::cte::filesystem
