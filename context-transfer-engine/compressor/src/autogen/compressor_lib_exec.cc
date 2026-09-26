/**
 * Container virtual API dispatch for the compressor ChiMod (Run / Save /
 * Load / NewTask / Aggregate), switch-case over Method ids. Hand-maintained
 * to match clio_mod.yaml + compressor_methods.h (same shape as the
 * replication chimod's autogen/replication_lib_exec.cc).
 *
 * INTERPOSITION (issue #886): this pool speaks the CTE core's task
 * interface. The X-macro lists the methods this module implements —
 * including the core's kPutBlob/kGetBlob/kGetBlobSize/kMultiPutBlob ids
 * with the core's task structs — and every DEFAULT case delegates to the
 * next pool via the CoreInterposer base, so any core method this module
 * does not override behaves exactly as if the task had been addressed to
 * that pool (including wire serialization).
 */
#include "clio_cte/compressor/compressor_runtime.h"
#include "clio_cte/compressor/autogen/compressor_methods.h"
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>

namespace clio::cte::compressor {

// One case body per (function, method); declared once to keep the dispatch
// functions in sync.
#define CLIO_COMP_FOR_EACH_METHOD(X)               \
  X(kCreate, CreateTask, Create)                   \
  X(kDestroy, DestroyTask, Destroy)                \
  X(kMonitor, MonitorTask, Monitor)                \
  X(kDynamicSchedule, DynamicScheduleTask, DynamicSchedule) \
  X(kCompress, CompressTask, Compress)             \
  X(kDecompress, DecompressTask, Decompress)       \
  X(kPutBlob, clio::cte::core::PutBlobTask, PutBlob) \
  X(kGetBlob, clio::cte::core::GetBlobTask, GetBlob) \
  X(kGetBlobSize, clio::cte::core::GetBlobSizeTask, GetBlobSize) \
  X(kMultiPutBlob, clio::cte::core::MultiPutBlobTask, MultiPutBlob) \
  X(kPodPutBlob, clio::cte::core::PodPutBlobTask, CompressPodPutBlob) \
  X(kPodGetBlob, clio::cte::core::PodGetBlobTask, DecompressPodGetBlob) \
  X(kPodMultiPutBlob, clio::cte::core::PodMultiPutBlobTask, CompressPodMultiPutBlob) \
  X(kPodMultiGetBlob, clio::cte::core::PodMultiGetBlobTask, DecompressPodMultiGetBlob)

void Runtime::Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
                   clio::run::u32 container_id) {
  clio::run::Container::Init(pool_id, pool_name, container_id);
  client_ = Client(pool_id);
  DefineModel(Method::kMaxMethodId);
  SetMethodNames(Method::GetMethodNames());
}

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
    CLIO_COMP_FOR_EACH_METHOD(X)
#undef X
    default:
      // Interposition escape: execute any other (core) method on the next
      // pool's container, verbatim.
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
    CLIO_COMP_FOR_EACH_METHOD(X)
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
    CLIO_COMP_FOR_EACH_METHOD(X)
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
    CLIO_COMP_FOR_EACH_METHOD(X)
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
    CLIO_COMP_FOR_EACH_METHOD(X)
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
    CLIO_COMP_FOR_EACH_METHOD(X)
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
    CLIO_COMP_FOR_EACH_METHOD(X)
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
    CLIO_COMP_FOR_EACH_METHOD(X)
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

#undef CLIO_COMP_FOR_EACH_METHOD

}  // namespace clio::cte::compressor
