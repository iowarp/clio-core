/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
/**
 * Checkpoint chimod -- the fault handler behind vector.Copy's lazy copies.
 *
 * The core dispatches a GetBlobTask here when a fault-handled tag misses a
 * blob (see Runtime::GetBlobImpl / PutBlobImpl in core_runtime.cc). The
 * task arrives with:
 *
 *   tag_id_                 the COPY tag (the one that faulted)
 *   blob_name_              the missing blob
 *   offset_/size_/data_     the original request (size 0, null data = a
 *                           put-side materialise-only fault)
 *   context_.fault_params_  the SOURCE tag name, copied from the tag's
 *                           registration at fault time
 *
 * Every core operation issued here carries Context::kNoFault -- the
 * handler's own gets and puts must never re-enter the fault path.
 */
#include <clio_cte/checkpoint/checkpoint_runtime.h>

#include <algorithm>
#include <cstring>
#include <string>

namespace clio::cte::checkpoint {

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;  // base forwarding target
  if (!config_.next_pool_id_.IsNull()) {
    next_client_ =
        std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::cte::core::Client *Runtime::GetNextClient() {
  if (!next_client_) {
    next_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  }
  return next_client_.get();
}

/**
 * A core get as a RAW task, never the client wrapper: AsyncGetBlob's
 * zero-IPC SHM fast path synthesizes a client-origin future whose teardown
 * is not valid inside a runtime worker coroutine (free() of an SHM task).
 * The task path is always safe here, and a fault is not a hot path.
 */
clio::run::Future<clio::cte::core::GetBlobTask> Runtime::SendGet(
    const clio::cte::core::TagId &tag_id, const std::string &blob_name,
    clio::run::u64 offset, clio::run::u64 size, clio::run::u32 flags,
    ctp::ipc::ShmPtr<> blob_data, const clio::cte::core::Context &context) {
  const clio::run::PoolId pool = interposer_next_pool_.IsNull()
                                     ? CorePoolId()
                                     : interposer_next_pool_;
  auto *ipc = CLIO_CPU_IPC;
  auto sub = ipc->NewTask<clio::cte::core::GetBlobTask>(
      clio::run::CreateTaskId(), pool, clio::run::PoolQuery::Dynamic(),
      tag_id, blob_name, offset, size, flags, blob_data, context);
  return ipc->Send(sub);
}

clio::run::TaskResume Runtime::GetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string blob_name = task->blob_name_.str();
  const TagId copy_tag = task->tag_id_;
  const clio::run::u64 req_off = task->offset_;
  const clio::run::u64 req_size = task->size_;
  const bool materialize_only = (req_size == 0 && task->blob_data_.IsNull());
  // The source tag name arrives in the POD fault-params array; it is
  // NUL-padded, never necessarily NUL-terminated at capacity.
  std::string src_tag_name(
      task->context_.fault_params_,
      strnlen(task->context_.fault_params_, Context::kFaultParamsSize));

  auto *core = GetNextClient();
  Context nofault = task->context_;
  nofault.op_flags_ |= Context::kNoFault;
  // The materialising put must not inherit read-side directives.
  nofault.create_on_get_ = false;

  if (src_tag_name.empty()) {
    // A fault with no source registered is a registration bug, not a read
    // error the caller can act on: fail loudly.
    HLOG(kError, "checkpoint: fault for '{}' with empty fault_params", blob_name);
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  // 1. Resolve the source tag (idempotent lookup).
  TagId src_tag = TagId::GetNull();
  {
    auto tf = core->AsyncGetOrCreateTag(src_tag_name);
    CLIO_CO_AWAIT(tf);
    if (tf->GetReturnCode() != 0) {
      HLOG(kError, "checkpoint: source tag '{}' resolve failed rc={}",
           src_tag_name, tf->GetReturnCode());
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }
    src_tag = tf->tag_id_;
  }

  // 2. Source blob size. Absent or empty source content = the blob is
  // legitimately NEW under the copy too: fall back to the core's pre-fault
  // behaviour by forwarding the original request with kNoFault (so
  // create_on_get and error semantics stay exactly as they were).
  clio::run::u64 full = 0;
  {
    auto sz = core->AsyncGetBlobSize(src_tag, blob_name,
                                     clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(sz);
    if (sz->GetReturnCode() == 0) {
      full = sz->size_;
    }
  }
  if (full == 0) {
    if (materialize_only) {
      // Put-side fault of a blob with no source content: nothing to
      // materialise; the put proceeds against a fresh blob.
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
    Context fwd = task->context_;
    fwd.op_flags_ |= Context::kNoFault;
    auto gf = SendGet(copy_tag, blob_name, req_off, req_size, task->flags_,
                      task->blob_data_, fwd);
    CLIO_CO_AWAIT(gf);
    task->return_code_ = gf->GetReturnCode();
    CLIO_CO_RETURN;
  }

  // 3. Read the WHOLE source blob and materialise it into the copy tag.
  // Whole-blob, not the requested range: blob existence is what gates the
  // fault, so a partial materialisation would leave silent holes that no
  // later access could detect.
  {
    auto buf = CLIO_IPC->AllocateBuffer(full);
    if (buf.IsNull()) {
      task->return_code_ = 2;
      CLIO_CO_RETURN;
    }
    ctp::ipc::ShmPtr<> buf_shm = buf.shm_.template Cast<void>();
    int rc = 0;
    {
      auto gf = SendGet(src_tag, blob_name, 0, full, /*flags=*/0, buf_shm,
                        nofault);
      CLIO_CO_AWAIT(gf);
      rc = gf->GetReturnCode();
    }
    if (rc == 0) {
      auto pf = core->AsyncPutBlob(copy_tag, blob_name.c_str(), 0, full,
                                   buf_shm, /*score=*/-1.0f, nofault);
      CLIO_CO_AWAIT(pf);
      rc = pf->GetReturnCode();
    }
    CLIO_IPC->FreeBuffer(buf);
    // 4. Serve the originally requested range by re-issuing the get against
    // the now-materialised copy, with kNoFault. NOT a memcpy from the
    // staging buffer: task->blob_data_ can be DEVICE memory (a gpu_vector
    // page fault), and only the core's serve path knows how to fill that.
    if (rc == 0 && !materialize_only) {
      Context fwd = task->context_;
      fwd.op_flags_ |= Context::kNoFault;
      auto gf = SendGet(copy_tag, blob_name, req_off, req_size, task->flags_,
                        task->blob_data_, fwd);
      CLIO_CO_AWAIT(gf);
      rc = gf->GetReturnCode();
    }
    task->return_code_ = rc;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::checkpoint

// ChiMod entry points (alloc/new/name/destroy) for the module manager.
CLIO_TASK_CC(clio::cte::checkpoint::Runtime)
