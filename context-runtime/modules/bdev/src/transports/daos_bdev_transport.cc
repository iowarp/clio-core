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

#include <clio_runtime/bdev/transports/daos_bdev_transport.h>
#include <clio_runtime/bdev/transports/daos_dfs_file.h>

#include <clio_ctp/io/io_error.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/manager.h>
#include <clio_runtime/work_orchestrator.h>
#include <clio_runtime/worker.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <utility>

namespace clio::run::bdev {

DaosBdevTransport::DaosBdevTransport() = default;

DaosBdevTransport::~DaosBdevTransport() { Destroy(); }

bool DaosBdevTransport::Init(const CreateParams &params,
                             const std::string &pool_name, Runtime *runtime) {
  (void)runtime;
  name_ = pool_name;
  std::string error;
  if (!path_.Parse(pool_name, &error)) {
    HLOG(kError, "DAOS bdev '{}': unusable path: {}", name_, error);
    return false;
  }

  clio::run::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  const size_t num_workers =
      work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;

  daosio::DfsFileSpec spec;
  spec.pool_ = path_.pool_;
  spec.cont_ = path_.cont_;
  spec.dirs_ = path_.dirs_;
  spec.file_ = path_.file_;
  spec.chunk_size_ = path_.chunk_size_;
  spec.oclass_ = path_.oclass_;
  // One event-queue slot per worker id the orchestrator can hand out (same
  // elastic headroom as FsBdevTransport::io_contexts_); queues themselves are
  // created on a worker's first I/O.
  file_ = std::make_unique<daosio::DfsFile>();
  if (!file_->Open(spec,
                   num_workers + clio::run::WorkOrchestrator::ElasticHeadroom(),
                   &error)) {
    HLOG(kError, "DAOS bdev '{}': {}; refusing the device", name_, error);
    file_.reset();
    return false;
  }

  clio::run::u64 capacity = 0;
  if (!ResolveCapacity(params, &capacity)) {
    Destroy();
    return false;
  }

  // Create this worker's event queue now: a process that cannot create one
  // (network contexts exhausted, fabric unreachable) would fail every I/O
  // later, so refuse the device instead.
  clio::run::Worker *worker = CLIO_CUR_WORKER;
  const size_t worker_id = worker ? worker->GetId() : 0;
  daosio::DaosProcess *process = daosio::DaosProcess::GetInstance();
  daos_handle_t eqh{};
  const int eq_rc = process->GetWorkerEq(worker_id, &eqh);
  if (eq_rc != 0) {
    HLOG(kError,
         "DAOS bdev '{}': daos_eq_create failed for worker {} (pool '{}' "
         "container '{}'): {}; refusing the device",
         name_, worker_id, path_.pool_, path_.cont_, daosio::DerText(eq_rc));
    Destroy();
    return false;
  }

  allocator_.Init(num_workers, capacity, params.alignment_);
  if (!OpenAllocLog(params)) {
    Destroy();
    return false;
  }
  HLOG(kInfo,
       "DAOS bdev ready: pool='{}' container='{}' file='{}' chunk_size={} "
       "oclass='{}' prezero={} capacity={} bytes",
       path_.pool_, path_.cont_, path_.FilePath(), path_.chunk_size_,
       path_.oclass_.empty() ? "<container default>" : path_.oclass_.c_str(),
       path_.prezero_ ? 1 : 0, capacity);
  return true;
}

bool DaosBdevTransport::ResolveCapacity(const CreateParams &params,
                                        clio::run::u64 *capacity) {
  if (params.total_size_ > 0) {
    *capacity = params.total_size_;
    return true;
  }
  uint64_t existing = 0;
  std::string error;
  if (!file_->GetSize(&existing, &error)) {
    HLOG(kError, "DAOS bdev '{}': {}; refusing the device", name_, error);
    return false;
  }
  if (existing == 0) {
    HLOG(kError,
         "DAOS bdev '{}': no capacity configured (capacity / capacity_limit) "
         "and file '{}' in pool '{}' container '{}' is empty; refusing the "
         "device",
         name_, path_.FilePath(), path_.pool_, path_.cont_);
    return false;
  }
  *capacity = existing;
  return true;
}

bool DaosBdevTransport::OpenAllocLog(const CreateParams &params) {
  // The log is a LOCAL file (AllocatorLog is file-based); there is no
  // default location for a device that is not itself a local file.
  const std::string path = !params.alloc_log_path_.empty()
                               ? params.alloc_log_path_
                               : path_.alloc_log_;
  if (path.empty()) {
    auto *manager = CLIO_RUNTIME_MANAGER;
    if (manager != nullptr && manager->is_restart_ &&
        !params.SkipsAllocLog()) {
      HLOG(kWarning,
           "DAOS bdev '{}': recovering start without an allocator log -- the "
           "allocator starts empty, so blocks written before the restart may "
           "be handed out again. Give the device a local alloc_log path "
           "(alloc_log=<path> in the daos:// URI) to recover allocations.",
           name_);
    }
    return true;
  }
  // An explicit log always recovers, exactly as in FsBdevTransport.
  if (!alloc_log_.Open(path, /*recover=*/true)) {
    HLOG(kError, "Cannot open allocator log {} for DAOS bdev '{}' ({})", path,
         name_, strerror(errno));
    return false;
  }
  has_alloc_log_ = true;
  const std::vector<LiveBlock> &live = alloc_log_.live(/*group_id=*/0);
  if (!live.empty()) {
    std::vector<std::pair<clio::run::u64, clio::run::u64>> ext;
    ext.reserve(live.size());
    for (const auto &b : live) {
      ext.emplace_back(b.offset, b.size);
    }
    allocator_.InitFromLive(ext);
    HLOG(kInfo, "DAOS bdev '{}': recovered {} live blocks from {}; {} bytes "
         "free", name_, live.size(), path, allocator_.GetRemainingSize());
  }
  // Start from a compact log so it tracks the live set, not history.
  alloc_log_.Compact();
  sync_stop_ = false;
  sync_thread_ = std::thread(&DaosBdevTransport::AllocLogSyncLoop, this);
  return true;
}

void DaosBdevTransport::AllocLogSyncLoop() {
  std::unique_lock<std::mutex> lock(sync_mu_);
  while (!sync_stop_) {
    sync_cv_.wait_for(lock, std::chrono::milliseconds(kAllocLogSyncPeriodMs),
                      [this] { return sync_stop_; });
    lock.unlock();
    FlushAllocLog();
    lock.lock();
  }
}

void DaosBdevTransport::StopAllocLogSync() {
  {
    std::lock_guard<std::mutex> lock(sync_mu_);
    sync_stop_ = true;
  }
  sync_cv_.notify_all();
  if (sync_thread_.joinable()) {
    sync_thread_.join();
  }
}

void DaosBdevTransport::LogBlocks(const std::vector<Block> &blocks,
                                  bool is_free) {
  if (!has_alloc_log_) return;
  for (const Block &b : blocks) {
    if (is_free) {
      alloc_log_.LogFree(/*group_id=*/0, b.offset_, b.size_, b.block_type_);
    } else {
      alloc_log_.LogAlloc(/*group_id=*/0, b.offset_, b.size_, b.block_type_);
    }
  }
  alloc_log_.Append();
}

void DaosBdevTransport::FlushAllocLog() {
  if (!has_alloc_log_) return;
  alloc_log_.SyncAppended();
  // Compact once history dominates (same policy as FsBdevTransport).
  constexpr clio::run::u64 kMinCompactRecords = 4096;
  const clio::run::u64 live = alloc_log_.live_block_count();
  if (alloc_log_.records_on_disk() > std::max(kMinCompactRecords, 2 * live)) {
    alloc_log_.Compact();
  }
}

bool DaosBdevTransport::Sync() {
  FlushAllocLog();
  return true;
}

void DaosBdevTransport::Destroy() {
  StopAllocLogSync();
  if (has_alloc_log_) {
    alloc_log_.Close();
    has_alloc_log_ = false;
  }
  if (file_) {
    std::string warnings;
    file_->Close(&warnings);
    if (!warnings.empty()) {
      HLOG(kWarning, "DAOS bdev '{}' (pool '{}' container '{}'): errors while "
           "closing:{}", name_, path_.pool_, path_.cont_, warnings);
    }
    file_.reset();
  }
}

bool DaosBdevTransport::AllocateBlocks(size_t size, int worker_id,
                                       std::vector<Block> &blocks) {
  if (!allocator_.AllocateBlocks(size, worker_id, blocks)) {
    return false;
  }
  // Logged BEFORE the caller can write into or reference the blocks. Nothing
  // to back: DAOS allocates space when the bytes are written.
  LogBlocks(blocks, /*is_free=*/false);
  return true;
}

void DaosBdevTransport::FreeBlocks(int worker_id,
                                   const std::vector<Block> &blocks) {
  // Logged BEFORE reuse is possible (a crash in between leaks the blocks,
  // never frees ones still in use).
  LogBlocks(blocks, /*is_free=*/true);
  allocator_.FreeBlocks(worker_id, blocks);
}

void DaosBdevTransport::LogIoFailure(const char *op, int raw_rc,
                                     clio::run::u64 offset,
                                     clio::run::u64 size) {
  const clio::run::u64 n =
      io_failures_.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n > 20 && (n & (n - 1)) != 0) {
    return;
  }
  HLOG(kError,
       "DAOS bdev {} failed #{}: {} (pool '{}' container '{}' file '{}' "
       "offset={} size={})",
       op, n, daosio::CompletionText(raw_rc), path_.pool_, path_.cont_,
       path_.FilePath(), offset, size);
}

clio::run::TaskResume DaosBdevTransport::WriteBlocks(
    ctp::ipc::FullPtr<WriteTask> task) {
  CLIO_TASK_BODY_BEGIN
  clio::run::Worker *worker = CLIO_CUR_WORKER;
  const size_t worker_id = worker ? worker->GetId() : 0;

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr =
      ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  const bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
    ctp::DeviceAwareMemcpy(staging.data(), data_ptr.ptr_, task->length_);
  }
  char *data = data_on_device ? staging.data() : data_ptr.ptr_;

  task->io_error_ = static_cast<clio::run::u32>(ctp::IoError::kOk);
  task->bytes_written_ = 0;

  // This worker's event queue (created on its first I/O).
  daosio::DaosProcess *process = daosio::DaosProcess::GetInstance();
  daos_handle_t eqh{};
  const int eq_rc = (file_ && file_->IsOpen())
                        ? process->GetWorkerEq(worker_id, &eqh)
                        : -DER_NO_HDL;
  if (eq_rc != 0) {
    LogIoFailure("write (event queue)", eq_rc, 0, task->length_);
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  clio::run::u64 total_bytes_written = 0;
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    const clio::run::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    const clio::run::u64 size = std::min(remaining, block.size_);
    if (size == 0) continue;

    // Submit with an event on this worker's queue, then yield until a
    // non-blocking poll of the queue reports the event complete.
    daosio::IoRequestPtr req = daosio::NewRequest();
    daosio::SubmitWrite(*file_, eqh, req.get(), data + total_bytes_written,
                        static_cast<size_t>(size), block.offset_);
    while (!req->IsDone()) {
      const int polled = daosio::PollEq(eqh);
      if (polled < 0) {
        daosio::FailLostRequest(req.get(), polled);
        break;
      }
      if (req->IsDone()) break;
      CLIO_CO_AWAIT(clio::run::yield(kPollYieldUs));
    }

    if (req->rc_ != 0) {
      LogIoFailure("write", req->raw_rc_, block.offset_, size);
      task->return_code_ = 4;
      task->io_error_ =
          static_cast<clio::run::u32>(ctp::ClassifyErrno(req->rc_));
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }
    total_bytes_written += size;
  }

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume DaosBdevTransport::ReadBlocks(
    ctp::ipc::FullPtr<ReadTask> task) {
  CLIO_TASK_BODY_BEGIN
  clio::run::Worker *worker = CLIO_CUR_WORKER;
  const size_t worker_id = worker ? worker->GetId() : 0;

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr =
      ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  const bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
  }
  char *data = data_on_device ? staging.data() : data_ptr.ptr_;

  task->io_error_ = static_cast<clio::run::u32>(ctp::IoError::kOk);
  task->bytes_read_ = 0;

  // This worker's event queue (created on its first I/O).
  daosio::DaosProcess *process = daosio::DaosProcess::GetInstance();
  daos_handle_t eqh{};
  const int eq_rc = (file_ && file_->IsOpen())
                        ? process->GetWorkerEq(worker_id, &eqh)
                        : -DER_NO_HDL;
  if (eq_rc != 0) {
    LogIoFailure("read (event queue)", eq_rc, 0, task->length_);
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  clio::run::u64 total_bytes_read = 0;
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    const clio::run::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    const clio::run::u64 size = std::min(remaining, block.size_);
    if (size == 0) continue;
    char *dst = data + total_bytes_read;
    if (path_.prezero_) {
      // Holes inside the file (allocated, never written): whether the DAOS
      // client zero-fills them could not be verified, so do not depend on it.
      std::memset(dst, 0, static_cast<size_t>(size));
    }

    // Submit with an event on this worker's queue, then yield until a
    // non-blocking poll of the queue reports the event complete.
    daosio::IoRequestPtr req = daosio::NewRequest();
    daosio::SubmitRead(*file_, eqh, req.get(), dst, static_cast<size_t>(size),
                       block.offset_);
    while (!req->IsDone()) {
      const int polled = daosio::PollEq(eqh);
      if (polled < 0) {
        daosio::FailLostRequest(req.get(), polled);
        break;
      }
      if (req->IsDone()) break;
      CLIO_CO_AWAIT(clio::run::yield(kPollYieldUs));
    }

    if (req->rc_ != 0) {
      LogIoFailure("read", req->raw_rc_, block.offset_, size);
      task->return_code_ = 4;
      task->io_error_ =
          static_cast<clio::run::u32>(ctp::ClassifyErrno(req->rc_));
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    // A read that ran past the end of the file is short. Inside the device
    // those bytes belong to blocks that were allocated but never written and
    // read as zeros (what the file transport's backed-but-unwritten blocks
    // return); a block outside the capacity keeps its short count.
    clio::run::u64 got =
        std::min(static_cast<clio::run::u64>(req->read_size_), size);
    if (got < size && block.offset_ + size <= allocator_.GetCapacity()) {
      std::memset(dst + got, 0, static_cast<size_t>(size - got));
      got = size;
    }
    total_bytes_read += got;
  }

  if (data_on_device && total_bytes_read > 0) {
    ctp::DeviceAwareMemcpy(data_ptr.ptr_, staging.data(), total_bytes_read);
  }

  task->return_code_ = 0;
  task->bytes_read_ = total_bytes_read;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::run::bdev
