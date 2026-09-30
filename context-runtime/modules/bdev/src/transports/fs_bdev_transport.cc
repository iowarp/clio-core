/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <cerrno>
#include <chrono>
#include <cstring>
#include <clio_runtime/bdev/transports/fs_bdev_transport.h>
#include <clio_ctp/introspect/system_info.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/manager.h>
#include <clio_runtime/worker.h>
#include <clio_runtime/work_orchestrator.h>
#include <fcntl.h>
#ifndef _WIN32
#include <unistd.h>  // close (ReserveFileSpace)
#endif

namespace clio::run::bdev {

namespace {

// Open `file_path` through the best AsyncIO backend that actually works at
// runtime. AsyncIoFactory's kDefault selects the preferred backend at *compile*
// time (NIXL > io_uring > libaio > POSIX), but a backend can be compiled in yet
// be unavailable at runtime: under a container's default seccomp profile
// io_uring_queue_init() fails, returning a negative errno as its return value
// while leaving the global errno untouched, so IoUringAsyncIO::Open() reports
// failure (errno=0) even though the file itself opened fine. When the preferred
// backend cannot open the file, transparently fall back to POSIX AIO, which
// works in restricted environments (containers/CI). Returns an opened AsyncIO,
// or nullptr only if even POSIX AIO cannot open the file (a genuine error).
std::unique_ptr<ctp::AsyncIO> OpenBackingFile(clio::run::u32 io_depth,
                                              const std::string &file_path) {
  auto io = ctp::AsyncIoFactory::Get(io_depth);
  if (io && io->Open(file_path, O_RDWR | O_CREAT, 0644)) {
    return io;
  }
  // Preferred backend is unusable in this environment. IoUringAsyncIO::Open()
  // closes any fds it opened before returning false, so it is safe to discard
  // it and retry with POSIX AIO.
  io = ctp::AsyncIoFactory::Get(io_depth, ctp::AsyncIoBackend::kPosixAio);
  if (io && io->Open(file_path, O_RDWR | O_CREAT, 0644)) {
    return io;
  }
  return nullptr;
}

/**
 * Reserve real disk blocks for [from, to) of the backing file. ftruncate
 * alone makes a SPARSE extent: it never fails for lack of space, so a full
 * disk showed up later as failed writes (EIO to the application) instead
 * of as a full device at allocation time.
 * @param file_path backing file
 * @param from start of the new extent
 * @param to end of the new extent
 * @return 0, or the errno (ENOSPC when the disk cannot hold it). A
 *         filesystem without fallocate support keeps the sparse extent (0).
 */
int ReserveFileSpace(const std::string &file_path, clio::run::u64 from,
                     clio::run::u64 to) {
#ifdef __linux__
  if (to <= from) return 0;
  const int fd = open(file_path.c_str(), O_RDWR);
  if (fd < 0) return errno;
  const int rc = posix_fallocate(fd, static_cast<off_t>(from),
                                 static_cast<off_t>(to - from));
  close(fd);
  if (rc == EOPNOTSUPP || rc == EINVAL) return 0;
  return rc;
#else
  (void)file_path;
  (void)from;
  (void)to;
  return 0;
#endif
}

}  // namespace

bool WorkerIOContext::Init(const std::string &file_path, clio::run::u32 io_depth,
                           clio::run::u32 worker_id) {
  if (is_initialized_) return true;

  async_io_ = OpenBackingFile(io_depth, file_path);
  if (!async_io_) {
    HLOG(kError, "Worker {} failed to open file {} ({})", worker_id,
         file_path, std::strerror(errno));
    return false;
  }

  is_initialized_ = true;
  return true;
}

void WorkerIOContext::Cleanup() {
  if (is_initialized_) {
    if (async_io_) {
      async_io_->Close();
      async_io_.reset();
    }
    is_initialized_ = false;
  }
}

bool FsBdevTransport::Init(const CreateParams& params,
                           const std::string& pool_name, Runtime* runtime) {
  // The pool name doubles as the backing file path.
  file_path_ = pool_name;
  io_depth_ = params.io_depth_;

  auto setup_io = OpenBackingFile(io_depth_, file_path_);
  if (!setup_io) {
    // errno from the last open attempt. Without it this message says only
    // "it did not work", and a permission problem on a leftover file is
    // indistinguishable from a missing io_uring or a full disk -- which is
    // exactly the ambiguity that made a stale scratch file look like a broken
    // driver.
    HLOG(kError, "Failed to open bdev file: {} ({})", file_path_,
         strerror(errno));
    return false;
  }

  off_t current_size = setup_io->GetFileSize();
  if (current_size < 0) {
    HLOG(kError, "Failed to get file size for: {}", file_path_);
    setup_io->Close();
    return false;
  }
  clio::run::u64 existing = static_cast<clio::run::u64>(current_size);

  // Device capacity (what the allocator hands out): the configured size when
  // one is given, else the existing file's size, else a 1GB default.
  //
  // #858 semantic change: a configured capacity is no longer clamped to a
  // smaller existing file. With lazy growth, "backing file smaller than
  // capacity" is the NORMAL state after any restart (the file only ever grew
  // as far as allocations reached), so the old min(existing, configured)
  // clamp would silently shrink the device to its previously-touched prefix
  // on every restart. The existing file is instead treated as the
  // already-backed prefix, and growth resumes from there.
  clio::run::u64 file_size;
  if (params.total_size_ > 0) {
    file_size = params.total_size_;
  } else if (existing > 0) {
    file_size = existing;
  } else {
    file_size = 1ULL << 30;
  }

  // #858: growth granularity for the lazy backing-file extension
  // (EnsureFileBacked). 0 disables stepping — each allocation extends the
  // file exactly as far as it needs.
  growth_unit_ = params.growth_unit_;

  if (existing == 0) {
    // Fresh (empty) backing file: truncate only the FIRST growth unit into
    // existence. The rest of the file materializes in EnsureFileBacked as
    // allocations cross the backed frontier. This keeps a 500GB
    // capacity_limit from claiming 500GB at compose time on filesystems that
    // allocate eagerly on a size set (NTFS SetEndOfFile; any FS without
    // sparse files) and gives ENOSPC a chance to surface at allocation time
    // instead of mid-write.
    clio::run::u64 initial = file_size;
    if (growth_unit_ > 0 && growth_unit_ < initial) {
      initial = growth_unit_;
    }
    if (!setup_io->Truncate(static_cast<size_t>(initial))) {
      HLOG(kError, "Failed to truncate file");
      setup_io->Close();
      return false;
    }
    const int rrc = ReserveFileSpace(file_path_, 0, initial);
    if (rrc != 0) {
      HLOG(kError, "Cannot reserve {} bytes for bdev file {} ({})", initial,
           file_path_, std::strerror(rrc));
      setup_io->Truncate(0);
      setup_io->Close();
      return false;
    }
    file_backed_bytes_.store(initial, std::memory_order_relaxed);
  } else {
    // Existing file: its current extent is the already-backed prefix (capped
    // at the device capacity — a larger file than the device just has spare
    // tail the allocator will never address).
    file_backed_bytes_.store(existing < file_size ? existing : file_size,
                             std::memory_order_relaxed);
  }

  setup_io->Close();

  if (!InitializeWorkerIOContexts()) {
    // A worker that cannot open the file fails every I/O it is handed: refuse
    // the device rather than create one that errors intermittently.
    HLOG(kError, "Failed to open {} on every worker; refusing the device",
         file_path_);
    CleanupWorkerIOContexts();  // release the descriptors that did open
    return false;
  }

  clio::run::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;
  allocator_.Init(num_workers, file_size, params.alignment_);

  return OpenAllocLog(params);
}

bool FsBdevTransport::OpenAllocLog(const CreateParams& params) {
  // An explicit alloc_log always recovers. The default log recovers on every
  // recovering start (a plain `clio_run start`); after `start --fresh` no
  // metadata references the old bytes, so their allocations are garbage and
  // the log starts empty.
  const bool explicit_path = !params.alloc_log_path_.empty();
  const std::string path =
      explicit_path ? params.alloc_log_path_ : file_path_ + ".alloc_log";
  auto *manager = CLIO_RUNTIME_MANAGER;
  const bool recover =
      explicit_path || (manager != nullptr && manager->is_restart_);
  if (!recover) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  if (!alloc_log_.Open(path, recover)) {
    HLOG(kError, "Cannot open allocator log {} for bdev {} ({})", path,
         file_path_, strerror(errno));
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
    HLOG(kInfo, "bdev {}: recovered {} live blocks from {}; {} bytes free",
         file_path_, live.size(), path, allocator_.GetRemainingSize());
  }
  // Start from a compact log so it tracks the live set, not history.
  alloc_log_.Compact();
  sync_stop_ = false;
  sync_thread_ = std::thread(&FsBdevTransport::AllocLogSyncLoop, this);
  return true;
}

bool FsBdevTransport::Sync() {
  // Data first, then the allocator state that references it: a crash in
  // between leaves synced bytes in blocks the log may not show yet (the
  // CTE's own WAL still does), never a logged block whose bytes are lost.
  const int fd = ::open(file_path_.c_str(), O_RDONLY);
  if (fd < 0) {
    HLOG(kError, "bdev Sync: cannot open {} ({})", file_path_,
         strerror(errno));
    return false;
  }
  const int rc = ::fdatasync(fd);
  const int err = errno;
  ::close(fd);
  if (rc != 0) {
    HLOG(kError, "bdev Sync: fdatasync {} failed ({})", file_path_,
         strerror(err));
    return false;
  }
  FlushAllocLog();
  return true;
}

void FsBdevTransport::AllocLogSyncLoop() {
  std::unique_lock<std::mutex> lock(sync_mu_);
  while (!sync_stop_) {
    sync_cv_.wait_for(lock, std::chrono::milliseconds(kAllocLogSyncPeriodMs),
                      [this] { return sync_stop_; });
    lock.unlock();
    FlushAllocLog();
    lock.lock();
  }
}

void FsBdevTransport::StopAllocLogSync() {
  {
    std::lock_guard<std::mutex> lock(sync_mu_);
    sync_stop_ = true;
  }
  sync_cv_.notify_all();
  if (sync_thread_.joinable()) {
    sync_thread_.join();
  }
}

void FsBdevTransport::LogBlocks(const std::vector<Block>& blocks,
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

void FsBdevTransport::FlushAllocLog() {
  if (!has_alloc_log_) return;
  alloc_log_.Flush();
  // Compact once history dominates: the log is then bounded by ~2x the
  // live-block count instead of growing with every allocate/free.
  constexpr clio::run::u64 kMinCompactRecords = 4096;
  const clio::run::u64 live = alloc_log_.live_block_count();
  if (alloc_log_.records_on_disk() > std::max(kMinCompactRecords, 2 * live)) {
    alloc_log_.Compact();
  }
}

void FsBdevTransport::Destroy() {
  CleanupWorkerIOContexts();
  StopAllocLogSync();
  if (has_alloc_log_) {
    alloc_log_.Close();
  }
}

bool FsBdevTransport::AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) {
  if (!allocator_.AllocateBlocks(size, worker_id, blocks)) {
    return false;
  }
  // #858: back every returned block with real file before the caller can
  // touch it. Reads of allocated-but-unwritten blocks must land inside the
  // file (and return zeros) exactly as they did when the whole capacity was
  // truncated up front — a block past EOF would short-read instead.
  clio::run::u64 end = 0;
  for (const Block &b : blocks) {
    clio::run::u64 e = b.offset_ + b.size_;
    if (e > end) end = e;
  }
  if (!EnsureFileBacked(end)) {
    // The disk cannot actually provide the space the allocator promised.
    // Fail the allocation cleanly (the caller sees the same "no space"
    // result as an exhausted allocator) instead of handing out blocks whose
    // writes would EIO later.
    allocator_.FreeBlocks(worker_id, blocks);
    blocks.clear();
    return false;
  }
  // Logged BEFORE the caller can write into or reference the blocks.
  LogBlocks(blocks, /*is_free=*/false);
  return true;
}

bool FsBdevTransport::GrowBackingFile(clio::run::u64 backed,
                                      clio::run::u64 target) {
  auto io = OpenBackingFile(io_depth_, file_path_);
  if (!io) {
    HLOG(kError, "EnsureFileBacked: failed to open {} to grow it", file_path_);
    return false;
  }
  bool ok = io->Truncate(static_cast<size_t>(target));
  if (ok) {
    const int rrc = ReserveFileSpace(file_path_, backed, target);
    if (rrc != 0) {
      // Out of disk: undo the sparse growth so the file never claims space
      // it does not have; the allocation fails as a full device.
      HLOG(kWarning, "EnsureFileBacked: no disk space to grow {} to {} "
           "bytes ({})", file_path_, target, std::strerror(rrc));
      io->Truncate(static_cast<size_t>(backed));
      ok = false;
    }
  }
  io->Close();
  return ok;
}

bool FsBdevTransport::EnsureFileBacked(clio::run::u64 end_offset) {
  // Fast path: recycled or low blocks are already inside the backed prefix.
  if (end_offset <= file_backed_bytes_.load(std::memory_order_acquire)) {
    return true;
  }
  std::lock_guard<std::mutex> lock(grow_mu_);
  clio::run::u64 backed = file_backed_bytes_.load(std::memory_order_relaxed);
  if (end_offset <= backed) {
    return true;  // another thread grew past us while we waited
  }
  // Grow in growth-unit steps (capped at capacity) so a bump-allocation
  // stream costs one truncate per unit, not one per allocation.
  clio::run::u64 target = end_offset;
  if (growth_unit_ > 0) {
    target = ((end_offset + growth_unit_ - 1) / growth_unit_) * growth_unit_;
  }
  clio::run::u64 capacity = allocator_.GetCapacity();
  if (capacity > 0 && target > capacity) {
    target = capacity;
  }
  const clio::run::u64 now_ns = static_cast<clio::run::u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
  if (grow_fail_end_ != 0 && end_offset >= grow_fail_end_ &&
      now_ns - grow_fail_ns_ < kGrowRetryNs) {
    return false;  // the disk was just found full: fail fast
  }
  // A whole growth unit first; if the disk cannot hold it, just what this
  // allocation needs (the unit can be gigabytes more than what is left).
  bool ok = GrowBackingFile(backed, target);
  if (!ok && target > end_offset) {
    target = end_offset;
    ok = GrowBackingFile(backed, target);
  }
  if (!ok) {
    if (grow_fail_end_ == 0 || end_offset < grow_fail_end_ ||
        now_ns - grow_fail_ns_ >= kGrowRetryNs) {
      grow_fail_end_ = end_offset;
    }
    grow_fail_ns_ = now_ns;
    HLOG(kError,
         "EnsureFileBacked: failed to grow {} from {} to {} bytes — treating "
         "as out of space",
         file_path_, backed, end_offset);
    return false;
  }
  grow_fail_end_ = 0;
  HLOG(kDebug, "EnsureFileBacked: grew {} from {} to {} bytes", file_path_,
       backed, target);
  file_backed_bytes_.store(target, std::memory_order_release);
  return true;
}

void FsBdevTransport::FreeBlocks(int worker_id, const std::vector<Block>& blocks) {
  // Logged BEFORE reuse is possible: a crash between the two leaves the
  // blocks allocated in the log (a leak), never free while still in use.
  LogBlocks(blocks, /*is_free=*/true);
  allocator_.FreeBlocks(worker_id, blocks);
}

bool FsBdevTransport::InitializeWorkerIOContexts() {
  clio::run::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;

  // Reserve slots for workers that do not exist yet. The worker pool GROWS at
  // runtime: WorkOrchestrator::SpawnAdditionalWorker() hands out ids past the
  // startup count (the #781/#785 lane-rescue path, and any scheduler that
  // sizes pools on demand). This vector used to be sized once at pool
  // creation, so GetWorkerIOContext(id) returned nullptr for every such worker
  // and its I/O silently failed — writes logged "WriteToFile called with
  // invalid I/O context" and reads returned 0 bytes, which the CTE read path
  // treats as a hole rather than an error. Sizing to the same elastic headroom
  // the lane table reserves keeps a context available for any id the
  // orchestrator can produce.
  //
  // Only the startup contexts are opened eagerly; the reserved tail is opened
  // lazily by GetWorkerIOContext the first time a spawned worker uses it, so
  // this costs a vector of empty structs, not file descriptors.
  const size_t reserved =
      num_workers + clio::run::WorkOrchestrator::ElasticHeadroom();
  io_contexts_.resize(reserved);
  bool success = true;
  for (size_t i = 0; i < num_workers; ++i) {
    if (!io_contexts_[i].Init(file_path_, io_depth_, static_cast<clio::run::u32>(i))) {
      success = false;
    }
  }
  return success;
}

void FsBdevTransport::CleanupWorkerIOContexts() {
  for (auto &ctx : io_contexts_) {
    ctx.Cleanup();
  }
}

WorkerIOContext *FsBdevTransport::GetWorkerIOContext(size_t worker_id) {
  if (worker_id >= io_contexts_.size()) {
    return nullptr;
  }
  WorkerIOContext *ctx = &io_contexts_[worker_id];
  if (!ctx->is_initialized_) {
    if (!ctx->Init(file_path_, io_depth_, static_cast<clio::run::u32>(worker_id))) {
      return nullptr;
    }
  }
  return ctx;
}

clio::run::TaskResume FsBdevTransport::WriteBlocks(ctp::ipc::FullPtr<WriteTask> task) {
  CLIO_TASK_BODY_BEGIN
  clio::run::Worker *worker = CLIO_CUR_WORKER;
  size_t worker_id = worker ? worker->GetId() : 0;
  WorkerIOContext *io_ctx = GetWorkerIOContext(worker_id);

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
    ctp::DeviceAwareMemcpy(staging.data(), data_ptr.ptr_, task->length_);
  }

  clio::run::u64 total_bytes_written = 0;
  clio::run::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    clio::run::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    clio::run::u64 block_write_size = std::min(remaining, block.size_);

    void *block_data = data_on_device
                           ? static_cast<void *>(staging.data() + data_offset)
                           : static_cast<void *>(data_ptr.ptr_ + data_offset);

    if (io_ctx == nullptr || !io_ctx->is_initialized_ || !io_ctx->async_io_) {
      HLOG(kError, "WriteToFile called with invalid I/O context");
      task->return_code_ = 1;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    ctp::IoToken token = io_ctx->async_io_->Write(
        block_data, static_cast<size_t>(block_write_size),
        static_cast<off_t>(block.offset_));
    if (token == ctp::kInvalidIoToken) {
      task->return_code_ = 2;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    ctp::IoResult result;
    while (!io_ctx->async_io_->IsComplete(token, result)) {
      CLIO_CO_AWAIT(clio::run::yield(10.0));
    }

    if (result.error_code != 0) {
      task->return_code_ = 4;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    clio::run::u64 actual_bytes = std::min(
        static_cast<clio::run::u64>(result.bytes_transferred), block_write_size);
    total_bytes_written += actual_bytes;
    data_offset += actual_bytes;
  }

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume FsBdevTransport::ReadBlocks(ctp::ipc::FullPtr<ReadTask> task) {
  CLIO_TASK_BODY_BEGIN
  clio::run::Worker *worker = CLIO_CUR_WORKER;
  size_t worker_id = worker ? worker->GetId() : 0;
  WorkerIOContext *io_ctx = GetWorkerIOContext(worker_id);

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
  }

  clio::run::u64 total_bytes_read = 0;
  clio::run::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    clio::run::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    clio::run::u64 block_read_size = std::min(remaining, block.size_);

    void *block_data = data_on_device
                           ? static_cast<void *>(staging.data() + data_offset)
                           : static_cast<void *>(data_ptr.ptr_ + data_offset);

    if (io_ctx == nullptr || !io_ctx->is_initialized_ || !io_ctx->async_io_) {
      task->return_code_ = 1;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    ctp::IoToken token = io_ctx->async_io_->Read(
        block_data, static_cast<size_t>(block_read_size),
        static_cast<off_t>(block.offset_));
    if (token == ctp::kInvalidIoToken) {
      task->return_code_ = 2;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    ctp::IoResult result;
    while (!io_ctx->async_io_->IsComplete(token, result)) {
      CLIO_CO_AWAIT(clio::run::yield(10.0));
    }

    if (result.error_code != 0) {
      task->return_code_ = 4;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    clio::run::u64 actual_bytes = std::min(
        static_cast<clio::run::u64>(result.bytes_transferred), block_read_size);
    total_bytes_read += actual_bytes;
    data_offset += actual_bytes;
  }

  if (data_on_device && total_bytes_read > 0) {
    ctp::DeviceAwareMemcpy(data_ptr.ptr_, staging.data(), total_bytes_read);
  }

  task->return_code_ = 0;
  task->bytes_read_ = total_bytes_read;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

} // namespace clio::run::bdev
