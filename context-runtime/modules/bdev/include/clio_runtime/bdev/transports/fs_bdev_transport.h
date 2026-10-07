/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#ifndef CLIO_BDEV_FS_TRANSPORT_H_
#define CLIO_BDEV_FS_TRANSPORT_H_

#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <clio_runtime/bdev/transports/block_allocator.h>
#include <clio_runtime/bdev/bdev_alloc_log.h>
#include <clio_ctp/io/async_io_factory.h>
#include <clio_ctp/introspect/system_info.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace clio::run::bdev {

/**
 * Worker-local I/O context for filesystem backend
 */
struct WorkerIOContext {
  bool is_initialized_{false};
  std::unique_ptr<ctp::AsyncIO> async_io_;

  bool Init(const std::string &file_path, clio::run::u32 io_depth, clio::run::u32 worker_id);
  void Cleanup();
};

class FsBdevTransport : public BdevTransport {
 public:
  FsBdevTransport() = default;
  ~FsBdevTransport() override { Destroy(); }

  bool Init(const CreateParams& params, const std::string& pool_name,
            Runtime* runtime) override;
  void Destroy() override;

  bool AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) override;
  void FreeBlocks(int worker_id, const std::vector<Block>& blocks) override;

  clio::run::TaskResume WriteBlocks(ctp::ipc::FullPtr<WriteTask> task) override;
  clio::run::TaskResume ReadBlocks(ctp::ipc::FullPtr<ReadTask> task) override;

  clio::run::u64 GetCapacity() const override { return allocator_.GetCapacity(); }
  /**
   * Bytes the device can still hand out AND back with disk (#1108): the
   * allocator's free space, capped by the free part of the backed prefix
   * plus what the host filesystem can still give the file. A capacity set
   * above the disk's free space otherwise reported gigabytes free while
   * every growth failed, and placement kept choosing a full target.
   * @return remaining bytes
   */
  clio::run::u64 GetRemainingSize() const override;

  void FlushAllocLog() override;
  bool Sync() override;

 private:
  /**
   * Persistent allocator state. Without it the bump allocator restarts at
   * offset 0 and a restart hands out bytes that live data -- placed by ANY
   * client, local or remote -- still occupies.
   */
  AllocatorLog alloc_log_;
  bool has_alloc_log_ = false;

  // The log's fsync runs on its own thread, owned by the transport, so it
  // ends with the pool: a runtime periodic task outlives a destroyed pool
  // and retries forever.
  static constexpr int kAllocLogSyncPeriodMs = 50;
  std::thread sync_thread_;
  std::mutex sync_mu_;
  std::condition_variable sync_cv_;
  bool sync_stop_ = false;

  /** Sync-thread body: FlushAllocLog every kAllocLogSyncPeriodMs until
   *  StopAllocLogSync. */
  void AllocLogSyncLoop();

  /** Stop and join the sync thread (idempotent). */
  void StopAllocLogSync();
  StandardBlockAllocator allocator_;
  std::vector<WorkerIOContext> io_contexts_;
  std::string file_path_;
  clio::run::u32 io_depth_;

  // #1210: the identity of the file Init opened. Later code opens the file
  // again BY PATH (worker contexts, growth, sync); if the path no longer
  // names this file (deleted, replaced), those opens must fail loudly rather
  // than create a new empty file or sync the wrong one.
  ctp::SystemInfo::FileIdentity file_id_;
  bool file_id_valid_ = false;

  /**
   * Whether file_path_ still names the file Init opened.
   * @param what the operation about to reopen it (for the error message)
   * @return true when it does (or when no identity could be recorded)
   */
  bool BackingFileIntact(const char *what) const;

  // #858: lazy backing-file growth. The file is truncated to at most one
  // growth unit at Init and extended in growth-unit steps as allocations
  // cross the backed frontier — never the full capacity up front (on NTFS
  // SetEndOfFile claims clusters eagerly, so the old full-capacity truncate
  // physically reserved the whole tier at compose).
  clio::run::u64 growth_unit_ = clio::run::u64(1) << 30;
  std::atomic<clio::run::u64> file_backed_bytes_{0};
  std::mutex grow_mu_;
  /** Smallest end offset a grow could not reserve disk for (guarded by
   *  grow_mu_; 0 = none), and when (steady ns): requests reaching it fail
   *  fast for kGrowRetryNs instead of retrying the reservation. */
  clio::run::u64 grow_fail_end_ = 0;
  clio::run::u64 grow_fail_ns_ = 0;
  /** Host filesystem free bytes (cached for kHostFreeTtlNs) and when it
   *  was sampled (steady ns); 0 ns = never. */
  mutable std::atomic<clio::run::u64> host_free_{0};
  mutable std::atomic<clio::run::u64> host_free_ns_{0};
  static constexpr clio::run::u64 kHostFreeTtlNs = 1000000000ull;
  /** @return free bytes on the backing file's filesystem (cached). */
  clio::run::u64 HostFreeBytes() const;
  /** How long a failed grow is trusted before disk space is probed again. */
  static constexpr clio::run::u64 kGrowRetryNs = 2000000000ull;
  /**
   * Extend the backing file to `target` bytes and reserve its blocks.
   * @param backed current backed size
   * @param target new size
   * @return true on success (on failure the file is left at `backed`)
   */
  bool GrowBackingFile(clio::run::u64 backed, clio::run::u64 target);

  /**
   * TEST-ONLY fault injection. While a file named `<backing file>.fail`
   * exists, every read, write and sync of this device fails with an I/O
   * error (io_error_ = DeviceFault), as a dead disk would; deleting the file
   * makes the device work again (a disk that comes back, or is replaced in
   * place). Off unless the marker exists. The marker is stat'ed at most once
   * per kFailMarkerPollNs per transport and the answer cached, so the I/O
   * path pays one clock read, not a syscall.
   */
  std::string fail_marker_path_;
  /** Cached "marker present" answer from the last stat. */
  std::atomic<bool> fail_marker_present_{false};
  /** Steady-clock ns of the last marker stat (0 = never). */
  std::atomic<clio::run::u64> fail_marker_checked_ns_{0};
  /** Marker re-check period: 100 ms. */
  static constexpr clio::run::u64 kFailMarkerPollNs = 100000000ull;

  /**
   * Whether the TEST-ONLY fault-injection marker is present (see
   * fail_marker_path_). Logs once on every transition.
   * @return true if every I/O of this device must fail
   */
  bool FaultInjected();

  bool InitializeWorkerIOContexts();
  void CleanupWorkerIOContexts();
  WorkerIOContext* GetWorkerIOContext(size_t worker_id);

  /** Extend the backing file so [0, end_offset) is inside it (growth-unit
   *  granularity, capped at capacity). Returns false if the extension fails
   *  (e.g. the disk is genuinely full). */
  bool EnsureFileBacked(clio::run::u64 end_offset);

  /**
   * Open the allocator-state log and, when recovering, rebuild the allocator
   * from it. Called from Init after allocator_.Init.
   * @param params Create parameters (alloc_log_path_ overrides the default
   *               "<file>.alloc_log" and always recovers)
   * @return false if the log cannot be opened
   */
  bool OpenAllocLog(const CreateParams& params);

  /**
   * Append one record per block to the allocator log and hand them to the OS.
   * @param blocks Blocks just allocated or freed
   * @param is_free true for frees, false for allocations
   */
  void LogBlocks(const std::vector<Block>& blocks, bool is_free);
};

} // namespace clio::run::bdev

#endif // CLIO_BDEV_FS_TRANSPORT_H_
