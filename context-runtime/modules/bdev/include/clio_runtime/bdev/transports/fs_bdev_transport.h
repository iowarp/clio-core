/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#ifndef CLIO_BDEV_FS_TRANSPORT_H_
#define CLIO_BDEV_FS_TRANSPORT_H_

#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <clio_runtime/bdev/transports/block_allocator.h>
#include <clio_ctp/io/async_io_factory.h>

#include <atomic>
#include <mutex>

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
  clio::run::u64 GetRemainingSize() const override { return allocator_.GetRemainingSize(); }

 private:
  StandardBlockAllocator allocator_;
  std::vector<WorkerIOContext> io_contexts_;
  std::string file_path_;
  clio::run::u32 io_depth_;

  // #858: lazy backing-file growth. The file is truncated to at most one
  // growth unit at Init and extended in growth-unit steps as allocations
  // cross the backed frontier — never the full capacity up front (on NTFS
  // SetEndOfFile claims clusters eagerly, so the old full-capacity truncate
  // physically reserved the whole tier at compose).
  clio::run::u64 growth_unit_ = clio::run::u64(1) << 30;
  std::atomic<clio::run::u64> file_backed_bytes_{0};
  std::mutex grow_mu_;

  bool InitializeWorkerIOContexts();
  void CleanupWorkerIOContexts();
  WorkerIOContext* GetWorkerIOContext(size_t worker_id);

  /** Extend the backing file so [0, end_offset) is inside it (growth-unit
   *  granularity, capped at capacity). Returns false if the extension fails
   *  (e.g. the disk is genuinely full). */
  bool EnsureFileBacked(clio::run::u64 end_offset);
};

} // namespace clio::run::bdev

#endif // CLIO_BDEV_FS_TRANSPORT_H_
