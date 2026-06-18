/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#ifndef CLIO_BDEV_FS_TRANSPORT_H_
#define CLIO_BDEV_FS_TRANSPORT_H_

#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <clio_runtime/bdev/transports/block_allocator.h>
#include <clio_ctp/io/async_io_factory.h>

namespace clio::run::bdev {

/**
 * Worker-local I/O context for filesystem backend
 */
struct WorkerIOContext {
  bool is_initialized_{false};
  std::unique_ptr<ctp::AsyncIO> async_io_;

  bool Init(const std::string &file_path, chi::u32 io_depth, chi::u32 worker_id);
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

  chi::TaskResume WriteBlocks(ctp::ipc::FullPtr<WriteTask> task, chi::RunContext &ctx) override;
  chi::TaskResume ReadBlocks(ctp::ipc::FullPtr<ReadTask> task, chi::RunContext &ctx) override;

  chi::u64 GetCapacity() const override { return allocator_.GetCapacity(); }
  chi::u64 GetRemainingSize() const override { return allocator_.GetRemainingSize(); }

 private:
  StandardBlockAllocator allocator_;
  std::vector<WorkerIOContext> io_contexts_;
  std::string file_path_;
  chi::u32 io_depth_;

  bool InitializeWorkerIOContexts();
  void CleanupWorkerIOContexts();
  WorkerIOContext* GetWorkerIOContext(size_t worker_id);
};

} // namespace clio::run::bdev

#endif // CLIO_BDEV_FS_TRANSPORT_H_
