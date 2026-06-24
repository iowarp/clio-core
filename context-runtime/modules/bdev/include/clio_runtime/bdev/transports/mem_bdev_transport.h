/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#ifndef CLIO_BDEV_MEM_TRANSPORT_H_
#define CLIO_BDEV_MEM_TRANSPORT_H_

#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <clio_runtime/bdev/transports/block_allocator.h>
#include <mutex>
#include <limits>

namespace clio::run::bdev {

class MemBdevTransport : public BdevTransport {
 public:
  MemBdevTransport() = default;
  ~MemBdevTransport() override { Destroy(); }

  bool Init(const CreateParams& params, const std::string& pool_name,
            Runtime* runtime) override;
  void Destroy() override;

  bool AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) override;
  void FreeBlocks(int worker_id, const std::vector<Block>& blocks) override;

  clio::run::TaskResume WriteBlocks(ctp::ipc::FullPtr<WriteTask> task, clio::run::RunContext &rctx) override;
  clio::run::TaskResume ReadBlocks(ctp::ipc::FullPtr<ReadTask> task, clio::run::RunContext &rctx) override;

  clio::run::u64 GetCapacity() const override { return allocator_.GetCapacity(); }
  clio::run::u64 GetRemainingSize() const override { return allocator_.GetRemainingSize(); }

 private:
  StandardBlockAllocator allocator_;
  clio::run::u64 ram_capacity_{0};

  static constexpr size_t kRamPageSize = 1ULL << 30; // 1 GiB pages

  mutable std::mutex ram_pages_mu_;
  std::vector<std::unique_ptr<char[]>> ram_pages_;

  char* EnsureRamPage(size_t page_idx);
  char* GetRamPage(size_t page_idx) const;
};

} // namespace clio::run::bdev

#endif // CLIO_BDEV_MEM_TRANSPORT_H_
