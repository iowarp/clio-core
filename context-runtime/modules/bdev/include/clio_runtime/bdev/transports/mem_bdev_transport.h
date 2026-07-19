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

  clio::run::TaskResume WriteBlocks(ctp::ipc::FullPtr<WriteTask> task) override;
  clio::run::TaskResume ReadBlocks(ctp::ipc::FullPtr<ReadTask> task) override;

  clio::run::u64 GetCapacity() const override { return allocator_.GetCapacity(); }
  clio::run::u64 GetRemainingSize() const override { return allocator_.GetRemainingSize(); }

 private:
  StandardBlockAllocator allocator_;
  clio::run::u64 ram_capacity_{0};
  BdevType bdev_type_{BdevType::kRam};

  // Benchmark-only knob (env CLIO_BDEV_FORCE_SYNC): when set, the device copy
  // path blocks the worker on cudaStreamSynchronize right after enqueueing —
  // reproducing the old synchronous behaviour — instead of yield-polling the
  // stream. Lets one binary A/B the worker-block-vs-yield change over an
  // identical device workload. Off in all normal operation.
  bool force_sync_gpu_{false};

  static constexpr size_t kRamPageSize = 1ULL << 30; // 1 GiB pages

  // A lazily-allocated RAM page. A kPinned pool allocates page-locked host
  // memory through GpuApi (cudaMallocHost / hipHostMalloc / sycl::malloc_host)
  // so GPU DMA runs truly asynchronously and concurrent transfers overlap;
  // every other pool — and any build or host without a usable GPU backend —
  // uses ordinary pageable `new char[]`. The `pinned` flag records which
  // allocator owns `data` so it is released through the matching free path.
  struct RamPage {
    char* data = nullptr;
    bool pinned = false;
  };

  mutable std::mutex ram_pages_mu_;
  std::vector<RamPage> ram_pages_;

  char* EnsureRamPage(size_t page_idx);
  char* GetRamPage(size_t page_idx) const;
  static void FreeRamPage(RamPage& page);

  // Host-source / host-dest copies: fully synchronous, set the task's result
  // fields directly. A host->host memcpy cannot be accelerated by a GPU stream.
  void WriteBlocksCpu(const ctp::ipc::FullPtr<WriteTask>& task, char* data);
  void ReadBlocksCpu(const ctp::ipc::FullPtr<ReadTask>& task, char* data);

  // Device-source / device-dest copies: enqueue every chunk on `stream` without
  // waiting, so the caller can yield the worker while the transfers run. Return
  // the transport return_code_ (0 = ok, 1 = capacity exceeded) and report the
  // number of bytes actually enqueued via the out-parameter. The caller waits on
  // the stream before publishing the result.
  int LaunchWriteBlocksGpu(const ctp::ipc::FullPtr<WriteTask>& task, char* data,
                           void* stream, clio::run::u64& bytes_written);
  int LaunchReadBlocksGpu(const ctp::ipc::FullPtr<ReadTask>& task, char* data,
                          void* stream, clio::run::u64& bytes_read);
};

} // namespace clio::run::bdev

#endif // CLIO_BDEV_MEM_TRANSPORT_H_
