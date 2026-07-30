/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#ifndef CLIO_BDEV_MEM_TRANSPORT_H_
#define CLIO_BDEV_MEM_TRANSPORT_H_

#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <clio_runtime/bdev/transports/block_allocator.h>
// Must follow the runtime headers above: posix_shm_mmap.h needs the memory
// backend base types they pull in.
#include <clio_ctp/memory/backend/posix_shm_mmap.h>
#include <atomic>
#include <limits>
#include <mutex>
#include <string>

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

 public:
  /**
   * Header at the start of a RAM bdev's shared-memory segment (issue #783).
   *
   * Lets a client sanity-check a segment it attached by NAME before trusting
   * any offset in it. POD, fixed layout, no pointers.
   */
  struct ShmRamHeader {
    static constexpr clio::run::u32 kVersion = 1;
    clio::run::u32 version_;
    clio::run::u32 ready_;      /**< 0 until the mapping is usable */
    clio::run::u64 capacity_;   /**< bytes of addressable device space */
    clio::run::u64 data_off_;   /**< offset from header start to byte 0 */
    clio::run::u64 reserved_[5];
  };

  /**
   * Deterministic segment name for a node-local RAM bdev.
   *
   * Derived purely from (runtime pid, pool id) so a client can construct it
   * from information it already has -- the server pid from ClientConnect and
   * the target pool id carried in each cached block descriptor. That removes
   * any need to publish a name or a page table separately.
   */
  static std::string ShmSegmentName(clio::run::u32 server_pid,
                                    const clio::run::PoolId &pool_id) {
    return "clio_rambdev_" + std::to_string(server_pid) + "_" +
           std::to_string(pool_id.major_) + "_" +
           std::to_string(pool_id.minor_);
  }

 private:
  // issue #783: when this is a node-local kRam device, its bytes live in a
  // dedicated shared-memory segment so client processes can read blob payloads
  // directly instead of paying a round-trip.
  //
  // The whole capacity is mapped as ONE contiguous sparse region rather than a
  // page table: these segments are memfd-backed, so an untouched reservation
  // costs nothing, and a flat mapping means a reader resolves a block offset
  // with an addition instead of chasing an index. kRamPageSize is retained
  // only to keep the existing paged call sites unchanged.
  /** Create the shared-memory backing for a node-local RAM device. */
  void InitShmBacking(const clio::run::PoolId &pool_id);

  /**
   * True if this page's START lies inside the shared mapping.
   *
   * Deliberately NOT "the whole 1 GiB page fits": a device smaller than one
   * page has a partial page 0, and requiring the full page rejected it --
   * which silently pushed every RAM bdev back to the private heap and broke
   * direct payload reads. Callers clamp their access within the device
   * capacity, and the mapping is sized to cover it (see kBackendHeaderSlack).
   */
  bool ShmPageInBounds(size_t page_idx) const {
    if (kRamPageSize != 0 && page_idx > shm_usable_ / kRamPageSize) {
      return false;  // guards the multiply below from overflowing
    }
    return page_idx * kRamPageSize < shm_usable_;
  }

  ctp::ipc::PosixShmMmap shm_backend_;
  size_t shm_usable_ = 0;  /**< bytes of device space actually mapped */
  char *shm_base_ = nullptr;       /**< byte 0 of device space, or nullptr */
  ShmRamHeader *shm_header_ = nullptr;
  bool shm_backed_ = false;
  std::string shm_name_;

  // Incremental population of the sparse SHM mapping (SystemInfo::BulkFault).
  // First-touch demand faulting made cold placement ~20x slower than warm on
  // WSL2 (one #PF per 4KB of memcpy), so when an allocation crosses this
  // watermark the NEXT populate-unit of pages is bulk-faulted in. The
  // watermark advances by CAS — no lock — because population is ADVISORY: a
  // thread that observes an advanced watermark before the winner's BulkFault
  // finishes just demand-faults those pages exactly as before. Committed
  // memory stays <= round_up(allocation high-water mark, unit): capacity that
  // is never allocated is never touched.
  clio::run::u64 populate_unit_ = 64ULL << 20;
  std::atomic<clio::run::u64> populated_bytes_{0};
  /** Bulk-fault [populated_bytes_, round_up(end, unit)) if end crosses it. */
  void EnsurePopulated(clio::run::u64 end);

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
