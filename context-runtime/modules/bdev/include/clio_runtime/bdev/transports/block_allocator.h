/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * ...
 */

#ifndef CLIO_BDEV_BLOCK_ALLOCATOR_H_
#define CLIO_BDEV_BLOCK_ALLOCATOR_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_tasks.h>
#include <clio_runtime/comutex.h>
#include <atomic>
#include <map>
#include <mutex>
#include <vector>
#include <list>

namespace clio::run::bdev {

// Free-list bucket sizes. MUST match kBlockSizes[] in block_allocator.cc.
// Sub-4KB buckets (issue #862): a ~1KB blob no longer pins a whole 4KB block
// on byte-addressable tiers — the RAM bdev allocates at 512B granularity (see
// mem_bdev_transport.cc); device-aligned tiers (file O_DIRECT) keep a 4KB
// alignment quantum, so their requests still round up past the small buckets.
enum class BlockSizeCategory : clio::run::u32 {
  k512B = 0,
  k1KB = 1,
  k2KB = 2,
  k4KB = 3,
  k16KB = 4,
  k32KB = 5,
  k64KB = 6,
  k128KB = 7,
  k1MB = 8,
  kMaxCategories = 9
};

extern const size_t kBlockSizes[];

/**
 * Worker-local block map to reduce lock contention
 */
class WorkerBlockMap {
 public:
  WorkerBlockMap();

  bool AllocateBlock(int block_type, Block& block, size_t min_size = 0);
  /** Pull ANY freed block whose size_ is <= max_bytes (largest first), for
   *  fragmentation reuse (issue #820). Unlike AllocateBlock, which finds a
   *  block >= a size in one category, this scans every bucket for a block that
   *  FITS what is left -- a freed 8 KiB block bucketed under 16 KiB is exactly
   *  the case the >= match could never reach. */
  bool AllocateAnyUpTo(size_t max_bytes, Block& block);
  void FreeBlock(Block block);

 private:
  std::vector<std::list<Block>> blocks_;
};

/**
 * Global block map with per-worker caching and locking
 */
class GlobalBlockMap {
 public:
  GlobalBlockMap();

  void Init(size_t num_workers);
  bool AllocateBlock(int worker, size_t io_size, Block& block);
  /** @copydoc WorkerBlockMap::AllocateAnyUpTo -- searches this worker's map
   *  first, then steals from the others. */
  bool AllocateAnyUpTo(int worker, size_t max_bytes, Block& block);
  bool FreeBlock(int worker, Block& block);

  /**
   * File the free range [offset, offset + size) into worker 0's free list,
   * carved largest-size-class first so AllocateBlock can reuse it.
   * @param offset Start of the free range (bytes)
   * @param size Length of the free range (bytes, alignment multiple)
   */
  void SeedFreeRange(clio::run::u64 offset, clio::run::u64 size);

  /** Map an I/O size to its block-size category, or -1 if larger than all. */
  static int FindBlockType(size_t io_size);

 private:
  std::vector<WorkerBlockMap> worker_maps_;
  std::vector<clio::run::CoMutex> worker_locks_;
};

/**
 * Heap allocator for new blocks
 */
class Heap {
 public:
  Heap();

  void Init(clio::run::u64 total_size, clio::run::u32 alignment = 4096);
  bool Allocate(size_t block_size, int block_type, Block& block);
  clio::run::u64 GetRemainingSize() const;

  /**
   * Move the bump cursor to `cursor` (used on recovery: past every live
   * block, so the heap can never hand out a live offset).
   * @param cursor New bump offset in bytes
   */
  void SetCursor(clio::run::u64 cursor) { heap_.store(cursor); }

 private:
  std::atomic<clio::run::u64> heap_;
  clio::run::u64 total_size_;
  clio::run::u32 alignment_;
};

/**
 * Standard Allocator containing GlobalBlockMap and Heap
 */
class StandardBlockAllocator {
 public:
  StandardBlockAllocator() : alignment_(4096), capacity_(0) {}

  void Init(size_t num_workers, clio::run::u64 capacity, clio::run::u32 alignment) {
    capacity_ = capacity;
    alignment_ = alignment;
    global_block_map_.Init(num_workers);
    heap_.Init(capacity, alignment);
  }

  bool AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks);

  /**
   * Rebuild the allocator from a recovered set of live extents (after Init).
   * The heap cursor moves past the highest live byte, every gap below it is
   * returned to the free list, and the live bytes are charged as allocated,
   * so no live extent is ever handed out again and freed space is reused.
   * @param live Live extents as (offset, size) pairs; sizes may be unaligned
   */
  void InitFromLive(
      const std::vector<std::pair<clio::run::u64, clio::run::u64>>& live);
  void FreeBlocks(int worker_id, const std::vector<Block>& blocks);

  clio::run::u64 GetRemainingSize() const;
  clio::run::u64 GetCapacity() const { return capacity_; }

  /** @return whether CLIO_BDEV_CHECK_ALLOC=1 enables the live-extent checker */
  static bool CheckEnabled();

 private:
  /**
   * Checker (CLIO_BDEV_CHECK_ALLOC=1): record newly allocated extents,
   * logging an error with a backtrace if one overlaps an extent already
   * live -- two owners of the same bytes.
   * @param blocks the blocks just handed out
   */
  void CheckAllocated(const std::vector<Block> &blocks);
  /**
   * Checker: drop freed extents, logging an error with a backtrace for a
   * range that is not (entirely) live -- a double free.
   * @param blocks the blocks being freed
   */
  void CheckFreed(const std::vector<Block> &blocks);
  /** AllocateBlocks without the checker. */
  bool AllocateBlocksImpl(size_t size, int worker_id,
                          std::vector<Block> &blocks);
  std::mutex check_mu_;
  /** Live extents (offset -> aligned end), only with the checker on. */
  std::map<clio::run::u64, clio::run::u64> live_;

  GlobalBlockMap global_block_map_;
  Heap heap_;
  clio::run::u32 alignment_;
  clio::run::u64 capacity_;
  std::atomic<clio::run::u64> allocated_bytes_{0};

  clio::run::u64 AlignSize(clio::run::u64 size) {
    if (alignment_ == 0) alignment_ = 4096;
    return ((size + alignment_ - 1) / alignment_) * alignment_;
  }
};

} // namespace clio::run::bdev

#endif // CLIO_BDEV_BLOCK_ALLOCATOR_H_
