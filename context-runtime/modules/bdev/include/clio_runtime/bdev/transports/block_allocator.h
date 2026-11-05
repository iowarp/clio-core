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
#include <iterator>
#include <set>

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
 * Coalescing free-extent index shared by every worker (issue #1333).
 *
 * Freed space is kept as maximal extents: a free merges with the extents on
 * either side, so 4 KiB blocks freed next to each other become one extent a
 * later 1 MiB request can take whole. The old design filed every freed block
 * in a per-worker, per-size-class list and never merged them; after small-I/O
 * churn a 1 MiB write was assembled from 256 scattered 4 KiB blocks and hit
 * the device as 256 separate writes (~4x slower on NVMe).
 *
 * Two indexes over the same extents: by offset (merge on free) and by
 * (length, offset) (best fit on allocate). All offsets/lengths are physical
 * footprints in allocator-alignment units. One CoMutex guards both; every
 * operation is O(log n) and never yields while holding it.
 */
class GlobalBlockMap {
 public:
  GlobalBlockMap();
  /** Kept for API compatibility; the index is shared, not per worker. */
  void Init(size_t num_workers);
  /** Best fit: take the smallest free extent >= footprint, split off the
   *  rest. On success block = {offset, footprint, type of footprint}. */
  bool AllocateBlock(int worker, size_t footprint, Block& block);
  /** Take the LARGEST free extent, clipped to max_bytes (the remainder stays
   *  free), so a request assembled from pieces uses as few as possible. */
  bool AllocateAnyUpTo(int worker, size_t max_bytes, Block& block);
  /** Return [offset_, offset_ + size_) (size_ = footprint), merging with
   *  adjacent free extents. */
  bool FreeBlock(int worker, Block& block);
  /** Return a free range (recovery). Same as FreeBlock. */
  void SeedFreeRange(clio::run::u64 offset, clio::run::u64 size);
  /** While the highest free extent [start, end) is accepted by
   *  retract(start, end) (run under the index lock), remove it -- hands the
   *  heap's free tail back to the bump cursor. @return true if any removed. */
  template <typename Fn>
  bool RetractTail(Fn&& retract);
  /** Map an I/O size to its block-size category, or -1 if larger than all. */
  static int FindBlockType(size_t io_size);
  /** Total free bytes and extent count (stats / tests). */
  clio::run::u64 FreeBytes();
  size_t NumExtents();

 private:
  void InsertLocked(clio::run::u64 off, clio::run::u64 len);
  void EraseLocked(std::map<clio::run::u64, clio::run::u64>::iterator it);
  void AddLocked(clio::run::u64 off, clio::run::u64 len);
  clio::run::CoMutex mu_;
  std::map<clio::run::u64, clio::run::u64> by_off_;                 // off -> len
  std::set<std::pair<clio::run::u64, clio::run::u64>> by_size_;     // (len, off)
  clio::run::u64 free_bytes_ = 0;
};

template <typename Fn>
bool GlobalBlockMap::RetractTail(Fn&& retract) {
  bool any = false;
  clio::run::ScopedCoMutex lock(mu_);
  for (;;) {
    if (by_off_.empty()) return any;
    auto last = std::prev(by_off_.end());
    const clio::run::u64 start = last->first;
    const clio::run::u64 end = start + last->second;
    if (!retract(start, end)) return any;
    EraseLocked(last);
    any = true;
  }
}


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
  /**
   * Move the cursor back from `end` to `start` iff it is exactly `end`: the
   * free extent [start, end) is the heap's tail and becomes untouched heap
   * again. A concurrent Allocate that already advanced the cursor makes the
   * CAS fail, so a byte is never both free and handed out.
   */
  bool Retract(clio::run::u64 start, clio::run::u64 end) {
    clio::run::u64 expected = end;
    return heap_.compare_exchange_strong(expected, start);
  }
  clio::run::u64 GetCursor() const { return heap_.load(); }

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
  /** Free-index stats and heap cursor (tests / diagnostics). */
  clio::run::u64 GetFreeIndexBytes() { return global_block_map_.FreeBytes(); }
  size_t GetFreeExtentCount() { return global_block_map_.NumExtents(); }
  clio::run::u64 GetHeapCursor() const { return heap_.GetCursor(); }

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
  /** Hand free extents at the top of the used range back to the heap. */
  void RetractHeapTail();
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
