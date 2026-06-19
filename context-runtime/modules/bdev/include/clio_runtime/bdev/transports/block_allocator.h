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
#include <vector>
#include <list>

namespace clio::run::bdev {

enum class BlockSizeCategory : chi::u32 {
  k256B = 0,
  k1KB = 1,
  k4KB = 2,
  k64KB = 3,
  k128KB = 4,
  k1MB = 5,
  kMaxCategories = 6
};

extern const size_t kBlockSizes[];

/**
 * Worker-local block map to reduce lock contention
 */
class WorkerBlockMap {
 public:
  WorkerBlockMap();

  bool AllocateBlock(int block_type, Block& block, size_t min_size = 0);
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
  bool FreeBlock(int worker, Block& block);

  /** Map an I/O size to its block-size category, or -1 if larger than all. */
  static int FindBlockType(size_t io_size);

 private:
  std::vector<WorkerBlockMap> worker_maps_;
  std::vector<chi::CoMutex> worker_locks_;
};

/**
 * Heap allocator for new blocks
 */
class Heap {
 public:
  Heap();

  void Init(chi::u64 total_size, chi::u32 alignment = 4096);
  bool Allocate(size_t block_size, int block_type, Block& block);
  chi::u64 GetRemainingSize() const;

 private:
  std::atomic<chi::u64> heap_;
  chi::u64 total_size_;
  chi::u32 alignment_;
};

/**
 * Standard Allocator containing GlobalBlockMap and Heap
 */
class StandardBlockAllocator {
 public:
  StandardBlockAllocator() : alignment_(4096), capacity_(0) {}

  void Init(size_t num_workers, chi::u64 capacity, chi::u32 alignment) {
    capacity_ = capacity;
    alignment_ = alignment;
    global_block_map_.Init(num_workers);
    heap_.Init(capacity, alignment);
  }

  bool AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks);
  void FreeBlocks(int worker_id, const std::vector<Block>& blocks);

  chi::u64 GetRemainingSize() const;
  chi::u64 GetCapacity() const { return capacity_; }

 private:
  GlobalBlockMap global_block_map_;
  Heap heap_;
  chi::u32 alignment_;
  chi::u64 capacity_;
  std::atomic<chi::u64> allocated_bytes_{0};

  chi::u64 AlignSize(chi::u64 size) {
    if (alignment_ == 0) alignment_ = 4096;
    return ((size + alignment_ - 1) / alignment_) * alignment_;
  }
};

} // namespace clio::run::bdev

#endif // CLIO_BDEV_BLOCK_ALLOCATOR_H_
