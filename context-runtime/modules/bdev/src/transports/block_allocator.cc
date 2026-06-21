/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * ...
 */

#include <clio_runtime/bdev/transports/block_allocator.h>

namespace clio::run::bdev {

// The implementation of WorkerBlockMap, GlobalBlockMap, Heap 
// will be moved here from bdev_runtime.cc

const size_t kBlockSizes[] = {
    4096,     // 4KB
    16384,    // 16KB
    32768,    // 32KB
    65536,    // 64KB
    131072,   // 128KB
    1048576   // 1MB
};

WorkerBlockMap::WorkerBlockMap() {
  blocks_.resize(static_cast<size_t>(BlockSizeCategory::kMaxCategories));
}

bool WorkerBlockMap::AllocateBlock(int block_type, Block &block, size_t min_size) {
  if (block_type < 0 ||
      block_type >= static_cast<int>(BlockSizeCategory::kMaxCategories)) {
    return false;
  }
  auto &list = blocks_[block_type];
  if (list.empty()) return false;

  for (auto it = list.begin(); it != list.end(); ++it) {
    if (it->size_ >= min_size) {
      block = *it;
      list.erase(it);
      return true;
    }
  }
  return false;
}

void WorkerBlockMap::FreeBlock(Block block) {
  int block_type = static_cast<int>(block.block_type_);
  if (block_type >= 0 &&
      block_type < static_cast<int>(BlockSizeCategory::kMaxCategories)) {
    blocks_[block_type].push_back(block);
  }
}

GlobalBlockMap::GlobalBlockMap() {}

void GlobalBlockMap::Init(size_t num_workers) {
  worker_maps_.resize(num_workers);
  worker_locks_.resize(num_workers);
}

int GlobalBlockMap::FindBlockType(size_t io_size) {
  for (int i = 0; i < static_cast<int>(BlockSizeCategory::kMaxCategories); ++i) {
    if (io_size <= kBlockSizes[i]) return i;
  }
  return -1;
}

bool GlobalBlockMap::AllocateBlock(int worker, size_t io_size, Block &block) {
  int block_type = FindBlockType(io_size);
  if (block_type == -1) {
    block_type = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
  }

  size_t worker_idx = static_cast<size_t>(worker) % worker_maps_.size();

  {
    clio::run::ScopedCoMutex lock(worker_locks_[worker_idx]);
    if (worker_maps_[worker_idx].AllocateBlock(block_type, block, io_size)) {
      return true;
    }
  }

  for (size_t i = 1; i < worker_maps_.size(); ++i) {
    size_t other_worker = (worker_idx + i) % worker_maps_.size();
    clio::run::ScopedCoMutex lock(worker_locks_[other_worker]);
    if (worker_maps_[other_worker].AllocateBlock(block_type, block, io_size)) {
      return true;
    }
  }

  return false;
}

bool GlobalBlockMap::FreeBlock(int worker, Block &block) {
  if (worker_maps_.empty()) return false;
  size_t worker_idx = static_cast<size_t>(worker) % worker_maps_.size();
  clio::run::ScopedCoMutex lock(worker_locks_[worker_idx]);
  worker_maps_[worker_idx].FreeBlock(block);
  return true;
}

Heap::Heap() : heap_(0), total_size_(0), alignment_(4096) {}

void Heap::Init(clio::run::u64 total_size, clio::run::u32 alignment) {
  heap_.store(0);
  total_size_ = total_size;
  alignment_ = alignment;
}

bool Heap::Allocate(size_t block_size, int block_type, Block &block) {
  clio::run::u32 alignment = (alignment_ == 0) ? 4096 : alignment_;
  clio::run::u64 aligned_size =
      ((block_size + alignment - 1) / alignment) * alignment;

  clio::run::u64 old_heap = heap_.fetch_add(aligned_size);
  if (total_size_ > 0 && old_heap + aligned_size > total_size_) {
    heap_.fetch_sub(aligned_size);
    return false;
  }

  block.offset_ = old_heap;
  block.size_ = static_cast<clio::run::u64>(block_size);
  block.block_type_ = static_cast<clio::run::u32>(block_type);
  return true;
}

clio::run::u64 Heap::GetRemainingSize() const {
  clio::run::u64 current_heap = heap_.load();
  if (total_size_ > current_heap) {
    return total_size_ - current_heap;
  }
  return 0;
}

bool StandardBlockAllocator::AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) {
  clio::run::u64 total_size = size;
  if (total_size == 0) {
    blocks.clear();
    return true;
  }

  Block block;
  if (global_block_map_.AllocateBlock(worker_id, total_size, block)) {
    blocks.push_back(block);
    allocated_bytes_.fetch_add(block.size_, std::memory_order_relaxed);
    return true;
  }

  clio::run::u64 aligned_total_size = AlignSize(total_size);
  // Classify the fresh allocation by its size category so callers (and the
  // free list it returns to) see the correct block_type_. Previously this
  // hardcoded the largest category, so e.g. a 4KB allocation came back tagged
  // as 1MB (block_type_ != 0).
  int block_type = GlobalBlockMap::FindBlockType(aligned_total_size);
  if (block_type == -1) {
    block_type = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
  }
  if (heap_.Allocate(aligned_total_size, block_type, block)) {
    block.size_ = total_size;
    blocks.push_back(block);
    allocated_bytes_.fetch_add(block.size_, std::memory_order_relaxed);
    return true;
  }

  return false;
}

void StandardBlockAllocator::FreeBlocks(int worker_id, const std::vector<Block>& blocks) {
  clio::run::u64 freed_bytes = 0;
  for (const auto& block : blocks) {
    Block block_copy = block;
    clio::run::u64 aligned_size = AlignSize(block.size_);
    block_copy.size_ = aligned_size;
    int bt = -1;
    for (int i = 0; i < static_cast<int>(BlockSizeCategory::kMaxCategories); ++i) {
      if (aligned_size <= kBlockSizes[i]) {
        bt = i;
        break;
      }
    }
    if (bt == -1) bt = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
    block_copy.block_type_ = static_cast<clio::run::u32>(bt);
    freed_bytes += block_copy.size_;
    global_block_map_.FreeBlock(worker_id, block_copy);
  }

  clio::run::u64 cur = allocated_bytes_.load(std::memory_order_relaxed);
  clio::run::u64 dec = std::min(cur, freed_bytes);
  allocated_bytes_.fetch_sub(dec, std::memory_order_relaxed);
}

clio::run::u64 StandardBlockAllocator::GetRemainingSize() const {
  clio::run::u64 live = allocated_bytes_.load(std::memory_order_relaxed);
  return (capacity_ > live) ? (capacity_ - live) : 0;
}

} // namespace clio::run::bdev
