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
    512,      // 512B
    1024,     // 1KB
    2048,     // 2KB
    4096,     // 4KB
    16384,    // 16KB
    32768,    // 32KB
    65536,    // 64KB
    131072,   // 128KB
    1048576   // 1MB
};
static_assert(sizeof(kBlockSizes) / sizeof(kBlockSizes[0]) ==
                  static_cast<size_t>(BlockSizeCategory::kMaxCategories),
              "kBlockSizes must match BlockSizeCategory");

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

bool WorkerBlockMap::AllocateAnyUpTo(size_t max_bytes, Block &block) {
  // Largest category first so a big request drains few large freed extents
  // rather than many tiny ones. Match on the block's PHYSICAL FOOTPRINT (its
  // 4 KiB-aligned size, which is what it actually occupies), not its logical
  // size_ -- taking a block whose footprint exceeds the remainder would occupy
  // more space than is charged for it. Alignment is 4 KiB everywhere in this
  // allocator (the block-size categories and the heap slab are all 4 KiB
  // multiples), so it is safe to compute here without the allocator's field.
  for (int bt = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1; bt >= 0;
       --bt) {
    auto &list = blocks_[bt];
    for (auto it = list.begin(); it != list.end(); ++it) {
      clio::run::u64 footprint = ((it->size_ + 4095) / 4096) * 4096;
      if (it->size_ > 0 && footprint <= max_bytes) {
        block = *it;
        list.erase(it);
        return true;
      }
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

bool GlobalBlockMap::AllocateAnyUpTo(int worker, size_t max_bytes,
                                     Block &block) {
  if (worker_maps_.empty()) return false;
  size_t worker_idx = static_cast<size_t>(worker) % worker_maps_.size();
  {
    clio::run::ScopedCoMutex lock(worker_locks_[worker_idx]);
    if (worker_maps_[worker_idx].AllocateAnyUpTo(max_bytes, block)) {
      return true;
    }
  }
  for (size_t i = 1; i < worker_maps_.size(); ++i) {
    size_t other = (worker_idx + i) % worker_maps_.size();
    clio::run::ScopedCoMutex lock(worker_locks_[other]);
    if (worker_maps_[other].AllocateAnyUpTo(max_bytes, block)) {
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

  // 1. Reuse a single free extent that already fits (the common case).
  Block block;
  if (global_block_map_.AllocateBlock(worker_id, total_size, block)) {
    blocks.push_back(block);
    allocated_bytes_.fetch_add(block.size_, std::memory_order_relaxed);
    return true;
  }

  // 2. Take one contiguous block from the heap. This is tried BEFORE
  //    fragmenting so that a request the heap can satisfy stays a SINGLE
  //    block -- fragmenting a request the heap could serve cleanly only
  //    scatters it for no benefit (and broke callers that assume one
  //    allocation is one block, e.g. bdev_file_vs_ram_comparison).
  clio::run::u64 aligned_total_size = AlignSize(total_size);
  int whole_type = GlobalBlockMap::FindBlockType(aligned_total_size);
  if (whole_type == -1) {
    whole_type = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
  }
  if (heap_.Allocate(aligned_total_size, whole_type, block)) {
    block.size_ = total_size;  // logical size the caller reads/writes
    blocks.push_back(block);
    // Charge the ALIGNED size (the heap advanced by it, and FreeBlocks credits
    // AlignSize(size_) -- keeping the two in step; see #798).
    allocated_bytes_.fetch_add(aligned_total_size, std::memory_order_relaxed);
    return true;
  }

  // 3. LAST RESORT: the heap cannot give a contiguous block (the tier is at or
  //    near capacity). ASSEMBLE the request from several smaller freed extents
  //    (issue #820). Without this, the free list is bucketed by size category
  //    -- a 1 MiB ask never sees freed 64 KiB blocks -- so a large ask can
  //    never reuse a fragmented pool and the bump-only heap climbs to capacity
  //    -> write EIO (xfstests 074/521/522). This is the allocator's job, not
  //    the CTE's (which used to cap every block at 64 KiB to dodge it); the
  //    blocks_ vector carries multi-block extents transparently.
  //
  //    Accounting is in PHYSICAL footprint bytes (AlignSize(size_)): each
  //    reused block is reported at its footprint and covers that many bytes, so
  //    a big ask consumes one slab per slab of need, not one slab per (possibly
  //    sub-slab) freed logical size. The caller distributes the logical request
  //    across these physical blocks.
  clio::run::u64 remaining_phys = aligned_total_size;
  Block fb;
  while (remaining_phys > 0 &&
         global_block_map_.AllocateAnyUpTo(worker_id, remaining_phys, fb)) {
    clio::run::u64 fp = AlignSize(fb.size_);       // true footprint
    if (fp > remaining_phys) fp = remaining_phys;  // last partial block
    fb.size_ = fp;                                 // report the footprint
    blocks.push_back(fb);
    allocated_bytes_.fetch_add(fp, std::memory_order_relaxed);
    remaining_phys -= fp;
  }
  if (remaining_phys == 0) {
    return true;  // satisfied entirely from reused space
  }

  // Freed extents did not cover it and the heap is full: genuinely out of
  // space. Roll everything back so the caller sees all-or-nothing (it frees
  // `blocks` only on a true return).
  for (auto &b : blocks) {
    global_block_map_.FreeBlock(worker_id, b);
    allocated_bytes_.fetch_sub(AlignSize(b.size_), std::memory_order_relaxed);
  }
  blocks.clear();
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
