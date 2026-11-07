/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * ...
 */

#include <clio_runtime/bdev/transports/block_allocator.h>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <set>
#ifdef __linux__
#include <execinfo.h>
#endif

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

GlobalBlockMap::GlobalBlockMap() {}

void GlobalBlockMap::Init(size_t /*num_workers*/) {
  clio::run::ScopedCoMutex lock(mu_);
  by_off_.clear();
  by_size_.clear();
  free_bytes_ = 0;
}

int GlobalBlockMap::FindBlockType(size_t io_size) {
  for (int i = 0; i < static_cast<int>(BlockSizeCategory::kMaxCategories); ++i) {
    if (io_size <= kBlockSizes[i]) return i;
  }
  return -1;
}

namespace {
clio::run::u32 TypeOf(clio::run::u64 footprint) {
  int bt = GlobalBlockMap::FindBlockType(static_cast<size_t>(footprint));
  if (bt < 0) bt = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
  return static_cast<clio::run::u32>(bt);
}
}  // namespace

void GlobalBlockMap::AddLocked(clio::run::u64 off, clio::run::u64 len) {
  by_off_.emplace(off, len);
  by_size_.emplace(len, off);
  free_bytes_ += len;
}

void GlobalBlockMap::EraseLocked(
    std::map<clio::run::u64, clio::run::u64>::iterator it) {
  by_size_.erase({it->second, it->first});
  free_bytes_ -= it->second;
  by_off_.erase(it);
}

void GlobalBlockMap::InsertLocked(clio::run::u64 off, clio::run::u64 len) {
  if (len == 0) return;
  // Merge with the right neighbour, then the left one.
  auto next = by_off_.lower_bound(off);
  if (next != by_off_.end() && next->first < off + len) {
    // Overlap means a double free or a stale recovery record: never let the
    // index hold the same byte twice. Keep the existing extent, add only the
    // parts outside it (rare; logged by the CLIO_BDEV_CHECK_ALLOC checker).
    HLOG(kWarning, "bdev free index: freed [{}, {}) overlaps free [{}, {})",
         off, off + len, next->first, next->first + next->second);
    clio::run::u64 left_len = next->first > off ? next->first - off : 0;
    clio::run::u64 right_start = next->first + next->second;
    clio::run::u64 end = off + len;
    if (left_len) InsertLocked(off, left_len);
    if (right_start < end) InsertLocked(right_start, end - right_start);
    return;
  }
  if (next != by_off_.end() && next->first == off + len) {
    len += next->second;
    EraseLocked(next);
  }
  auto it = by_off_.lower_bound(off);
  if (it != by_off_.begin()) {
    auto prev = std::prev(it);
    const clio::run::u64 prev_end = prev->first + prev->second;
    if (prev_end > off) {
      HLOG(kWarning, "bdev free index: freed [{}, {}) overlaps free [{}, {})",
           off, off + len, prev->first, prev_end);
      if (prev_end >= off + len) return;
      len = off + len - prev_end;
      off = prev_end;
    }
    if (prev->first + prev->second == off) {
      off = prev->first;
      len += prev->second;
      EraseLocked(prev);
    }
  }
  AddLocked(off, len);
}

bool GlobalBlockMap::AllocateBlock(int /*worker*/, size_t footprint,
                                   Block &block) {
  if (footprint == 0) return false;
  clio::run::ScopedCoMutex lock(mu_);
  auto fit = by_size_.lower_bound({static_cast<clio::run::u64>(footprint), 0});
  if (fit == by_size_.end()) return false;
  const clio::run::u64 len = fit->first;
  const clio::run::u64 off = fit->second;
  EraseLocked(by_off_.find(off));
  if (len > footprint) AddLocked(off + footprint, len - footprint);
  block = Block(off, footprint, TypeOf(footprint));
  return true;
}

bool GlobalBlockMap::AllocateAnyUpTo(int /*worker*/, size_t max_bytes,
                                     Block &block) {
  if (max_bytes == 0) return false;
  clio::run::ScopedCoMutex lock(mu_);
  if (by_size_.empty()) return false;
  auto big = std::prev(by_size_.end());
  const clio::run::u64 len = big->first;
  const clio::run::u64 off = big->second;
  const clio::run::u64 take = std::min<clio::run::u64>(len, max_bytes);
  EraseLocked(by_off_.find(off));
  if (len > take) AddLocked(off + take, len - take);
  block = Block(off, take, TypeOf(take));
  return true;
}

bool GlobalBlockMap::FreeBlock(int /*worker*/, Block &block) {
  clio::run::ScopedCoMutex lock(mu_);
  InsertLocked(block.offset_, block.size_);
  return true;
}

void GlobalBlockMap::SeedFreeRange(clio::run::u64 offset, clio::run::u64 size) {
  if (size == 0) return;
  clio::run::ScopedCoMutex lock(mu_);
  InsertLocked(offset, size);
}

clio::run::u64 GlobalBlockMap::FreeBytes() {
  clio::run::ScopedCoMutex lock(mu_);
  return free_bytes_;
}

size_t GlobalBlockMap::NumExtents() {
  clio::run::ScopedCoMutex lock(mu_);
  return by_off_.size();
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

namespace {
/** Log the calling stack to stderr (checker diagnostics). */
void DumpStack() {
#ifdef __linux__
  void *frames[32];
  int n = backtrace(frames, 32);
  backtrace_symbols_fd(frames, n, 2);
#endif
}
}  // namespace

bool StandardBlockAllocator::CheckEnabled() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_BDEV_CHECK_ALLOC");
    return e != nullptr && *e == '1';
  }();
  return on;
}

void StandardBlockAllocator::CheckAllocated(const std::vector<Block> &blocks) {
  std::lock_guard<std::mutex> g(check_mu_);
  for (const auto &b : blocks) {
    const clio::run::u64 lo = b.offset_;
    const clio::run::u64 hi = b.offset_ + AlignSize(b.size_);
    auto it = live_.upper_bound(lo);
    bool overlap = false;
    if (it != live_.begin() && std::prev(it)->second > lo) overlap = true;
    if (it != live_.end() && it->first < hi) overlap = true;
    if (overlap) {
      HLOG(kError, "BDEV-CHECK: allocated [{}, {}) overlaps a live extent: "
           "two owners of the same bytes", lo, hi);
      DumpStack();
    }
    live_[lo] = std::max(live_[lo], hi);
  }
}

void StandardBlockAllocator::CheckFreed(const std::vector<Block> &blocks) {
  std::lock_guard<std::mutex> g(check_mu_);
  for (const auto &b : blocks) {
    const clio::run::u64 lo = b.offset_;
    const clio::run::u64 hi = b.offset_ + AlignSize(b.size_);
    auto it = live_.upper_bound(lo);
    if (it == live_.begin() || std::prev(it)->second < hi) {
      HLOG(kError, "BDEV-CHECK: freed [{}, {}) is not live: double free",
           lo, hi);
      DumpStack();
      continue;
    }
    --it;  // the live extent [s, e) containing [lo, hi)
    const clio::run::u64 s = it->first;
    const clio::run::u64 e = it->second;
    live_.erase(it);
    if (s < lo) live_[s] = lo;  // a partial free (a shrink) keeps the rest
    if (hi < e) live_[hi] = e;
  }
}

bool StandardBlockAllocator::AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) {
  const size_t first_new = blocks.size();
  const bool ok = AllocateBlocksImpl(size, worker_id, blocks);
  if (ok && CheckEnabled() && blocks.size() > first_new) {
    CheckAllocated(std::vector<Block>(blocks.begin() + first_new, blocks.end()));
  }
  return ok;
}

bool StandardBlockAllocator::AllocateBlocksImpl(size_t size, int worker_id,
                                                std::vector<Block>& blocks) {
  clio::run::u64 total_size = size;
  if (total_size == 0) {
    blocks.clear();
    return true;
  }
  const clio::run::u64 footprint = AlignSize(total_size);
  int type = GlobalBlockMap::FindBlockType(static_cast<size_t>(total_size));
  if (type == -1) type = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;

  // 1. Best fit from the coalescing free index: freed space is reused first,
  //    and because frees merge, a large request finds a large extent whenever
  //    the freed space it needs is contiguous (#1333).
  Block block;
  if (global_block_map_.AllocateBlock(worker_id, footprint, block)) {
    block.size_ = total_size;  // logical size the caller reads/writes
    block.block_type_ = static_cast<clio::run::u32>(type);
    blocks.push_back(block);
    allocated_bytes_.fetch_add(footprint, std::memory_order_relaxed);
    return true;
  }

  // 2. One contiguous block from the untouched heap.
  if (heap_.Allocate(footprint, type, block)) {
    block.size_ = total_size;
    blocks.push_back(block);
    // Charge the ALIGNED size (the heap advanced by it, and FreeBlocks credits
    // AlignSize(size_) -- keeping the two in step; see #798).
    allocated_bytes_.fetch_add(footprint, std::memory_order_relaxed);
    return true;
  }

  // 3. LAST RESORT (#820): no single free extent or heap span fits. Assemble
  //    the request from the LARGEST free extents first so it spans as few
  //    pieces as possible. Pieces report their physical footprint; the caller
  //    distributes the logical request across them.
  clio::run::u64 remaining_phys = footprint;
  Block fb;
  while (remaining_phys > 0 &&
         global_block_map_.AllocateAnyUpTo(worker_id, remaining_phys, fb)) {
    blocks.push_back(fb);
    allocated_bytes_.fetch_add(fb.size_, std::memory_order_relaxed);
    remaining_phys -= fb.size_;
  }
  if (remaining_phys == 0) {
    return true;
  }

  // Genuinely out of space: roll back so the caller sees all-or-nothing.
  for (auto &b : blocks) {
    global_block_map_.FreeBlock(worker_id, b);
    allocated_bytes_.fetch_sub(b.size_, std::memory_order_relaxed);
  }
  blocks.clear();
  RetractHeapTail();
  return false;
}

void StandardBlockAllocator::RetractHeapTail() {
  global_block_map_.RetractTail(
      [this](clio::run::u64 start, clio::run::u64 end) {
        return heap_.Retract(start, end);
      });
}

void StandardBlockAllocator::InitFromLive(
    const std::vector<std::pair<clio::run::u64, clio::run::u64>>& live) {
  // Aligned footprints, sorted by offset; overlapping records (a stale alloc
  // record whose free was lost) merge rather than double count.
  std::vector<std::pair<clio::run::u64, clio::run::u64>> ext;
  ext.reserve(live.size());
  for (const auto& e : live) {
    if (e.second == 0) continue;
    ext.emplace_back(e.first, e.first + AlignSize(e.second));
  }
  std::sort(ext.begin(), ext.end());
  clio::run::u64 cursor = 0;
  clio::run::u64 used = 0;
  for (const auto& e : ext) {
    if (e.first > cursor) {
      global_block_map_.SeedFreeRange(cursor, e.first - cursor);
    }
    clio::run::u64 start = std::max(e.first, cursor);
    if (e.second > start) {
      used += e.second - start;
      cursor = e.second;
    }
  }
  heap_.SetCursor(cursor);
  allocated_bytes_.store(used, std::memory_order_relaxed);
  if (CheckEnabled()) {
    std::lock_guard<std::mutex> g(check_mu_);
    live_.clear();
    for (const auto &e : ext) live_[e.first] = std::max(live_[e.first], e.second);
  }
}

void StandardBlockAllocator::FreeBlocks(int worker_id, const std::vector<Block>& blocks) {
  if (CheckEnabled()) CheckFreed(blocks);
  clio::run::u64 freed_bytes = 0;
  for (const auto& block : blocks) {
    if (block.size_ == 0) continue;
    Block block_copy = block;
    block_copy.size_ = AlignSize(block.size_);  // physical footprint
    block_copy.block_type_ = TypeOf(block_copy.size_);
    freed_bytes += block_copy.size_;
    global_block_map_.FreeBlock(worker_id, block_copy);  // merges neighbours
  }
  // Freed space at the top of the used range goes back to the heap, so the
  // heap -- not a pile of extents -- serves the next big contiguous request.
  RetractHeapTail();

  clio::run::u64 cur = allocated_bytes_.load(std::memory_order_relaxed);
  clio::run::u64 dec = std::min(cur, freed_bytes);
  allocated_bytes_.fetch_sub(dec, std::memory_order_relaxed);
}

clio::run::u64 StandardBlockAllocator::GetRemainingSize() const {
  clio::run::u64 live = allocated_bytes_.load(std::memory_order_relaxed);
  return (capacity_ > live) ? (capacity_ - live) : 0;
}

} // namespace clio::run::bdev
