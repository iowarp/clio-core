/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CTP_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_
#define CTP_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_

#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/memory/allocator/heap.h"
#include "clio_ctp/data_structures/ipc/slist_pre.h"
#include "clio_ctp/data_structures/ipc/rb_tree_pre.h"
#include "clio_ctp/thread/lock/mutex.h"
#include <cmath>
#include <vector>

namespace ctp::ipc {

/**
 * Metadata stored after each allocation
 *
 * NOTE: This structure acts as both an allocation header AND a free list node.
 * When allocated: next_ is unused, size_ holds the data size (excluding header)
 * When free: next_ links to next free page in slist, size_ holds data size (excluding header)
 * This structure is 16 bytes to accommodate slist_node requirements.
 *
 * Bit 63 of size_ is the free flag (kFreeMask). All size reads must use
 * GetSize() to mask out this bit.
 *
 * @tparam MODE MemMode::kShared uses offset-based slist_node,
 *              MemMode::kPrivate uses raw-pointer priv_slist_node.
 */
template <MemMode MODE = MemMode::kShared>
struct BuddyPage : public std::conditional_t<MODE == MemMode::kPrivate,
                                              pre::priv_slist_node,
                                              pre::slist_node> {
  using BaseNodeT = std::conditional_t<MODE == MemMode::kPrivate,
                                        pre::priv_slist_node,
                                        pre::slist_node>;
  size_t size_;  /**< Size of data portion (always excludes BuddyPage header).
                  *   Bit 63 is the free flag (kFreeMask). */

  static constexpr size_t kFreeMask = (size_t)1 << (sizeof(size_t) * 8 - 1);

  CTP_INLINE_CROSS_FUN BuddyPage() : BaseNodeT(), size_(0) {}
  CTP_INLINE_CROSS_FUN explicit BuddyPage(size_t size) : BaseNodeT(), size_(size) {}

  CTP_INLINE_CROSS_FUN void MarkFree()      { size_ |=  kFreeMask; }
  CTP_INLINE_CROSS_FUN void MarkAllocated() { size_ &= ~kFreeMask; }
  CTP_INLINE_CROSS_FUN bool IsFree()  const { return (size_ & kFreeMask) != 0; }
  CTP_INLINE_CROSS_FUN size_t GetSize() const { return size_ & ~kFreeMask; }
};

/** Type aliases for readability */
template <MemMode MODE = MemMode::kShared>
using FreeSmallBuddyPage = BuddyPage<MODE>;
template <MemMode MODE = MemMode::kShared>
using FreeLargeBuddyPage = BuddyPage<MODE>;

/**
 * Coalesce page node for RB tree-based coalescing
 */
struct CoalesceBuddyPage : public pre::rb_node {
  OffsetPtr<> key_;  /**< Offset pointer used as key for RB tree */
  size_t size_;      /**< Size of the page */

  CoalesceBuddyPage() : pre::rb_node(), key_(OffsetPtr<>::GetNull()), size_(0) {}
  explicit CoalesceBuddyPage(const OffsetPtr<> &k, size_t size)
      : pre::rb_node(), key_(k), size_(size) {}

  // Comparison operators required by rb_tree
  bool operator<(const CoalesceBuddyPage &other) const {
    return key_.load() < other.key_.load();
  }
  bool operator>(const CoalesceBuddyPage &other) const {
    return key_.load() > other.key_.load();
  }
  bool operator==(const CoalesceBuddyPage &other) const {
    return key_.load() == other.key_.load();
  }
};

/**
 * Maps old user-data offsets to new offsets after compaction.
 *
 * Entries are recorded in ascending old_off order (left-to-right heap scan),
 * so Resolve() uses binary search.
 *
 * Usage:
 *   ForwardingTable table = alloc->Compact();
 *   my_ptr = table.Resolve(my_ptr);   // update every stored OffsetPtr
 */
class ForwardingTable {
 public:
  struct Entry {
    size_t old_off;
    size_t new_off;
  };

 private:
  std::vector<Entry> entries_;

 public:
  void Record(size_t old_off, size_t new_off) {
    entries_.push_back({old_off, new_off});
  }

  /** Resolve a raw offset. Returns old_off unchanged if not found. */
  size_t Resolve(size_t old_off) const {
    size_t lo = 0, hi = entries_.size();
    while (lo < hi) {
      size_t mid = (lo + hi) / 2;
      if (entries_[mid].old_off == old_off) return entries_[mid].new_off;
      if (entries_[mid].old_off < old_off) lo = mid + 1;
      else hi = mid;
    }
    return old_off;
  }

  /** Resolve a typed OffsetPtr. Null pointers are returned unchanged. */
  template <typename T>
  OffsetPtr<T> Resolve(OffsetPtr<T> ptr) const {
    if (ptr.IsNull()) return ptr;
    return OffsetPtr<T>(Resolve(ptr.load()));
  }

  bool   Empty() const { return entries_.empty(); }
  size_t Size()  const { return entries_.size(); }
};

/**
 * Buddy allocator using power-of-two free lists
 *
 * This allocator manages memory using segregated free lists for different
 * size classes. Small allocations (<16KB) use round-up sizing, while large
 * allocations (>16KB) use round-down sizing with best-fit search.
 *
 * @tparam MODE MemMode::kShared uses offset-based slist (shared memory safe),
 *              MemMode::kPrivate uses raw-pointer priv_slist with cached base
 *              pointer (faster, not shared-memory safe).
 */
template <MemMode MODE = MemMode::kShared>
class _BuddyAllocator : public Allocator {
 public:
  using PageT = BuddyPage<MODE>;

 private:
  /** Free list type: shared uses offset-based slist, private uses raw-pointer priv_slist */
  using PageListT = std::conditional_t<MODE == MemMode::kPrivate,
                                        pre::priv_slist<PageT>,
                                        pre::slist<PageT, false>>;

  static constexpr size_t kMinSize = 32;           /**< Minimum allocation size (2^5) */
  static constexpr size_t kSmallThreshold = 16384; /**< 16KB threshold (2^14) */
  static constexpr size_t kMaxSize = 1048576;      /**< Maximum size class (2^20 = 1MB) */

  static constexpr size_t kMinLog2 = 5;    /**< log2(32) */
  static constexpr size_t kSmallLog2 = 14; /**< log2(16384) */
  static constexpr size_t kMaxLog2 = 20;   /**< log2(1048576) */

  static constexpr size_t kMaxSmallPages = kSmallLog2 - kMinLog2 + 1; /**< 5 to 14 = 10 lists */
  static constexpr size_t kMaxLargePages = kMaxLog2 - kSmallLog2;     /**< 15 to 20 = 6 lists */

  static constexpr size_t kSmallArenaSize = 65536; /**< 64KB arena size */
  static constexpr size_t kSmallArenaPages = 128;  /**< Max pages in small arena */

  Heap<false> big_heap_;   /**< Heap for large allocations */
  Heap<false> small_arena_; /**< Arena for small allocations */

  PageListT small_pages_[kMaxSmallPages];   /**< Free lists for sizes 32B - 16KB */
  PageListT large_pages_[kMaxLargePages]; /**< Free lists for sizes 16KB - 1MB */
  PageListT regions_;   /**< List of big_heap_ regions */

  ArenaState cur_arena_;  /**< Current bump arena (if active) */

  // Serializes Allocate/Free. The free lists are intrusive slists with no
  // internal synchronization; "kShared" only means position-independent
  // (offset-based) nodes, NOT thread-safety. Concurrent allocators must hold
  // this. (Thread-locality used to come from ProducerConsumerAllocator's
  // per-thread PcThreadBlocks; this lock makes a single shared BuddyAllocator
  // safe to use directly from many threads.)
  ctp::Mutex lock_;

#ifdef CTP_BUDDY_ALLOC_DEBUG
  size_t dbg_alloc_count_;
  size_t dbg_free_count_;
  size_t dbg_net_bytes_;
  size_t dbg_big_heap_alloc_count_;
#endif

  // _MultiProcessAllocator needs access to reconstruct pointers when attaching
  friend class _MultiProcessAllocator;

 public:
#ifdef CTP_BUDDY_ALLOC_DEBUG
  CTP_CROSS_FUN size_t DbgAllocCount() const { return dbg_alloc_count_; }
  CTP_CROSS_FUN size_t DbgFreeCount() const { return dbg_free_count_; }
  CTP_CROSS_FUN size_t DbgNetBytes() const { return dbg_net_bytes_; }
  CTP_CROSS_FUN size_t DbgBigHeapOffset() const { return big_heap_.GetOffset(); }
  CTP_CROSS_FUN size_t DbgBigHeapMax() const { return big_heap_.GetMaxOffset(); }
#endif

  /** Convert offset to raw pointer (position-independent, safe for multi-process) */
  CTP_INLINE_CROSS_FUN PageT *OffsetToPage(size_t offset) {
    return reinterpret_cast<PageT *>(GetBackendData() + offset);
  }

  /** Convert raw pointer back to offset (position-independent, safe for multi-process) */
  CTP_INLINE_CROSS_FUN size_t PageToOffset(PageT *ptr) {
    return reinterpret_cast<char *>(ptr) - GetBackendData();
  }

  /**
   * Initialize the buddy allocator
   */
  CTP_CROSS_FUN bool shm_init(const MemoryBackend &backend,
                               size_t region_size = 0,
                               bool shifted = false) {
    SetBackend(backend);
    alloc_header_size_ = sizeof(_BuddyAllocator);
    this_ = reinterpret_cast<char *>(this) -
            reinterpret_cast<char *>(backend.data_);

    if (shifted) {
      shift_ = this_;
      data_start_ = 0;
    } else {
      shift_ = 0;
      data_start_ = sizeof(_BuddyAllocator);
    }

    if (region_size == 0) {
      region_size = shifted ? backend.data_capacity_
                            : backend.data_capacity_ - this_;
    }
    region_size_ = region_size;

    if (region_size < kMinSize) {
      return false;
    }

    regions_.Init();
    big_heap_.Init(0, 0);
    small_arena_.Init(0, 0);
    cur_arena_ = ArenaState{};
    lock_.Init();
    for (size_t i = 0; i < kMaxSmallPages; ++i) {
      small_pages_[i].Init();
    }
    for (size_t i = 0; i < kMaxLargePages; ++i) {
      large_pages_[i].Init();
    }

    Expand(OffsetPtr<>(GetAllocatorDataOff()), GetAllocatorDataSize());
#ifdef CTP_BUDDY_ALLOC_DEBUG
    dbg_alloc_count_ = 0;
    dbg_free_count_ = 0;
    dbg_net_bytes_ = 0;
    dbg_big_heap_alloc_count_ = 0;
#endif
    return true;
  }

  /**
   * Attach to an existing buddy allocator
   */
  bool shm_attach(const MemoryBackend &backend) {
    (void)backend;
    return true;
  }

  /**
   * Allocate memory of specified size
   */
  CTP_CROSS_FUN OffsetPtr<> AllocateOffset(size_t requested_size) {
    ScopedMutex guard(lock_, 0);
    // Fast path: bump-allocate from active arena
    if (cur_arena_.IsActive()) {
      constexpr size_t kAlign = 16;
#if CTP_IS_GPU
      // GPU: multiple threads in a warp may share this allocator partition.
      // Use atomicCAS loop for thread-safe bump allocation.
      while (true) {
        size_t cur = *(volatile size_t *)&cur_arena_.arena_cur_;
        size_t aligned_cur = (cur + kAlign - 1) & ~(kAlign - 1);
        size_t new_cur = aligned_cur + requested_size;
        if (new_cur > cur_arena_.arena_end_) break;
        size_t old = atomicCAS(
            reinterpret_cast<unsigned long long *>(&cur_arena_.arena_cur_),
            static_cast<unsigned long long>(cur),
            static_cast<unsigned long long>(new_cur));
        if (old == cur) {
          return OffsetPtr<>(aligned_cur);
        }
      }
#else
      size_t aligned_cur = (cur_arena_.arena_cur_ + kAlign - 1) & ~(kAlign - 1);
      if (aligned_cur + requested_size <= cur_arena_.arena_end_) {
        cur_arena_.arena_cur_ = aligned_cur + requested_size;
        return OffsetPtr<>(aligned_cur);
      }
#endif
    }

    // Slow path: regular BuddyAllocator
    if (requested_size < kMinSize) {
      requested_size = kMinSize;
    }
    constexpr size_t kAlign = alignof(PageT);
    requested_size = (requested_size + kAlign - 1) & ~(kAlign - 1);

    if (requested_size <= kSmallThreshold) {
      return AllocateSmall(requested_size);
    }
    return AllocateLarge(requested_size);
  }

  /**
   * Reallocate previously allocated memory to a new size
   */
  CTP_CROSS_FUN OffsetPtr<> ReallocateOffset(OffsetPtr<> offset, size_t new_size) {
    if (offset.IsNull()) {
      return AllocateOffset(new_size);
    }

    size_t page_offset = offset.load() - sizeof(PageT);
    PageT *page = OffsetToPage(page_offset);
    size_t old_size = page->GetSize();

    if (new_size <= old_size) {
      return offset;
    }

    OffsetPtr<> new_offset = AllocateOffset(new_size);
    if (new_offset.IsNull()) {
      return new_offset;
    }

    char *base = GetBackendData();
    memcpy(base + new_offset.load(), base + offset.load(), old_size);
    FreeOffset(offset);
    return new_offset;
  }

  /**
   * Free previously allocated memory
   */
  CTP_CROSS_FUN void FreeOffset(OffsetPtr<> offset) {
    if (offset.IsNull()) {
      return;
    }
    FreeOffsetNoNullCheck(offset);
  }

  /**
   * Free previously allocated memory (without null check)
   */
  CTP_CROSS_FUN void FreeOffsetNoNullCheck(OffsetPtr<> offset) {
    ScopedMutex guard(lock_, 0);
    size_t off = offset.load();
    // No-op for arena allocations
    if (cur_arena_.IsActive() &&
        off >= cur_arena_.arena_off_ && off < cur_arena_.arena_end_) {
      return;
    }

    size_t page_offset = off - sizeof(PageT);
    PageT *page = OffsetToPage(page_offset);
    size_t data_size = page->GetSize();
    page->MarkFree();

#ifdef CTP_BUDDY_ALLOC_DEBUG
    dbg_free_count_++;
    dbg_net_bytes_ -= data_size;
#endif

    size_t list_idx;
    if (data_size <= kSmallThreshold) {
      list_idx = GetSmallPageListIndexForFree(data_size);
      EmplaceToList(small_pages_[list_idx], page_offset);
    } else {
      list_idx = GetLargePageListIndexForFree(data_size);
      EmplaceToList(large_pages_[list_idx], page_offset);
    }
  }

  /** Push a new bump arena */
  CTP_CROSS_FUN bool PushArenaState(ArenaState &prior, OffsetPtr<> &block, size_t size) {
    prior = cur_arena_;
    cur_arena_ = ArenaState{};
    block = AllocateOffset(size);
    if (block.IsNull()) {
      cur_arena_ = prior;
      return false;
    }
    cur_arena_.arena_off_ = block.load();
    cur_arena_.arena_cur_ = block.load();
    cur_arena_.arena_end_ = block.load() + size;
    return true;
  }

  /** Pop a bump arena */
  CTP_CROSS_FUN void PopArenaState(const ArenaState &prior, OffsetPtr<> block) {
    cur_arena_ = prior;
    if (!block.IsNull()) {
      FreeOffset(block);
    }
  }

  /**
   * Expand the allocator with new memory region
   */
  CTP_CROSS_FUN void Expand(OffsetPtr<> region, size_t region_size) {
    if (region.IsNull() || region_size == 0) {
      return;
    }
    if (region_size <= sizeof(PageT)) {
      return;
    }
    PageT *node = OffsetToPage(region.load());
    node->size_ = region_size;
    EmplaceToList(regions_, region.load());
    region += sizeof(PageT);
    region_size -= sizeof(PageT);
    DivideArenaIntoPages(big_heap_);
    big_heap_.Init(region.load(),
                   region.load() + region_size);
  }

  /**
   * Compact the heap (host-only, shared mode only)
   */
  ForwardingTable Compact() {
    char *base = GetBackendData();
    size_t scan_start = GetAllocatorDataOff() + sizeof(PageT);
    size_t scan_end = big_heap_.GetOffset();
    size_t arena_tail_start = small_arena_.GetOffset();
    size_t arena_tail_end = small_arena_.GetMaxOffset();

    ForwardingTable table;
    size_t cursor = scan_start;
    size_t write_pos = scan_start;

    while (cursor < scan_end) {
      if (cursor == arena_tail_start && arena_tail_start < arena_tail_end) {
        cursor = arena_tail_end;
        continue;
      }

      PageT *page = reinterpret_cast<PageT *>(base + cursor);
      size_t data_size = page->GetSize();
      size_t blk_size = sizeof(PageT) + data_size;

      if (!page->IsFree()) {
        size_t old_user_off = cursor + sizeof(PageT);
        size_t new_user_off = write_pos + sizeof(PageT);

        if (write_pos != cursor) {
          memmove(base + write_pos, base + cursor, blk_size);
        }
        reinterpret_cast<PageT *>(base + write_pos)->MarkAllocated();
        table.Record(old_user_off, new_user_off);
        write_pos += blk_size;
      }
      cursor += blk_size;
    }

    for (size_t i = 0; i < kMaxSmallPages; ++i) small_pages_[i].Init();
    for (size_t i = 0; i < kMaxLargePages; ++i) large_pages_[i].Init();
    small_arena_.Init(0, 0);
    big_heap_.Init(write_pos, big_heap_.GetMaxOffset());
    return table;
  }

 private:
  /**
   * Add a page to a free list. In private mode, uses raw pointers.
   * In shared mode, uses FullPtr + offset-based slist.
   */
  CTP_INLINE_CROSS_FUN void EmplaceToList(PageListT &list, size_t page_offset) {
    PageT *page = OffsetToPage(page_offset);
    if constexpr (MODE == MemMode::kPrivate) {
      page->next_ = nullptr;
      list.emplace(page);
    } else {
      page->next_ = OffsetPtr<>::GetNull();
      FullPtr<PageT> node_ptr(this, page_offset);
      list.emplace(this, node_ptr);
    }
  }

  /**
   * Pop from a free list. Returns offset of popped page, or 0 if empty.
   */
  CTP_INLINE_CROSS_FUN size_t PopFromList(PageListT &list) {
    if (list.empty()) {
      return 0;
    }
    if constexpr (MODE == MemMode::kPrivate) {
      PageT *node = list.pop();
      if (!node) return 0;
      return PageToOffset(node);
    } else {
      auto node = list.pop(this);
      if (node.IsNull()) return 0;
      return node.shm_.off_.load();
    }
  }

  /**
   * Allocate small memory (<16KB) using round-up sizing
   */
  CTP_CROSS_FUN OffsetPtr<> AllocateSmall(size_t size) {
    size_t list_idx = GetSmallPageListIndexForAlloc(size);

    // Check free lists from this size class upward
    for (size_t i = list_idx; i < kMaxSmallPages; ++i) {
      size_t off = PopFromList(small_pages_[i]);
      if (off != 0) {
        // Preserve actual page data size to prevent free-list migration.
        // Without this, popping from a larger list and storing the smaller
        // requested size causes the page to shrink on free, permanently
        // losing the excess bytes.
        PageT *page = OffsetToPage(off);
        return FinalizeAllocation(off, page->GetSize());
      }
    }

    // Try allocating from small_arena_
    size_t total_size = size + sizeof(PageT);
    size_t arena_offset = small_arena_.Allocate(total_size);
    if (arena_offset != 0) {
      return FinalizeAllocation(arena_offset, size);
    }

    // Repopulate the small arena
    if (RepopulateSmallArena()) {
      for (size_t i = list_idx; i < kMaxSmallPages; ++i) {
        size_t off = PopFromList(small_pages_[i]);
        if (off != 0) {
          PageT *page = OffsetToPage(off);
          return FinalizeAllocation(off, page->GetSize());
        }
      }
    }

    // Allocate directly from big_heap_
    size_t heap_offset = big_heap_.Allocate(total_size);
    if (heap_offset != 0) {
      return FinalizeAllocation(heap_offset, size);
    }

    return OffsetPtr<>::GetNull();
  }

  /**
   * Allocate large memory (>16KB) using round-down sizing with best-fit
   */
  CTP_CROSS_FUN OffsetPtr<> AllocateLarge(size_t size) {
    size_t total_size = size + sizeof(PageT);
    size_t list_idx = GetLargePageListIndexForAlloc(size);

    for (size_t i = list_idx; i < kMaxLargePages; ++i) {
      size_t found_offset = FindFirstFit(i, total_size);
      if (found_offset != 0) {
        PageT *page = OffsetToPage(found_offset);
        size_t page_data_size = page->GetSize();
        size_t page_total_size = page_data_size + sizeof(PageT);

        if (page_total_size > total_size &&
            (page_total_size - total_size) >= sizeof(PageT) + kMinSize) {
          AddRemainderToFreeList(found_offset + total_size,
                                page_total_size - total_size);
          return FinalizeAllocation(found_offset, size);
        }
        // Remainder too small to form a valid page (data < kMinSize) — keep
        // full page size to avoid both losing the tail bytes AND handing out
        // an undersized page that the caller would overflow (#680).
        return FinalizeAllocation(found_offset, page_data_size);
      }
    }

    size_t heap_offset = big_heap_.Allocate(total_size);
    if (heap_offset != 0) {
      return FinalizeAllocation(heap_offset, size);
    }
    return OffsetPtr<>::GetNull();
  }

  /**
   * Find the first fit in a large page free list
   */
  CTP_CROSS_FUN size_t FindFirstFit(size_t list_idx, size_t required_size) {
    if (large_pages_[list_idx].empty()) {
      return 0;
    }

    if constexpr (MODE == MemMode::kPrivate) {
      for (auto it = large_pages_[list_idx].begin();
           it != large_pages_[list_idx].end(); ++it) {
        PageT *page = it.Get();
        size_t page_total_size = page->GetSize() + sizeof(PageT);
        if (page_total_size >= required_size) {
          size_t offset = PageToOffset(page);
          (void)large_pages_[list_idx].PopAt(it);
          return offset;
        }
      }
    } else {
      for (auto it = large_pages_[list_idx].begin(this);
           it != large_pages_[list_idx].end(); ++it) {
        ctp::ipc::FullPtr<PageT> free_page(this, it.GetCurrent().load());
        size_t page_total_size = free_page.ptr_->GetSize() + sizeof(PageT);
        if (page_total_size >= required_size) {
          size_t offset = it.GetCurrent().load();
          (void)large_pages_[list_idx].PopAt(this, it);
          return offset;
        }
      }
    }
    return 0;
  }

  /**
   * Repopulate small arena with more space
   */
  CTP_CROSS_FUN bool RepopulateSmallArena() {
    size_t arena_size = kSmallArenaSize + kSmallArenaPages * sizeof(PageT);

    DivideArenaIntoPages(small_arena_);

    size_t heap_offset = big_heap_.Allocate(arena_size);
    if (heap_offset != 0) {
      small_arena_.Init(heap_offset, heap_offset + arena_size);
      return true;
    }

    for (size_t list_idx = 0; list_idx < kMaxLargePages; ++list_idx) {
      size_t offset = FindFirstFit(list_idx, arena_size);
      if (offset != 0) {
        PageT *page = OffsetToPage(offset);
        // Use GetSize() to mask off kFreeMask (bit 63). FindFirstFit pops the
        // page but leaves it MarkFree'd, so reading page->size_ raw carries the
        // free flag into the arithmetic: page_total_size explodes to ~2^63 and
        // AddRemainderToFreeList below emplaces a bogus multi-GB "remainder"
        // that runs off the real page into live memory, corrupting the free
        // list. Every other size read here already uses GetSize().
        size_t page_total_size = page->GetSize() + sizeof(PageT);

        small_arena_.Init(offset, offset + arena_size);

        if (page_total_size > arena_size &&
            (page_total_size - arena_size) >= sizeof(PageT) + kMinSize) {
          AddRemainderToFreeList(offset + arena_size,
                                page_total_size - arena_size);
        }
        return true;
      }
    }
    return false;
  }

  /**
   * Divide the current small_arena_ into pages using greedy algorithm
   */
  CTP_CROSS_FUN void DivideArenaIntoPages(Heap<false> &heap) {
    size_t arena_begin = heap.GetOffset();
    size_t arena_end = heap.GetMaxOffset();
    size_t remaining_offset = arena_begin;
    size_t remaining_size = arena_end - arena_begin;

    if (remaining_size == 0) {
      return;
    }

    for (int i = static_cast<int>(kMaxSmallPages) - 1; i >= 0; --i) {
      size_t page_data_size = static_cast<size_t>(1) << (i + kMinLog2);
      size_t page_total_size = page_data_size + sizeof(PageT);

      while (remaining_size >= page_total_size) {
        PageT *page = OffsetToPage(remaining_offset);
        page->size_ = page_data_size;
        page->MarkFree();
        EmplaceToList(small_pages_[i], remaining_offset);
        remaining_offset += page_total_size;
        remaining_size -= page_total_size;
      }
    }

    heap.Init(arena_end, arena_end);
  }

  /**
   * Add a remainder page back to the appropriate free list
   */
  CTP_CROSS_FUN void AddRemainderToFreeList(size_t page_offset,
                                               size_t total_size) {
    // A page must hold at least kMinSize of data; otherwise it is filed in
    // small_pages_[0] and later handed out for a kMinSize request, letting the
    // caller overflow its short buffer into the next page's header and corrupt
    // the free list (#680).
    if (total_size < sizeof(PageT) + kMinSize) {
      return;
    }
    size_t data_size = total_size - sizeof(PageT);

    PageT *page = OffsetToPage(page_offset);
    page->size_ = data_size;
    page->MarkFree();

    if (data_size <= kSmallThreshold) {
      size_t list_idx = GetSmallPageListIndexForFree(data_size);
      EmplaceToList(small_pages_[list_idx], page_offset);
    } else {
      size_t list_idx = GetLargePageListIndexForFree(data_size);
      EmplaceToList(large_pages_[list_idx], page_offset);
    }
  }

  /**
   * Finalize allocation by setting page header and returning user offset
   */
  CTP_CROSS_FUN OffsetPtr<> FinalizeAllocation(size_t page_offset,
                                                  size_t user_size) {
    PageT *bp = OffsetToPage(page_offset);
    bp->size_ = user_size;
    bp->MarkAllocated();
#ifdef CTP_BUDDY_ALLOC_DEBUG
    dbg_alloc_count_++;
    dbg_net_bytes_ += user_size;
#endif
    return OffsetPtr<>(page_offset + sizeof(PageT));
  }

  /**
   * Get free list index for small allocations (round up)
   */
  static CTP_CROSS_FUN size_t GetSmallPageListIndexForAlloc(size_t &alloc_size) {
    if (alloc_size <= kMinSize) {
      alloc_size = kMinSize;
      return 0;
    }
    size_t log2 = ctp::CeilLog2(alloc_size);
    if (log2 < kMinLog2) {
      alloc_size = kMinSize;
      return 0;
    }
    if (log2 > kSmallLog2) {
      alloc_size = kSmallThreshold;
      return kMaxSmallPages - 1;
    }
    alloc_size = static_cast<size_t>(1) << log2;
    return log2 - kMinLog2;
  }

  /**
   * Get free list index for small pages when freeing (round down)
   */
  static CTP_CROSS_FUN size_t GetSmallPageListIndexForFree(size_t size) {
    if (size <= kMinSize) {
      return 0;
    }
    size_t log2 = ctp::FloorLog2(size);
    if (log2 < kMinLog2) {
      return 0;
    }
    if (log2 > kSmallLog2) {
      return kMaxSmallPages - 1;
    }
    return log2 - kMinLog2;
  }

  /**
   * Get free list index for large allocations (round down)
   */
  static CTP_CROSS_FUN size_t GetLargePageListIndexForAlloc(size_t size) {
    if (size <= kSmallThreshold) {
      return 0;
    }
    size_t log2 = ctp::FloorLog2(size);
    if (log2 <= kSmallLog2) {
      return 0;
    }
    if (log2 > kMaxLog2) {
      return kMaxLargePages - 1;
    }
    return log2 - kSmallLog2 - 1;
  }

  /**
   * Get free list index for large pages when freeing
   */
  CTP_CROSS_FUN size_t GetLargePageListIndexForFree(size_t size) {
    return GetLargePageListIndexForAlloc(size);
  }
};

/** Typedef for the complete BuddyAllocator with BaseAllocator wrapper */
using BuddyAllocator = BaseAllocator<_BuddyAllocator<>>;

/** Private-mode BuddyAllocator (raw pointers, cached base, for PartitionedAllocator) */
using PrivateBuddyAllocator = BaseAllocator<_BuddyAllocator<MemMode::kPrivate>>;

}  // namespace ctp::ipc

#endif  // CTP_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_
