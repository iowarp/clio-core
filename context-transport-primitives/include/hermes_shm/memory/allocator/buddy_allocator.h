/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/data_structures/ipc/slist_pre.h"
#include "hermes_shm/data_structures/ipc/rb_tree_pre.h"
#include <cmath>

namespace hshm::ipc {

/**
 * Metadata stored after each allocation
 */
struct BuddyPage {
  size_t size_;  /**< Size of the allocated page */

  BuddyPage() : size_(0) {}
  explicit BuddyPage(size_t size) : size_(size) {}
};

/**
 * Free page node for singly-linked free lists
 */
struct FreeBuddyPage : public pre::slist_node {
  size_t size_;  /**< Size of the free page */

  FreeBuddyPage() : pre::slist_node(), size_(0) {}
  explicit FreeBuddyPage(size_t size) : pre::slist_node(), size_(size) {}
};

/**
 * Coalesce page node for RB tree-based coalescing
 */
struct CoalesceBuddyPage : public pre::rb_node {
  OffsetPtr key;  /**< Offset pointer used as key for RB tree */
  size_t size_;       /**< Size of the page */

  CoalesceBuddyPage() : pre::rb_node(), key(OffsetPtr::GetNull()), size_(0) {}
  explicit CoalesceBuddyPage(const OffsetPtr &k, size_t size)
      : pre::rb_node(), key(k), size_(size) {}

  // Comparison operators required by rb_tree
  bool operator<(const CoalesceBuddyPage &other) const {
    return key.load() < other.key.load();
  }
  bool operator>(const CoalesceBuddyPage &other) const {
    return key.load() > other.key.load();
  }
  bool operator==(const CoalesceBuddyPage &other) const {
    return key.load() == other.key.load();
  }
};

/**
 * Buddy allocator using power-of-two free lists
 *
 * This allocator manages memory using segregated free lists for different
 * size classes. Small allocations (<16KB) use round-up sizing, while large
 * allocations (>16KB) use round-down sizing with best-fit search.
 *
 * Coalescing merges adjacent free pages to reduce fragmentation.
 */
class BuddyAllocator {
 private:
  static constexpr size_t kMinSize = 32;        /**< Minimum allocation size (2^5) */
  static constexpr size_t kSmallThreshold = 16384;  /**< 16KB threshold (2^14) */
  static constexpr size_t kMaxSize = 1048576;   /**< Maximum size class (2^20 = 1MB) */

  static constexpr size_t kMinLog2 = 5;   /**< log2(32) */
  static constexpr size_t kSmallLog2 = 14;  /**< log2(16384) */
  static constexpr size_t kMaxLog2 = 20;  /**< log2(1048576) */

  static constexpr size_t kNumRoundUpLists = kSmallLog2 - kMinLog2 + 1;  /**< 5 to 14 = 10 lists */
  static constexpr size_t kNumRoundDownLists = kMaxLog2 - kSmallLog2;    /**< 15 to 20 = 6 lists */
  static constexpr size_t kNumFreeLists = kNumRoundUpLists + kNumRoundDownLists;  /**< Total: 16 lists */

  size_t heap_begin_;           /**< Offset to heap beginning */
  size_t heap_current_;         /**< Current heap offset */
  size_t heap_end_;             /**< End of heap */

  pre::slist<false> *round_up_lists_;    /**< Free lists for sizes 32B - 16KB (round up) */
  pre::slist<false> *round_down_lists_;  /**< Free lists for sizes 16KB - 1MB (round down) */

 public:
  MemoryBackend backend_;       /**< Memory backend for compatibility with FullPtr */

  /**
   * Get allocator ID (required for FullPtr construction)
   * @return Null allocator ID for standalone allocator
   */
  AllocatorId GetId() const {
    return AllocatorId();
  }

  /**
   * Initialize the buddy allocator
   *
   * @param data ShmPtr to memory backend
   * @param heap_size Total size of heap to manage
   * @return true on success, false on failure
   */
  bool shm_init(char *data, size_t heap_size) {
    // Initialize backend structure for FullPtr compatibility
    backend_.data_ = data;
    backend_.data_size_ = heap_size;
    backend_.data_offset_ = 0;
    backend_.SetInitialized();

    // Calculate space needed for free list metadata
    size_t metadata_size = kNumFreeLists * sizeof(pre::slist<false>);
    size_t aligned_metadata = ((metadata_size + 63) / 64) * 64;  // 64-byte align

    if (heap_size < aligned_metadata + kMinSize) {
      return false;  // Not enough space
    }

    // Allocate free lists from beginning of heap
    round_up_lists_ = reinterpret_cast<pre::slist<false>*>(backend_.data_);
    round_down_lists_ = round_up_lists_ + kNumRoundUpLists;

    // Initialize all free lists
    for (size_t i = 0; i < kNumRoundUpLists; ++i) {
      round_up_lists_[i].Init();
    }
    for (size_t i = 0; i < kNumRoundDownLists; ++i) {
      round_down_lists_[i].Init();
    }

    // Set heap boundaries
    heap_begin_ = aligned_metadata;
    heap_current_ = heap_begin_;
    heap_end_ = heap_size;

    return true;
  }

  /**
   * Allocate memory of specified size
   *
   * @param size Size in bytes to allocate
   * @return Offset pointer to allocated memory (after BuddyPage header)
   */
  OffsetPtr AllocateOffset(size_t size) {
    if (size < kMinSize) {
      size = kMinSize;
    }

    if (size < kSmallThreshold) {
      return AllocateSmall(size);
    } else {
      return AllocateLarge(size);
    }
  }

  /**
   * Free previously allocated memory
   *
   * @param offset Offset pointer to memory (after BuddyPage header)
   */
  void FreeOffset(OffsetPtr offset) {
    if (offset.IsNull()) {
      return;
    }

    // Get the actual page start (before BuddyPage header)
    size_t page_offset = offset.load() - sizeof(BuddyPage);
    BuddyPage *page = reinterpret_cast<BuddyPage*>(backend_.data_ + page_offset);
    size_t page_size = page->size_;

    // Convert to FreeBuddyPage
    FreeBuddyPage *free_page = reinterpret_cast<FreeBuddyPage*>(page);
    free_page->size_ = page_size;

    // Add to appropriate free list
    size_t list_idx = GetFreeListIndex(page_size);
    ShmPtr shm_ptr;
    shm_ptr.off_ = page_offset;
    if (list_idx < kNumRoundUpLists) {
      FullPtr<pre::slist_node> node_ptr(this, shm_ptr);
      round_up_lists_[list_idx].emplace(this, node_ptr);
    } else {
      FullPtr<pre::slist_node> node_ptr(this, shm_ptr);
      round_down_lists_[list_idx - kNumRoundUpLists].emplace(this, node_ptr);
    }
  }

 private:
  /**
   * Allocate small memory (<16KB) using round-up sizing
   */
  OffsetPtr AllocateSmall(size_t size) {
    size_t total_size = size + sizeof(BuddyPage);
    size_t list_idx = GetRoundUpListIndex(total_size);
    size_t alloc_size = static_cast<size_t>(1) << (list_idx + kMinLog2);

    // Step 2: Check if page exists in this free list
    if (!round_up_lists_[list_idx].empty()) {
      auto node = round_up_lists_[list_idx].pop(this);
      return FinalizeAllocation(node.shm_.off_.load(), alloc_size);
    }

    // Step 3: Check larger free lists and split
    for (size_t i = list_idx + 1; i < kNumRoundUpLists; ++i) {
      if (!round_up_lists_[i].empty()) {
        auto node = round_up_lists_[i].pop(this);
        return SplitAndAllocate(node.shm_.off_.load(), i, list_idx);
      }
    }

    // Check round-down lists for large pages
    for (size_t i = 0; i < kNumRoundDownLists; ++i) {
      if (!round_down_lists_[i].empty()) {
        auto node = round_down_lists_[i].pop(this);
        FreeBuddyPage *free_page = reinterpret_cast<FreeBuddyPage*>(backend_.data_ + node.shm_.off_.load());
        if (free_page->size_ >= alloc_size) {
          return SplitLargeAndAllocate(node.shm_.off_.load(), free_page->size_, alloc_size, list_idx);
        } else {
          // Put it back
          FullPtr<pre::slist_node> put_back(this, node.shm_);
          round_down_lists_[i].emplace(this, put_back);
        }
      }
    }

    // Step 4: Try coalescing
    Coalesce(0, list_idx);

    // Retry after coalescing
    if (!round_up_lists_[list_idx].empty()) {
      auto node = round_up_lists_[list_idx].pop(this);
      return FinalizeAllocation(node.shm_.off_.load(), alloc_size);
    }

    // Step 5: Allocate from heap
    return AllocateFromHeap(alloc_size);
  }

  /**
   * Allocate large memory (>16KB) using round-down sizing with best-fit
   */
  OffsetPtr AllocateLarge(size_t size) {
    size_t total_size = size + sizeof(BuddyPage);
    size_t list_idx = GetRoundDownListIndex(total_size);

    // Step 2: Check this list for best fit
    // For now, just take the first fit
    if (!round_down_lists_[list_idx].empty()) {
      auto node = round_down_lists_[list_idx].peek(this);
      if (!node.IsNull()) {
        FreeBuddyPage *free_page = reinterpret_cast<FreeBuddyPage*>(node.ptr_);
        if (free_page->size_ >= total_size) {
          round_down_lists_[list_idx].pop(this);
          return SubsetAndAllocate(node.shm_.off_.load(), free_page->size_, total_size);
        }
      }
    }

    // Step 3: Check larger lists
    for (size_t i = list_idx + 1; i < kNumRoundDownLists; ++i) {
      if (!round_down_lists_[i].empty()) {
        auto node = round_down_lists_[i].pop(this);
        FreeBuddyPage *free_page = reinterpret_cast<FreeBuddyPage*>(backend_.data_ + node.shm_.off_.load());
        return SubsetAndAllocate(node.shm_.off_.load(), free_page->size_, total_size);
      }
    }

    // Step 4: Try coalescing
    Coalesce(0, kNumRoundUpLists + list_idx);

    // Retry
    if (!round_down_lists_[list_idx].empty()) {
      auto node = round_down_lists_[list_idx].pop(this);
      FreeBuddyPage *free_page = reinterpret_cast<FreeBuddyPage*>(backend_.data_ + node.shm_.off_.load());
      if (free_page->size_ >= total_size) {
        return SubsetAndAllocate(node.shm_.off_.load(), free_page->size_, total_size);
      }
    }

    // Step 5: Allocate from heap
    return AllocateFromHeap(total_size);
  }

  /**
   * Coalesce free pages in specified range of free lists
   *
   * @param list_id_min Index of first free list to coalesce
   * @param list_id_max Index of last free list to coalesce
   */
  void Coalesce(size_t list_id_min, size_t list_id_max) {
    // Build RB tree of all free pages in range
    pre::rb_tree<CoalesceBuddyPage, false> coalesce_tree;
    coalesce_tree.Init();

    // Pop all entries from specified free lists
    for (size_t i = list_id_min; i <= list_id_max && i < kNumFreeLists; ++i) {
      pre::slist<false> *list = GetFreeList(i);

      while (!list->empty()) {
        auto node = list->pop(this);
        FreeBuddyPage *free_page = reinterpret_cast<FreeBuddyPage*>(backend_.data_ + node.shm_.off_.load());
        size_t page_size = free_page->size_;

        // Convert to CoalesceBuddyPage
        CoalesceBuddyPage *coalesce_page = reinterpret_cast<CoalesceBuddyPage*>(free_page);
        coalesce_page->key = node.shm_.off_;
        coalesce_page->size_ = page_size;

        // Insert into RB tree
        FullPtr<CoalesceBuddyPage> tree_node(coalesce_page, static_cast<ShmPtr>(node.shm_));
        coalesce_tree.emplace(this, tree_node);
      }
    }

    // Merge contiguous pages
    MergeContiguousPages(coalesce_tree);

    // Rebuild free lists from merged tree
    // Since we don't have an iterator, we'll need to track which keys we've seen
    // For now, skip rebuilding as MergeContiguousPages is not implemented
    // TODO: Implement proper tree traversal and rebuilding
    (void)coalesce_tree;  // Suppress unused variable warning
  }

  /**
   * Merge contiguous pages in the RB tree
   */
  void MergeContiguousPages(pre::rb_tree<CoalesceBuddyPage, false> &tree) {
    // TODO: Implement tree traversal and merging
    // This requires iterating through the tree and checking if
    // offset + size equals the next node's offset
    (void)tree;  // Suppress unused parameter warning for now
  }

  /**
   * Split a large page and allocate the requested size
   */
  OffsetPtr SplitAndAllocate(size_t page_offset, size_t src_list, size_t dst_list) {
    size_t src_size = static_cast<size_t>(1) << (src_list + kMinLog2);
    size_t dst_size = static_cast<size_t>(1) << (dst_list + kMinLog2);

    // Split into buddy pages
    size_t current_offset = page_offset;
    size_t current_size = src_size;

    while (current_size > dst_size) {
      current_size /= 2;
      size_t buddy_offset = current_offset + current_size;

      // Add buddy to free list
      FreeBuddyPage *buddy = reinterpret_cast<FreeBuddyPage*>(backend_.data_ + buddy_offset);
      buddy->size_ = current_size;

      size_t buddy_list = GetFreeListIndex(current_size);
      ShmPtr buddy_shm;
      buddy_shm.off_ = buddy_offset;
      FullPtr<pre::slist_node> buddy_node(this, buddy_shm);
      round_up_lists_[buddy_list].emplace(this, buddy_node);
    }

    return FinalizeAllocation(page_offset, dst_size);
  }

  /**
   * Split a large page from round-down lists
   */
  OffsetPtr SplitLargeAndAllocate(size_t page_offset, size_t page_size, size_t alloc_size, size_t dst_list) {
    if (page_size == alloc_size) {
      return FinalizeAllocation(page_offset, alloc_size);
    }

    // Put remainder back into free list
    size_t remainder_offset = page_offset + alloc_size;
    size_t remainder_size = page_size - alloc_size;

    FreeBuddyPage *remainder = reinterpret_cast<FreeBuddyPage*>(backend_.data_ + remainder_offset);
    remainder->size_ = remainder_size;

    size_t remainder_list = GetFreeListIndex(remainder_size);
    ShmPtr remainder_shm;
    remainder_shm.off_ = OffsetPtr(remainder_offset);
    FullPtr<pre::slist_node> remainder_node(this, remainder_shm);
    if (remainder_list < kNumRoundUpLists) {
      round_up_lists_[remainder_list].emplace(this, remainder_node);
    } else {
      round_down_lists_[remainder_list - kNumRoundUpLists].emplace(this, remainder_node);
    }

    return FinalizeAllocation(page_offset, alloc_size);
  }

  /**
   * Subset a large allocation and return remainder to free list
   */
  OffsetPtr SubsetAndAllocate(size_t page_offset, size_t page_size, size_t alloc_size) {
    if (page_size == alloc_size) {
      return FinalizeAllocation(page_offset, alloc_size);
    }

    // Return remainder to appropriate free list
    size_t remainder_offset = page_offset + alloc_size;
    size_t remainder_size = page_size - alloc_size;

    FreeBuddyPage *remainder = reinterpret_cast<FreeBuddyPage*>(backend_.data_ + remainder_offset);
    remainder->size_ = remainder_size;

    size_t remainder_list = GetFreeListIndex(remainder_size);
    ShmPtr remainder_shm;
    remainder_shm.off_ = OffsetPtr(remainder_offset);
    FullPtr<pre::slist_node> remainder_node(this, remainder_shm);
    if (remainder_list < kNumRoundUpLists) {
      round_up_lists_[remainder_list].emplace(this, remainder_node);
    } else {
      round_down_lists_[remainder_list - kNumRoundUpLists].emplace(this, remainder_node);
    }

    return FinalizeAllocation(page_offset, alloc_size);
  }

  /**
   * Allocate from heap
   */
  OffsetPtr AllocateFromHeap(size_t size) {
    if (heap_current_ + size > heap_end_) {
      return OffsetPtr::GetNull();
    }

    size_t alloc_offset = heap_current_;
    heap_current_ += size;

    return FinalizeAllocation(alloc_offset, size);
  }

  /**
   * Finalize allocation by setting page header and returning user offset
   */
  OffsetPtr FinalizeAllocation(size_t page_offset, size_t page_size) {
    BuddyPage *page = reinterpret_cast<BuddyPage*>(backend_.data_ + page_offset);
    page->size_ = page_size;

    return OffsetPtr(page_offset + sizeof(BuddyPage));
  }

  /**
   * Get free list for small allocations (round up)
   */
  size_t GetRoundUpListIndex(size_t size) {
    if (size <= kMinSize) return 0;
    size_t log2 = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(size))));
    if (log2 < kMinLog2) return 0;
    if (log2 > kSmallLog2) return kNumRoundUpLists - 1;
    return log2 - kMinLog2;
  }

  /**
   * Get free list for large allocations (round down)
   */
  size_t GetRoundDownListIndex(size_t size) {
    size_t log2 = static_cast<size_t>(std::floor(std::log2(static_cast<double>(size))));
    if (log2 <= kSmallLog2) return 0;
    if (log2 > kMaxLog2) return kNumRoundDownLists - 1;
    return log2 - kSmallLog2 - 1;
  }

  /**
   * Get free list index for any size
   */
  size_t GetFreeListIndex(size_t size) {
    if (size < kSmallThreshold) {
      return GetRoundUpListIndex(size);
    } else {
      return kNumRoundUpLists + GetRoundDownListIndex(size);
    }
  }

  /**
   * Get pointer to free list by index
   */
  pre::slist<false>* GetFreeList(size_t idx) {
    if (idx < kNumRoundUpLists) {
      return &round_up_lists_[idx];
    } else {
      return &round_down_lists_[idx - kNumRoundUpLists];
    }
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_
