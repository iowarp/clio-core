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

#ifndef HSHM_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/memory/allocator/heap.h"
#include "hermes_shm/data_structures/ipc/slist_pre.h"
#include "hermes_shm/data_structures/ipc/rb_tree_pre.h"
#include <cmath>

namespace hshm::ipc {

/**
 * Metadata stored after each allocation
 *
 * NOTE: This structure acts as both an allocation header AND a free list node.
 * When allocated: next_ is unused, size_ holds the data size (excluding header)
 * When free: next_ links to next free page in slist, size_ holds data size (excluding header)
 * This structure is 16 bytes to accommodate slist_node requirements.
 */
struct BuddyPage : public pre::slist_node {
  size_t size_;  /**< Size of data portion (always excludes BuddyPage header) */

  BuddyPage() : pre::slist_node(), size_(0) {}
  explicit BuddyPage(size_t size) : pre::slist_node(), size_(size) {}
};

/**
 * Free page node for small allocations (<16KB)
 * NOTE: Now a simple type alias for BuddyPage since BuddyPage inherits from slist_node
 */
using FreeSmallBuddyPage = BuddyPage;

/**
 * Free page node for large allocations (>16KB)
 * NOTE: Now a simple type alias for BuddyPage since BuddyPage inherits from slist_node
 */
using FreeLargeBuddyPage = BuddyPage;

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
 * Buddy allocator using power-of-two free lists
 *
 * This allocator manages memory using segregated free lists for different
 * size classes. Small allocations (<16KB) use round-up sizing, while large
 * allocations (>16KB) use round-down sizing with best-fit search.
 */
class _BuddyAllocator : public Allocator {
 private:
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

  pre::slist<BuddyPage, false> small_pages_[kMaxSmallPages];   /**< Free lists for sizes 32B - 16KB */
  pre::slist<BuddyPage, false>
      large_pages_[kMaxLargePages]; /**< Free lists for sizes 16KB - 1MB */
  pre::slist<BuddyPage, false> regions_;   /**< List of big_heap_ regions */

  // _MultiProcessAllocator needs access to reconstruct pointers when attaching
  friend class _MultiProcessAllocator;

 public:
  /**
   * Initialize the buddy allocator
   *
   * @param backend Memory backend
   * @param region_size Size of the region in bytes. If 0, defaults to backend.data_capacity_
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, size_t region_size = 0) {
    // Store backend
    SetBackend(backend);
    alloc_header_size_ = sizeof(_BuddyAllocator);

    // Calculate and store the offset of this allocator object within the
    // backend data This must be calculated BEFORE any GetBackendData() calls
    this_ = reinterpret_cast<char *>(this) -
            reinterpret_cast<char *>(backend.data_);

    // Default region_size to data_capacity_ if not specified
    if (region_size == 0) {
      region_size = backend.data_capacity_ - this_;
    }

    // Store region_size for use in GetAllocatorDataSize()
    region_size_ = region_size;

    // Calculate data_start_ - where the allocator's managed region begins
    // For BuddyAllocator, data starts immediately after the allocator object
    data_start_ = sizeof(_BuddyAllocator);

    if (region_size < kMinSize) {
      return false;  // Not enough space
    }

    // Initialize all free lists
    for (size_t i = 0; i < kMaxSmallPages; ++i) {
      small_pages_[i].Init();
    }
    for (size_t i = 0; i < kMaxLargePages; ++i) {
      large_pages_[i].Init();
    }

    // Big heap gets all available space initially
    Expand(OffsetPtr<>(GetAllocatorDataOff()), GetAllocatorDataSize());

    // Small arena is initially empty - will be populated on first small allocation
    small_arena_.Init(0, 0);

    return true;
  }

  /**
   * Attach to an existing buddy allocator
   *
   * BuddyAllocator does not require any process-specific initialization,
   * so this is a no-op. The allocator state is fully shared.
   *
   * @param backend Memory backend (unused)
   * @return true (always succeeds)
   */
  bool shm_attach(const MemoryBackend &backend) {
    (void)backend;  // Unused - no process-specific initialization needed
    return true;
  }

  /**
   * Allocate memory of specified size
   *
   * @param requested_size Size in bytes to allocate (excluding BuddyPage header)
   * @return Offset pointer to allocated memory (after BuddyPage header)
   */
  OffsetPtr<> AllocateOffset(size_t requested_size) {
    if (requested_size < kMinSize) {
      requested_size = kMinSize;
    }
    // Round up to BuddyPage alignment so every heap bump stays aligned
    constexpr size_t kAlign = alignof(BuddyPage);
    requested_size = (requested_size + kAlign - 1) & ~(kAlign - 1);

    OffsetPtr<> ptr;
    if (requested_size <= kSmallThreshold) {
      ptr = AllocateSmall(requested_size);
    } else {
      ptr = AllocateLarge(requested_size);
    }

    return ptr;
  }

  /**
   * Reallocate previously allocated memory to a new size
   *
   * @param offset Offset pointer to existing memory (after BuddyPage header)
   * @param new_size New size in bytes (excluding BuddyPage header)
   * @return Offset pointer to reallocated memory (may be same or different location)
   */
  OffsetPtr<> ReallocateOffset(OffsetPtr<> offset, size_t new_size) {
    // Handle null pointer case
    if (offset.IsNull()) {
      return AllocateOffset(new_size);
    }

    // Get the BuddyPage header (offset points after header)
    size_t page_offset = offset.load() - sizeof(BuddyPage);
    hipc::FullPtr<BuddyPage> page(this, OffsetPtr<BuddyPage>(page_offset));
    size_t old_size = page.ptr_->size_;  // Size without header

    // If new size fits in existing allocation, reuse it
    if (new_size <= old_size) {
      return offset;
    }

    // Allocate new memory
    OffsetPtr<> new_offset = AllocateOffset(new_size);
    if (new_offset.IsNull()) {
      return new_offset;  // Allocation failed
    }

    // Copy old data to new location
    hipc::FullPtr<char> old_data(this, OffsetPtr<char>(offset.load()));
    hipc::FullPtr<char> new_data(this, OffsetPtr<char>(new_offset.load()));
    memcpy(new_data.ptr_, old_data.ptr_, old_size);

    // Free old allocation
    FreeOffset(offset);

    return new_offset;
  }

  /**
   * Free previously allocated memory
   *
   * @param offset Offset pointer to memory (after BuddyPage header)
   */
  void FreeOffset(OffsetPtr<> offset) {
    if (offset.IsNull()) {
      return;
    }
    FreeOffsetNoNullCheck(offset);
  }

  /**
   * Free previously allocated memory (without null check)
   *
   * @param offset Offset pointer to memory (after BuddyPage header) - must not be null
   */
  void FreeOffsetNoNullCheck(OffsetPtr<> offset) {
    // Get the BuddyPage header (offset points after header)
    size_t page_offset = offset.load() - sizeof(BuddyPage);
    hipc::FullPtr<BuddyPage> page(this, OffsetPtr<BuddyPage>(page_offset));
    size_t data_size = page.ptr_->size_;  // Size without header

    // Determine which free list to add to based on data size
    if (data_size <= kSmallThreshold) {
      // Small page - add to small_pages_ list using exact size match
      size_t list_idx = GetSmallPageListIndexForFree(data_size);

      // Initialize BuddyPage as a free list node
      hipc::FullPtr<BuddyPage> free_page(this, OffsetPtr<BuddyPage>(page_offset));
      free_page.ptr_->next_ = OffsetPtr<>::GetNull();  // Initialize slist_node
      // size_ already contains data_size from the allocation, keep it as-is

      // Add to free list - BuddyPage inherits from slist_node
      FullPtr<BuddyPage> node_ptr(this, OffsetPtr<BuddyPage>(page_offset));
      small_pages_[list_idx].emplace(this, node_ptr);
    } else {
      // Large page - add to large_pages_ list
      size_t list_idx = GetLargePageListIndexForFree(data_size);

      // Initialize BuddyPage as a free list node
      hipc::FullPtr<BuddyPage> free_page(this, OffsetPtr<BuddyPage>(page_offset));
      free_page.ptr_->next_ = OffsetPtr<>::GetNull();  // Initialize slist_node
      // size_ already contains data_size from the allocation, keep it as-is

      // Add to free list - BuddyPage inherits from slist_node
      FullPtr<BuddyPage> node_ptr(this, OffsetPtr<BuddyPage>(page_offset));
      large_pages_[list_idx].emplace(this, node_ptr);
    }
  }

  /**
   * Expand the allocator with new memory region
   *
   * This method expands the big_heap_ to include a new memory region,
   * making it available for allocation and arena repopulation.
   *
   * @param region Offset pointer to the new region
   * @param region_size Size of the new region in bytes
   */
  void Expand(OffsetPtr<> region, size_t region_size) { 
    if (region.IsNull() || region_size == 0) {
      return;
    }
    FullPtr<BuddyPage> node(this, OffsetPtr<BuddyPage>(region.load()));
    node->size_ = region_size;
    regions_.emplace(this, node);
    region += sizeof(BuddyPage);
    region_size -= sizeof(BuddyPage);
    DivideArenaIntoPages(big_heap_);
    big_heap_.Init(region.load(),
                                        region.load() + region_size);
  }

 private:
  /**
   * Allocate small memory (<16KB) using round-up sizing
   */
  OffsetPtr<> AllocateSmall(size_t size) {
    // Step 1: Get the free list for this size (round up to next largest)
    // This also modifies size to be the rounded-up value
    size_t list_idx = GetSmallPageListIndexForAlloc(size);

    // Step 2: Check if page exists in this free list
    if (!small_pages_[list_idx].empty()) {
      auto node = small_pages_[list_idx].pop(this);
      if (!node.IsNull()) {
        return FinalizeAllocation(node.shm_.off_.load(), size);
      }
      // If pop returned null despite list not being empty, fall through to next step
    }

    // Step 3: Try allocating from small_arena_
    size_t total_size = size + sizeof(BuddyPage);
    size_t arena_offset = small_arena_.Allocate(total_size);
    if (arena_offset != 0) {
      return FinalizeAllocation(arena_offset, size);
    }

    // Step 4: Repopulate the small arena
    if (RepopulateSmallArena()) {
      // After repopulating, the pages are in free lists, not in the arena
      // Retry allocation from the free list
      if (!small_pages_[list_idx].empty()) {
        auto node = small_pages_[list_idx].pop(this);
        if (!node.IsNull()) {
          return FinalizeAllocation(node.shm_.off_.load(), size);
        }
      }
    }

    // Step 5: Allocate directly from big_heap_ if all else fails
    // This handles the case where free lists and arena don't have suitable pages
    size_t heap_offset = big_heap_.Allocate(total_size);
    if (heap_offset != 0) {
      return FinalizeAllocation(heap_offset, size);
    }

    // Step 6: Out of memory
    return OffsetPtr<>::GetNull();
  }

  /**
   * Allocate large memory (>16KB) using round-down sizing with best-fit
   */
  OffsetPtr<> AllocateLarge(size_t size) {
    size_t total_size = size + sizeof(BuddyPage);

    // Step 1: Identify the free list using round down
    size_t list_idx = GetLargePageListIndexForAlloc(size);

    // Step 2: Check each entry in this list for a fit
    for (size_t i = list_idx; i < kMaxLargePages; ++i) {
        size_t found_offset = FindFirstFit(list_idx, total_size);
        if (found_offset != 0) {
        hipc::FullPtr<FreeLargeBuddyPage> free_page(this, OffsetPtr<FreeLargeBuddyPage>(found_offset));
        size_t page_data_size = free_page.ptr_->size_;
        size_t page_total_size = page_data_size + sizeof(BuddyPage);

        // If there's remainder, add it back to appropriate list using exact size
        if (page_total_size > total_size) {
            AddRemainderToFreeList(found_offset + total_size, page_total_size - total_size);
        }

        return FinalizeAllocation(found_offset, size);
        }
    }
    
    // Step 4: Try allocating from heap
    size_t heap_offset = big_heap_.Allocate(total_size);
    if (heap_offset != 0) {
      return FinalizeAllocation(heap_offset, size);
    }

    // Step 5: Out of memory
    return OffsetPtr<>::GetNull();
  }

  /**
   * Find the first fit in a large page free list
   *
   * Iterates through all entries in the specified free list to find the first
   * entry with size >= required_size. Removes the entry from the list using PopAt.
   *
   * @param list_idx Index of the free list to search
   * @param required_size Minimum size required (including BuddyPage header)
   * @return Offset to the found page, or 0 if no fit found
   */
  size_t FindFirstFit(size_t list_idx, size_t required_size) {
    if (large_pages_[list_idx].empty()) {
      return 0;
    }

    // Iterate through all entries in the list using the iterator
    for (auto it = large_pages_[list_idx].begin(this);
         it != large_pages_[list_idx].end(); ++it) {
      hipc::FullPtr<FreeLargeBuddyPage> free_page(
          this, OffsetPtr<FreeLargeBuddyPage>(it.GetCurrent().load()));

      // Check if this page size (including header) is large enough
      size_t page_total_size = free_page.ptr_->size_ + sizeof(BuddyPage);
      if (page_total_size >= required_size) {
        // Found a fit - capture offset before removing from list
        size_t offset = it.GetCurrent().load();
        // Remove from list using PopAt (iterator is invalidated after this)
        (void)large_pages_[list_idx].PopAt(this, it);
        return offset;
      }
    }

    return 0;  // No fit found
  }

  /**
   * Repopulate small arena with more space
   * @return true if arena was successfully repopulated
   */
  bool RepopulateSmallArena() {
    size_t arena_size = kSmallArenaSize + kSmallArenaPages * sizeof(BuddyPage);

    // Divide small arena into pages
    DivideArenaIntoPages(small_arena_);

    // Step 4.2.1: Try allocating from big_heap_ first
    size_t heap_offset = big_heap_.Allocate(arena_size);

    if (heap_offset != 0) {
      // Step 4.1: Set arena bounds
      small_arena_.Init(heap_offset, heap_offset + arena_size);

      return true;
    }

    // Step 4.2.2: If heap fails, search all large_pages_ lists for space
    {
      for (size_t list_idx = 0; list_idx < kMaxLargePages; ++list_idx) {
        size_t offset = FindFirstFit(list_idx, arena_size);
        if (offset != 0) {
          small_arena_.Init(offset, offset + arena_size);
          return true;
        }
      }
    }

    return false;  // Could not repopulate
  }

  /**
   * Divide the current small_arena_ into pages using greedy algorithm
   *
   * This function operates on the current small_arena_ state and divides
   * the arena space into free pages, adding them to the appropriate free lists.
   * After this function, the arena is marked as fully consumed.
   */
  void DivideArenaIntoPages(Heap<false> &heap) {
    // Get the arena bounds from the heap
    // After Init(), heap_ points to the beginning and hasn't moved yet
    size_t arena_begin = heap.GetOffset();
    size_t arena_end = heap.GetMaxOffset();
    size_t remaining_offset = arena_begin;
    size_t remaining_size = arena_end - arena_begin;

    // Check if arena is empty
    if (remaining_size == 0) {
      // Arena is empty, nothing to divide
      return;
    }

    // Greedy algorithm: divide by largest page sizes first
    // Start from largest small page (16KB) down to smallest (32B)
    for (int i = static_cast<int>(kMaxSmallPages) - 1; i >= 0; --i) {
      size_t page_data_size = static_cast<size_t>(1) << (i + kMinLog2);  // 2^(i+5)
      size_t page_total_size = page_data_size + sizeof(BuddyPage);

      while (remaining_size >= page_total_size) {
        // Create a free page
        hipc::FullPtr<BuddyPage> free_page(this, OffsetPtr<BuddyPage>(remaining_offset));
        if (free_page.IsNull()) {
          break;
        }
        free_page.ptr_->next_ = OffsetPtr<>::GetNull();  // Initialize slist_node
        free_page.ptr_->size_ = page_data_size;  // Data size excluding header

        // Add to free list
        FullPtr<BuddyPage> node_ptr(this, OffsetPtr<BuddyPage>(remaining_offset));
        small_pages_[i].emplace(this, node_ptr);

        remaining_offset += page_total_size;
        remaining_size -= page_total_size;
      }
    }

    // Any remaining space smaller than kMinSize + sizeof(BuddyPage) is wasted

    // Mark arena as fully consumed by setting it to point to the end
    heap.Init(arena_end, arena_end);
  }

  /**
   * Add a remainder page back to the appropriate free list
   *
   * @param page_offset Offset to the remainder page
   * @param total_size Total size of remainder (including BuddyPage header)
   */
  void AddRemainderToFreeList(size_t page_offset, size_t total_size) {
    size_t data_size = total_size - sizeof(BuddyPage);

    if (data_size <= kSmallThreshold) {
      // Small remainder - use exact size match (round down)
      size_t rem_list_idx = GetSmallPageListIndexForFree(data_size);

      hipc::FullPtr<BuddyPage> remainder(this, OffsetPtr<BuddyPage>(page_offset));
      remainder.ptr_->next_ = OffsetPtr<>::GetNull();  // Initialize slist_node
      remainder.ptr_->size_ = data_size;  // Data size excluding header

      FullPtr<BuddyPage> rem_node(this, OffsetPtr<BuddyPage>(page_offset));
      small_pages_[rem_list_idx].emplace(this, rem_node);
    } else {
      // Large remainder - use exact size match
      size_t rem_list_idx = GetLargePageListIndexForFree(data_size);

      hipc::FullPtr<BuddyPage> remainder(this, OffsetPtr<BuddyPage>(page_offset));
      remainder.ptr_->next_ = OffsetPtr<>::GetNull();  // Initialize slist_node
      remainder.ptr_->size_ = data_size;  // Data size excluding header

      FullPtr<BuddyPage> rem_node(this, OffsetPtr<BuddyPage>(page_offset));
      large_pages_[rem_list_idx].emplace(this, rem_node);
    }
  }

  /**
   * Finalize allocation by setting page header and returning user offset
   *
   * @param page_offset Offset to the page (including BuddyPage header)
   * @param user_size Size of the data portion (excluding BuddyPage header)
   * @return Offset pointer to usable memory (after BuddyPage header)
   */
  OffsetPtr<> FinalizeAllocation(size_t page_offset, size_t user_size) {
    hipc::FullPtr<BuddyPage> page(this, OffsetPtr<BuddyPage>(page_offset));
    page.ptr_->size_ = user_size;  // Store size without header

    size_t result_offset = page_offset + sizeof(BuddyPage);
    return OffsetPtr<>(result_offset);
  }

  /**
   * Get free list index for small allocations when allocating (round up to next largest)
   *
   * @param alloc_size Reference to size - will be modified to the rounded-up power-of-2 size
   * @return Index into the small_pages_ array
   */
  static size_t GetSmallPageListIndexForAlloc(size_t &alloc_size) {
    if (alloc_size <= kMinSize) {
      alloc_size = kMinSize;
      return 0;
    }

    // Round up to next power of 2
    size_t log2 = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(alloc_size))));

    if (log2 < kMinLog2) {
      alloc_size = kMinSize;
      return 0;
    }
    if (log2 > kSmallLog2) {
      alloc_size = kSmallThreshold;
      return kMaxSmallPages - 1;
    }

    // Calculate the rounded-up size
    alloc_size = static_cast<size_t>(1) << log2;  // 2^log2
    return log2 - kMinLog2;
  }

  /**
   * Get free list index for small pages when freeing (round down to exact or next smallest)
   */
  static size_t GetSmallPageListIndexForFree(size_t size) {
    if (size <= kMinSize) {
      return 0;
    }

    // Round down to exact power of 2
    size_t log2 = static_cast<size_t>(std::floor(std::log2(static_cast<double>(size))));

    if (log2 < kMinLog2) {
      return 0;
    }
    if (log2 > kSmallLog2) {
      return kMaxSmallPages - 1;
    }

    return log2 - kMinLog2;
  }

  /**
   * Get free list index for large allocations when allocating (round down)
   */
  static size_t GetLargePageListIndexForAlloc(size_t size) {
    if (size <= kSmallThreshold) {
      return 0;
    }

    // Round down to previous power of 2
    size_t log2 = static_cast<size_t>(std::floor(std::log2(static_cast<double>(size))));

    if (log2 <= kSmallLog2) {
      return 0;
    }
    if (log2 > kMaxLog2) {
      return kMaxLargePages - 1;
    }

    return log2 - kSmallLog2 - 1;
  }

  /**
   * Get free list index for large pages when freeing (round down to exact)
   */
  size_t GetLargePageListIndexForFree(size_t size) {
    return GetLargePageListIndexForAlloc(size);  // Same logic for large pages
  }
};

/** Typedef for the complete BuddyAllocator with BaseAllocator wrapper */
using BuddyAllocator = BaseAllocator<_BuddyAllocator>;

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_BUDDY_ALLOCATOR_H_
