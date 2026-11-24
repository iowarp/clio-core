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

#ifndef CTP_MEMORY_ALLOCATOR_SLAB_ALLOCATOR_H_
#define CTP_MEMORY_ALLOCATOR_SLAB_ALLOCATOR_H_

#include "clio_ctp/memory/allocator/allocator.h"

namespace ctp::ipc {

/**
 * High-performance slab allocator with O(1) alloc/free.
 *
 * Uses a bump pointer for initial allocation and per-size-class free stacks
 * for recycling. Each allocation has a small header (SlabHeader) that stores
 * the size class index, enabling O(1) free without any lookup.
 *
 * Size classes (data size, excluding header):
 *   0: 32B, 1: 64B, 2: 128B, 3: 256B, 4: 512B,
 *   5: 1024B, 6: 2048B, 7: 4096B
 *
 * For allocations > 4096B, falls back to bump pointer (no recycling).
 *
 * Designed for GPU where BuddyAllocator's free list traversal is expensive
 * (~950 clocks). Slab alloc/free should be ~10-30 clocks (stack push/pop).
 */
class PrivateSlabAllocator : public Allocator {
 public:
  /**
   * Per-allocation header. Minimal: just stores size for free-list routing.
   * When free, first 8 bytes of the data region are repurposed as next pointer.
   */
  struct SlabHeader {
    size_t size_;  /**< Data size (excluding this header). Bit 63 = free flag. */

    static constexpr size_t kFreeMask = (size_t)1 << (sizeof(size_t) * 8 - 1);

    CTP_INLINE_CROSS_FUN void Init(size_t data_size) { size_ = data_size; }
    CTP_INLINE_CROSS_FUN size_t GetSize() const { return size_ & ~kFreeMask; }
    CTP_INLINE_CROSS_FUN void MarkFree() { size_ |= kFreeMask; }
    CTP_INLINE_CROSS_FUN void MarkAllocated() { size_ &= ~kFreeMask; }
    CTP_INLINE_CROSS_FUN bool IsFree() const { return (size_ & kFreeMask) != 0; }
  };

  static constexpr size_t kHeaderSize = 16;  // Padded to 16 for alignment
  static_assert(sizeof(SlabHeader) <= kHeaderSize, "SlabHeader too large");

  // Size classes: 32, 64, 128, 256, 512, 1024, 2048, 4096
  static constexpr int kNumClasses = 8;
  static constexpr size_t kMinLog2 = 5;   // log2(32)
  static constexpr size_t kMaxClassSize = 4096;
  static constexpr size_t kRingCap = 1024;  // single-thread fast path

  /**
   * Free stack entry. When a slot is free, we store a pointer to the next
   * free slot in the data region (right after SlabHeader).
   */
  struct FreeNode {
    FreeNode *next_;
  };

  /**
   * Single-threaded ring buffer of freed offsets (per size class).
   * Stored as a fixed-size circular buffer in the slab-managed region.
   */
  struct FreeRing {
    size_t *buf_;  /**< pointer to ring storage (size kRingCap) */
    size_t head_;  /**< pop index */
    size_t tail_;  /**< push index */

    CTP_INLINE_CROSS_FUN void Init(size_t *buf) {
      buf_ = buf;
      head_ = 0;
      tail_ = 0;
    }

    CTP_INLINE_CROSS_FUN bool Empty() const { return head_ == tail_; }

    CTP_INLINE_CROSS_FUN bool Full() const {
      return ((tail_ + 1) % kRingCap) == head_;
    }

    CTP_INLINE_CROSS_FUN bool Push(size_t off) {
      if (Full()) {
        return false;
      }
      buf_[tail_] = off;
      tail_ = (tail_ + 1) % kRingCap;
      return true;
    }

    CTP_INLINE_CROSS_FUN bool Pop(size_t &off) {
      if (Empty()) {
        return false;
      }
      off = buf_[head_];
      head_ = (head_ + 1) % kRingCap;
      return true;
    }
  };

 private:
  FreeNode *free_lists_[kNumClasses];  /**< Per-class free stacks */
  FreeRing free_rings_[kNumClasses];   /**< Per-class fixed ring buffer */
  size_t bump_;      /**< Current bump pointer offset (relative to backend base) */
  size_t bump_end_;  /**< End of managed region */
  char * __restrict__ base_;       /**< Cached base pointer */

 public:
  CTP_CROSS_FUN
  PrivateSlabAllocator() : bump_(0), bump_end_(0), base_(nullptr) {
    for (int i = 0; i < kNumClasses; ++i) {
      free_lists_[i] = nullptr;
      free_rings_[i].Init(nullptr);
    }
  }

  /**
   * Initialize the slab allocator.
   *
   * @param backend Memory backend
   * @param region_size Size of region (0 = entire backend minus header)
   */
  CTP_CROSS_FUN
  bool shm_init(const MemoryBackend &backend, size_t region_size = 0) {
    SetBackend(backend);
    alloc_header_size_ = sizeof(PrivateSlabAllocator);
    this_ = reinterpret_cast<char *>(this) -
            reinterpret_cast<char *>(backend.data_);
    base_ = reinterpret_cast<char *>(backend.data_);

    if (region_size == 0) {
      region_size = backend.data_capacity_ - this_;
    }
    region_size_ = region_size;
    data_start_ = sizeof(PrivateSlabAllocator);

    // Bump pointer starts right after the allocator header
    bump_ = this_ + sizeof(PrivateSlabAllocator);
    bump_end_ = this_ + region_size;

    for (int i = 0; i < kNumClasses; ++i) {
      free_lists_[i] = nullptr;
      free_rings_[i].Init(nullptr);
    }

    // Reserve ring storage up-front (single-thread fast free-list).
    // This avoids pointer chasing through freed objects on GPU.
    for (int cls = 0; cls < kNumClasses; ++cls) {
      size_t ring_bytes = kRingCap * sizeof(size_t);
      ring_bytes = (ring_bytes + 15) & ~(size_t)15;
      if (bump_ + ring_bytes > bump_end_) {
        break;  // out of space; rings remain disabled for remaining classes
      }
      auto *buf = reinterpret_cast<size_t *>(base_ + bump_);
      bump_ += ring_bytes;
      free_rings_[cls].Init(buf);
    }
    return true;
  }

  /**
   * Allocate memory.
   *
   * Fast path: pop from free list for the size class.
   * Slow path: bump allocate from the remaining region.
   */
  CTP_CROSS_FUN
  OffsetPtr<> AllocateOffset(size_t requested_size) {
    // Round up to minimum and align to 16 bytes
    if (requested_size < 32) {
      requested_size = 32;
    }
    requested_size = (requested_size + 15) & ~(size_t)15;

    int cls = GetSizeClass(requested_size);

    // Fast path: pop from free list (most common after warmup)
    if (cls >= 0) {
      size_t off = 0;
      if (free_rings_[cls].buf_ != nullptr && free_rings_[cls].Pop(off)) {
        SlabHeader *hdr = reinterpret_cast<SlabHeader *>(base_ + off - kHeaderSize);
        hdr->MarkAllocated();
        return OffsetPtr<>(off);
      }
      FreeNode *node = free_lists_[cls];
      if (node != nullptr) {
        free_lists_[cls] = node->next_;
        SlabHeader *hdr = reinterpret_cast<SlabHeader *>(
            reinterpret_cast<char *>(node) - kHeaderSize);
        hdr->MarkAllocated();
        return OffsetPtr<>(reinterpret_cast<char *>(node) - base_);
      }
    }

    // Slow path: bump allocate
    size_t slot_data_size = (cls >= 0) ? ClassToSize(cls) : requested_size;
    size_t total = kHeaderSize + slot_data_size;
    if (bump_ + total > bump_end_) {
      return OffsetPtr<>::GetNull();  // Out of memory
    }

    SlabHeader *hdr = reinterpret_cast<SlabHeader *>(base_ + bump_);
    hdr->Init(slot_data_size);
    size_t data_off = bump_ + kHeaderSize;
    bump_ += total;

    return OffsetPtr<>(data_off);
  }

  /**
   * Free memory (without null check).
   *
   * Reads the size from the header, computes size class, pushes to free stack.
   */
  CTP_CROSS_FUN
  void FreeOffsetNoNullCheck(OffsetPtr<> offset) {
    size_t off = offset.load();
    SlabHeader *hdr = reinterpret_cast<SlabHeader *>(
        base_ + off - kHeaderSize);
    size_t data_size = hdr->GetSize();
    hdr->MarkFree();

    int cls = GetSizeClass(data_size);
    if (cls >= 0) {
      // Prefer ring buffer (single-thread fast path).
      if (free_rings_[cls].buf_ != nullptr &&
          free_rings_[cls].Push(off)) {
        return;
      }

      // Fallback: push to free list — reuse data region as FreeNode
      FreeNode *node = reinterpret_cast<FreeNode *>(base_ + off);
      node->next_ = free_lists_[cls];
      free_lists_[cls] = node;
    }
    // For oversized allocations (cls < 0), memory is leaked.
    // This is acceptable for GPU scratch buffers that are short-lived.
  }

  /**
   * Free memory (null-safe).
   */
  CTP_CROSS_FUN
  void FreeOffset(OffsetPtr<> offset) {
    if (offset.IsNull()) {
      return;
    }
    FreeOffsetNoNullCheck(offset);
  }

  /**
   * Reallocate (not supported — allocate new + copy manually).
   */
  CTP_CROSS_FUN
  OffsetPtr<> ReallocateOffsetNoNullCheck(OffsetPtr<> p, size_t new_size) {
    (void)p; (void)new_size;
    return OffsetPtr<>::GetNull();
  }

  /** Push arena state — bump allocator subset */
  CTP_CROSS_FUN bool PushArenaState(ArenaState &prior, OffsetPtr<> &block,
                                      size_t size) {
    prior.arena_off_ = bump_;
    prior.arena_cur_ = bump_;
    prior.arena_end_ = bump_end_;
    block = AllocateOffset(size);
    if (block.IsNull()) {
      return false;
    }
    // Set up arena within the allocated block
    size_t block_off = block.load();
    prior.arena_off_ = bump_;  // Save state after allocation
    return true;
  }

  /** Pop arena state */
  CTP_CROSS_FUN void PopArenaState(const ArenaState &prior, OffsetPtr<> block) {
    if (!block.IsNull()) {
      FreeOffset(block);
    }
  }

  /** No-op TLS management */
  CTP_CROSS_FUN void CreateTls() {}
  CTP_CROSS_FUN void FreeTls() {}

 private:
  /**
   * Get size class index for a given size (branchless).
   * Returns -1 if size exceeds kMaxClassSize.
   * Uses bit tricks: cls = max(0, ceil_log2(size) - 5)
   */
  CTP_INLINE_CROSS_FUN
  static int GetSizeClass(size_t size) {
    if (size > kMaxClassSize) {
      return -1;
    }
    if (size <= 32) {
      return 0;
    }
    // ceil_log2: find position of highest set bit, +1 if not power of 2
    unsigned int v = static_cast<unsigned int>(size - 1);
    // __clz counts leading zeros from bit 31
#if CTP_IS_GPU
    int log2_val = 31 - __clz(v);
#else
    int log2_val = 31 - __builtin_clz(v);
#endif
    int cls = log2_val - 4;  // subtract (kMinLog2 - 1) = 4
    if (cls < 0) {
      cls = 0;
    }
    if (cls >= kNumClasses) {
      cls = kNumClasses - 1;
    }
    return cls;
  }

  /**
   * Convert size class index to actual slot size.
   */
  CTP_INLINE_CROSS_FUN
  static size_t ClassToSize(int cls) {
    return (size_t)32 << cls;
  }
};

}  // namespace ctp::ipc

#endif  // CTP_MEMORY_ALLOCATOR_SLAB_ALLOCATOR_H_
