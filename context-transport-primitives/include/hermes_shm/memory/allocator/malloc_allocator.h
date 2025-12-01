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

#ifndef HSHM_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_

#include <cstdint>
#include <cstdlib>

#include "allocator.h"
#include "hermes_shm/thread/lock.h"

namespace hshm::ipc {

class _MallocAllocator;
typedef BaseAllocator<_MallocAllocator> MallocAllocator;

struct MallocPage {
  size_t page_size_;
};

struct _MallocAllocatorHeader : public AllocatorHeader {
  HSHM_CROSS_FUN
  _MallocAllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId alloc_id, size_t custom_header_size) {
    AllocatorHeader::Configure(alloc_id, custom_header_size);
  }
};

class _MallocAllocator : public Allocator {
 private:
  _MallocAllocatorHeader *header_;

 public:
  /**
   * Allocator constructor
   * */
  HSHM_CROSS_FUN
  _MallocAllocator() : header_(nullptr) {}

  /**
   * Initialize the allocator in shared memory
   * */
  HSHM_CROSS_FUN
  void shm_init(AllocatorId id, size_t custom_header_size,
                MemoryBackend backend) {
    id_ = id;
    backend_ = backend;
    accel_id_ = -1;
    header_ = ConstructHeader<_MallocAllocatorHeader>(
        malloc(sizeof(_MallocAllocatorHeader) + custom_header_size));
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    header_->Configure(id, custom_header_size);
  }

  /**
   * Attach an existing allocator from shared memory
   * */
  HSHM_CROSS_FUN
  void shm_attach(MemoryBackend backend) {
    HSHM_THROW_ERROR(NOT_IMPLEMENTED, "_MallocAllocator::shm_attach");
  }

  /**
   * Allocate a memory of \a size size with optional alignment.
   * */
  HSHM_CROSS_FUN
  OffsetPtr AllocateOffset(const hipc::MemContext &ctx, size_t size, size_t alignment = 0) {
#if HSHM_IS_HOST
    void *mem;
    if (alignment > 0) {
      mem = SystemInfo::AlignedAlloc(alignment, sizeof(MallocPage) + size);
    } else {
      mem = malloc(sizeof(MallocPage) + size);
    }
    auto page = reinterpret_cast<MallocPage *>(mem);
    page->page_size_ = size;
    header_->AddSize(size);
    return OffsetPtr((size_t)(page + 1));
#else
    return OffsetPtr(0);
#endif
  }

  /**
   * Reallocate \a p pointer to \a new_size new size.
   *
   * @return whether or not the pointer p was changed
   * */
  HSHM_CROSS_FUN
  OffsetPtr ReallocateOffsetNoNullCheck(const hipc::MemContext &ctx,
                                            OffsetPtr p, size_t new_size) {
#if HSHM_IS_HOST
    // Get the input page
    auto page =
        reinterpret_cast<MallocPage *>(p.off_.load() - sizeof(MallocPage));
    header_->AddSize(new_size - page->page_size_);

    // Reallocate the input page
    auto new_page = reinterpret_cast<MallocPage *>(
        realloc(page, sizeof(MallocPage) + new_size));
    new_page->page_size_ = new_size;

    // Create the pointer
    return OffsetPtr(size_t(new_page + 1));
#else
    return OffsetPtr(0);
#endif
  }

  /**
   * Free \a ptr pointer. Null check is performed elsewhere.
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const hipc::MemContext &ctx, OffsetPtr p) {
    auto page =
        reinterpret_cast<MallocPage *>(p.off_.load() - sizeof(MallocPage));
    header_->SubSize(page->page_size_);
    free(page);
  }

  /**
   * Get the current amount of data allocated. Can be used for leak
   * checking.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() {
    return (size_t)header_->GetCurrentlyAllocatedSize();
  }

  /**
   * Create a globally-unique thread ID
   * */
  HSHM_CROSS_FUN
  void CreateTls(MemContext &ctx) {}

  /**
   * Free a thread-local memory storage
   * */
  HSHM_CROSS_FUN
  void FreeTls(const hipc::MemContext &ctx) {}
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_
