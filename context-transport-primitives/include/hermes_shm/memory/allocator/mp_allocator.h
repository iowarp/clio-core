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

#ifndef HSHM_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/data_structures/ipc/slist_pre.h"
#include "hermes_shm/thread/lock/mutex.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include <sys/types.h>
#include <unistd.h>

namespace hshm::ipc {

/**
 * Forward declarations
 */
class ProcessBlock;
class _MultiProcessAllocator;

/**
 * Typedef for the complete MultiProcessAllocator with BaseAllocator wrapper
 */
typedef BaseAllocator<_MultiProcessAllocator> MultiProcessAllocator;

/**
 * Per-thread allocator block providing lock-free fast-path allocation.
 *
 * Each thread has its own ThreadBlock with a private BuddyAllocator,
 * enabling concurrent allocations without contention. When a ThreadBlock
 * runs out of memory, it requests expansion from its parent ProcessBlock.
 */
class ThreadBlock : public pre::slist_node {
 public:
  int tid_;                    /**< Thread ID */
  BuddyAllocator alloc_;       /**< Private buddy allocator for this thread (MUST BE LAST) */

  /**
   * Default constructor
   */
  ThreadBlock() : tid_(-1) {}

  /**
   * Initialize the thread block with a memory region
   *
   * @param backend Memory backend from the allocator
   * @param region FullPtr to the memory region for this thread
   * @param region_size Size of the memory region in bytes
   * @param tid Thread ID for this block
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, FullPtr<void> region, size_t region_size, int tid) {
    tid_ = tid;

    // The region already excludes the ThreadBlock header, so just use ShiftTo
    MemoryBackend thread_backend = backend.ShiftTo(region, region_size);

    // Initialize the buddy allocator with the shifted backend
    alloc_.shm_init(thread_backend);
    return true;
  }

  /**
   * Expand this ThreadBlock by freeing a region back to it
   *
   * This is called when memory is returned to the ThreadBlock allocator,
   * typically when a ThreadBlock is freed or when coalescing occurs.
   *
   * @param ptr Offset pointer to memory region to return
   */
  void Expand(OffsetPtr<> ptr) { alloc_.FreeOffset(ptr); }
};

/**
 * Per-process allocator block managing multiple ThreadBlocks.
 *
 * Each process has one ProcessBlock that manages a pool of ThreadBlocks.
 * The ProcessBlock uses a mutex to protect its thread list and can
 * expand by requesting more memory from the global allocator.
 */
class ProcessBlock : public pre::slist_node {
 public:
  int pid_;                    /**< Process ID */
  int tid_count_;              /**< Number of thread blocks allocated */
  hshm::Mutex lock_;           /**< Mutex protecting thread list */
  pre::slist<false> threads_;  /**< List of ThreadBlocks */
  ThreadLocalKey tblock_key_;  /**< TLS key for ThreadBlock* */
  ThreadLocalKey pblock_key_;  /**< TLS key for ProcessBlock* */
  BuddyAllocator alloc_;       /**< Allocator for managing ThreadBlock regions (MUST BE LAST) */

  /**
   * Default constructor
   */
  ProcessBlock() : pid_(-1), tid_count_(0) {}

  /**
   * Set up thread-local storage keys
   * @return true on success, false on failure
   */
  bool SetupTls() {
    void *null_ptr = nullptr;
    if (!HSHM_THREAD_MODEL->CreateTls<void>(tblock_key_, null_ptr)) {
      return false;
    }
    if (!HSHM_THREAD_MODEL->CreateTls<void>(pblock_key_, null_ptr)) {
      return false;
    }
    return true;
  }

  /**
   * Initialize the process block with a memory region
   *
   * @param backend Memory backend from the allocator
   * @param region FullPtr to the memory region for this process
   * @param region_size Size of the memory region in bytes
   * @param pid Process ID
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, FullPtr<void> region, size_t region_size, int pid) {
    pid_ = pid;
    tid_count_ = 0;
    lock_.Init();
    threads_.Init();

    // The region already excludes the ProcessBlock header, so just use ShiftTo
    MemoryBackend process_backend = backend.ShiftTo(region, region_size);

    // Initialize buddy allocator for managing thread blocks
    alloc_.shm_init(process_backend);

    // Set up TLS for this process block
    if (!SetupTls()) {
      return false;
    }

    return true;
  }

  /**
   * Expand this ProcessBlock by freeing a region back to it
   *
   * This is called when memory is returned to the ProcessBlock allocator,
   * typically when a ThreadBlock is freed or when coalescing occurs.
   *
   * @param ptr Offset pointer to memory region to return
   */
  void Expand(OffsetPtr<> ptr) {
    alloc_.FreeOffset(ptr);
  }
};

/**
 * Private header stored in backend's private region (process-local, not shared)
 */
struct MultiProcessAllocatorPrivateHeader {
  ProcessBlock *process_block_;  /**< Pointer to this process's ProcessBlock */

  MultiProcessAllocatorPrivateHeader() : process_block_(nullptr) {}
};

/**
 * Multi-process allocator with thread-local storage for lock-free fast path.
 *
 * Architecture:
 * - Global BuddyAllocator manages the entire shared memory region
 * - ProcessBlocks are allocated per process, each managing ThreadBlocks
 * - ThreadBlocks provide lock-free allocation for individual threads
 *
 * Allocation Strategy (3-tier fallback):
 * 1. Fast path: Allocate from thread-local ThreadBlock (no locks)
 * 2. Medium path: Expand ThreadBlock from ProcessBlock (one lock)
 * 3. Slow path: Expand ProcessBlock from global allocator (global lock)
 *
 * This design minimizes contention and provides malloc-like performance
 * for the common case of thread-local allocations.
 *
 * Memory Layout:
 * The allocator itself is placed at the beginning of shared memory.
 * Header fields (pid_count_, lists, lock) are part of the allocator structure.
 * The BuddyAllocator follows immediately after the allocator header.
 */
class _MultiProcessAllocator : public Allocator {
 public:
  // Header fields (shared memory compatible)
  int pid_count_;                /**< Number of processes */
  pre::slist<false> alloc_procs_;/**< Active ProcessBlocks */
  pre::slist<false> free_procs_; /**< Free ProcessBlocks */
  hshm::Mutex lock_;             /**< Mutex protecting process lists */

  // Allocator state (shared memory compatible)
  size_t process_unit_;          /**< Default ProcessBlock size (1GB) */
  size_t thread_unit_;           /**< Default ThreadBlock size (16MB) */
  BuddyAllocator alloc_;         /**< Global buddy allocator */

 public:
  /**
   * Default constructor
   */
  _MultiProcessAllocator() : pid_count_(0), process_unit_(0), thread_unit_(0) {}

  /**
   * Set the process block for this process
   *
   * Stores the ProcessBlock pointer in the private header region
   * allocated by GetPrivateHeader().
   *
   * @param pblock Pointer to ProcessBlock for current process
   */
  void SetProcessBlock(ProcessBlock *pblock) {
    auto *header = GetPrivateHeader<MultiProcessAllocatorPrivateHeader>();
    if (header != nullptr) {
      header->process_block_ = pblock;
    }
  }

  /**
   * Get the process block for this process
   *
   * Retrieves the ProcessBlock pointer from the private header region.
   *
   * @return Pointer to ProcessBlock for current process, or nullptr if not set
   */
  ProcessBlock* GetProcessBlock() const {
    auto *header = GetPrivateHeader<MultiProcessAllocatorPrivateHeader>();
    if (header != nullptr) {
      return header->process_block_;
    }
    return nullptr;
  }


  /**
   * Initialize the allocator with a new memory region
   *
   * @param backend Memory backend to use (allocator will be placed at backend.data_)
   * @param custom_header_size Size of custom header (default: 0)
   * @param size Total size of the memory region (optional, defaults to backend.data_size_)
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, size_t custom_header_size = 0) {
    SetBackend(backend);
    alloc_header_size_ = sizeof(_MultiProcessAllocator);
    data_start_ = sizeof(_MultiProcessAllocator);

    // Initialize header fields directly (we are the header!)
    pid_count_ = 0;
    lock_.Init();
    alloc_procs_.Init();
    free_procs_.Init();

    // Create a shifted backend for the global allocator
    // It starts after this allocator object
    MemoryBackend alloc_backend = backend.ShiftTo(GetAllocatorDataOff());

    // Initialize global buddy allocator with shifted backend
    alloc_.shm_init(alloc_backend);

    // Set default sizes based on available memory
    // For testing with smaller memory regions, use smaller defaults
    size_t available_size = backend.data_size_;
    if (available_size < 1ULL * 1024 * 1024 * 1024) {
      // For regions < 1GB, use smaller units
      process_unit_ = available_size / 4;        // Use 1/4 of available space
      thread_unit_ = 4ULL * 1024 * 1024;         // 4MB
    } else {
      process_unit_ = 1ULL * 1024 * 1024 * 1024;  // 1GB
      thread_unit_ = 16ULL * 1024 * 1024;         // 16MB
    }

    // Allocate first ProcessBlock using AllocateProcessBlock()
    FullPtr<ProcessBlock> pblock_ptr = AllocateProcessBlock();
    if (pblock_ptr.IsNull()) {
      return false;
    }

    return true;
  }

  /**
   * Attach to an existing allocator (for multi-process scenarios)
   *
   * This method allows a process to attach to shared memory that was
   * initialized by another process. It assumes shm_init was previously called
   * by another process. It only allocates a ProcessBlock and sets up the
   * private header for this process.
   *
   * @param backend The memory backend (unused, for API compatibility)
   * @return true on success, false on failure
   */
  bool shm_attach(const MemoryBackend &backend) {
    (void)backend;  // Unused - backend is already set during construction

    // Allocate a ProcessBlock for this process using AllocateProcessBlock()
    FullPtr<ProcessBlock> pblock_ptr = AllocateProcessBlock();
    if (pblock_ptr.IsNull()) {
      return false;
    }

    return true;
  }

  /**
   * Detach from the allocator (cleanup TLS)
   */
  void shm_detach() {
    // TLS cleanup is handled automatically by pthread_key_create destructor
  }

 public:
  /**
   * Ensure thread-local storage is set up for the current thread
   *
   * This is the critical function for lock-free fast path allocation.
   * It checks TLS for a ThreadBlock, and if not found, allocates one
   * from the ProcessBlock (assumes ProcessBlock already exists).
   *
   * Uses SINGLE allocation for both ThreadBlock object and managed region.
   *
   * @return Pointer to the thread's ThreadBlock, or nullptr on failure
   */
  ThreadBlock* EnsureTls() {
    printf("[EnsureTls] Start\n");
    // Get ProcessBlock (should already exist from shm_init or shm_attach)
    ProcessBlock *pblock = GetProcessBlock();
    if (pblock == nullptr) {
      printf("[EnsureTls] No ProcessBlock\n");
      return nullptr;
    }

    // Check if we already have a ThreadBlock in TLS
    void *tblock_data = HSHM_THREAD_MODEL->GetTls<void>(pblock->tblock_key_);
    if (tblock_data != nullptr) {
      printf("[EnsureTls] Found existing ThreadBlock in TLS\n");
      return reinterpret_cast<ThreadBlock*>(tblock_data);
    }

    printf("[EnsureTls] Allocating new ThreadBlock with SINGLE allocation\n");
    // Allocate SINGLE region for both ThreadBlock object and managed region
    FullPtr<ThreadBlock> tblock_ptr = AllocateFromProcessOrGlobal<ThreadBlock>(
        sizeof(ThreadBlock) + thread_unit_);
    if (tblock_ptr.IsNull()) {
      printf("[EnsureTls] Failed to allocate ThreadBlock region\n");
      return nullptr;
    }

    // Construct ThreadBlock object at the region start
    new (tblock_ptr.ptr_) ThreadBlock();

    // Create managed_region starting AFTER the ThreadBlock object
    OffsetPtr<> shifted_off(tblock_ptr.shm_.off_.load() + sizeof(ThreadBlock));
    FullPtr<void> managed_region(this, shifted_off);

    printf("[EnsureTls] Initializing ThreadBlock\n");
    // Initialize ThreadBlock with the managed_region and store in TLS
    tblock_ptr.ptr_->shm_init(GetBackend(), managed_region, thread_unit_, pblock->tid_count_++);
    HSHM_THREAD_MODEL->SetTls<void>(pblock->tblock_key_, reinterpret_cast<void*>(tblock_ptr.ptr_));
    printf("[EnsureTls] ThreadBlock initialized and stored in TLS\n");
    return tblock_ptr.ptr_;
  }

  /**
   * Allocate memory from ProcessBlock or fallback to expanding from global allocator
   *
   * This is a helper function used by EnsureTls to allocate ThreadBlock objects
   * and their data regions. It implements a 2-tier strategy:
   * 1. Try allocating from ProcessBlock (with lock)
   * 2. If that fails, expand ProcessBlock from global allocator and retry
   *
   * @tparam T The type to allocate
   * @param size Size in bytes to allocate
   * @return FullPtr to allocated memory, or null on failure
   */
  template <typename T>
  FullPtr<T> AllocateFromProcessOrGlobal(size_t size) {
    ProcessBlock *pblock = GetProcessBlock();
    if (pblock == nullptr) {
      return FullPtr<T>::GetNull();
    }

    // Tier 1: Try allocating from ProcessBlock (requires lock)
    {
      printf("[AllocFromProcOrGlobal] Acquiring pblock lock (Tier 1)\n");
      ScopedMutex scoped_lock(pblock->lock_, 0);
      printf("[AllocFromProcOrGlobal] Acquired pblock lock (Tier 1)\n");
      FullPtr<T> ptr = pblock->alloc_.Allocate<T>(size);
      if (!ptr.IsNull()) {
        printf("[AllocFromProcOrGlobal] Success (Tier 1), releasing pblock lock\n");
        return ptr;
      }
      printf("[AllocFromProcOrGlobal] Failed (Tier 1), releasing pblock lock\n");
    }

    // Tier 2: Expand ProcessBlock from global allocator and retry
    {
      // First acquire global lock to allocate expansion memory
      printf("[AllocFromProcOrGlobal] Acquiring global lock (Tier 2)\n");
      ScopedMutex global_lock(lock_, 0);
      printf("[AllocFromProcOrGlobal] Acquired global lock (Tier 2)\n");
      OffsetPtr<> expand_ptr = alloc_.AllocateOffset(process_unit_);
      if (expand_ptr.IsNull()) {
        printf("[AllocFromProcOrGlobal] Expand failed, releasing global lock\n");
        return FullPtr<T>::GetNull();
      }

      // Now acquire ProcessBlock lock to expand and retry allocation
      printf("[AllocFromProcOrGlobal] Acquiring pblock lock (Tier 2)\n");
      ScopedMutex pblock_lock(pblock->lock_, 0);
      printf("[AllocFromProcOrGlobal] Acquired pblock lock (Tier 2)\n");
      pblock->Expand(expand_ptr);

      // Retry allocation from expanded ProcessBlock
      FullPtr<T> ptr = pblock->alloc_.Allocate<T>(size);
      printf("[AllocFromProcOrGlobal] Tier 2 complete, releasing locks\n");
      return ptr;  // May still be null if expansion wasn't enough
    }
  }

  /**
   * Allocate a ProcessBlock for the current process
   *
   * This function implements the following logic:
   * 1. Check if there are any free process blocks in the free_procs_ list
   * 2. If found, pop the first one and get the pid from that block
   * 3. Otherwise, allocate a new one with SINGLE allocation:
   *    a. Create a pid using pid_count_++
   *    b. Allocate SINGLE region of (sizeof(ProcessBlock) + process_unit_) size
   *    c. Cast the region start to ProcessBlock*
   *    d. Create managed_region starting after ProcessBlock object
   *    e. Call shm_init on the ProcessBlock, passing the managed_region
   * 4. Add the ProcessBlock to the alloc_procs_ list
   * 5. Call SetProcessBlock(pblock)
   * 6. Return the ProcessBlock pointer
   *
   * @return FullPtr to ProcessBlock, or null FullPtr on failure
   */
  FullPtr<ProcessBlock> AllocateProcessBlock() {
    // Lock the global process list
    printf("[AllocateProcessBlock] Acquiring global lock\n");
    ScopedMutex scoped_lock(lock_, 0);
    printf("[AllocateProcessBlock] Acquired global lock\n");

    FullPtr<ProcessBlock> pblock_ptr = FullPtr<ProcessBlock>::GetNull();
    int pid;

    // Step 1-2: Try to pop from free_procs_ first
    if (!free_procs_.empty()) {
      auto node = free_procs_.pop(&alloc_);
      if (!node.IsNull()) {
        pblock_ptr = node.Cast<ProcessBlock>();
        pid = pblock_ptr.ptr_->pid_;  // Get the pid from the existing block
      }
    }

    // Step 3: If not reused, allocate new ProcessBlock
    if (pblock_ptr.IsNull()) {
      // Step 3a: Create a pid using pid_count_
      pid = pid_count_++;

      // Step 3b: Allocate SINGLE region for both ProcessBlock object and managed region
      pblock_ptr = alloc_.AllocateRegion<ProcessBlock>(process_unit_);
      if (pblock_ptr.IsNull()) {
        return FullPtr<ProcessBlock>::GetNull();
      }

      // Step 3c: Create managed_region starting AFTER the ProcessBlock object
      OffsetPtr<> shifted_off(pblock_ptr.shm_.off_.load() + sizeof(ProcessBlock));
      FullPtr<void> managed_region(this, shifted_off);

      // Step 3d: Call shm_init on the ProcessBlock with the managed_region
      if (!pblock_ptr.ptr_->shm_init(GetBackend(), managed_region, process_unit_, pid)) {
        alloc_.Free(pblock_ptr);  // Free the SINGLE allocation on error
        return FullPtr<ProcessBlock>::GetNull();
      }
    }

    // Step 4: Add to allocated process list
    FullPtr<pre::slist_node> pblock_node = pblock_ptr.Cast<pre::slist_node>();
    alloc_procs_.emplace(&alloc_, pblock_node);

    // Step 5: Call SetProcessBlock(pblock)
    SetProcessBlock(pblock_ptr.ptr_);

    // Step 6: Return the ProcessBlock FullPtr
    return pblock_ptr;
  }

  /**
   * Allocate memory from the multi-process allocator
   *
   * Implements a 3-tier fallback strategy:
   * 1. Fast path: Thread-local ThreadBlock (lock-free)
   * 2. Medium path: ProcessBlock (one lock)
   * 3. Slow path: Global allocator (global lock)
   *
   * @param size Size in bytes to allocate
   * @return Offset pointer to allocated memory, or null on failure
   */
  OffsetPtr<> AllocateOffset(size_t size) {
    OffsetPtr<> ptr = AllocateOffsetFromTblock(size);
    if (!ptr.IsNull()) {
      return ptr;
    }
    ptr = AllocateOffsetFromPblock(size);
    if (!ptr.IsNull()) {
      return ptr;
    }
    ptr = AllocateOffsetFromGlobal(size);
    if (!ptr.IsNull()) {
      return ptr;
    }
    return ptr;
  }

  /**
   * Allocate offset with alignment (alignment parameter ignored)
   *
   * @param size Size in bytes to allocate
   * @param alignment Alignment (ignored - no alignment support)
   * @return OffsetPtr to allocated memory, or null on failure
   */
  OffsetPtr<> AllocateOffset(size_t size, size_t alignment) {
    (void)alignment;  // Alignment not supported
    return AllocateOffset(size);
  }

  /**
   * Reallocate memory (NOT IMPLEMENTED)
   *
   * MultiProcessAllocator does not support reallocation.
   * Users should allocate new memory and manually migrate data.
   *
   * @param p The original offset pointer
   * @param new_size The new size in bytes
   * @return Null offset pointer (reallocation not supported)
   */
  OffsetPtr<> ReallocateOffsetNoNullCheck(OffsetPtr<> p, size_t new_size) {
    (void)p;
    (void)new_size;
    HSHM_THROW_ERROR(NOT_IMPLEMENTED,
                     "MultiProcessAllocator does not support reallocation");
    return OffsetPtr<>();
  }

  /**
   * Free memory allocated from the multi-process allocator
   *
   * Returns the memory to the appropriate allocator tier:
   * - If from ThreadBlock: Returns to ThreadBlock allocator
   * - If from ProcessBlock: Returns to ProcessBlock allocator
   * - If from Global: Returns to global allocator
   *
   * @param p The offset pointer to free
   */
  void FreeOffsetNoNullCheck(OffsetPtr<> p) {
    ProcessBlock *pblock = GetProcessBlock();
    if (pblock == nullptr) {
      return;
    }

    // Try to get ThreadBlock from TLS
    ThreadBlock *tblock = reinterpret_cast<ThreadBlock*>(
        HSHM_THREAD_MODEL->GetTls<void>(pblock->tblock_key_));
    if (tblock == nullptr) {
      return;
    }

    // Free from ThreadBlock allocator
    tblock->alloc_.FreeOffset(p);
  }

  /**
   * Create thread-local storage for the allocator
   *
   * Initializes TLS infrastructure for the current thread.
   * The MultiProcessAllocator uses TLS to store ThreadBlock pointers.
   */
  void CreateTls() {
    // TLS is created on-demand by EnsureTls()
    // No explicit initialization needed here
  }

  /**
   * Free thread-local storage for the allocator
   *
   * Cleans up TLS infrastructure for the current thread.
   */
  void FreeTls() {
    // TLS cleanup is handled by the pthread_key_create destructor
    // No explicit cleanup needed here
  }

  /**
   * Tier 1: Allocate from thread-local ThreadBlock allocator (lock-free fast path)
   *
   * This is the fastest allocation path with no locking required.
   * Attempts to allocate from the current thread's ThreadBlock allocator.
   *
   * @param size Size in bytes to allocate
   * @return Offset pointer to allocated memory, or null if ThreadBlock exhausted
   */
  OffsetPtr<> AllocateOffsetFromTblock(size_t size) {
    ThreadBlock *tblock = EnsureTls();
    if (tblock != nullptr) {
      return tblock->alloc_.AllocateOffset(size);
    }
    return OffsetPtr<>::GetNull();
  }

  /**
   * Tier 2: Allocate from ProcessBlock and expand ThreadBlock (medium path)
   *
   * When ThreadBlock is exhausted, this allocates thread_unit_ bytes from
   * the ProcessBlock allocator, expands the ThreadBlock with this memory,
   * and retries the allocation from the expanded ThreadBlock.
   *
   * Requires ProcessBlock lock.
   *
   * @param size Size in bytes to allocate
   * @return Offset pointer to allocated memory, or null if ProcessBlock exhausted
   */
  OffsetPtr<> AllocateOffsetFromPblock(size_t size) {
    ProcessBlock *pblock = GetProcessBlock();
    ThreadBlock *tblock = EnsureTls();
    if (pblock == nullptr || tblock == nullptr) {
      return OffsetPtr<>::GetNull();
    }

    printf("[AllocOffsetFromPblock] Acquiring pblock lock\n");
    ScopedMutex scoped_lock(pblock->lock_, 0);
    printf("[AllocOffsetFromPblock] Acquired pblock lock\n");
    OffsetPtr<> expand_offset = pblock->alloc_.AllocateOffset(thread_unit_);
    if (!expand_offset.IsNull()) {
      // Expand the thread block and try reallocating
      tblock->Expand(expand_offset);
      OffsetPtr<> ptr = tblock->alloc_.AllocateOffset(size);
      if (!ptr.IsNull()) {
        printf("[AllocOffsetFromPblock] Success, releasing pblock lock\n");
        return ptr;
      }
    }
    printf("[AllocOffsetFromPblock] Failed, releasing pblock lock\n");
    return OffsetPtr<>::GetNull();
  }

  /**
   * Tier 3: Allocate from global allocator and expand ProcessBlock (slow path)
   *
   * When ProcessBlock is exhausted, this allocates process_unit_ bytes from
   * the global allocator, expands the ProcessBlock with this memory, and
   * then retries allocation through the ProcessBlock tier.
   *
   * Requires both global and ProcessBlock locks (acquired sequentially).
   *
   * @param size Size in bytes to allocate
   * @return Offset pointer to allocated memory, or null if global allocator exhausted
   */
  OffsetPtr<> AllocateOffsetFromGlobal(size_t size) {
    ProcessBlock *pblock = GetProcessBlock();
    if (pblock == nullptr) {
      return OffsetPtr<>::GetNull();
    }

    // Acquire global lock to allocate expansion memory
    OffsetPtr<> expand_ptr;
    {
      printf("[AllocOffsetFromGlobal] Acquiring global lock\n");
      ScopedMutex global_lock(lock_, 0);
      printf("[AllocOffsetFromGlobal] Acquired global lock\n");
      expand_ptr = alloc_.AllocateOffset(process_unit_);
      if (expand_ptr.IsNull()) {
        printf("[AllocOffsetFromGlobal] Expand failed, releasing global lock\n");
        return OffsetPtr<>::GetNull();
      }
      printf("[AllocOffsetFromGlobal] Releasing global lock\n");
    }

    // Acquire ProcessBlock lock to expand
    {
      printf("[AllocOffsetFromGlobal] Acquiring pblock lock\n");
      ScopedMutex pblock_lock(pblock->lock_, 0);
      printf("[AllocOffsetFromGlobal] Acquired pblock lock\n");
      pblock->Expand(expand_ptr);
      printf("[AllocOffsetFromGlobal] Releasing pblock lock\n");
    }

    // Retry through ProcessBlock tier (which will expand ThreadBlock if needed)
    printf("[AllocOffsetFromGlobal] Calling AllocateOffsetFromPblock\n");
    return AllocateOffsetFromPblock(size);
  }

  /**
   * Reallocate memory to a new size
   *
   * @param offset Original offset pointer
   * @param new_size New size in bytes
   * @return New offset pointer, or null on failure
   */
  OffsetPtr<> ReallocateOffset(OffsetPtr<> offset, size_t new_size) {
    if (offset.IsNull()) {
      return AllocateOffset(new_size);
    }

    // Try to reallocate from the tblock.
    ThreadBlock *tblock = EnsureTls();
    OffsetPtr<> new_offset = tblock->alloc_.ReallocateOffset(offset, new_size);
    if (!new_offset.IsNull()) {
      return new_offset;
    }

    // Allocate new memory
    new_offset = AllocateOffset(new_size);
    if (new_offset.IsNull()) {
      return new_offset;
    }

    // Get old allocation size by reading the BuddyPage header
    // The page header is located just before the user data
    size_t page_offset = offset.load() - sizeof(BuddyPage);
    BuddyPage *page = reinterpret_cast<BuddyPage*>(GetBackendData() + page_offset);
    size_t old_size = page->size_;

    // Copy old data to new location (copy the minimum of old and new sizes)
    char *old_data = GetBackendData() + offset.load();
    char *new_data = GetBackendData() + new_offset.load();
    size_t copy_size = (old_size < new_size) ? old_size : new_size;
    memcpy(new_data, old_data, copy_size);

    // Free old allocation
    FreeOffset(offset);

    return new_offset;
  }

  /**
   * Free allocated memory
   *
   * @param offset Offset pointer to memory to free
   */
  void FreeOffset(OffsetPtr<> offset) {
    if (offset.IsNull()) {
      return;
    }

    // Determine which allocator owns this memory
    // For simplicity, try thread-local first, then global
    ProcessBlock *pblock = GetProcessBlock();
    if (pblock != nullptr) {
      void *tblock_data = HSHM_THREAD_MODEL->GetTls<void>(pblock->tblock_key_);
      ThreadBlock *tblock = reinterpret_cast<ThreadBlock*>(tblock_data);
      if (tblock != nullptr) {
        // Check if offset is within thread block range
        // For now, just try to free it
        tblock->alloc_.FreeOffset(offset);
        return;
      }
    }

    // Fall back to global allocator
    ScopedMutex scoped_lock(lock_, 0);
    alloc_.FreeOffset(offset);
  }

  /**
   * Get the shared header (stored in backend's shared header region)
   *
   * MultiProcessAllocator does not support shared headers, so this returns nullptr.
   *
   * @return nullptr
   */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *GetSharedHeader() {
    return nullptr;
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_
