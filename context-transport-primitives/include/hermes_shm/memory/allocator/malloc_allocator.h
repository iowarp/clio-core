#ifndef HERMES_SHM_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_
#define HERMES_SHM_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include <memory>

namespace hshm::ipc {

/**
 * Page header for malloc allocations
 * Stored before each allocated block
 */
struct MallocPage {
  size_t page_size_;  // Size of this allocation including header

  MallocPage() : page_size_(0) {}
  explicit MallocPage(size_t size) : page_size_(size) {}
};

/**
 * Allocator that wraps standard malloc/free
 * Uses null allocator ID since memory is not shared
 */
template <bool ATOMIC>
class _MallocAllocator : public Allocator {
 public:
  /**
   * Default constructor - initializes with null backend
   */
  _MallocAllocator() {
    // Create a null backend with max capacity so ContainsPtr always returns true
    MemoryBackend null_backend;
    null_backend.id_ = MemoryBackendId::GetNull();
    null_backend.backend_size_ = 0;
    null_backend.data_capacity_ = SIZE_MAX;  // Accept all pointer offsets
    null_backend.header_ = nullptr;
    null_backend.region_ = nullptr;
    null_backend.data_ = nullptr;
    SetBackend(null_backend);

    // Minimal header size
    alloc_header_size_ = sizeof(_MallocAllocator);

    // No fixed region size (allocates on demand via malloc)
    region_size_ = 0;
    data_start_ = 0;
    // Set this_ so that GetBackendData() returns 0
    // Formula: GetBackendData() = this - this_ = 0
    // Then: 0 + raw_pointer_as_offset = raw_pointer
    this_ = reinterpret_cast<size_t>(this);

#ifdef HSHM_ALLOC_TRACK_SIZE
    total_alloc_ = 0;
#endif
  }

  /**
   * Initialize allocator (for compatibility with backend pattern)
   */
  void shm_init(const MemoryBackend &backend, size_t region_size = 0) {
    // Ignore backend - we use malloc directly
    (void)backend;
    (void)region_size;
  }

  /**
   * Attach to existing allocator
   * Not supported for malloc allocator
   */
  void shm_attach(const MemoryBackend &backend) {
    (void)backend;
    throw SHMEM_NOT_SUPPORTED.format();
  }

  /**
   * Allocate memory via malloc
   */
  OffsetPtr<> AllocateOffset(size_t size) {
    // Allocate space for header + user data
    size_t total_size = sizeof(MallocPage) + size;
    void *ptr = malloc(total_size);

    if (!ptr) {
      return OffsetPtr<>();  // Null pointer
    }

    // Initialize page header
    MallocPage *page = reinterpret_cast<MallocPage*>(ptr);
    page->page_size_ = total_size;

    // User data starts after header
    void *user_ptr = reinterpret_cast<char*>(ptr) + sizeof(MallocPage);

#ifdef HSHM_ALLOC_TRACK_SIZE
    total_alloc_ += size;
#endif

    // Return offset (cast pointer to offset for interface compatibility)
    return OffsetPtr<>(reinterpret_cast<size_t>(user_ptr));
  }

  /**
   * Reallocate memory via realloc
   */
  OffsetPtr<> ReallocateOffsetNoNullCheck(OffsetPtr<> p, size_t new_size) {
    // Get page header (before user data)
    void *user_ptr = reinterpret_cast<void*>(p.load());
    MallocPage *old_page = reinterpret_cast<MallocPage*>(
        reinterpret_cast<char*>(user_ptr) - sizeof(MallocPage));

    size_t old_size = old_page->page_size_ - sizeof(MallocPage);

    // Reallocate
    size_t new_total_size = sizeof(MallocPage) + new_size;
    void *new_ptr = realloc(old_page, new_total_size);

    if (!new_ptr) {
      return OffsetPtr<>();  // Allocation failed, original pointer still valid
    }

    // Update page header
    MallocPage *new_page = reinterpret_cast<MallocPage*>(new_ptr);
    new_page->page_size_ = new_total_size;

    // New user data pointer
    void *new_user_ptr = reinterpret_cast<char*>(new_ptr) + sizeof(MallocPage);

#ifdef HSHM_ALLOC_TRACK_SIZE
    total_alloc_ += (new_size - old_size);
#endif

    return OffsetPtr<>(reinterpret_cast<size_t>(new_user_ptr));
  }

  /**
   * Free memory via free
   */
  void FreeOffsetNoNullCheck(OffsetPtr<> p) {
    // Get page header (before user data)
    void *user_ptr = reinterpret_cast<void*>(p.load());
    MallocPage *page = reinterpret_cast<MallocPage*>(
        reinterpret_cast<char*>(user_ptr) - sizeof(MallocPage));

#ifdef HSHM_ALLOC_TRACK_SIZE
    size_t size = page->page_size_ - sizeof(MallocPage);
    total_alloc_ -= size;
#endif

    // Free entire allocation (header + data)
    free(page);
  }

  /**
   * Get currently allocated size
   */
  size_t GetCurrentlyAllocatedSize() {
#ifdef HSHM_ALLOC_TRACK_SIZE
    return total_alloc_;
#else
    return 0;
#endif
  }

  /**
   * Thread-local storage (no-op for malloc)
   */
  void CreateTls() {}
  void FreeTls() {}
};

// Type alias
typedef BaseAllocator<_MallocAllocator<false>> MallocAllocator;

/**
 * Singleton wrapper for MallocAllocator
 * Self-initializing and globally accessible
 */
class MallocAllocatorSingleton {
 public:
  /**
   * Get the singleton instance
   * Initializes lazily on first access
   */
  static MallocAllocator* Get() {
    static MallocAllocatorSingleton instance;
    return &instance.allocator_;
  }

 private:
  MallocAllocatorSingleton() = default;
  ~MallocAllocatorSingleton() = default;

  // Non-copyable, non-movable
  MallocAllocatorSingleton(const MallocAllocatorSingleton&) = delete;
  MallocAllocatorSingleton& operator=(const MallocAllocatorSingleton&) = delete;

  MallocAllocator allocator_;
};

}  // namespace hshm::ipc

// Global accessor macro
#define HSHM_MALLOC ::hshm::ipc::MallocAllocatorSingleton::Get()

#endif  // HERMES_SHM_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_
