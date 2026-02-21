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

#ifndef HSHM_DATA_STRUCTURES_IPC_RING_BUFFER_H_
#define HSHM_DATA_STRUCTURES_IPC_RING_BUFFER_H_

#ifdef _WIN32
using pid_t = int;
#else
#include <sys/types.h>
#endif

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/data_structures/ipc/shm_container.h"
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/types/bitfield.h"

namespace hshm::ipc {

/**
 * Ring buffer configuration flags for compile-time customization.
 * These flags control thread-safety models and buffer behavior.
 */
enum RingQueueFlag : uint32_t {
  /** Single producer single consumer mode (no atomics needed) */
  RING_BUFFER_SPSC_FLAGS = 0x01,
  /** Multiple producers single consumer mode (atomic operations required) */
  RING_BUFFER_MPSC_FLAGS = 0x02,
  /** Wait for space (block until space is available) */
  RING_BUFFER_WAIT_FOR_SPACE = 0x04,
  /** Error on no space (return error if no space) */
  RING_BUFFER_ERROR_ON_NO_SPACE = 0x08,
  /** Dynamic size (resize buffer when full) */
  RING_BUFFER_DYNAMIC_SIZE = 0x10,
  /** Fixed-size buffer (no dynamic resizing) */
  RING_BUFFER_FIXED_SIZE = 0x20
};

/**
 * Ring buffer entry with atomic ready flag.
 *
 * This structure combines an atomic bitfield for the ready flag with user data.
 * The ready flag is used to mark when data is ready for consumption,
 * with proper memory ordering to ensure data visibility across threads.
 *
 * @tparam T The type of data to store in the entry
 */
template <typename T>
struct RingBufferEntry {
  abitfield32_t flags_; /**< Atomic flags (bit 0 = data ready) */
  T data_;              /**< The actual data */

  /**
   * Default constructor
   */
  HSHM_INLINE_CROSS_FUN
  RingBufferEntry() : flags_(0) {}

  /**
   * Constructor with data
   *
   * @param data The data to store
   */
  HSHM_INLINE_CROSS_FUN
  explicit RingBufferEntry(const T& data) : flags_(0), data_(data) {}

  /**
   * Check if entry is ready for consumption (acquire semantics)
   *
   * @return True if entry is ready
   */
  HSHM_INLINE_CROSS_FUN
  bool IsReady() const { return flags_.Any(1); }

  /**
   * Mark entry as ready (release semantics)
   * Call this AFTER writing data to ensure visibility
   */
  HSHM_INLINE_CROSS_FUN
  void SetReady() { flags_.SetBits(1); }

  /**
   * Clear ready flag
   */
  HSHM_INLINE_CROSS_FUN
  void ClearReady() { flags_.UnsetBits(1); }

  /**
   * Get reference to data
   *
   * @return Reference to the data
   */
  HSHM_INLINE_CROSS_FUN
  T& GetData() { return data_; }

  /**
   * Get const reference to data
   *
   * @return Const reference to the data
   */
  HSHM_INLINE_CROSS_FUN
  const T& GetData() const { return data_; }
};

/**
 * Lock-free ring buffer (circular queue) for shared memory.
 *
 * This is a high-performance circular queue implementation designed for
 * shared memory environments. It uses a vector internally for storage and
 * supports Single Producer Single Consumer (SPSC) and Multiple Producer
 * Single Consumer (MPSC) modes with optional lock-free atomic operations.
 *
 * Features:
 * - Fixed-size circular buffer using vector for storage
 * - Configurable at compile-time via FLAGS template parameter
 * - Lock-free operation in MPSC mode using atomic fetch_add
 * - Proper handling of wrap-around at buffer boundaries
 * - Process-independent storage using OffsetPtr via vector
 * - RingBufferEntry combines validation flags with user data
 *
 * @tparam T The element type to store in the buffer
 * @tparam AllocT The allocator type for shared memory allocation
 * @tparam FLAGS Configuration flags controlling buffer behavior
 */
template <typename T, typename AllocT,
          uint32_t FLAGS = (RING_BUFFER_SPSC_FLAGS | RING_BUFFER_FIXED_SIZE |
                            RING_BUFFER_ERROR_ON_NO_SPACE)>
class ring_buffer : public ShmContainer<AllocT> {
 public:
  using allocator_type = AllocT;
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = size_t;
  using entry_type = RingBufferEntry<T>;

  // Configuration constants derived from FLAGS
  static constexpr bool IsSPSC = (FLAGS & RING_BUFFER_SPSC_FLAGS) != 0;
  static constexpr bool IsMPSC = (FLAGS & RING_BUFFER_MPSC_FLAGS) != 0;
  static constexpr bool WaitForSpace =
      (FLAGS & RING_BUFFER_WAIT_FOR_SPACE) != 0;
  static constexpr bool ErrorOnNoSpace =
      (FLAGS & RING_BUFFER_ERROR_ON_NO_SPACE) != 0;
  static constexpr bool DynamicSize = (FLAGS & RING_BUFFER_DYNAMIC_SIZE) != 0;
  static constexpr bool IsAtomic = IsMPSC;

  using entry_vector = vector<entry_type, AllocT>;
  using head_type = hipc::opt_atomic<u64, IsAtomic>;
  using tail_type = hipc::opt_atomic<u64, IsAtomic>;

 private:
  entry_vector queue_;     /**< Internal vector storing entries */
  head_type head_;         /**< Consumer head pointer */
  tail_type tail_;         /**< Producer tail pointer */
  u32 assigned_worker_id_; /**< Assigned worker ID for this lane (set by
                              orchestrator) */
  int signal_fd_;          /**< Signal file descriptor for awakening worker */
  pid_t tid_;              /**< Thread ID of the worker owning this lane */
  hipc::opt_atomic<bool, IsAtomic>
      active_; /**< Whether worker is accepting tasks (true) or blocked in
                  epoll_wait (false) */

 public:
  /**
   * Calculate exact size needed for a ring_buffer with given depth
   *
   * @param depth The queue depth (number of usable entries)
   * @return Exact size in bytes needed to allocate this ring_buffer
   */
  static size_t CalculateSize(size_t depth) {
    // Base size includes all member variables
    size_t base_size = sizeof(ring_buffer);

    // Vector entries: (depth + 1) entries, each is RingBufferEntry<T>
    size_t entry_size = sizeof(entry_type);
    size_t entries_size = (depth + 1) * entry_size;

    return base_size + entries_size;
  }

  /**
   * Constructor
   *
   * @param alloc The allocator to use for memory allocation
   * @param depth The initial capacity (number of entries)
   */
  HSHM_CROSS_FUN
  explicit ring_buffer(AllocT* alloc, size_t depth = 1024)
      : ShmContainer<AllocT>(alloc),
        queue_(alloc, depth + 1),
        head_(0),
        tail_(0),
        assigned_worker_id_(0),
        signal_fd_(-1),
        tid_(0),
        active_(true) {
    // Allocate depth + 1 to account for the one reserved slot
  }

  /**
   * Copy constructor
   *
   * Creates a new ring_buffer with the same configuration and contents as
   * another. Used when ring_buffers are stored in shared memory containers like
   * vector.
   */
  HSHM_CROSS_FUN
  ring_buffer(const ring_buffer& other)
      : ShmContainer<AllocT>(other.GetAllocator()),
        queue_(other.GetAllocator(), other.queue_.size() - 1),
        head_(other.head_),
        tail_(other.tail_),
        assigned_worker_id_(other.assigned_worker_id_),
        signal_fd_(other.signal_fd_),
        tid_(other.tid_),
        active_(other.active_.load()) {
    // Copy the contents of the queue from other
    for (size_t i = 0; i < other.queue_.size(); ++i) {
      queue_[i] = other.queue_[i];
    }
  }

  /**
   * Move constructor (deleted)
   *
   * IPC data structures must be allocated via allocator, not moved on stack.
   */
  ring_buffer(ring_buffer&& other) noexcept = delete;

  /**
   * Destructor
   */
  HSHM_CROSS_FUN
  ~ring_buffer() {
    // Vector destructor handles cleanup automatically
  }

  /**
   * Get assigned worker ID for this lane
   *
   * @return The worker ID assigned to this lane
   */
  HSHM_INLINE_CROSS_FUN
  u32 GetAssignedWorkerId() const { return assigned_worker_id_; }

  /**
   * Set assigned worker ID for this lane
   *
   * @param worker_id The worker ID to assign
   */
  HSHM_INLINE_CROSS_FUN
  void SetAssignedWorkerId(u32 worker_id) { assigned_worker_id_ = worker_id; }

  /**
   * Get signal file descriptor for this lane
   *
   * @return The signal file descriptor
   */
  HSHM_INLINE_CROSS_FUN
  int GetSignalFd() const { return signal_fd_; }

  /**
   * Set signal file descriptor for this lane
   *
   * @param signal_fd The signal file descriptor to set
   */
  HSHM_INLINE_CROSS_FUN
  void SetSignalFd(int signal_fd) { signal_fd_ = signal_fd; }

  /**
   * Get thread ID of the worker owning this lane
   *
   * @return The thread ID
   */
  HSHM_INLINE_CROSS_FUN
  pid_t GetTid() const { return tid_; }

  /**
   * Set thread ID of the worker owning this lane
   *
   * @param tid The thread ID to set
   */
  HSHM_INLINE_CROSS_FUN
  void SetTid(pid_t tid) { tid_ = tid; }

  /**
   * Check if worker is active (accepting tasks) or blocked in epoll_wait
   *
   * @return true if worker is active, false if blocked
   */
  HSHM_INLINE_CROSS_FUN
  bool IsActive() const { return active_.load(); }

  /**
   * Set worker active status
   *
   * @param active true if worker is active, false if blocked in epoll_wait
   */
  HSHM_INLINE_CROSS_FUN
  void SetActive(bool active) { active_.store(active); }

  /**
   * Get current size (number of items in buffer)
   *
   * @return Number of items currently in the buffer
   */
  HSHM_INLINE_CROSS_FUN
  size_t Size() const {
    u64 head = head_.load();
    u64 tail = tail_.load();
    if (tail >= head) {
      return tail - head;
    }
    return 0;
  }

  /**
   * Get capacity of the buffer
   *
   * @return Maximum number of items the buffer can hold
   */
  HSHM_INLINE_CROSS_FUN
  size_t Capacity() const {
    // Ring buffer capacity is one less than vector size (one slot reserved)
    // But we return capacity as the usable entries
    size_t depth = queue_.size();
    return (depth > 0) ? (depth - 1) : 0;
  }

  /**
   * Get depth (number of allocated slots)
   *
   * @return Number of allocated slots in vector
   */
  HSHM_INLINE_CROSS_FUN
  size_t GetDepth() const { return queue_.size(); }

  /**
   * Check if buffer is empty
   *
   * @return True if buffer contains no items
   */
  HSHM_INLINE_CROSS_FUN
  bool Empty() const { return head_.load() == tail_.load(); }

  /**
   * Check if buffer is full
   *
   * @return True if buffer has no more space
   */
  HSHM_INLINE_CROSS_FUN
  bool Full() const {
    u64 head = head_.load();
    u64 tail = tail_.load();
    // Buffer is full when number of items equals capacity
    // Since we have capacity+1 slots, full means tail - head == capacity
    return (tail - head) == (queue_.size() - 1);
  }

  /**
   * Push an element into the buffer
   *
   * @param val The value to push
   * @return True if push succeeded, false if buffer is full (when using
   * ErrorOnNoSpace)
   */
  HSHM_CROSS_FUN
  bool Push(const T& val) { return Emplace(val); }

  /**
   * Try to push an element (alias for Push)
   *
   * @param val The value to push
   * @return True if push succeeded, false if buffer is full
   */
  HSHM_INLINE_CROSS_FUN
  bool TryPush(const T& val) { return Push(val); }

  /**
   * Emplace an element (same as push)
   *
   * @param args Arguments to construct the element
   * @return True if emplace succeeded, false if buffer is full
   */
  template <typename... Args>
  HSHM_CROSS_FUN bool Emplace(Args&&... args) {
    // Load head and allocate a slot atomically
    u64 head = head_.load();
    u64 tail = tail_.fetch_add(1);
    entry_vector& queue = queue_;

    // Check if there's space in the queue
    // We need to keep one slot empty as a sentinel, so size must be <
    // queue.size()
    if constexpr (WaitForSpace) {
      size_t size = tail - head + 1;
      while (size >= queue.size()) {
        head = head_.load();
        size = tail - head + 1;
      }
    } else if constexpr (ErrorOnNoSpace) {
      size_t size = tail - head + 1;
      if (size >= queue.size()) {
        tail_.fetch_sub(1);
        return false;
      }
    } else if constexpr (DynamicSize) {
      size_t size = tail - head + 1;
      if (size >= queue.size()) {
        // Would need to resize vector - not implemented for now
        return false;
      }
    }

    // Emplace into queue at our slot
    size_t idx = tail % queue.size();
    auto& entry = queue[idx];
    entry.data_ = T(std::forward<Args>(args)...);
    entry.SetReady();  // Mark as ready with release semantics

    return true;
  }

  /**
   * Pop an element from the buffer
   *
   * @param val Reference to store the popped value
   * @return True if pop succeeded, false if buffer is empty
   */
  HSHM_CROSS_FUN
  bool Pop(T& val) {
    // Don't pop if there's no entries
    u64 head = head_.load();
    u64 tail = tail_.load();
    if (head >= tail) {
      return false;
    }

    // Pop the element, but only if it's marked valid
    size_t idx = head % queue_.size();
    entry_type& entry = queue_[idx];
    if (entry.IsReady()) {  // Acquire semantics ensure data visibility
      val = entry.data_;
      entry.ClearReady();
      head_.fetch_add(1);
      return true;
    }
    return false;
  }

  /**
   * Try to pop an element (alias for Pop)
   *
   * @param val Reference to store the popped value
   * @return True if pop succeeded, false if buffer is empty
   */
  HSHM_INLINE_CROSS_FUN
  bool TryPop(T& val) { return Pop(val); }

  /**
   * Clear the buffer
   */
  HSHM_CROSS_FUN
  void Clear() {
    head_ = 0;
    tail_ = 0;
    // Clear all entries
    for (size_t i = 0; i < queue_.size(); ++i) {
      queue_[i].flags_.Clear();
    }
  }

  /**
   * Reset the buffer (alias for Clear)
   */
  HSHM_INLINE_CROSS_FUN
  void Reset() { Clear(); }

  /**
   * Resize the buffer to a new depth
   *
   * @param new_depth The new capacity
   */
  HSHM_CROSS_FUN
  void Resize(size_t new_depth) {
    ring_buffer new_queue(this->GetAllocator(), new_depth);
    T val;
    while (Pop(val)) {
      new_queue.Push(val);
    }
    // Move new_queue data into this
    queue_ = std::move(new_queue.queue_);
  }
};

/**
 * Typedef for extensible ring buffer (single-thread only).
 *
 * This ring buffer will dynamically resize when capacity is reached,
 * making it suitable for scenarios where size cannot be predicted upfront.
 * NOT thread-safe for multiple producers.
 */
template <typename T, typename AllocT = hipc::Allocator>
using ext_ring_buffer =
    ring_buffer<T, AllocT, (RING_BUFFER_SPSC_FLAGS | RING_BUFFER_DYNAMIC_SIZE)>;

/**
 * Typedef for fixed-size SPSC (Single Producer Single Consumer) ring buffer.
 *
 * This ring buffer is optimized for single-threaded scenarios and will
 * return an error when attempting to push beyond capacity.
 */
template <typename T, typename AllocT = hipc::Allocator>
using spsc_ring_buffer =
    ring_buffer<T, AllocT,
                (RING_BUFFER_SPSC_FLAGS | RING_BUFFER_FIXED_SIZE |
                 RING_BUFFER_ERROR_ON_NO_SPACE)>;

/**
 * Typedef for fixed-size MPSC (Multiple Producer Single Consumer) ring buffer.
 *
 * This ring buffer is optimized for scenarios where multiple threads push
 * but only one thread consumes. Uses atomic operations for thread-safe
 * multi-producer access while supporting single consumer.
 */
template <typename T, typename AllocT = hipc::Allocator>
using mpsc_ring_buffer =
    ring_buffer<T, AllocT,
                (RING_BUFFER_MPSC_FLAGS | RING_BUFFER_FIXED_SIZE |
                 RING_BUFFER_WAIT_FOR_SPACE)>;

/**
 * Typedef for circular fixed-size MPSC (Multiple Producer Single Consumer) ring
 * buffer.
 *
 * This ring buffer is optimized for scenarios where multiple threads push
 * but only one thread consumes. Uses atomic operations for thread-safe
 * multi-producer access while supporting single consumer. Wraps around
 * when full instead of waiting.
 */
template <typename T, typename AllocT = hipc::Allocator>
using circular_mpsc_ring_buffer =
    ring_buffer<T, AllocT, (RING_BUFFER_MPSC_FLAGS | RING_BUFFER_FIXED_SIZE)>;

}  // namespace hshm::ipc

#endif  // HSHM_DATA_STRUCTURES_IPC_RING_BUFFER_H_
