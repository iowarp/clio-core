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

#ifndef CTP_DATA_STRUCTURES_IPC_MULTI_RING_BUFFER_H_
#define CTP_DATA_STRUCTURES_IPC_MULTI_RING_BUFFER_H_

#include "clio_ctp/data_structures/ipc/shm_container.h"
#include "clio_ctp/data_structures/ipc/vector.h"
#include "clio_ctp/data_structures/ipc/ring_buffer.h"
#include "clio_ctp/constants/macros.h"
#include <cassert>

namespace ctp::ipc {

/**
 * Multi-lane ring buffer container for shared memory.
 *
 * This is a container of multiple ring_buffer instances organized as a vector.
 * It provides lane-based access to independent ring buffers, where each lane
 * can have multiple priority levels. This is useful for scenarios where you need
 * to multiplex data across multiple independent queues (e.g., task scheduling
 * with multiple lanes and priorities).
 *
 * Features:
 * - Multiple independent ring buffers organized by lane and priority
 * - Lane-based indexing with automatic index calculation
 * - Configurable depth for all buffers
 * - Process-independent storage using vector of ring_buffers
 * - Type-safe access to ring buffers via GetLane()
 *
 * @tparam T The element type to store in each ring buffer
 * @tparam AllocT The allocator type for shared memory allocation
 * @tparam FLAGS Configuration flags controlling ring buffer behavior
 */
template<typename T,
         typename AllocT,
         uint32_t FLAGS = (RING_BUFFER_SPSC_FLAGS | RING_BUFFER_FIXED_SIZE | RING_BUFFER_ERROR_ON_NO_SPACE)>
class multi_ring_buffer : public ShmContainer<AllocT> {
 public:
  using allocator_type = AllocT;
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = size_t;
  using entry_type = RingBufferEntry<T>;
  using ring_buffer_type = ring_buffer<T, AllocT, FLAGS>;
  using lanes_vector = vector<ring_buffer_type, AllocT>;

 private:
  lanes_vector lanes_;       /**< Vector of OffsetPtrs to ring buffers */
  size_t num_lanes_;         /**< Number of lanes */
  size_t num_prios_;         /**< Number of priority levels per lane */

 public:
  /**
   * Calculate exact size needed for a multi_ring_buffer with given parameters
   *
   * @param num_lanes The number of lanes
   * @param num_prios The number of priority levels per lane
   * @param depth The queue depth per ring buffer
   * @return Exact size in bytes needed to allocate this multi_ring_buffer
   */
  static size_t CalculateSize(size_t num_lanes, size_t num_prios, size_t depth) {
    // Base size includes all member variables
    size_t base_size = sizeof(multi_ring_buffer);

    // Each ring buffer's size
    size_t per_ring_buffer_size = ring_buffer_type::CalculateSize(depth);

    // Total ring buffers = num_lanes * num_prios
    size_t total_ring_buffers = num_lanes * num_prios;

    return base_size + (total_ring_buffers * per_ring_buffer_size);
  }

  /**
   * Constructor
   *
   * Initializes a multi-lane ring buffer with the specified number of lanes,
   * priority levels per lane, and depth for each individual ring buffer.
   * Total number of ring buffers created = num_lanes * num_prios.
   *
   * @param alloc The allocator to use for memory allocation
   * @param num_lanes The number of lanes
   * @param num_prios The number of priority levels per lane
   * @param depth The depth (capacity) of each individual ring buffer
   */
  CTP_CROSS_FUN
  explicit multi_ring_buffer(AllocT *alloc, size_t num_lanes, size_t num_prios,
                             size_t depth)
      : ShmContainer<AllocT>(alloc),
        num_lanes_(num_lanes),
        num_prios_(num_prios),
        lanes_(alloc, num_lanes * num_prios, depth) {
  }

  /**
   * Copy constructor (deleted)
   *
   * IPC data structures must be allocated via allocator, not copied on stack.
   */
  multi_ring_buffer(const multi_ring_buffer &other) = delete;

  /**
   * Move constructor (deleted)
   *
   * IPC data structures must be allocated via allocator, not moved on stack.
   */
  multi_ring_buffer(multi_ring_buffer &&other) noexcept = delete;

  /**
   * Destructor
   */
  CTP_CROSS_FUN
  ~multi_ring_buffer() {
    // Vector destructor handles cleanup automatically
  }

  /**
   * Get ring buffer for a specific lane and priority level
   *
   * Returns a reference to the ring buffer at the specified lane and priority.
   * The actual index is calculated as: lane_id * num_prios + prio.
   *
   * @param lane_id The lane identifier (must be < num_lanes_)
   * @param prio The priority level (must be < num_prios_)
   * @return Reference to the ring_buffer at the specified lane and priority
   *
   * Asserts that lane_id < num_lanes_ and prio < num_prios_
   */
  CTP_INLINE_CROSS_FUN
  ring_buffer_type& GetLane(size_t lane_id, size_t prio) {
    assert(lane_id < num_lanes_);
    assert(prio < num_prios_);
    size_t idx = lane_id * num_prios_ + prio;
    // lanes_[idx] is a raw pointer to ring_buffer_type
    return lanes_[idx];
  }

  /**
   * Get ring buffer for a specific lane and priority level (const version)
   *
   * Returns a const reference to the ring buffer at the specified lane and
   * priority. The actual index is calculated as: lane_id * num_prios + prio.
   *
   * @param lane_id The lane identifier (must be < num_lanes_)
   * @param prio The priority level (must be < num_prios_)
   * @return Const reference to the ring_buffer at the specified lane and priority
   *
   * Asserts that lane_id < num_lanes_ and prio < num_prios_
   */
  CTP_INLINE_CROSS_FUN
  const ring_buffer_type& GetLane(size_t lane_id, size_t prio) const {
    assert(lane_id < num_lanes_);
    assert(prio < num_prios_);
    size_t idx = lane_id * num_prios_ + prio;
    // lanes_[idx] is a raw pointer to ring_buffer_type
    return lanes_[idx];
  }

  /**
   * Get the number of lanes
   *
   * @return The number of lanes in this multi-ring buffer
   */
  CTP_INLINE_CROSS_FUN
  size_t GetNumLanes() const {
    return num_lanes_;
  }

  /**
   * Get the number of priority levels per lane
   *
   * @return The number of priority levels per lane
   */
  CTP_INLINE_CROSS_FUN
  size_t GetNumPrios() const {
    return num_prios_;
  }

  /**
   * Get the total number of ring buffers
   *
   * @return Total number of ring buffers (num_lanes * num_prios)
   */
  CTP_INLINE_CROSS_FUN
  size_t GetTotalBuffers() const {
    return num_lanes_ * num_prios_;
  }

};

/**
 * Typedef for extensible ring buffer (single-thread only).
 *
 * This ring buffer will dynamically resize when capacity is reached,
 * making it suitable for scenarios where size cannot be predicted upfront.
 * NOT thread-safe for multiple producers.
 */
template <typename T, typename AllocT = ctp::ipc::Allocator>
using multi_ext_ring_buffer =
    multi_ring_buffer<T, AllocT, (RING_BUFFER_SPSC_FLAGS | RING_BUFFER_DYNAMIC_SIZE)>;

/**
 * Typedef for fixed-size SPSC (Single Producer Single Consumer) ring buffer.
 *
 * This ring buffer is optimized for single-threaded scenarios and will
 * return an error when attempting to push beyond capacity.
 */
template <typename T, typename AllocT = ctp::ipc::Allocator>
using multi_spsc_ring_buffer =
    multi_ring_buffer<T, AllocT,
                (RING_BUFFER_SPSC_FLAGS | RING_BUFFER_FIXED_SIZE |
                 RING_BUFFER_ERROR_ON_NO_SPACE)>;

/**
 * Typedef for fixed-size MPSC (Multiple Producer Single Consumer) ring buffer.
 *
 * This ring buffer is optimized for scenarios where multiple threads push
 * but only one thread consumes. Uses atomic operations for thread-safe
 * multi-producer access while supporting single consumer.
 */
template <typename T, typename AllocT = ctp::ipc::Allocator>
using multi_mpsc_ring_buffer =
    multi_ring_buffer<T, AllocT,
                      (RING_BUFFER_MPSC_FLAGS | RING_BUFFER_FIXED_SIZE |
                       RING_BUFFER_WAIT_FOR_SPACE)>;

/**
 * Typedef for multi-lane MPMC ring buffer where each lane signals its
 * owning worker thread on an empty→non-empty push.
 */
template <typename T, typename AllocT = ctp::ipc::Allocator>
using multi_signal_mpmc_queue =
    multi_ring_buffer<T, AllocT,
                      (RING_BUFFER_MPSC_FLAGS | RING_BUFFER_FIXED_SIZE |
                       RING_BUFFER_WAIT_FOR_SPACE | RING_BUFFER_LOCK_POP |
                       RING_BUFFER_SIGNAL_ON_0)>;

}  // namespace ctp::ipc

#endif  // CTP_DATA_STRUCTURES_IPC_MULTI_RING_BUFFER_H_
