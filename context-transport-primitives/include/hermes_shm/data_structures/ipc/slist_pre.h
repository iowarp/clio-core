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

#ifndef HSHM_DATA_STRUCTURES_IPC_SLIST_PRE_H_
#define HSHM_DATA_STRUCTURES_IPC_SLIST_PRE_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/types/atomic.h"

namespace hshm::ipc::pre {

/**
 * Singly-linked list node for preallocated list
 *
 * This node structure is designed to be embedded in other data structures.
 * It does not own the data - it only maintains the list linkage.
 */
class slist_node {
 public:
  OffsetPtr next_;  /**< Offset pointer to next node */

  /**
   * Default constructor
   */
  HSHM_CROSS_FUN
  slist_node() : next_(OffsetPtr::GetNull()) {}

  /**
   * Get the next pointer
   */
  HSHM_CROSS_FUN
  OffsetPtr GetNext() const {
    return next_;
  }

  /**
   * Set the next pointer
   */
  HSHM_CROSS_FUN
  void SetNext(const OffsetPtr &next) {
    next_ = next;
  }
};

/**
 * Singly-linked list for preallocated nodes
 *
 * This is a shared-memory compatible singly-linked list that does not
 * perform allocations. All nodes must be preallocated by the caller.
 *
 * The list maintains only the linkage between nodes - it does not own
 * the node memory. Nodes are expected to embed slist_node as a member.
 *
 * @tparam ATOMIC Whether to use atomic operations for thread-safety
 */
template<bool ATOMIC = false>
class slist {
 private:
  opt_atomic<size_t, ATOMIC> size_;  /**< Number of elements in the list */
  OffsetPtr head_;               /**< Offset pointer to head node */

 public:
  /**
   * Default constructor
   */
  HSHM_CROSS_FUN
  slist() : size_(0), head_(OffsetPtr::GetNull()) {}

  /**
   * Initialize the list
   */
  HSHM_CROSS_FUN
  void Init() {
    size_.store(0);
    head_ = OffsetPtr::GetNull();
  }

  /**
   * Emplace a preallocated node at the front of the list
   *
   * @param alloc Allocator used for the node (for address translation)
   * @param node Preallocated node to add to the list
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void emplace(AllocT *alloc, FullPtr<slist_node> node) {
    // Set node's next to current head
    node.ptr_->next_ = head_;

    // Update head to point to new node
    head_ = node.shm_.off_;

    // Increment size
    size_.store(size_.load() + 1);
  }

  /**
   * Pop the first entry from the list
   *
   * @param alloc Allocator used for the node (for address translation)
   * @return FullPtr to the popped node, or null if list is empty
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  FullPtr<slist_node> pop(AllocT *alloc) {
    // Check if list is empty
    if (size_.load() == 0) {
      return FullPtr<slist_node>::GetNull();
    }

    // Get the head node
    auto head = FullPtr<slist_node>(alloc, head_);

    // Update head to next node
    head_ = head.ptr_->next_;

    // Decrement size
    size_.store(size_.load() - 1);

    return head;
  }

  /**
   * Get the number of elements in the list
   *
   * @return Number of elements
   */
  HSHM_CROSS_FUN
  size_t size() const {
    return size_.load();
  }

  /**
   * Check if the list is empty
   *
   * @return true if the list is empty, false otherwise
   */
  HSHM_CROSS_FUN
  bool empty() const {
    return size_.load() == 0;
  }

  /**
   * Get the head pointer (for debugging/inspection)
   *
   * @return Offset pointer to the head node
   */
  HSHM_CROSS_FUN
  OffsetPtr GetHead() const {
    return head_;
  }

  /**
   * Peek at the first element without removing it
   *
   * @param alloc Allocator used for the node (for address translation)
   * @return FullPtr to the head node, or null if list is empty
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  FullPtr<slist_node> peek(AllocT *alloc) const {
    if (size_.load() == 0) {
      return FullPtr<slist_node>::GetNull();
    }
    return FullPtr<slist_node>(alloc, head_);
  }
};

}  // namespace hshm::ipc::pre

#endif  // HSHM_DATA_STRUCTURES_IPC_SLIST_PRE_H_
