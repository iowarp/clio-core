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
  OffsetPtr<>next_;  /**< Offset pointer to next node */

  /**
   * Default constructor
   */
  HSHM_CROSS_FUN
  slist_node() : next_(OffsetPtr<>::GetNull()) {}

  /**
   * Get the next pointer
   */
  HSHM_CROSS_FUN
  OffsetPtr<>GetNext() const {
    return next_;
  }

  /**
   * Set the next pointer
   */
  HSHM_CROSS_FUN
  void SetNext(const OffsetPtr<> &next) {
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
 public:
  /**
   * Iterator for traversing slist_pre nodes
   *
   * This iterator maintains the current node position and the previous node
   * for efficient removal via PopAt. The iterator is forward-only and supports
   * dereferencing, incrementing, and comparison operations.
   *
   * The iterator stores an allocator pointer to enable self-contained
   * navigation without requiring the parent list object.
   */
  class Iterator {
   private:
    OffsetPtr<> current_;  /**< Current node offset */
    OffsetPtr<> prev_;     /**< Previous node offset (for PopAt functionality) */
    void *alloc_;          /**< Allocator pointer for node traversal */

   public:
    /**
     * Construct a null iterator
     */
    HSHM_CROSS_FUN
    Iterator() : current_(OffsetPtr<>::GetNull()), prev_(OffsetPtr<>::GetNull()), alloc_(nullptr) {}

    /**
     * Construct an iterator at a specific position
     *
     * @param current Offset pointer to current node
     * @param prev Offset pointer to previous node (or null for head)
     * @param alloc Allocator pointer for node traversal
     */
    HSHM_CROSS_FUN
    Iterator(const OffsetPtr<> &current, const OffsetPtr<> &prev, void *alloc = nullptr)
        : current_(current), prev_(prev), alloc_(alloc) {}

    /**
     * Get current node offset
     *
     * @return OffsetPtr to current node
     */
    HSHM_CROSS_FUN
    OffsetPtr<> GetCurrent() const {
      return current_;
    }

    /**
     * Get previous node offset
     *
     * @return OffsetPtr to previous node (or null if at head)
     */
    HSHM_CROSS_FUN
    OffsetPtr<> GetPrev() const {
      return prev_;
    }

    /**
     * Check if iterator is at head
     *
     * @return true if previous node is null
     */
    HSHM_CROSS_FUN
    bool IsAtHead() const {
      return prev_.IsNull();
    }

    /**
     * Check if iterator is null (not pointing to any node)
     *
     * @return true if current is null
     */
    HSHM_CROSS_FUN
    bool IsNull() const {
      return current_.IsNull();
    }

    /**
     * Equality comparison
     *
     * @param other Iterator to compare with
     * @return true if both point to the same node
     */
    HSHM_CROSS_FUN
    bool operator==(const Iterator &other) const {
      return current_.load() == other.current_.load();
    }

    /**
     * Inequality comparison
     *
     * @param other Iterator to compare with
     * @return true if they point to different nodes
     */
    HSHM_CROSS_FUN
    bool operator!=(const Iterator &other) const {
      return current_.load() != other.current_.load();
    }

    /**
     * Prefix increment operator for forward iteration
     *
     * Advances the iterator to the next node in the list.
     * This operator is self-contained and does not require the parent list.
     *
     * @return Reference to this iterator after advancement
     */
    HSHM_CROSS_FUN
    Iterator& operator++() {
      if (IsNull() || alloc_ == nullptr) {
        // Already at end, remain null
        return *this;
      }

      // Get the next pointer from current node
      auto current = FullPtr<slist_node>(static_cast<hipc::Allocator*>(alloc_),
                                         OffsetPtr<slist_node>(current_.load()));
      OffsetPtr<> next_off = current.ptr_->next_;

      if (next_off.IsNull()) {
        // Update to null iterator for end
        current_ = OffsetPtr<>::GetNull();
        prev_ = OffsetPtr<>::GetNull();
        return *this;
      }

      // Update previous to current, and advance current to next
      prev_ = current_;
      current_ = next_off;
      return *this;
    }
  };

 private:
  opt_atomic<size_t, ATOMIC> size_;  /**< Number of elements in the list */
  OffsetPtr<> head_;               /**< Offset pointer to head node */

 public:
  /**
   * Default constructor
   */
  HSHM_CROSS_FUN
  slist() : size_(0), head_(OffsetPtr<>::GetNull()) {}

  /**
   * Initialize the list
   */
  HSHM_CROSS_FUN
  void Init() {
    size_.store(0);
    head_ = OffsetPtr<>::GetNull();
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
    auto head = FullPtr<slist_node>(alloc, OffsetPtr<slist_node>(head_.load()));

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
  OffsetPtr<>GetHead() const {
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
    return FullPtr<slist_node>(alloc, OffsetPtr<slist_node>(head_.load()));
  }

  /**
   * Get iterator to the beginning of the list
   *
   * @param alloc Allocator pointer for iterator traversal
   * @return Iterator pointing to the head node, or null iterator if list is empty
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  Iterator begin(AllocT *alloc) {
    return Iterator(head_, OffsetPtr<>::GetNull(), alloc);
  }

  /**
   * Get a null iterator (end marker)
   *
   * @return Null iterator
   */
  HSHM_CROSS_FUN
  Iterator end() const {
    return Iterator();
  }

  /**
   * Remove node at iterator position and return it
   *
   * Removes the node pointed to by the iterator from the list. The node
   * is not deallocated - the caller is responsible for managing its memory.
   *
   * @param alloc Allocator used for address translation
   * @param it Iterator pointing to the node to remove
   * @return FullPtr to the removed node, or null if iterator is invalid
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  FullPtr<slist_node> PopAt(AllocT *alloc, const Iterator &it) {
    // Check if iterator is valid
    if (it.IsNull()) {
      return FullPtr<slist_node>::GetNull();
    }

    // Check if list is empty
    if (size_.load() == 0) {
      return FullPtr<slist_node>::GetNull();
    }

    // Get the current node
    auto current = FullPtr<slist_node>(alloc, OffsetPtr<slist_node>(it.GetCurrent().load()));

    if (it.IsAtHead()) {
      // Removing head node - update head_ directly
      head_ = current.ptr_->next_;
    } else {
      // Removing middle node - update prev's next pointer
      auto prev = FullPtr<slist_node>(alloc, OffsetPtr<slist_node>(it.GetPrev().load()));
      prev.ptr_->next_ = current.ptr_->next_;
    }

    // Decrement size
    size_.store(size_.load() - 1);

    return current;
  }
};

}  // namespace hshm::ipc::pre

#endif  // HSHM_DATA_STRUCTURES_IPC_SLIST_PRE_H_
