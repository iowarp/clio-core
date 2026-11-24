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

#ifndef CTP_DATA_STRUCTURES_IPC_SLIST_PRE_H_
#define CTP_DATA_STRUCTURES_IPC_SLIST_PRE_H_

#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/types/atomic.h"

namespace ctp::ipc::pre {

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
  CTP_CROSS_FUN
  slist_node() : next_(OffsetPtr<>::GetNull()) {}

  /**
   * Get the next pointer
   */
  CTP_CROSS_FUN
  OffsetPtr<>GetNext() const {
    return next_;
  }

  /**
   * Set the next pointer
   */
  CTP_CROSS_FUN
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
 * the node memory. Nodes are expected to inherit from slist_node.
 *
 * @tparam NodeT The actual node type that inherits from slist_node
 * @tparam ATOMIC Whether to use atomic operations for thread-safety
 */
template<typename NodeT, bool ATOMIC = false>
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
    OffsetPtr<NodeT> current_;  /**< Current node offset */
    OffsetPtr<NodeT> prev_;     /**< Previous node offset (for PopAt functionality) */
    void *alloc_;               /**< Allocator pointer for node traversal */

   public:
    /**
     * Construct a null iterator
     */
    CTP_CROSS_FUN
    Iterator() : current_(OffsetPtr<NodeT>::GetNull()), prev_(OffsetPtr<NodeT>::GetNull()), alloc_(nullptr) {}

    /**
     * Construct an iterator at a specific position
     *
     * @param current Offset pointer to current node
     * @param prev Offset pointer to previous node (or null for head)
     * @param alloc Allocator pointer for node traversal
     */
    CTP_CROSS_FUN
    Iterator(const OffsetPtr<NodeT> &current, const OffsetPtr<NodeT> &prev, void *alloc = nullptr)
        : current_(current), prev_(prev), alloc_(alloc) {}

    /**
     * Get current node offset
     *
     * @return OffsetPtr to current node
     */
    CTP_CROSS_FUN
    OffsetPtr<NodeT> GetCurrent() const {
      return current_;
    }

    /**
     * Get previous node offset
     *
     * @return OffsetPtr to previous node (or null if at head)
     */
    CTP_CROSS_FUN
    OffsetPtr<NodeT> GetPrev() const {
      return prev_;
    }

    /**
     * Check if iterator is at head
     *
     * @return true if previous node is null
     */
    CTP_CROSS_FUN
    bool IsAtHead() const {
      return prev_.IsNull();
    }

    /**
     * Check if iterator is null (not pointing to any node)
     *
     * @return true if current is null
     */
    CTP_CROSS_FUN
    bool IsNull() const {
      return current_.IsNull();
    }

    /**
     * Set this iterator to null (equal to end())
     *
     * Makes this iterator equivalent to the end marker by setting
     * both current and previous pointers to null.
     */
    CTP_CROSS_FUN
    void SetNull() {
      current_ = OffsetPtr<NodeT>::GetNull();
      prev_ = OffsetPtr<NodeT>::GetNull();
    }

    /**
     * Equality comparison
     *
     * @param other Iterator to compare with
     * @return true if both point to the same node
     */
    CTP_CROSS_FUN
    bool operator==(const Iterator &other) const {
      return current_.load() == other.current_.load();
    }

    /**
     * Inequality comparison
     *
     * @param other Iterator to compare with
     * @return true if they point to different nodes
     */
    CTP_CROSS_FUN
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
    CTP_CROSS_FUN
    Iterator& operator++() {
      if (IsNull() || alloc_ == nullptr) {
        // Already at end, remain null
        SetNull();
        return *this;
      }

      // Get the next pointer from current node
      auto current = FullPtr<NodeT>(static_cast<ctp::ipc::Allocator*>(alloc_),
                                    current_);
      OffsetPtr<> next_off = current.ptr_->next_;

      if (next_off.IsNull()) {
        // Update to null iterator for end
        current_ = OffsetPtr<NodeT>::GetNull();
        prev_ = OffsetPtr<NodeT>::GetNull();
        return *this;
      }

      // Update previous to current, and advance current to next
      prev_ = current_;
      current_ = OffsetPtr<NodeT>(next_off);
      return *this;
    }
  };

 private:
  opt_atomic<size_t, ATOMIC> size_;  /**< Number of elements in the list */
  OffsetPtr<NodeT> head_;            /**< Offset pointer to head node */

 public:
  /**
   * Default constructor
   */
  CTP_CROSS_FUN
  slist() : size_(0), head_(OffsetPtr<NodeT>::GetNull()) {}

  /**
   * Initialize the list
   */
  CTP_CROSS_FUN
  void Init() {
    size_.store(0);
    head_ = OffsetPtr<NodeT>::GetNull();
  }

  /**
   * Emplace a preallocated node at the front of the list
   *
   * @param alloc Allocator used for the node (for address translation)
   * @param node Preallocated node to add to the list
   */
  template<typename AllocT>
  CTP_CROSS_FUN
  void emplace(AllocT *alloc, FullPtr<NodeT> node) {
    // Set node's next to current head
    node.ptr_->next_ = head_.template Cast<void>();

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
  CTP_CROSS_FUN
  FullPtr<NodeT> pop(AllocT *alloc) {
    // Check if list is empty
    if (size_.load() == 0) {
      return FullPtr<NodeT>::GetNull();
    }

    // Double-check that head is not null
    if (head_.IsNull()) {
      // This shouldn't happen if size > 0, but let's handle it gracefully
      // Reset the list to empty state
      size_.store(0);
      return FullPtr<NodeT>::GetNull();
    }

    // Get the head node
    auto head = FullPtr<NodeT>(alloc, head_);

    // Check if head was successfully created
    if (head.IsNull()) {
      // This shouldn't happen, but let's handle it gracefully
      // Reset the list to empty state
      size_.store(0);
      head_ = OffsetPtr<NodeT>::GetNull();
      return FullPtr<NodeT>::GetNull();
    }

    // Update head to next node
    head_ = OffsetPtr<NodeT>(head.ptr_->next_);

    // Decrement size
    size_.store(size_.load() - 1);

    return head;
  }

  /**
   * Get the number of elements in the list
   *
   * @return Number of elements
   */
  CTP_CROSS_FUN
  size_t size() const {
    return size_.load();
  }

  /**
   * Check if the list is empty
   *
   * @return true if the list is empty, false otherwise
   */
  CTP_CROSS_FUN
  bool empty() const {
    return size_.load() == 0;
  }

  /**
   * Get the head pointer (for debugging/inspection)
   *
   * @return Offset pointer to the head node
   */
  CTP_CROSS_FUN
  OffsetPtr<NodeT> GetHead() const {
    return head_;
  }

  /**
   * Peek at the first element without removing it
   *
   * @param alloc Allocator used for the node (for address translation)
   * @return FullPtr to the head node, or null if list is empty
   */
  template<typename AllocT>
  CTP_CROSS_FUN
  FullPtr<NodeT> peek(AllocT *alloc) const {
    if (size_.load() == 0) {
      return FullPtr<NodeT>::GetNull();
    }
    return FullPtr<NodeT>(alloc, head_);
  }

  /**
   * Get iterator to the beginning of the list
   *
   * @param alloc Allocator pointer for iterator traversal
   * @return Iterator pointing to the head node, or null iterator if list is empty
   */
  template<typename AllocT>
  CTP_CROSS_FUN
  Iterator begin(AllocT *alloc) {
    return Iterator(head_, OffsetPtr<NodeT>::GetNull(), alloc);
  }

  /**
   * Get a null iterator (end marker)
   *
   * @return Null iterator
   */
  CTP_CROSS_FUN
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
  CTP_CROSS_FUN
  FullPtr<NodeT> PopAt(AllocT *alloc, const Iterator &it) {
    // Check if iterator is valid
    if (it.IsNull()) {
      return FullPtr<NodeT>::GetNull();
    }

    // Check if list is empty
    if (size_.load() == 0) {
      return FullPtr<NodeT>::GetNull();
    }

    // Get the current node
    auto current = FullPtr<NodeT>(alloc, it.GetCurrent());

    if (it.IsAtHead()) {
      // Removing head node - update head_ directly
      head_ = OffsetPtr<NodeT>(current.ptr_->next_);
    } else {
      // Removing middle node - update prev's next pointer
      auto prev = FullPtr<NodeT>(alloc, it.GetPrev());
      prev.ptr_->next_ = current.ptr_->next_;
    }

    // Decrement size
    size_.store(size_.load() - 1);

    return current;
  }
};

/**
 * Private-memory singly-linked list node (raw pointers, no offset math)
 *
 * Used when the allocator operates in MemMode::kPrivate and does not
 * need shared-memory offset-based linkage.
 */
class priv_slist_node {
 public:
  priv_slist_node *next_;

  CTP_CROSS_FUN priv_slist_node() : next_(nullptr) {}
};

/**
 * Private-memory singly-linked list (raw pointers, no allocator needed)
 *
 * @tparam NodeT Node type inheriting from priv_slist_node
 */
template <typename NodeT>
class priv_slist {
 public:
  class Iterator {
   private:
    NodeT *current_;
    NodeT *prev_;

   public:
    CTP_CROSS_FUN Iterator() : current_(nullptr), prev_(nullptr) {}
    CTP_CROSS_FUN Iterator(NodeT *current, NodeT *prev)
        : current_(current), prev_(prev) {}

    CTP_CROSS_FUN NodeT *Get() const { return current_; }
    CTP_CROSS_FUN NodeT *GetPrev() const { return prev_; }
    CTP_CROSS_FUN bool IsAtHead() const { return prev_ == nullptr; }
    CTP_CROSS_FUN bool IsNull() const { return current_ == nullptr; }

    CTP_CROSS_FUN bool operator==(const Iterator &o) const {
      return current_ == o.current_;
    }
    CTP_CROSS_FUN bool operator!=(const Iterator &o) const {
      return current_ != o.current_;
    }

    CTP_CROSS_FUN Iterator &operator++() {
      if (!current_) return *this;
      prev_ = current_;
      current_ = static_cast<NodeT *>(current_->next_);
      return *this;
    }
  };

 private:
  size_t size_;
  NodeT *head_;

 public:
  CTP_CROSS_FUN priv_slist() : size_(0), head_(nullptr) {}

  CTP_CROSS_FUN void Init() {
    size_ = 0;
    head_ = nullptr;
  }

  CTP_CROSS_FUN bool empty() const { return size_ == 0; }
  CTP_CROSS_FUN size_t size() const { return size_; }

  CTP_CROSS_FUN void emplace(NodeT *node) {
    node->next_ = head_;
    head_ = node;
    ++size_;
  }

  CTP_CROSS_FUN NodeT *pop() {
    if (size_ == 0 || !head_) return nullptr;
    NodeT *h = head_;
    head_ = static_cast<NodeT *>(h->next_);
    --size_;
    return h;
  }

  CTP_CROSS_FUN Iterator begin() { return Iterator(head_, nullptr); }
  CTP_CROSS_FUN Iterator end() const { return Iterator(); }

  CTP_CROSS_FUN NodeT *PopAt(const Iterator &it) {
    if (it.IsNull() || size_ == 0) return nullptr;
    NodeT *cur = it.Get();
    if (it.IsAtHead()) {
      head_ = static_cast<NodeT *>(cur->next_);
    } else {
      it.GetPrev()->next_ = cur->next_;
    }
    --size_;
    return cur;
  }
};

}  // namespace ctp::ipc::pre

#endif  // CTP_DATA_STRUCTURES_IPC_SLIST_PRE_H_
