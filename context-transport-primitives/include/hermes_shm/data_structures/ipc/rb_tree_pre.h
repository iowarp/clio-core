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

#ifndef HSHM_DATA_STRUCTURES_IPC_RB_TREE_PRE_H_
#define HSHM_DATA_STRUCTURES_IPC_RB_TREE_PRE_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/types/atomic.h"

namespace hshm::ipc::pre {

/**
 * Color enumeration for Red-Black tree nodes
 */
enum class RBColor : uint8_t {
  RED = 0,
  BLACK = 1
};

/**
 * Red-Black tree node base class for preallocated tree
 *
 * This node structure is designed to be inherited by user-defined node types.
 * It maintains the RB tree properties and linkage without storing the key.
 * The derived NodeT class must provide:
 * - A 'key' member variable for ordering
 * - Comparison operators (operator<, operator>, operator==)
 */
class rb_node {
 public:
  OffsetPtr left_;            /**< Offset pointer to left child */
  OffsetPtr right_;           /**< Offset pointer to right child */
  OffsetPtr parent_;          /**< Offset pointer to parent node */
  RBColor color_;                 /**< Node color (RED or BLACK) */

  /**
   * Default constructor
   */
  HSHM_CROSS_FUN
  rb_node() : left_(OffsetPtr::GetNull()),
              right_(OffsetPtr::GetNull()),
              parent_(OffsetPtr::GetNull()),
              color_(RBColor::RED) {}

  /**
   * Check if this is a null node
   */
  HSHM_CROSS_FUN
  bool IsNull() const {
    return left_.IsNull() && right_.IsNull() && parent_.IsNull();
  }
};

/**
 * Red-Black tree for preallocated nodes
 *
 * This is a shared-memory compatible balanced binary search tree that
 * does not perform allocations. All nodes must be preallocated by the caller.
 *
 * The tree maintains RB tree invariants:
 * 1. Every node is either red or black
 * 2. The root is black
 * 3. All leaves (NULL) are black
 * 4. Red nodes have black children
 * 5. All paths from root to leaves have the same number of black nodes
 *
 * @tparam NodeT The node type that inherits from rb_node and provides:
 *               - A 'key' member variable for ordering
 *               - Comparison operators (operator<, operator>, operator==)
 * @tparam ATOMIC Whether to use atomic operations for thread-safety
 */
template<typename NodeT, bool ATOMIC = false>
class rb_tree {
 public:
  using KeyT = decltype(std::declval<NodeT>().key);  /**< Deduced key type from NodeT */

 private:
  opt_atomic<size_t, ATOMIC> size_;  /**< Number of nodes in the tree */
  OffsetPtr root_;                /**< Offset pointer to root node */

 public:
  /**
   * Default constructor
   */
  HSHM_CROSS_FUN
  rb_tree() : size_(0), root_(OffsetPtr::GetNull()) {}

  /**
   * Initialize the tree
   */
  HSHM_CROSS_FUN
  void Init() {
    size_.store(0);
    root_ = OffsetPtr::GetNull();
  }

  /**
   * Get the number of nodes in the tree
   */
  HSHM_CROSS_FUN
  size_t size() const {
    return size_.load();
  }

  /**
   * Check if the tree is empty
   */
  HSHM_CROSS_FUN
  bool empty() const {
    return size_.load() == 0;
  }

  /**
   * Get the root pointer (for debugging/inspection)
   */
  HSHM_CROSS_FUN
  OffsetPtr GetRoot() const {
    return root_;
  }

  /**
   * Emplace a preallocated node into the tree
   *
   * @param alloc Allocator for address translation
   * @param node Preallocated node to insert
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void emplace(AllocT *alloc, FullPtr<NodeT> node) {
    node.ptr_->color_ = RBColor::RED;
    node.ptr_->left_ = OffsetPtr::GetNull();
    node.ptr_->right_ = OffsetPtr::GetNull();
    node.ptr_->parent_ = OffsetPtr::GetNull();

    // Standard BST insertion
    if (root_.IsNull()) {
      root_ = node.shm_.off_;
      node.ptr_->color_ = RBColor::BLACK;
      size_.store(size_.load() + 1);
      return;
    }

    // Find insertion point
    OffsetPtr curr_off = root_;
    OffsetPtr parent_off = OffsetPtr::GetNull();

    while (!curr_off.IsNull()) {
      FullPtr<NodeT> curr(alloc, curr_off);
      parent_off = curr_off;

      if (*node.ptr_ < *curr.ptr_) {
        curr_off = curr.ptr_->left_;
      } else if (*node.ptr_ > *curr.ptr_) {
        curr_off = curr.ptr_->right_;
      } else {
        // Key already exists - don't insert duplicate
        return;
      }
    }

    // Insert node
    node.ptr_->parent_ = parent_off;
    FullPtr<NodeT> parent(alloc, parent_off);

    if (*node.ptr_ < *parent.ptr_) {
      parent.ptr_->left_ = node.shm_.off_;
    } else {
      parent.ptr_->right_ = node.shm_.off_;
    }

    size_.store(size_.load() + 1);

    // Fix RB tree properties
    FixInsert(alloc, node);
  }

  /**
   * Remove a node from the tree
   *
   * @param alloc Allocator for address translation
   * @param key Key of the node to remove
   * @return FullPtr to the removed node, or null if not found
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  FullPtr<NodeT> pop(AllocT *alloc, const KeyT &key) {
    if (root_.IsNull()) {
      return FullPtr<NodeT>::GetNull();
    }

    // Find the node to delete
    OffsetPtr node_off = FindNode(alloc, key);
    if (node_off.IsNull()) {
      return FullPtr<NodeT>::GetNull();
    }

    FullPtr<NodeT> node(alloc, node_off);
    FullPtr<NodeT> result = node;  // Save for return

    // Node to be deleted and node to replace it
    OffsetPtr replace_off;
    RBColor original_color = node.ptr_->color_;

    // Track where the black node was actually removed (for FixDeleteFromParent)
    OffsetPtr deleted_parent;
    bool deleted_was_left;

    if (node.ptr_->left_.IsNull()) {
      // Case 1: No left child
      replace_off = node.ptr_->right_;
      deleted_parent = node.ptr_->parent_;
      deleted_was_left = false;
      if (!deleted_parent.IsNull()) {
        FullPtr<NodeT> parent(alloc, deleted_parent);
        deleted_was_left = (parent.ptr_->left_.load() == node_off.load());
      }
      Transplant(alloc, node_off, node.ptr_->right_);
    } else if (node.ptr_->right_.IsNull()) {
      // Case 2: No right child
      replace_off = node.ptr_->left_;
      deleted_parent = node.ptr_->parent_;
      deleted_was_left = false;
      if (!deleted_parent.IsNull()) {
        FullPtr<NodeT> parent(alloc, deleted_parent);
        deleted_was_left = (parent.ptr_->left_.load() == node_off.load());
      }
      Transplant(alloc, node_off, node.ptr_->left_);
    } else {
      // Case 3: Two children - find successor
      OffsetPtr successor_off = Minimum(alloc, node.ptr_->right_);
      FullPtr<NodeT> successor(alloc, successor_off);
      original_color = successor.ptr_->color_;
      replace_off = successor.ptr_->right_;

      // Track where successor was originally (that's where black node is removed)
      deleted_parent = successor.ptr_->parent_;
      deleted_was_left = false;
      if (!deleted_parent.IsNull()) {
        FullPtr<NodeT> parent(alloc, deleted_parent);
        deleted_was_left = (parent.ptr_->left_.load() == successor_off.load());
      }

      if (successor.ptr_->parent_.load() == node_off.load()) {
        // Successor is direct child
        if (!replace_off.IsNull()) {
          FullPtr<NodeT> replace(alloc, replace_off);
          replace.ptr_->parent_ = successor_off;
        }
        // When successor is direct child of node, the deleted position's parent becomes the successor itself after transplant
        deleted_parent = successor_off;
      } else {
        Transplant(alloc, successor_off, successor.ptr_->right_);
        successor.ptr_->right_ = node.ptr_->right_;
        if (!successor.ptr_->right_.IsNull()) {
          FullPtr<NodeT> right_child(alloc, successor.ptr_->right_);
          right_child.ptr_->parent_ = successor_off;
        }
      }

      Transplant(alloc, node_off, successor_off);
      successor.ptr_->left_ = node.ptr_->left_;
      if (!successor.ptr_->left_.IsNull()) {
        FullPtr<NodeT> left_child(alloc, successor.ptr_->left_);
        left_child.ptr_->parent_ = successor_off;
      }
      successor.ptr_->color_ = node.ptr_->color_;
    }

    size_.store(size_.load() - 1);

    // Fix RB properties if we deleted a black node
    if (original_color == RBColor::BLACK) {
      if (!replace_off.IsNull()) {
        FixDelete(alloc, replace_off);
      } else if (!deleted_parent.IsNull() && size_.load() > 0) {
        // Deleted a black leaf - need to fix from parent
        // The "double-black" null is at the position where the black node was removed
        FixDeleteFromParent(alloc, deleted_parent, deleted_was_left);
      }
    }

    // Ensure root is black
    if (!root_.IsNull()) {
      FullPtr<NodeT> root_node(alloc, root_);
      root_node.ptr_->color_ = RBColor::BLACK;
    }

    // Clear the removed node's pointers
    result.ptr_->left_ = OffsetPtr::GetNull();
    result.ptr_->right_ = OffsetPtr::GetNull();
    result.ptr_->parent_ = OffsetPtr::GetNull();

    return result;
  }

  /**
   * Find a node by key
   *
   * @param alloc Allocator for address translation
   * @param key Key to search for
   * @return FullPtr to the node, or null if not found
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  FullPtr<NodeT> find(AllocT *alloc, const KeyT &key) const {
    OffsetPtr node_off = FindNode(alloc, key);
    if (node_off.IsNull()) {
      return FullPtr<NodeT>::GetNull();
    }
    return FullPtr<NodeT>(alloc, node_off);
  }

 private:
  /**
   * Find node by key (internal helper)
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  OffsetPtr FindNode(AllocT *alloc, const KeyT &key) const {
    OffsetPtr curr_off = root_;

    while (!curr_off.IsNull()) {
      FullPtr<NodeT> curr(alloc, curr_off);

      if (key < curr.ptr_->key) {
        curr_off = curr.ptr_->left_;
      } else if (key > curr.ptr_->key) {
        curr_off = curr.ptr_->right_;
      } else {
        return curr_off;
      }
    }

    return OffsetPtr::GetNull();
  }

  /**
   * Find minimum node in subtree
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  OffsetPtr Minimum(AllocT *alloc, OffsetPtr node_off) const {
    while (!node_off.IsNull()) {
      FullPtr<NodeT> node(alloc, node_off);
      if (node.ptr_->left_.IsNull()) {
        break;
      }
      node_off = node.ptr_->left_;
    }
    return node_off;
  }

  /**
   * Replace subtree rooted at u with subtree rooted at v
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void Transplant(AllocT *alloc, OffsetPtr u_off, OffsetPtr v_off) {
    FullPtr<NodeT> u(alloc, u_off);

    if (u.ptr_->parent_.IsNull()) {
      root_ = v_off;
    } else {
      FullPtr<NodeT> parent(alloc, u.ptr_->parent_);
      if (u_off.load() == parent.ptr_->left_.load()) {
        parent.ptr_->left_ = v_off;
      } else {
        parent.ptr_->right_ = v_off;
      }
    }

    if (!v_off.IsNull()) {
      FullPtr<NodeT> v(alloc, v_off);
      v.ptr_->parent_ = u.ptr_->parent_;
    }
  }

  /**
   * Left rotation
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void RotateLeft(AllocT *alloc, OffsetPtr x_off) {
    FullPtr<NodeT> x(alloc, x_off);
    OffsetPtr y_off = x.ptr_->right_;
    FullPtr<NodeT> y(alloc, y_off);

    x.ptr_->right_ = y.ptr_->left_;
    if (!y.ptr_->left_.IsNull()) {
      FullPtr<NodeT> left_child(alloc, y.ptr_->left_);
      left_child.ptr_->parent_ = x_off;
    }

    y.ptr_->parent_ = x.ptr_->parent_;
    if (x.ptr_->parent_.IsNull()) {
      root_ = y_off;
    } else {
      FullPtr<NodeT> parent(alloc, x.ptr_->parent_);
      if (x_off.load() == parent.ptr_->left_.load()) {
        parent.ptr_->left_ = y_off;
      } else {
        parent.ptr_->right_ = y_off;
      }
    }

    y.ptr_->left_ = x_off;
    x.ptr_->parent_ = y_off;
  }

  /**
   * Right rotation
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void RotateRight(AllocT *alloc, OffsetPtr y_off) {
    FullPtr<NodeT> y(alloc, y_off);
    OffsetPtr x_off = y.ptr_->left_;
    FullPtr<NodeT> x(alloc, x_off);

    y.ptr_->left_ = x.ptr_->right_;
    if (!x.ptr_->right_.IsNull()) {
      FullPtr<NodeT> right_child(alloc, x.ptr_->right_);
      right_child.ptr_->parent_ = y_off;
    }

    x.ptr_->parent_ = y.ptr_->parent_;
    if (y.ptr_->parent_.IsNull()) {
      root_ = x_off;
    } else {
      FullPtr<NodeT> parent(alloc, y.ptr_->parent_);
      if (y_off.load() == parent.ptr_->left_.load()) {
        parent.ptr_->left_ = x_off;
      } else {
        parent.ptr_->right_ = x_off;
      }
    }

    x.ptr_->right_ = y_off;
    y.ptr_->parent_ = x_off;
  }

  /**
   * Fix RB tree properties after insertion
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void FixInsert(AllocT *alloc, FullPtr<NodeT> node) {
    OffsetPtr node_off = node.shm_.off_;

    while (!node.ptr_->parent_.IsNull()) {
      FullPtr<NodeT> parent(alloc, node.ptr_->parent_);
      if (parent.ptr_->color_ == RBColor::BLACK) {
        break;
      }

      if (parent.ptr_->parent_.IsNull()) {
        break;
      }

      FullPtr<NodeT> grandparent(alloc, parent.ptr_->parent_);

      if (node.ptr_->parent_.load() == grandparent.ptr_->left_.load()) {
        // Parent is left child
        OffsetPtr uncle_off = grandparent.ptr_->right_;

        if (!uncle_off.IsNull()) {
          FullPtr<NodeT> uncle(alloc, uncle_off);
          if (uncle.ptr_->color_ == RBColor::RED) {
            // Case 1: Uncle is red
            parent.ptr_->color_ = RBColor::BLACK;
            uncle.ptr_->color_ = RBColor::BLACK;
            grandparent.ptr_->color_ = RBColor::RED;
            node_off = parent.ptr_->parent_;
            node = FullPtr<NodeT>(alloc, node_off);
            continue;
          }
        }

        if (node_off.load() == parent.ptr_->right_.load()) {
          // Case 2: Node is right child
          node_off = node.ptr_->parent_;
          RotateLeft(alloc, node_off);
          node = FullPtr<NodeT>(alloc, node_off);
          parent = FullPtr<NodeT>(alloc, node.ptr_->parent_);
          grandparent = FullPtr<NodeT>(alloc, parent.ptr_->parent_);
        }

        // Case 3: Node is left child
        parent.ptr_->color_ = RBColor::BLACK;
        grandparent.ptr_->color_ = RBColor::RED;
        RotateRight(alloc, parent.ptr_->parent_);
      } else {
        // Parent is right child (symmetric)
        OffsetPtr uncle_off = grandparent.ptr_->left_;

        if (!uncle_off.IsNull()) {
          FullPtr<NodeT> uncle(alloc, uncle_off);
          if (uncle.ptr_->color_ == RBColor::RED) {
            parent.ptr_->color_ = RBColor::BLACK;
            uncle.ptr_->color_ = RBColor::BLACK;
            grandparent.ptr_->color_ = RBColor::RED;
            node_off = parent.ptr_->parent_;
            node = FullPtr<NodeT>(alloc, node_off);
            continue;
          }
        }

        if (node_off.load() == parent.ptr_->left_.load()) {
          node_off = node.ptr_->parent_;
          RotateRight(alloc, node_off);
          node = FullPtr<NodeT>(alloc, node_off);
          parent = FullPtr<NodeT>(alloc, node.ptr_->parent_);
          grandparent = FullPtr<NodeT>(alloc, parent.ptr_->parent_);
        }

        parent.ptr_->color_ = RBColor::BLACK;
        grandparent.ptr_->color_ = RBColor::RED;
        RotateLeft(alloc, parent.ptr_->parent_);
      }
    }

    // Ensure root is black
    if (!root_.IsNull()) {
      FullPtr<NodeT> root(alloc, root_);
      root.ptr_->color_ = RBColor::BLACK;
    }
  }

  /**
   * Fix RB tree properties after deletion
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void FixDelete(AllocT *alloc, OffsetPtr node_off) {
    while (node_off.load() != root_.load()) {
      FullPtr<NodeT> node(alloc, node_off);
      if (node.ptr_->color_ == RBColor::RED) {
        break;
      }

      if (node.ptr_->parent_.IsNull()) {
        break;
      }

      FullPtr<NodeT> parent(alloc, node.ptr_->parent_);

      if (node_off.load() == parent.ptr_->left_.load()) {
        OffsetPtr sibling_off = parent.ptr_->right_;
        if (sibling_off.IsNull()) {
          node_off = node.ptr_->parent_;
          continue;
        }

        FullPtr<NodeT> sibling(alloc, sibling_off);

        if (sibling.ptr_->color_ == RBColor::RED) {
          sibling.ptr_->color_ = RBColor::BLACK;
          parent.ptr_->color_ = RBColor::RED;
          RotateLeft(alloc, node.ptr_->parent_);
          sibling_off = parent.ptr_->right_;
          sibling = FullPtr<NodeT>(alloc, sibling_off);
        }

        bool left_black = sibling.ptr_->left_.IsNull();
        bool right_black = sibling.ptr_->right_.IsNull();

        if (!sibling.ptr_->left_.IsNull()) {
          FullPtr<NodeT> left(alloc, sibling.ptr_->left_);
          left_black = (left.ptr_->color_ == RBColor::BLACK);
        }
        if (!sibling.ptr_->right_.IsNull()) {
          FullPtr<NodeT> right(alloc, sibling.ptr_->right_);
          right_black = (right.ptr_->color_ == RBColor::BLACK);
        }

        if (left_black && right_black) {
          sibling.ptr_->color_ = RBColor::RED;
          node_off = node.ptr_->parent_;
        } else {
          if (right_black) {
            if (!sibling.ptr_->left_.IsNull()) {
              FullPtr<NodeT> left(alloc, sibling.ptr_->left_);
              left.ptr_->color_ = RBColor::BLACK;
            }
            sibling.ptr_->color_ = RBColor::RED;
            RotateRight(alloc, sibling_off);
            sibling_off = parent.ptr_->right_;
            sibling = FullPtr<NodeT>(alloc, sibling_off);
          }

          sibling.ptr_->color_ = parent.ptr_->color_;
          parent.ptr_->color_ = RBColor::BLACK;
          if (!sibling.ptr_->right_.IsNull()) {
            FullPtr<NodeT> right(alloc, sibling.ptr_->right_);
            right.ptr_->color_ = RBColor::BLACK;
          }
          RotateLeft(alloc, node.ptr_->parent_);
          node_off = root_;
        }
      } else {
        // Symmetric case
        OffsetPtr sibling_off = parent.ptr_->left_;
        if (sibling_off.IsNull()) {
          node_off = node.ptr_->parent_;
          continue;
        }

        FullPtr<NodeT> sibling(alloc, sibling_off);

        if (sibling.ptr_->color_ == RBColor::RED) {
          sibling.ptr_->color_ = RBColor::BLACK;
          parent.ptr_->color_ = RBColor::RED;
          RotateRight(alloc, node.ptr_->parent_);
          sibling_off = parent.ptr_->left_;
          sibling = FullPtr<NodeT>(alloc, sibling_off);
        }

        bool left_black = sibling.ptr_->left_.IsNull();
        bool right_black = sibling.ptr_->right_.IsNull();

        if (!sibling.ptr_->left_.IsNull()) {
          FullPtr<NodeT> left(alloc, sibling.ptr_->left_);
          left_black = (left.ptr_->color_ == RBColor::BLACK);
        }
        if (!sibling.ptr_->right_.IsNull()) {
          FullPtr<NodeT> right(alloc, sibling.ptr_->right_);
          right_black = (right.ptr_->color_ == RBColor::BLACK);
        }

        if (left_black && right_black) {
          sibling.ptr_->color_ = RBColor::RED;
          node_off = node.ptr_->parent_;
        } else {
          if (left_black) {
            if (!sibling.ptr_->right_.IsNull()) {
              FullPtr<NodeT> right(alloc, sibling.ptr_->right_);
              right.ptr_->color_ = RBColor::BLACK;
            }
            sibling.ptr_->color_ = RBColor::RED;
            RotateLeft(alloc, sibling_off);
            sibling_off = parent.ptr_->left_;
            sibling = FullPtr<NodeT>(alloc, sibling_off);
          }

          sibling.ptr_->color_ = parent.ptr_->color_;
          parent.ptr_->color_ = RBColor::BLACK;
          if (!sibling.ptr_->left_.IsNull()) {
            FullPtr<NodeT> left(alloc, sibling.ptr_->left_);
            left.ptr_->color_ = RBColor::BLACK;
          }
          RotateRight(alloc, node.ptr_->parent_);
          node_off = root_;
        }
      }
    }

    if (!node_off.IsNull()) {
      FullPtr<NodeT> node(alloc, node_off);
      node.ptr_->color_ = RBColor::BLACK;
    }
  }

  /**
   * Fix RB properties when a black leaf was deleted (replacement is NULL)
   * This is called with the parent of the deleted node
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void FixDeleteFromParent(AllocT *alloc, OffsetPtr parent_off, bool deleted_was_left) {
    while (true) {
      FullPtr<NodeT> parent(alloc, parent_off);

      if (deleted_was_left) {
        // Deleted node was left child
        OffsetPtr sibling_off = parent.ptr_->right_;
        if (sibling_off.IsNull()) break;  // Shouldn't happen in valid RB tree

        FullPtr<NodeT> sibling(alloc, sibling_off);

        // Case 1: Red sibling
        if (sibling.ptr_->color_ == RBColor::RED) {
          sibling.ptr_->color_ = RBColor::BLACK;
          parent.ptr_->color_ = RBColor::RED;
          RotateLeft(alloc, parent_off);
          sibling_off = parent.ptr_->right_;
          if (sibling_off.IsNull()) break;
          sibling = FullPtr<NodeT>(alloc, sibling_off);
        }

        // Check sibling's children colors
        bool left_black = sibling.ptr_->left_.IsNull();
        bool right_black = sibling.ptr_->right_.IsNull();
        if (!sibling.ptr_->left_.IsNull()) {
          FullPtr<NodeT> left(alloc, sibling.ptr_->left_);
          left_black = (left.ptr_->color_ == RBColor::BLACK);
        }
        if (!sibling.ptr_->right_.IsNull()) {
          FullPtr<NodeT> right(alloc, sibling.ptr_->right_);
          right_black = (right.ptr_->color_ == RBColor::BLACK);
        }

        // Case 2: Sibling and both nephews are black
        if (left_black && right_black) {
          sibling.ptr_->color_ = RBColor::RED;
          if (parent.ptr_->color_ == RBColor::RED) {
            parent.ptr_->color_ = RBColor::BLACK;
            return;
          }
          // If parent is root, we're done (reduced black height of whole tree)
          if (parent_off.load() == root_.load()) {
            return;
          }
          // Continue fixing from parent
          if (parent.ptr_->parent_.IsNull()) break;
          OffsetPtr grandparent_off = parent.ptr_->parent_;
          FullPtr<NodeT> grandparent(alloc, grandparent_off);
          deleted_was_left = (grandparent.ptr_->left_.load() == parent_off.load());
          parent_off = grandparent_off;
        } else {
          // Case 3: Right nephew is black (left is red)
          if (right_black) {
            if (!sibling.ptr_->left_.IsNull()) {
              FullPtr<NodeT> left(alloc, sibling.ptr_->left_);
              left.ptr_->color_ = RBColor::BLACK;
            }
            sibling.ptr_->color_ = RBColor::RED;
            RotateRight(alloc, sibling_off);
            sibling_off = parent.ptr_->right_;
            sibling = FullPtr<NodeT>(alloc, sibling_off);
          }

          // Case 4: Right nephew is red
          sibling.ptr_->color_ = parent.ptr_->color_;
          parent.ptr_->color_ = RBColor::BLACK;
          if (!sibling.ptr_->right_.IsNull()) {
            FullPtr<NodeT> right(alloc, sibling.ptr_->right_);
            right.ptr_->color_ = RBColor::BLACK;
          }
          RotateLeft(alloc, parent_off);
          return;
        }
      } else {
        // Deleted node was right child (symmetric)
        OffsetPtr sibling_off = parent.ptr_->left_;
        if (sibling_off.IsNull()) break;

        FullPtr<NodeT> sibling(alloc, sibling_off);

        if (sibling.ptr_->color_ == RBColor::RED) {
          sibling.ptr_->color_ = RBColor::BLACK;
          parent.ptr_->color_ = RBColor::RED;
          RotateRight(alloc, parent_off);
          sibling_off = parent.ptr_->left_;
          if (sibling_off.IsNull()) break;
          sibling = FullPtr<NodeT>(alloc, sibling_off);
        }

        bool left_black = sibling.ptr_->left_.IsNull();
        bool right_black = sibling.ptr_->right_.IsNull();
        if (!sibling.ptr_->left_.IsNull()) {
          FullPtr<NodeT> left(alloc, sibling.ptr_->left_);
          left_black = (left.ptr_->color_ == RBColor::BLACK);
        }
        if (!sibling.ptr_->right_.IsNull()) {
          FullPtr<NodeT> right(alloc, sibling.ptr_->right_);
          right_black = (right.ptr_->color_ == RBColor::BLACK);
        }

        if (left_black && right_black) {
          sibling.ptr_->color_ = RBColor::RED;
          if (parent.ptr_->color_ == RBColor::RED) {
            parent.ptr_->color_ = RBColor::BLACK;
            return;
          }
          // If parent is root, we're done (reduced black height of whole tree)
          if (parent_off.load() == root_.load()) {
            return;
          }
          if (parent.ptr_->parent_.IsNull()) break;
          OffsetPtr grandparent_off = parent.ptr_->parent_;
          FullPtr<NodeT> grandparent(alloc, grandparent_off);
          deleted_was_left = (grandparent.ptr_->left_.load() == parent_off.load());
          parent_off = grandparent_off;
        } else {
          if (left_black) {
            if (!sibling.ptr_->right_.IsNull()) {
              FullPtr<NodeT> right(alloc, sibling.ptr_->right_);
              right.ptr_->color_ = RBColor::BLACK;
            }
            sibling.ptr_->color_ = RBColor::RED;
            RotateLeft(alloc, sibling_off);
            sibling_off = parent.ptr_->left_;
            sibling = FullPtr<NodeT>(alloc, sibling_off);
          }

          sibling.ptr_->color_ = parent.ptr_->color_;
          parent.ptr_->color_ = RBColor::BLACK;
          if (!sibling.ptr_->left_.IsNull()) {
            FullPtr<NodeT> left(alloc, sibling.ptr_->left_);
            left.ptr_->color_ = RBColor::BLACK;
          }
          RotateRight(alloc, parent_off);
          return;
        }
      }
    }
  }

};

}  // namespace hshm::ipc::pre

#endif  // HSHM_DATA_STRUCTURES_IPC_RB_TREE_PRE_H_
