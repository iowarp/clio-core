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

#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <algorithm>
#include <functional>
#include "hermes_shm/data_structures/ipc/rb_tree_pre.h"
#include "hermes_shm/memory/backend/malloc_backend.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"

using namespace hshm::ipc;

/**
 * Test node structure that inherits from rb_node
 */
template<typename KeyT>
struct TestRBNode : public pre::rb_node {
  KeyT key;      // Key for ordering (required by rb_tree)
  int value_;    // Test data

  TestRBNode() : pre::rb_node(), key(), value_(0) {}
  explicit TestRBNode(const KeyT &k, int val = 0) : pre::rb_node(), key(k), value_(val) {}

  // Comparison operators (required by rb_tree)
  bool operator<(const TestRBNode &other) const { return key < other.key; }
  bool operator>(const TestRBNode &other) const { return key > other.key; }
  bool operator==(const TestRBNode &other) const { return key == other.key; }
};

/**
 * Helper function to create an ArenaAllocator for testing
 */
template<bool ATOMIC>
ArenaAllocator<ATOMIC>* CreateTestAllocator(MallocBackend &backend, size_t arena_size) {
  backend.shm_init(MemoryBackendId(0, 0), arena_size);

  // Use MakeAlloc to create the allocator properly
  return backend.MakeAlloc<ArenaAllocator<ATOMIC>>();
}

/**
 * Helper to verify RB tree properties
 */
template<typename NodeT, typename AllocT>
bool VerifyRBProperties(AllocT *alloc, pre::rb_tree<NodeT, false> &tree) {
  using KeyT = typename pre::rb_tree<NodeT, false>::KeyT;

  if (tree.empty()) {
    return true;
  }

  FullPtr<NodeT> root(alloc, OffsetPtr<NodeT>(tree.GetRoot().load()));

  // Property 2: Root must be black
  if (root.ptr_->color_ != pre::RBColor::BLACK) {
    return false;
  }

  // Helper lambda to check properties recursively
  std::function<int(OffsetPtr<NodeT>, KeyT*, KeyT*)> check_node =
      [&](OffsetPtr<NodeT> node_off, KeyT *min_key, KeyT *max_key) -> int {
    if (node_off.IsNull()) {
      return 1;  // Null nodes are black (property 3)
    }

    FullPtr<NodeT> node(alloc, node_off);

    // Check BST property
    if (min_key && node.ptr_->key <= *min_key) return -1;
    if (max_key && node.ptr_->key >= *max_key) return -1;

    // Property 4: Red nodes have black children
    if (node.ptr_->color_ == pre::RBColor::RED) {
      if (!node.ptr_->left_.IsNull()) {
        FullPtr<NodeT> left(alloc, OffsetPtr<NodeT>(node.ptr_->left_.load()));
        if (left.ptr_->color_ == pre::RBColor::RED) return -1;
      }
      if (!node.ptr_->right_.IsNull()) {
        FullPtr<NodeT> right(alloc, OffsetPtr<NodeT>(node.ptr_->right_.load()));
        if (right.ptr_->color_ == pre::RBColor::RED) return -1;
      }
    }

    // Property 5: Same number of black nodes on all paths
    int left_black = check_node(OffsetPtr<NodeT>(node.ptr_->left_), min_key, &node.ptr_->key);
    int right_black = check_node(OffsetPtr<NodeT>(node.ptr_->right_), &node.ptr_->key, max_key);

    if (left_black == -1 || right_black == -1 || left_black != right_black) {
      return -1;
    }

    return left_black + (node.ptr_->color_ == pre::RBColor::BLACK ? 1 : 0);
  };

  return check_node(OffsetPtr<NodeT>(tree.GetRoot().load()), nullptr, nullptr) != -1;
}

TEST_CASE("rb_tree_pre - Basic Operations", "[rb_tree_pre]") {
  MallocBackend backend;
  size_t arena_size = 10 * 1024 * 1024;  // 10 MB
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);
  

  SECTION("Initialization") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    REQUIRE(tree.size() == 0);
    REQUIRE(tree.empty());
    REQUIRE(tree.GetRoot().IsNull());
  }

  SECTION("Single element insert and find") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    // Allocate a test node
    auto node_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
    node_ptr.ptr_->key = 42;
    node_ptr.ptr_->value_ = 100;

    // Insert the node (TestRBNode inherits from rb_node, so we can cast)
    hipc::ShmPtrBase<TestRBNode<int>> test_shm(node_ptr.shm_.alloc_id_, node_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr(alloc, test_shm);
    test_ptr.ptr_ = node_ptr.ptr_;
    tree.emplace(alloc, test_ptr);

    REQUIRE(tree.size() == 1);
    REQUIRE_FALSE(tree.empty());
    REQUIRE(VerifyRBProperties(alloc, tree));

    // Find the node
    auto found = tree.find(alloc, 42);
    REQUIRE_FALSE(found.IsNull());
    REQUIRE(found.ptr_->value_ == 100);
  }

  SECTION("Multiple elements in order") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    // Insert nodes in ascending order
    for (int i = 0; i < 10; ++i) {
      auto node_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
      node_ptr.ptr_->key = i;
      node_ptr.ptr_->value_ = i * 10;

      hipc::ShmPtrBase<TestRBNode<int>> test_shm(node_ptr.shm_.alloc_id_, node_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr(alloc, test_shm);
    test_ptr.ptr_ = node_ptr.ptr_;
      tree.emplace(alloc, test_ptr);
    }

    REQUIRE(tree.size() == 10);
    REQUIRE(VerifyRBProperties(alloc, tree));

    // Verify all nodes can be found
    for (int i = 0; i < 10; ++i) {
      auto found = tree.find(alloc, i);
      REQUIRE_FALSE(found.IsNull());
      REQUIRE(found.ptr_->value_ == i * 10);
    }
  }

  SECTION("Multiple elements reverse order") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    // Insert nodes in descending order
    for (int i = 9; i >= 0; --i) {
      auto node_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
      node_ptr.ptr_->key = i;
      node_ptr.ptr_->value_ = i * 10;

      hipc::ShmPtrBase<TestRBNode<int>> test_shm(node_ptr.shm_.alloc_id_, node_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr(alloc, test_shm);
    test_ptr.ptr_ = node_ptr.ptr_;
      tree.emplace(alloc, test_ptr);
    }

    REQUIRE(tree.size() == 10);
    REQUIRE(VerifyRBProperties(alloc, tree));
  }

  SECTION("Random insertion order") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    std::vector<int> keys = {50, 25, 75, 10, 30, 60, 90, 5, 15, 27, 35, 55, 65, 85, 95};

    for (int key : keys) {
      auto node_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
      node_ptr.ptr_->key = key;
      node_ptr.ptr_->value_ = key;

      hipc::ShmPtrBase<TestRBNode<int>> test_shm(node_ptr.shm_.alloc_id_, node_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr(alloc, test_shm);
    test_ptr.ptr_ = node_ptr.ptr_;
      tree.emplace(alloc, test_ptr);

      REQUIRE(VerifyRBProperties(alloc, tree));
    }

    REQUIRE(tree.size() == keys.size());

    // Verify all keys can be found
    for (int key : keys) {
      auto found = tree.find(alloc, key);
      REQUIRE_FALSE(found.IsNull());
    }
  }

  SECTION("Duplicate key insertion") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    // Insert node with key 42
    auto node1_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
    node1_ptr.ptr_->key = 42;
    node1_ptr.ptr_->value_ = 100;

    hipc::ShmPtrBase<TestRBNode<int>> test_shm1(node1_ptr.shm_.alloc_id_, node1_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr1(alloc, test_shm1);
    test_ptr1.ptr_ = node1_ptr.ptr_;
    tree.emplace(alloc, test_ptr1);

    REQUIRE(tree.size() == 1);

    // Try to insert another node with same key
    auto node2_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
    node2_ptr.ptr_->key = 42;
    node2_ptr.ptr_->value_ = 200;

    hipc::ShmPtrBase<TestRBNode<int>> test_shm2(node2_ptr.shm_.alloc_id_, node2_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr2(alloc, test_shm2);
    test_ptr2.ptr_ = node2_ptr.ptr_;
    tree.emplace(alloc, test_ptr2);

    // Size should remain 1 (duplicate not inserted)
    REQUIRE(tree.size() == 1);

    // Original value should be preserved
    auto found = tree.find(alloc, 42);
    REQUIRE(found.ptr_->value_ == 100);
  }

}

TEST_CASE("rb_tree_pre - Deletion", "[rb_tree_pre]") {
  MallocBackend backend;
  size_t arena_size = 10 * 1024 * 1024;
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);
  

  SECTION("Delete from empty tree") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    auto popped = tree.pop(alloc, 42);
    REQUIRE(popped.IsNull());
    REQUIRE(tree.size() == 0);
  }

  SECTION("Delete single element") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    auto node_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
    node_ptr.ptr_->key = 42;
    node_ptr.ptr_->value_ = 100;

    hipc::ShmPtrBase<TestRBNode<int>> test_shm(node_ptr.shm_.alloc_id_, node_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr(alloc, test_shm);
    test_ptr.ptr_ = node_ptr.ptr_;
    tree.emplace(alloc, test_ptr);

    REQUIRE(tree.size() == 1);

    auto popped = tree.pop(alloc, 42);
    REQUIRE_FALSE(popped.IsNull());
    REQUIRE(tree.size() == 0);
    REQUIRE(tree.empty());

    REQUIRE(popped.ptr_->value_ == 100);
  }

  SECTION("Delete non-existent key") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    // Insert a few nodes
    for (int i = 0; i < 5; ++i) {
      auto node_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
      node_ptr.ptr_->key = i * 10;
      node_ptr.ptr_->value_ = i;

      hipc::ShmPtrBase<TestRBNode<int>> test_shm(node_ptr.shm_.alloc_id_, node_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr(alloc, test_shm);
    test_ptr.ptr_ = node_ptr.ptr_;
      tree.emplace(alloc, test_ptr);
    }

    size_t initial_size = tree.size();

    // Try to delete non-existent key
    auto popped = tree.pop(alloc, 99);
    REQUIRE(popped.IsNull());
    REQUIRE(tree.size() == initial_size);
  }

  SECTION("Delete all elements") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    std::vector<int> keys = {50, 25, 75, 10, 30, 60, 90};

    // Insert all
    for (int key : keys) {
      auto node_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
      node_ptr.ptr_->key = key;
      node_ptr.ptr_->value_ = key;

      hipc::ShmPtrBase<TestRBNode<int>> test_shm(node_ptr.shm_.alloc_id_, node_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr(alloc, test_shm);
    test_ptr.ptr_ = node_ptr.ptr_;
      tree.emplace(alloc, test_ptr);
    }

    REQUIRE(tree.size() == keys.size());

    // Delete all in different order
    std::vector<int> delete_order = {25, 90, 50, 10, 75, 60, 30};
    for (int key : delete_order) {
      auto popped = tree.pop(alloc, key);
      REQUIRE_FALSE(popped.IsNull());
      bool valid = VerifyRBProperties(alloc, tree);
      if (!valid) {
        INFO("RB properties violated after deleting key: " << key);
        INFO("Remaining size: " << tree.size());
      }
      REQUIRE(valid);
    }

    REQUIRE(tree.size() == 0);
    REQUIRE(tree.empty());
  }

  SECTION("Interleaved insert and delete") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    // Insert 5 nodes
    for (int i = 0; i < 5; ++i) {
      auto node_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
      node_ptr.ptr_->key = i;
      node_ptr.ptr_->value_ = i;

      hipc::ShmPtrBase<TestRBNode<int>> test_shm(node_ptr.shm_.alloc_id_, node_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr(alloc, test_shm);
    test_ptr.ptr_ = node_ptr.ptr_;
      tree.emplace(alloc, test_ptr);
    }

    // Delete 2
    tree.pop(alloc, 1);
    tree.pop(alloc, 3);
    REQUIRE(tree.size() == 3);
    REQUIRE(VerifyRBProperties(alloc, tree));

    // Insert 3 more
    for (int i = 10; i < 13; ++i) {
      auto node_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
      node_ptr.ptr_->key = i;
      node_ptr.ptr_->value_ = i;

      hipc::ShmPtrBase<TestRBNode<int>> test_shm(node_ptr.shm_.alloc_id_, node_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr(alloc, test_shm);
    test_ptr.ptr_ = node_ptr.ptr_;
      tree.emplace(alloc, test_ptr);
    }

    REQUIRE(tree.size() == 6);
    REQUIRE(VerifyRBProperties(alloc, tree));

    // Verify correct keys remain
    REQUIRE_FALSE(tree.find(alloc, 0).IsNull());
    REQUIRE(tree.find(alloc, 1).IsNull());
    REQUIRE_FALSE(tree.find(alloc, 2).IsNull());
    REQUIRE(tree.find(alloc, 3).IsNull());
    REQUIRE_FALSE(tree.find(alloc, 4).IsNull());
    REQUIRE_FALSE(tree.find(alloc, 10).IsNull());
    REQUIRE_FALSE(tree.find(alloc, 11).IsNull());
    REQUIRE_FALSE(tree.find(alloc, 12).IsNull());
  }

}

TEST_CASE("rb_tree_pre - Large Tree", "[rb_tree_pre]") {
  MallocBackend backend;
  size_t arena_size = 50 * 1024 * 1024;  // 50 MB
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);
  

  SECTION("1000 elements sequential") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    const int NUM_NODES = 1000;

    // Insert
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
      node_ptr.ptr_->key = i;
      node_ptr.ptr_->value_ = i;

      hipc::ShmPtrBase<TestRBNode<int>> test_shm(node_ptr.shm_.alloc_id_, node_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr(alloc, test_shm);
    test_ptr.ptr_ = node_ptr.ptr_;
      tree.emplace(alloc, test_ptr);
    }

    REQUIRE(tree.size() == NUM_NODES);
    REQUIRE(VerifyRBProperties(alloc, tree));

    // Verify all present
    for (int i = 0; i < NUM_NODES; ++i) {
      REQUIRE_FALSE(tree.find(alloc, i).IsNull());
    }

    // Delete half
    for (int i = 0; i < NUM_NODES; i += 2) {
      tree.pop(alloc, i);
    }

    REQUIRE(tree.size() == NUM_NODES / 2);
    REQUIRE(VerifyRBProperties(alloc, tree));
  }

}

TEST_CASE("rb_tree_pre - Atomic Version", "[rb_tree_pre][atomic]") {
  MallocBackend backend;
  size_t arena_size = 10 * 1024 * 1024;
  auto *alloc = CreateTestAllocator<true>(backend, arena_size);


  SECTION("Basic atomic operations") {
    pre::rb_tree<TestRBNode<int>, true> tree;
    tree.Init();

    // Insert nodes
    for (int i = 0; i < 20; ++i) {
      auto node_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
      node_ptr.ptr_->key = i;
      node_ptr.ptr_->value_ = i * 2;

      hipc::ShmPtrBase<TestRBNode<int>> test_shm(node_ptr.shm_.alloc_id_, node_ptr.shm_.off_.load());
    FullPtr<TestRBNode<int>> test_ptr(alloc, test_shm);
    test_ptr.ptr_ = node_ptr.ptr_;
      tree.emplace(alloc, test_ptr);
    }

    REQUIRE(tree.size() == 20);

    // Delete some nodes
    for (int i = 0; i < 20; i += 3) {
      auto popped = tree.pop(alloc, i);
      REQUIRE_FALSE(popped.IsNull());
    }

    REQUIRE(tree.size() == 13);
  }

}

TEST_CASE("rb_tree_pre - Deletion Edge Cases", "[rb_tree_pre][deletion]") {
  MallocBackend backend;
  size_t arena_size = 10 * 1024 * 1024;
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);

  SECTION("Delete root node") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    // Insert root and two children
    auto root = alloc->Allocate<TestRBNode<int>>(sizeof(TestRBNode<int>));
    root.ptr_->key = 50;
    hipc::ShmPtrBase<TestRBNode<int>> root_shm(root.shm_.alloc_id_, root.shm_.off_.load());
    FullPtr<TestRBNode<int>> root_ptr(alloc, root_shm);
    root_ptr.ptr_ = root.ptr_;
    tree.emplace(alloc, root_ptr);

    auto left = alloc->Allocate<TestRBNode<int>>(sizeof(TestRBNode<int>));
    left.ptr_->key = 25;
    hipc::ShmPtrBase<TestRBNode<int>> left_shm(left.shm_.alloc_id_, left.shm_.off_.load());
    FullPtr<TestRBNode<int>> left_ptr(alloc, left_shm);
    left_ptr.ptr_ = left.ptr_;
    tree.emplace(alloc, left_ptr);

    auto right = alloc->Allocate<TestRBNode<int>>(sizeof(TestRBNode<int>));
    right.ptr_->key = 75;
    hipc::ShmPtrBase<TestRBNode<int>> right_shm(right.shm_.alloc_id_, right.shm_.off_.load());
    FullPtr<TestRBNode<int>> right_ptr(alloc, right_shm);
    right_ptr.ptr_ = right.ptr_;
    tree.emplace(alloc, right_ptr);

    REQUIRE(tree.size() == 3);
    REQUIRE(VerifyRBProperties(alloc, tree));

    // Delete root
    auto popped = tree.pop(alloc, 50);
    REQUIRE_FALSE(popped.IsNull());
    REQUIRE(tree.size() == 2);
    REQUIRE(VerifyRBProperties(alloc, tree));

    // Both children should still be findable
    REQUIRE_FALSE(tree.find(alloc, 25).IsNull());
    REQUIRE_FALSE(tree.find(alloc, 75).IsNull());
  }

  SECTION("Delete node with only left child") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    // Create structure: 50 -> 25 -> 10
    std::vector<int> keys = {50, 25, 75, 10};
    for (int key : keys) {
      auto node = alloc->Allocate<TestRBNode<int>>(sizeof(TestRBNode<int>));
      node.ptr_->key = key;
      hipc::ShmPtrBase<TestRBNode<int>> shm(node.shm_.alloc_id_, node.shm_.off_.load());
      FullPtr<TestRBNode<int>> ptr(alloc, shm);
      ptr.ptr_ = node.ptr_;
      tree.emplace(alloc, ptr);
    }

    REQUIRE(tree.size() == 4);

    // Delete 25 (has left child 10)
    auto popped = tree.pop(alloc, 25);
    REQUIRE_FALSE(popped.IsNull());
    REQUIRE(tree.size() == 3);
    REQUIRE(VerifyRBProperties(alloc, tree));
    REQUIRE_FALSE(tree.find(alloc, 10).IsNull());
  }

  SECTION("Delete node with only right child") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    // Create structure with right-only child
    std::vector<int> keys = {50, 25, 75, 90};
    for (int key : keys) {
      auto node = alloc->Allocate<TestRBNode<int>>(sizeof(TestRBNode<int>));
      node.ptr_->key = key;
      hipc::ShmPtrBase<TestRBNode<int>> shm(node.shm_.alloc_id_, node.shm_.off_.load());
      FullPtr<TestRBNode<int>> ptr(alloc, shm);
      ptr.ptr_ = node.ptr_;
      tree.emplace(alloc, ptr);
    }

    REQUIRE(tree.size() == 4);

    // Delete 75 (has right child 90)
    auto popped = tree.pop(alloc, 75);
    REQUIRE_FALSE(popped.IsNull());
    REQUIRE(tree.size() == 3);
    REQUIRE(VerifyRBProperties(alloc, tree));
    REQUIRE_FALSE(tree.find(alloc, 90).IsNull());
  }

  SECTION("Delete node with two children - successor is direct child") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    // Create structure where successor is direct right child
    std::vector<int> keys = {50, 25, 75, 60};
    for (int key : keys) {
      auto node = alloc->Allocate<TestRBNode<int>>(sizeof(TestRBNode<int>));
      node.ptr_->key = key;
      hipc::ShmPtrBase<TestRBNode<int>> shm(node.shm_.alloc_id_, node.shm_.off_.load());
      FullPtr<TestRBNode<int>> ptr(alloc, shm);
      ptr.ptr_ = node.ptr_;
      tree.emplace(alloc, ptr);
    }

    // Delete 50 - successor 60 is left child of 75
    auto popped = tree.pop(alloc, 50);
    REQUIRE_FALSE(popped.IsNull());
    REQUIRE(VerifyRBProperties(alloc, tree));
  }

  SECTION("Delete black leaf causing fix") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    // Insert nodes in order that creates black leaves
    std::vector<int> keys = {50, 25, 75, 10, 30, 60, 90, 5};
    for (int key : keys) {
      auto node = alloc->Allocate<TestRBNode<int>>(sizeof(TestRBNode<int>));
      node.ptr_->key = key;
      hipc::ShmPtrBase<TestRBNode<int>> shm(node.shm_.alloc_id_, node.shm_.off_.load());
      FullPtr<TestRBNode<int>> ptr(alloc, shm);
      ptr.ptr_ = node.ptr_;
      tree.emplace(alloc, ptr);
    }

    REQUIRE(VerifyRBProperties(alloc, tree));

    // Delete various nodes and verify RB properties after each
    std::vector<int> delete_order = {5, 30, 10, 90, 60};
    for (int key : delete_order) {
      auto popped = tree.pop(alloc, key);
      REQUIRE_FALSE(popped.IsNull());
      REQUIRE(VerifyRBProperties(alloc, tree));
    }
  }

  SECTION("Stress test - random deletion order") {
    pre::rb_tree<TestRBNode<int>, false> tree;
    tree.Init();

    const int NUM_NODES = 100;
    std::vector<int> keys;
    for (int i = 0; i < NUM_NODES; ++i) {
      keys.push_back(i);
      auto node = alloc->Allocate<TestRBNode<int>>(sizeof(TestRBNode<int>));
      node.ptr_->key = i;
      hipc::ShmPtrBase<TestRBNode<int>> shm(node.shm_.alloc_id_, node.shm_.off_.load());
      FullPtr<TestRBNode<int>> ptr(alloc, shm);
      ptr.ptr_ = node.ptr_;
      tree.emplace(alloc, ptr);
    }

    REQUIRE(tree.size() == NUM_NODES);
    REQUIRE(VerifyRBProperties(alloc, tree));

    // Delete in reverse order (different pattern than insertion)
    for (int i = NUM_NODES - 1; i >= 0; --i) {
      auto popped = tree.pop(alloc, i);
      REQUIRE_FALSE(popped.IsNull());
      if (tree.size() > 0) {
        REQUIRE(VerifyRBProperties(alloc, tree));
      }
    }

    REQUIRE(tree.empty());
  }
}

TEST_CASE("rb_tree_pre - String Keys", "[rb_tree_pre][string_keys]") {
  MallocBackend backend;
  size_t arena_size = 10 * 1024 * 1024;
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);

  SECTION("String key operations") {
    pre::rb_tree<TestRBNode<std::string>, false> tree;
    tree.Init();

    std::vector<std::string> keys = {"apple", "banana", "cherry", "date", "elderberry"};

    for (const auto& key : keys) {
      auto node = alloc->Allocate<TestRBNode<std::string>>(sizeof(TestRBNode<std::string>));
      new (node.ptr_) TestRBNode<std::string>(key, 0);
      hipc::ShmPtrBase<TestRBNode<std::string>> shm(node.shm_.alloc_id_, node.shm_.off_.load());
      FullPtr<TestRBNode<std::string>> ptr(alloc, shm);
      ptr.ptr_ = node.ptr_;
      tree.emplace(alloc, ptr);
    }

    REQUIRE(tree.size() == keys.size());

    // Find operations
    REQUIRE_FALSE(tree.find(alloc, std::string("cherry")).IsNull());
    REQUIRE(tree.find(alloc, std::string("fig")).IsNull());

    // Delete
    auto popped = tree.pop(alloc, std::string("banana"));
    REQUIRE_FALSE(popped.IsNull());
    REQUIRE(tree.size() == keys.size() - 1);
  }
}
