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

  ArenaAllocator<ATOMIC> *alloc = new ArenaAllocator<ATOMIC>();
  alloc->shm_init(backend, 0);

  return alloc;
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
  std::function<int(OffsetPtr<>, KeyT*, KeyT*)> check_node =
      [&](OffsetPtr<> node_off, KeyT *min_key, KeyT *max_key) -> int {
    if (node_off.IsNull()) {
      return 1;  // Null nodes are black (property 3)
    }

    FullPtr<NodeT> node(alloc, OffsetPtr<NodeT>(node_off.load()));

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
    int left_black = check_node(node.ptr_->left_, min_key, &node.ptr_->key);
    int right_black = check_node(node.ptr_->right_, &node.ptr_->key, max_key);

    if (left_black == -1 || right_black == -1 || left_black != right_black) {
      return -1;
    }

    return left_black + (node.ptr_->color_ == pre::RBColor::BLACK ? 1 : 0);
  };

  return check_node(tree.GetRoot(), nullptr, nullptr) != -1;
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
    FullPtr<TestRBNode<int>> test_ptr(node_ptr.ptr_, static_cast<ShmPtr<>>(node_ptr.shm_));
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

      FullPtr<TestRBNode<int>> test_ptr(node_ptr.ptr_, static_cast<ShmPtr<>>(node_ptr.shm_));
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

      FullPtr<TestRBNode<int>> test_ptr(node_ptr.ptr_, static_cast<ShmPtr<>>(node_ptr.shm_));
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

      FullPtr<TestRBNode<int>> test_ptr(node_ptr.ptr_, static_cast<ShmPtr<>>(node_ptr.shm_));
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

    FullPtr<TestRBNode<int>> test_ptr1(node1_ptr.ptr_, static_cast<ShmPtr<>>(node1_ptr.shm_));
    tree.emplace(alloc, test_ptr1);

    REQUIRE(tree.size() == 1);

    // Try to insert another node with same key
    auto node2_ptr = alloc->Allocate<TestRBNode<int>>( sizeof(TestRBNode<int>));
    node2_ptr.ptr_->key = 42;
    node2_ptr.ptr_->value_ = 200;

    FullPtr<TestRBNode<int>> test_ptr2(node2_ptr.ptr_, static_cast<ShmPtr<>>(node2_ptr.shm_));
    tree.emplace(alloc, test_ptr2);

    // Size should remain 1 (duplicate not inserted)
    REQUIRE(tree.size() == 1);

    // Original value should be preserved
    auto found = tree.find(alloc, 42);
    REQUIRE(found.ptr_->value_ == 100);
  }

  delete alloc;
  backend.shm_destroy();
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

    FullPtr<TestRBNode<int>> test_ptr(node_ptr.ptr_, static_cast<ShmPtr<>>(node_ptr.shm_));
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

      FullPtr<TestRBNode<int>> test_ptr(node_ptr.ptr_, static_cast<ShmPtr<>>(node_ptr.shm_));
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

      FullPtr<TestRBNode<int>> test_ptr(node_ptr.ptr_, static_cast<ShmPtr<>>(node_ptr.shm_));
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

      FullPtr<TestRBNode<int>> test_ptr(node_ptr.ptr_, static_cast<ShmPtr<>>(node_ptr.shm_));
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

      FullPtr<TestRBNode<int>> test_ptr(node_ptr.ptr_, static_cast<ShmPtr<>>(node_ptr.shm_));
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

  delete alloc;
  backend.shm_destroy();
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

      FullPtr<TestRBNode<int>> test_ptr(node_ptr.ptr_, static_cast<ShmPtr<>>(node_ptr.shm_));
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

  delete alloc;
  backend.shm_destroy();
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

      FullPtr<TestRBNode<int>> test_ptr(node_ptr.ptr_, static_cast<ShmPtr<>>(node_ptr.shm_));
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

  delete alloc;
  backend.shm_destroy();
}
