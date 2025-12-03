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
#include "hermes_shm/data_structures/ipc/slist_pre.h"
#include "hermes_shm/memory/backend/malloc_backend.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"

using namespace hshm::ipc;

/**
 * Test node structure that embeds slist_node
 */
struct TestNode {
  pre::slist_node link_;  // List linkage
  int value_;             // Test data

  TestNode() : value_(0) {}
  explicit TestNode(int val) : value_(val) {}
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

TEST_CASE("slist_pre - Basic Operations", "[slist_pre]") {
  MallocBackend backend;
  size_t arena_size = 1024 * 1024;  // 1 MB
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);
  

  SECTION("Initialization") {
    pre::slist<false> list;
    list.Init();

    REQUIRE(list.size() == 0);
    REQUIRE(list.empty());
    REQUIRE(list.GetHead().IsNull());
  }

  SECTION("Single element emplace and pop") {
    pre::slist<false> list;
    list.Init();

    // Allocate a test node
    auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
    node_ptr.ptr_->value_ = 42;

    // Emplace the node (cast to slist_node*)
    auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
    FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
    list.emplace(alloc, link_ptr);

    REQUIRE(list.size() == 1);
    REQUIRE_FALSE(list.empty());
    REQUIRE_FALSE(list.GetHead().IsNull());

    // Pop the node
    auto popped = list.pop(alloc);
    REQUIRE_FALSE(popped.IsNull());
    REQUIRE(list.size() == 0);
    REQUIRE(list.empty());

    // Verify the data
    auto popped_node = reinterpret_cast<TestNode*>(popped.ptr_);
    REQUIRE(popped_node->value_ == 42);
  }

  SECTION("Multiple elements - LIFO order") {
    pre::slist<false> list;
    list.Init();

    // Allocate and emplace 5 nodes
    const int NUM_NODES = 5;
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
      node_ptr.ptr_->value_ = i;

      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    REQUIRE(list.size() == NUM_NODES);

    // Pop all nodes - should come out in reverse order (LIFO)
    for (int i = NUM_NODES - 1; i >= 0; --i) {
      auto popped = list.pop(alloc);
      REQUIRE_FALSE(popped.IsNull());

      auto popped_node = reinterpret_cast<TestNode*>(popped.ptr_);
      REQUIRE(popped_node->value_ == i);
    }

    REQUIRE(list.size() == 0);
    REQUIRE(list.empty());
  }

  SECTION("Pop from empty list") {
    pre::slist<false> list;
    list.Init();

    auto popped = list.pop(alloc);
    REQUIRE(popped.IsNull());
    REQUIRE(list.size() == 0);
  }

  SECTION("Peek operation") {
    pre::slist<false> list;
    list.Init();

    // Peek empty list
    auto peeked = list.peek(alloc);
    REQUIRE(peeked.IsNull());

    // Add a node
    auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
    node_ptr.ptr_->value_ = 100;
    auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
    list.emplace(alloc, link_ptr);

    // Peek should return the head without removing it
    peeked = list.peek(alloc);
    REQUIRE_FALSE(peeked.IsNull());
    REQUIRE(list.size() == 1);  // Size unchanged

    auto peeked_node = reinterpret_cast<TestNode*>(peeked.ptr_);
    REQUIRE(peeked_node->value_ == 100);
  }

  SECTION("Interleaved emplace and pop") {
    pre::slist<false> list;
    list.Init();

    // Emplace 3 nodes
    for (int i = 0; i < 3; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }
    REQUIRE(list.size() == 3);

    // Pop 2 nodes
    list.pop(alloc);
    list.pop(alloc);
    REQUIRE(list.size() == 1);

    // Emplace 2 more nodes
    for (int i = 10; i < 12; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }
    REQUIRE(list.size() == 3);

    // Pop all and verify order
    auto n1 = list.pop(alloc);
    auto n2 = list.pop(alloc);
    auto n3 = list.pop(alloc);

    REQUIRE(reinterpret_cast<TestNode*>(n1.ptr_)->value_ == 11);
    REQUIRE(reinterpret_cast<TestNode*>(n2.ptr_)->value_ == 10);
    REQUIRE(reinterpret_cast<TestNode*>(n3.ptr_)->value_ == 0);
  }

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("slist_pre - Atomic Version", "[slist_pre][atomic]") {
  MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  auto *alloc = CreateTestAllocator<true>(backend, arena_size);
  

  SECTION("Basic atomic operations") {
    pre::slist<true> list;
    list.Init();

    // Allocate and emplace nodes
    const int NUM_NODES = 10;
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    REQUIRE(list.size() == NUM_NODES);

    // Pop all nodes
    for (int i = NUM_NODES - 1; i >= 0; --i) {
      auto popped = list.pop(alloc);
      REQUIRE_FALSE(popped.IsNull());
      REQUIRE(reinterpret_cast<TestNode*>(popped.ptr_)->value_ == i);
    }

    REQUIRE(list.size() == 0);
    REQUIRE(list.empty());
  }

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("slist_pre - Node Reuse", "[slist_pre]") {
  MallocBackend backend;
  size_t arena_size = 1024 * 1024;
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);
  

  SECTION("Reuse popped nodes") {
    pre::slist<false> list;
    list.Init();

    // Allocate a node
    auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
    node_ptr.ptr_->value_ = 1;

    // Emplace, pop, modify, and re-emplace
    auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
    list.emplace(alloc, link_ptr);

    auto popped = list.pop(alloc);
    REQUIRE_FALSE(popped.IsNull());
    REQUIRE(list.size() == 0);

    // Modify the node
    auto reused_node = reinterpret_cast<TestNode*>(popped.ptr_);
    reused_node->value_ = 999;

    // Re-emplace the same node
    list.emplace(alloc, popped);
    REQUIRE(list.size() == 1);

    // Pop and verify
    auto final = list.pop(alloc);
    REQUIRE(reinterpret_cast<TestNode*>(final.ptr_)->value_ == 999);
  }

  delete alloc;
  backend.shm_destroy();
}

TEST_CASE("slist_pre - Large List", "[slist_pre]") {
  MallocBackend backend;
  size_t arena_size = 10 * 1024 * 1024;  // 10 MB for large test
  auto *alloc = CreateTestAllocator<false>(backend, arena_size);
  

  SECTION("1000 elements") {
    pre::slist<false> list;
    list.Init();

    const int NUM_NODES = 1000;

    // Emplace many nodes
    for (int i = 0; i < NUM_NODES; ++i) {
      auto node_ptr = alloc->Allocate<TestNode>( sizeof(TestNode));
      node_ptr.ptr_->value_ = i;
      auto *link_node_ptr = reinterpret_cast<pre::slist_node*>(node_ptr.ptr_);
      FullPtr<pre::slist_node> link_ptr(link_node_ptr, static_cast<ShmPtr<>>(node_ptr.shm_));
      list.emplace(alloc, link_ptr);
    }

    REQUIRE(list.size() == NUM_NODES);

    // Pop and verify all
    int count = 0;
    while (!list.empty()) {
      auto popped = list.pop(alloc);
      REQUIRE_FALSE(popped.IsNull());
      count++;
    }

    REQUIRE(count == NUM_NODES);
    REQUIRE(list.size() == 0);
  }

  delete alloc;
  backend.shm_destroy();
}
