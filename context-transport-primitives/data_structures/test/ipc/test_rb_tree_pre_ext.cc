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

/**
 * Extended randomized rb_tree tests targeting the deep rebalancing paths:
 * FixInsert case 2/3 rotations (both sides), FixDelete sibling recoloring and
 * rotation cases (both sides), FixDeleteFromParent black-leaf deletion fixes,
 * and Transplant with deep successors.
 */

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <functional>
#include <random>
#include <set>
#include <vector>
#include "clio_ctp/data_structures/ipc/rb_tree_pre.h"
#include "clio_ctp/memory/backend/malloc_backend.h"
#include "clio_ctp/memory/allocator/arena_allocator.h"

using namespace ctp::ipc;

/** Test node structure that inherits from rb_node */
struct ExtRBNode : public pre::rb_node {
  int key;
  int value_;

  ExtRBNode() : pre::rb_node(), key(0), value_(0) {}

  bool operator<(const ExtRBNode &other) const { return key < other.key; }
  bool operator>(const ExtRBNode &other) const { return key > other.key; }
  bool operator==(const ExtRBNode &other) const { return key == other.key; }
};

/** Helper to create an ArenaAllocator for testing */
static ArenaAllocator<false>* CreateExtTestAllocator(MallocBackend &backend,
                                                     size_t arena_size) {
  backend.shm_init(MemoryBackendId(0, 0), arena_size);
  return backend.MakeAlloc<ArenaAllocator<false>>();
}

/** Insert a key into the tree */
static void TreeInsert(ArenaAllocator<false> *alloc,
                       pre::rb_tree<ExtRBNode, false> &tree, int key) {
  auto node_ptr = alloc->Allocate<ExtRBNode>(sizeof(ExtRBNode));
  node_ptr.ptr_->key = key;
  node_ptr.ptr_->value_ = key * 2;
  ctp::ipc::ShmPtrBase<ExtRBNode> shm(node_ptr.shm_.alloc_id_,
                                      node_ptr.shm_.off_.load());
  FullPtr<ExtRBNode> ptr(alloc, shm);
  ptr.ptr_ = node_ptr.ptr_;
  tree.emplace(alloc, ptr);
}

/** Verify RB tree invariants (BST order, red-red, black height) */
static bool VerifyExtRBProperties(ArenaAllocator<false> *alloc,
                                  pre::rb_tree<ExtRBNode, false> &tree) {
  if (tree.empty()) {
    return true;
  }

  FullPtr<ExtRBNode> root(alloc, OffsetPtr<ExtRBNode>(tree.GetRoot().load()));
  if (root.ptr_->color_ != pre::RBColor::BLACK) {
    return false;
  }

  std::function<int(OffsetPtr<ExtRBNode>, const int*, const int*)> check_node =
      [&](OffsetPtr<ExtRBNode> node_off, const int *min_key,
          const int *max_key) -> int {
    if (node_off.IsNull()) {
      return 1;
    }

    FullPtr<ExtRBNode> node(alloc, node_off);
    if (min_key && node.ptr_->key <= *min_key) return -1;
    if (max_key && node.ptr_->key >= *max_key) return -1;

    if (node.ptr_->color_ == pre::RBColor::RED) {
      if (!node.ptr_->left_.IsNull()) {
        FullPtr<ExtRBNode> left(alloc,
                                OffsetPtr<ExtRBNode>(node.ptr_->left_.load()));
        if (left.ptr_->color_ == pre::RBColor::RED) return -1;
      }
      if (!node.ptr_->right_.IsNull()) {
        FullPtr<ExtRBNode> right(
            alloc, OffsetPtr<ExtRBNode>(node.ptr_->right_.load()));
        if (right.ptr_->color_ == pre::RBColor::RED) return -1;
      }
    }

    int left_black = check_node(OffsetPtr<ExtRBNode>(node.ptr_->left_),
                                min_key, &node.ptr_->key);
    int right_black = check_node(OffsetPtr<ExtRBNode>(node.ptr_->right_),
                                 &node.ptr_->key, max_key);
    if (left_black == -1 || right_black == -1 || left_black != right_black) {
      return -1;
    }
    return left_black + (node.ptr_->color_ == pre::RBColor::BLACK ? 1 : 0);
  };

  return check_node(OffsetPtr<ExtRBNode>(tree.GetRoot().load()),
                    nullptr, nullptr) != -1;
}

TEST_CASE("rb_tree_pre_ext - Randomized insert/erase stress",
          "[rb_tree_pre_ext]") {
  MallocBackend backend;
  auto *alloc = CreateExtTestAllocator(backend, 64 * 1024 * 1024);

  const int kNumKeys = 256;
  for (unsigned seed = 1; seed <= 6; ++seed) {
    std::mt19937 rng(seed);
    pre::rb_tree<ExtRBNode, false> tree;
    tree.Init();

    std::vector<int> keys(kNumKeys);
    for (int i = 0; i < kNumKeys; ++i) keys[i] = i;
    std::shuffle(keys.begin(), keys.end(), rng);

    // Insert in random order, verifying invariants periodically
    for (size_t i = 0; i < keys.size(); ++i) {
      TreeInsert(alloc, tree, keys[i]);
      if (i % 17 == 0) {
        REQUIRE(VerifyExtRBProperties(alloc, tree));
      }
    }
    REQUIRE(tree.size() == static_cast<size_t>(kNumKeys));
    REQUIRE(VerifyExtRBProperties(alloc, tree));

    // Erase in a different random order
    std::shuffle(keys.begin(), keys.end(), rng);
    for (size_t i = 0; i < keys.size(); ++i) {
      auto popped = tree.pop(alloc, keys[i]);
      REQUIRE_FALSE(popped.IsNull());
      REQUIRE(popped.ptr_->key == keys[i]);
      REQUIRE(VerifyExtRBProperties(alloc, tree));
    }
    REQUIRE(tree.empty());
  }
}

TEST_CASE("rb_tree_pre_ext - Interleaved random operations vs std::set",
          "[rb_tree_pre_ext]") {
  MallocBackend backend;
  auto *alloc = CreateExtTestAllocator(backend, 64 * 1024 * 1024);

  for (unsigned seed = 10; seed <= 13; ++seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> key_dist(0, 127);
    std::uniform_int_distribution<int> op_dist(0, 99);

    pre::rb_tree<ExtRBNode, false> tree;
    tree.Init();
    std::set<int> reference;

    for (int step = 0; step < 1500; ++step) {
      int key = key_dist(rng);
      int op = op_dist(rng);
      if (op < 55) {
        // Insert (duplicates rejected by both containers)
        TreeInsert(alloc, tree, key);
        reference.insert(key);
      } else if (op < 90) {
        // Erase
        auto popped = tree.pop(alloc, key);
        bool ref_had = reference.erase(key) > 0;
        REQUIRE(popped.IsNull() == !ref_had);
      } else {
        // Find
        auto found = tree.find(alloc, key);
        bool ref_has = reference.count(key) > 0;
        REQUIRE(found.IsNull() == !ref_has);
      }
      REQUIRE(tree.size() == reference.size());
      if (step % 50 == 0) {
        REQUIRE(VerifyExtRBProperties(alloc, tree));
      }
    }

    // Validate full content at the end
    REQUIRE(VerifyExtRBProperties(alloc, tree));
    for (int key = 0; key <= 127; ++key) {
      bool ref_has = reference.count(key) > 0;
      REQUIRE(tree.find(alloc, key).IsNull() == !ref_has);
    }

    // Drain remaining keys (descending to vary deletion direction)
    for (auto it = reference.rbegin(); it != reference.rend(); ++it) {
      auto popped = tree.pop(alloc, *it);
      REQUIRE_FALSE(popped.IsNull());
    }
    REQUIRE(tree.empty());
  }
}

TEST_CASE("rb_tree_pre_ext - Deep successor transplant",
          "[rb_tree_pre_ext]") {
  MallocBackend backend;
  auto *alloc = CreateExtTestAllocator(backend, 16 * 1024 * 1024);

  // Build trees of varying shapes and erase internal nodes whose successor
  // is deep in the right subtree (not a direct child), with and without a
  // right child on the successor.
  for (int shape = 0; shape < 4; ++shape) {
    pre::rb_tree<ExtRBNode, false> tree;
    tree.Init();

    // Dense balanced range so internal nodes have deep successors
    for (int key = 0; key < 64; ++key) {
      TreeInsert(alloc, tree, key * 2);  // even keys
    }
    REQUIRE(VerifyExtRBProperties(alloc, tree));

    // Optionally add odd keys near the erased nodes so successors have
    // right children
    if (shape % 2 == 1) {
      for (int key = 1; key < 128; key += 8) {
        TreeInsert(alloc, tree, key);
      }
    }
    REQUIRE(VerifyExtRBProperties(alloc, tree));

    // Erase every internal node from a varying starting point
    for (int key = shape * 2; key < 128; key += 6) {
      auto popped = tree.pop(alloc, key);
      if (!popped.IsNull()) {
        REQUIRE(VerifyExtRBProperties(alloc, tree));
      }
    }
    // Erase the root repeatedly (always a two-child internal node when big)
    while (!tree.empty()) {
      FullPtr<ExtRBNode> root(alloc,
                              OffsetPtr<ExtRBNode>(tree.GetRoot().load()));
      int root_key = root.ptr_->key;
      auto popped = tree.pop(alloc, root_key);
      REQUIRE_FALSE(popped.IsNull());
      REQUIRE(VerifyExtRBProperties(alloc, tree));
    }
  }
}
