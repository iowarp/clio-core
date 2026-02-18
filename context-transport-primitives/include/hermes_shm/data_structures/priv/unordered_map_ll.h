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

#ifndef HSHM_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LL_H_
#define HSHM_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LL_H_

#include <vector>
#include <list>
#include <functional>
#include <utility>
#include <algorithm>

namespace hshm::priv {

/**
 * Unordered map implementation using vector of lists
 *
 * This map partitions the hash space into multiple buckets (vector elements).
 * Each bucket contains a list of key-value pairs.
 * External locking is required for thread safety.
 *
 * Template parameters:
 * @tparam Key Key type
 * @tparam T Mapped value type
 * @tparam Hash Hash function type (defaults to std::hash<Key>)
 * @tparam KeyEqual Key equality comparison (defaults to std::equal_to<Key>)
 */
template<typename Key, typename T, typename Hash = std::hash<Key>,
         typename KeyEqual = std::equal_to<Key>>
class unordered_map_ll {
public:
  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<const Key, T>;
  using size_type = std::size_t;
  using hasher = Hash;
  using key_equal = KeyEqual;

private:
  // Each bucket contains a list of key-value pairs
  struct Bucket {
    std::list<value_type> entries;

    Bucket() = default;
    Bucket(const Bucket&) = default;
    Bucket& operator=(const Bucket&) = default;
    Bucket(Bucket&&) noexcept = default;
    Bucket& operator=(Bucket&&) noexcept = default;
  };

  std::vector<Bucket> buckets_;
  Hash hash_fn_;
  KeyEqual key_eq_;

  /**
   * Get bucket index for a given key
   * @param key Key to hash
   * @return Bucket index
   */
  size_type get_bucket_index(const Key& key) const {
    return hash_fn_(key) % buckets_.size();
  }

  /**
   * Find entry in a bucket
   * @param bucket Bucket to search
   * @param key Key to find
   * @return Iterator to the entry, or bucket.entries.end() if not found
   */
  typename std::list<value_type>::iterator
  find_entry_in_bucket(Bucket& bucket, const Key& key) {
    return std::find_if(bucket.entries.begin(), bucket.entries.end(),
                        [this, &key](const value_type& item) {
                          return key_eq_(item.first, key);
                        });
  }

  /**
   * Find entry in a bucket (const version)
   * @param bucket Bucket to search
   * @param key Key to find
   * @return Const iterator to the entry, or bucket.entries.end() if not found
   */
  typename std::list<value_type>::const_iterator
  find_entry_in_bucket(const Bucket& bucket, const Key& key) const {
    return std::find_if(bucket.entries.begin(), bucket.entries.end(),
                        [this, &key](const value_type& item) {
                          return key_eq_(item.first, key);
                        });
  }

public:
  /**
   * Constructor
   * @param max_concurrency Number of buckets (determines maximum useful concurrency)
   * @param hash Hash function
   * @param equal Key equality function
   */
  explicit unordered_map_ll(size_type max_concurrency = 16,
                           const Hash& hash = Hash(),
                           const KeyEqual& equal = KeyEqual())
      : buckets_(max_concurrency), hash_fn_(hash), key_eq_(equal) {}

  /**
   * Destructor
   */
  ~unordered_map_ll() = default;

  // Copyable and movable
  unordered_map_ll(const unordered_map_ll&) = default;
  unordered_map_ll& operator=(const unordered_map_ll&) = default;
  unordered_map_ll(unordered_map_ll&&) noexcept = default;
  unordered_map_ll& operator=(unordered_map_ll&&) noexcept = default;

  /**
   * Insert or update a key-value pair
   * @param key Key to insert/update
   * @param value Value to insert/update
   * @return Pair of iterator to the element and bool indicating if insertion occurred
   */
  std::pair<bool, T*> insert_or_assign(const Key& key, const T& value) {
    size_type bucket_idx = get_bucket_index(key);
    Bucket& bucket = buckets_[bucket_idx];

    auto it = find_entry_in_bucket(bucket, key);
    if (it != bucket.entries.end()) {
      // Key exists, update value
      it->second = value;
      return {false, &it->second};
    } else {
      // Key doesn't exist, insert new entry
      bucket.entries.emplace_back(key, value);
      return {true, &bucket.entries.back().second};
    }
  }

  /**
   * Insert a key-value pair (only if key doesn't exist)
   * @param key Key to insert
   * @param value Value to insert
   * @return Pair of bool indicating success and pointer to value (or nullptr)
   */
  std::pair<bool, T*> insert(const Key& key, const T& value) {
    size_type bucket_idx = get_bucket_index(key);
    Bucket& bucket = buckets_[bucket_idx];

    auto it = find_entry_in_bucket(bucket, key);
    if (it != bucket.entries.end()) {
      // Key exists, insertion failed
      return {false, &it->second};
    } else {
      // Key doesn't exist, insert new entry
      bucket.entries.emplace_back(key, value);
      return {true, &bucket.entries.back().second};
    }
  }

  /**
   * Emplace a key-value pair
   * @param args Arguments to construct the key-value pair
   * @return Pair of bool indicating success and pointer to value (or nullptr)
   */
  template<typename... Args>
  std::pair<bool, T*> emplace(Args&&... args) {
    value_type temp(std::forward<Args>(args)...);
    return insert(temp.first, temp.second);
  }

  /**
   * Access element with given key (creates if doesn't exist)
   * @param key Key to access
   * @return Reference to the mapped value
   */
  T& operator[](const Key& key) {
    size_type bucket_idx = get_bucket_index(key);
    Bucket& bucket = buckets_[bucket_idx];

    auto it = find_entry_in_bucket(bucket, key);
    if (it != bucket.entries.end()) {
      return it->second;
    } else {
      // Insert default-constructed value
      bucket.entries.emplace_back(key, T());
      return bucket.entries.back().second;
    }
  }

  /**
   * Access element with given key (const version, throws if not found)
   * @param key Key to access
   * @return Const reference to the mapped value
   * @throws std::out_of_range if key not found
   */
  const T& at(const Key& key) const {
    size_type bucket_idx = get_bucket_index(key);
    const Bucket& bucket = buckets_[bucket_idx];

    auto it = find_entry_in_bucket(bucket, key);
    if (it != bucket.entries.end()) {
      return it->second;
    } else {
      throw std::out_of_range("unordered_map_ll::at: key not found");
    }
  }

  /**
   * Access element with given key (non-const version, throws if not found)
   * @param key Key to access
   * @return Reference to the mapped value
   * @throws std::out_of_range if key not found
   */
  T& at(const Key& key) {
    size_type bucket_idx = get_bucket_index(key);
    Bucket& bucket = buckets_[bucket_idx];

    auto it = find_entry_in_bucket(bucket, key);
    if (it != bucket.entries.end()) {
      return it->second;
    } else {
      throw std::out_of_range("unordered_map_ll::at: key not found");
    }
  }

  /**
   * Find an element
   * @param key Key to find
   * @return Pointer to the value if found, nullptr otherwise
   */
  T* find(const Key& key) {
    size_type bucket_idx = get_bucket_index(key);
    Bucket& bucket = buckets_[bucket_idx];

    auto it = find_entry_in_bucket(bucket, key);
    if (it != bucket.entries.end()) {
      return &it->second;
    }
    return nullptr;
  }

  /**
   * Find an element (const version)
   * @param key Key to find
   * @return Const pointer to the value if found, nullptr otherwise
   */
  const T* find(const Key& key) const {
    size_type bucket_idx = get_bucket_index(key);
    const Bucket& bucket = buckets_[bucket_idx];

    auto it = find_entry_in_bucket(bucket, key);
    if (it != bucket.entries.end()) {
      return &it->second;
    }
    return nullptr;
  }

  /**
   * Check if key exists
   * @param key Key to check
   * @return true if key exists, false otherwise
   */
  bool contains(const Key& key) const {
    return find(key) != nullptr;
  }

  /**
   * Count occurrences of key (always 0 or 1)
   * @param key Key to count
   * @return 1 if key exists, 0 otherwise
   */
  size_type count(const Key& key) const {
    return contains(key) ? 1 : 0;
  }

  /**
   * Erase element with given key
   * @param key Key to erase
   * @return Number of elements erased (0 or 1)
   */
  size_type erase(const Key& key) {
    size_type bucket_idx = get_bucket_index(key);
    Bucket& bucket = buckets_[bucket_idx];

    auto it = find_entry_in_bucket(bucket, key);
    if (it != bucket.entries.end()) {
      bucket.entries.erase(it);
      return 1;
    }
    return 0;
  }

  /**
   * Clear all elements
   */
  void clear() {
    for (auto& bucket : buckets_) {
      bucket.entries.clear();
    }
  }

  /**
   * Get total number of elements
   * @return Total number of elements
   */
  size_type size() const {
    size_type total = 0;
    for (const auto& bucket : buckets_) {
      total += bucket.entries.size();
    }
    return total;
  }

  /**
   * Check if map is empty
   * @return true if empty, false otherwise
   */
  bool empty() const {
    return size() == 0;
  }

  /**
   * Get number of buckets
   * @return Number of buckets
   */
  size_type bucket_count() const {
    return buckets_.size();
  }

  /**
   * Apply a function to each element
   * @param fn Function to apply (takes const Key&, T&)
   */
  template<typename Func>
  void for_each(Func fn) {
    for (auto& bucket : buckets_) {
      for (auto& entry : bucket.entries) {
        fn(entry.first, entry.second);
      }
    }
  }

  /**
   * Apply a function to each element (const version)
   * @param fn Function to apply (takes const Key&, const T&)
   */
  template<typename Func>
  void for_each(Func fn) const {
    for (const auto& bucket : buckets_) {
      for (const auto& entry : bucket.entries) {
        fn(entry.first, entry.second);
      }
    }
  }
};

}  // namespace hshm::priv

#endif  // HSHM_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LL_H_
