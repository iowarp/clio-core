/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Stress tests for ctp::priv::unordered_map_ll, focused on the property that
 * matters most: automatic growth (rehash) is SAFE to interleave with concurrent
 * single-key operations. The map starts deliberately tiny so a multi-threaded
 * insert storm triggers many rehashes WHILE other threads are inserting,
 * finding, and erasing — the exact race the map-wide RwLock is there to make
 * safe. With the pre-RwLock in-place rehash these tests crash / corrupt.
 */
#include "../../../context-runtime/test/simple_test.h"
#include "clio_ctp/data_structures/priv/unordered_map_ll.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using ctp::priv::unordered_map_ll;

// ---------------------------------------------------------------------------
// Single-threaded correctness across growth
// ---------------------------------------------------------------------------

TEST_CASE("umap_ll: single-thread grows and keeps all entries", "[priv_umap_ll]") {
  // Tiny initial bucket count + 0.6 growth => dozens of rehashes.
  unordered_map_ll<uint64_t, uint64_t> map(4, /*ext_percent=*/0.6,
                                           /*ext_mult=*/2);
  constexpr uint64_t kN = 20000;
  for (uint64_t i = 0; i < kN; ++i) {
    auto r = map.insert(i, i * 3 + 7);
    REQUIRE(r.inserted);
  }
  REQUIRE(map.size() == kN);
  REQUIRE(map.bucket_count() > 4);  // it actually grew
  for (uint64_t i = 0; i < kN; ++i) {
    auto *v = map.find(i);
    REQUIRE(v != nullptr);
    REQUIRE(*v == i * 3 + 7);
  }
  // Erase half, the rest must remain correct.
  for (uint64_t i = 0; i < kN; i += 2) REQUIRE(map.erase(i) == 1);
  REQUIRE(map.size() == kN / 2);
  for (uint64_t i = 1; i < kN; i += 2) {
    auto *v = map.find(i);
    REQUIRE(v != nullptr);
    REQUIRE(*v == i * 3 + 7);
  }
}

// ---------------------------------------------------------------------------
// Concurrent placement with growth: many threads insert disjoint key ranges
// into a tiny map; every key must survive the storm of concurrent rehashes.
// ---------------------------------------------------------------------------

TEST_CASE("umap_ll: concurrent inserts survive concurrent growth",
          "[priv_umap_ll]") {
  unordered_map_ll<uint64_t, uint64_t> map(4, 0.6, 2);
  constexpr int kThreads = 8;
  constexpr uint64_t kPer = 6000;  // 48k keys into a 4-bucket map

  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&map, t]() {
      for (uint64_t i = 0; i < kPer; ++i) {
        uint64_t k = static_cast<uint64_t>(t) * kPer + i;
        map.insert(k, k ^ 0xA5A5A5A5ULL);  // value derived from key
      }
    });
  }
  for (auto &th : ts) th.join();

  REQUIRE(map.size() == static_cast<size_t>(kThreads) * kPer);
  for (int t = 0; t < kThreads; ++t) {
    for (uint64_t i = 0; i < kPer; ++i) {
      uint64_t k = static_cast<uint64_t>(t) * kPer + i;
      auto *v = map.find(k);
      REQUIRE(v != nullptr);
      REQUIRE(*v == (k ^ 0xA5A5A5A5ULL));
    }
  }
}

// ---------------------------------------------------------------------------
// Readers running DURING expansion: writers grow the table while readers
// hammer find(). Any value a reader observes must be the correct one (no torn
// reads / use-after-free from a concurrent rehash), and the final contents
// must be exactly correct.
// ---------------------------------------------------------------------------

TEST_CASE("umap_ll: concurrent find during expansion is consistent",
          "[priv_umap_ll]") {
  unordered_map_ll<uint64_t, uint64_t> map(2, 0.6, 2);
  constexpr int kWriters = 6;
  constexpr int kReaders = 4;
  constexpr uint64_t kPer = 5000;
  const uint64_t kKeys = static_cast<uint64_t>(kWriters) * kPer;

  std::atomic<bool> done{false};
  std::atomic<uint64_t> bad{0};   // wrong value observed
  std::atomic<uint64_t> reads{0};

  auto value_of = [](uint64_t k) { return k * 1315423911ULL + 1; };

  std::vector<std::thread> writers;
  for (int t = 0; t < kWriters; ++t) {
    writers.emplace_back([&map, &value_of, t]() {
      for (uint64_t i = 0; i < kPer; ++i) {
        uint64_t k = static_cast<uint64_t>(t) * kPer + i;
        map.insert_or_assign(k, value_of(k));
      }
    });
  }
  std::vector<std::thread> readers;
  for (int r = 0; r < kReaders; ++r) {
    readers.emplace_back([&]() {
      while (!done.load(std::memory_order_relaxed)) {
        for (uint64_t k = 0; k < kKeys; ++k) {
          auto *v = map.find(k);  // may or may not be present yet
          if (v != nullptr) {
            if (*v != value_of(k)) bad.fetch_add(1);
            reads.fetch_add(1);
          }
        }
      }
    });
  }

  for (auto &w : writers) w.join();
  done.store(true, std::memory_order_relaxed);
  for (auto &r : readers) r.join();

  REQUIRE(bad.load() == 0);            // never observed a corrupt value
  REQUIRE(reads.load() > 0);           // readers actually found live entries
  REQUIRE(map.size() == kKeys);
  for (uint64_t k = 0; k < kKeys; ++k) {
    auto *v = map.find(k);
    REQUIRE(v != nullptr);
    REQUIRE(*v == value_of(k));
  }
}

// ---------------------------------------------------------------------------
// Mixed churn: insert + erase + find all racing growth on disjoint ranges.
// Each thread owns its key range so the final state is deterministic.
// ---------------------------------------------------------------------------

TEST_CASE("umap_ll: concurrent insert/erase/find churn with growth",
          "[priv_umap_ll]") {
  unordered_map_ll<uint64_t, uint64_t> map(4, 0.6, 2);
  constexpr int kThreads = 8;
  constexpr uint64_t kPer = 4000;

  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&map, t]() {
      const uint64_t base = static_cast<uint64_t>(t) * kPer;
      // Insert all, erase the even ones, re-find the odd ones.
      for (uint64_t i = 0; i < kPer; ++i) map.insert(base + i, base + i);
      for (uint64_t i = 0; i < kPer; i += 2) map.erase(base + i);
      for (uint64_t i = 1; i < kPer; i += 2) {
        auto *v = map.find(base + i);
        // Our own odd keys are never erased by anyone => must be present.
        if (v == nullptr || *v != base + i) {
          // record via a known-bad write so the assert below trips
          map.insert_or_assign(~0ULL, 1);
        }
      }
    });
  }
  for (auto &th : ts) th.join();

  REQUIRE(map.find(~0ULL) == nullptr);  // no thread hit a missing/odd key
  REQUIRE(map.size() == static_cast<size_t>(kThreads) * (kPer / 2));
  for (int t = 0; t < kThreads; ++t) {
    const uint64_t base = static_cast<uint64_t>(t) * kPer;
    for (uint64_t i = 1; i < kPer; i += 2) {
      auto *v = map.find(base + i);
      REQUIRE(v != nullptr);
      REQUIRE(*v == base + i);
    }
    for (uint64_t i = 0; i < kPer; i += 2) {
      REQUIRE(map.find(base + i) == nullptr);
    }
  }
}

// ---------------------------------------------------------------------------
// for_each(WRITE lock) concurrent with insert/erase/find (read locks) + growth.
//
// This mirrors the CTE-core workload that wedges under generic/089: DelTag
// scans the whole blob map with for_each() -- which takes the map-wide RwLock
// in WRITE (exclusive) mode -- while other coroutines (PutBlob/DelBlob/GetBlob)
// insert/erase/find blobs, i.e. take the RwLock in READ mode and (on growth)
// WRITE mode via rehash. Keys are compound "major.minor.blobname" strings like
// the real blob keys, and the for_each body does the same read-only
// prefix-collect DelTag does. If for_each's write lock deadlocks or is starved
// by the sustained reader stream, the scanner threads stop making progress; a
// watchdog detects "no progress" and fails (set CTP_UMAP_HANG_GDB=1 to instead
// pause so a debugger can attach to the wedged for_each).
// ---------------------------------------------------------------------------
TEST_CASE("umap_ll: for_each(write) vs concurrent insert/erase/find (089 repro)",
          "[priv_umap_ll]") {
  // Heap + leaked: if we bail out on a detected deadlock we detach the still
  // wedged worker threads, so the map must outlive them.
  auto *map = new unordered_map_ll<std::string, uint64_t>(4, 0.6, 2);

  constexpr uint64_t kSpace = 3000;  // distinct blob keys touched
  auto key = [](uint64_t i) {
    // "major.minor.blobname", like tag_blob_name_to_info_ keys.
    return std::string("1.0.blob") + std::to_string(i % kSpace);
  };
  const std::string kPrefix = "1.0.";
  for (uint64_t i = 0; i < kSpace; i += 2) map->insert(key(i), i);

  std::atomic<bool> stop{false};
  std::atomic<uint64_t> scans{0}, writer_ops{0}, finder_ops{0};

  // Writers: insert/erase (read lock + bucket write lock + rehash write lock).
  auto writer = [&](uint64_t seed) {
    uint64_t x = seed;
    while (!stop.load(std::memory_order_relaxed)) {
      x = x * 6364136223846793005ULL + 1442695040888963407ULL;
      uint64_t i = x % kSpace;
      if (x & 0x10) {
        map->insert(key(i), i);
      } else {
        map->erase(key(i));
      }
      writer_ops.fetch_add(1, std::memory_order_relaxed);
    }
  };
  // Scanners: for_each == the DelTag full-map scan (GLOBAL WRITE lock).
  auto scanner = [&]() {
    while (!stop.load(std::memory_order_relaxed)) {
      std::vector<std::string> matched;  // exactly what DelTag collects
      map->for_each([&](const std::string &k, const uint64_t &) {
        if (k.compare(0, kPrefix.size(), kPrefix) == 0) matched.push_back(k);
      });
      scans.fetch_add(1, std::memory_order_relaxed);
    }
  };
  // Finders: find (read lock).
  auto finder = [&](uint64_t seed) {
    uint64_t x = seed;
    while (!stop.load(std::memory_order_relaxed)) {
      x = x * 2862933555777941757ULL + 3037000493ULL;
      map->find(key(x % kSpace));
      finder_ops.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> ts;
  for (int i = 0; i < 4; ++i) ts.emplace_back(writer, 1000 + i);
  for (int i = 0; i < 3; ++i) ts.emplace_back(scanner);
  for (int i = 0; i < 2; ++i) ts.emplace_back(finder, 7000 + i);

  // Watchdog: a deadlock shows up as "combined progress counter stops moving".
  uint64_t last = 0, frozen_secs = 0;
  bool deadlock = false;
  for (int sec = 0; sec < 30 && !deadlock; ++sec) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    uint64_t cur = scans.load() + writer_ops.load() + finder_ops.load();
    if (cur == last) {
      if (++frozen_secs >= 5) deadlock = true;
    } else {
      frozen_secs = 0;
      last = cur;
    }
  }

  if (deadlock) {
    // scans stalled but writer/finder maybe too (writer_priority defers new
    // readers behind the waiting for_each writer). Report the split.
    if (std::getenv("CTP_UMAP_HANG_GDB") != nullptr) {
      // Pause (threads still wedged) so a debugger can dump the stuck for_each.
      std::this_thread::sleep_for(std::chrono::seconds(120));
    }
    for (auto &th : ts) th.detach();
    FAIL(std::string("for_each(write) DEADLOCK/STARVATION: no progress for 5s"
                     " -- scans=") + std::to_string(scans.load()) +
         " writer_ops=" + std::to_string(writer_ops.load()) +
         " finder_ops=" + std::to_string(finder_ops.load()));
  }

  stop.store(true, std::memory_order_relaxed);
  for (auto &th : ts) th.join();
  REQUIRE(scans.load() > 0);
  REQUIRE(writer_ops.load() > 0);
  REQUIRE(finder_ops.load() > 0);
  delete map;
}

SIMPLE_TEST_MAIN()
