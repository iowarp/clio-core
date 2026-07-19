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

#include <algorithm>
#include <atomic>
#include <catch2/catch_all.hpp>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "clio_ctp/search/regex_search_engine.h"

using ctp::search::RegexSearchEngine;

namespace {

// Sorted matching keys via the engine, for easy comparison.
template <typename V>
std::vector<std::string> SearchKeys(const RegexSearchEngine<V> &eng,
                                    const std::string &pat) {
  std::vector<std::string> out;
  for (const auto &kv : eng.Search(pat)) {
    out.push_back(kv.first);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Ground-truth: brute-force std::regex_match over a key list.
std::vector<std::string> BruteForce(const std::vector<std::string> &keys,
                                    const std::string &pat) {
  std::regex re(pat);
  std::vector<std::string> out;
  for (const auto &k : keys) {
    if (std::regex_match(k, re)) {
      out.push_back(k);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

TEST_CASE("RegexSearch: insert/find/contains/size", "[regex_search]") {
  RegexSearchEngine<int> eng;
  REQUIRE(eng.Empty());
  REQUIRE(eng.Insert("/a/b/c", 1));
  REQUIRE(eng.Insert("/a/b/d", 2));
  REQUIRE_FALSE(eng.Insert("/a/b/c", 99));  // overwrite -> not new
  REQUIRE(eng.Size() == 2);
  REQUIRE(eng.Contains("/a/b/c"));
  REQUIRE(eng.Find("/a/b/c") != nullptr);
  REQUIRE(*eng.Find("/a/b/c") == 99);  // value was overwritten
  REQUIRE(eng.Find("/nope") == nullptr);
}

TEST_CASE("RegexSearch: exact-path match", "[regex_search]") {
  RegexSearchEngine<int> eng;
  eng.Insert("/home/user/file.txt", 1);
  eng.Insert("/home/user/other.txt", 2);
  eng.Insert("/var/log/file.txt", 3);

  auto r = SearchKeys(eng, "^/home/user/file\\.txt$");
  REQUIRE(r == std::vector<std::string>{"/home/user/file.txt"});
}

TEST_CASE("RegexSearch: direct-children pattern", "[regex_search]") {
  RegexSearchEngine<int> eng;
  eng.Insert("/dir/a", 1);
  eng.Insert("/dir/b", 2);
  eng.Insert("/dir/sub/c", 3);  // grandchild, must NOT match
  eng.Insert("/dir", 4);        // the dir itself, must NOT match
  eng.Insert("/other/x", 5);

  auto r = SearchKeys(eng, "^/dir/[^/]+$");
  REQUIRE(r == std::vector<std::string>{"/dir/a", "/dir/b"});
}

TEST_CASE("RegexSearch: substring via .* and unanchored-style", "[regex_search]") {
  RegexSearchEngine<int> eng;
  eng.Insert("alpha_foo_beta", 1);
  eng.Insert("xfooy", 2);
  eng.Insert("nomatch", 3);

  auto r = SearchKeys(eng, ".*foo.*");
  REQUIRE(r == std::vector<std::string>{"alpha_foo_beta", "xfooy"});
}

TEST_CASE("RegexSearch: delete removes from results and index", "[regex_search]") {
  RegexSearchEngine<int> eng;
  eng.Insert("/dir/a", 1);
  eng.Insert("/dir/b", 2);
  REQUIRE(eng.Delete("/dir/a"));
  REQUIRE_FALSE(eng.Delete("/dir/a"));  // already gone
  REQUIRE_FALSE(eng.Contains("/dir/a"));

  auto r = SearchKeys(eng, "^/dir/[^/]+$");
  REQUIRE(r == std::vector<std::string>{"/dir/b"});
  // The trigrams of the deleted key must no longer surface it.
  REQUIRE(SearchKeys(eng, ".*/a$").empty());
}

TEST_CASE("RegexSearch: rename preserves value and updates index",
          "[regex_search]") {
  RegexSearchEngine<std::string> eng;
  eng.Insert("/old/name", "payload");

  REQUIRE(eng.Rename("/old/name", "/new/place"));
  REQUIRE_FALSE(eng.Contains("/old/name"));
  REQUIRE(eng.Contains("/new/place"));
  REQUIRE(*eng.Find("/new/place") == "payload");

  // Old name no longer searchable; new name is.
  REQUIRE(SearchKeys(eng, "^/old/.*").empty());
  REQUIRE(SearchKeys(eng, "^/new/place$") ==
          std::vector<std::string>{"/new/place"});

  REQUIRE_FALSE(eng.Rename("/missing", "/whatever"));
  // Rename onto itself is a no-op success.
  REQUIRE(eng.Rename("/new/place", "/new/place"));
  REQUIRE(eng.Contains("/new/place"));
}

TEST_CASE("RegexSearch: short keys (< 3 chars) found via scan fallback",
          "[regex_search]") {
  RegexSearchEngine<int> eng;
  eng.Insert("a", 1);
  eng.Insert("ab", 2);
  eng.Insert("abc", 3);

  // "^ab$" has a literal run "ab" of length 2 -> no trigram -> scan fallback.
  REQUIRE(SearchKeys(eng, "^ab$") == std::vector<std::string>{"ab"});
  REQUIRE(SearchKeys(eng, "^a$") == std::vector<std::string>{"a"});
  REQUIRE(SearchKeys(eng, "^abc$") == std::vector<std::string>{"abc"});
}

TEST_CASE("RegexSearch: alternation/groups fall back to scan but stay correct",
          "[regex_search]") {
  RegexSearchEngine<int> eng;
  eng.Insert("cat", 1);
  eng.Insert("dog", 2);
  eng.Insert("bird", 3);

  REQUIRE(SearchKeys(eng, "^(cat|dog)$") ==
          std::vector<std::string>{"cat", "dog"});
  REQUIRE(SearchKeys(eng, "^(bird)$") == std::vector<std::string>{"bird"});
}

TEST_CASE("RegexSearch: match-all and no-match patterns", "[regex_search]") {
  RegexSearchEngine<int> eng;
  eng.Insert("x1", 1);
  eng.Insert("y2", 2);
  REQUIRE(SearchKeys(eng, ".*").size() == 2);
  REQUIRE(SearchKeys(eng, "^zzz$").empty());
  // A required trigram that indexes nothing -> empty fast path.
  REQUIRE(SearchKeys(eng, ".*qqq.*").empty());
}

TEST_CASE("RegexSearch: invalid pattern throws", "[regex_search]") {
  RegexSearchEngine<int> eng;
  eng.Insert("abc", 1);
  REQUIRE_THROWS_AS(eng.Search("([unclosed"), std::regex_error);
}

TEST_CASE("RegexSearch: iterator yields key and value", "[regex_search]") {
  RegexSearchEngine<int> eng;
  eng.Insert("/p/one", 11);
  eng.Insert("/p/two", 22);

  int sum = 0;
  int count = 0;
  for (const auto &kv : eng.Search("^/p/[^/]+$")) {
    REQUIRE(kv.first.rfind("/p/", 0) == 0);
    sum += kv.second;
    ++count;
  }
  REQUIRE(count == 2);
  REQUIRE(sum == 33);
}

// The core correctness guarantee: for a large, varied corpus and many patterns,
// the engine's result must EXACTLY equal brute-force regex over all keys. This
// is what proves the trigram prefilter never drops a real match.
TEST_CASE("RegexSearch: matches brute force across many patterns",
          "[regex_search]") {
  RegexSearchEngine<int> eng;
  std::vector<std::string> keys;

  std::mt19937 rng(1234567);  // fixed seed: deterministic
  const char *dirs[] = {"home", "var", "usr", "etc", "data", "tmp", "opt"};
  const char *leaves[] = {"file", "log", "config", "data", "cache",
                          "index", "blob", "meta", "foo", "bar"};
  const char *exts[] = {"txt", "log", "dat", "bin", "json", "csv"};
  std::uniform_int_distribution<int> depthD(1, 4);
  auto pick = [&](auto &arr, size_t len) { return arr[rng() % len]; };

  for (int i = 0; i < 400; ++i) {
    std::string path;
    int depth = depthD(rng);
    for (int d = 0; d < depth; ++d) {
      path += "/";
      path += pick(dirs, 7);
    }
    path += "/";
    path += pick(leaves, 10);
    path += "_";
    path += std::to_string(rng() % 50);
    path += ".";
    path += pick(exts, 6);
    if (eng.Insert(path, i)) {
      keys.push_back(path);
    }
  }

  std::vector<std::string> patterns = {
      ".*foo.*",
      ".*\\.log$",
      "^/home/.*",
      "^/var/log/.*",
      "^/[^/]+/[^/]+$",          // exactly two components
      ".*/file_[0-9]+\\..*",
      ".*config.*\\.json$",
      "^/(home|var)/.*",         // alternation -> scan fallback
      ".*data.*",
      ".*/bar_[0-9]+\\.csv$",
      "^/etc/.*/meta_[0-9]+\\.dat$",
      ".*",                       // everything
      ".*zzzzz.*",                // nothing
  };

  for (const auto &pat : patterns) {
    INFO("pattern: " << pat);
    REQUIRE(SearchKeys(eng, pat) == BruteForce(keys, pat));
  }
}

// Same correctness guarantee, but after a churn of deletes and renames, to
// confirm the inverted index stays consistent under mutation.
TEST_CASE("RegexSearch: consistent after delete/rename churn", "[regex_search]") {
  RegexSearchEngine<int> eng;
  std::vector<std::string> keys;
  std::mt19937 rng(987654);

  for (int i = 0; i < 300; ++i) {
    std::string k = "/base/d" + std::to_string(rng() % 20) + "/item_" +
                    std::to_string(i) + ".dat";
    if (eng.Insert(k, i)) keys.push_back(k);
  }
  // Delete ~1/3.
  for (size_t i = 0; i < keys.size(); i += 3) {
    eng.Delete(keys[i]);
  }
  std::vector<std::string> alive;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (i % 3 != 0) alive.push_back(keys[i]);
  }
  // Rename ~1/2 of the survivors into a new subtree.
  std::vector<std::string> final_keys;
  for (size_t i = 0; i < alive.size(); ++i) {
    if (i % 2 == 0) {
      std::string nk = "/moved/m" + std::to_string(i) + ".dat";
      REQUIRE(eng.Rename(alive[i], nk));
      final_keys.push_back(nk);
    } else {
      final_keys.push_back(alive[i]);
    }
  }

  REQUIRE(eng.Size() == final_keys.size());
  for (const auto &pat : std::vector<std::string>{
           ".*\\.dat$", "^/base/.*", "^/moved/.*", ".*item_.*", ".*"}) {
    INFO("pattern: " << pat);
    REQUIRE(SearchKeys(eng, pat) == BruteForce(final_keys, pat));
  }
}

// Concurrent Insert / Delete / Search stress test. RegexSearchEngine is
// internally synchronized (a shared_mutex), so many threads may mutate and
// query it simultaneously without external locking. This exercises exactly
// that: writers hammer Insert/Delete while readers run Search in parallel.
// Without the internal lock this races entries_/index_ and corrupts/crashes
// (best surfaced under ThreadSanitizer, but the stress alone catches most).
TEST_CASE("RegexSearch: parallel insert/delete/search", "[regex_search][parallel]") {
  RegexSearchEngine<int> eng;

  constexpr int kKeySpace = 400;   // distinct keys touched
  constexpr int kIters = 4000;     // mutations per writer thread
  constexpr int kInserters = 4;
  constexpr int kDeleters = 4;
  constexpr int kSearchers = 3;

  // Keys live in 20 directories so the search prefilter has real work.
  auto key = [](int i) {
    return "/p/dir" + std::to_string(i % 20) + "/file" + std::to_string(i) +
           ".dat";
  };

  // Seed half the space so deleters have targets from the start.
  for (int i = 0; i < kKeySpace; i += 2) {
    eng.Insert(key(i), i);
  }

  std::atomic<bool> stop{false};
  std::atomic<size_t> search_count{0};
  std::atomic<size_t> bad_match{0};  // any Search result not matching the regex

  auto writer = [&](int seed, bool do_insert) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, kKeySpace - 1);
    for (int n = 0; n < kIters; ++n) {
      int i = pick(rng);
      if (do_insert) {
        eng.Insert(key(i), i);
      } else {
        eng.Delete(key(i));
      }
    }
  };

  auto searcher = [&]() {
    // The pattern matches every key in the space; results are a live subset.
    const std::string pat = "^/p/dir[0-9]+/file[0-9]+\\.dat$";
    std::regex re(pat);
    while (!stop.load(std::memory_order_relaxed)) {
      auto res = eng.Search(pat);
      // keys() is a snapshot — safe to walk with no external lock. Every key
      // it reports must genuinely match the pattern (proves Search returned a
      // consistent, non-corrupt view even under concurrent mutation).
      for (const auto &k : res.keys()) {
        if (!std::regex_match(k, re)) {
          bad_match.fetch_add(1, std::memory_order_relaxed);
        }
      }
      search_count.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < kInserters; ++t) threads.emplace_back(writer, 100 + t, true);
  for (int t = 0; t < kDeleters; ++t) threads.emplace_back(writer, 900 + t, false);
  for (int t = 0; t < kSearchers; ++t) threads.emplace_back(searcher);

  // Join the writers first, then signal searchers to stop.
  for (int t = 0; t < kInserters + kDeleters; ++t) threads[t].join();
  stop.store(true, std::memory_order_relaxed);
  for (int t = kInserters + kDeleters;
       t < kInserters + kDeleters + kSearchers; ++t) {
    threads[t].join();
  }

  // No search ever saw a non-matching (corrupted) key.
  REQUIRE(bad_match.load() == 0);
  // Searchers actually ran concurrently with the writers.
  REQUIRE(search_count.load() > 0);

  // Engine is still fully consistent after the storm: every key a final Search
  // returns is present via Contains(), and Size() agrees with a full scan.
  auto final_res = eng.Search("^/p/.*");
  for (const auto &k : final_res.keys()) {
    REQUIRE(eng.Contains(k));
  }
  REQUIRE(eng.Size() == final_res.keys().size());
}
