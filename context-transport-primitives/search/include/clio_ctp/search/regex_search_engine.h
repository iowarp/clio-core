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

#ifndef CLIO_CTP_SEARCH_REGEX_SEARCH_ENGINE_H_
#define CLIO_CTP_SEARCH_REGEX_SEARCH_ENGINE_H_

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <regex>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ctp::search {

/**
 * A regex-queryable string index that maps string keys to user-defined values.
 *
 * Keys are indexed by their character trigrams (overlapping 3-grams) in an
 * inverted index `trigram -> {keys}`. A `Search(pattern)` derives the set of
 * trigrams that EVERY match of the regex must contain, intersects their posting
 * lists to obtain a small candidate set, and then verifies each candidate with
 * the full regex. This is the technique behind Google Code Search (Cox, "Regular
 * Expression Matching with a Trigram Index").
 *
 * Correctness: the trigram extractor is CONSERVATIVE — it only treats a trigram
 * as "required" when it provably appears in every match (it comes from a literal
 * run that is neither optional nor inside an alternation/group). Whenever it
 * cannot prove a useful required trigram (alternation, groups, short literals,
 * etc.) it falls back to scanning all keys. So the candidate set is always a
 * superset of the true matches, and the final regex verification makes the
 * result exact regardless of how good the prefilter was.
 *
 * Match semantics: a key matches when std::regex_match(key, pattern) is true
 * (the whole key must match), using the ECMAScript grammar (std::regex default).
 *
 * Thread-safety: internally synchronized with a shared_mutex — mutators
 * (Insert/Delete/Rename/Clear) take it exclusively, queries (Search/Contains/
 * Find/Size/Empty) take it shared. Search returns a SNAPSHOT of the matching
 * (key, value) pairs, so a SearchResult — keys() and the value-yielding
 * iterator alike — is stable under concurrent mutation. The previous design
 * fetched values lazily via Find() at iteration time; a concurrent Delete then
 * made the iterator dereference Find()'s nullptr — a NULL TagId read that
 * crashed the runtime under the #807 CFS stress (readdir's TagQuery racing
 * unlink).
 *
 * Search is POINT-IN-TIME consistent (issue #919): the pairs it returns are all
 * live at one instant, never a blend of two. It gets there optimistically — the
 * expensive regex pass still runs with the lock released (see Search), but the
 * result is only accepted if no mutation landed while it ran; otherwise the
 * whole query is retried, and a query that keeps losing the race falls back to
 * a single fully-locked pass so it cannot livelock.
 *
 * Dropping that guarantee is not merely a stale read — it can make a live key
 * vanish entirely. A key renamed A->B between the candidate scan and the value
 * snapshot was captured as A (still live then) but is looked up as A after the
 * move, while B was never a candidate at all. Both names match a readdir's
 * "^<dir>/[^/]+$", so the file disappeared from a listing of its own directory:
 * the cr_cli_cfs_rename flake, whose D4 case asserts exactly that a concurrent
 * rename may relabel an entry but must never drop it.
 */
template <typename ValueT>
class RegexSearchEngine {
 public:
  RegexSearchEngine() = default;

  /**
   * Bind `value` to `key`. If `key` already exists its value is overwritten.
   * @return true if a new key was added, false if an existing one was updated.
   */
  bool Insert(const std::string &key, const ValueT &value) {
    std::unique_lock<std::shared_mutex> lk(mtx_);
    const bool is_new = InsertLocked(key, value);
    BumpVersionLocked();
    return is_new;
  }

  /** Remove `key`. @return true if it existed. */
  bool Delete(const std::string &key) {
    std::unique_lock<std::shared_mutex> lk(mtx_);
    const bool existed = DeleteLocked(key);
    BumpVersionLocked();
    return existed;
  }

 private:
  /** Insert body; caller must hold mtx_ exclusively. */
  bool InsertLocked(const std::string &key, const ValueT &value) {
    auto res = entries_.insert_or_assign(key, value);
    const bool is_new = res.second;
    if (is_new) {
      // Postings store a stable pointer to the single key copy owned by
      // entries_ (node pointers/refs survive rehash; only erasing that key
      // invalidates it, and Delete scrubs the postings first). This avoids
      // copying the (possibly long) key into every one of its trigram postings.
      const std::string *kp = &res.first->first;
      std::vector<std::string> tg;
      Trigrams(key, tg);
      for (const auto &t : tg) {
        index_[t].insert(kp);
      }
    }
    return is_new;
  }

  /** Delete body; caller must hold mtx_ exclusively. */
  bool DeleteLocked(const std::string &key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return false;
    }
    // Scrub the postings (by pointer) BEFORE erasing the entries_ node, so the
    // stored pointer never dangles.
    const std::string *kp = &it->first;
    std::vector<std::string> tg;
    Trigrams(key, tg);
    for (const auto &t : tg) {
      auto p = index_.find(t);
      if (p != index_.end()) {
        p->second.erase(kp);
        if (p->second.empty()) {
          index_.erase(p);
        }
      }
    }
    entries_.erase(it);
    return true;
  }

  /**
   * Move the entry at `old_key` to `new_key`, preserving its value. If
   * `new_key` already exists its value is overwritten by the moved one.
   * @return false if `old_key` does not exist.
   */
 public:
  bool Rename(const std::string &old_key, const std::string &new_key) {
    std::unique_lock<std::shared_mutex> lk(mtx_);
    auto it = entries_.find(old_key);
    if (it == entries_.end()) {
      return false;
    }
    if (old_key == new_key) {
      return true;
    }
    ValueT moved = std::move(it->second);
    DeleteLocked(old_key);
    InsertLocked(new_key, moved);
    BumpVersionLocked();
    return true;
  }

  bool Contains(const std::string &key) const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return entries_.find(key) != entries_.end();
  }

  /** @return pointer to the value bound to `key`, or nullptr if absent. NOTE:
   *  the pointer is only stable while no concurrent mutation occurs. */
  const ValueT *Find(const std::string &key) const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : &it->second;
  }

  size_t Size() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return entries_.size();
  }
  bool Empty() const {
    std::shared_lock<std::shared_mutex> lk(mtx_);
    return entries_.empty();
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lk(mtx_);
    entries_.clear();
    index_.clear();
    BumpVersionLocked();
  }

  /**
   * Iterable result of a Search. Holds a snapshot of the matching keys; the
   * bound values are fetched live from the engine on dereference, so the engine
   * must outlive the result and must not be mutated while iterating.
   */
  class SearchResult {
   public:
    /** (key, value) view yielded on dereference. */
    using reference = std::pair<const std::string &, const ValueT &>;

    class iterator {
     public:
      iterator(const SearchResult *r, size_t i) : r_(r), i_(i) {}
      bool operator==(const iterator &o) const { return i_ == o.i_; }
      bool operator!=(const iterator &o) const { return i_ != o.i_; }
      iterator &operator++() {
        ++i_;
        return *this;
      }
      reference operator*() const {
        // Both halves come from the result's own snapshot — no live engine
        // access, so iteration is safe against concurrent mutation.
        return reference(r_->keys_[i_], r_->values_[i_]);
      }

     private:
      const SearchResult *r_;
      size_t i_;
    };

    iterator begin() const { return iterator(this, 0); }
    iterator end() const { return iterator(this, keys_.size()); }
    size_t size() const { return keys_.size(); }
    bool empty() const { return keys_.empty(); }
    /** The matching keys (sorted), e.g. for callers that only need names. */
    const std::vector<std::string> &keys() const { return keys_; }

   private:
    friend class RegexSearchEngine;
    SearchResult(std::vector<std::string> keys, std::vector<ValueT> values)
        : keys_(std::move(keys)), values_(std::move(values)) {}
    std::vector<std::string> keys_;   // sorted; aligned with values_
    std::vector<ValueT> values_;      // snapshot taken under the shared lock
  };

  /**
   * Return every (key, value) whose key fully matches `pattern`.
   *
   * The returned pairs were all live at a single instant: a mutation that lands
   * mid-query invalidates the pass and the query is retried (#919).
   *
   * @throws std::regex_error if `pattern` is not a valid ECMAScript regex.
   */
  SearchResult Search(const std::string &pattern) const {
    // Compile the regex OUTSIDE the lock -- it touches no shared state and is
    // ~50-100us, which would otherwise be held under the shared lock on every
    // query and starve writers (#680: 3 tight-loop searchers vs 8 writers hung
    // the parallel test intermittently under load).
    //
    // MEASURED, do not "optimize" this into a cache: a thread_local
    // compiled-pattern cache was tried against readdir, whose pattern
    // ("^<dir>/[^/]+$") repeats on every call and is the best case such a
    // cache can have. Four runs each, readdir over a 10-entry directory:
    // cached median 18,913 ops/s vs uncached 18,549 -- a 2% difference inside
    // this benchmark's noise, and the cache was slightly WORSE on the mean.
    // Whatever dominates a small query, it is not this compile.
    std::regex re(pattern);  // throws on bad pattern

    std::vector<std::string> required;
    const bool prefilterable = ExtractRequiredTrigrams(pattern, required);
    const bool use_prefilter = prefilterable && !required.empty();

    // Optimistic pass: snapshot candidate keys under the shared lock, release
    // it for the expensive regex_match (that release is the whole point of
    // #680), then re-acquire to snapshot the values. Accept the pass only if
    // the mutation counter did not move across it -- otherwise the two locked
    // sections saw different states, and a key that was renamed in between
    // would silently drop out of the result (#919).
    constexpr int kOptimisticAttempts = 3;
    for (int attempt = 0; attempt < kOptimisticAttempts; ++attempt) {
      std::vector<std::string> candidates;
      uint64_t version_before = 0;
      {
        std::shared_lock<std::shared_mutex> lk(mtx_);
        version_before = version_.load(std::memory_order_relaxed);
        if (!CollectCandidatesLocked(use_prefilter, required, &candidates)) {
          // A required trigram indexes no key at all: no match is possible, and
          // that verdict needed only the one consistent look.
          return SearchResult(std::vector<std::string>(),
                              std::vector<ValueT>());
        }
      }

      std::vector<std::string> matches = MatchAndSort(&candidates, re);

      std::shared_lock<std::shared_mutex> lk(mtx_);
      if (version_.load(std::memory_order_relaxed) != version_before) {
        continue;  // the index moved under us -- redo the query
      }
      return BuildResultLocked(&matches);
    }

    // Repeatedly outraced by writers. Fall back to ONE fully-locked pass: it is
    // consistent by construction and cannot be invalidated, so a directory hot
    // enough to defeat the optimistic path still terminates. Writers wait for a
    // single regex pass here, which is why this is the exception and not the
    // rule.
    std::shared_lock<std::shared_mutex> lk(mtx_);
    std::vector<std::string> candidates;
    if (!CollectCandidatesLocked(use_prefilter, required, &candidates)) {
      return SearchResult(std::vector<std::string>(), std::vector<ValueT>());
    }
    std::vector<std::string> matches = MatchAndSort(&candidates, re);
    return BuildResultLocked(&matches);
  }

 private:
  /**
   * Gather the keys the regex could possibly match into `*out`. Caller must
   * hold mtx_ (shared is enough).
   * @return false if a required trigram indexes no key, i.e. the pattern cannot
   *         match anything and `*out` is meaningless.
   */
  bool CollectCandidatesLocked(bool use_prefilter,
                               const std::vector<std::string> &required,
                               std::vector<std::string> *out) const {
    out->clear();
    if (!use_prefilter) {
      // No usable prefilter: every key is a candidate.
      out->reserve(entries_.size());
      for (const auto &kv : entries_) {
        out->push_back(kv.first);
      }
      return true;
    }
    // Candidates = keys present in the posting lists of ALL required trigrams.
    // Walk the smallest posting list and test membership in the others; this is
    // a superset of the true matches.
    const std::unordered_set<const std::string *> *smallest = nullptr;
    for (const auto &t : required) {
      auto it = index_.find(t);
      if (it == index_.end()) {
        return false;
      }
      if (smallest == nullptr || it->second.size() < smallest->size()) {
        smallest = &it->second;
      }
    }
    for (const std::string *kp : *smallest) {
      bool in_all = true;
      for (const auto &t : required) {
        const auto &posting = index_.find(t)->second;
        if (posting.find(kp) == posting.end()) {
          in_all = false;
          break;
        }
      }
      if (in_all) {
        out->push_back(*kp);  // copy under the lock
      }
    }
    return true;
  }

  /**
   * Verify each candidate against `re` and return the survivors, sorted.
   * Deliberately callable with no lock held -- matching is the expensive part.
   * Consumes `*candidates`.
   */
  static std::vector<std::string> MatchAndSort(
      std::vector<std::string> *candidates, const std::regex &re) {
    std::vector<std::string> matches;
    for (auto &c : *candidates) {
      if (std::regex_match(c, re)) {
        matches.push_back(std::move(c));
      }
    }
    std::sort(matches.begin(), matches.end());
    return matches;
  }

  /**
   * Pair each matched key with its value. Caller must hold mtx_ (shared) AND
   * have established that the index did not change since the candidate scan, so
   * every key is still present. Consumes `*matches`.
   */
  SearchResult BuildResultLocked(std::vector<std::string> *matches) const {
    std::vector<std::string> keys;
    std::vector<ValueT> values;
    keys.reserve(matches->size());
    values.reserve(matches->size());
    for (auto &k : *matches) {
      auto it = entries_.find(k);
      if (it == entries_.end()) {
        continue;  // unreachable while the version check holds; cheap guard
      }
      values.push_back(it->second);
      keys.push_back(std::move(k));
    }
    return SearchResult(std::move(keys), std::move(values));
  }

  /**
   * Stamp a mutation. Caller must hold mtx_ exclusively, which is what lets a
   * Search compare the counter under a mere shared lock: any writer that could
   * change it is excluded for the whole of both the reader's locked sections.
   */
  void BumpVersionLocked() {
    version_.store(version_.load(std::memory_order_relaxed) + 1,
                   std::memory_order_relaxed);
  }

  /** Append the unique overlapping trigrams of `s` to `out` (empty if <3). */
  static void Trigrams(const std::string &s, std::vector<std::string> &out) {
    out.clear();
    if (s.size() < 3) {
      return;
    }
    std::set<std::string> uniq;
    for (size_t i = 0; i + 3 <= s.size(); ++i) {
      uniq.insert(s.substr(i, 3));
    }
    out.assign(uniq.begin(), uniq.end());
  }

  /**
   * Derive the trigrams that EVERY match of `pattern` must contain (AND
   * semantics). Conservative: returns false (caller must scan all keys) for any
   * construct it can't reason about safely — alternation `|`, groups `()`, a
   * dangling escape. Returns true with `out` = required trigrams otherwise; an
   * empty `out` means "no useful constraint, scan all". The function NEVER emits
   * a trigram that is not guaranteed to appear in every match.
   */
  static bool ExtractRequiredTrigrams(const std::string &p,
                                      std::vector<std::string> &out) {
    out.clear();
    std::vector<std::string> runs;
    std::string run;
    auto flush = [&]() {
      if (run.size() >= 3) {
        runs.push_back(run);
      }
      run.clear();
    };

    const size_t n = p.size();
    size_t i = 0;
    while (i < n) {
      const char c = p[i];
      switch (c) {
        case '|':
        case '(':
        case ')':
          // Alternation / groups: a literal here is not guaranteed in every
          // match. Bail to a full scan.
          return false;
        case '\\': {
          if (i + 1 >= n) {
            return false;  // dangling escape -> bail
          }
          const char nx = p[i + 1];
          if (std::isalnum(static_cast<unsigned char>(nx))) {
            // \d \w \s \b ... a class, not a reliable literal.
            flush();
          } else {
            run.push_back(nx);  // \. \/ \+ \\ ... a literal character
          }
          i += 2;
          continue;
        }
        case '[': {
          // Character class matches one of a set -> not a fixed literal.
          flush();
          i++;
          if (i < n && p[i] == '^') {
            i++;
          }
          if (i < n && p[i] == ']') {
            i++;  // a leading ']' is a literal member of the class
          }
          while (i < n && p[i] != ']') {
            if (p[i] == '\\' && i + 1 < n) {
              i += 2;
            } else {
              i++;
            }
          }
          if (i < n) {
            i++;  // consume ']'
          }
          continue;
        }
        case '.':
        case '^':
        case '$':
          flush();
          i++;
          continue;
        case '*':
        case '?':
          // The preceding atom is optional: drop it and break the run.
          if (!run.empty()) {
            run.pop_back();
          }
          flush();
          i++;
          continue;
        case '+':
          // The preceding atom occurs >=1 times: it stays in the run, but the
          // run cannot extend across the repetition.
          flush();
          i++;
          continue;
        case '{': {
          // Possible {n}, {n,}, {n,m} quantifier. Parse the minimum count.
          size_t j = i + 1;
          std::string mn;
          while (j < n && std::isdigit(static_cast<unsigned char>(p[j]))) {
            mn.push_back(p[j]);
            j++;
          }
          size_t k = j;
          while (k < n && p[k] != '}') {
            k++;
          }
          const bool is_quant =
              !mn.empty() && k < n && (p[j] == '}' || p[j] == ',');
          if (!is_quant) {
            run.push_back('{');  // a literal '{'
            i++;
            continue;
          }
          const long min_count = std::atol(mn.c_str());
          if (min_count == 0 && !run.empty()) {
            run.pop_back();  // preceding atom optional
          }
          flush();
          i = k + 1;
          continue;
        }
        default:
          run.push_back(c);
          i++;
          continue;
      }
    }
    flush();

    std::set<std::string> tri;
    for (const auto &r : runs) {
      for (size_t a = 0; a + 3 <= r.size(); ++a) {
        tri.insert(r.substr(a, 3));
      }
    }
    out.assign(tri.begin(), tri.end());
    return true;
  }

  std::unordered_map<std::string, ValueT> entries_;
  // trigram -> set of pointers to keys (owned by entries_). Pointers, not
  // copies, to keep the index small for long keys.
  std::unordered_map<std::string, std::unordered_set<const std::string *>>
      index_;
  // Guards entries_ + index_: exclusive for mutators, shared for queries.
  mutable std::shared_mutex mtx_;
  // Mutation counter. Bumped under the exclusive lock, read under the shared
  // one, so Search can tell whether the key set moved under it (#919).
  std::atomic<uint64_t> version_{0};
};

}  // namespace ctp::search

#endif  // CLIO_CTP_SEARCH_REGEX_SEARCH_ENGINE_H_
