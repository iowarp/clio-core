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
#include <cstdlib>
#include <regex>
#include <set>
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
 * Thread-safety: NOT internally synchronized. Callers that mutate and query
 * concurrently must provide external synchronization. References returned by a
 * SearchResult are valid only while the engine is unmodified.
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
    auto res = entries_.insert_or_assign(key, value);
    const bool is_new = res.second;
    if (is_new) {
      std::vector<std::string> tg;
      Trigrams(key, tg);
      for (const auto &t : tg) {
        index_[t].insert(key);
      }
    }
    return is_new;
  }

  /** Remove `key`. @return true if it existed. */
  bool Delete(const std::string &key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return false;
    }
    entries_.erase(it);
    std::vector<std::string> tg;
    Trigrams(key, tg);
    for (const auto &t : tg) {
      auto p = index_.find(t);
      if (p != index_.end()) {
        p->second.erase(key);
        if (p->second.empty()) {
          index_.erase(p);
        }
      }
    }
    return true;
  }

  /**
   * Move the entry at `old_key` to `new_key`, preserving its value. If
   * `new_key` already exists its value is overwritten by the moved one.
   * @return false if `old_key` does not exist.
   */
  bool Rename(const std::string &old_key, const std::string &new_key) {
    auto it = entries_.find(old_key);
    if (it == entries_.end()) {
      return false;
    }
    if (old_key == new_key) {
      return true;
    }
    ValueT moved = std::move(it->second);
    Delete(old_key);
    Insert(new_key, moved);
    return true;
  }

  bool Contains(const std::string &key) const {
    return entries_.find(key) != entries_.end();
  }

  /** @return pointer to the value bound to `key`, or nullptr if absent. */
  const ValueT *Find(const std::string &key) const {
    auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : &it->second;
  }

  size_t Size() const { return entries_.size(); }
  bool Empty() const { return entries_.empty(); }

  void Clear() {
    entries_.clear();
    index_.clear();
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
        const std::string &k = r_->keys_[i_];
        return reference(k, *r_->eng_->Find(k));
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
    SearchResult(std::vector<std::string> keys, const RegexSearchEngine *eng)
        : keys_(std::move(keys)), eng_(eng) {}
    std::vector<std::string> keys_;
    const RegexSearchEngine *eng_;
  };

  /**
   * Return every (key, value) whose key fully matches `pattern`.
   * @throws std::regex_error if `pattern` is not a valid ECMAScript regex.
   */
  SearchResult Search(const std::string &pattern) const {
    std::regex re(pattern);  // compiled once per query; throws on bad pattern

    std::vector<std::string> required;
    const bool prefilterable = ExtractRequiredTrigrams(pattern, required);

    std::vector<std::string> matches;
    if (!prefilterable || required.empty()) {
      // No usable prefilter: verify the regex against every key.
      for (const auto &kv : entries_) {
        if (std::regex_match(kv.first, re)) {
          matches.push_back(kv.first);
        }
      }
    } else {
      // Candidates = keys present in the posting lists of ALL required
      // trigrams. Walk the smallest posting list and test membership in the
      // others; this is a superset of the true matches.
      const std::unordered_set<std::string> *smallest = nullptr;
      for (const auto &t : required) {
        auto it = index_.find(t);
        if (it == index_.end()) {
          // Some required trigram indexes no key at all => no match possible.
          return SearchResult(std::vector<std::string>(), this);
        }
        if (smallest == nullptr || it->second.size() < smallest->size()) {
          smallest = &it->second;
        }
      }
      for (const auto &key : *smallest) {
        bool in_all = true;
        for (const auto &t : required) {
          const auto &posting = index_.find(t)->second;
          if (posting.find(key) == posting.end()) {
            in_all = false;
            break;
          }
        }
        if (in_all && std::regex_match(key, re)) {
          matches.push_back(key);
        }
      }
    }

    std::sort(matches.begin(), matches.end());
    return SearchResult(std::move(matches), this);
  }

 private:
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
  std::unordered_map<std::string, std::unordered_set<std::string>> index_;
};

}  // namespace ctp::search

#endif  // CLIO_CTP_SEARCH_REGEX_SEARCH_ENGINE_H_
