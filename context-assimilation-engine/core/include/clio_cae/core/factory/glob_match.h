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

#ifndef CLIO_CAE_CORE_GLOB_MATCH_H_
#define CLIO_CAE_CORE_GLOB_MATCH_H_

#include <string>
#include <vector>

#ifndef _WIN32
#include <fnmatch.h>
#endif

namespace clio::cae::core {

namespace glob_detail {

/**
 * Match one character against a bracket expression in a glob pattern.
 *
 * Supports "[abc]", ranges "[a-z]" and negation "[!abc]" / "[^abc]", with a
 * leading ']' taken literally, mirroring fnmatch(3). An unterminated bracket
 * is treated as a literal '[' character, as fnmatch does.
 *
 * @param pattern The full glob pattern.
 * @param pos In: index of the opening '['. Out: index just past the
 *            expression (or past the literal '[' when unterminated).
 * @param c The character to test.
 * @return true if c matches the bracket expression.
 */
inline bool MatchBracket(const std::string& pattern, size_t& pos, char c) {
  size_t i = pos + 1;
  bool negate = false;
  if (i < pattern.size() && (pattern[i] == '!' || pattern[i] == '^')) {
    negate = true;
    ++i;
  }
  bool matched = false;
  bool first = true;
  while (i < pattern.size() && (first || pattern[i] != ']')) {
    first = false;
    char lo = pattern[i];
    char hi = lo;
    if (i + 2 < pattern.size() && pattern[i + 1] == '-' &&
        pattern[i + 2] != ']') {
      hi = pattern[i + 2];
      i += 2;
    }
    if (lo <= c && c <= hi) {
      matched = true;
    }
    ++i;
  }
  if (i >= pattern.size()) {  // no closing ']': literal '['
    pos = pos + 1;
    return c == '[';
  }
  pos = i + 1;
  return matched != negate;
}

}  // namespace glob_detail

/**
 * Portable glob matcher equivalent to fnmatch(pattern, str, 0).
 *
 * '*' matches any sequence (including '/'), '?' any single character,
 * "[...]" a bracket expression, and '\\' escapes the next pattern character.
 * Uses the classic single-backtrack-point algorithm for '*'.
 *
 * @param pattern Glob pattern.
 * @param str String to test.
 * @return true if the whole string matches the whole pattern.
 */
inline bool GlobMatch(const std::string& pattern, const std::string& str) {
  const size_t npos = std::string::npos;
  size_t p = 0, s = 0, star_p = npos, star_s = 0;
  while (s < str.size()) {
    if (p < pattern.size()) {
      size_t next_p = p + 1;
      bool matched = false;
      char pc = pattern[p];
      if (pc == '*') {
        star_p = p;
        star_s = s;
        ++p;
        continue;
      } else if (pc == '?') {
        matched = true;
      } else if (pc == '[') {
        next_p = p;
        matched = glob_detail::MatchBracket(pattern, next_p, str[s]);
      } else {
        if (pc == '\\' && p + 1 < pattern.size()) {
          pc = pattern[p + 1];
          next_p = p + 2;
        }
        matched = (pc == str[s]);
      }
      if (matched) {
        p = next_p;
        ++s;
        continue;
      }
    }
    if (star_p == npos) {
      return false;
    }
    p = star_p + 1;  // backtrack: let the last '*' absorb one more char
    s = ++star_s;
  }
  while (p < pattern.size() && pattern[p] == '*') {
    ++p;
  }
  return p == pattern.size();
}

/**
 * Glob match with the semantics every CAE dataset_filter uses: fnmatch(3)
 * with flags 0 on POSIX ('*' also crosses '/'), the portable matcher above on
 * Windows, which has no fnmatch.
 *
 * @param pattern Glob pattern.
 * @param str String to test.
 * @return true if the whole string matches the whole pattern.
 */
inline bool MatchGlob(const std::string& pattern, const std::string& str) {
#ifdef _WIN32
  return GlobMatch(pattern, str);
#else
  return fnmatch(pattern.c_str(), str.c_str(), 0) == 0;
#endif
}

/**
 * Apply a dataset_filter: any exclude match rejects; otherwise an empty
 * include list accepts, else at least one include pattern must match.
 *
 * @param name Dataset path to test (e.g. "/temperature").
 * @param include_patterns Glob patterns to include (empty: include all).
 * @param exclude_patterns Glob patterns to exclude (checked first).
 * @return true if the dataset is selected.
 */
inline bool MatchesDatasetFilter(
    const std::string& name, const std::vector<std::string>& include_patterns,
    const std::vector<std::string>& exclude_patterns) {
  for (const auto& p : exclude_patterns) {
    if (MatchGlob(p, name)) {
      return false;
    }
  }
  if (include_patterns.empty()) {
    return true;
  }
  for (const auto& p : include_patterns) {
    if (MatchGlob(p, name)) {
      return true;
    }
  }
  return false;
}

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_GLOB_MATCH_H_
