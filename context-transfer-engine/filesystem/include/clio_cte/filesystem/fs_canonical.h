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
 * Canonical names: the one key every spelling of a name maps to.
 *
 * On a case-sensitive volume a name is its own canonical name. On a
 * case-insensitive volume it is its Unicode simple case fold (fs_fold_table.h),
 * applied one code point at a time: no normalization, no locale rules, and no
 * full folding (so "Straße" and "STRASSE" stay distinct, as on NTFS).
 * Malformed UTF-8 folds only its ASCII bytes and keeps the rest, so such a
 * name stays addressable by its exact bytes.
 *
 * Shared by clients and the runtime, so both always agree on a key.
 */
#ifndef CLIO_CTE_FILESYSTEM_FS_CANONICAL_H_
#define CLIO_CTE_FILESYSTEM_FS_CANONICAL_H_

#include <algorithm>
#include <cstdint>
#include <string>

#include <clio_cte/filesystem/fs_fold_table.h>

namespace clio::cte::filesystem {

/**
 * Simple case fold of one code point.
 * @param cp code point
 * @return its fold, or cp when it has none
 */
inline uint32_t FsFoldCodePoint(uint32_t cp) {
  if (cp < 0x80) return (cp >= 'A' && cp <= 'Z') ? cp + ('a' - 'A') : cp;
  const FsFoldPair *end = kFsFoldTable + kFsFoldTableSize;
  const FsFoldPair *it = std::lower_bound(
      kFsFoldTable, end, cp,
      [](const FsFoldPair &p, uint32_t v) { return p.from_ < v; });
  return (it != end && it->from_ == cp) ? it->to_ : cp;
}

/**
 * Decode one UTF-8 sequence starting at s[i].
 * @param s string
 * @param i start index
 * @param cp receives the code point
 * @return the sequence length, or 0 when the bytes are not well-formed
 *         UTF-8 (overlong, surrogate, truncated, or out of range)
 */
inline size_t FsUtf8Decode(const std::string &s, size_t i, uint32_t *cp) {
  const auto b = [&s](size_t k) { return static_cast<unsigned char>(s[k]); };
  const unsigned char c = b(i);
  size_t n = 0;
  uint32_t v = 0, min = 0;
  if (c < 0x80) {
    *cp = c;
    return 1;
  } else if ((c & 0xE0) == 0xC0) {
    n = 2, v = c & 0x1F, min = 0x80;
  } else if ((c & 0xF0) == 0xE0) {
    n = 3, v = c & 0x0F, min = 0x800;
  } else if ((c & 0xF8) == 0xF0) {
    n = 4, v = c & 0x07, min = 0x10000;
  } else {
    return 0;
  }
  if (i + n > s.size()) return 0;
  for (size_t k = 1; k < n; ++k) {
    if ((b(i + k) & 0xC0) != 0x80) return 0;
    v = (v << 6) | (b(i + k) & 0x3F);
  }
  if (v < min || v > 0x10FFFF || (v >= 0xD800 && v <= 0xDFFF)) return 0;
  *cp = v;
  return n;
}

/**
 * Append the UTF-8 encoding of a code point.
 * @param cp code point (valid, not a surrogate)
 * @param out string to append to
 */
inline void FsUtf8Append(uint32_t cp, std::string *out) {
  if (cp < 0x80) {
    out->push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

/**
 * Simple case fold of a UTF-8 string (see the file comment).
 * @param name UTF-8 bytes
 * @return the folded name
 */
inline std::string FsFold(const std::string &name) {
  std::string out;
  out.reserve(name.size());
  for (size_t i = 0; i < name.size();) {
    const unsigned char c = static_cast<unsigned char>(name[i]);
    if (c < 0x80) {
      out.push_back(static_cast<char>(FsFoldCodePoint(c)));
      ++i;
      continue;
    }
    uint32_t cp = 0;
    const size_t n = FsUtf8Decode(name, i, &cp);
    if (n == 0) {
      out.push_back(name[i]);  // not UTF-8: keep the byte as is
      ++i;
      continue;
    }
    FsUtf8Append(FsFoldCodePoint(cp), &out);
    i += n;
  }
  return out;
}

/**
 * The canonical name of a single path component.
 * @param name the component as spelled
 * @param case_insensitive whether the volume folds case
 * @return the key the inode table stores it under
 */
inline std::string FsCanonicalName(const std::string &name,
                                   bool case_insensitive) {
  return case_insensitive ? FsFold(name) : name;
}

/**
 * The canonical form of a whole path ('/' separators are never folded).
 * Client caches key on it, so every spelling of a path hits one entry.
 * @param path path as spelled
 * @param case_insensitive whether the volume folds case
 * @return the canonical path
 */
inline std::string FsCanonicalPath(const std::string &path,
                                   bool case_insensitive) {
  return case_insensitive ? FsFold(path) : path;
}

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FS_CANONICAL_H_
