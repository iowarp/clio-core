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

#ifndef CLIO_BDEV_DAOS_PATH_H_
#define CLIO_BDEV_DAOS_PATH_H_

#include <cstdint>
#include <string>
#include <vector>

#include "clio_ctp/util/config_parse.h"

namespace clio::run::bdev {

/**
 * Where a DAOS bdev target lives, parsed from its pool name:
 *
 *   daos://<pool>/<container>/<file path>[?key=value[&key=value...]]
 *
 * <pool> and <container> are DAOS labels (or UUID strings). <file path> is
 * the path of the backing file inside the (POSIX) container; it may contain
 * directories, which the transport creates. Recognised keys:
 *
 *   chunk_size  DFS chunk size of the file, a size string ("1m", "4MB").
 *               Default 1 MiB. Used only when the file is created.
 *   oclass      DAOS object class name of the file ("SX", "EC_16P2GX", ...).
 *               Default: the container's default. Used only at creation.
 *   alloc_log   LOCAL path of the allocator write-ahead log. Default: none
 *               (CreateParams::alloc_log_path_, when set, takes precedence).
 *   prezero     1 (default) zeroes the destination buffer before every read,
 *               so a block that was allocated but never written reads as
 *               zeros whether or not the DAOS client fills holes itself.
 *               0 skips that pass over the buffer; use it only on a DAOS
 *               version known to return zeros for holes.
 *   chunk_align 1 (default) starts every block of at least chunk_size bytes
 *               on a chunk boundary. A chunk is the unit DAOS stores (and,
 *               with a redundant object class, erasure-codes): a 1 MiB write
 *               that straddles two chunks was measured 3.4x slower than an
 *               aligned one. 0 packs blocks back to back as other devices do.
 *   io_threads  Number of dedicated DAOS I/O threads of the process, 1..64.
 *               Default: CLIO_DAOS_IO_THREADS, else 4. The threads are shared
 *               by all DAOS devices of a process, so only the first device
 *               opened decides the count.
 */
struct DaosBdevPath {
  std::string pool_;       /**< Pool label or UUID string. */
  std::string cont_;       /**< Container label or UUID string. */
  std::vector<std::string> dirs_; /**< Parent directories, outermost first. */
  std::string file_;       /**< File name inside the innermost directory. */
  uint64_t chunk_size_ = kDefaultChunkSize; /**< DFS chunk size in bytes. */
  std::string oclass_;     /**< Object class name; empty = container default. */
  std::string alloc_log_;  /**< Local allocator-log path; empty = none. */
  bool prezero_ = true;    /**< Zero the destination before each read. */
  bool chunk_align_ = true; /**< Start blocks >= chunk_size on a chunk. */
  uint32_t io_threads_ = 0; /**< DAOS I/O threads; 0 = environment/default. */

  /** Default DFS chunk size: 1 MiB. */
  static constexpr uint64_t kDefaultChunkSize = uint64_t(1) << 20;
  /** URI scheme of a DAOS bdev pool name. */
  static constexpr const char *kScheme = "daos://";
  /** Largest accepted io_threads value. */
  static constexpr uint32_t kMaxIoThreads = 64;

  /**
   * The file's path inside the container, for messages.
   * @return "/dir/.../file"
   */
  std::string FilePath() const {
    std::string out;
    for (const std::string &d : dirs_) {
      out += "/" + d;
    }
    return out + "/" + file_;
  }

  /**
   * Split a trailing "_node<digits>" off `text`. The CTE names every storage
   * target "<configured path>_node<N>", so for a path that carries a query
   * string the suffix lands on the last option value instead of on the file
   * name; the parser moves it back.
   * @param text in: string to inspect; out: the string without the suffix
   * @return the suffix ("_node3"), or empty when `text` does not end in one
   */
  static std::string TakeNodeSuffix(std::string &text) {
    static const std::string kTag = "_node";
    const size_t pos = text.rfind(kTag);
    if (pos == std::string::npos || pos + kTag.size() >= text.size()) {
      return std::string();
    }
    for (size_t i = pos + kTag.size(); i < text.size(); ++i) {
      if (text[i] < '0' || text[i] > '9') {
        return std::string();
      }
    }
    std::string suffix = text.substr(pos);
    text.erase(pos);
    return suffix;
  }

  /**
   * Whether `text` is a size string ConfigParse::ParseSize accepts: digits
   * (with an optional fraction) followed by nothing or by a k/m/g/t unit.
   * Checked first because ParseSize treats an unknown unit as fatal.
   * @param text candidate size string
   * @return true if it can be handed to ParseSize
   */
  static bool IsSizeString(const std::string &text) {
    size_t i = 0;
    while (i < text.size() &&
           ((text[i] >= '0' && text[i] <= '9') || text[i] == '.')) {
      ++i;
    }
    if (i == 0) {
      return false;
    }
    if (i == text.size()) {
      return true;
    }
    return std::string("kKmMgGtT").find(text[i]) != std::string::npos;
  }

  /**
   * Apply one "key=value" option of the query string.
   * @param option the option text
   * @param error out: why the option was rejected
   * @return false if the key is unknown or the value is unusable
   */
  bool ApplyOption(const std::string &option, std::string *error) {
    const size_t eq = option.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 >= option.size()) {
      *error = "option '" + option + "' is not key=value";
      return false;
    }
    const std::string key = option.substr(0, eq);
    const std::string value = option.substr(eq + 1);
    if (key == "chunk_size") {
      if (!IsSizeString(value)) {
        *error = "chunk_size '" + value + "' is not a size (e.g. 1m, 4MB)";
        return false;
      }
      chunk_size_ = ctp::ConfigParse::ParseSize(value);
      if (chunk_size_ == 0) {
        *error = "chunk_size '" + value + "' must be greater than 0";
        return false;
      }
    } else if (key == "oclass") {
      oclass_ = value;
    } else if (key == "alloc_log") {
      alloc_log_ = value;
    } else if (key == "prezero") {
      if (value != "0" && value != "1") {
        *error = "prezero '" + value + "' must be 0 or 1";
        return false;
      }
      prezero_ = (value == "1");
    } else if (key == "chunk_align") {
      if (value != "0" && value != "1") {
        *error = "chunk_align '" + value + "' must be 0 or 1";
        return false;
      }
      chunk_align_ = (value == "1");
    } else if (key == "io_threads") {
      return ApplyIoThreads(value, error);
    } else {
      *error = "unknown option '" + key +
               "' (expected chunk_size, oclass, alloc_log, prezero, "
               "chunk_align or io_threads)";
      return false;
    }
    return true;
  }

  /**
   * Apply the io_threads option.
   * @param value the option value
   * @param error out: why the value was rejected
   * @return false unless the value is an integer in 1..kMaxIoThreads
   */
  bool ApplyIoThreads(const std::string &value, std::string *error) {
    uint32_t n = 0;
    bool ok = !value.empty() && value.size() <= 2;
    for (size_t i = 0; ok && i < value.size(); ++i) {
      ok = value[i] >= '0' && value[i] <= '9';
      n = n * 10 + static_cast<uint32_t>(value[i] - '0');
    }
    if (!ok || n == 0 || n > kMaxIoThreads) {
      *error = "io_threads '" + value + "' must be an integer from 1 to " +
               std::to_string(kMaxIoThreads);
      return false;
    }
    io_threads_ = n;
    return true;
  }

  /**
   * Split "<pool>/<container>/<dir>/.../<file>" into its parts.
   * @param location the URI without scheme and query string
   * @param error out: why the location was rejected
   * @return false unless pool, container and file name are all present
   */
  bool ParseLocation(const std::string &location, std::string *error) {
    std::vector<std::string> parts;
    size_t begin = 0;
    while (begin <= location.size()) {
      size_t end = location.find('/', begin);
      if (end == std::string::npos) end = location.size();
      if (end > begin) {
        parts.push_back(location.substr(begin, end - begin));
      }
      begin = end + 1;
    }
    if (parts.size() < 3) {
      *error = "expected daos://<pool>/<container>/<file path>";
      return false;
    }
    for (const std::string &p : parts) {
      if (p == "." || p == "..") {
        *error = "'.' and '..' are not allowed in the path";
        return false;
      }
    }
    pool_ = parts[0];
    cont_ = parts[1];
    dirs_.assign(parts.begin() + 2, parts.end() - 1);
    file_ = parts.back();
    return true;
  }

  /**
   * Parse a DAOS bdev pool name.
   * @param pool_name the bdev pool name (the storage target's `path`)
   * @param error out: why the name was rejected
   * @return false if the name is not a usable daos:// URI
   */
  bool Parse(const std::string &pool_name, std::string *error) {
    const std::string scheme = kScheme;
    if (pool_name.rfind(scheme, 0) != 0) {
      *error = "expected daos://<pool>/<container>/<file path>";
      return false;
    }
    std::string location = pool_name.substr(scheme.size());
    std::string query;
    const size_t qmark = location.find('?');
    if (qmark != std::string::npos) {
      query = location.substr(qmark + 1);
      location.erase(qmark);
      // "<file>?<options>_node<N>" -> "<file>_node<N>" + "<options>".
      location += TakeNodeSuffix(query);
    }
    if (!ParseLocation(location, error)) {
      return false;
    }
    size_t begin = 0;
    while (begin < query.size()) {
      size_t end = query.find('&', begin);
      if (end == std::string::npos) end = query.size();
      if (end > begin &&
          !ApplyOption(query.substr(begin, end - begin), error)) {
        return false;
      }
      begin = end + 1;
    }
    return true;
  }
};

}  // namespace clio::run::bdev

#endif  // CLIO_BDEV_DAOS_PATH_H_
