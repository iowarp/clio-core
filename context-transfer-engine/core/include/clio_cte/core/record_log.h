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
 * Append-only, self-checking record log for one container's durable state.
 *
 * Shared by CTE modules that keep a node-local slice of metadata (the
 * clio-fs namespace shard, stream sizes and merge plans). Every mutation
 * appends one record with a single write(2), so a daemon crash (kill -9)
 * loses nothing that returned to the caller. At startup the log is replayed,
 * a torn tail is cut off, and the owner rewrites its live state as a compact
 * snapshot.
 *
 * Record: [u32 magic][u32 len][u8 type][payload(len)][u32 fnv1a(type+payload)]
 */
#ifndef CLIO_CTE_CORE_RECORD_LOG_H_
#define CLIO_CTE_CORE_RECORD_LOG_H_

#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <clio_runtime/clio_runtime.h>

namespace clio::cte::core {

/** One container's append-only record log (thread-safe). */
class RecordLog {
 public:
  RecordLog() = default;
  ~RecordLog();
  RecordLog(const RecordLog &) = delete;
  RecordLog &operator=(const RecordLog &) = delete;

  /**
   * Open (creating parent directories and the file) for replay + append.
   * @param path log file path
   * @return true on success
   */
  bool Open(const std::string &path);

  /** @return true once Open succeeded. */
  bool IsOpen() const { return fd_ >= 0; }

  /**
   * Replay every intact record in order, then cut the file after the last
   * intact one (a crash can leave a torn final record).
   * @param fn called with (record type, payload) per record
   * @return number of records replayed
   */
  size_t Replay(
      const std::function<void(clio::run::u32, const std::string &)> &fn);

  /**
   * Append one record with a single write(2). No-op when not open.
   * @param type record type (low 8 bits are stored)
   * @param payload encoded record body
   */
  void Append(clio::run::u32 type, const std::string &payload);

  /**
   * Atomically replace the log with `records` (write temp, fsync, rename)
   * and keep appending to the new file.
   * @param records full snapshot of the live state
   * @return true on success (the old log stays in place on failure)
   */
  bool Rewrite(
      const std::vector<std::pair<clio::run::u32, std::string>> &records);

  /** @return bytes appended since the last Open/Rewrite. */
  clio::run::u64 BytesSinceCompact() const { return since_compact_; }

 private:
  /**
   * Serialize one record into `out`.
   * @param type record type
   * @param payload record body
   * @param out destination buffer (appended to)
   */
  static void Frame(clio::run::u32 type, const std::string &payload,
                    std::string *out);

  std::string path_;
  int fd_ = -1;
  std::mutex mu_;
  clio::run::u64 since_compact_ = 0;
};

}  // namespace clio::cte::core

#endif  // CLIO_CTE_CORE_RECORD_LOG_H_
