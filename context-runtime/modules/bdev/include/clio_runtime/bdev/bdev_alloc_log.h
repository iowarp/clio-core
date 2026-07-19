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

#ifndef BDEV_ALLOC_LOG_H_
#define BDEV_ALLOC_LOG_H_

#include <clio_runtime/clio_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * Reusable persistent allocator-state log (write-ahead log) for the bdev
 * allocator. This is a SHARED mechanism: the plain bdev uses group_id 0,
 * and safe-bdev (erasure-coding) will namespace its per-group state by a
 * distinct group_id. Local storage only — no remote concerns.
 *
 * On-disk format is a stream of fixed-size 40-byte packed POD records in
 * host byte order. The minimal record set that reproduces the current live
 * state is: one kGroupOpen per live group, then one kAlloc per live block.
 */

namespace clio::run::bdev {

/** Record types for the allocator WAL */
enum class AllocLogType : clio::run::u32 {
  kAlloc = 1,       /**< A block was allocated */
  kFree = 2,        /**< A block was freed */
  kGroupOpen = 3,   /**< A group/stripe was opened */
  kGroupFreeze = 4  /**< A group/stripe was frozen (final row count) */
};

/**
 * Fixed-size binary record (packed POD, host byte order, 40 bytes).
 *
 * Field meaning depends on `type`:
 *  - kAlloc / kFree:   group_id (0 for plain bdev), block_type, offset, size.
 *  - kGroupOpen:       group_id, block_type = k, offset = logical_base,
 *                      size = num_rows, row = first_row.
 *  - kGroupFreeze:     group_id, size = num_rows.
 */
struct AllocLogRecord {
  clio::run::u32 type;        /**< AllocLogType */
  clio::run::u32 group_id;    /**< Allocator/group namespace (0 = plain bdev) */
  clio::run::u32 block_type;  /**< Block size category, or k for kGroupOpen */
  clio::run::u32 reserved;    /**< Padding / reserved for future use */
  clio::run::u64 offset;      /**< Block offset, or logical_base for kGroupOpen */
  clio::run::u64 size;        /**< Block size, or num_rows for group records */
  clio::run::u64 row;         /**< first_row for kGroupOpen */
};
static_assert(sizeof(AllocLogRecord) == 40,
              "AllocLogRecord must be exactly 40 bytes");

/** A recovered live block within a group. */
struct LiveBlock {
  clio::run::u64 offset;
  clio::run::u64 size;
  clio::run::u32 block_type;
};

/** A recovered group record. */
struct GroupRec {
  clio::run::u32 group_id;
  clio::run::u32 k;
  clio::run::u64 first_row;
  clio::run::u64 num_rows;
  clio::run::u64 logical_base;
};

/**
 * Persistent allocator-state log.
 *
 * Holds an in-memory append buffer plus the file path. Logging is
 * thread-safe (a brief std::mutex around buffer mutations). An empty path
 * means logging is disabled (every API is a no-op) — this preserves the
 * pre-WAL behaviour where no file is created.
 */
class AllocatorLog {
 public:
  AllocatorLog() = default;
  ~AllocatorLog() { Close(); }

  AllocatorLog(const AllocatorLog &) = delete;
  AllocatorLog &operator=(const AllocatorLog &) = delete;

  /**
   * Open (or create) the log file.
   * @param path Path to the log file. Empty => logging disabled (no-op).
   * @param recover If true and the file exists, replay it into recovered
   *                state (accessible via groups() / live()).
   * @return true on success (including the disabled/empty-path case).
   */
  bool Open(const std::string &path, bool recover) {
    std::lock_guard<std::mutex> lock(mu_);
    path_ = path;
    buffer_.clear();
    enabled_ = !path.empty();
    live_.clear();
    groups_.clear();
    live_cache_valid_ = false;
    if (!enabled_) {
      return true;  // Disabled: success, but nothing on disk.
    }
    namespace fs = std::filesystem;
    bool exists = fs::exists(path_);
    if (recover && exists) {
      ReplayFileLocked();
    }
    // Ensure the file exists so a later Flush()/append always has a target.
    std::FILE *f = std::fopen(path_.c_str(), "ab");
    if (f == nullptr) {
      enabled_ = false;
      return false;
    }
    std::fclose(f);
    return true;
  }

  /** Append an allocation record to the in-memory buffer. */
  void LogAlloc(clio::run::u32 group_id, clio::run::u64 offset, clio::run::u64 size,
                clio::run::u32 block_type) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) return;
    AllocLogRecord rec{};
    rec.type = static_cast<clio::run::u32>(AllocLogType::kAlloc);
    rec.group_id = group_id;
    rec.block_type = block_type;
    rec.offset = offset;
    rec.size = size;
    buffer_.push_back(rec);
  }

  /** Append a free record to the in-memory buffer. */
  void LogFree(clio::run::u32 group_id, clio::run::u64 offset, clio::run::u64 size,
               clio::run::u32 block_type) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) return;
    AllocLogRecord rec{};
    rec.type = static_cast<clio::run::u32>(AllocLogType::kFree);
    rec.group_id = group_id;
    rec.block_type = block_type;
    rec.offset = offset;
    rec.size = size;
    buffer_.push_back(rec);
  }

  /** Append a group-open record (used by safe-bdev). */
  void LogGroupOpen(clio::run::u32 group_id, clio::run::u32 k, clio::run::u64 first_row,
                    clio::run::u64 num_rows, clio::run::u64 logical_base) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) return;
    AllocLogRecord rec{};
    rec.type = static_cast<clio::run::u32>(AllocLogType::kGroupOpen);
    rec.group_id = group_id;
    rec.block_type = k;
    rec.offset = logical_base;
    rec.size = num_rows;
    rec.row = first_row;
    buffer_.push_back(rec);
  }

  /** Append a group-freeze record (used by safe-bdev). */
  void LogGroupFreeze(clio::run::u32 group_id, clio::run::u64 num_rows) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) return;
    AllocLogRecord rec{};
    rec.type = static_cast<clio::run::u32>(AllocLogType::kGroupFreeze);
    rec.group_id = group_id;
    rec.size = num_rows;
    buffer_.push_back(rec);
  }

  /**
   * Append the buffered records to the file (write + fsync). Idempotent
   * when the buffer is empty. Records flushed here are also folded into the
   * in-memory recovered model so a later Compact() sees a consistent state.
   */
  void Flush() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_ || buffer_.empty()) return;
    std::FILE *f = std::fopen(path_.c_str(), "ab");
    if (f == nullptr) return;
    std::fwrite(buffer_.data(), sizeof(AllocLogRecord), buffer_.size(), f);
    std::fflush(f);
#ifndef _WIN32
    int fd = ::fileno(f);
    if (fd >= 0) {
      ::fsync(fd);
    }
#endif
    std::fclose(f);
    // Fold the just-persisted records into the recovered model.
    for (const auto &rec : buffer_) {
      ApplyRecordLocked(rec);
    }
    records_on_disk_ += buffer_.size();
    buffer_.clear();
  }

  /**
   * Recompute the current state by replaying the WHOLE log (file + buffer)
   * in memory, then rewrite the file as the minimal record set that
   * reproduces it: one kGroupOpen per live group, then one kAlloc per live
   * block (no frees). Written to a temp file and atomically renamed over the
   * original. Bounds the file size to ~live-block count.
   */
  void Compact() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) return;
    // Fold any buffered (not-yet-flushed) records into the model first so we
    // compact against the complete current state.
    for (const auto &rec : buffer_) {
      ApplyRecordLocked(rec);
    }
    buffer_.clear();

    namespace fs = std::filesystem;
    std::string tmp = path_ + ".compact.tmp";
    std::FILE *f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) return;

    clio::run::u64 written = 0;
    // One kGroupOpen per live group.
    for (const auto &kv : groups_) {
      const GroupRec &g = kv.second;
      AllocLogRecord rec{};
      rec.type = static_cast<clio::run::u32>(AllocLogType::kGroupOpen);
      rec.group_id = g.group_id;
      rec.block_type = g.k;
      rec.offset = g.logical_base;
      rec.size = g.num_rows;
      rec.row = g.first_row;
      std::fwrite(&rec, sizeof(rec), 1, f);
      ++written;
    }
    // One kAlloc per live block (no frees).
    for (const auto &gkv : live_) {
      clio::run::u32 group_id = gkv.first;
      for (const auto &bkv : gkv.second) {
        const LiveBlock &b = bkv.second;
        AllocLogRecord rec{};
        rec.type = static_cast<clio::run::u32>(AllocLogType::kAlloc);
        rec.group_id = group_id;
        rec.block_type = b.block_type;
        rec.offset = b.offset;
        rec.size = b.size;
        std::fwrite(&rec, sizeof(rec), 1, f);
        ++written;
      }
    }
    std::fflush(f);
#ifndef _WIN32
    int fd = ::fileno(f);
    if (fd >= 0) {
      ::fsync(fd);
    }
#endif
    std::fclose(f);

    std::error_code ec;
    fs::rename(tmp, path_, ec);
    if (ec) {
      fs::remove(tmp, ec);
      return;
    }
    records_on_disk_ = written;
    live_cache_valid_ = false;
  }

  /** Sync then close the file handle (flushes any buffered records). */
  void Close() {
    Flush();
  }

  /** Recovered group table (after replay). */
  const std::vector<GroupRec> &groups() {
    std::lock_guard<std::mutex> lock(mu_);
    RebuildGroupCacheLocked();
    return group_cache_;
  }

  /**
   * Recovered live blocks for a group (after replaying alloc/free; a free
   * removes the matching live alloc by (group_id, offset)).
   */
  const std::vector<LiveBlock> &live(clio::run::u32 group_id) {
    std::lock_guard<std::mutex> lock(mu_);
    RebuildLiveCacheLocked();
    auto it = live_cache_.find(group_id);
    if (it == live_cache_.end()) {
      return empty_live_;
    }
    return it->second;
  }

  /** Number of fixed-size records currently on disk. */
  clio::run::u64 records_on_disk() {
    std::lock_guard<std::mutex> lock(mu_);
    return records_on_disk_;
  }

  /** Current live-block count across all groups (recovered model). */
  clio::run::u64 live_block_count() {
    std::lock_guard<std::mutex> lock(mu_);
    clio::run::u64 n = 0;
    for (const auto &gkv : live_) {
      n += gkv.second.size();
    }
    return n;
  }

  /** Whether logging is enabled (non-empty path, file opened). */
  bool enabled() {
    std::lock_guard<std::mutex> lock(mu_);
    return enabled_;
  }

  /** On-disk file size in bytes (for tests). */
  clio::run::u64 file_size() {
    std::lock_guard<std::mutex> lock(mu_);
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(path_, ec)) return 0;
    return static_cast<clio::run::u64>(fs::file_size(path_, ec));
  }

 private:
  std::string path_;
  std::vector<AllocLogRecord> buffer_;
  bool enabled_ = false;
  clio::run::u64 records_on_disk_ = 0;
  std::mutex mu_;

  // Recovered in-memory model. live_[group_id][offset] = LiveBlock.
  std::map<clio::run::u32, std::map<clio::run::u64, LiveBlock>> live_;
  std::map<clio::run::u32, GroupRec> groups_;

  // Lazily-rebuilt flat caches for the accessors (so we can hand out
  // stable references). Invalidated whenever the model changes.
  bool live_cache_valid_ = false;
  std::map<clio::run::u32, std::vector<LiveBlock>> live_cache_;
  std::vector<GroupRec> group_cache_;
  std::vector<LiveBlock> empty_live_;

  /** Apply a single record to the in-memory recovered model. */
  void ApplyRecordLocked(const AllocLogRecord &rec) {
    switch (static_cast<AllocLogType>(rec.type)) {
      case AllocLogType::kAlloc: {
        LiveBlock b{rec.offset, rec.size, rec.block_type};
        live_[rec.group_id][rec.offset] = b;
        break;
      }
      case AllocLogType::kFree: {
        auto git = live_.find(rec.group_id);
        if (git != live_.end()) {
          git->second.erase(rec.offset);
        }
        break;
      }
      case AllocLogType::kGroupOpen: {
        GroupRec g{};
        g.group_id = rec.group_id;
        g.k = rec.block_type;
        g.first_row = rec.row;
        g.num_rows = rec.size;
        g.logical_base = rec.offset;
        groups_[rec.group_id] = g;
        break;
      }
      case AllocLogType::kGroupFreeze: {
        auto it = groups_.find(rec.group_id);
        if (it != groups_.end()) {
          it->second.num_rows = rec.size;
        } else {
          GroupRec g{};
          g.group_id = rec.group_id;
          g.num_rows = rec.size;
          groups_[rec.group_id] = g;
        }
        break;
      }
      default:
        break;
    }
    live_cache_valid_ = false;
  }

  /** Replay the on-disk file into the recovered model. */
  void ReplayFileLocked() {
    std::FILE *f = std::fopen(path_.c_str(), "rb");
    if (f == nullptr) return;
    AllocLogRecord rec{};
    clio::run::u64 count = 0;
    while (std::fread(&rec, sizeof(rec), 1, f) == 1) {
      ApplyRecordLocked(rec);
      ++count;
    }
    std::fclose(f);
    records_on_disk_ = count;
  }

  void RebuildLiveCacheLocked() {
    if (live_cache_valid_) return;
    live_cache_.clear();
    for (const auto &gkv : live_) {
      std::vector<LiveBlock> &out = live_cache_[gkv.first];
      out.reserve(gkv.second.size());
      for (const auto &bkv : gkv.second) {
        out.push_back(bkv.second);
      }
    }
    live_cache_valid_ = true;
  }

  void RebuildGroupCacheLocked() {
    group_cache_.clear();
    group_cache_.reserve(groups_.size());
    for (const auto &kv : groups_) {
      group_cache_.push_back(kv.second);
    }
  }
};

}  // namespace clio::run::bdev

#endif  // BDEV_ALLOC_LOG_H_
