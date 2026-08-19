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

#ifndef WRPCTE_CORE_TRANSACTION_LOG_H_
#define WRPCTE_CORE_TRANSACTION_LOG_H_

#include <clio_runtime/clio_runtime.h>

#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace clio::cte::core {

/** Transaction types for the WAL */
enum class TxnType : uint8_t {
  kCreateNewBlob = 0,
  kExtendBlob = 1,
  kClearBlob = 2,
  kDelBlob = 3,
  kCreateTag = 4,
  kDelTag = 5,
  /**
   * Blob's stored bytes were transformed (issue #818).
   *
   * A separate record rather than a field on kCreateNewBlob for two reasons.
   * The transform is only known at the END of a put, after the bytes have been
   * written, whereas kCreateNewBlob is logged when the blob is first created.
   * And replay must be able to re-apply the mark on top of a blob that
   * kCreateNewBlob has just reinserted with default (untransformed) state --
   * see ReplayTransactionLogs.
   *
   * Appending a type is safe for an older runtime: its replay chain ignores
   * types it does not recognise, which leaves it exactly where it was before
   * this record existed.
   */
  kSetBlobTransform = 6,
  /**
   * Replace one replica's block layout (issue #886).
   *
   * A separate record from kExtendBlob for the same appending-is-safe reason
   * as kSetBlobTransform: an older runtime replaying a newer WAL skips the
   * unknown type, losing only replica layouts it could not have addressed
   * anyway -- the primary's kExtendBlob records replay untouched. Folding a
   * replica index into kExtendBlob instead would make every old record
   * unreadable.
   *
   * Semantics mirror kExtendBlob: the record holds the replica's FULL block
   * layout at log time (full replacement on replay, not a delta).
   */
  kExtendReplica = 7,
};

/** A single block entry within TxnExtendBlob */
struct TxnExtendBlobBlock {
  clio::run::u32 bdev_major_;
  clio::run::u32 bdev_minor_;
  clio::run::PoolQuery target_query_;
  clio::run::u64 target_offset_;
  clio::run::u64 size_;
};

/** Payload: create a new blob (metadata only, no blocks yet) */
struct TxnCreateNewBlob {
  clio::run::u32 tag_major_;
  clio::run::u32 tag_minor_;
  std::string blob_name_;
  float score_;
};

/** Payload: extend (or replace) blob blocks */
struct TxnExtendBlob {
  clio::run::u32 tag_major_;
  clio::run::u32 tag_minor_;
  std::string blob_name_;
  std::vector<TxnExtendBlobBlock> new_blocks_;
};

/** Payload: replace one replica's block layout (issue #886) */
struct TxnExtendReplica {
  clio::run::u32 tag_major_;
  clio::run::u32 tag_minor_;
  std::string blob_name_;
  clio::run::u32 replica_;  // 1-based, Context::replica_ semantics
  std::string replica_name_;
  float score_ = -1.0f;       // the replica's own placement score
  clio::run::u32 flags_ = 0;  // REPLICA_* bits
  clio::run::u32 transform_flags_ = 0;  // THIS copy's transform state (#886)
  float min_score_ = -1.0f;   // organizer rescore floor (-1 = none)
  std::vector<TxnExtendBlobBlock> new_blocks_;
};

/** Payload: clear all blocks from a blob */
struct TxnClearBlob {
  clio::run::u32 tag_major_;
  clio::run::u32 tag_minor_;
  std::string blob_name_;
};

/** Payload: mark a blob's stored bytes as transformed (issue #818) */
struct TxnSetBlobTransform {
  clio::run::u32 tag_major_;
  clio::run::u32 tag_minor_;
  std::string blob_name_;
  clio::run::u32 transform_flags_;
};

/** Payload: delete a blob */
struct TxnDelBlob {
  clio::run::u32 tag_major_;
  clio::run::u32 tag_minor_;
  std::string blob_name_;
};

/** Payload: create a tag */
struct TxnCreateTag {
  std::string tag_name_;
  clio::run::u32 tag_major_;
  clio::run::u32 tag_minor_;
};

/** Payload: delete a tag */
struct TxnDelTag {
  std::string tag_name_;
  clio::run::u32 tag_major_;
  clio::run::u32 tag_minor_;
};

/**
 * Header-only Write-Ahead Transaction Log.
 *
 * Record format on disk:
 *   [u8 txn_type][u32 payload_size][payload bytes]
 *
 * The payload bytes are a simple binary serialization (not cereal) so the
 * on-disk format is self-contained.
 */
class TransactionLog {
 public:
  TransactionLog() = default;
  ~TransactionLog() { Close(); }

  /** Open (or create) the WAL file in append mode. */
  void Open(const std::string &file_path, clio::run::u64 capacity_bytes) {
    file_path_ = file_path;
    capacity_bytes_ = capacity_bytes;
    buffer_.reserve(4096);
    ofs_.open(file_path_,
              std::ios::binary | std::ios::app);
  }

  // ---- Log helpers for each transaction type ----

  void Log(TxnType type, const TxnCreateNewBlob &txn) {
    std::lock_guard<std::mutex> txn_lk(mtx_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.blob_name_);
    WriteFloat(buffer_, txn.score_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnExtendBlob &txn) {
    std::lock_guard<std::mutex> txn_lk(mtx_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.blob_name_);
    WriteU32(buffer_, static_cast<clio::run::u32>(txn.new_blocks_.size()));
    for (const auto &blk : txn.new_blocks_) {
      WriteU32(buffer_, blk.bdev_major_);
      WriteU32(buffer_, blk.bdev_minor_);
      WriteRaw(buffer_, &blk.target_query_, sizeof(clio::run::PoolQuery));
      WriteU64(buffer_, blk.target_offset_);
      WriteU64(buffer_, blk.size_);
    }
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnExtendReplica &txn) {
    std::lock_guard<std::mutex> txn_lk(mtx_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.blob_name_);
    WriteU32(buffer_, txn.replica_);
    WriteString(buffer_, txn.replica_name_);
    WriteFloat(buffer_, txn.score_);
    WriteU32(buffer_, txn.flags_);
    WriteU32(buffer_, txn.transform_flags_);
    WriteFloat(buffer_, txn.min_score_);
    WriteU32(buffer_, static_cast<clio::run::u32>(txn.new_blocks_.size()));
    for (const auto &blk : txn.new_blocks_) {
      WriteU32(buffer_, blk.bdev_major_);
      WriteU32(buffer_, blk.bdev_minor_);
      WriteRaw(buffer_, &blk.target_query_, sizeof(clio::run::PoolQuery));
      WriteU64(buffer_, blk.target_offset_);
      WriteU64(buffer_, blk.size_);
    }
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnClearBlob &txn) {
    std::lock_guard<std::mutex> txn_lk(mtx_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.blob_name_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnDelBlob &txn) {
    std::lock_guard<std::mutex> txn_lk(mtx_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.blob_name_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnSetBlobTransform &txn) {
    std::lock_guard<std::mutex> txn_lk(mtx_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.blob_name_);
    WriteU32(buffer_, txn.transform_flags_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnCreateTag &txn) {
    std::lock_guard<std::mutex> txn_lk(mtx_);
    buffer_.clear();
    WriteString(buffer_, txn.tag_name_);
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnDelTag &txn) {
    std::lock_guard<std::mutex> txn_lk(mtx_);
    buffer_.clear();
    WriteString(buffer_, txn.tag_name_);
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteRecord(type, buffer_);
  }

  /** Flush pending writes to disk */
  void Sync() {
    if (ofs_.is_open()) {
      ofs_.flush();
    }
  }

  /** Return current on-disk file size */
  clio::run::u64 Size() const {
    namespace fs = std::filesystem;
    if (fs::exists(file_path_)) {
      return static_cast<clio::run::u64>(fs::file_size(file_path_));
    }
    return 0;
  }

  /**
   * Load all entries from the WAL file on disk.
   * Returns a vector of (TxnType, raw payload bytes).
   */
  std::vector<std::pair<TxnType, std::vector<char>>> Load() const {
    std::vector<std::pair<TxnType, std::vector<char>>> entries;
    namespace fs = std::filesystem;
    if (!fs::exists(file_path_)) return entries;

    std::ifstream ifs(file_path_, std::ios::binary);
    if (!ifs.is_open()) return entries;
    const std::uintmax_t file_size = fs::file_size(file_path_);

    while (ifs.peek() != EOF) {
      uint8_t type_byte;
      ifs.read(reinterpret_cast<char *>(&type_byte), sizeof(type_byte));
      if (!ifs.good()) break;

      uint32_t payload_size;
      ifs.read(reinterpret_cast<char *>(&payload_size), sizeof(payload_size));
      if (!ifs.good()) break;

      // A framed size larger than what is left in the file cannot be a real
      // record. Stop instead of sizing a vector from it: an interleaved write
      // once put a blob name where this field belongs, and allocating from
      // that value is an OOM (or a very large pointless allocation) before any
      // parsing even starts. Everything after a bad frame is unusable anyway.
      const std::streampos here = ifs.tellg();
      if (here < 0 || static_cast<std::uintmax_t>(payload_size) >
                          static_cast<std::uintmax_t>(file_size) -
                              static_cast<std::uintmax_t>(here)) {
        break;
      }

      std::vector<char> payload(payload_size);
      ifs.read(payload.data(), payload_size);
      if (!ifs.good() && static_cast<uint32_t>(ifs.gcount()) != payload_size)
        break;

      entries.emplace_back(static_cast<TxnType>(type_byte), std::move(payload));
    }
    return entries;
  }

  /** Truncate the WAL file (called after a full snapshot compaction) */
  void Truncate() {
    if (ofs_.is_open()) {
      ofs_.close();
    }
    // Re-open in truncate mode then re-open in append mode
    ofs_.open(file_path_, std::ios::binary | std::ios::trunc);
    ofs_.close();
    ofs_.open(file_path_, std::ios::binary | std::ios::app);
  }

  /** Sync then close the file handle */
  void Close() {
    if (ofs_.is_open()) {
      ofs_.flush();
      ofs_.close();
    }
  }

  // ---- Static deserialization helpers ----

  static TxnCreateNewBlob DeserializeCreateNewBlob(const std::vector<char> &data) {
    TxnCreateNewBlob txn;
    size_t off = 0;
    txn.tag_major_ = ReadU32(data, off);
    txn.tag_minor_ = ReadU32(data, off);
    txn.blob_name_ = ReadString(data, off);
    txn.score_ = ReadFloat(data, off);
    return txn;
  }

  static TxnExtendBlob DeserializeExtendBlob(const std::vector<char> &data) {
    TxnExtendBlob txn;
    size_t off = 0;
    txn.tag_major_ = ReadU32(data, off);
    txn.tag_minor_ = ReadU32(data, off);
    txn.blob_name_ = ReadString(data, off);
    clio::run::u32 num_blocks = ReadU32(data, off);
    txn.new_blocks_.resize(num_blocks);
    for (clio::run::u32 i = 0; i < num_blocks; ++i) {
      txn.new_blocks_[i].bdev_major_ = ReadU32(data, off);
      txn.new_blocks_[i].bdev_minor_ = ReadU32(data, off);
      ReadRaw(data, off, &txn.new_blocks_[i].target_query_,
              sizeof(clio::run::PoolQuery));
      txn.new_blocks_[i].target_offset_ = ReadU64(data, off);
      txn.new_blocks_[i].size_ = ReadU64(data, off);
    }
    return txn;
  }

  static TxnExtendReplica DeserializeExtendReplica(
      const std::vector<char> &data) {
    TxnExtendReplica txn;
    size_t off = 0;
    txn.tag_major_ = ReadU32(data, off);
    txn.tag_minor_ = ReadU32(data, off);
    txn.blob_name_ = ReadString(data, off);
    txn.replica_ = ReadU32(data, off);
    txn.replica_name_ = ReadString(data, off);
    txn.score_ = ReadFloat(data, off);
    txn.flags_ = ReadU32(data, off);
    txn.transform_flags_ = ReadU32(data, off);
    txn.min_score_ = ReadFloat(data, off);
    clio::run::u32 num_blocks = ReadU32(data, off);
    txn.new_blocks_.resize(num_blocks);
    for (clio::run::u32 i = 0; i < num_blocks; ++i) {
      txn.new_blocks_[i].bdev_major_ = ReadU32(data, off);
      txn.new_blocks_[i].bdev_minor_ = ReadU32(data, off);
      ReadRaw(data, off, &txn.new_blocks_[i].target_query_,
              sizeof(clio::run::PoolQuery));
      txn.new_blocks_[i].target_offset_ = ReadU64(data, off);
      txn.new_blocks_[i].size_ = ReadU64(data, off);
    }
    return txn;
  }

  static TxnSetBlobTransform DeserializeSetBlobTransform(
      const std::vector<char> &data) {
    TxnSetBlobTransform txn;
    size_t off = 0;
    txn.tag_major_ = ReadU32(data, off);
    txn.tag_minor_ = ReadU32(data, off);
    txn.blob_name_ = ReadString(data, off);
    txn.transform_flags_ = ReadU32(data, off);
    return txn;
  }

  static TxnClearBlob DeserializeClearBlob(const std::vector<char> &data) {
    TxnClearBlob txn;
    size_t off = 0;
    txn.tag_major_ = ReadU32(data, off);
    txn.tag_minor_ = ReadU32(data, off);
    txn.blob_name_ = ReadString(data, off);
    return txn;
  }

  static TxnDelBlob DeserializeDelBlob(const std::vector<char> &data) {
    TxnDelBlob txn;
    size_t off = 0;
    txn.tag_major_ = ReadU32(data, off);
    txn.tag_minor_ = ReadU32(data, off);
    txn.blob_name_ = ReadString(data, off);
    return txn;
  }

  static TxnCreateTag DeserializeCreateTag(const std::vector<char> &data) {
    TxnCreateTag txn;
    size_t off = 0;
    txn.tag_name_ = ReadString(data, off);
    txn.tag_major_ = ReadU32(data, off);
    txn.tag_minor_ = ReadU32(data, off);
    return txn;
  }

  static TxnDelTag DeserializeDelTag(const std::vector<char> &data) {
    TxnDelTag txn;
    size_t off = 0;
    txn.tag_name_ = ReadString(data, off);
    txn.tag_major_ = ReadU32(data, off);
    txn.tag_minor_ = ReadU32(data, off);
    return txn;
  }

 private:
  std::string file_path_;
  clio::run::u64 capacity_bytes_ = 0;
  std::ofstream ofs_;
  std::vector<char> buffer_;  // Reusable serialization buffer
  /**
   * Serializes Log(), which mutates buffer_ and ofs_.
   *
   * Log() is called from worker threads, and the per-worker log array is sized
   * ONCE, at container create, from GetTotalWorkerCount(). The scheduler grows
   * the pool afterwards, so the `wid % blob_txn_logs_.size()` at the call site
   * aliases several live workers onto a single log -- an alias that did not
   * exist when the array was sized.
   *
   * Two of them interleaving in Log() emitted records whose framed size did
   * not match their contents. Replay then read a blob name where a length
   * prefix belonged, took the ASCII "V_00" as a length of 808,476,502, and
   * segfaulted -- every restart after that write, including `clio_run restart`.
   */
  mutable std::mutex mtx_;

  /** Write a complete record: [u8 type][u32 size][payload] */
  void WriteRecord(TxnType type, const std::vector<char> &payload) {
    if (!ofs_.is_open()) return;
    uint8_t type_byte = static_cast<uint8_t>(type);
    uint32_t payload_size = static_cast<uint32_t>(payload.size());
    ofs_.write(reinterpret_cast<const char *>(&type_byte), sizeof(type_byte));
    ofs_.write(reinterpret_cast<const char *>(&payload_size),
               sizeof(payload_size));
    ofs_.write(payload.data(), payload_size);
  }

  // ---- Serialization primitives ----
  static void WriteU32(std::vector<char> &buf, clio::run::u32 val) {
    const char *p = reinterpret_cast<const char *>(&val);
    buf.insert(buf.end(), p, p + sizeof(val));
  }
  static void WriteU64(std::vector<char> &buf, clio::run::u64 val) {
    const char *p = reinterpret_cast<const char *>(&val);
    buf.insert(buf.end(), p, p + sizeof(val));
  }
  static void WriteFloat(std::vector<char> &buf, float val) {
    const char *p = reinterpret_cast<const char *>(&val);
    buf.insert(buf.end(), p, p + sizeof(val));
  }
  static void WriteString(std::vector<char> &buf, const std::string &s) {
    WriteU32(buf, static_cast<clio::run::u32>(s.size()));
    buf.insert(buf.end(), s.data(), s.data() + s.size());
  }
  static void WriteRaw(std::vector<char> &buf, const void *ptr, size_t len) {
    const char *p = reinterpret_cast<const char *>(ptr);
    buf.insert(buf.end(), p, p + len);
  }

  // ---- Deserialization primitives ----
  /**
   * Bounds gate for every read below.
   *
   * A WAL is exactly the structure that must tolerate a torn tail: the writer
   * can die mid-record, and a reader that trusts the bytes turns a recoverable
   * truncation into a crash. These primitives previously indexed straight into
   * the payload, so one bad length (see mtx_) was an out-of-bounds read of
   * ~808 MB rather than a rejected record.
   *
   * Throws so the caller can stop replaying THIS file at the first damaged
   * record and keep everything already applied -- the correct WAL response,
   * since nothing after a torn record can be trusted anyway.
   */
  static void Need(const std::vector<char> &data, size_t off, size_t n) {
    if (off > data.size() || n > data.size() - off) {
      throw std::runtime_error(
          "TransactionLog: truncated record (need " + std::to_string(n) +
          " bytes at offset " + std::to_string(off) + " of " +
          std::to_string(data.size()) + ")");
    }
  }
  static clio::run::u32 ReadU32(const std::vector<char> &data, size_t &off) {
    clio::run::u32 val;
    Need(data, off, sizeof(val));
    std::memcpy(&val, data.data() + off, sizeof(val));
    off += sizeof(val);
    return val;
  }
  static clio::run::u64 ReadU64(const std::vector<char> &data, size_t &off) {
    clio::run::u64 val;
    Need(data, off, sizeof(val));
    std::memcpy(&val, data.data() + off, sizeof(val));
    off += sizeof(val);
    return val;
  }
  static float ReadFloat(const std::vector<char> &data, size_t &off) {
    float val;
    Need(data, off, sizeof(val));
    std::memcpy(&val, data.data() + off, sizeof(val));
    off += sizeof(val);
    return val;
  }
  static std::string ReadString(const std::vector<char> &data, size_t &off) {
    clio::run::u32 len = ReadU32(data, off);
    Need(data, off, len);
    std::string s(data.data() + off, len);
    off += len;
    return s;
  }
  static void ReadRaw(const std::vector<char> &data, size_t &off, void *ptr,
                      size_t len) {
    Need(data, off, len);
    std::memcpy(ptr, data.data() + off, len);
    off += len;
  }
};

}  // namespace clio::cte::core

#endif  // WRPCTE_CORE_TRANSACTION_LOG_H_
