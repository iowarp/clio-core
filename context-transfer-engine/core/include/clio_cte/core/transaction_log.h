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

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
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
  /**
   * Blob is an expendable cache copy (kCtePutDroppable).
   *
   * Separate from kCreateNewBlob because droppability is classified under the
   * per-blob write token, which is acquired after the create record is
   * written. Without this record, replay restores every blob as authoritative,
   * and since droppability is write-once it can never be marked again.
   *
   * Appending a type is safe for an older runtime: replay ignores types it
   * does not recognise.
   */
  kSetBlobDroppable = 8,
  /**
   * Full replacement of a tag's name identity: its canonical name plus every
   * alias (hard link) bound to it (issue: renames/hard links lost on crash).
   *
   * RenameTag and GetOrCreateTagAlias/DelTag's alias-unlink and
   * promote-alias-to-canonical paths mutated TagInfo::tag_name_/aliases_ and
   * tag_name_to_id_ purely in memory, with no WAL record at all -- only the
   * ORIGINAL kCreateTag binding survived a restart, so a rename or hard link
   * made after the last FlushMetadata snapshot (and before a crash) vanished.
   * A full-identity record (not a delta) after ANY of those mutations is
   * simplest to replay correctly regardless of which shard/seq order the
   * records land in -- see ApplyWalSetTagIdentity.
   */
  kSetTagIdentity = 9,
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
struct TxnSetBlobDroppable {
  clio::run::u32 tag_major_;
  clio::run::u32 tag_minor_;
  std::string blob_name_;
  clio::run::u32 droppable_;
};

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

/** Payload: delete a tag. `aliases_` is the FULL alias list at the moment of
 *  delete, captured so replay can drop every alias's tag_name_to_id_
 *  binding, not just the canonical name -- otherwise a hard-linked tag
 *  deleted before a crash left its alias names dangling after restart. */
struct TxnDelTag {
  std::string tag_name_;
  clio::run::u32 tag_major_;
  clio::run::u32 tag_minor_;
  std::vector<std::string> aliases_;
};

/** Payload: full replacement of a tag's name identity (issue: renames/hard
 *  links lost on crash). See TxnType::kSetTagIdentity. */
struct TxnSetTagIdentity {
  clio::run::u32 tag_major_;
  clio::run::u32 tag_minor_;
  std::string canonical_name_;
  std::vector<std::string> aliases_;
};

/**
 * One record read back from a WAL shard by TransactionLog::Load(): the
 * transaction type, its GLOBAL sequence number (see seq_counter_ below), and
 * the raw (type-specific) payload bytes to hand to the matching
 * Deserialize*() helper.
 */
struct WalRecord {
  TxnType type_;
  clio::run::u64 seq_;
  std::vector<char> payload_;
};

/**
 * Header-only Write-Ahead Transaction Log.
 *
 * Record format on disk:
 *   [u8 txn_type][u64 seq][u32 payload_size][payload bytes]
 *
 * The payload bytes are a simple binary serialization (not cereal) so the
 * on-disk format is self-contained.
 *
 * Every shard file opens with a 4-byte format magic (kWalMagic) so a reader
 * can tell this (post-sequence-number) format apart from the pre-existing
 * one that had no magic and no per-record seq. This branch is new (no
 * deployed WAL predates it), so an old-format file is simply rejected with a
 * log line rather than parsed -- see Load().
 */
class TransactionLog {
 public:
  /** Format marker written as the first 4 bytes of every WAL shard file
   *  created by this (sequence-numbered) record format. */
  static constexpr uint32_t kWalMagic = 0x57414C32;  // "WAL2"

  TransactionLog() = default;
  ~TransactionLog() { Close(); }

  /** Open (or create) the WAL file in append mode. A brand-new (empty or
   *  absent) file gets the format magic written first; an existing file is
   *  left exactly as it is -- appending must never rewrite bytes a
   *  not-yet-replayed reader may still need. */
  void Open(const std::string &file_path, clio::run::u64 capacity_bytes) {
    file_path_ = file_path;
    capacity_bytes_ = capacity_bytes;
    buffer_.reserve(4096);
    namespace fs = std::filesystem;
    const bool is_new = !fs::exists(file_path_) ||
                        fs::file_size(file_path_) == 0;
    ofs_.open(file_path_, std::ios::binary | std::ios::app);
    if (is_new && ofs_.is_open()) {
      WriteMagic();
    }
  }

  /**
   * Global sequence-number source for records this log writes (issue: WAL
   * cross-shard replay ordering). Records are sharded per-worker
   * (blob_txn_logs_[wid % N] / tag_txn_logs_[wid % N]), so two records for
   * the SAME blob can land in different shard files; ReplayTransactionLogs
   * used to apply shards strictly in file-index order, which is not the
   * same as the order the records were actually written in and could apply
   * a full-block-replacement record (kExtendBlob/kExtendReplica) OUT of
   * order, regressing a blob to an earlier, shorter layout. Every shard of
   * one core container now draws its per-record seq from the SAME atomic
   * counter (owned by Runtime, one per container), so merging every
   * shard's records by seq at replay recovers true write order regardless
   * of which shard each record landed in. Not owned: the counter outlives
   * this object for the container's whole lifetime.
   */
  void SetSeqCounter(std::atomic<clio::run::u64> *seq_counter) {
    seq_counter_ = seq_counter;
  }

  // ---- Log helpers for each transaction type ----

  void Log(TxnType type, const TxnSetBlobDroppable &txn) {
    std::lock_guard<std::mutex> lk(mu_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.blob_name_);
    WriteU32(buffer_, txn.droppable_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnCreateNewBlob &txn) {
    std::lock_guard<std::mutex> lk(mu_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.blob_name_);
    WriteFloat(buffer_, txn.score_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnExtendBlob &txn) {
    std::lock_guard<std::mutex> lk(mu_);
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
    std::lock_guard<std::mutex> lk(mu_);
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
    std::lock_guard<std::mutex> lk(mu_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.blob_name_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnDelBlob &txn) {
    std::lock_guard<std::mutex> lk(mu_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.blob_name_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnSetBlobTransform &txn) {
    std::lock_guard<std::mutex> lk(mu_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.blob_name_);
    WriteU32(buffer_, txn.transform_flags_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnCreateTag &txn) {
    std::lock_guard<std::mutex> lk(mu_);
    buffer_.clear();
    WriteString(buffer_, txn.tag_name_);
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnDelTag &txn) {
    std::lock_guard<std::mutex> lk(mu_);
    buffer_.clear();
    WriteString(buffer_, txn.tag_name_);
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteStringVector(buffer_, txn.aliases_);
    WriteRecord(type, buffer_);
  }

  void Log(TxnType type, const TxnSetTagIdentity &txn) {
    std::lock_guard<std::mutex> lk(mu_);
    buffer_.clear();
    WriteU32(buffer_, txn.tag_major_);
    WriteU32(buffer_, txn.tag_minor_);
    WriteString(buffer_, txn.canonical_name_);
    WriteStringVector(buffer_, txn.aliases_);
    WriteRecord(type, buffer_);
  }

  /** Flush pending writes to disk */
  void Sync() {
    std::lock_guard<std::mutex> lk(mu_);
    if (ofs_.is_open()) {
      ofs_.flush();
      FsyncPath(file_path_);
    }
  }

  /**
   * fsync a file by path (works on any open of the inode, so a read-only
   * descriptor suffices).
   * @param path file to sync
   * @return true on success
   */
  static bool FsyncPath(const std::string &path) {
#ifdef _WIN32
    // _commit (FlushFileBuffers) needs a writable handle.
    const int fd = ::_open(path.c_str(), _O_RDWR | _O_BINARY);
    if (fd < 0) return false;
    const bool ok = ::_commit(fd) == 0;
    ::_close(fd);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
#endif
    return ok;
  }

  /**
   * fsync the directory holding `path`, making a create or rename of it
   * durable.
   * @param path file whose parent directory is synced
   * @return true on success
   */
  static bool FsyncParentDir(const std::string &path) {
#ifdef _WIN32
    // A directory cannot be opened or flushed through the CRT; NTFS journals
    // the rename itself.
    (void)path;
    return true;
#else
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) parent = ".";
    return FsyncPath(parent.string());
#endif
  }

  /**
   * Durably replace `path` with `tmp_path`: fsync the new file, rename it
   * over the old one, fsync the directory. A crash at any point leaves
   * either the complete old file or the complete new one.
   * @param tmp_path fully written replacement
   * @param path destination
   * @return true on success
   */
  static bool DurableReplace(const std::string &tmp_path,
                             const std::string &path) {
    if (!FsyncPath(tmp_path)) return false;
    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) return false;
    return FsyncParentDir(path);
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
   * Load every record from the WAL file on disk, each carrying the GLOBAL
   * seq it was written with (see SetSeqCounter). A file that does not open
   * with kWalMagic predates the sequence-numbered format (or is corrupt);
   * it is not parseable as this format, so it is rejected wholesale with a
   * log line rather than guessed at record-by-record -- this branch is new,
   * so no deployed WAL is expected to be in the old, magic-less format.
   */
  std::vector<WalRecord> Load() const {
    std::vector<WalRecord> entries;
    namespace fs = std::filesystem;
    if (!fs::exists(file_path_)) return entries;

    std::ifstream ifs(file_path_, std::ios::binary);
    if (!ifs.is_open()) return entries;

    uint32_t magic = 0;
    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (!ifs.good() || magic != kWalMagic) {
      HLOG(kWarning,
           "TransactionLog::Load: '{}' is not a sequence-numbered WAL file "
           "(missing/mismatched format magic) -- ignoring it. Old-format "
           "WALs from before this change are not replayable; wipe the "
           "metadata_log_path to start a fresh log.",
           file_path_);
      return entries;
    }

    while (ifs.peek() != EOF) {
      uint8_t type_byte;
      ifs.read(reinterpret_cast<char *>(&type_byte), sizeof(type_byte));
      if (!ifs.good()) break;

      clio::run::u64 seq = 0;
      ifs.read(reinterpret_cast<char *>(&seq), sizeof(seq));
      if (!ifs.good()) break;

      uint32_t payload_size;
      ifs.read(reinterpret_cast<char *>(&payload_size), sizeof(payload_size));
      if (!ifs.good()) break;

      std::vector<char> payload(payload_size);
      ifs.read(payload.data(), payload_size);
      if (!ifs.good() && static_cast<uint32_t>(ifs.gcount()) != payload_size)
        break;

      entries.push_back(
          WalRecord{static_cast<TxnType>(type_byte), seq, std::move(payload)});
    }
    return entries;
  }

  /** Truncate the WAL file (called after a full snapshot compaction). The
   *  freshly-truncated file gets the format magic rewritten immediately, so
   *  a crash right after truncation still leaves a well-formed (if empty)
   *  file rather than one Load() would reject. */
  void Truncate() { TruncateThrough(~clio::run::u64(0)); }

  /**
   * Drop every record whose global seq is <= `seq` -- the ones a snapshot
   * taken after `seq` was issued already contains -- and keep the rest. A
   * record logged while the snapshot was being built (seq > `seq`) may
   * describe state the snapshot never saw, so dropping it would lose that
   * update on the next restart. The shard is rewritten to a temp file and
   * durably renamed over the original.
   * @param seq highest seq the snapshot covers
   */
  void TruncateThrough(clio::run::u64 seq) {
    std::lock_guard<std::mutex> lk(mu_);
    if (ofs_.is_open()) {
      ofs_.flush();
      ofs_.close();
    }
    std::vector<WalRecord> keep;
    for (auto &rec : Load()) {
      if (rec.seq_ > seq) keep.push_back(std::move(rec));
    }
    const std::string tmp = file_path_ + ".tmp";
    {
      std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
      uint32_t magic = kWalMagic;
      out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
      for (const auto &rec : keep) {
        WriteRecordTo(out, rec.type_, rec.seq_, rec.payload_);
      }
    }
    if (!DurableReplace(tmp, file_path_)) {
      HLOG(kError, "TransactionLog: could not rewrite {}; keeping it whole",
           file_path_);
    }
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

  static TxnSetBlobDroppable DeserializeSetBlobDroppable(
      const std::vector<char> &data) {
    TxnSetBlobDroppable txn;
    size_t off = 0;
    txn.tag_major_ = ReadU32(data, off);
    txn.tag_minor_ = ReadU32(data, off);
    txn.blob_name_ = ReadString(data, off);
    txn.droppable_ = ReadU32(data, off);
    return txn;
  }

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
    // Older (pre-alias-tracking) records end here: no aliases_ to read.
    if (off < data.size()) {
      txn.aliases_ = ReadStringVector(data, off);
    }
    return txn;
  }

  static TxnSetTagIdentity DeserializeSetTagIdentity(
      const std::vector<char> &data) {
    TxnSetTagIdentity txn;
    size_t off = 0;
    txn.tag_major_ = ReadU32(data, off);
    txn.tag_minor_ = ReadU32(data, off);
    txn.canonical_name_ = ReadString(data, off);
    txn.aliases_ = ReadStringVector(data, off);
    return txn;
  }

 private:
  std::string file_path_;
  clio::run::u64 capacity_bytes_ = 0;
  std::ofstream ofs_;
  std::vector<char> buffer_;  // Reusable serialization buffer
  /**
   * One log is shared by every worker whose id maps onto it: the runtime
   * shards logs by `worker_id % num_logs`, and elastic workers (ids past the
   * configured count) alias the base workers' logs. Two threads then raced on
   * buffer_ and ofs_ -- heap corruption that surfaced as zeroed regions in a
   * blob written by 8 threads with the metadata log on (RELIABILITY.md
   * defect 14). Log() never suspends, so a plain mutex is correct here.
   */
  std::mutex mu_;

  /** Shared global sequence-number source (see SetSeqCounter). Null means
   *  "not wired up yet" (e.g. a standalone TransactionLog used in a unit
   *  test), in which case every record logs seq 0 -- replay still works
   *  within a single shard (insertion order == seq order there), it just
   *  cannot be merged across shards. */
  std::atomic<clio::run::u64> *seq_counter_ = nullptr;

  /** Next global seq for a record this log is about to write. */
  clio::run::u64 NextSeq() {
    if (seq_counter_ == nullptr) return 0;
    return seq_counter_->fetch_add(1, std::memory_order_relaxed) + 1;
  }

  /** Write the format magic as the first bytes of the (empty) file. */
  void WriteMagic() {
    uint32_t magic = kWalMagic;
    ofs_.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  }

  /**
   * Write a complete record: [u8 type][u64 seq][u32 size][payload], then
   * flush() the C++ stream buffer to the OS immediately.
   *
   * Without this, a record sits in the ofstream's userspace buffer (libstdc++
   * default ~a few KB) until it fills, something else calls Sync(), or the
   * file is closed -- none of which a SIGKILL waits for. The periodic
   * FlushMetadata task only Sync()s every flush_metadata_period_ms_ (default
   * 5000 ms), so a crash between periods could lose every WAL record written
   * since the last one, however long ago that data was actually acked to the
   * client via fsync(). flush() pushes the buffer out via write(2); the
   * bytes then survive THIS process dying (they are in the kernel page
   * cache), which is the durability contract callers of fsync()/close() are
   * relying on. It is not an fsync(2) -- surviving raw power loss still
   * needs Sync()/Close(), which already run at the existing checkpoints.
   */
  void WriteRecord(TxnType type, const std::vector<char> &payload) {
    if (!ofs_.is_open()) return;
    WriteRecordTo(ofs_, type, NextSeq(), payload);
    ofs_.flush();
  }

  /** Serialize one record [u8 type][u64 seq][u32 size][payload] to `out`. */
  static void WriteRecordTo(std::ofstream &out, TxnType type,
                            clio::run::u64 seq,
                            const std::vector<char> &payload) {
    uint8_t type_byte = static_cast<uint8_t>(type);
    uint32_t payload_size = static_cast<uint32_t>(payload.size());
    out.write(reinterpret_cast<const char *>(&type_byte), sizeof(type_byte));
    out.write(reinterpret_cast<const char *>(&seq), sizeof(seq));
    out.write(reinterpret_cast<const char *>(&payload_size),
              sizeof(payload_size));
    out.write(payload.data(), payload_size);
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
  /** [u32 count][WriteString each] -- used for TagInfo::aliases_. */
  static void WriteStringVector(std::vector<char> &buf,
                                const std::vector<std::string> &v) {
    WriteU32(buf, static_cast<clio::run::u32>(v.size()));
    for (const auto &s : v) {
      WriteString(buf, s);
    }
  }

  // ---- Deserialization primitives ----
  static clio::run::u32 ReadU32(const std::vector<char> &data, size_t &off) {
    clio::run::u32 val;
    std::memcpy(&val, data.data() + off, sizeof(val));
    off += sizeof(val);
    return val;
  }
  static clio::run::u64 ReadU64(const std::vector<char> &data, size_t &off) {
    clio::run::u64 val;
    std::memcpy(&val, data.data() + off, sizeof(val));
    off += sizeof(val);
    return val;
  }
  static float ReadFloat(const std::vector<char> &data, size_t &off) {
    float val;
    std::memcpy(&val, data.data() + off, sizeof(val));
    off += sizeof(val);
    return val;
  }
  static std::string ReadString(const std::vector<char> &data, size_t &off) {
    clio::run::u32 len = ReadU32(data, off);
    std::string s(data.data() + off, len);
    off += len;
    return s;
  }
  static std::vector<std::string> ReadStringVector(
      const std::vector<char> &data, size_t &off) {
    clio::run::u32 count = ReadU32(data, off);
    std::vector<std::string> v;
    v.reserve(count);
    for (clio::run::u32 i = 0; i < count; ++i) {
      v.push_back(ReadString(data, off));
    }
    return v;
  }
  static void ReadRaw(const std::vector<char> &data, size_t &off, void *ptr,
                      size_t len) {
    std::memcpy(ptr, data.data() + off, len);
    off += len;
  }
};

}  // namespace clio::cte::core

#endif  // WRPCTE_CORE_TRANSACTION_LOG_H_
