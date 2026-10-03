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

#ifndef SAFE_BDEV_JOURNAL_H_
#define SAFE_BDEV_JOURNAL_H_

#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace clio::run::safe_bdev {

/**
 * Degraded-write journal (#1137): closes the RAID write hole while a data
 * member is down.
 *
 * With a data column down, its chunk in a stripe exists only through the
 * stripe's parity. A write to that stripe changes live members and then the
 * parity; a crash between the two leaves parity that matches neither
 * version, and the down column's chunk -- fsynced long before -- can no
 * longer be decoded. Before such a write lands anything, the array records
 * the down column's chunk here (reconstructed while the parity was still
 * consistent). After a crash, the stripe is re-encoded from its live
 * members plus the journaled chunk, and the down column survives.
 *
 * A record is valid only while the write's stripe intent (kIntentGroup key)
 * is live: an intent cleared means the stripe's parity became current again
 * and the record is superseded. Open() drops records whose key is no longer
 * live. Records are appended to one file; it is compacted when it grows
 * past a limit.
 */
class StripeJournal {
 public:
  StripeJournal() = default;
  ~StripeJournal() { Close(); }
  StripeJournal(const StripeJournal &) = delete;
  StripeJournal &operator=(const StripeJournal &) = delete;

  /**
   * Open (creating if absent) the journal file and load the records that are
   * still valid.
   * @param path journal file; empty disables the journal
   * @param chunk_len bytes per record payload (the array's chunk length)
   * @param live_keys intent keys still live in the intent log
   * @return true on success (or when disabled)
   */
  bool Open(const std::string &path, uint64_t chunk_len,
            const std::set<uint64_t> &live_keys);

  /** Close the file. */
  void Close();

  /** @return true when a journal file is open. */
  bool enabled() const { return fd_ >= 0; }

  /**
   * Record a down column's chunk before a degraded write to its stripe lands.
   * Reaches the kernel before returning (survives the process dying); call
   * Sync() for power-loss durability.
   * @param key the write's intent key for this stripe
   * @param slot the stripe
   * @param col the down data column
   * @param data chunk_len bytes: the column's current chunk
   * @return true when the record was written
   */
  bool Append(uint64_t key, uint64_t slot, uint32_t col, const uint8_t *data);

  /** fdatasync the journal. @return true on success */
  bool Sync();

  /**
   * Read the newest valid record of (slot, col).
   * @param slot the stripe
   * @param col the data column
   * @param out receives chunk_len bytes
   * @return true when a record exists and was read intact
   */
  bool Read(uint64_t slot, uint32_t col, std::vector<uint8_t> &out);

  /** Forget every record of a stripe (its parity is current again; the
   *  caller holds the stripe). */
  void DropSlot(uint64_t slot);

  /** Forget a stripe's records written under one intent key (that write
   *  brought the stripe's parity current). */
  void DropKey(uint64_t slot, uint64_t key);

  /** @return the number of stripes with records. */
  size_t NumSlots();

 private:
  /** On-disk record header; chunk_len payload bytes follow. */
  struct RecordHeader {
    uint64_t magic;
    uint64_t key;
    uint64_t slot;
    uint32_t col;
    uint32_t pad;
    uint64_t sum;  // checksum of the fields above and the payload
  };
  /** Where a record sits in the file. */
  struct Loc {
    uint64_t key;
    uint64_t off;
  };
  static constexpr uint64_t kMagic = 0x534a524e4c4f4731ULL;  // "SJRNLOG1"
  // Compact once the file passes this and its live records are fewer.
  static constexpr uint64_t kCompactBytes = 256ULL << 20;

  /**
   * Checksum a record.
   * @param h the header (sum ignored)
   * @param data the payload
   * @param len payload bytes
   * @return the checksum
   */
  static uint64_t Checksum(const RecordHeader &h, const uint8_t *data,
                           uint64_t len);
  /** Scan the file, keeping valid records whose key is live. */
  void Load(const std::set<uint64_t> &live_keys);
  /** Rewrite the file with only the live records (mu_ held). @return ok */
  bool CompactLocked();
  /** Write one record at `off` (mu_ held). @return ok */
  bool WriteAt(uint64_t off, uint64_t key, uint64_t slot, uint32_t col,
               const uint8_t *data);

  std::mutex mu_;
  int fd_ = -1;
  std::string path_;
  uint64_t chunk_len_ = 0;
  uint64_t end_ = 0;  // append offset
  // slot -> column -> newest record
  std::map<uint64_t, std::map<uint32_t, Loc>> recs_;
};

}  // namespace clio::run::safe_bdev

#endif  // SAFE_BDEV_JOURNAL_H_
