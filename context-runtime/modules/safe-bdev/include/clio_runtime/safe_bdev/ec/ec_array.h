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

#ifndef SAFE_BDEV_EC_EC_ARRAY_H_
#define SAFE_BDEV_EC_EC_ARRAY_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "reed_solomon.h"

/**
 * EcArray — daemon-independent striping + recovery layer.
 *
 * An EcArray maps a fixed set of `k` data members and 0..m_max parity members
 * onto a Reed-Solomon code. Each member is backed by a MemberStore (an abstract
 * byte device — backed by RAM for tests, by a bdev pool in the runtime). The
 * array is sliced into fixed-size stripes; stripe `s` occupies bytes
 * [s*shard_len, (s+1)*shard_len) on every member.
 *
 * It implements the operations that back safe_bdev's management API:
 *   - AddParityDrive  : raise fault tolerance by one, computing a single new
 *                       parity row across existing data (incremental, no rewrite
 *                       of data or existing parity).
 *   - MarkFaulty / RemoveDrive : take a member out of service.
 *   - RecoverMember   : reconstruct a failed member's shards onto a fresh store.
 *   - ReadStripeData  : reconstruct-on-read when a data member is unavailable.
 *
 * This layer is intentionally free of any runtime/SHM dependency so it can be
 * unit-tested directly.
 */

namespace clio::run::safe_bdev::ec {

/** Abstract byte-addressable member device. */
class MemberStore {
 public:
  virtual ~MemberStore() = default;
  /** Read `len` bytes at `offset` into `out`. @return true on success. */
  virtual bool Read(uint64_t offset, uint64_t len, uint8_t *out) = 0;
  /** Write `len` bytes from `in` at `offset`. @return true on success. */
  virtual bool Write(uint64_t offset, uint64_t len, const uint8_t *in) = 0;
  /** Total capacity in bytes. */
  virtual uint64_t Capacity() const = 0;
};

/** Simple in-memory MemberStore for tests and small buffers. */
class MemoryStore : public MemberStore {
 public:
  explicit MemoryStore(uint64_t capacity) : buf_(capacity, 0) {}

  bool Read(uint64_t offset, uint64_t len, uint8_t *out) override {
    if (offset + len > buf_.size()) {
      return false;
    }
    std::memcpy(out, buf_.data() + offset, len);
    return true;
  }
  bool Write(uint64_t offset, uint64_t len, const uint8_t *in) override {
    if (offset + len > buf_.size()) {
      return false;
    }
    std::memcpy(buf_.data() + offset, in, len);
    return true;
  }
  uint64_t Capacity() const override { return buf_.size(); }

  /** Direct access (test convenience). */
  const std::vector<uint8_t> &bytes() const { return buf_; }

 private:
  std::vector<uint8_t> buf_;
};

/** Member role within the array. */
enum class EcRole : uint32_t { kData = 0, kParity = 1 };

/** Member availability. */
enum class EcState : uint32_t { kActive = 0, kFaulty = 1, kRemoved = 2 };

/** One member of the array. */
struct EcMember {
  MemberStore *store = nullptr;  // non-owning
  EcRole role = EcRole::kData;
  EcState state = EcState::kActive;
  int index = -1;  // data column (role==kData) or parity row (role==kParity)
};

class EcArray {
 public:
  /**
   * @param shard_len    stripe block size per member, in bytes
   * @param num_stripes  number of stripes (capacity = num_stripes*shard_len)
   * @param k            number of data members
   * @param m_max        maximum number of parity members (== max failures)
   * @param data_stores  exactly k data member stores
   */
  EcArray(size_t shard_len, size_t num_stripes, int k, int m_max,
          const std::vector<MemberStore *> &data_stores)
      : shard_len_(shard_len),
        num_stripes_(num_stripes),
        k_(k),
        m_max_(m_max),
        rs_(k, m_max),
        parity_level_(0) {
    for (int c = 0; c < k_; ++c) {
      EcMember mem;
      mem.store = data_stores[static_cast<size_t>(c)];
      mem.role = EcRole::kData;
      mem.state = EcState::kActive;
      mem.index = c;
      members_.push_back(mem);
    }
  }

  size_t shard_len() const { return shard_len_; }
  size_t num_stripes() const { return num_stripes_; }
  int k() const { return k_; }
  int m_max() const { return m_max_; }
  int parity_level() const { return parity_level_; }
  size_t num_members() const { return members_.size(); }
  const EcMember &member(size_t i) const { return members_[i]; }

  /** Index of the data member for column `c` (always c by construction). */
  int DataMemberIndex(int c) const { return c; }
  /** Index of the parity member for row `r` (k + r). */
  int ParityMemberIndex(int r) const { return k_ + r; }

  /**
   * Write one stripe: store the k data shards on the data members and (re)compute
   * parity for every active parity row. `data_shards` must hold k buffers of
   * shard_len bytes each.
   */
  bool WriteStripe(size_t stripe,
                   const std::vector<std::vector<uint8_t>> &data_shards) {
    if (stripe >= num_stripes_ ||
        static_cast<int>(data_shards.size()) != k_) {
      return false;
    }
    const uint64_t off = static_cast<uint64_t>(stripe) * shard_len_;
    for (int c = 0; c < k_; ++c) {
      EcMember &m = members_[static_cast<size_t>(c)];
      if (m.state == EcState::kActive &&
          !m.store->Write(off, shard_len_, data_shards[c].data())) {
        return false;
      }
    }
    return WriteParityForStripe(stripe, data_shards);
  }

  /**
   * Read the k data shards of a stripe, reconstructing any whose data member is
   * not active. Fails if too many members are down to reconstruct.
   */
  bool ReadStripeData(size_t stripe,
                      std::vector<std::vector<uint8_t>> *out) {
    if (stripe >= num_stripes_) {
      return false;
    }
    bool all_data_active = true;
    for (int c = 0; c < k_; ++c) {
      if (members_[static_cast<size_t>(c)].state != EcState::kActive) {
        all_data_active = false;
        break;
      }
    }
    if (all_data_active) {
      const uint64_t off = static_cast<uint64_t>(stripe) * shard_len_;
      out->assign(static_cast<size_t>(k_),
                  std::vector<uint8_t>(shard_len_, 0));
      for (int c = 0; c < k_; ++c) {
        if (!members_[static_cast<size_t>(c)].store->Read(
                off, shard_len_, (*out)[c].data())) {
          return false;
        }
      }
      return true;
    }
    return ReconstructStripeData(stripe, out);
  }

  /** Read a single data shard (column `c`), reconstructing if needed. */
  bool ReadDataShard(size_t stripe, int c, std::vector<uint8_t> *out) {
    EcMember &m = members_[static_cast<size_t>(c)];
    out->assign(shard_len_, 0);
    if (m.state == EcState::kActive) {
      const uint64_t off = static_cast<uint64_t>(stripe) * shard_len_;
      return m.store->Read(off, shard_len_, out->data());
    }
    std::vector<std::vector<uint8_t>> data;
    if (!ReconstructStripeData(stripe, &data)) {
      return false;
    }
    *out = data[static_cast<size_t>(c)];
    return true;
  }

  /**
   * Add a parity drive, raising fault tolerance by one. Computes the single new
   * Cauchy parity row across the existing data for every stripe — existing data
   * shards and existing parity rows are not touched. Caps at m_max.
   *
   * @return the new parity level, or -1 on failure / cap reached.
   */
  int AddParityDrive(MemberStore *store) {
    if (parity_level_ >= m_max_) {
      return -1;
    }
    const int r = parity_level_;
    EcMember mem;
    mem.store = store;
    mem.role = EcRole::kParity;
    mem.state = EcState::kActive;
    mem.index = r;
    members_.push_back(mem);
    // Compute parity row r for every stripe from the current data.
    for (size_t s = 0; s < num_stripes_; ++s) {
      std::vector<std::vector<uint8_t>> data;
      if (!ReadStripeData(s, &data)) {
        members_.pop_back();
        return -1;
      }
      std::vector<const uint8_t *> ptrs(static_cast<size_t>(k_));
      for (int c = 0; c < k_; ++c) {
        ptrs[static_cast<size_t>(c)] = data[static_cast<size_t>(c)].data();
      }
      std::vector<uint8_t> parity(shard_len_, 0);
      rs_.EncodeParityShard(r, ptrs, shard_len_, parity.data());
      const uint64_t off = static_cast<uint64_t>(s) * shard_len_;
      if (!store->Write(off, shard_len_, parity.data())) {
        members_.pop_back();
        return -1;
      }
    }
    ++parity_level_;
    return parity_level_;
  }

  /** Mark a member faulty (kept as a recovery candidate, excluded from I/O). */
  void MarkFaulty(size_t member_index) {
    members_[member_index].state = EcState::kFaulty;
  }

  /**
   * Remove a member. If `was_faulty`, the member is marked faulty (a later
   * RecoverMember can reconstruct it); otherwise it is unlinked. Neither path
   * migrates data (clean removal assumes the member's data is reconstructable
   * from the remaining redundancy or is being decommissioned).
   */
  void RemoveDrive(size_t member_index, bool was_faulty) {
    members_[member_index].state =
        was_faulty ? EcState::kFaulty : EcState::kRemoved;
  }

  /**
   * Reconstruct a failed member's shards onto `new_store` and bring it back
   * online. Works for both data and parity members.
   * @return true on success.
   */
  bool RecoverMember(size_t member_index, MemberStore *new_store) {
    EcMember &m = members_[member_index];
    for (size_t s = 0; s < num_stripes_; ++s) {
      std::vector<std::vector<uint8_t>> data;
      if (!ReconstructStripeData(s, &data, /*exclude=*/static_cast<int>(
                                              member_index))) {
        return false;
      }
      std::vector<uint8_t> shard(shard_len_, 0);
      if (m.role == EcRole::kData) {
        shard = data[static_cast<size_t>(m.index)];
      } else {
        std::vector<const uint8_t *> ptrs(static_cast<size_t>(k_));
        for (int c = 0; c < k_; ++c) {
          ptrs[static_cast<size_t>(c)] = data[static_cast<size_t>(c)].data();
        }
        rs_.EncodeParityShard(m.index, ptrs, shard_len_, shard.data());
      }
      const uint64_t off = static_cast<uint64_t>(s) * shard_len_;
      if (!new_store->Write(off, shard_len_, shard.data())) {
        return false;
      }
    }
    m.store = new_store;
    m.state = EcState::kActive;
    return true;
  }

 private:
  /** Compute + write all active parity rows for a stripe given its data. */
  bool WriteParityForStripe(
      size_t stripe, const std::vector<std::vector<uint8_t>> &data_shards) {
    if (parity_level_ == 0) {
      return true;
    }
    std::vector<const uint8_t *> ptrs(static_cast<size_t>(k_));
    for (int c = 0; c < k_; ++c) {
      ptrs[static_cast<size_t>(c)] = data_shards[static_cast<size_t>(c)].data();
    }
    const uint64_t off = static_cast<uint64_t>(stripe) * shard_len_;
    for (int r = 0; r < parity_level_; ++r) {
      EcMember &pm = members_[static_cast<size_t>(ParityMemberIndex(r))];
      if (pm.state != EcState::kActive) {
        continue;
      }
      std::vector<uint8_t> parity(shard_len_, 0);
      rs_.EncodeParityShard(r, ptrs, shard_len_, parity.data());
      if (!pm.store->Write(off, shard_len_, parity.data())) {
        return false;
      }
    }
    return true;
  }

  /**
   * Reconstruct all k data shards of a stripe from the active members,
   * optionally excluding one member index (used during its own recovery).
   */
  bool ReconstructStripeData(size_t stripe,
                             std::vector<std::vector<uint8_t>> *out,
                             int exclude = -1) {
    const uint64_t off = static_cast<uint64_t>(stripe) * shard_len_;
    std::vector<int> survivor_index;
    std::vector<std::vector<uint8_t>> survivor_buf;
    for (size_t i = 0; i < members_.size(); ++i) {
      if (static_cast<int>(i) == exclude) {
        continue;
      }
      const EcMember &m = members_[i];
      if (m.state != EcState::kActive) {
        continue;
      }
      const int global = (m.role == EcRole::kData) ? m.index : (k_ + m.index);
      std::vector<uint8_t> buf(shard_len_, 0);
      if (!m.store->Read(off, shard_len_, buf.data())) {
        return false;
      }
      survivor_index.push_back(global);
      survivor_buf.push_back(std::move(buf));
      if (static_cast<int>(survivor_index.size()) == k_) {
        break;  // k survivors is enough
      }
    }
    if (static_cast<int>(survivor_index.size()) < k_) {
      return false;  // too many failures to reconstruct
    }
    std::vector<const uint8_t *> ptrs(survivor_buf.size());
    for (size_t i = 0; i < survivor_buf.size(); ++i) {
      ptrs[i] = survivor_buf[i].data();
    }
    return rs_.DecodeData(survivor_index, ptrs, shard_len_, out);
  }

  size_t shard_len_;
  size_t num_stripes_;
  int k_;
  int m_max_;
  ReedSolomon rs_;
  int parity_level_;
  std::vector<EcMember> members_;  // [0,k) data, then parity in add order
};

}  // namespace clio::run::safe_bdev::ec

#endif  // SAFE_BDEV_EC_EC_ARRAY_H_
