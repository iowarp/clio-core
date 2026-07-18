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

#ifndef SAFE_BDEV_EC_DECLUSTERED_H_
#define SAFE_BDEV_EC_DECLUSTERED_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ec_array.h"      // MemberStore, MemoryStore, EcState
#include "reed_solomon.h"

/**
 * Declustered, variable-width erasure-coded layout (design.md Part C),
 * daemon-independent so it can be unit-tested directly.
 *
 * Two ideas on top of the fixed EcArray:
 *
 *  - GENERATIONS (variable width). A generation is a frozen (k, m, member-set)
 *    config that owns a contiguous range of GLOBAL stripe indices. Adding a data
 *    drive opens a NEW generation with k+1 data shards for FUTURE stripes;
 *    existing generations keep their geometry, so no data is ever rewritten when
 *    the array widens.
 *
 *  - ROTATED-PARITY DECLUSTERING. Within a generation of N = k+m members, the
 *    role each member plays (which data column, or which parity row) ROTATES by
 *    the stripe index. Over N consecutive stripes every member is parity exactly
 *    m times, so parity — and therefore rebuild-read load — is spread evenly
 *    across all members instead of pinned to dedicated parity drives.
 *
 * Slot model: a member's shard for global stripe g is stored at slot g on that
 * member (byte offset g * shard_len). Generations partition the global stripe
 * range, so a member only writes slots for stripes in generations it belongs to.
 *
 * (This is the N = k+m form of declustering — rotated parity. Subset
 * declustering with N > k+m is a further extension.)
 */

namespace clio::run::safe_bdev::ec {

class DeclusteredArray {
 public:
  explicit DeclusteredArray(size_t shard_len) : shard_len_(shard_len) {}

  /** Register a (non-owning) member store; returns its global member id. */
  int AddMemberStore(MemberStore *store) {
    members_.push_back(store);
    member_state_.push_back(EcState::kActive);
    return static_cast<int>(members_.size()) - 1;
  }

  /**
   * Open a generation covering the next `num_stripes` global stripes.
   * @param k            data shards per stripe
   * @param m            parity shards per stripe (N = k+m == gen_members.size())
   * @param gen_members  global member ids participating (size k+m)
   * @return epoch index
   */
  int OpenGeneration(int k, int m, const std::vector<int> &gen_members,
                     uint64_t num_stripes) {
    Generation g;
    g.k = k;
    g.m = m;
    g.first_stripe = next_stripe_;
    g.num_stripes = num_stripes;
    g.seed = static_cast<uint64_t>(gens_.size());  // vary rotation per gen
    g.members = gen_members;
    g.rs = std::make_unique<ReedSolomon>(k, m);
    next_stripe_ += num_stripes;
    gens_.push_back(std::move(g));
    return static_cast<int>(gens_.size()) - 1;
  }

  size_t shard_len() const { return shard_len_; }
  uint64_t total_stripes() const { return next_stripe_; }
  int num_generations() const { return static_cast<int>(gens_.size()); }
  int GenerationK(int epoch) const { return gens_[epoch].k; }

  /** Epoch owning global stripe g, or -1. */
  int EpochOf(uint64_t g) const {
    for (size_t e = 0; e < gens_.size(); ++e) {
      if (g >= gens_[e].first_stripe &&
          g < gens_[e].first_stripe + gens_[e].num_stripes) {
        return static_cast<int>(e);
      }
    }
    return -1;
  }

  /**
   * RS shard index q played by the member at position p (in gen.members) for
   * global stripe g. q in [0,k) is data column q; q in [k,N) is parity row q-k.
   */
  int ShardOfPosition(int epoch, uint64_t g, int p) const {
    const Generation &gen = gens_[epoch];
    const int N = static_cast<int>(gen.members.size());
    const uint64_t local = g - gen.first_stripe;
    const int r = static_cast<int>((local + gen.seed) % static_cast<uint64_t>(N));
    return ((p - r) % N + N) % N;
  }

  /** Member position holding RS shard q for global stripe g (inverse). */
  int PositionOfShard(int epoch, uint64_t g, int q) const {
    const Generation &gen = gens_[epoch];
    const int N = static_cast<int>(gen.members.size());
    const uint64_t local = g - gen.first_stripe;
    const int r = static_cast<int>((local + gen.seed) % static_cast<uint64_t>(N));
    return (q + r) % N;
  }

  /** Global member id holding RS shard q for global stripe g. */
  int MemberOfShard(int epoch, uint64_t g, int q) const {
    return gens_[epoch].members[
        static_cast<size_t>(PositionOfShard(epoch, g, q))];
  }

  /**
   * Write one stripe: `data_shards` holds gen.k buffers of shard_len bytes.
   * Data shards and freshly-encoded parity are placed on the rotated members.
   */
  bool WriteStripe(uint64_t g, const std::vector<std::vector<uint8_t>> &data_shards) {
    const int e = EpochOf(g);
    if (e < 0) {
      return false;
    }
    const Generation &gen = gens_[e];
    if (static_cast<int>(data_shards.size()) != gen.k) {
      return false;
    }
    const uint64_t off = g * shard_len_;
    // Data shards onto their rotated members.
    for (int c = 0; c < gen.k; ++c) {
      const int mid = MemberOfShard(e, g, c);
      if (member_state_[static_cast<size_t>(mid)] == EcState::kActive &&
          !members_[static_cast<size_t>(mid)]->Write(off, shard_len_,
                                                     data_shards[c].data())) {
        return false;
      }
    }
    // Parity rows onto their rotated members.
    std::vector<const uint8_t *> ptrs(static_cast<size_t>(gen.k));
    for (int c = 0; c < gen.k; ++c) {
      ptrs[static_cast<size_t>(c)] = data_shards[c].data();
    }
    for (int pr = 0; pr < gen.m; ++pr) {
      const int mid = MemberOfShard(e, g, gen.k + pr);
      if (member_state_[static_cast<size_t>(mid)] != EcState::kActive) {
        continue;
      }
      std::vector<uint8_t> parity(shard_len_, 0);
      gen.rs->EncodeParityShard(pr, ptrs, shard_len_, parity.data());
      if (!members_[static_cast<size_t>(mid)]->Write(off, shard_len_,
                                                    parity.data())) {
        return false;
      }
    }
    return true;
  }

  /** Read the gen.k data shards of stripe g, reconstructing if needed. */
  bool ReadStripe(uint64_t g, std::vector<std::vector<uint8_t>> *out) {
    const int e = EpochOf(g);
    if (e < 0) {
      return false;
    }
    const Generation &gen = gens_[e];
    bool all_data_active = true;
    for (int c = 0; c < gen.k; ++c) {
      const int mid = MemberOfShard(e, g, c);
      if (member_state_[static_cast<size_t>(mid)] != EcState::kActive) {
        all_data_active = false;
        break;
      }
    }
    const uint64_t off = g * shard_len_;
    if (all_data_active) {
      out->assign(static_cast<size_t>(gen.k),
                  std::vector<uint8_t>(shard_len_, 0));
      for (int c = 0; c < gen.k; ++c) {
        const int mid = MemberOfShard(e, g, c);
        if (!members_[static_cast<size_t>(mid)]->Read(off, shard_len_,
                                                     (*out)[c].data())) {
          return false;
        }
      }
      return true;
    }
    return ReconstructStripe(g, /*exclude_mid=*/-1, out);
  }

  void MarkFaulty(int mid) { member_state_[static_cast<size_t>(mid)] = EcState::kFaulty; }

  /**
   * Reconstruct a failed member's shards onto `new_store` and bring it online.
   * Spans every generation the member belongs to.
   */
  bool RecoverMember(int mid, MemberStore *new_store) {
    for (size_t e = 0; e < gens_.size(); ++e) {
      const Generation &gen = gens_[e];
      const int p = PositionInGen(static_cast<int>(e), mid);
      if (p < 0) {
        continue;  // member not in this generation
      }
      for (uint64_t g = gen.first_stripe;
           g < gen.first_stripe + gen.num_stripes; ++g) {
        std::vector<std::vector<uint8_t>> data;
        if (!ReconstructStripe(g, /*exclude_mid=*/mid, &data)) {
          return false;
        }
        const int q = ShardOfPosition(static_cast<int>(e), g, p);
        std::vector<uint8_t> shard(shard_len_, 0);
        if (q < gen.k) {
          shard = data[static_cast<size_t>(q)];
        } else {
          std::vector<const uint8_t *> ptrs(static_cast<size_t>(gen.k));
          for (int c = 0; c < gen.k; ++c) {
            ptrs[static_cast<size_t>(c)] = data[static_cast<size_t>(c)].data();
          }
          gen.rs->EncodeParityShard(q - gen.k, ptrs, shard_len_, shard.data());
        }
        if (!new_store->Write(g * shard_len_, shard_len_, shard.data())) {
          return false;
        }
      }
    }
    members_[static_cast<size_t>(mid)] = new_store;
    member_state_[static_cast<size_t>(mid)] = EcState::kActive;
    return true;
  }

  /** Count how many of `count` consecutive stripes place parity on member mid
   *  (test helper for verifying declustering spread). */
  int ParityCountForMember(int epoch, int mid, uint64_t count) const {
    const Generation &gen = gens_[epoch];
    const int p = PositionInGen(epoch, mid);
    if (p < 0) {
      return 0;
    }
    int n = 0;
    for (uint64_t i = 0; i < count; ++i) {
      const uint64_t g = gen.first_stripe + i;
      if (ShardOfPosition(epoch, g, p) >= gen.k) {
        ++n;
      }
    }
    return n;
  }

 private:
  struct Generation {
    int k = 0;
    int m = 0;
    uint64_t first_stripe = 0;
    uint64_t num_stripes = 0;
    uint64_t seed = 0;
    std::vector<int> members;  // global member ids, size k+m
    std::unique_ptr<ReedSolomon> rs;
  };

  /** Position of global member mid within gen.members, or -1. */
  int PositionInGen(int epoch, int mid) const {
    const Generation &gen = gens_[epoch];
    for (size_t p = 0; p < gen.members.size(); ++p) {
      if (gen.members[p] == mid) {
        return static_cast<int>(p);
      }
    }
    return -1;
  }

  /** Reconstruct all gen.k data shards for stripe g from active survivors. */
  bool ReconstructStripe(uint64_t g, int exclude_mid,
                         std::vector<std::vector<uint8_t>> *out) {
    const int e = EpochOf(g);
    if (e < 0) {
      return false;
    }
    const Generation &gen = gens_[e];
    const uint64_t off = g * shard_len_;
    std::vector<int> survivor_index;
    std::vector<std::vector<uint8_t>> survivor_buf;
    for (size_t p = 0; p < gen.members.size(); ++p) {
      const int mid = gen.members[p];
      if (mid == exclude_mid ||
          member_state_[static_cast<size_t>(mid)] != EcState::kActive) {
        continue;
      }
      std::vector<uint8_t> buf(shard_len_, 0);
      if (!members_[static_cast<size_t>(mid)]->Read(off, shard_len_,
                                                   buf.data())) {
        return false;
      }
      survivor_index.push_back(ShardOfPosition(e, g, static_cast<int>(p)));
      survivor_buf.push_back(std::move(buf));
      if (static_cast<int>(survivor_index.size()) == gen.k) {
        break;
      }
    }
    if (static_cast<int>(survivor_index.size()) < gen.k) {
      return false;
    }
    std::vector<const uint8_t *> ptrs(survivor_buf.size());
    for (size_t i = 0; i < survivor_buf.size(); ++i) {
      ptrs[i] = survivor_buf[i].data();
    }
    return gen.rs->DecodeData(survivor_index, ptrs, shard_len_, out);
  }

  size_t shard_len_;
  uint64_t next_stripe_ = 0;
  std::vector<MemberStore *> members_;     // global registry (non-owning)
  std::vector<EcState> member_state_;      // parallel to members_
  std::vector<Generation> gens_;
};

}  // namespace clio::run::safe_bdev::ec

#endif  // SAFE_BDEV_EC_DECLUSTERED_H_
