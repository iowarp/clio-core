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
 * Daemon-independent unit tests for the safe_bdev erasure-coding core (#543).
 *
 * These tests exercise the math and the member-array/recovery layer directly
 * against in-memory member stores — no Chimaera runtime is started. They cover:
 *   - GF(2^8) arithmetic invariants
 *   - Reed-Solomon encode/decode (reconstruct from any k survivors)
 *   - incremental parity (each parity row is independent)
 *   - EcArray: add a (parity) drive, remove a faulty drive with degraded reads,
 *     and recover a failed drive with exact byte-for-byte reconstruction.
 *
 * NOTE: simple_test.h SECTIONs are sequential and share state (NOT Catch2-style
 * isolated re-runs), so each independent scenario is its own TEST_CASE and all
 * member stores are owned for the lifetime of the case (see EcLab).
 */

#include <cstdint>
#include <memory>
#include <vector>

#include "clio_runtime/safe_bdev/ec/declustered.h"
#include "clio_runtime/safe_bdev/ec/ec_array.h"
#include "clio_runtime/safe_bdev/ec/gf256.h"
#include "clio_runtime/safe_bdev/ec/reed_solomon.h"
#include "simple_test.h"

namespace ec = clio::run::safe_bdev::ec;

namespace {

// Deterministic shard contents so every run is reproducible.
std::vector<uint8_t> GenShard(int col, int stripe, size_t len) {
  std::vector<uint8_t> v(len);
  for (size_t b = 0; b < len; ++b) {
    v[b] = static_cast<uint8_t>((col * 37 + stripe * 101 + b * 7 + 13) & 0xFF);
  }
  return v;
}

// First-k survivors helper for the RS-level tests: given the set of lost global
// shard indices, return the first k surviving (index, pointer) pairs.
bool PickSurvivors(int k, int total, const std::vector<int> &lost,
                   const std::vector<const uint8_t *> &all_shards,
                   std::vector<int> *idx,
                   std::vector<const uint8_t *> *ptr) {
  idx->clear();
  ptr->clear();
  for (int g = 0; g < total && static_cast<int>(idx->size()) < k; ++g) {
    bool is_lost = false;
    for (int l : lost) {
      if (l == g) {
        is_lost = true;
        break;
      }
    }
    if (is_lost) {
      continue;
    }
    idx->push_back(g);
    ptr->push_back(all_shards[static_cast<size_t>(g)]);
  }
  return static_cast<int>(idx->size()) == k;
}

/**
 * Test harness owning every member store (data, parity, and recovery targets)
 * for the lifetime of a TEST_CASE, so the EcArray never dangles.
 */
struct EcLab {
  size_t shard_len;
  size_t stripes;
  int k;
  int m_max;
  std::vector<std::unique_ptr<ec::MemoryStore>> stores;  // owns ALL stores
  std::unique_ptr<ec::EcArray> arr;

  EcLab(size_t sl, size_t st, int kk, int mm)
      : shard_len(sl), stripes(st), k(kk), m_max(mm) {
    std::vector<ec::MemberStore *> data_ptrs;
    for (int c = 0; c < k; ++c) {
      stores.push_back(std::make_unique<ec::MemoryStore>(sl * st));
      data_ptrs.push_back(stores.back().get());
    }
    arr = std::make_unique<ec::EcArray>(sl, st, k, mm, data_ptrs);
    for (size_t s = 0; s < st; ++s) {
      std::vector<std::vector<uint8_t>> ds(k);
      for (int c = 0; c < k; ++c) {
        ds[c] = GenShard(c, static_cast<int>(s), sl);
      }
      arr->WriteStripe(s, ds);
    }
  }

  // Allocate a fresh store owned by the lab (parity or recovery target).
  ec::MemoryStore *NewStore() {
    stores.push_back(std::make_unique<ec::MemoryStore>(shard_len * stripes));
    return stores.back().get();
  }

  // Add `n` parity drives, raising tolerance by n.
  void AddParity(int n) {
    for (int i = 0; i < n; ++i) {
      arr->AddParityDrive(NewStore());
    }
  }

  // The expected full byte image of data column `col` across all stripes.
  std::vector<uint8_t> ExpectedColumnImage(int col) const {
    std::vector<uint8_t> img;
    for (size_t s = 0; s < stripes; ++s) {
      std::vector<uint8_t> shard = GenShard(col, static_cast<int>(s), shard_len);
      img.insert(img.end(), shard.begin(), shard.end());
    }
    return img;
  }

  // Verify every stripe's data reads back as written.
  void RequireAllDataReadable() const {
    for (size_t s = 0; s < stripes; ++s) {
      std::vector<std::vector<uint8_t>> rd;
      REQUIRE(arr->ReadStripeData(s, &rd));
      for (int c = 0; c < k; ++c) {
        REQUIRE(rd[c] == GenShard(c, static_cast<int>(s), shard_len));
      }
    }
  }
};

}  // namespace

TEST_CASE("gf256_field_axioms", "[safe_bdev][ec][gf256]") {
  for (int a = 0; a < 256; ++a) {
    REQUIRE(ec::GfMul(static_cast<uint8_t>(a), 0) == 0);
    REQUIRE(ec::GfMul(0, static_cast<uint8_t>(a)) == 0);
    REQUIRE(ec::GfMul(static_cast<uint8_t>(a), 1) == static_cast<uint8_t>(a));
  }
  for (int a = 1; a < 256; ++a) {
    uint8_t av = static_cast<uint8_t>(a);
    REQUIRE(ec::GfMul(av, ec::GfInv(av)) == 1);
    for (int b = 1; b < 256; ++b) {
      uint8_t bv = static_cast<uint8_t>(b);
      REQUIRE(ec::GfDiv(ec::GfMul(av, bv), bv) == av);
    }
  }
}

TEST_CASE("reed_solomon_reconstruct_any_k", "[safe_bdev][ec][rs]") {
  const int k = 4;
  const int m = 3;
  const size_t len = 64;
  ec::ReedSolomon rs(k, m);

  std::vector<std::vector<uint8_t>> data(k);
  std::vector<const uint8_t *> dptr(k);
  for (int c = 0; c < k; ++c) {
    data[c] = GenShard(c, /*stripe=*/0, len);
    dptr[c] = data[c].data();
  }
  std::vector<std::vector<uint8_t>> parity;
  rs.Encode(dptr, len, m, &parity);

  std::vector<const uint8_t *> all(k + m);
  for (int c = 0; c < k; ++c) all[c] = data[c].data();
  for (int r = 0; r < m; ++r) all[k + r] = parity[r].data();

  std::vector<std::vector<int>> loss_sets = {
      {},        {0},         {3},          {k},    {k + 2},
      {0, 1},    {0, k},      {k, k + 1},   {1, 3}, {0, 1, 2},
      {0, 1, k}, {0, k, k + 1}, {k, k + 1, k + 2}};

  for (const auto &lost : loss_sets) {
    if (static_cast<int>(lost.size()) > m) continue;  // beyond tolerance
    std::vector<int> sidx;
    std::vector<const uint8_t *> sptr;
    REQUIRE(PickSurvivors(k, k + m, lost, all, &sidx, &sptr));
    std::vector<std::vector<uint8_t>> recovered;
    REQUIRE(rs.DecodeData(sidx, sptr, len, &recovered));
    for (int c = 0; c < k; ++c) {
      REQUIRE(recovered[c] == data[c]);
    }
  }
}

TEST_CASE("reed_solomon_incremental_parity", "[safe_bdev][ec][incremental]") {
  const int k = 5;
  const int m = 3;
  const size_t len = 48;
  ec::ReedSolomon rs(k, m);

  std::vector<std::vector<uint8_t>> data(k);
  std::vector<const uint8_t *> dptr(k);
  for (int c = 0; c < k; ++c) {
    data[c] = GenShard(c, 7, len);
    dptr[c] = data[c].data();
  }

  // Computing each parity row in isolation (the incremental-upgrade primitive)
  // must yield exactly the same bytes as the batch encode, proving the rows do
  // not depend on each other.
  std::vector<std::vector<uint8_t>> full;
  rs.Encode(dptr, len, m, &full);
  for (int r = 0; r < m; ++r) {
    std::vector<uint8_t> one(len, 0);
    rs.EncodeParityShard(r, dptr, len, one.data());
    REQUIRE(one == full[r]);
  }
}

TEST_CASE("ec_array_add_parity_drive", "[safe_bdev][ec][add]") {
  EcLab lab(/*shard_len=*/32, /*stripes=*/5, /*k=*/3, /*m_max=*/2);
  REQUIRE(lab.arr->parity_level() == 0);

  REQUIRE(lab.arr->AddParityDrive(lab.NewStore()) == 1);
  REQUIRE(lab.arr->parity_level() == 1);

  REQUIRE(lab.arr->AddParityDrive(lab.NewStore()) == 2);
  REQUIRE(lab.arr->parity_level() == 2);

  // Parity is capped at m_max — we never grow beyond the target tolerance.
  REQUIRE(lab.arr->AddParityDrive(lab.NewStore()) == -1);
  REQUIRE(lab.arr->parity_level() == 2);

  // Data still reads back correctly after parity was layered on incrementally.
  lab.RequireAllDataReadable();
}

TEST_CASE("ec_array_remove_one_faulty_data_drive", "[safe_bdev][ec][remove]") {
  EcLab lab(/*shard_len=*/32, /*stripes=*/4, /*k=*/3, /*m_max=*/2);
  lab.AddParity(1);  // tolerate one failure

  lab.arr->RemoveDrive(lab.arr->DataMemberIndex(1), /*was_faulty=*/true);

  // Degraded reads reconstruct the failed data drive's contents.
  lab.RequireAllDataReadable();
}

TEST_CASE("ec_array_remove_two_faulty_drives", "[safe_bdev][ec][remove]") {
  EcLab lab(/*shard_len=*/32, /*stripes=*/4, /*k=*/3, /*m_max=*/2);
  lab.AddParity(2);  // tolerate two failures

  lab.arr->RemoveDrive(lab.arr->DataMemberIndex(0), /*was_faulty=*/true);
  lab.arr->RemoveDrive(lab.arr->DataMemberIndex(2), /*was_faulty=*/true);

  // Two data drives down at tolerance 2: survivors (1 data + 2 parity) still
  // reconstruct every stripe.
  lab.RequireAllDataReadable();
}

TEST_CASE("ec_array_recover_data_drive", "[safe_bdev][ec][recover]") {
  EcLab lab(/*shard_len=*/32, /*stripes=*/6, /*k=*/4, /*m_max=*/2);
  lab.AddParity(2);

  const int col = 1;
  lab.arr->RemoveDrive(lab.arr->DataMemberIndex(col), /*was_faulty=*/true);

  ec::MemoryStore *replacement = lab.NewStore();
  REQUIRE(lab.arr->RecoverMember(lab.arr->DataMemberIndex(col), replacement));

  // The reconstructed member must byte-match the original data member exactly.
  REQUIRE(replacement->bytes() == lab.ExpectedColumnImage(col));
  REQUIRE(replacement->bytes() == lab.stores[static_cast<size_t>(col)]->bytes());

  // After recovery the member is active again; reads are non-degraded.
  for (size_t s = 0; s < lab.stripes; ++s) {
    std::vector<uint8_t> shard;
    REQUIRE(lab.arr->ReadDataShard(s, col, &shard));
    REQUIRE(shard == GenShard(col, static_cast<int>(s), lab.shard_len));
  }
}

TEST_CASE("ec_array_recover_parity_drive", "[safe_bdev][ec][recover]") {
  EcLab lab(/*shard_len=*/32, /*stripes=*/6, /*k=*/4, /*m_max=*/2);
  lab.AddParity(2);

  // Parity row 0's store is the first store added after the k data stores.
  const size_t parity0_store = static_cast<size_t>(lab.k);
  const std::vector<uint8_t> original_parity0 = lab.stores[parity0_store]->bytes();

  lab.arr->RemoveDrive(lab.arr->ParityMemberIndex(0), /*was_faulty=*/true);

  ec::MemoryStore *replacement = lab.NewStore();
  REQUIRE(lab.arr->RecoverMember(lab.arr->ParityMemberIndex(0), replacement));

  // Recovered parity must match the originally-computed parity bytes.
  REQUIRE(replacement->bytes() == original_parity0);
}

TEST_CASE("declustered_parity_spread", "[safe_bdev][ec][declustered]") {
  // k=3, m=1, N=4: over N consecutive stripes each member must be parity
  // exactly m=1 time — i.e., parity is spread evenly (declustered), not pinned.
  const size_t shard_len = 16;
  ec::DeclusteredArray arr(shard_len);
  std::vector<std::unique_ptr<ec::MemoryStore>> stores;
  std::vector<int> ids;
  for (int i = 0; i < 4; ++i) {
    stores.push_back(std::make_unique<ec::MemoryStore>(shard_len * 8));
    ids.push_back(arr.AddMemberStore(stores.back().get()));
  }
  const int e = arr.OpenGeneration(3, 1, ids, /*num_stripes=*/4);
  for (int mid : ids) {
    REQUIRE(arr.ParityCountForMember(e, mid, 4) == 1);
  }
}

TEST_CASE("declustered_variable_width_recover", "[safe_bdev][ec][declustered]") {
  // Two generations of different width over a shared member pool:
  //   gen0: k=3, m=1 over members {0,1,2,3}    -> stripes [0,4)
  //   gen1: k=4, m=1 over members {0,1,2,4 | 3} -> stripes [4,8)  (widened)
  // Existing (gen0) stripes keep their geometry; gen1 uses the added member 4.
  const size_t shard_len = 32;
  const uint64_t total = 8;
  ec::DeclusteredArray arr(shard_len);
  std::vector<std::unique_ptr<ec::MemoryStore>> stores;
  std::vector<int> ids;
  for (int i = 0; i < 5; ++i) {
    stores.push_back(std::make_unique<ec::MemoryStore>(shard_len * total));
    ids.push_back(arr.AddMemberStore(stores.back().get()));
  }
  arr.OpenGeneration(3, 1, {ids[0], ids[1], ids[2], ids[3]}, 4);
  arr.OpenGeneration(4, 1, {ids[0], ids[1], ids[2], ids[4], ids[3]}, 4);
  REQUIRE(arr.total_stripes() == total);

  auto write_all = [&]() {
    for (uint64_t g = 0; g < total; ++g) {
      const int k = arr.GenerationK(arr.EpochOf(g));
      std::vector<std::vector<uint8_t>> ds(static_cast<size_t>(k));
      for (int c = 0; c < k; ++c) {
        ds[c] = GenShard(c, static_cast<int>(g), shard_len);
      }
      REQUIRE(arr.WriteStripe(g, ds));
    }
  };
  auto verify_all = [&]() {
    for (uint64_t g = 0; g < total; ++g) {
      const int k = arr.GenerationK(arr.EpochOf(g));
      std::vector<std::vector<uint8_t>> rd;
      REQUIRE(arr.ReadStripe(g, &rd));
      REQUIRE(static_cast<int>(rd.size()) == k);
      for (int c = 0; c < k; ++c) {
        REQUIRE(rd[c] == GenShard(c, static_cast<int>(g), shard_len));
      }
    }
  };

  write_all();
  verify_all();  // happy path across both generations

  // Fault member 0 (participates in BOTH generations): degraded reads
  // reconstruct in each generation under its own (k,m) geometry.
  arr.MarkFaulty(ids[0]);
  verify_all();

  // Recover member 0 onto a fresh store; reads are non-degraded again.
  auto repl = std::make_unique<ec::MemoryStore>(shard_len * total);
  REQUIRE(arr.RecoverMember(ids[0], repl.get()));
  verify_all();

  // Redundancy restored: fault a different member, still reconstructs.
  arr.MarkFaulty(ids[1]);
  verify_all();
}

SIMPLE_TEST_MAIN()
