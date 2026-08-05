/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * 4-NODE CACHE COHERENCE SUITE (issue #886 distributed coherence).
 *
 * Every node runs this binary against its own daemon (rank = NODE_ID-1),
 * clients bound to the chain top via CLIO_CTE_POOL. Node-to-node
 * synchronization is blob-based barriers through the store itself.
 *
 *  1. Put Once, Read Many — each rank puts its own blob; every rank then
 *     repeatedly reads its neighbor's blob. After the first read a
 *     NODE-LOCAL DRAM copy must exist (reader-side caching) and every
 *     read returns the writer's bytes.
 *  2. Segmented PORM — each rank partial-puts a disjoint range of ONE
 *     shared blob; every rank full-reads it and must see all segments.
 *  3. Repeated Segmented PORM — iterate (2) with round-seeded payloads;
 *     every round must be coherent (stale local caches must be
 *     invalidated by the next round's writes).
 *  4. Fragmented PORM — all ranks write the ENTIRE blob simultaneously;
 *     each rank re-reads until the content converges to ONE writer's
 *     complete round-stamped pattern (eventual coherence).
 *  5. Repeated Fragmented PORM — iterate (4); every round converges.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace {

constexpr int kNumNodes = 4;
constexpr clio::run::u64 kPormSize = 1ULL * 1024 * 1024;
constexpr clio::run::u64 kSegSize = 512ULL * 1024;  // per-rank segment
constexpr clio::run::u64 kSharedSize = kSegSize * kNumNodes;
constexpr int kRepeatRounds = 5;
constexpr int kReadRounds = 10;

int Rank() {
  const char *env = std::getenv("NODE_ID");
  return env ? std::atoi(env) - 1 : 0;
}

/** Deterministic pattern: byte i of (writer, round) — the header-free
 *  encoding lets a reader DECODE which write it is looking at from any
 *  single byte pair and regenerate the full expected buffer. */
void FillPattern(char *buf, clio::run::u64 size, int writer, int round) {
  for (clio::run::u64 i = 0; i < size; ++i) {
    buf[i] = static_cast<char>((writer * 31 + round * 7 + i * 13) & 0x7f);
  }
}

bool MatchesPattern(const char *buf, clio::run::u64 size, int writer,
                    int round) {
  for (clio::run::u64 i = 0; i < size; ++i) {
    if (buf[i] != static_cast<char>((writer * 31 + round * 7 + i * 13) &
                                    0x7f)) {
      return false;
    }
  }
  return true;
}

clio::cte::core::Client *Chain() { return CLIO_CTE_CLIENT; }

/** Decode which (writer, round) pattern buf holds, for failure forensics.
 *  Returns "w<writer>r<round>", "torn(...)" for a recognizable prefix, or
 *  "unknown". */
std::string DecodePattern(const char *buf, clio::run::u64 size) {
  for (int w = 0; w < kNumNodes; ++w) {
    for (int r = 0; r < kRepeatRounds; ++r) {
      if (MatchesPattern(buf, size, w, r)) {
        return "w" + std::to_string(w) + "r" + std::to_string(r);
      }
    }
  }
  for (int w = 0; w < kNumNodes; ++w) {
    for (int r = 0; r < kRepeatRounds; ++r) {
      if (buf[0] == static_cast<char>((w * 31 + r * 7) & 0x7f) &&
          buf[1] == static_cast<char>((w * 31 + r * 7 + 13) & 0x7f)) {
        return "torn(w" + std::to_string(w) + "r" + std::to_string(r) +
               " prefix)";
      }
    }
  }
  return "unknown";
}

/** Blob-based rendezvous: every rank puts its arrival blob and waits for
 *  the other ranks' — the store itself is the message board. Epochs make
 *  names unique so no cleanup is needed between barriers. */
int g_barrier_epoch = 0;
bool Barrier(const clio::cte::core::TagId &tag_id) {
  const int epoch = g_barrier_epoch++;
  char one = 1;
  {
    std::string mine = "barrier_" + std::to_string(epoch) + "_" +
                       std::to_string(Rank());
    auto put = Chain()->AsyncPutBlob(tag_id, mine, 0, 1, &one);
    put.Wait();
    if (put->GetReturnCode() != 0) {
      return false;
    }
  }
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  for (int r = 0; r < kNumNodes; ++r) {
    std::string name =
        "barrier_" + std::to_string(epoch) + "_" + std::to_string(r);
    while (true) {
      auto sz = Chain()->AsyncGetBlobSize(tag_id, name);
      sz.Wait();
      if (sz->GetReturnCode() == 0 && sz->size_ == 1) {
        break;
      }
      if (std::chrono::steady_clock::now() > deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  return true;
}

/** Read tag/name fully through the chain into buf; true on rc==0. */
bool ReadFull(const clio::cte::core::TagId &tag_id, const std::string &name,
              char *buf, clio::run::u64 size) {
  auto get = Chain()->AsyncGetBlob(tag_id, name, 0, size, /*flags=*/0, buf);
  get.Wait();
  return get->GetReturnCode() == 0;
}

}  // namespace

TEST_CASE("Coherence - put once read many (reader-local DRAM)",
          "[coherence][porm]") {
  clio::cte::core::Tag tag("coherence_porm");
  const clio::cte::core::TagId tag_id = tag.GetTagId();
  const int rank = Rank();

  // Everybody publishes its own blob...
  std::vector<char> mine(kPormSize);
  FillPattern(mine.data(), kPormSize, rank, 0);
  const std::string my_name = "porm_" + std::to_string(rank);
  {
    auto put = Chain()->AsyncPutBlob(tag_id, my_name, 0, kPormSize,
                                     mine.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
  }
  REQUIRE(Barrier(tag_id));

  // ...then repeatedly reads its neighbor's.
  const int peer = (rank + 1) % kNumNodes;
  const std::string peer_name = "porm_" + std::to_string(peer);
  std::vector<char> got(kPormSize);

  REQUIRE(ReadFull(tag_id, peer_name, got.data(), kPormSize));
  REQUIRE(MatchesPattern(got.data(), kPormSize, peer, 0));

  // The first read must leave a NODE-LOCAL DRAM copy behind (reader-side
  // caching): the LOCAL core container answers for the blob. Poll — the
  // population is best-effort-async by contract, only its EVENTUAL
  // existence is guaranteed.
  {
    clio::cte::core::Client direct(clio::cte::core::kCtePoolId);
    bool local = false;
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      auto sz = direct.AsyncGetBlobSize(tag_id, peer_name,
                                        clio::run::PoolQuery::Local(),
                                        clio::cte::core::kCacheReplica);
      sz.Wait();
      if (sz->GetReturnCode() == 0 && sz->size_ >= kPormSize) {
        local = true;
        break;
      }
      // The blob's OWNER node keeps no cache-slot copy by design (issue
      // #894): there the authoritative primary IS the node-local DRAM
      // copy. A Local primary probe answers only on the owner node.
      auto psz = direct.AsyncGetBlobSize(tag_id, peer_name,
                                         clio::run::PoolQuery::Local());
      psz.Wait();
      if (psz->GetReturnCode() == 0 && psz->size_ >= kPormSize) {
        local = true;
        break;
      }
      // Re-read: population may piggyback on reads.
      REQUIRE(ReadFull(tag_id, peer_name, got.data(), kPormSize));
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    REQUIRE(local);
  }

  // Every subsequent read stays correct.
  for (int i = 0; i < kReadRounds; ++i) {
    REQUIRE(ReadFull(tag_id, peer_name, got.data(), kPormSize));
    REQUIRE(MatchesPattern(got.data(), kPormSize, peer, 0));
  }
  REQUIRE(Barrier(tag_id));
}

static void SegmentedRound(const clio::cte::core::TagId &tag_id,
                           const std::string &blob, int round) {
  const int rank = Rank();
  // Disjoint per-rank segment writes to ONE shared blob.
  std::vector<char> seg(kSegSize);
  FillPattern(seg.data(), kSegSize, rank, round);
  {
    auto put = Chain()->AsyncPutBlob(tag_id, blob, rank * kSegSize, kSegSize,
                                     seg.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
  }
  REQUIRE(Barrier(tag_id));

  // Full read: every segment must be exactly its writer's ROUND-STAMPED
  // pattern — a stale cached copy of a previous round fails here.
  std::vector<char> got(kSharedSize);
  REQUIRE(ReadFull(tag_id, blob, got.data(), kSharedSize));
  for (int w = 0; w < kNumNodes; ++w) {
    REQUIRE(MatchesPattern(got.data() + w * kSegSize, kSegSize, w, round));
  }
  REQUIRE(Barrier(tag_id));
}

TEST_CASE("Coherence - segmented put once read many",
          "[coherence][segmented]") {
  clio::cte::core::Tag tag("coherence_seg");
  SegmentedRound(tag.GetTagId(), "seg_blob", 0);
}

TEST_CASE("Coherence - repeated segmented put once read many",
          "[coherence][segmented][repeated]") {
  clio::cte::core::Tag tag("coherence_seg_rep");
  for (int round = 0; round < kRepeatRounds; ++round) {
    SegmentedRound(tag.GetTagId(), "seg_rep_blob", round);
  }
}

static void FragmentedRound(const clio::cte::core::TagId &tag_id,
                            const std::string &blob, int round) {
  const int rank = Rank();
  // ALL ranks write the ENTIRE blob simultaneously (racing full puts).
  std::vector<char> mine(kSharedSize);
  FillPattern(mine.data(), kSharedSize, rank, round);
  {
    auto put = Chain()->AsyncPutBlob(tag_id, blob, 0, kSharedSize,
                                     mine.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
  }
  REQUIRE(Barrier(tag_id));

  // Eventual coherence: re-read until the content is ONE writer's complete
  // round-stamped pattern. (The racing puts serialized under the blob's
  // write token; once the dust settles every reader must see the last
  // writer's bytes — never a stale round, never a torn blend forever.)
  std::vector<char> got(kSharedSize);
  bool uniform = false;
  int winner = -1;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (std::chrono::steady_clock::now() < deadline) {
    REQUIRE(ReadFull(tag_id, blob, got.data(), kSharedSize));
    for (int w = 0; w < kNumNodes; ++w) {
      if (MatchesPattern(got.data(), kSharedSize, w, round)) {
        uniform = true;
        winner = w;
        break;
      }
    }
    if (uniform) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  REQUIRE(uniform);

  // After a rendezvous the store is quiescent: one more read must still be
  // the SAME winner on every node (agreement, not just per-node
  // stability). Publish each node's winner and cross-check.
  REQUIRE(Barrier(tag_id));
  {
    char w = static_cast<char>('0' + winner);
    std::string mine_name = "winner_" + std::to_string(g_barrier_epoch) +
                            "_" + std::to_string(rank);
    auto put = Chain()->AsyncPutBlob(tag_id, mine_name, 0, 1, &w);
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
  }
  const int epoch = g_barrier_epoch;
  REQUIRE(Barrier(tag_id));
  for (int r = 0; r < kNumNodes; ++r) {
    char w = 0;
    std::string name =
        "winner_" + std::to_string(epoch) + "_" + std::to_string(r);
    REQUIRE(ReadFull(tag_id, name, &w, 1));
    if (w != static_cast<char>('0' + winner)) {
      // Forensics for the winner split: whose pattern is this node actually
      // holding, and does a fresh read still return it?
      std::vector<char> now(kSharedSize);
      std::string now_dec = "readfail";
      if (ReadFull(tag_id, blob, now.data(), kSharedSize)) {
        now_dec = DecodePattern(now.data(), kSharedSize);
      }
      fprintf(stderr,
              "[COH-DIAG] rank=%d round=%d winner=%d but rank %d published "
              "'%c'; my stable read decodes as %s, fresh re-read decodes as "
              "%s\n",
              rank, round, winner, r, w, DecodePattern(got.data(),
              kSharedSize).c_str(), now_dec.c_str());
    }
    REQUIRE(w == static_cast<char>('0' + winner));
  }
  REQUIRE(Barrier(tag_id));
}

TEST_CASE("Coherence - fragmented put once read many",
          "[coherence][fragmented]") {
  clio::cte::core::Tag tag("coherence_frag");
  FragmentedRound(tag.GetTagId(), "frag_blob", 0);
}

TEST_CASE("Coherence - repeated fragmented put once read many",
          "[coherence][fragmented][repeated]") {
  clio::cte::core::Tag tag("coherence_frag_rep");
  for (int round = 0; round < kRepeatRounds; ++round) {
    FragmentedRound(tag.GetTagId(), "frag_rep_blob", round);
  }
}

int main(int argc, char *argv[]) {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    fprintf(stderr, "CLIO_INIT failed\n");
    return 2;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    fprintf(stderr, "CTE client init failed\n");
    return 2;
  }
  fprintf(stderr, "coherence: rank %d ready\n", Rank());
  std::string filter = argc > 1 ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);
  if (SimpleTest::g_test_finalize) SimpleTest::g_test_finalize();
  return result;
}
