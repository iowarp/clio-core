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
 * Cross-node coherence of the shm write-through cache. Every node runs this
 * binary with its client bound to the cache pool (CLIO_CTE_POOL=565.0), so
 * puts and covered gets complete in each node's own segment; the scenarios
 * check that what one node writes is what every other node reads once the
 * writer has synced, including after other nodes cached the old bytes.
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

using clio::cte::core::TagId;
using clio::run::u64;

/** Cluster size (CTE_NUM_NODES, default 4). */
int NumNodes() {
  const char *e = std::getenv("CTE_NUM_NODES");
  const int v = (e != nullptr && *e) ? std::atoi(e) : 4;
  return v > 0 ? v : 4;
}

/** This node's rank (NODE_ID is 1-based). */
int Rank() {
  const char *env = std::getenv("NODE_ID");
  return env != nullptr ? std::atoi(env) - 1 : 0;
}

/** The process client, bound to the cache pool. */
clio::cte::core::Client *Io() { return CLIO_CTE_CLIENT; }

/** Deterministic bytes for (writer, round). */
std::string Pattern(u64 size, int writer, int round) {
  std::string v(size, '\0');
  for (u64 i = 0; i < size; ++i) {
    v[i] = static_cast<char>((writer * 31 + round * 7 + i * 13) & 0x7f);
  }
  return v;
}

/** Make this node's buffered writes durable and visible (fsync). */
bool Sync(const TagId &tag) {
  auto s = Io()->AsyncSyncTag(tag);
  s.Wait();
  return s->GetReturnCode() == 0;
}

int g_epoch = 0;
/**
 * Cluster barrier over blobs: each rank publishes a marker and syncs it, then
 * waits until every rank's marker is visible.
 */
bool Barrier(const TagId &tag) {
  const int epoch = g_epoch++;
  char one = 1;
  const std::string mine =
      "barrier_" + std::to_string(epoch) + "_" + std::to_string(Rank());
  auto put = Io()->AsyncPutBlob(tag, mine, 0, 1, &one);
  put.Wait();
  if (put->GetReturnCode() != 0 || !Sync(tag)) return false;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  for (int r = 0; r < NumNodes(); ++r) {
    const std::string name =
        "barrier_" + std::to_string(epoch) + "_" + std::to_string(r);
    while (true) {
      auto sz = Io()->AsyncGetBlobSize(tag, name);
      sz.Wait();
      if (sz->GetReturnCode() == 0 && sz->size_ == 1) break;
      if (std::chrono::steady_clock::now() > deadline) {
        fprintf(stderr, "[BARRIER-TIMEOUT] rank=%d epoch=%d waiting_on=%s\n",
                Rank(), epoch, name.c_str());
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
  return true;
}

/** Read a whole blob through the cache. */
bool Read(const TagId &tag, const std::string &name, std::string *out) {
  auto get = Io()->AsyncGetBlob(tag, name, 0, out->size(), 0, out->data());
  get.Wait();
  return get->GetReturnCode() == 0;
}

/** Shared tag (every rank resolves the same id). */
TagId SharedTag(const std::string &name) {
  auto t = Io()->AsyncGetOrCreateTag(name);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  return t->tag_id_;
}

}  // namespace

TEST_CASE("ShmWtc coherence - every node reads every node's writes",
          "[shm_wtc][coherence]") {
  REQUIRE(Io()->HasWtc());
  const TagId tag = SharedTag("wtc_coh_porm");
  const int rank = Rank();
  const u64 size = 256 * 1024;
  const std::string mine = Pattern(size, rank, 0);
  auto put = Io()->AsyncPutBlob(tag, "rank_" + std::to_string(rank), 0, size,
                                mine.data());
  put.Wait();
  REQUIRE(put->GetReturnCode() == 0);
  REQUIRE(Sync(tag));
  REQUIRE(Barrier(tag));
  for (int r = 0; r < NumNodes(); ++r) {
    std::string back(size, '\0');
    REQUIRE(Read(tag, "rank_" + std::to_string(r), &back));
    REQUIRE(back == Pattern(size, r, 0));
    // Twice: the second read is served from this node's cache.
    std::string again(size, '\0');
    REQUIRE(Read(tag, "rank_" + std::to_string(r), &again));
    REQUIRE(again == back);
  }
  REQUIRE(Barrier(tag));
}

TEST_CASE("ShmWtc coherence - an overwrite invalidates other nodes' copies",
          "[shm_wtc][coherence][invalidate]") {
  const TagId tag = SharedTag("wtc_coh_inval");
  const int rank = Rank();
  const u64 size = 128 * 1024;
  if (rank == 0) {
    const std::string v0 = Pattern(size, 0, 0);
    auto p = Io()->AsyncPutBlob(tag, "hot", 0, size, v0.data());
    p.Wait();
    REQUIRE(p->GetReturnCode() == 0);
    REQUIRE(Sync(tag));
  }
  REQUIRE(Barrier(tag));
  // Every node caches round 0.
  std::string back(size, '\0');
  REQUIRE(Read(tag, "hot", &back));
  REQUIRE(back == Pattern(size, 0, 0));
  REQUIRE(Barrier(tag));
  for (int round = 1; round <= 3; ++round) {
    const int writer = round % NumNodes();
    if (rank == writer) {
      const std::string v = Pattern(size, writer, round);
      auto p = Io()->AsyncPutBlob(tag, "hot", 0, size, v.data());
      p.Wait();
      REQUIRE(p->GetReturnCode() == 0);
      REQUIRE(Sync(tag));
    }
    REQUIRE(Barrier(tag));
    std::string now(size, '\0');
    REQUIRE(Read(tag, "hot", &now));
    if (now != Pattern(size, writer, round)) {
      fprintf(stderr, "[STALE] rank=%d round=%d writer=%d\n", rank, round,
              writer);
    }
    REQUIRE(now == Pattern(size, writer, round));
    REQUIRE(Barrier(tag));
  }
}

TEST_CASE("ShmWtc coherence - N-1 writers to disjoint ranges of one blob",
          "[shm_wtc][coherence][n1]") {
  const TagId tag = SharedTag("wtc_coh_n1");
  const int rank = Rank();
  const u64 seg = 64 * 1024;
  const int n = NumNodes();
  // Each rank writes its segment in 4 KB pieces (many small writes).
  const std::string mine = Pattern(seg, rank, 1);
  for (u64 off = 0; off < seg; off += 4096) {
    auto p = Io()->AsyncPutBlob(tag, "shared", rank * seg + off, 4096,
                                mine.data() + off);
    p.Wait();
    REQUIRE(p->GetReturnCode() == 0);
  }
  REQUIRE(Sync(tag));
  REQUIRE(Barrier(tag));
  std::string all(seg * n, '\0');
  REQUIRE(Read(tag, "shared", &all));
  for (int r = 0; r < n; ++r) {
    REQUIRE(all.compare(r * seg, seg, Pattern(seg, r, 1)) == 0);
  }
  REQUIRE(Barrier(tag));
}

TEST_CASE("ShmWtc coherence - a delete is seen everywhere",
          "[shm_wtc][coherence][del]") {
  const TagId tag = SharedTag("wtc_coh_del");
  const int rank = Rank();
  const u64 size = 32 * 1024;
  if (rank == 0) {
    const std::string v = Pattern(size, 0, 2);
    auto p = Io()->AsyncPutBlob(tag, "doomed", 0, size, v.data());
    p.Wait();
    REQUIRE(p->GetReturnCode() == 0);
    REQUIRE(Sync(tag));
  }
  REQUIRE(Barrier(tag));
  std::string back(size, '\0');
  REQUIRE(Read(tag, "doomed", &back));  // every node caches it
  REQUIRE(Barrier(tag));
  if (rank == 0) {
    auto d = Io()->AsyncDelBlob(tag, "doomed");
    d.Wait();
    REQUIRE(d->GetReturnCode() == 0);
  }
  REQUIRE(Barrier(tag));
  auto sz = Io()->AsyncGetBlobSize(tag, "doomed");
  sz.Wait();
  REQUIRE(sz->GetReturnCode() != 0);
  REQUIRE(Barrier(tag));
}

int main(int argc, char *argv[]) {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    fprintf(stderr, "CLIO_INIT failed
");
    return 2;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    fprintf(stderr, "CTE client init failed
");
    return 2;
  }
  fprintf(stderr, "shm_wtc coherence: rank %d ready (cache %s)
", Rank(),
          Io()->HasWtc() ? "attached" : "NOT attached");
  std::string filter = argc > 1 ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);
  if (SimpleTest::g_test_finalize) SimpleTest::g_test_finalize();
  return result;
}
