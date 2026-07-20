/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Multi-node CTE memory-coherency tests: WRITE-ONLY, READ-ONLY, APPEND-ONLY.
 *
 * Unlike test_core_functionality's "Distributed Execution Validation" (a single
 * client on one node doing sequential put/get), EVERY rank here is a client, so
 * different NODES touch the SAME blobs -- which is what coherency actually
 * means. Blobs hash-route by (tag_id, blob_name) to a container that may live on
 * any node (core_runtime.cc:4795), so a blob written on node A is generally
 * stored on node B and must read back identically from node C.
 *
 * Launched by test/integration/distributed_slurm (one rank per node). Ranks
 * coordinate through files on a shared filesystem:
 *   CTE_RANK      - this rank's id (0..N-1)
 *   CTE_NUM_NODES - N
 *   CTE_RUNDIR    - shared dir for barrier files
 *
 * Each rank runs as a client (CLIO_WITH_RUNTIME=0) against its LOCAL daemon.
 */

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

int g_rank = 0;
int g_nnodes = 1;
std::string g_rundir;
bool g_initialized = false;
std::unique_ptr<clio::cte::core::Client> g_client;

constexpr size_t kBlobSize = 4096;
constexpr int kBlobsPerRank = 8;

int EnvInt(const char *k, int dflt) {
  const char *v = std::getenv(k);
  return v ? std::atoi(v) : dflt;
}

/** Deterministic content for global blob index `idx`, verifiable by any rank. */
std::vector<char> MakeData(int idx, size_t size = kBlobSize) {
  std::vector<char> d(size);
  for (size_t j = 0; j < size; ++j)
    d[j] = static_cast<char>((idx * 31 + static_cast<int>(j)) & 0xFF);
  return d;
}

bool FileExists(const std::string &p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

/**
 * File-based cross-rank barrier. Coherency needs ranks SYNCHRONIZED, not merely
 * alive: a reader must not start until every writer has finished, or a "stale
 * read" is indistinguishable from a read that simply raced ahead.
 */
void Barrier(const std::string &name, int timeout_s = 180) {
  if (g_nnodes <= 1 || g_rundir.empty()) return;
  const std::string mine =
      g_rundir + "/bar_" + name + "." + std::to_string(g_rank);
  { std::ofstream f(mine); f << "1\n"; }
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
  while (std::chrono::steady_clock::now() < deadline) {
    int seen = 0;
    for (int r = 0; r < g_nnodes; ++r)
      if (FileExists(g_rundir + "/bar_" + name + "." + std::to_string(r))) ++seen;
    if (seen >= g_nnodes) return;
    std::this_thread::sleep_for(200ms);
  }
  std::fprintf(stderr, "[rank %d] BARRIER '%s' TIMEOUT\n", g_rank, name.c_str());
  REQUIRE(false);
}

void EnsureInit() {
  if (g_initialized) return;
  g_rank = EnvInt("CTE_RANK", 0);
  g_nnodes = EnvInt("CTE_NUM_NODES", 1);
  const char *rd = std::getenv("CTE_RUNDIR");
  g_rundir = rd ? rd : "";
  std::fprintf(stderr, "[COH] rank=%d/%d rundir=%s\n", g_rank, g_nnodes,
               g_rundir.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  // The compose-created CTE core pool (512.0), present on every node.
  g_client = std::make_unique<clio::cte::core::Client>(clio::cte::core::kCtePoolId);
  g_initialized = true;
}

clio::cte::core::TagId GetTag(const std::string &name) {
  auto t = g_client->AsyncGetOrCreateTag(name);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  return t->tag_id_;
}

/** PutBlob `data` at `offset` of blob `name`. */
void PutBlob(clio::cte::core::TagId tag, const std::string &name,
             const std::vector<char> &data, clio::run::u64 offset = 0) {
  auto buf = CLIO_IPC->AllocateBuffer(data.size());
  REQUIRE(!buf.IsNull());
  std::memcpy(buf.ptr_, data.data(), data.size());
  auto p = g_client->AsyncPutBlob(tag, name, offset, data.size(),
                                  buf.shm_.template Cast<void>(), 0.5f,
                                  clio::cte::core::Context(), 0);
  p.Wait();
  REQUIRE(p->GetReturnCode() == 0);
}

/** GetBlob `size` bytes at `offset` of blob `name`. */
std::vector<char> GetBlob(clio::cte::core::TagId tag, const std::string &name,
                          size_t size, clio::run::u64 offset = 0) {
  auto buf = CLIO_IPC->AllocateBuffer(size);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, 0, size);
  auto g = g_client->AsyncGetBlob(tag, name, offset, size,
                                  /*flags=*/0, buf.shm_.template Cast<void>());
  g.Wait();
  REQUIRE(g->GetReturnCode() == 0);
  std::vector<char> out(size);
  std::memcpy(out.data(), buf.ptr_, size);
  return out;
}

}  // namespace

/**
 * WRITE-ONLY: every rank writes a DISJOINT set of blobs concurrently, then all
 * ranks read back the WHOLE set. Verifies that writes issued from every node are
 * globally visible and correct -- no lost or cross-clobbered writes.
 */
TEST_CASE("COHERENCY - write-only: concurrent disjoint writes are globally visible",
          "[cte][core][distributed][coherency]") {
  EnsureInit();
  auto tag = GetTag("coh_write_only");

  for (int i = 0; i < kBlobsPerRank; ++i) {
    int idx = g_rank * kBlobsPerRank + i;
    PutBlob(tag, "wo_b" + std::to_string(idx), MakeData(idx));
  }
  std::fprintf(stderr, "[COH][write-only] rank %d wrote %d blobs\n", g_rank,
               kBlobsPerRank);
  Barrier("wo_written");

  const int total = g_nnodes * kBlobsPerRank;
  for (int idx = 0; idx < total; ++idx) {
    auto got = GetBlob(tag, "wo_b" + std::to_string(idx), kBlobSize);
    REQUIRE(got == MakeData(idx));
  }
  std::fprintf(stderr,
               "[COH][write-only] rank %d verified all %d blobs (written by %d "
               "nodes) -- PASS\n",
               g_rank, total, g_nnodes);
  Barrier("wo_done");
}

/**
 * READ-ONLY: rank 0 writes; every rank then reads the SAME blobs concurrently.
 * Verifies coherent reads of one dataset from all nodes -- no stale or torn data
 * on the nodes that did not write it.
 */
TEST_CASE("COHERENCY - read-only: all nodes read one dataset identically",
          "[cte][core][distributed][coherency]") {
  EnsureInit();
  auto tag = GetTag("coh_read_only");
  const int total = kBlobsPerRank * 2;

  if (g_rank == 0) {
    for (int idx = 0; idx < total; ++idx)
      PutBlob(tag, "ro_b" + std::to_string(idx), MakeData(idx + 1000));
    std::fprintf(stderr, "[COH][read-only] rank 0 wrote %d blobs\n", total);
  }
  Barrier("ro_written");

  for (int idx = 0; idx < total; ++idx) {
    auto got = GetBlob(tag, "ro_b" + std::to_string(idx), kBlobSize);
    REQUIRE(got == MakeData(idx + 1000));
  }
  std::fprintf(stderr,
               "[COH][read-only] rank %d read all %d blobs identically -- PASS\n",
               g_rank, total);
  Barrier("ro_done");
}

/**
 * APPEND-ONLY: all ranks append to ONE shared blob, each owning a disjoint slice
 * at increasing offsets (an append in a blob store is a write at a growing
 * offset). Verifies concurrent partial writes to a single blob assemble
 * coherently rather than clobbering each other.
 */
TEST_CASE("COHERENCY - append-only: concurrent slices assemble into one blob",
          "[cte][core][distributed][coherency]") {
  EnsureInit();
  auto tag = GetTag("coh_append_only");
  const std::string name = "ap_blob";
  const size_t slice = kBlobSize;

  // Each rank appends its own slice at offset rank*slice.
  auto mine = MakeData(g_rank + 2000, slice);
  PutBlob(tag, name, mine, /*offset=*/static_cast<clio::run::u64>(g_rank) * slice);
  std::fprintf(stderr, "[COH][append-only] rank %d appended slice at offset %zu\n",
               g_rank, g_rank * slice);
  Barrier("ap_written");

  // Every rank reads the assembled blob and verifies every rank's slice.
  for (int r = 0; r < g_nnodes; ++r) {
    auto got = GetBlob(tag, name, slice, static_cast<clio::run::u64>(r) * slice);
    REQUIRE(got == MakeData(r + 2000));
  }
  std::fprintf(stderr,
               "[COH][append-only] rank %d verified all %d slices assembled -- "
               "PASS\n",
               g_rank, g_nnodes);
  Barrier("ap_done");
}

SIMPLE_TEST_MAIN()
