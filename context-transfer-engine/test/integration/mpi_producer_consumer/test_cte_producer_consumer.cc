/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Multi-node CTE-core producer/consumer test (issue #714).
 *
 * Reproduces the cross-node blob-data gap: after #685 the CTE tag namespace is
 * cluster-wide (a tag created on node A is visible from node B), but a blob's
 * PAYLOAD written on node A is not retrievable from node B (a cross-node
 * GetBlob returns 0 bytes). This test drives that path directly with the CTE
 * core client, one rank per node, synchronizing with MPI barriers (MPI is used
 * ONLY for cross-rank coordination and to hand the tag id to the consumer via
 * distributed memory — never for the data movement under test).
 *
 * Launched as `mpirun -np 2` across two containers (see docker-compose in this
 * directory); rank 0 lands on node 1, rank 1 on node 2, and each attaches to
 * its OWN local daemon (CLIO_INIT(kClient), CLIO_WITH_RUNTIME=0).
 *
 *   Rank 0 (producer, node A):
 *     PutBlob("0")            -- write the payload (Dynamic-routed)
 *     MPI_Bcast(tag_id)       -- share the tag handle with the consumer
 *     MPI_Barrier             -- publish: the blob is written
 *     GetTagSize() / GetBlobSize("0")  -- own-node sanity
 *
 *   Rank 1 (consumer, node B):
 *     MPI_Bcast(tag_id)       -- receive the tag handle
 *     MPI_Barrier             -- wait until the producer has written
 *     GetTagSize()            -- tag metadata must be visible cross-node (#685)
 *     GetBlobSize("0")        -- blob size must be visible cross-node
 *     GetBlob("0")            -- THE #714 CHECK: payload must come back intact
 *
 * The consumer's GetBlob is the authoritative cross-node assertion. Until #714
 * is fixed it returns 0 bytes and this test FAILS by design (a reproducer); once
 * cross-node blob transfer works it reads the producer's bytes and passes.
 *
 * Exit code 0 == all ranks' checks passed (reduced to rank 0).
 */
#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

namespace {

// The CTE core pool composed on every node in clio_config.yaml.
constexpr clio::run::u32 kCorePoolMajor = 512;
// Shared, deterministic payload so the consumer proves it read the PRODUCER's
// bytes cross-node (not zeros/garbage). One blob named "0" under a shared tag.
constexpr clio::run::u64 kBlobSize = 1024u * 1024u;  // 1 MiB
constexpr const char *kTagName = "/mpi_pc/cte_shared_tag";
constexpr const char *kBlobName = "0";

int EnvSecs(const char *name, int def) {
  const char *e = std::getenv(name);
  if (!e || !*e) return def;
  int v = std::atoi(e);
  return v > 0 ? v : def;
}

unsigned char PayloadByte(clio::run::u64 i) {
  return static_cast<unsigned char>((i * 131u + 7u) & 0xFFu);
}

void Log(int rank, const char *role, const std::string &msg) {
  std::fprintf(stderr, "[cte-pc rank%d %s] %s\n", rank, role, msg.c_str());
  std::fflush(stderr);
}

// Attach to this node's local daemon as a client (never spawn a runtime).
bool AttachClient(int rank, const char *role) {
  setenv("CLIO_WITH_RUNTIME", "0", 1);
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    Log(rank, role, "FAIL: CLIO_INIT(kClient) failed");
    return false;
  }
  return true;
}

// ---- rank 0: producer -----------------------------------------------------
// Writes the blob, hands the tag id to the consumer, then checks its own view.
int RunProducer() {
  const char *role = "producer";
  if (!AttachClient(0, role)) {
    // Still participate in the collectives so the consumer does not hang.
    std::uint64_t z = 0;
    MPI_Bcast(&z, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return 2;
  }
  clio::cte::core::Client core;
  core.Init(clio::run::PoolId(kCorePoolMajor, 0));
  auto *ipc = CLIO_IPC;

  // Create the shared tag (cluster-wide namespace) and PutBlob the payload.
  auto mk = core.AsyncGetOrCreateTag(kTagName, clio::cte::core::TagId::GetNull(),
                                     clio::run::PoolQuery::Dynamic());
  mk.Wait();
  clio::cte::core::TagId tag_id = mk->tag_id_;
  int failed = 0;
  if (mk->GetReturnCode() != 0 || tag_id.IsNull()) {
    Log(0, role, "FAIL: GetOrCreateTag rc=" + std::to_string(mk->GetReturnCode()));
    failed = 1;
  }

  if (!failed) {
    ctp::ipc::FullPtr<char> wbuf = ipc->AllocateBuffer(kBlobSize);
    for (clio::run::u64 j = 0; j < kBlobSize; ++j) wbuf.ptr_[j] = PayloadByte(j);
    auto pb = core.AsyncPutBlob(tag_id, kBlobName, 0, kBlobSize,
                                wbuf.shm_.template Cast<void>(), -1.0f,
                                clio::cte::core::Context(), 0u,
                                clio::run::PoolQuery::Dynamic());
    pb.Wait();
    if (pb->GetReturnCode() != 0) {
      Log(0, role, "FAIL: PutBlob rc=" + std::to_string(pb->GetReturnCode()));
      failed = 1;
    }
    ipc->FreeBuffer(wbuf);
  }

  // Hand the tag id to the consumer (distributed-memory share), then publish
  // "blob written" via the barrier.
  std::uint64_t tag_u64 = failed ? 0 : tag_id.ToU64();
  MPI_Bcast(&tag_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  // Own-node sanity: the producer must see its own tag/blob sizes.
  if (!failed) {
    auto ts = core.AsyncGetTagSize(tag_id, clio::run::PoolQuery::Dynamic());
    ts.Wait();
    auto bs = core.AsyncGetBlobSize(tag_id, kBlobName,
                                    clio::run::PoolQuery::Dynamic());
    bs.Wait();
    Log(0, role, "local view tag_size=" + std::to_string(ts->tag_size_) +
                     " blob_size=" + std::to_string(bs->size_));
    if (bs->GetReturnCode() != 0 || bs->size_ != kBlobSize) {
      Log(0, role, "FAIL: producer-local GetBlobSize mismatch");
      failed = 1;
    }
  }

  // Stay alive so this node's daemon (owner of the blob's container) survives
  // the consumer's read window.
  int keepalive = EnvSecs("PC_KEEPALIVE_SEC", 30);
  std::this_thread::sleep_for(std::chrono::seconds(keepalive));
  return failed ? 4 : 0;
}

// ---- rank 1: consumer -----------------------------------------------------
// Receives the tag id and must read the producer's payload cross-node.
int RunConsumer() {
  const char *role = "consumer";
  bool attached = AttachClient(1, role);

  std::uint64_t tag_u64 = 0;
  MPI_Bcast(&tag_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!attached) return 3;
  if (tag_u64 == 0) {
    Log(1, role, "FAIL: producer reported no tag (see producer log)");
    return 5;
  }
  clio::cte::core::Client core;
  core.Init(clio::run::PoolId(kCorePoolMajor, 0));
  auto *ipc = CLIO_IPC;
  clio::cte::core::TagId tag_id = clio::cte::core::TagId::FromU64(tag_u64);

  // Tag metadata must be visible cross-node (this already works post-#685).
  auto ts = core.AsyncGetTagSize(tag_id, clio::run::PoolQuery::Dynamic());
  ts.Wait();
  auto bs = core.AsyncGetBlobSize(tag_id, kBlobName,
                                  clio::run::PoolQuery::Dynamic());
  bs.Wait();
  Log(1, role, "cross-node view tag_size=" + std::to_string(ts->tag_size_) +
                   " blob_size=" + std::to_string(bs->size_));
  if (bs->GetReturnCode() != 0 || bs->size_ != kBlobSize) {
    Log(1, role, "FAIL: cross-node GetBlobSize rc=" +
                     std::to_string(bs->GetReturnCode()) + " size=" +
                     std::to_string(bs->size_) + " (expected " +
                     std::to_string(kBlobSize) + ")");
    return 6;
  }

  // THE #714 CHECK: the payload written on the producer node must come back
  // byte-for-byte here. On the current (buggy) runtime this returns 0 bytes.
  ctp::ipc::FullPtr<char> rbuf = ipc->AllocateBuffer(kBlobSize);
  std::memset(rbuf.ptr_, 0, kBlobSize);
  auto gb = core.AsyncGetBlob(tag_id, kBlobName, 0, kBlobSize, 0u,
                              rbuf.shm_.template Cast<void>(),
                              clio::run::PoolQuery::Dynamic());
  gb.Wait();
  // GetBlob is all-or-nothing: rc==0 means all kBlobSize bytes landed in rbuf;
  // a short/failed cross-node read reports rc!=0 (GetBlobTask has no separate
  // bytes-read field). The byte-for-byte check below is the payload assertion —
  // it also catches the #714 "reads zeros" symptom even if rc were 0.
  if (gb->GetReturnCode() != 0) {
    Log(1, role, "FAIL: cross-node GetBlob rc=" +
                     std::to_string(gb->GetReturnCode()) + " (expected " +
                     std::to_string(kBlobSize) +
                     " bytes) — blob payload is node-local (#714)");
    ipc->FreeBuffer(rbuf);
    return 7;
  }
  for (clio::run::u64 i = 0; i < kBlobSize; ++i) {
    if (static_cast<unsigned char>(rbuf.ptr_[i]) != PayloadByte(i)) {
      Log(1, role, "FAIL: cross-node payload mismatch at " + std::to_string(i));
      ipc->FreeBuffer(rbuf);
      return 8;
    }
  }
  ipc->FreeBuffer(rbuf);
  Log(1, role, "PASS: cross-node GetBlob of " + std::to_string(kBlobSize) +
                   " bytes matched the producer");
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "test_cte_producer_consumer requires exactly 2 ranks "
                   "(got %d)\n",
                   size);
    }
    MPI_Finalize();
    return 64;
  }

  int local_rc = (rank == 0) ? RunProducer() : RunConsumer();

  // Reduce every rank's result: the job passes only if all ranks passed. The
  // consumer (rank 1) carries the authoritative cross-node signal.
  int global_rc = 0;
  MPI_Reduce(&local_rc, &global_rc, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return (rank == 0) ? global_rc : local_rc;
}
