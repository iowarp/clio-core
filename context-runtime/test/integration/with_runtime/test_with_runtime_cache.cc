/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * "Clio as a cache" multi-node MPI test (issue #1015).
 *
 * Every rank asks for a runtime — CLIO_WITH_RUNTIME=1, no daemon started ahead
 * of time — and there is no coordination between them beyond MPI itself. On
 * each node exactly one of the four co-resident ranks must WIN and bring the
 * runtime up in-process; the other three must find it and ATTACH as plain
 * clients instead of failing or clobbering the winner's shared memory.
 *
 * Launched as `mpirun -np 8` over a 2-node docker cluster, 4 ranks per node
 * (see docker-compose.yaml + mpi_hostfile in this directory).
 *
 * What each phase asserts:
 *
 *   1. init      Every rank's CLIO_INIT(kClient, with_runtime=true) succeeds.
 *                Before #1015 the racing ranks either failed to bind the
 *                runtime's ports or reaped the winner's memfd segments.
 *   2. one-owner Reduced per node: IsRuntime() is true on exactly ONE rank of
 *                the four. Two runtimes on a port is the failure this issue
 *                exists to prevent, and "all eight attached to nothing" is the
 *                opposite failure — both are caught here.
 *   3. put/get   Each rank PutBlobs a rank-stamped payload under a shared tag,
 *                barrier, then GetBlobs the payload written by its node-local
 *                predecessor — a rank that STARTED the runtime reads what a
 *                rank that ATTACHED wrote and vice versa, so the two bring-up
 *                paths are proven to land in the same runtime. Each rank also
 *                reads its own blob back, and rank 0 additionally reads a blob
 *                written on the other node.
 *
 * Exit code 0 == every rank's checks passed (max-reduced onto rank 0).
 */
#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <clio_cte/core/core_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/manager.h>

namespace {

/** The CTE core pool composed on every node by clio_config.yaml. */
constexpr clio::run::u32 kCorePoolMajor = 512;
/** One blob per rank, small enough that eight of them are cheap. */
constexpr clio::run::u64 kBlobSize = 64u * 1024u;
constexpr const char *kTagName = "/with_runtime/shared_tag";
/** Ranks per node — must match `slots=` in mpi_hostfile. */
constexpr int kRanksPerNode = 4;

/** Rank-stamped payload: proves the reader got THAT writer's bytes. */
unsigned char PayloadByte(int writer_rank, clio::run::u64 i) {
  return static_cast<unsigned char>((i * 131u + 7u + writer_rank * 29u) & 0xFFu);
}

std::string BlobName(int rank) { return "rank" + std::to_string(rank); }

void Log(int rank, const std::string &msg) {
  std::fprintf(stderr, "[with_runtime rank%d] %s\n", rank, msg.c_str());
  std::fflush(stderr);
}

/**
 * Write this rank's blob.
 * @param core the CTE core client
 * @param tag_id the shared tag
 * @param rank this rank
 * @return true on success
 */
bool PutOwnBlob(clio::cte::core::Client *core, clio::cte::core::TagId tag_id,
                int rank) {
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> wbuf = ipc->AllocateBuffer(kBlobSize);
  for (clio::run::u64 j = 0; j < kBlobSize; ++j) {
    wbuf.ptr_[j] = static_cast<char>(PayloadByte(rank, j));
  }
  auto pb = core->AsyncPutBlob(tag_id, BlobName(rank), 0, kBlobSize,
                               wbuf.shm_.template Cast<void>(), -1.0f,
                               clio::cte::core::Context(), 0u,
                               clio::run::PoolQuery::Dynamic());
  pb.Wait();
  const bool ok = pb->GetReturnCode() == 0;
  if (!ok) {
    Log(rank, "FAIL: PutBlob rc=" + std::to_string(pb->GetReturnCode()));
  }
  ipc->FreeBuffer(wbuf);
  return ok;
}

/**
 * Read back the blob a given rank wrote and verify it byte-for-byte.
 * @param core the CTE core client
 * @param tag_id the shared tag
 * @param rank this rank (for logging)
 * @param writer_rank the rank whose blob to read
 * @return true if the payload matched
 */
bool GetAndVerify(clio::cte::core::Client *core, clio::cte::core::TagId tag_id,
                  int rank, int writer_rank) {
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> rbuf = ipc->AllocateBuffer(kBlobSize);
  std::memset(rbuf.ptr_, 0, kBlobSize);
  auto gb = core->AsyncGetBlob(tag_id, BlobName(writer_rank), 0, kBlobSize, 0u,
                               rbuf.shm_.template Cast<void>(),
                               clio::run::PoolQuery::Dynamic());
  gb.Wait();
  if (gb->GetReturnCode() != 0) {
    Log(rank, "FAIL: GetBlob(" + BlobName(writer_rank) +
                  ") rc=" + std::to_string(gb->GetReturnCode()));
    ipc->FreeBuffer(rbuf);
    return false;
  }
  for (clio::run::u64 i = 0; i < kBlobSize; ++i) {
    if (static_cast<unsigned char>(rbuf.ptr_[i]) != PayloadByte(writer_rank, i)) {
      Log(rank, "FAIL: payload mismatch in " + BlobName(writer_rank) +
                    " at byte " + std::to_string(i));
      ipc->FreeBuffer(rbuf);
      return false;
    }
  }
  ipc->FreeBuffer(rbuf);
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  if (nranks % kRanksPerNode != 0) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "test_with_runtime_cache needs a multiple of %d ranks "
                   "(got %d)\n",
                   kRanksPerNode, nranks);
    }
    MPI_Finalize();
    return 64;
  }
  // Ranks are assigned node-by-node (mpi_hostfile gives each host 4 slots), so
  // rank/4 is the node index and rank%4 the rank's slot on that node.
  const int node_id = rank / kRanksPerNode;
  MPI_Comm node_comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, node_id, rank, &node_comm);

  // --- phase 1: every rank asks for a runtime, unsynchronized -------------
  // No barrier before this on purpose: the ranks must race, because the race
  // is what #1015 is about.
  int local_rc = 0;
  const bool inited =
      clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  if (!inited) {
    Log(rank, "FAIL: CLIO_INIT(kClient, with_runtime=true) returned false");
    local_rc = 2;
  }
  const int is_owner =
      (inited && CLIO_RUNTIME_MANAGER->IsRuntime()) ? 1 : 0;
  Log(rank, std::string("init ok=") + (inited ? "1" : "0") +
                " node=" + std::to_string(node_id) +
                " runtime_owner=" + std::to_string(is_owner));

  // --- phase 2: exactly one runtime owner per node ------------------------
  int owners_on_node = 0;
  MPI_Allreduce(&is_owner, &owners_on_node, 1, MPI_INT, MPI_SUM, node_comm);
  if (owners_on_node != 1) {
    Log(rank, "FAIL: node " + std::to_string(node_id) + " has " +
                  std::to_string(owners_on_node) +
                  " runtime owners (expected exactly 1)");
    if (local_rc == 0) local_rc = 3;
  }

  // --- phase 3: PutBlob / GetBlob through the (possibly shared) runtime ---
  std::uint64_t tag_u64 = 0;
  clio::cte::core::Client core;
  if (local_rc == 0) {
    core.Init(clio::run::PoolId(kCorePoolMajor, 0));
    // Rank 0 owns tag creation so every rank agrees on the handle.
    if (rank == 0) {
      auto mk = core.AsyncGetOrCreateTag(kTagName,
                                         clio::cte::core::TagId::GetNull(),
                                         clio::run::PoolQuery::Dynamic());
      mk.Wait();
      if (mk->GetReturnCode() != 0 || mk->tag_id_.IsNull()) {
        Log(0, "FAIL: GetOrCreateTag rc=" +
                   std::to_string(mk->GetReturnCode()));
        local_rc = 4;
      } else {
        tag_u64 = mk->tag_id_.ToU64();
      }
    }
  }
  MPI_Bcast(&tag_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  if (tag_u64 == 0 && local_rc == 0) {
    Log(rank, "FAIL: no shared tag (see rank 0)");
    local_rc = 5;
  }

  if (local_rc == 0) {
    const clio::cte::core::TagId tag_id =
        clio::cte::core::TagId::FromU64(tag_u64);
    if (!PutOwnBlob(&core, tag_id, rank)) {
      local_rc = 6;
    }
    // All writes must land before any read.
    MPI_Barrier(MPI_COMM_WORLD);

    if (local_rc == 0) {
      // Own blob: the simplest round trip through whichever runtime we bound to.
      if (!GetAndVerify(&core, tag_id, rank, rank)) {
        local_rc = 7;
      }
      // Node-local predecessor: a starter reads an attacher's bytes (and, for
      // the rank that started the runtime, vice versa) — the proof that both
      // bring-up paths ended up talking to the SAME runtime.
      const int peer =
          node_id * kRanksPerNode + ((rank + 1) % kRanksPerNode);
      if (local_rc == 0 && !GetAndVerify(&core, tag_id, rank, peer)) {
        local_rc = 8;
      }
      // Rank 0 additionally reads across the node boundary, so the two
      // independently-started runtimes are proven to have formed one cluster.
      if (local_rc == 0 && rank == 0 && nranks > kRanksPerNode) {
        if (!GetAndVerify(&core, tag_id, 0, kRanksPerNode)) {
          local_rc = 9;
        }
      }
    }
    if (local_rc == 0) {
      Log(rank, "PASS");
    }
  } else {
    // Keep the collectives balanced so a failing rank cannot hang the job.
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // A rank that started the runtime must not exit while its node's other ranks
  // are still using it, so everyone leaves together.
  MPI_Barrier(MPI_COMM_WORLD);

  int global_rc = 0;
  MPI_Reduce(&local_rc, &global_rc, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Comm_free(&node_comm);
  MPI_Finalize();
  return (rank == 0) ? global_rc : local_rc;
}
