/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Multi-node CFS (CTE filesystem) producer/consumer test (issue #714).
 *
 * The CFS sits on top of the CTE core, so it inherits the same cross-node gap:
 * a file created+written on node A is visible by PATH from node B (shared tag
 * namespace, #685), but its CONTENT reads back as 0 bytes cross-node (#714).
 * This drives the CFS client directly, one rank per node, using MPI barriers
 * for coordination (never for the data path under test).
 *
 * Launched as `mpirun -np 2` across two containers (see docker-compose here);
 * rank 0 -> node 1, rank 1 -> node 2, each attaching to its OWN local daemon.
 *
 *   Rank 0 (producer, node A):
 *     Open(O_CREAT|O_RDWR); Write(payload); Close
 *     MPI_Barrier            -- publish: the file is written
 *     Getattr(size)          -- own-node sanity
 *
 *   Rank 1 (consumer, node B):
 *     MPI_Barrier            -- wait until the producer has written
 *     Getattr(size)          -- file size must be visible cross-node
 *     Open(O_RDONLY); Read   -- THE #714 CHECK: content must come back intact
 *
 * The consumer's Read is the authoritative cross-node assertion: until #714 is
 * fixed it returns 0 bytes and this test FAILS by design; once cross-node blob
 * transfer works it reads the producer's bytes and passes.
 */
#include <fcntl.h>
#include <mpi.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/filesystem/filesystem_client.h>

namespace {

// The CFS filesystem pool composed on every node in clio_config.yaml.
constexpr clio::run::u32 kCfsPoolMajor = 560;
constexpr clio::run::u64 kFileSize = 1024u * 1024u;  // 1 MiB
constexpr const char *kSharedPath = "/mpi_pc/cfs_shared.bin";

int EnvSecs(const char *name, int def) {
  const char *e = std::getenv(name);
  if (!e || !*e) return def;
  int v = std::atoi(e);
  return v > 0 ? v : def;
}

unsigned char PayloadByte(clio::run::u64 i) {
  return static_cast<unsigned char>((i * 197u + 23u) & 0xFFu);
}

void Log(int rank, const char *role, const std::string &msg) {
  std::fprintf(stderr, "[cfs-pc rank%d %s] %s\n", rank, role, msg.c_str());
  std::fflush(stderr);
}

bool AttachClient(int rank, const char *role) {
  setenv("CLIO_WITH_RUNTIME", "0", 1);
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    Log(rank, role, "FAIL: CLIO_INIT(kClient) failed");
    return false;
  }
  return true;
}

// ---- rank 0: producer -----------------------------------------------------
int RunProducer() {
  const char *role = "producer";
  if (!AttachClient(0, role)) {
    MPI_Barrier(MPI_COMM_WORLD);
    return 2;
  }
  clio::cte::filesystem::Client cfs;
  cfs.Init(clio::run::PoolId(kCfsPoolMajor, 0));
  auto *ipc = CLIO_IPC;
  int failed = 0;

  auto open = cfs.AsyncOpen(kSharedPath, O_CREAT | O_RDWR, 0644);
  open.Wait();
  clio::run::u64 h = open->handle_;
  if (open->GetReturnCode() != 0 || h == 0) {
    Log(0, role, "FAIL: Open(create) rc=" + std::to_string(open->GetReturnCode()));
    failed = 1;
  }

  if (!failed) {
    ctp::ipc::FullPtr<char> wbuf = ipc->AllocateBuffer(kFileSize);
    for (clio::run::u64 j = 0; j < kFileSize; ++j) wbuf.ptr_[j] = PayloadByte(j);
    auto w = cfs.AsyncWrite(h, 0, kFileSize, wbuf.shm_.template Cast<void>());
    w.Wait();
    if (w->GetReturnCode() != 0 || w->bytes_written_ != kFileSize) {
      Log(0, role, "FAIL: Write rc=" + std::to_string(w->GetReturnCode()) +
                       " bytes=" + std::to_string(w->bytes_written_));
      failed = 1;
    }
    ipc->FreeBuffer(wbuf);
    auto cl = cfs.AsyncClose(h);
    cl.Wait();
  }

  // Publish "file written".
  MPI_Barrier(MPI_COMM_WORLD);

  // Own-node sanity: the producer must see the exact logical size.
  if (!failed) {
    auto ga = cfs.AsyncGetattr(kSharedPath);
    ga.Wait();
    Log(0, role, "local view exists=" + std::to_string(ga->exists_) +
                     " size=" + std::to_string(ga->size_));
    if (ga->GetReturnCode() != 0 || ga->size_ != kFileSize) {
      Log(0, role, "FAIL: producer-local Getattr size mismatch");
      failed = 1;
    }
  }

  // Keep this node's daemon alive through the consumer's read window.
  std::this_thread::sleep_for(std::chrono::seconds(EnvSecs("PC_KEEPALIVE_SEC", 30)));
  return failed ? 4 : 0;
}

// ---- rank 1: consumer -----------------------------------------------------
int RunConsumer() {
  const char *role = "consumer";
  bool attached = AttachClient(1, role);
  MPI_Barrier(MPI_COMM_WORLD);
  if (!attached) return 3;

  clio::cte::filesystem::Client cfs;
  cfs.Init(clio::run::PoolId(kCfsPoolMajor, 0));
  auto *ipc = CLIO_IPC;

  // File metadata (size) must be visible cross-node by path (post-#685).
  auto ga = cfs.AsyncGetattr(kSharedPath);
  ga.Wait();
  Log(1, role, "cross-node view exists=" + std::to_string(ga->exists_) +
                   " size=" + std::to_string(ga->size_));
  if (ga->GetReturnCode() != 0 || ga->exists_ != 1 || ga->size_ != kFileSize) {
    Log(1, role, "FAIL: cross-node Getattr rc=" +
                     std::to_string(ga->GetReturnCode()) + " exists=" +
                     std::to_string(ga->exists_) + " size=" +
                     std::to_string(ga->size_) + " (expected " +
                     std::to_string(kFileSize) + ")");
    return 6;
  }

  // THE #714 CHECK: the content written on the producer node must read back
  // byte-for-byte here. On the current (buggy) runtime this returns 0 bytes.
  auto open = cfs.AsyncOpen(kSharedPath, O_RDONLY, 0644);
  open.Wait();
  clio::run::u64 h = open->handle_;
  if (open->GetReturnCode() != 0 || h == 0) {
    Log(1, role, "FAIL: cross-node Open(RDONLY) rc=" +
                     std::to_string(open->GetReturnCode()));
    return 7;
  }
  ctp::ipc::FullPtr<char> rbuf = ipc->AllocateBuffer(kFileSize);
  std::memset(rbuf.ptr_, 0, kFileSize);
  auto r = cfs.AsyncRead(h, 0, kFileSize, rbuf.shm_.template Cast<void>());
  r.Wait();
  auto cl = cfs.AsyncClose(h);
  cl.Wait();
  if (r->GetReturnCode() != 0 || r->bytes_read_ != kFileSize) {
    Log(1, role, "FAIL: cross-node Read rc=" +
                     std::to_string(r->GetReturnCode()) + " bytes_read=" +
                     std::to_string(r->bytes_read_) + " (expected " +
                     std::to_string(kFileSize) +
                     ") — file content is node-local (#714)");
    ipc->FreeBuffer(rbuf);
    return 8;
  }
  for (clio::run::u64 i = 0; i < kFileSize; ++i) {
    if (static_cast<unsigned char>(rbuf.ptr_[i]) != PayloadByte(i)) {
      Log(1, role, "FAIL: cross-node content mismatch at " + std::to_string(i));
      ipc->FreeBuffer(rbuf);
      return 9;
    }
  }
  ipc->FreeBuffer(rbuf);
  Log(1, role, "PASS: cross-node Read of " + std::to_string(kFileSize) +
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
                   "test_cfs_producer_consumer requires exactly 2 ranks "
                   "(got %d)\n",
                   size);
    }
    MPI_Finalize();
    return 64;
  }

  int local_rc = (rank == 0) ? RunProducer() : RunConsumer();
  int global_rc = 0;
  MPI_Reduce(&local_rc, &global_rc, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return (rank == 0) ? global_rc : local_rc;
}
