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
 * clio_cte_safe_bench.cc
 *
 * Benchmark that:
 *  1. Attaches to a running CLIO runtime (clio_run start).
 *  2. Creates 5 RAM bdevs and a SAFE bdev over 4 of them.
 *  3. Registers the SAFE bdev as a CTE storage target (attach_existing=1).
 *  4. Issues PutBlob I/O against CTE while a background thread performs
 *     a live migration: adds the 5th RAM bdev and removes the 1st.
 *  5. Reports throughput before, during, and after the migration event.
 */

#include "bench_common.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <mpi.h>

#include "clio_runtime/admin/admin_client.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/safe_bdev/safe_bdev_client.h"
#include "clio_cte/core/core_client.h"
#include "clio_ctp/util/logging.h"

using namespace std::chrono;

// ---------------------------------------------------------------------------
// Setup helpers
// ---------------------------------------------------------------------------

/** @brief Create one RAM bdev and return its pool ID.
 *  @param bdev_client  Reference to bdev client.
 *  @param name         Unique pool name.
 *  @param major        PoolId major number.
 *  @param size_bytes   Capacity of the RAM bdev in bytes.
 *  @return Assigned PoolId on success; aborts via MPI_Abort on failure.
 */
static clio::run::PoolId CreateRamBdev(clio::run::bdev::Client &bdev_client,
                                       const std::string &name,
                                       clio::run::u32 major,
                                       clio::run::u64 size_bytes) {
  clio::run::PoolId id(major, 0);
  auto t = bdev_client.AsyncCreate(clio::run::PoolQuery::Dynamic(), name, id,
                                   clio::run::bdev::BdevType::kRam,
                                   size_bytes);
  t.Wait();
  if (t->return_code_.load() != 0) {
    HLOG(kError, "Failed to create RAM bdev '{}' (rc={})", name,
         t->return_code_.load());
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  HLOG(kInfo, "Created RAM bdev '{}' id={}.{}", name, major, 0);
  return id;
}

/** @brief Run the migration: add spare bdev, remove one active member.
 *  @param safe_client  Reference to safe_bdev client.
 *  @param safe_pool_id PoolId of the SAFE bdev.
 *  @param spare_name   Name of the spare RAM bdev.
 *  @param spare_id     PoolId of the spare RAM bdev.
 *  @param remove_major Major of the member to remove.
 */
static void RunMigration(clio::run::safe_bdev::Client &safe_client,
                         const clio::run::PoolId &safe_pool_id,
                         const std::string &spare_name,
                         const clio::run::PoolId &spare_id,
                         clio::run::u32 remove_major) {
  HLOG(kInfo, "=== MIGRATION START: adding spare '{}' ===", spare_name);
  auto t1 = safe_client.AsyncAddBdev(clio::run::PoolQuery::Dynamic(),
                                     spare_name, 0, spare_id, 0);
  t1.Wait();
  HLOG(kInfo, "Migration AddBdev rc={}", t1->return_code_.load());

  clio::run::PoolId remove_id(remove_major, 0);
  auto t2 = safe_client.AsyncRemoveBdev(clio::run::PoolQuery::Dynamic(),
                                        remove_id, 0 /* not faulty */);
  t2.Wait();
  HLOG(kInfo, "Migration RemoveBdev rc={}", t2->return_code_.load());
  HLOG(kInfo, "=== MIGRATION COMPLETE ===");
}

// ---------------------------------------------------------------------------
// I/O workload
// ---------------------------------------------------------------------------

/** @brief Run PutBlob I/O loop for a fixed number of operations.
 *  @param cte_client   Reference to CTE core client.
 *  @param tag_id       TagId under which blobs are stored.
 *  @param buf_shm      Pre-allocated SHM buffer to use as payload.
 *  @param io_size      Bytes per operation.
 *  @param num_ops      Total number of PutBlob operations to issue.
 *  @param label        Human-readable label for timing output.
 *  @return Elapsed microseconds.
 */
static long long RunPutOps(clio::cte::core::Client *cte_client,
                            clio::cte::core::TagId tag_id,
                            const ctp::ipc::ShmPtr<> &buf_shm,
                            clio_bench::u64 io_size, int num_ops,
                            const std::string &label) {
  auto start = steady_clock::now();
  for (int i = 0; i < num_ops; ++i) {
    auto t = cte_client->AsyncPutBlob(tag_id, label + "_" + std::to_string(i),
                                      0, io_size, buf_shm, 0.8f,
                                      clio::cte::core::Context(), 0,
                                      clio::run::PoolQuery::Dynamic());
    t.Wait();
    if (t->return_code_.load() != 0) {
      HLOG(kError, "PutBlob failed at op {} (rc={})", i,
           t->return_code_.load());
      break;
    }
  }
  auto end = steady_clock::now();
  return duration_cast<microseconds>(end - start).count();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  clio_bench::BenchArgs args = clio_bench::ParseBenchArgs(argc, argv);
  if (!args.ok) {
    MPI_Finalize();
    return 1;
  }

  // --- Runtime init (client mode — server must already be running) ----------
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "CLIO_INIT failed");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  struct FinalizeGuard {
    ~FinalizeGuard() {
      auto *mgr = CLIO_RUNTIME_MANAGER;
      if (mgr) mgr->ClientFinalize();
      MPI_Finalize();
    }
  } guard;

  std::this_thread::sleep_for(milliseconds(500));

  // CLIO_CTE_CLIENT_INIT looks up the already-running CTE pool (cte_main)
  // and wires up the global CTE client. We do NOT create a new CTE pool.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    HLOG(kError, "CLIO_CTE_CLIENT_INIT failed");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::this_thread::sleep_for(milliseconds(200));

  // Use the global CTE client initialized by CLIO_CTE_CLIENT_INIT
  auto *cte_client = CLIO_CTE_CLIENT;

  // Only rank 0 creates the bdevs & safe_bdev
  clio::run::bdev::Client bdev_client;
  clio::run::safe_bdev::Client safe_client;

  // PoolId space: majors 2100-2105 for RAM bdevs, 2200 for safe_bdev
  const clio::run::u32 kBdevBase = 2100;
  const clio::run::u32 kSafeMajor = 2200;
  const clio::run::u64 kBdevSize = 1ULL << 30;  // 1 GiB each
  const clio::run::u64 kSafeTotal = 4ULL << 30; // 4 GiB aggregate
  const std::string kSafeName = "bench_safe_bdev";
  clio::run::PoolId safe_pool_id(kSafeMajor, 0);

  std::vector<clio::run::PoolId> member_ids;

  if (world_rank == 0) {
    // Create 5 RAM bdevs
    for (int m = 0; m < 5; ++m) {
      std::string name = "bench_ram_" + std::to_string(m);
      auto id = CreateRamBdev(bdev_client, name, kBdevBase + m, kBdevSize);
      member_ids.push_back(id);
    }

    // Create SAFE bdev over first 4 members (1 parity / failure tolerance)
    std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
    for (int m = 0; m < 4; ++m) {
      std::string name = "bench_ram_" + std::to_string(m);
      members.emplace_back(name, 0, member_ids[m]);
    }
    auto sct = safe_client.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), kSafeName, safe_pool_id,
        1 /* max_failures */, members, "/tmp/cte_safe_bench.alog");
    sct.Wait();
    if (sct->return_code_.load() != 0) {
      HLOG(kError, "SafeBdev Create failed (rc={})", sct->return_code_.load());
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
    HLOG(kInfo, "SAFE bdev '{}' created ({}.{})", kSafeName, kSafeMajor, 0);

    // Register SAFE bdev as a CTE target (attach_existing=1 means CTE uses
    // the existing safe_pool_id pool rather than creating a new bdev)
    auto rt = cte_client->AsyncRegisterTarget(
        kSafeName, clio::run::bdev::BdevType::kFile, kSafeTotal,
        clio::run::PoolQuery::Dynamic(), safe_pool_id,
        clio::run::PoolQuery::Dynamic());
    rt.Wait();
    if (rt->return_code_.load() != 0) {
      HLOG(kError, "RegisterTarget failed (rc={})", rt->return_code_.load());
      MPI_Abort(MPI_COMM_WORLD, 4);
    }
    HLOG(kInfo, "SAFE bdev registered as CTE target '{}'", kSafeName);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // --- Shared memory buffer -------------------------------------------------
  auto *ipc = CLIO_CPU_IPC;
  auto write_buf = ipc->AllocateBuffer(args.io_size);
  std::memset(write_buf.ptr_, 0xAB, args.io_size);
  auto buf_shm = write_buf.shm_.template Cast<void>();

  // --- Create tag -----------------------------------------------------------
  auto tag_t = cte_client->AsyncGetOrCreateTag("/bench_safe/workload");
  tag_t.Wait();
  clio::cte::core::TagId tag_id = tag_t->tag_id_;

  // --- Phase 1: baseline ops before migration --------------------------------
  int phase_ops = args.io_count / 3;
  HLOG(kInfo, "[Phase 1] Baseline: {} ops x {} bytes", phase_ops,
       args.io_size);
  long long us1 = RunPutOps(cte_client, tag_id, buf_shm,
                             args.io_size, phase_ops, "pre");
  double mb1 = static_cast<double>(phase_ops * args.io_size) / (1 << 20);
  HLOG(kInfo, "[Phase 1] {:.2f} MB in {:.0f} ms -> {:.2f} MB/s",
       mb1, us1 / 1000.0, mb1 / (us1 / 1e6));

  // --- Trigger migration on rank 0 in background ----------------------------
  std::thread migration_thread;
  if (world_rank == 0) {
    migration_thread = std::thread([&]() {
      RunMigration(safe_client, safe_pool_id,
                   "bench_ram_4", member_ids[4], kBdevBase + 0);
    });
  }

  // --- Phase 2: I/O during migration ----------------------------------------
  HLOG(kInfo, "[Phase 2] During migration: {} ops x {} bytes", phase_ops,
       args.io_size);
  long long us2 = RunPutOps(cte_client, tag_id, buf_shm,
                             args.io_size, phase_ops, "mid");
  double mb2 = static_cast<double>(phase_ops * args.io_size) / (1 << 20);
  HLOG(kInfo, "[Phase 2] {:.2f} MB in {:.0f} ms -> {:.2f} MB/s",
       mb2, us2 / 1000.0, mb2 / (us2 / 1e6));

  if (world_rank == 0 && migration_thread.joinable()) {
    migration_thread.join();
  }

  // --- Phase 3: post-migration ops ------------------------------------------
  HLOG(kInfo, "[Phase 3] Post-migration: {} ops x {} bytes", phase_ops,
       args.io_size);
  long long us3 = RunPutOps(cte_client, tag_id, buf_shm,
                             args.io_size, phase_ops, "post");
  double mb3 = static_cast<double>(phase_ops * args.io_size) / (1 << 20);
  HLOG(kInfo, "[Phase 3] {:.2f} MB in {:.0f} ms -> {:.2f} MB/s",
       mb3, us3 / 1000.0, mb3 / (us3 / 1e6));

  ipc->FreeBuffer(write_buf);

  // --- Summary (rank 0 only) ------------------------------------------------
  if (world_rank == 0) {
    double total_mb = mb1 + mb2 + mb3;
    long long total_us = us1 + us2 + us3;
    HLOG(kInfo, "=== SUMMARY ===");
    HLOG(kInfo, "  Pre-migration  BW: {:.2f} MB/s ({:.0f} ms)", mb1 / (us1 / 1e6), us1 / 1000.0);
    HLOG(kInfo, "  During-migrat  BW: {:.2f} MB/s ({:.0f} ms)", mb2 / (us2 / 1e6), us2 / 1000.0);
    HLOG(kInfo, "  Post-migration BW: {:.2f} MB/s ({:.0f} ms)", mb3 / (us3 / 1e6), us3 / 1000.0);
    HLOG(kInfo, "  Overall        BW: {:.2f} MB/s ({:.0f} ms total)",
         total_mb / (total_us / 1e6), total_us / 1000.0);
  }

  return 0;
}
