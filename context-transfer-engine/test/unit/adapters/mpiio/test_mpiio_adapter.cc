/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * MPI-IO ADAPTER UNIT TESTS
 *
 * Exercise the WRP CTE MPI-IO adapter (libclio_cte_mpiio.so) under the
 * new clio:: prefix gating. Runs as a single-rank MPI program: enough
 * to validate the open/read/write/close paths and the prefix check
 * without needing a multi-node launcher in CI.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>
#include <mpi.h>

#include <cstring>
#include <filesystem>
#include <vector>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_cte/core/core_client.h"

namespace stdfs = std::filesystem;

namespace {
constexpr int kPayload = 16 * 1024;  // 16 KiB
const std::string kBackend = "/tmp/clio_cte_mpiio_test.dat";
const std::string kClio = "clio::" + kBackend;

bool initializeRuntime() {
  static bool initialized = false;
  if (initialized) return true;

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    INFO("Clio init failed");
    return false;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    INFO("CTE init failed");
    return false;
  }
  // Give the CTE core pool a file-backed storage target.
  auto *cte_client = CLIO_CTE_CLIENT;
  clio::run::PoolId bdev_pool_id(952, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto create_task = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), kBackend, bdev_pool_id,
      clio::run::bdev::BdevType::kFile);
  create_task.Wait();
  auto reg_task = cte_client->AsyncRegisterTarget(
      kBackend, clio::run::bdev::BdevType::kFile,
      static_cast<clio::run::u64>(kPayload) * 64, clio::run::PoolQuery::Local(),
      bdev_pool_id);
  reg_task.Wait();
  if (reg_task->GetReturnCode() != 0) {
    INFO("Failed to register target, rc=" << reg_task->GetReturnCode());
    return false;
  }
  initialized = true;
  return true;
}
}  // namespace

TEST_CASE("MPI-IO Adapter: independent write + read round-trip",
          "[mpiio][adapter]") {
  REQUIRE(initializeRuntime());
  stdfs::remove(kBackend);

  MPI_File fh;
  int rc = MPI_File_open(MPI_COMM_SELF, kClio.c_str(),
                         MPI_MODE_CREATE | MPI_MODE_RDWR,
                         MPI_INFO_NULL, &fh);
  REQUIRE(rc == MPI_SUCCESS);

  std::vector<char> w(kPayload);
  for (int i = 0; i < kPayload; ++i) w[i] = char((i * 7) & 0xff);

  MPI_Status st;
  rc = MPI_File_write(fh, w.data(), kPayload, MPI_CHAR, &st);
  REQUIRE(rc == MPI_SUCCESS);

  // Seek back and read.
  REQUIRE(MPI_File_seek(fh, 0, MPI_SEEK_SET) == MPI_SUCCESS);
  std::vector<char> r(kPayload, 0);
  rc = MPI_File_read(fh, r.data(), kPayload, MPI_CHAR, &st);
  REQUIRE(rc == MPI_SUCCESS);
  REQUIRE(r == w);

  REQUIRE(MPI_File_close(&fh) == MPI_SUCCESS);
  stdfs::remove(kBackend);
}

TEST_CASE("MPI-IO Adapter: explicit-offset (read_at/write_at) ops",
          "[mpiio][adapter][offset]") {
  REQUIRE(initializeRuntime());
  stdfs::remove(kBackend);

  MPI_File fh;
  REQUIRE(MPI_File_open(MPI_COMM_SELF, kClio.c_str(),
                        MPI_MODE_CREATE | MPI_MODE_RDWR,
                        MPI_INFO_NULL, &fh) == MPI_SUCCESS);

  std::vector<char> a(1024, 'a'), b(1024, 'b');
  MPI_Status st;
  // Write 'a' block at offset 0, 'b' block at offset 4096 (leaves a hole).
  REQUIRE(MPI_File_write_at(fh, 0, a.data(), 1024, MPI_CHAR, &st) ==
          MPI_SUCCESS);
  REQUIRE(MPI_File_write_at(fh, 4096, b.data(), 1024, MPI_CHAR, &st) ==
          MPI_SUCCESS);

  std::vector<char> r(1024, 0);
  REQUIRE(MPI_File_read_at(fh, 4096, r.data(), 1024, MPI_CHAR, &st) ==
          MPI_SUCCESS);
  REQUIRE(r == b);

  REQUIRE(MPI_File_close(&fh) == MPI_SUCCESS);
  stdfs::remove(kBackend);
}

TEST_CASE("MPI-IO Adapter: bare path is not intercepted",
          "[mpiio][adapter][prefix]") {
  REQUIRE(initializeRuntime());
  const std::string backend = "/tmp/clio_cte_mpiio_bare.dat";
  stdfs::remove(backend);

  // No clio:: prefix → MPI-IO call goes straight to the underlying
  // MPI implementation, which still works fine. We can't easily check
  // "did interception happen" without internal APIs, but a successful
  // open + close on a path the CTE wasn't told about is the right
  // negative-control behaviour.
  MPI_File fh;
  REQUIRE(MPI_File_open(MPI_COMM_SELF, backend.c_str(),
                        MPI_MODE_CREATE | MPI_MODE_RDWR,
                        MPI_INFO_NULL, &fh) == MPI_SUCCESS);
  REQUIRE(MPI_File_close(&fh) == MPI_SUCCESS);
  REQUIRE(stdfs::exists(backend));
  stdfs::remove(backend);
}

/*
 * Custom main: bracket Catch around MPI_Init/MPI_Finalize. With more
 * than one rank, this run is a smoke test only — the test cases above
 * assume MPI_COMM_SELF semantics, not collective semantics.
 */
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
