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
#include "clio_ctp/introspect/system_info.h"

namespace stdfs = std::filesystem;

namespace {
constexpr int kPayload = 16 * 1024;  // 16 KiB
const std::string kBackend = "/tmp/clio_cte_mpiio_test.dat";
const std::string kClio = "clio::" + kBackend;

bool initializeRuntime() {
  static bool initialized = false;
  if (initialized) return true;
  // kBackend is the storage target's backing file: remove a stale one now,
  // never once the target is up -- deleting a live device file loses every
  // write made after it (the test cases used to do exactly that).
  stdfs::remove(kBackend);

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
}

TEST_CASE("MPI-IO Adapter: explicit-offset (read_at/write_at) ops",
          "[mpiio][adapter][offset]") {
  REQUIRE(initializeRuntime());

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

TEST_CASE("MPI-IO Adapter: status reports bytes actually transferred",
          "[mpiio][adapter][status]") {
  REQUIRE(initializeRuntime());
  // Its own file: the other cases leave 16 KiB in kClio, and open does not
  // truncate, so the short read below would not be short.
  const std::string path = "clio::/tmp/clio_cte_mpiio_status.dat";

  MPI_File fh;
  REQUIRE(MPI_File_open(MPI_COMM_SELF, path.c_str(),
                        MPI_MODE_CREATE | MPI_MODE_RDWR,
                        MPI_INFO_NULL, &fh) == MPI_SUCCESS);

  constexpr int kInts = 250;  // 1000 bytes
  std::vector<int> w(kInts, 7);
  MPI_Status st;
  REQUIRE(MPI_File_write_at(fh, 0, w.data(), kInts, MPI_INT, &st) ==
          MPI_SUCCESS);
  int count = -1;
  REQUIRE(MPI_Get_count(&st, MPI_INT, &count) == MPI_SUCCESS);
  REQUIRE(count == kInts);
  REQUIRE(MPI_File_sync(fh) == MPI_SUCCESS);

  // Ask for twice what the file holds: the read is short, and the status
  // must say so (#1187: it used to be left unset).
  std::vector<char> r(2 * kInts * sizeof(int), 0);
  REQUIRE(MPI_File_read_at(fh, 0, r.data(), static_cast<int>(r.size()),
                           MPI_CHAR, &st) == MPI_SUCCESS);
  REQUIRE(MPI_Get_count(&st, MPI_CHAR, &count) == MPI_SUCCESS);
  REQUIRE(count == static_cast<int>(kInts * sizeof(int)));
  const char *wb = reinterpret_cast<const char *>(w.data());
  size_t first_diff = 0;
  while (first_diff < kInts * sizeof(int) && r[first_diff] == wb[first_diff]) {
    ++first_diff;
  }
  INFO("first differing byte " << first_diff << ": read "
       << static_cast<int>(r[first_diff % r.size()]) << " wrote "
       << static_cast<int>(wb[first_diff % (kInts * sizeof(int))]));
  REQUIRE(first_diff == kInts * sizeof(int));

  REQUIRE(MPI_File_close(&fh) == MPI_SUCCESS);
}

TEST_CASE("MPI-IO Adapter: a failed write is reported at sync",
          "[mpiio][adapter][errors]") {
  REQUIRE(initializeRuntime());

  MPI_File fh;
  REQUIRE(MPI_File_open(MPI_COMM_SELF, kClio.c_str(),
                        MPI_MODE_CREATE | MPI_MODE_RDWR,
                        MPI_INFO_NULL, &fh) == MPI_SUCCESS);

  // The runtime runs in this process, so capping this process's file size
  // makes the storage target's write fail with EFBIG (the signal that would
  // otherwise kill the process is ignored).
  ctp::SystemInfo::IgnoreFileSizeSignal();
  const int pid = ctp::SystemInfo::GetPid();
  uint64_t orig = 0;
  REQUIRE(ctp::SystemInfo::SetProcessFileSizeLimit(pid, 1, &orig));

  std::vector<char> w(kPayload, 'x');
  MPI_Status st;
  int write_rc = MPI_File_write_at(fh, 0, w.data(), kPayload, MPI_CHAR, &st);
  // Writes are deferred, so the failure may only surface here. Before #1187
  // both calls returned MPI_SUCCESS and the data was silently lost.
  int sync_rc = MPI_File_sync(fh);
  REQUIRE(ctp::SystemInfo::SetProcessFileSizeLimit(pid, orig));
  INFO("write_rc=" << write_rc << " sync_rc=" << sync_rc);
  REQUIRE((write_rc != MPI_SUCCESS || sync_rc != MPI_SUCCESS));

  // The failure is reported once; with the limit lifted the file works.
  REQUIRE(MPI_File_write_at(fh, 0, w.data(), kPayload, MPI_CHAR, &st) ==
          MPI_SUCCESS);
  REQUIRE(MPI_File_sync(fh) == MPI_SUCCESS);
  REQUIRE(MPI_File_close(&fh) == MPI_SUCCESS);
}

namespace {
/** Open a fresh clio:: file holding bytes 0, 1, 2, ... (mod 251). */
MPI_File OpenPattern(const std::string& path, int nbytes) {
  MPI_File_delete(path.c_str(), MPI_INFO_NULL);
  MPI_File fh;
  REQUIRE(MPI_File_open(MPI_COMM_SELF, path.c_str(),
                        MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL,
                        &fh) == MPI_SUCCESS);
  std::vector<char> w(nbytes);
  for (int i = 0; i < nbytes; ++i) w[i] = static_cast<char>(i % 251);
  MPI_Status st;
  REQUIRE(MPI_File_write_at(fh, 0, w.data(), nbytes, MPI_CHAR, &st) ==
          MPI_SUCCESS);
  REQUIRE(MPI_File_sync(fh) == MPI_SUCCESS);
  return fh;
}
char Pattern(long off) { return static_cast<char>(off % 251); }
}  // namespace

TEST_CASE("MPI-IO Adapter: file metadata calls parallel HDF5 makes",
          "[mpiio][adapter][hdf5]") {
  REQUIRE(initializeRuntime());
  // Before these were wrapped, each one handed the adapter's handle to the
  // real MPI library, which crashed (H5Fopen calls MPI_File_get_info).
  const std::string path = "clio::/tmp/clio_cte_mpiio_meta.dat";
  MPI_File fh = OpenPattern(path, 4096);

  MPI_Offset size = -1;
  REQUIRE(MPI_File_get_size(fh, &size) == MPI_SUCCESS);
  REQUIRE(size == 4096);
  REQUIRE(MPI_File_set_size(fh, 1000) == MPI_SUCCESS);
  REQUIRE(MPI_File_get_size(fh, &size) == MPI_SUCCESS);
  REQUIRE(size == 1000);

  MPI_Info info = MPI_INFO_NULL;
  REQUIRE(MPI_File_get_info(fh, &info) == MPI_SUCCESS);
  REQUIRE(info != MPI_INFO_NULL);
  REQUIRE(MPI_File_set_info(fh, info) == MPI_SUCCESS);
  MPI_Info_free(&info);

  int amode = 0;
  REQUIRE(MPI_File_get_amode(fh, &amode) == MPI_SUCCESS);
  REQUIRE(amode == (MPI_MODE_CREATE | MPI_MODE_RDWR));

  int flag = 0;
  REQUIRE(MPI_File_set_atomicity(fh, 1) == MPI_SUCCESS);
  REQUIRE(MPI_File_get_atomicity(fh, &flag) == MPI_SUCCESS);
  REQUIRE(flag == 1);

  REQUIRE(MPI_File_close(&fh) == MPI_SUCCESS);
  REQUIRE(MPI_File_delete(path.c_str(), MPI_INFO_NULL) == MPI_SUCCESS);
  REQUIRE(MPI_File_open(MPI_COMM_SELF, path.c_str(), MPI_MODE_RDONLY,
                        MPI_INFO_NULL, &fh) != MPI_SUCCESS);
}

TEST_CASE("MPI-IO Adapter: non-contiguous file view",
          "[mpiio][adapter][view]") {
  REQUIRE(initializeRuntime());
  const std::string path = "clio::/tmp/clio_cte_mpiio_view.dat";
  MPI_File fh = OpenPattern(path, 1024);

  // Tile: 4 blocks of 4 bytes every 16 bytes, tiled every 64 bytes,
  // after an 8-byte displacement.
  MPI_Datatype vec, tile;
  MPI_Type_vector(4, 4, 16, MPI_BYTE, &vec);
  MPI_Type_create_resized(vec, 0, 64, &tile);
  MPI_Type_commit(&tile);
  auto file_off = [](long pos) {  // data-stream byte -> file byte
    return 8 + (pos / 16) * 64 + ((pos % 16) / 4) * 16 + pos % 4;
  };

  REQUIRE(MPI_File_set_view(fh, 8, MPI_BYTE, tile, "native", MPI_INFO_NULL) ==
          MPI_SUCCESS);
  SECTION("read across tiles, starting mid-block") {
    std::vector<char> r(40);
    MPI_Status st;
    REQUIRE(MPI_File_read_at_all(fh, 3, r.data(), 40, MPI_CHAR, &st) ==
            MPI_SUCCESS);
    int count = -1;
    MPI_Get_count(&st, MPI_CHAR, &count);
    REQUIRE(count == 40);
    for (int i = 0; i < 40; ++i) {
      INFO("stream byte " << 3 + i);
      REQUIRE(r[i] == Pattern(file_off(3 + i)));
    }
  }
  SECTION("write through the view lands in the selected bytes only") {
    std::vector<char> w(32, 'Z');
    MPI_Status st;
    REQUIRE(MPI_File_write_at(fh, 0, w.data(), 32, MPI_CHAR, &st) ==
            MPI_SUCCESS);
    REQUIRE(MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native",
                              MPI_INFO_NULL) == MPI_SUCCESS);
    std::vector<char> r(200);
    REQUIRE(MPI_File_read_at(fh, 0, r.data(), 200, MPI_CHAR, &st) ==
            MPI_SUCCESS);
    std::vector<bool> hit(200, false);
    for (int i = 0; i < 32; ++i) hit[file_off(i)] = true;
    for (int i = 0; i < 200; ++i) {
      INFO("file byte " << i);
      REQUIRE(r[i] == (hit[i] ? 'Z' : Pattern(i)));
    }
  }
  SECTION("etype offsets count etypes, not bytes") {
    REQUIRE(MPI_File_set_view(fh, 0, MPI_INT, MPI_INT, "native",
                              MPI_INFO_NULL) == MPI_SUCCESS);
    std::vector<char> r(8);
    MPI_Status st;
    REQUIRE(MPI_File_read_at(fh, 5, r.data(), 8, MPI_CHAR, &st) == MPI_SUCCESS);
    for (int i = 0; i < 8; ++i) REQUIRE(r[i] == Pattern(20 + i));
  }
  SECTION("file-pointer ops refuse a non-default view") {
    std::vector<char> r(4);
    MPI_Status st;
    REQUIRE(MPI_File_read(fh, r.data(), 4, MPI_CHAR, &st) != MPI_SUCCESS);
  }
  MPI_Type_free(&tile);
  MPI_Type_free(&vec);
  REQUIRE(MPI_File_close(&fh) == MPI_SUCCESS);
}

TEST_CASE("MPI-IO Adapter: derived memory datatype", "[mpiio][adapter][view]") {
  REQUIRE(initializeRuntime());
  const std::string path = "clio::/tmp/clio_cte_mpiio_memtype.dat";
  MPI_File fh = OpenPattern(path, 256);

  // Memory: 4 blocks of 4 bytes every 8 bytes; the gaps must be untouched.
  MPI_Datatype mem;
  MPI_Type_vector(4, 4, 8, MPI_BYTE, &mem);
  MPI_Type_commit(&mem);
  std::vector<char> r(32, '.');
  MPI_Status st;
  REQUIRE(MPI_File_read_at(fh, 100, r.data(), 1, mem, &st) == MPI_SUCCESS);
  for (int i = 0; i < 28; ++i) {
    INFO("memory byte " << i);
    if (i % 8 < 4) {
      REQUIRE(r[i] == Pattern(100 + (i / 8) * 4 + i % 8));
    } else {
      REQUIRE(r[i] == '.');
    }
  }
  MPI_Type_free(&mem);
  REQUIRE(MPI_File_close(&fh) == MPI_SUCCESS);
}

TEST_CASE("MPI-IO Adapter: an undecoded filetype is refused",
          "[mpiio][adapter][view]") {
  REQUIRE(initializeRuntime());
  const std::string path = "clio::/tmp/clio_cte_mpiio_subarray.dat";
  MPI_File fh = OpenPattern(path, 256);

  // Subarray is not decoded: the view must be refused, not ignored, or reads
  // would silently return the wrong bytes.
  int sizes[1] = {16}, subsizes[1] = {4}, starts[1] = {2};
  MPI_Datatype sub;
  MPI_Type_create_subarray(1, sizes, subsizes, starts, MPI_ORDER_C, MPI_BYTE,
                           &sub);
  MPI_Type_commit(&sub);
  REQUIRE(MPI_File_set_view(fh, 0, MPI_BYTE, sub, "native", MPI_INFO_NULL) !=
          MPI_SUCCESS);
  // The previous (default) view still applies.
  char c = 0;
  MPI_Status st;
  REQUIRE(MPI_File_read_at(fh, 7, &c, 1, MPI_CHAR, &st) == MPI_SUCCESS);
  REQUIRE(c == Pattern(7));
  MPI_Type_free(&sub);
  REQUIRE(MPI_File_close(&fh) == MPI_SUCCESS);
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
