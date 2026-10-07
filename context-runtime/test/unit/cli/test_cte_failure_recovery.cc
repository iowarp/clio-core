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
 * CTE failure-recovery tests against a real daemon.
 *
 * CorruptRestore (#725): a corrupt metadata snapshot or transaction-log
 *   record used to kill the daemon during restart -- a garbage length sized a
 *   std::string / vector or drove a read past the record -- before it bound
 *   its port, with nothing logged. The daemon must now stay up and say why.
 *
 * BlobTimes (#796): a restored blob came back with last_modified_ == 0 --
 *   neither the WAL nor the snapshot carried its times -- which TemporalSearch
 *   reads as "never written", so every pre-restart blob vanished from time
 *   queries. One blob is restored from the snapshot, one from the WAL.
 *
 * PutBlobRollback (#1059, Linux): when a PutBlob's data write fails after
 *   the blob was already grown, the blob kept pointing at unwritten blocks
 *   and every later GetBlob of it failed. The write failure is real: the
 *   daemon gets an RLIMIT_FSIZE below the end of its backing file (SIGXFSZ
 *   ignored), so placement succeeds and the write past the limit fails with
 *   EFBIG. The blob must keep its committed size and bytes.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/transaction_log.h>
#include <clio_ctp/introspect/system_info.h>

#include "runtime_server.h"
#include "simple_test.h"

namespace fs = std::filesystem;

namespace {

/**
 * Run the clio_run binary with a hard kill deadline.
 * @param args arguments after the executable
 * @param timeout_sec seconds before the child is killed
 * @return exit code, -1 if it could not start, -3 if killed or signalled
 */
int RunCliTimed(const std::vector<std::string> &args, int timeout_sec) {
  const std::string log =
      (fs::temp_directory_path() / "cte_failure_recovery_cli.log").string();
  ctp::SpawnedProcess proc =
      ctp::SystemInfo::SpawnProcess(CLIO_RUN_EXE, args, log);
  if (!proc.valid) return -1;
  int rc = 0;
  if (!ctp::SystemInfo::WaitForChild(proc, timeout_sec * 1000, &rc)) {
    return -3;
  }
  return rc;
}

/**
 * Wait (bounded) for a stopped daemon to exit.
 * @param server the daemon
 */
void WaitForExit(clio::run::test::RuntimeServer &server) {
  for (int i = 0; i < 600 && server.IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

/**
 * Point the daemon and CLI at test-private locations.
 * @param work_dir scratch directory for this test
 */
void SetTestEnv(const fs::path &work_dir) {
  ctp::SystemInfo::Setenv("CLIO_WAIT_SERVER", "15", 1);
  ctp::SystemInfo::Setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);
  ctp::SystemInfo::Setenv("CLIO_RESTART_LOG",
                          (work_dir / "restart_log.bin").string(), 1);
  // Daemon output goes here (RuntimeServer truncates it on every Start).
  ctp::SystemInfo::Setenv("CLIO_TEST_SERVER_LOG",
                          (work_dir / "daemon.log").string(), 1);
}

/**
 * Read a whole file.
 * @param path file to read
 * @return its contents ("" if missing)
 */
std::string Slurp(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

/**
 * Append raw bytes to a file.
 * @param path file to append to (created if missing)
 * @param bytes bytes to append
 */
void AppendBytes(const fs::path &path, const std::vector<uint8_t> &bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::app);
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

/**
 * Little-endian bytes of a 64-bit value.
 * @param v the value
 * @return its 8 bytes, least significant first
 */
std::vector<uint8_t> U64(uint64_t v) {
  std::vector<uint8_t> out(8);
  for (int i = 0; i < 8; ++i) out[i] = static_cast<uint8_t>(v >> (8 * i));
  return out;
}

/**
 * Little-endian bytes of a u32.
 * @param v value
 * @return its four bytes
 */
std::vector<uint8_t> U32(uint32_t v) {
  return {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8),
          static_cast<uint8_t>(v >> 16), static_cast<uint8_t>(v >> 24)};
}

/**
 * Put `size` bytes of `fill` at `off` through a shared-memory buffer.
 * @param cte CTE client
 * @param tag_id target tag
 * @param name blob name
 * @param off blob offset
 * @param size byte count
 * @param fill fill byte
 * @return the task's return code
 */
clio::run::u32 PutFill(clio::cte::core::Client *cte,
                       const clio::cte::core::TagId &tag_id,
                       const std::string &name, size_t off, size_t size,
                       char fill) {
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(size);
  REQUIRE(!buf.IsNull());
  memset(buf.ptr_, fill, size);
  ctp::ipc::ShmPtr<> shm_ref(buf.shm_);
  auto put = cte->AsyncPutBlob(tag_id, name, off, size, shm_ref, 1.0f);
  put.Wait();
  clio::run::u32 rc = put->GetReturnCode();
  ipc->FreeBuffer(buf);
  return rc;
}

}  // namespace

TEST_CASE("CorruptRestore - corrupt snapshot and WAL do not kill the daemon",
          "[cli][cte][restart]") {
  constexpr unsigned kPort = 10603;
  const fs::path work_dir = fs::temp_directory_path() / "cte_corrupt_restore";
  fs::remove_all(work_dir);
  fs::create_directories(work_dir);
  SetTestEnv(work_dir);
  const fs::path meta_log = work_dir / "meta_log";

  const fs::path compose_yaml = work_dir / "compose.yaml";
  {
    std::ofstream f(compose_yaml);
    f << "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: \"cte_corrupt_test\"\n"
         "    pool_query: local\n"
         "    pool_id: \"701.0\"\n"
         "    restart: true\n"
         "    storage:\n"
         "      - path: " << (work_dir / "ram_dev").string() << "\n"
         "        bdev_type: ram\n"
         "        capacity_limit: 64mb\n"
         "    dpe:\n"
         "      dpe_type: random\n"
         "    performance:\n"
         "      metadata_log_path: " << meta_log.string() << "\n";
  }

  // --- Phase 1: write a snapshot plus a post-snapshot WAL entry.
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort));
  REQUIRE(server.WaitForReady());
  REQUIRE(RunCliTimed({"compose", compose_yaml.string()}, 60) == 0);
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  auto *cte = CLIO_CTE_CLIENT;
  cte->Init(clio::run::PoolId(701, 0));
  auto tag = cte->AsyncGetOrCreateTag("corrupt_tag_a");
  tag.Wait();
  REQUIRE(tag->GetReturnCode() == 0);
  REQUIRE(PutFill(cte, tag->tag_id_, "blob", 0, 4096, 'k') == 0);
  auto flush_meta = cte->AsyncFlushMetadata(clio::run::PoolQuery::Local(), 0);
  flush_meta.Wait();
  auto tag_b = cte->AsyncGetOrCreateTag("corrupt_tag_post_snapshot");
  tag_b.Wait();
  REQUIRE(tag_b->GetReturnCode() == 0);

  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);
  WaitForExit(server);
  server.Stop();
  REQUIRE(fs::exists(meta_log));

  // --- Corrupt both persistence paths.
  // Snapshot: a tag entry (type 0) whose name length claims ~4 GiB.
  AppendBytes(meta_log, {0});
  AppendBytes(meta_log, U32(0xFFFFFF00u));
  // WAL: a well-framed kCreateTag record (type, sequence number, payload
  // size, payload) whose 4-byte payload is a string length of 2 GiB -- the
  // old reader copied 2 GiB out of a 4-byte buffer. The sequence number is
  // past every real record's, so replay reaches it last.
  const fs::path tag_wal = meta_log.string() + ".tag.0";
  std::vector<uint8_t> rec = {
      static_cast<uint8_t>(clio::cte::core::TxnType::kCreateTag)};
  for (uint8_t b : U64(1ull << 40)) rec.push_back(b);
  for (uint8_t b : U32(4)) rec.push_back(b);
  for (uint8_t b : U32(0x7FFFFFFFu)) rec.push_back(b);
  AppendBytes(tag_wal, rec);

  // --- Phase 2: restart. Drop the restart log first: `clio_run start`
  // replays it and re-creates the pool as a FRESH (non-restart) pool before
  // our compose runs, and the restore would never read the corrupt files.
  // The compose below, with `restart: true`, is then what creates the pool
  // and runs restore + replay; the daemon must survive it and report why.
  fs::remove(work_dir / "restart_log.bin");
  clio::run::test::RuntimeServer server2;
  REQUIRE(server2.Start(kPort));
  REQUIRE(server2.WaitForReady());
  REQUIRE(RunCliTimed({"compose", compose_yaml.string()}, 60) == 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(server2.IsRunning());
  const std::string log = Slurp((work_dir / "daemon.log").string());
  INFO("daemon log tail: " << clio::run::test::RuntimeServer::LogTail());
  REQUIRE(log.find("is corrupt") != std::string::npos);
  REQUIRE(log.find("metadata restore") != std::string::npos);

  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);
  WaitForExit(server2);
  server2.Stop();
  fs::remove_all(work_dir);
}


/** Set (to "1") in a child that runs one phase of a multi-process case. */
constexpr const char *kPhaseEnv = "CLIO_TEST_PHASE_CHILD";

/**
 * Whether this process was launched by RunPhase. Phase cases do nothing
 * otherwise, so running the whole binary unfiltered skips them.
 * @return true inside a RunPhase child
 */
bool InPhaseChild() { return ctp::SystemInfo::Getenv(kPhaseEnv) == "1"; }

/**
 * Run one phase case of this binary in a fresh process. Each client phase of
 * a restart test needs one: a client connects once, and the daemon it talked
 * to is gone after the restart.
 * @param phase the phase case's name (a test filter)
 * @param log_path where the child's output goes
 * @return the child's exit code (0 = the phase passed), -1 if it could not
 *         start, -3 if it hung or crashed
 */
int RunPhase(const std::string &phase, const std::string &log_path) {
  ctp::SystemInfo::Setenv(kPhaseEnv, "1", 1);
  ctp::SpawnedProcess proc = ctp::SystemInfo::SpawnProcess(
      ctp::SystemInfo::GetExecutablePath(), {phase}, log_path);
  ctp::SystemInfo::Unsetenv(kPhaseEnv);
  if (!proc.valid) return -1;
  int rc = 0;
  if (!ctp::SystemInfo::WaitForChild(proc, 120 * 1000, &rc)) return -3;
  return rc;
}

/**
 * Connect to the daemon as a CTE client on the given pool.
 * @param pool_major pool id major of the composed CTE pool
 * @return the client, or nullptr if init failed
 */
clio::cte::core::Client *ConnectCte(clio::run::u32 pool_major) {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    return nullptr;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) return nullptr;
  auto *cte = CLIO_CTE_CLIENT;
  cte->Init(clio::run::PoolId(pool_major, 0));
  return cte;
}

/**
 * Time-search one tag's blobs.
 * @param cte CTE client
 * @param tag tag name (a regex)
 * @return the hits, in ascending last_modified_ order
 */
std::vector<clio::cte::core::TemporalSearchResult> SearchTimes(
    clio::cte::core::Client *cte, const std::string &tag) {
  auto search = cte->AsyncTemporalSearch(tag, ".*", 0, 0, 0,
                                         clio::run::PoolQuery::Local());
  search.Wait();
  return search->results_;
}

/** Pool and tag shared by the BlobTimes case and its phases. */
constexpr clio::run::u32 kTimesPool = 703;
constexpr const char *kTimesTag = "times_tag";

TEST_CASE("TimesPhaseWrite - one blob in the snapshot, one in the WAL",
          "[cli][cte][restart][phase]") {
  if (!InPhaseChild()) return;
  auto *cte = ConnectCte(kTimesPool);
  REQUIRE(cte != nullptr);
  auto tag = cte->AsyncGetOrCreateTag(kTimesTag);
  tag.Wait();
  REQUIRE(tag->GetReturnCode() == 0);
  REQUIRE(PutFill(cte, tag->tag_id_, "in_snapshot", 0, 4096, 's') == 0);
  auto flush = cte->AsyncFlushMetadata(clio::run::PoolQuery::Local(), 0);
  flush.Wait();
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  REQUIRE(PutFill(cte, tag->tag_id_, "in_wal", 0, 4096, 'w') == 0);
  REQUIRE(SearchTimes(cte, kTimesTag).size() == 2);
}

TEST_CASE("TimesPhaseRead - both blobs are found by time, in write order",
          "[cli][cte][restart][phase]") {
  if (!InPhaseChild()) return;
  auto *cte = ConnectCte(kTimesPool);
  REQUIRE(cte != nullptr);
  auto hits = SearchTimes(cte, kTimesTag);
  INFO("TemporalSearch hits after restart: " << hits.size());
  REQUIRE(hits.size() == 2);
  REQUIRE(hits[0].blob_name_ == "in_snapshot");
  REQUIRE(hits[1].blob_name_ == "in_wal");
  REQUIRE(hits[0].last_modified_ < hits[1].last_modified_);
}

TEST_CASE("BlobTimes - restored blobs keep their write times",
          "[cli][cte][restart]") {
  constexpr unsigned kPort = 10605;
  const fs::path work_dir = fs::temp_directory_path() / "cte_blob_times";
  fs::remove_all(work_dir);
  fs::create_directories(work_dir);
  SetTestEnv(work_dir);
  const fs::path compose_yaml = work_dir / "compose.yaml";
  {
    std::ofstream f(compose_yaml);
    f << "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: \"cte_times_test\"\n"
         "    pool_query: local\n"
         "    pool_id: \"" << kTimesPool << ".0\"\n"
         "    restart: true\n"
         "    storage:\n"
         "      - path: " << (work_dir / "disk.bin").string() << "\n"
         "        bdev_type: file\n"
         "        capacity_limit: 64mb\n"
         "    dpe:\n"
         "      dpe_type: random\n"
         "    performance:\n"
         "      metadata_log_path: " << (work_dir / "meta_log").string()
      << "\n";
  }
  const std::string phase_log = (work_dir / "phase.log").string();

  // --- Phase 1: one blob in the snapshot, one only in the WAL.
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort));
  REQUIRE(server.WaitForReady());
  REQUIRE(RunCliTimed({"compose", compose_yaml.string()}, 60) == 0);
  const int write_rc = RunPhase("TimesPhaseWrite", phase_log);
  INFO("write phase output: " << Slurp(phase_log));
  REQUIRE(write_rc == 0);
  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);
  WaitForExit(server);
  server.Stop();

  // --- Phase 2: restart and restore (same restart-log handling as
  // CorruptRestore: the compose with `restart: true` must run the restore).
  fs::remove(work_dir / "restart_log.bin");
  clio::run::test::RuntimeServer server2;
  REQUIRE(server2.Start(kPort));
  REQUIRE(server2.WaitForReady());
  REQUIRE(RunCliTimed({"compose", compose_yaml.string()}, 60) == 0);

  // --- Phase 3: a fresh client finds both blobs by time, in write order.
  const int read_rc = RunPhase("TimesPhaseRead", phase_log);
  INFO("read phase output: " << Slurp(phase_log));
  REQUIRE(read_rc == 0);

  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);
  WaitForExit(server2);
  server2.Stop();
  fs::remove_all(work_dir);
}


TEST_CASE("PutBlobRollback - a failed data write leaves the blob committed",
          "[cli][cte][putblob]") {
  constexpr unsigned kPort = 10604;
  constexpr size_t kHead = 4096;               // committed bytes
  constexpr size_t kGrow = 4 * 1024 * 1024;    // append that will fail
  constexpr uint64_t kFsizeLimit = 1024 * 1024;  // writes past 1 MiB fail
  const fs::path work_dir = fs::temp_directory_path() / "cte_putblob_rollback";
  fs::remove_all(work_dir);
  fs::create_directories(work_dir);
  SetTestEnv(work_dir);

  const fs::path compose_yaml = work_dir / "compose.yaml";
  {
    // One 64 MiB file tier: smaller than the 1 GiB growth unit, so the whole
    // backing file exists from compose time and no allocation needs to grow
    // it -- placement succeeds and only the data write can fail.
    std::ofstream f(compose_yaml);
    f << "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: \"cte_rollback_test\"\n"
         "    pool_query: local\n"
         "    pool_id: \"702.0\"\n"
         "    storage:\n"
         "      - path: " << (work_dir / "file_dev.dat").string() << "\n"
         "        bdev_type: file\n"
         "        capacity_limit: 64mb\n"
         "    dpe:\n"
         "      dpe_type: random\n";
  }

  // Ignored dispositions survive exec: the daemon gets EFBIG, not a SIGXFSZ
  // that would kill it, once it writes past its file-size limit.
  ctp::SystemInfo::IgnoreFileSizeSignal();
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort));
  REQUIRE(server.WaitForReady());
  REQUIRE(RunCliTimed({"compose", compose_yaml.string()}, 60) == 0);
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  auto *cte = CLIO_CTE_CLIENT;
  cte->Init(clio::run::PoolId(702, 0));
  auto tag = cte->AsyncGetOrCreateTag("rollback_tag");
  tag.Wait();
  REQUIRE(tag->GetReturnCode() == 0);
  const auto tag_id = tag->tag_id_;

  auto blob_size = [&]() {
    auto t = cte->AsyncGetBlobSize(tag_id, "blob");
    t.Wait();
    return t->size_;
  };

  // Committed state: 4 KiB at the front of the device.
  REQUIRE(PutFill(cte, tag_id, "blob", 0, kHead, 'a') == 0);
  REQUIRE(blob_size() == kHead);

  // Cap the daemon's file size below where the append will land.
  // Lower only the soft limit (EFBIG is driven by it): an unprivileged
  // process cannot raise a hard limit back up afterwards.
  uint64_t orig = 0;
  REQUIRE(ctp::SystemInfo::SetProcessFileSizeLimit(server.Pid(), kFsizeLimit,
                                                   &orig));
  REQUIRE(PutFill(cte, tag_id, "blob", kHead, kGrow, 'b') != 0);

  // The failed append must not have grown the blob, and the committed bytes
  // must still read back.
  REQUIRE(blob_size() == kHead);
  std::vector<char> head(kHead, 0);
  auto get = cte->AsyncGetBlob(tag_id, "blob", 0, kHead, 0, head.data());
  get.Wait();
  REQUIRE((get.get() == nullptr || get->GetReturnCode() == 0));
  REQUIRE(std::all_of(head.begin(), head.end(),
                      [](char c) { return c == 'a'; }));

  // Lift the limit: the same append now succeeds.
  REQUIRE(ctp::SystemInfo::SetProcessFileSizeLimit(server.Pid(), orig));
  REQUIRE(PutFill(cte, tag_id, "blob", kHead, kGrow, 'b') == 0);
  REQUIRE(blob_size() == kHead + kGrow);

  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);
  WaitForExit(server);
  server.Stop();
  fs::remove_all(work_dir);
}

SIMPLE_TEST_MAIN()
