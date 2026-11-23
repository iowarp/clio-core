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
 * A persistent clio-fs (the configuration of the "persistent local
 * filesystem" tutorial) keeps files across a runtime restart.
 *
 * The runtime runs the full chain a mount uses -- shm cache over the
 * write-back cache, a volatile core, replication and a persistent core
 * with a long_term file tier, stream with a size log, filesystem with a
 * namespace log -- from one config file, so every start composes it the
 * same way. Each phase runs in a CHILD process (this
 * binary, filtered to the [persist_child] case) so a client never outlives
 * the runtime it attached to:
 *
 *   write  -> stop (graceful, or killed) -> start without --fresh -> verify
 *
 * Covered: file pages (several 1 MiB pages, through the shm cache), logical
 * size (a partial last page and a shrinking truncate), the namespace (a
 * rename), and a file never fsynced, which only a graceful stop must keep.
 */

#include <fcntl.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "clio_cte/core/core_client.h"
#include "clio_cte/filesystem/filesystem_client.h"
#include "clio_ctp/introspect/system_info.h"
#include "clio_runtime/clio_runtime.h"
#include "runtime_server.h"
#include "simple_test.h"

namespace fs = std::filesystem;
using clio::cte::filesystem::FsSsize;

namespace {

/** Own port (RESOURCE_LOCK keeps other daemon tests off it). */
constexpr unsigned kPort = 10633;
/** Synced file: three full pages and a partial fourth. */
constexpr size_t kSyncedBytes = 3 * 1024 * 1024 + 4097;
/** Truncated file: written at 2 MiB, truncated to 1.5 MiB. */
constexpr size_t kTruncWritten = 2 * 1024 * 1024;
constexpr size_t kTruncLength = 1536 * 1024;
/** Renamed and unsynced files. */
constexpr size_t kSmallBytes = 64 * 1024 + 7;

constexpr const char *kSynced = "clio::/persist_synced.bin";
constexpr const char *kTrunc = "clio::/persist_trunc.bin";
constexpr const char *kRenameSrc = "clio::/persist_rename_src.bin";
constexpr const char *kRenameDst = "clio::/persist_renamed.bin";
constexpr const char *kUnsynced = "clio::/persist_unsynced.bin";

/** Env var naming the phase a child runs ("write" or "verify"). */
constexpr const char *kPhaseEnv = "CLIO_PERSIST_PHASE";
/** Env var: "1" when the verify phase must also find the unsynced file. */
constexpr const char *kExpectUnsyncedEnv = "CLIO_PERSIST_EXPECT_UNSYNCED";

/**
 * Deterministic contents of one test file.
 * @param n bytes
 * @param seed distinguishes files
 * @return the bytes
 */
std::vector<char> Pattern(size_t n, int seed) {
  std::vector<char> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<char>((i * 31 + static_cast<size_t>(seed) * 7) % 251);
  }
  return v;
}

/**
 * The test's working directory (config, devices, logs).
 * @return a per-user path under the temp directory
 */
fs::path WorkDir() {
  std::string user = ctp::SystemInfo::Getenv("USER");
  if (user.empty()) user = ctp::SystemInfo::Getenv("USERNAME");
  if (user.empty()) user = "user";
  return fs::temp_directory_path() / ("clio_cfs_persistence_" + user);
}

/**
 * Write the runtime config: the persistent-filesystem chain.
 * @param work working directory
 * @return the config path
 */
std::string WriteConfig(const fs::path &work) {
  auto p = [&work](const char *name) {
    return (work / name).generic_string();
  };
  const std::string yaml = p("persistent.yaml");
  std::ofstream f(yaml);
  f << "compose:\n"
       // Persistent core: the durable copy of every file lives here.
       "  - mod_name: clio_cte_core\n"
       "    pool_name: cte_main\n"
       "    pool_query: local\n"
       "    pool_id: \"512.0\"\n"
       "    durability: persistent\n"
       "    storage:\n"
       "      - path: \"" << p("durable_tier.dat") << "\"\n"
       "        bdev_type: file\n"
       "        capacity_limit: 256mb\n"
       "        score: 0.2\n"
       "        persistence_level: long_term\n"
       "    performance:\n"
       "      metadata_log_path: \"" << p("cte_metadata_log") << "\"\n"
       // Volatile core: where the write-back cache lands writes first.
       "  - mod_name: clio_cte_core\n"
       "    pool_name: cte_volatile\n"
       "    pool_query: local\n"
       "    pool_id: \"540.0\"\n"
       "    durability: volatile\n"
       "    storage:\n"
       "      - path: \"ram::persist_ram\"\n"
       "        bdev_type: ram\n"
       "        capacity_limit: 128mb\n"
       "        score: 1.0\n"
       "  - mod_name: clio_cte_replication\n"
       "    pool_name: clio_cte_replication\n"
       "    pool_query: local\n"
       "    pool_id: \"561.0\"\n"
       "    next_pool_id: \"512.0\"\n"
       "    replication_factor: 1\n"
       "    replica_score: 0.2\n"
       // No periodic flush within the test: only fsync and the flush a
       // graceful stop runs can make bytes durable, so each is what is tested.
       "  - mod_name: clio_cte_core_persist_write_back\n"
       "    pool_name: clio_cte_core_persist_write_back\n"
       "    pool_query: local\n"
       "    pool_id: \"567.0\"\n"
       "    next_volatile: \"540.0\"\n"
       "    next_persistent: \"561.0\"\n"
       "    flush_period_ms: 600000\n"
       "    flush_pool: \"568.0\"\n"
       "    fallback_to_persistent: true\n"
       "  - mod_name: clio_cte_core_shm_cache_writethru\n"
       "    pool_name: clio_cte_core_shm_cache_writethru\n"
       "    pool_query: local\n"
       "    pool_id: \"568.0\"\n"
       "    next_pool_id: \"567.0\"\n"
       "    md_only: true\n"
       "    invalidate_on_flush: true\n"
       "    capacity: \"64M\"\n"
       "  - mod_name: clio_cte_stream\n"
       "    pool_name: clio_cte_stream\n"
       "    pool_query: local\n"
       "    pool_id: \"565.0\"\n"
       "    next_pool_id: \"568.0\"\n"
       "    staging_pool_id: \"561.0\"\n"
       "    log_path: \"" << p("cte_stream_log") << "\"\n"
       "  - mod_name: clio_cte_filesystem\n"
       "    pool_name: clio_cte_filesystem\n"
       "    pool_query: local\n"
       "    pool_id: \"560.0\"\n"
       "    next_pool_id: \"568.0\"\n"
       "    metadata_log_path: \"" << p("cfs_namespace_log") << "\"\n";
  return yaml;
}

/**
 * Run a program to completion.
 * @param exe program
 * @param args arguments
 * @param log file for its output
 * @param timeout_sec bound on the run
 * @return its exit code, -1 if it did not start, -3 on timeout
 */
int RunToCompletion(const std::string &exe,
                    const std::vector<std::string> &args,
                    const std::string &log, int timeout_sec) {
  ctp::SpawnedProcess proc = ctp::SystemInfo::SpawnProcess(exe, args, log);
  if (!proc.valid) return -1;
  int rc = 0;
  if (!ctp::SystemInfo::WaitForChild(proc, timeout_sec * 1000, &rc)) {
    ctp::SystemInfo::TerminateChild(proc);
    return -3;
  }
  return rc;
}

/**
 * Wait (bounded) for a daemon to exit on its own.
 * @param server the daemon
 * @param timeout_ms bound on the wait
 * @return true when it exited within the bound
 */
bool WaitDaemonExit(clio::run::test::RuntimeServer *server, int timeout_ms) {
  for (int waited = 0; waited < timeout_ms; waited += 100) {
    if (!server->IsRunning()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return !server->IsRunning();
}

/**
 * Run one phase in a child process of this binary.
 * @param phase "write" or "verify"
 * @param work working directory (for the child's log)
 * @return the child's exit code (0: every check passed)
 */
int RunPhase(const std::string &phase, const fs::path &work) {
  ctp::SystemInfo::Setenv(kPhaseEnv, phase, 1);
  const std::string log = (work / ("phase_" + phase + ".log")).string();
  const int rc = RunToCompletion(CFS_PERSIST_TEST_EXE, {"[persist_child]"},
                                 log, 180);
  ctp::SystemInfo::Unsetenv(kPhaseEnv);
  if (rc != 0) {
    std::ifstream in(log);
    std::string line;
    while (std::getline(in, line)) INFO("  child| " << line);
  }
  return rc;
}

/**
 * Write a whole file, optionally fsync it, and close it.
 * @param cfs filesystem client
 * @param path clio:: path
 * @param data contents
 * @param sync fsync before close
 */
void WriteFile(clio::cte::filesystem::Client *cfs, const std::string &path,
               const std::vector<char> &data, bool sync) {
  int fd = cfs->OpenFd(path, O_CREAT | O_RDWR | O_TRUNC, 0644);
  REQUIRE(fd >= 0);
  REQUIRE(cfs->WriteFd(fd, data.data(), data.size()) ==
          static_cast<FsSsize>(data.size()));
  if (sync) REQUIRE(cfs->SyncFd(fd) == 0);
  REQUIRE(cfs->CloseFd(fd) == 0);
}

/**
 * Size of a file as the filesystem reports it.
 * @param cfs filesystem client
 * @param path clio:: path
 * @param exists set to whether the path names a file
 * @return its logical size (0 when absent)
 */
clio::run::u64 SizeOf(clio::cte::filesystem::Client *cfs,
                      const std::string &path, bool *exists) {
  clio::run::u64 size = 0;
  *exists = false;
  // A missing path may also report failure; *exists stays false either way.
  (void)cfs->GetAttr(path.substr(sizeof("clio::") - 1), exists, &size);
  return size;
}

/**
 * Check one file's size and contents.
 * @param cfs filesystem client
 * @param path clio:: path
 * @param want expected contents
 */
void CheckFile(clio::cte::filesystem::Client *cfs, const std::string &path,
               const std::vector<char> &want) {
  INFO("checking " << path);
  bool exists = false;
  REQUIRE(SizeOf(cfs, path, &exists) == want.size());
  REQUIRE(exists);
  int fd = cfs->OpenFd(path, O_RDONLY, 0);
  REQUIRE(fd >= 0);
  std::vector<char> back(want.size(), 0);
  REQUIRE(cfs->PreadFd(fd, back.data(), back.size(), 0) ==
          static_cast<FsSsize>(back.size()));
  REQUIRE(back == want);
  REQUIRE(cfs->CloseFd(fd) == 0);
}

/** Child: write every file. Synced files are fsynced; one is not. */
void WritePhase(clio::cte::filesystem::Client *cfs) {
  WriteFile(cfs, kSynced, Pattern(kSyncedBytes, 1), true);

  int fd = cfs->OpenFd(kTrunc, O_CREAT | O_RDWR | O_TRUNC, 0644);
  REQUIRE(fd >= 0);
  const std::vector<char> t = Pattern(kTruncWritten, 2);
  REQUIRE(cfs->WriteFd(fd, t.data(), t.size()) ==
          static_cast<FsSsize>(t.size()));
  REQUIRE(cfs->FtruncateFd(fd, kTruncLength) == 0);
  REQUIRE(cfs->SyncFd(fd) == 0);
  REQUIRE(cfs->CloseFd(fd) == 0);

  WriteFile(cfs, kRenameSrc, Pattern(kSmallBytes, 3), true);
  REQUIRE(cfs->RenamePath(kRenameSrc, kRenameDst) == 0);
  fd = cfs->OpenFd(kRenameDst, O_RDWR, 0);
  REQUIRE(fd >= 0);
  REQUIRE(cfs->SyncFd(fd) == 0);  // makes the new name durable
  REQUIRE(cfs->CloseFd(fd) == 0);

  WriteFile(cfs, kUnsynced, Pattern(kSmallBytes, 4), false);
}

/** Child: every fsynced file is back, byte for byte. */
void VerifyPhase(clio::cte::filesystem::Client *cfs) {
  CheckFile(cfs, kSynced, Pattern(kSyncedBytes, 1));
  std::vector<char> t = Pattern(kTruncWritten, 2);
  t.resize(kTruncLength);
  CheckFile(cfs, kTrunc, t);
  CheckFile(cfs, kRenameDst, Pattern(kSmallBytes, 3));
  bool exists = true;
  (void)SizeOf(cfs, kRenameSrc, &exists);
  REQUIRE_FALSE(exists);
  if (ctp::SystemInfo::Getenv(kExpectUnsyncedEnv) == "1") {
    CheckFile(cfs, kUnsynced, Pattern(kSmallBytes, 4));
  }
}

/**
 * Driver: start a fresh runtime, write, stop it, restart it with recovery,
 * verify, stop.
 * @param graceful true: `clio_run stop`; false: kill the runtime
 */
void RunRestartCycle(bool graceful) {
  const fs::path work = WorkDir();
  fs::remove_all(work);
  fs::create_directories(work);
  const std::string conf = WriteConfig(work);
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", conf, 1);
  ctp::SystemInfo::Setenv("CLIO_STORAGE_ROOT", work.generic_string(), 1);
  ctp::SystemInfo::Setenv("CLIO_WAIT_SERVER", "30", 1);
  ctp::SystemInfo::Setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);
  ctp::SystemInfo::Setenv(kExpectUnsyncedEnv, graceful ? "1" : "0", 1);

  {
    // Each start keeps its own log in the work dir (left behind on failure).
    ctp::SystemInfo::Setenv("CLIO_TEST_SERVER_LOG",
                            (work / "runtime_first.log").string(), 1);
    clio::run::test::RuntimeServer server;
    REQUIRE(server.Start(kPort));
    REQUIRE(server.WaitForReady());
    REQUIRE(RunPhase("write", work) == 0);
    if (graceful) {
      const std::string log = (work / "stop.log").string();
      REQUIRE(RunToCompletion(CLIO_RUN_EXE, {"stop", "--grace-period", "10000"},
                              log, 120) == 0);
      REQUIRE(WaitDaemonExit(&server, 120000));
    }
    server.Stop();  // the crash case: kill it with everything in flight
  }
  {
    ctp::SystemInfo::Setenv("CLIO_TEST_SERVER_LOG",
                            (work / "runtime_recovered.log").string(), 1);
    clio::run::test::RuntimeServer server;
    REQUIRE(server.Start(kPort, "127.0.0.1", false, false, /*recover=*/true));
    REQUIRE(server.WaitForReady());
    REQUIRE(RunPhase("verify", work) == 0);
    const std::string log = (work / "stop2.log").string();
    (void)RunToCompletion(CLIO_RUN_EXE, {"stop", "--grace-period", "2000"},
                          log, 120);
    (void)WaitDaemonExit(&server, 60000);
    server.Stop();
  }
  ctp::SystemInfo::Unsetenv("CLIO_TEST_SERVER_LOG");
  fs::remove_all(work);
}

}  // namespace

TEST_CASE("cfs persistent config: phase body (runs in a child process)",
          "[persist_child]") {
  const std::string phase = ctp::SystemInfo::Getenv(kPhaseEnv);
  if (phase.empty()) return;  // only meaningful when spawned by a driver
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false));
  // Binds the filesystem pool (the fd calls do it lazily; GetAttr does not).
  REQUIRE(clio::cte::filesystem::CLIO_CFS_CLIENT_INIT());
  auto *cfs = CLIO_CFS_CLIENT;
  REQUIRE(cfs != nullptr);
  if (phase == "write") {
    WritePhase(cfs);
  } else {
    VerifyPhase(cfs);
  }
}

TEST_CASE("cfs persistent config: files survive a graceful stop",
          "[cfs][persist]") {
  RunRestartCycle(/*graceful=*/true);
}

TEST_CASE("cfs persistent config: fsynced files survive a runtime crash",
          "[cfs][persist]") {
  RunRestartCycle(/*graceful=*/false);
}

SIMPLE_TEST_MAIN()
