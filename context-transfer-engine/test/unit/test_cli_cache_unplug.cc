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
 * `clio_run pool stop|start|rm <cache>` end to end, against a real daemon.
 *
 * The orchestrator test case spawns a `clio_run` daemon on its own port,
 * composes a persistent file-backed core (WAL on), a RAM volatile core and a
 * write-back cache that never flushes by itself, and then drives three
 * phases. The client phases run as fresh processes (this binary re-launched
 * with a test-name filter and CLIO_CACHE_UNPLUG_PHASE set), because a process
 * can attach to one runtime only:
 *   1. write   - blobs go through the cache; they are dirty, not durable;
 *   2. the orchestrator runs `clio_run pool stop <cache>` and checks its exit
 *      code and that it reports the flushed blob count; then `pool start`
 *      brings the cache back from the pool log, `pool stop` stops it again,
 *      and `pool rm` removes the stopped pool (re-created first); the daemon
 *      is then stopped (the job "ends");
 *   3. verify  - a second daemon is started on the same storage (a recovering
 *      start, not --fresh); every blob must read back exactly from the
 *      persistent core.
 * A second case checks that `clio_run stop` without a pool still stops the
 * whole runtime.
 */

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "clio_ctp/serialize/msgpack_wrapper.h"

#include "clio_cte/core/core_client.h"
#include "clio_ctp/introspect/system_info.h"
#include "clio_runtime/clio_runtime.h"
#include "runtime_server.h"
#include "simple_test.h"

/**
 * Convert a msgpack object to T (the CTP msgpack shim has convert(), not as<>).
 * @param o the object
 * @return its value as T
 */
template <typename T>
T MsgAs(const msgpack::object &o) {
  T out{};
  o.convert(out);
  return out;
}


namespace fs = std::filesystem;
using clio::run::PoolId;
using clio::run::u32;
using clio::run::u64;

namespace {

/** Own port (RESOURCE_LOCK keeps other daemon tests off it). */
constexpr unsigned kPort = 10626;
/** Environment variable naming the client phase a re-launched binary runs. */
const char kPhaseEnv[] = "CLIO_CACHE_UNPLUG_PHASE";
const PoolId kPersistentPool(512, 0);
const PoolId kCachePool(567, 0);
const char kCacheName[] = "unplug_cache";
constexpr int kBlobs = 24;
constexpr u64 kKiB = 1024;
const u64 kSizes[] = {4 * kKiB, 12345, 64 * kKiB, 200000, 512 * kKiB,
                      1024 * kKiB};

/** @return the client phase this process was launched for ("" = orchestrator) */
std::string Phase() {
  const char *p = std::getenv(kPhaseEnv);
  return p == nullptr ? std::string() : std::string(p);
}

/** @return the directory the orchestrator keeps this run's storage in */
std::string WorkDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  const std::string base = (d && *d) ? d : ".";
  return base + "/cache_unplug_cli_work";
}

/** Deterministic payload; `seed` changes every byte. */
std::string Payload(u64 size, char seed) {
  std::string v(size, '\0');
  for (u64 i = 0; i < size; ++i) {
    v[i] = static_cast<char>(seed + (i * 7) % 61);
  }
  return v;
}

/** @return the expected content of blob `i` (name and bytes are a function of i) */
std::string BlobData(int i) {
  return Payload(kSizes[i % (sizeof(kSizes) / sizeof(kSizes[0]))],
                 static_cast<char>('A' + i % 40));
}

/** @return the name of blob `i` */
std::string BlobName(int i) { return "cli_blob_" + std::to_string(i); }

/**
 * Write the compose YAML: persistent file core (WAL on), RAM volatile core
 * and a write-back cache with every flush trigger off.
 * @param work directory for the file tier and metadata log
 * @param yaml path to write
 */
void WriteCompose(const std::string &work, const std::string &yaml) {
  std::ofstream f(yaml);
  REQUIRE(f.is_open());
  f << "compose:\n"
       "  - mod_name: clio_cte_core\n"
       "    pool_name: \"cache_unplug_cli_persistent\"\n"
       "    pool_query: local\n"
       "    restart: true\n"
       "    pool_id: \"512.0\"\n"
       "    durability: persistent\n"
       "    storage:\n"
       "      - path: \"" << work << "/file_dev.dat\"\n"
       "        bdev_type: file\n"
       "        capacity_limit: 256mb\n"
       "        score: 0.0\n"
       "        persistence_level: long_term\n"
       "    performance:\n"
       "      metadata_log_path: \"" << work << "/meta_log\"\n"
       "  - mod_name: clio_cte_core\n"
       "    pool_name: \"cache_unplug_cli_volatile\"\n"
       "    pool_query: local\n"
       "    pool_id: \"540.0\"\n"
       "    durability: volatile\n"
       "    storage:\n"
       "      - path: \"ram::cache_unplug_cli_ram\"\n"
       "        bdev_type: ram\n"
       "        capacity_limit: 128mb\n"
       "        score: 1.0\n"
       "  - mod_name: clio_cte_core_persist_write_back\n"
       "    pool_name: \"" << kCacheName << "\"\n"
       "    pool_query: local\n"
       "    pool_id: \"567.0\"\n"
       "    restart: true\n"
       "    next_volatile: \"540.0\"\n"
       "    next_persistent: \"512.0\"\n"
       "    flush_period_ms: 0\n"
       "    flush_fill_pct: 0\n"
       "    flush_dirty_mb: 0\n";
}

/** Read a whole file ("" if missing). */
std::string ReadFile(const std::string &path) {
  std::ifstream f(path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

/**
 * Run an executable to completion, output captured in `log`.
 * @param exe program to run
 * @param args its arguments
 * @param log file receiving stdout and stderr
 * @param timeout_sec bound on the run
 * @return the exit code, -1 if it did not start, -3 on timeout
 */
int RunProc(const std::string &exe, const std::vector<std::string> &args,
            const std::string &log, int timeout_sec) {
  ctp::SpawnedProcess proc = ctp::SystemInfo::SpawnProcess(exe, args, log);
  if (!proc.valid) return -1;
  int rc = 0;
  if (!ctp::SystemInfo::WaitForChild(proc, timeout_sec * 1000, &rc)) return -3;
  return rc;
}

/** Run `clio_run <args>`; output lands in `log`. */
int RunCli(const std::vector<std::string> &args, const std::string &log,
           int timeout_sec) {
  return RunProc(CLIO_RUN_EXE, args, log, timeout_sec);
}

/**
 * Re-launch this binary to run one client phase in a fresh process.
 * @param phase "write" or "verify"
 * @param test_filter name filter selecting that phase's test case
 * @param log file receiving the child's output
 * @return the child's exit code
 */
int RunPhase(const std::string &phase, const std::string &test_filter,
             const std::string &log) {
  ctp::SystemInfo::Setenv(kPhaseEnv, phase, 1);
  const int rc = RunProc(ctp::SystemInfo::GetExecutablePath(), {test_filter},
                         log, 240);
  ctp::SystemInfo::Unsetenv(kPhaseEnv);
  return rc;
}

/** Set the environment every daemon, CLI and client of this test shares. */
void SetCommonEnv() {
  ctp::SystemInfo::Setenv("CLIO_WAIT_SERVER", "15", 1);
  ctp::SystemInfo::Setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);
}

/**
 * Make the daemon hermetic: it reads `conf` as its server config (instead of
 * falling back to the user's ~/.clio/clio.yaml, whose own 512.0 would answer
 * the test's pools) and keeps every runtime-owned file under `work`.
 * @param work the run's storage directory
 * @param conf server config to use (its compose is the daemon's own)
 */
void SetServerEnv(const std::string &work, const std::string &conf) {
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", conf.c_str(), 1);
  ctp::SystemInfo::Setenv("CLIO_STORAGE_ROOT", work.c_str(), 1);
}

/** Undo SetServerEnv. */
void ClearServerEnv() {
  ctp::SystemInfo::Unsetenv("CLIO_SERVER_CONF");
  ctp::SystemInfo::Unsetenv("CLIO_STORAGE_ROOT");
}

/** Connect this process to the daemon as a client. */
void ConnectClient() {
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false));
}

/** A tag on the persistent core (the tag authority); same id after a restart. */
clio::cte::core::TagId MakeTag() {
  clio::cte::core::Client pers(kPersistentPool);
  auto f = pers.AsyncGetOrCreateTag("cache_unplug_cli_tag");
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  return f->tag_id_;
}

/** Blob size on a pool, or -1 when it does not have the blob. */
long long SizeOn(const PoolId &pool, const clio::cte::core::TagId &tag,
                 const std::string &name) {
  clio::cte::core::Client c(pool);
  auto f = c.AsyncGetBlobSize(tag, name, clio::run::PoolQuery::Local());
  f.Wait();
  return f->GetReturnCode() == 0 ? static_cast<long long>(f->size_) : -1;
}

/** Read the whole blob from a pool; empty on failure. */
std::string GetBlob(const PoolId &pool, const clio::cte::core::TagId &tag,
                    const std::string &name, u64 size) {
  clio::cte::core::Client c(pool);
  std::string out(size, '\0');
  auto f = c.AsyncGetBlob(tag, name, 0, size, 0, out.data());
  f.Wait();
  if (f.get() == nullptr || f->GetReturnCode() != 0) return std::string();
  return out;
}

/** The cache's "dirty_blobs" stat. */
u64 CacheDirtyBlobs() {
  clio::cte::core::Client cache(kCachePool);
  auto f = cache.AsyncMonitor(clio::run::PoolQuery::Local(), "stats");
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  u64 dirty = ~u64{0};
  for (const auto &kv : f->results_) {
    msgpack::object_handle oh =
        msgpack::unpack(kv.second.data(), kv.second.size());
    const msgpack::object &o = oh.get();
    REQUIRE(o.type == msgpack::type::MAP);
    for (uint32_t i = 0; i < o.via.map.size; ++i) {
      if (MsgAs<std::string>(o.via.map.ptr[i].key) == "dirty_blobs") {
        dirty = MsgAs<u64>(o.via.map.ptr[i].val);
      }
    }
  }
  return dirty;
}

/** Wait until the daemon exits on its own; true if it did. */
bool WaitDaemonGone(clio::run::test::RuntimeServer *server, int seconds) {
  for (int i = 0; i < seconds * 10 && server->IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return !server->IsRunning();
}

}  // namespace

TEST_CASE("cli_phase_write: blobs through the cache are dirty, not durable",
          "[cte][cache_unplug][cli][phase]") {
  if (Phase() != "write") return;
  ConnectClient();
  const auto tag = MakeTag();
  clio::cte::core::Client cache(kCachePool);
  for (int i = 0; i < kBlobs; ++i) {
    const std::string data = BlobData(i);
    auto f = cache.AsyncPutBlob(tag, BlobName(i), 0, data.size(), data.data());
    f.Wait();
    REQUIRE(f->GetReturnCode() == 0);
  }
  // A partial overwrite of a buffered blob: still one dirty blob.
  const std::string patch = Payload(300, 'q');
  auto f = cache.AsyncPutBlob(tag, BlobName(0), 50, patch.size(), patch.data());
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);

  REQUIRE(CacheDirtyBlobs() == (u64)kBlobs);
  for (int i = 0; i < kBlobs; ++i) {
    REQUIRE(SizeOn(kPersistentPool, tag, BlobName(i)) == -1);
  }
}

TEST_CASE("cli_phase_verify: every blob reads back exactly from persistent",
          "[cte][cache_unplug][cli][phase]") {
  if (Phase() != "verify") return;
  ConnectClient();
  const auto tag = MakeTag();
  for (int i = 0; i < kBlobs; ++i) {
    std::string want = BlobData(i);
    if (i == 0) want.replace(50, 300, Payload(300, 'q'));
    INFO("blob " << BlobName(i));
    REQUIRE(SizeOn(kPersistentPool, tag, BlobName(i)) == (long long)want.size());
    REQUIRE(GetBlob(kPersistentPool, tag, BlobName(i), want.size()) == want);
  }
}

TEST_CASE("cli: pool stop <cache> flushes; the data survives the runtime ending",
          "[cte][cache_unplug][cli]") {
  if (!Phase().empty()) return;
  const std::string work = WorkDir();
  std::error_code ec;
  fs::remove_all(work, ec);
  fs::create_directories(work);
  // The stack is the daemon's own compose: a restart re-creates it and the
  // persistent core (restart: true) replays its metadata log.
  const std::string yaml = work + "/server.yaml";
  WriteCompose(work, yaml);
  SetCommonEnv();
  SetServerEnv(work, yaml);

  {
    clio::run::test::RuntimeServer server;
    REQUIRE(server.Start(kPort));
    REQUIRE(server.WaitForReady());
    REQUIRE(RunPhase("write", "cli_phase_write", work + "/write.log") == 0);

    // A pool that is not running cannot be stopped.
    REQUIRE(RunCli({"pool", "stop", "no_such_pool"}, work + "/stop_bad.log",
                   60) != 0);
    const std::string stop_log = work + "/stop.log";
    REQUIRE(RunCli({"pool", "stop", kCacheName}, stop_log, 90) == 0);
    const std::string out = ReadFile(stop_log);
    INFO("clio_run stop output: " << out);
    if (clio::run::test::RuntimeServer::AdminPoolMarkerPossible()) {
      // The report is a kSuccess line; conservatively require INFO output.
      REQUIRE(out.find("flushed " + std::to_string(kBlobs) + " blobs") !=
              std::string::npos);
    }
    // Stopped, so a second stop finds no running pool.
    REQUIRE(RunCli({"pool", "stop", kCacheName}, work + "/stop_again.log",
                   60) != 0);
    // `pool start` re-creates the stopped cache from the pool log...
    REQUIRE(RunCli({"pool", "start", kCacheName}, work + "/start.log", 60) ==
            0);
    REQUIRE(RunCli({"pool", "start", kCacheName}, work + "/start_again.log",
                   60) == 0);  // already running: not an error
    // ...it runs again (so it can be stopped), and nothing is left to flush.
    REQUIRE(RunCli({"pool", "stop", kCacheName}, work + "/stop2.log", 90) ==
            0);
    if (clio::run::test::RuntimeServer::AdminPoolMarkerPossible()) {
      REQUIRE(ReadFile(work + "/stop2.log").find("flushed 0 blobs") !=
              std::string::npos);
    }
    // `pool rm` on a stopped pool re-creates it, then deletes its data and
    // forgets it: afterwards neither stop nor start can find it.
    REQUIRE(RunCli({"pool", "rm", kCacheName}, work + "/rm.log", 90) == 0);
    REQUIRE(RunCli({"pool", "start", kCacheName}, work + "/start_gone.log",
                   60) != 0);
    // An unknown pool subcommand is rejected.
    REQUIRE(RunCli({"pool", "bogus", kCacheName}, work + "/bogus.log", 30) !=
            0);
    // The job ends: the runtime goes away without any further flush.
    server.Stop();
  }

  // Restart on the same storage (a recovering start, not --fresh).
  clio::run::test::RuntimeServer restarted;
  REQUIRE(restarted.Start(kPort, "127.0.0.1", /*ephemeral=*/false,
                          /*detached=*/false, /*recover=*/true));
  REQUIRE(restarted.WaitForReady());
  REQUIRE(RunPhase("verify", "cli_phase_verify", work + "/verify.log") == 0);
  REQUIRE(RunCli({"stop", "--grace-period", "2000"}, work + "/final_stop.log",
                 90) == 0);
  WaitDaemonGone(&restarted, 60);
  restarted.Stop();
  ClearServerEnv();
  fs::remove_all(work, ec);
}

TEST_CASE("cli: stop with no pool still stops the whole runtime",
          "[cte][cache_unplug][cli][runtime_stop]") {
  if (!Phase().empty()) return;
  const std::string work = WorkDir();
  std::error_code ec;
  fs::remove_all(work, ec);
  fs::create_directories(work);
  SetCommonEnv();
  const std::string yaml = work + "/server.yaml";
  WriteCompose(work, yaml);
  SetServerEnv(work, yaml);
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort));
  REQUIRE(server.WaitForReady());
  REQUIRE(RunCli({"stop", "--grace-period", "2000"}, work + "/stop_rt.log",
                 90) == 0);
  REQUIRE(WaitDaemonGone(&server, 60));
  server.Stop();
  ClearServerEnv();
  fs::remove_all(work, ec);
}

SIMPLE_TEST_MAIN()
