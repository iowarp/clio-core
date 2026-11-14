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
 * fsync through the descriptor layer (the POSIX/STDIO/MPI-IO interceptors)
 * must put a file's bytes on a persistent device (issue #1188).
 *
 * Layout: a persistent core (file
 * tier) and a volatile core (RAM) behind a write-back cache, with the
 * filesystem over the cache. Writes are acked from the volatile core, so the
 * persistent file tier fills only once fsync makes the cache flush the
 * file's dirty copies -- which also proves fsync takes the filesystem's own
 * chain (learned from its config, no fixed pool id) rather than a guessed
 * core pool. The observable is the persistent tier's free space.
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

namespace {

/** Own port (RESOURCE_LOCK keeps other daemon tests off it). */
constexpr unsigned kPort = 10619;
/** Bytes written and fsynced: several pages. */
constexpr size_t kFileBytes = 3 * 1024 * 1024 + 4096;

/**
 * Run `clio_run <args>` to completion.
 * @param args CLI arguments
 * @param timeout_sec bound on the run
 * @return its exit code, -1 if it did not start, -3 on timeout
 */
int RunCli(const std::vector<std::string> &args, int timeout_sec) {
  const std::string log =
      (fs::temp_directory_path() / "cfs_fsync_durable_cli.log").string();
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
 * Free space left on one target. Blocks a SyncTag moves onto the target are
 * allocated there, so this drops by the bytes moved (the bytes_written_
 * counter only follows puts).
 * @param cte CTE client
 * @param target the target's registered name
 * @return free bytes, or ~0 if the query failed
 */
clio::run::u64 TargetFreeBytes(clio::cte::core::Client *cte,
                                  const std::string &target) {
  auto t = cte->AsyncGetTargetInfo(target, clio::run::PoolQuery::Local());
  t.Wait();
  if (t->GetReturnCode() != 0) return ~clio::run::u64{0};
  return t->remaining_space_;
}

/**
 * The registered name of the target whose name contains `needle` (compose
 * may decorate the configured path).
 * @param cte CTE client
 * @param needle substring of the target's path
 * @return the name, or "" when no target matches
 */
std::string FindTarget(clio::cte::core::Client *cte,
                       const std::string &needle) {
  auto t = cte->AsyncListTargets(clio::run::PoolQuery::Local());
  t.Wait();
  for (const auto &name : t->target_names_) {
    INFO("target: " << name);
    if (name.find(needle) != std::string::npos) return name;
  }
  return "";
}

}  // namespace

TEST_CASE("cfs fsync flushes a file's pages to the persistent core",
          "[cfs][fsync][durable]") {
  std::string user = ctp::SystemInfo::Getenv("USER");
  if (user.empty()) user = "user";
  const fs::path work =
      fs::temp_directory_path() / ("clio_cfs_fsync_durable_" + user);
  fs::remove_all(work);
  fs::create_directories(work);
  const std::string file_dev = (work / "file_dev.dat").string();
  const std::string yaml = (work / "compose.yaml").string();
  {
    std::ofstream f(yaml);
    f << "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: \"cfs_fsync_persistent\"\n"
         "    pool_query: local\n"
         "    pool_id: \"512.0\"\n"
         "    durability: persistent\n"
         "    storage:\n"
         "      - path: \"" << file_dev << "\"\n"
         "        bdev_type: file\n"
         "        capacity_limit: 64mb\n"
         "        score: 0.0\n"
         "        persistence_level: long_term\n"
         "    dpe:\n"
         "      dpe_type: max_bw\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: \"cfs_fsync_volatile\"\n"
         "    pool_query: local\n"
         "    pool_id: \"540.0\"\n"
         "    durability: volatile\n"
         "    storage:\n"
         "      - path: \"ram::cfs_fsync_ram\"\n"
         "        bdev_type: ram\n"
         "        capacity_limit: 64mb\n"
         "        score: 1.0\n"
         "  - mod_name: clio_cte_write_back_cache\n"
         "    pool_name: \"cfs_fsync_cache\"\n"
         "    pool_query: local\n"
         "    pool_id: \"567.0\"\n"
         "    next_volatile: \"540.0\"\n"
         "    next_persistent: \"512.0\"\n"
         "    flush_period_ms: 0\n"
         "  - mod_name: clio_cte_filesystem\n"
         "    pool_name: \"clio_cte_filesystem\"\n"
         "    pool_query: local\n"
         "    pool_id: \"560.0\"\n"
         "    next_pool_id: \"567.0\"\n";
  }
  ctp::SystemInfo::Setenv("CLIO_WAIT_SERVER", "15", 1);
  ctp::SystemInfo::Setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);

  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort));
  REQUIRE(server.WaitForReady());
  REQUIRE(RunCli({"compose", "start", yaml}, 60) == 0);
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false));
  REQUIRE(clio::cte::filesystem::CLIO_CFS_CLIENT_INIT());
  auto *cfs = CLIO_CFS_CLIENT;
  REQUIRE(cfs != nullptr);
  // The filesystem's chain is the cache, learned from its config.
  REQUIRE(cfs->Cte()->pool_id_ == clio::run::PoolId(567, 0));
  // Inspect the persistent core's file tier directly.
  clio::cte::core::Client persistent(clio::run::PoolId(512, 0));
  auto *cte = &persistent;

  const std::string file_target = FindTarget(cte, "file_dev.dat");
  REQUIRE(!file_target.empty());
  const clio::run::u64 before = TargetFreeBytes(cte, file_target);
  REQUIRE(before != ~clio::run::u64{0});

  const std::string path = "clio::" + (work / "durable.bin").string();
  std::vector<char> data(kFileBytes);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<char>((i * 13 + 5) % 251);
  }
  int fd = cfs->OpenFd(path, O_CREAT | O_RDWR | O_TRUNC, 0644);
  REQUIRE(fd >= 0);
  REQUIRE(cfs->WriteFd(fd, data.data(), data.size()) ==
          static_cast<clio::cte::filesystem::FsSsize>(data.size()));

  // Drain only: the pages are dirty in the volatile core and the persistent
  // file tier has not seen them (so the check below is about fsync).
  REQUIRE(cfs->Flush(path.substr(6)) == 0);
  const clio::run::u64 drained = TargetFreeBytes(cte, file_target);
  INFO("file tier free: before=" << before << " after drain=" << drained);
  REQUIRE(before - drained < kFileBytes);

  REQUIRE(cfs->SyncFd(fd) == 0);
  const clio::run::u64 synced = TargetFreeBytes(cte, file_target);
  INFO("file tier free after fsync=" << synced);
  REQUIRE(before - synced >= kFileBytes);

  // The bytes still read back.
  std::vector<char> back(kFileBytes, 0);
  REQUIRE(cfs->PreadFd(fd, back.data(), back.size(), 0) ==
          static_cast<clio::cte::filesystem::FsSsize>(back.size()));
  REQUIRE(back == data);
  REQUIRE(cfs->CloseFd(fd) == 0);

  REQUIRE(RunCli({"stop", "--grace-period", "2000"}, 90) == 0);
  for (int i = 0; i < 600 && server.IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  server.Stop();
  fs::remove_all(work);
}

SIMPLE_TEST_MAIN()
