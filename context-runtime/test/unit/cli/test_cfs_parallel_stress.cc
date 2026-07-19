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

// Parallel filesystem-adapter (CFS chimod) overlap stress.
//
// Many threads hammer read / write / append / delete-file / delete-directory /
// list at random over a DELIBERATELY TINY namespace -- 4 shared files and 4
// shared directories -- so every op contends with concurrent ops on the SAME
// path. This maximizes overlap on the CFS metadata (by_path_ handle map, tag
// resolution, the deferred-append pipeline, directory markers) to shake out
// races, use-after-free, and deadlocks.
//
// Success metric (per the request): NO crashes, NO hangs, NO memory leaks.
//   - crashes  -> the runtime/client aborts and the test fails to finish.
//   - hangs    -> every op is a bounded Async+Wait and all threads join; a hang
//                 trips the ctest per-test TIMEOUT.
//   - leaks    -> every buffer allocated is freed on every path, and every open
//                 handle is closed; run under ASan/MSan for enforcement.
// Individual ops are NOT required to succeed: a read/append/unlink/rmdir may
// legitimately race a concurrent delete and return non-zero. We only assert the
// system stays alive and responsive.
//
// Env knobs: STRESS_THREADS (8), STRESS_ITERS (1500).

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/filesystem/filesystem_client.h>

#include "runtime_server.h"
#include "simple_test.h"

namespace fs = std::filesystem;

namespace {

constexpr int kNumFiles = 4;
constexpr int kNumDirs = 4;

int FromEnv(const char *name, int dflt) {
  if (const char *e = std::getenv(name)) {
    int n = std::atoi(e);
    if (n > 0) return n;
  }
  return dflt;
}

int RunCliTimed(const std::vector<std::string> &args, int timeout_sec) {
  std::vector<std::string> full;
  full.push_back(CLIO_RUN_EXE);
  full.insert(full.end(), args.begin(), args.end());
  std::vector<char *> argv;
  for (auto &a : full) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    int n = open("/dev/null", O_WRONLY);
    if (n >= 0) { dup2(n, 1); dup2(n, 2); close(n); }
    execv(argv[0], argv.data());
    _exit(127);
  }
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
  int status = 0;
  while (true) {
    pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return -3;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

std::string FilePath(int k) { return "/pf" + std::to_string(k); }
std::string DirPath(int k) { return "/pd" + std::to_string(k); }

}  // namespace

TEST_CASE("CfsParallelStress - read/write/append/unlink/rmdir/list race on a "
          "tiny shared namespace (no crash/hang/leak)",
          "[cli][cfs][concurrent][stress]") {
  constexpr unsigned kPort = 10609;
  const fs::path work = fs::temp_directory_path() / "cfs_parallel_stress";
  fs::remove_all(work);
  fs::create_directories(work);

  const fs::path yaml = work / "compose.yaml";
  {
    std::ofstream f(yaml);
    f << "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: \"cfs_cte\"\n"
         "    pool_query: local\n"
         "    pool_id: \"512.0\"\n"
         "    storage:\n"
         "      - path: " << (work / "ram_dev").string() << "\n"
         "        bdev_type: ram\n"
         "        capacity_limit: 256mb\n"
         "    dpe:\n"
         "      dpe_type: random\n"
         "  - mod_name: clio_cte_filesystem\n"
         "    pool_name: \"cfs\"\n"
         "    pool_query: local\n"
         "    pool_id: \"600.0\"\n"
         "    next_pool_id: \"512.0\"\n";
  }

  setenv("CLIO_WAIT_SERVER", "15", 1);
  setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);

  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort));
  REQUIRE(server.WaitForReady());
  REQUIRE(RunCliTimed({"compose", "start", yaml.string()}, 60) == 0);

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false));
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  clio::cte::filesystem::Client cfs;
  cfs.Init(clio::run::PoolId(600, 0));

  // Seed the tiny shared namespace: 4 files + 4 dirs. (Best-effort; the stress
  // loop re-creates them anyway.)
  for (int k = 0; k < kNumFiles; ++k) {
    auto op = cfs.AsyncOpen(FilePath(k), O_CREAT | O_RDWR, 0644);
    op.Wait();
    if (op->GetReturnCode() == 0) cfs.AsyncClose(op->handle_).Wait();
  }
  for (int k = 0; k < kNumDirs; ++k) cfs.AsyncMkdir(DirPath(k)).Wait();

  const int kThreads = FromEnv("STRESS_THREADS", 8);
  const int kIters = FromEnv("STRESS_ITERS", 1500);
  constexpr clio::run::u64 kMaxIo = 8 * 1024;  // 8 KiB

  std::atomic<bool> failed{false};
  std::atomic<long> total_ops{0};

  auto worker = [&](int tid) {
    std::mt19937 rng(0xF00D ^ (tid * 2654435761u));
    std::uniform_int_distribution<int> fdist(0, kNumFiles - 1);
    std::uniform_int_distribution<int> ddist(0, kNumDirs - 1);
    std::uniform_int_distribution<int> opdist(0, 6);
    std::uniform_int_distribution<clio::run::u64> szdist(1, kMaxIo);

    for (int i = 0; i < kIters && !failed.load(std::memory_order_relaxed); ++i) {
      const int op = opdist(rng);
      switch (op) {
        case 0: {  // write: open(create) -> write -> close
          const std::string p = FilePath(fdist(rng));
          const clio::run::u64 sz = szdist(rng);
          auto o = cfs.AsyncOpen(p, O_CREAT | O_RDWR, 0644);
          o.Wait();
          if (o->GetReturnCode() == 0) {
            ctp::ipc::FullPtr<char> b = ipc->AllocateBuffer(sz);
            if (b.IsNull()) { failed.store(true); return; }
            std::memset(b.ptr_, static_cast<int>(tid + i), sz);
            cfs.AsyncWrite(o->handle_, 0, sz, b.shm_.template Cast<void>())
                .Wait();
            ipc->FreeBuffer(b);
            cfs.AsyncClose(o->handle_).Wait();
          }
          break;
        }
        case 1: {  // read: open -> read -> close
          const std::string p = FilePath(fdist(rng));
          const clio::run::u64 sz = szdist(rng);
          auto o = cfs.AsyncOpen(p, O_RDWR, 0644);
          o.Wait();
          if (o->GetReturnCode() == 0 && o->handle_ != 0) {
            ctp::ipc::FullPtr<char> b = ipc->AllocateBuffer(sz);
            if (b.IsNull()) { failed.store(true); return; }
            std::memset(b.ptr_, 0, sz);
            cfs.AsyncRead(o->handle_, 0, sz, b.shm_.template Cast<void>())
                .Wait();
            ipc->FreeBuffer(b);
            cfs.AsyncClose(o->handle_).Wait();
          }
          break;
        }
        case 2: {  // append: open(create) -> append -> close
          const std::string p = FilePath(fdist(rng));
          const clio::run::u64 sz = szdist(rng);
          auto o = cfs.AsyncOpen(p, O_CREAT | O_RDWR, 0644);
          o.Wait();
          if (o->GetReturnCode() == 0) {
            ctp::ipc::FullPtr<char> b = ipc->AllocateBuffer(sz);
            if (b.IsNull()) { failed.store(true); return; }
            std::memset(b.ptr_, static_cast<int>(i), sz);
            cfs.AsyncAppend(o->handle_, sz, b.shm_.template Cast<void>()).Wait();
            ipc->FreeBuffer(b);
            cfs.AsyncClose(o->handle_).Wait();
          }
          break;
        }
        case 3: {  // delete file
          cfs.AsyncUnlink(FilePath(fdist(rng))).Wait();
          break;
        }
        case 4: {  // delete directory
          cfs.AsyncRmdir(DirPath(ddist(rng))).Wait();
          break;
        }
        case 5: {  // list (readdir) — root and a subdir
          cfs.AsyncReaddir("/").Wait();
          cfs.AsyncReaddir(DirPath(ddist(rng))).Wait();
          break;
        }
        case 6: {  // replenish a dir so rmdir keeps having targets
          cfs.AsyncMkdir(DirPath(ddist(rng))).Wait();
          break;
        }
        default:
          break;
      }
      total_ops.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);
  for (auto &t : threads) t.join();

  // Reaching here without a crash / deadlock is the pass condition.
  REQUIRE_FALSE(failed.load());
  REQUIRE(total_ops.load() > 0);

  server.Stop();
}

SIMPLE_TEST_MAIN()
