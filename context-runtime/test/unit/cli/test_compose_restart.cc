/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Integration tests for `clio_run compose start/stop/rm/list` and the restart
 * write-ahead log. Each test drives the real clio_run binary against a live
 * daemon (clio::run::test::RuntimeServer) and asserts on the captured output
 * of `compose list` / `compose list --restartable`.
 *
 *   1. ComposeRestart_RestartSurvives — compose a bdev with `restart: true`,
 *      stop the daemon, start a fresh one; the bdev is auto-restarted from the
 *      WAL, so `compose list` shows it again.
 *   2. ComposeRestart_StopKeepsRestartable — `compose stop` removes the bdev
 *      from `compose list` but `compose list --restartable` still shows it.
 *   3. ComposeRestart_RmUnregisters — `compose rm` removes the bdev from both
 *      `compose list` and `compose list --restartable`.
 *
 * `compose list` output is asserted via fresh clio_run subprocesses (each is a
 * new client), which sidesteps in-process client reconnection across a daemon
 * restart. HOME is redirected to the test's work dir so the WAL lives at
 * <work>/.clio/restart_log.bin and never touches the developer's ~/.clio.
 */

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>

#include "runtime_server.h"
#include "simple_test.h"

namespace fs = std::filesystem;

namespace {

// Run the clio_run binary with a hard kill deadline. If capture_path is
// non-empty, the child's stdout is redirected there (stderr always silenced).
int RunCliTimed(const std::vector<std::string>& args, int timeout_sec,
                const std::string& capture_path = "") {
  std::vector<std::string> full;
  full.push_back(CLIO_RUN_EXE);
  full.insert(full.end(), args.begin(), args.end());
  std::vector<char*> argv;
  argv.reserve(full.size() + 1);
  for (auto& a : full) {
    argv.push_back(a.data());
  }
  argv.push_back(nullptr);

  pid_t pid = fork();
  if (pid < 0) {
    return -1;
  }
  if (pid == 0) {
    int out_fd = capture_path.empty()
                     ? open("/dev/null", O_WRONLY)
                     : open(capture_path.c_str(),
                            O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int null_fd = open("/dev/null", O_WRONLY);
    if (out_fd >= 0) {
      dup2(out_fd, 1);
    }
    if (null_fd >= 0) {
      dup2(null_fd, 2);
    }
    execv(argv[0], argv.data());
    _exit(127);
  }

  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
  int status = 0;
  while (true) {
    pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) {
      if (WIFEXITED(status)) return WEXITSTATUS(status);
      return -2;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return -3;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

// Run a clio_run command and return its captured stdout.
std::string RunCliCapture(const std::vector<std::string>& args,
                          int timeout_sec) {
  const fs::path tmp =
      fs::temp_directory_path() / "clio_compose_list_capture.txt";
  RunCliTimed(args, timeout_sec, tmp.string());
  std::ifstream f(tmp);
  std::stringstream ss;
  ss << f.rdbuf();
  fs::remove(tmp);
  return ss.str();
}

void WaitForExit(clio::run::test::RuntimeServer& server) {
  for (int i = 0; i < 600 && server.IsRunning(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

// Write a single-file-bdev compose file. pool_id is fixed at 720.0 so tests
// can assert presence by id regardless of the (path-valued) pool name.
void WriteBdevCompose(const fs::path& yaml, const fs::path& work_dir,
                      bool restart) {
  std::ofstream f(yaml);
  f << "compose:\n"
       "  - mod_name: clio_bdev\n"
       "    pool_name: \""
    << (work_dir / "bdev0.dat").string()
    << "\"\n"
       "    pool_query: local\n"
       "    pool_id: \"720.0\"\n"
       "    bdev_type: file\n"
       "    capacity: \"16MB\"\n"
       "    restart: "
    << (restart ? "true" : "false")
    << "\n"
       "    alloc_log: \""
    << (work_dir / "bdev0.alog").string() << "\"\n";
}

// Set up an isolated work dir + HOME (so the restart WAL is sandboxed).
fs::path SetupWork(const std::string& name) {
  const fs::path work = fs::temp_directory_path() / name;
  fs::remove_all(work);
  fs::create_directories(work);
  setenv("HOME", work.c_str(), 1);
  setenv("CLIO_WAIT_SERVER", "15", 1);
  setenv("CLIO_BIND_ADDR", "127.0.0.1", 1);
  return work;
}

}  // namespace

TEST_CASE("ComposeRestart_RestartSurvives - restart:true survives daemon "
          "restart",
          "[cli][compose][restart]") {
  const fs::path work = SetupWork("clio_compose_restart_1");
  const fs::path yaml = work / "bdev.yaml";
  WriteBdevCompose(yaml, work, /*restart=*/true);
  constexpr unsigned kPort = 10610;

  // Phase 1: daemon up, compose the restartable bdev.
  clio::run::test::RuntimeServer s1;
  REQUIRE(s1.Start(kPort));
  REQUIRE(s1.WaitForReady());
  REQUIRE(RunCliTimed({"compose", "start", yaml.string()}, 60) == 0);
  {
    std::string out = RunCliCapture({"compose", "list"}, 30);
    REQUIRE(out.find("720.0") != std::string::npos);
  }

  // Phase 2: stop the daemon cleanly.
  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);
  WaitForExit(s1);
  s1.Stop();

  // Phase 3: fresh daemon. No re-compose — startup WAL replay must bring the
  // bdev back, so `compose list` shows pool 720.0 again.
  clio::run::test::RuntimeServer s2;
  REQUIRE(s2.Start(kPort));
  REQUIRE(s2.WaitForReady());
  {
    std::string out = RunCliCapture({"compose", "list"}, 30);
    REQUIRE(out.find("720.0") != std::string::npos);
  }

  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);
  WaitForExit(s2);
  s2.Stop();
  fs::remove_all(work);
}

TEST_CASE("ComposeRestart_StopKeepsRestartable - stop drops from list but "
          "keeps restartable",
          "[cli][compose][restart]") {
  const fs::path work = SetupWork("clio_compose_restart_2");
  const fs::path yaml = work / "bdev.yaml";
  WriteBdevCompose(yaml, work, /*restart=*/true);
  constexpr unsigned kPort = 10611;

  clio::run::test::RuntimeServer s;
  REQUIRE(s.Start(kPort));
  REQUIRE(s.WaitForReady());

  REQUIRE(RunCliTimed({"compose", "start", yaml.string()}, 60) == 0);
  REQUIRE(RunCliTimed({"compose", "stop", yaml.string()}, 60) == 0);

  // Active list no longer has the bdev...
  {
    std::string out = RunCliCapture({"compose", "list"}, 30);
    REQUIRE(out.find("720.0") == std::string::npos);
  }
  // ...but it is still registered for restart.
  {
    std::string out = RunCliCapture({"compose", "list", "--restartable"}, 30);
    REQUIRE(out.find("bdev.yaml") != std::string::npos);
  }

  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);
  WaitForExit(s);
  s.Stop();
  fs::remove_all(work);
}

TEST_CASE("ComposeRestart_RmUnregisters - rm drops from list AND restartable",
          "[cli][compose][restart]") {
  const fs::path work = SetupWork("clio_compose_restart_3");
  const fs::path yaml = work / "bdev.yaml";
  WriteBdevCompose(yaml, work, /*restart=*/true);
  constexpr unsigned kPort = 10612;

  clio::run::test::RuntimeServer s;
  REQUIRE(s.Start(kPort));
  REQUIRE(s.WaitForReady());

  REQUIRE(RunCliTimed({"compose", "start", yaml.string()}, 60) == 0);
  // rm = stop the pool AND unregister it from restart (file is NOT deleted).
  REQUIRE(RunCliTimed({"compose", "rm", yaml.string()}, 60) == 0);
  REQUIRE(fs::exists(yaml));  // compose file itself survives rm

  {
    std::string out = RunCliCapture({"compose", "list"}, 30);
    REQUIRE(out.find("720.0") == std::string::npos);
  }
  {
    std::string out = RunCliCapture({"compose", "list", "--restartable"}, 30);
    REQUIRE(out.find("bdev.yaml") == std::string::npos);
  }

  REQUIRE(RunCliTimed({"stop", "--grace-period", "2000"}, 90) == 0);
  WaitForExit(s);
  s.Stop();
  fs::remove_all(work);
}

SIMPLE_TEST_MAIN()
