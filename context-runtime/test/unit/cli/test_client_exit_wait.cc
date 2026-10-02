/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Regression test for issue #970: a task submitted after ClientFinalize must
 * fail, not hang the process in exit() forever.
 *
 * The reported symptom is an HDF5 VOL application that never exits. The cause
 * is an atexit ordering that the connector cannot influence:
 *
 *   1. HDF5 registers atexit(H5_term_library) during its FIRST API call
 *      (hdf5/src/H5.c, H5_init_library).
 *   2. The VOL connector initialises the clio client lazily, inside that same
 *      first H5Fcreate, so clio's atexit handler is registered SECOND.
 *   3. atexit runs handlers in reverse order, so clio tears the client down
 *      FIRST, and H5_term_library then closes the file HDF5 still held open.
 *   4. clio_file_close -> clio_write_stamp submits a DelBlob on the dead client
 *      and waits on a response that can never arrive.
 *
 * Registering clio's handler earlier cannot fix this: earlier registration
 * means later execution, and the connector is not loaded until HDF5 is already
 * initialised.
 *
 * This test reproduces the ordering WITHOUT HDF5, because HDF5 contributes
 * nothing to the mechanism except "a handler registered before clio's". The
 * child registers its own handler before CLIO_INIT and submits a DelBlob from
 * it -- the same call clio_write_stamp makes. Keeping HDF5 out means this runs
 * in every CI configuration, not just the ones that build the VOL.
 *
 * Why the existing suites miss it: vol_compat_suite.py drives the connector
 * through h5py, which closes its objects, so H5_term_library never has a file
 * to close and the exit path is never exercised.
 *
 * The assertion is that the child EXITS. Its exit code additionally
 * distinguishes a real pass from a vacuous one -- a child that could not reach
 * a live runtime reports a distinct code and fails the test rather than
 * "passing" by exiting early.
 *
 * POSIX-only (fork()/waitpid); the whole cli/ directory is gated on NOT WIN32.
 */
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

#include "runtime_server.h"
#include "simple_test.h"

namespace fs = std::filesystem;

namespace {

// Distinct from the other live-runtime cli tests (which also serialize on the
// "clio_runtime" RESOURCE_LOCK).
constexpr unsigned kPort = 10617;

// How long the child gets to exit. The failure mode is an UNBOUNDED wait, so
// any finite budget separates pass from fail; this one is generous enough to
// absorb a slow CI runner without making a real hang cost the whole ctest
// timeout.
constexpr int kExitBudgetSec = 30;

// Child exit codes. Anything other than kOk fails the test with a specific
// cause, so a child that never reached the runtime cannot masquerade as a pass.
constexpr int kOk = 0;
constexpr int kInitFailed = 2;
constexpr int kCteInitFailed = 3;
constexpr int kPutFailed = 4;
constexpr int kAtexitRegFailed = 5;

/** Run the clio_run binary as a subprocess with a timeout; returns its exit
 *  code (negative on spawn/timeout failure). Mirrors test_cte_fallback.cc. */
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

const char *kTagName = "issue970_exit_tag";
const char *kBlobName = "stamp";

clio::cte::core::Client *g_cte = nullptr;
clio::cte::core::TagId g_tag_id;

/**
 * Stands in for H5_term_library. Registered BEFORE CLIO_INIT, so atexit's LIFO
 * ordering runs it AFTER clio's teardown -- exactly HDF5's position.
 *
 * The body is the shape of clio_write_stamp's ambiguous branch
 * (clio_vol.cc, AsyncDelBlob followed by an unconditional Wait). Before the
 * fix this parks forever in IpcCpu2Cpu::RecvOut; after it, Wait returns false
 * and the task carries an error code, which is what the VOL's existing
 * "stamp could not be dropped" warning path already handles.
 */
void LateTeardown() {
  if (g_cte == nullptr) return;
  auto del = g_cte->AsyncDelBlob(g_tag_id, std::string(kBlobName));
  del.Wait();  // must RETURN; the defect is that it never does
}

/** Runs in the forked child. Never returns: ends via exit() so atexit runs. */
[[noreturn]] void ExitClientMain() {
  // Silence the child's client chatter; the parent asserts on its exit code.
  int devnull = open("/dev/null", O_WRONLY);
  if (devnull >= 0) { dup2(devnull, 1); dup2(devnull, 2); close(devnull); }

  // THE ORDERING UNDER TEST. Registered before CLIO_INIT, therefore run after
  // clio's own handler. Moving this line below CLIO_INIT would make the test
  // pass for the wrong reason.
  if (std::atexit(LateTeardown) != 0) _exit(kAtexitRegFailed);

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    _exit(kInitFailed);
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) _exit(kCteInitFailed);
  g_cte = CLIO_CTE_CLIENT;

  clio::cte::core::Tag tag{std::string(kTagName)};
  g_tag_id = tag.GetTagId();

  // Put the blob the late handler will try to delete, and prove in passing that
  // this child really did reach a healthy runtime.
  char payload[8] = {'s', 't', 'a', 'm', 'p', '!', '\0', '\0'};
  auto put = g_cte->AsyncPutBlob(g_tag_id, std::string(kBlobName), 0,
                                 sizeof(payload), payload);
  put.Wait();
  if (put->GetReturnCode() != 0) _exit(kPutFailed);

  // exit(), NOT _exit(): the whole point is to run the atexit handlers.
  std::exit(kOk);
}

}  // namespace

TEST_CASE("ClientExit - task submitted after ClientFinalize does not hang exit",
          "[cr][cli][issue970]") {
  clio::run::test::SetEnvVar("CLIO_PORT", std::to_string(kPort));

  const fs::path work = fs::temp_directory_path() / "clio_client_exit_test";
  fs::remove_all(work);
  fs::create_directories(work);

  // Self-contained CTE core pool (512.0) on a ram device; CTE creates its own
  // bdev target locally. Mirrors test_client_crash_putblob.cc.
  const fs::path compose_yaml = work / "compose.yaml";
  {
    std::ofstream f(compose_yaml);
    f << "compose:\n"
         "  - mod_name: clio_cte_core\n"
         "    pool_name: cte_client_exit\n"
         "    pool_query: local\n"
         "    pool_id: \"512.0\"\n"
         "    storage:\n"
         "      - path: " << (work / "ram_dev").string() << "\n"
         "        bdev_type: ram\n"
         "        capacity_limit: 256mb\n"
         "    dpe:\n"
         "      dpe_type: random\n";
  }

  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(kPort, "127.0.0.1", /*ephemeral=*/true));
  REQUIRE(server.WaitForReady());
  REQUIRE(RunCliTimed({"compose", "start", compose_yaml.string()}, 60) == 0);

  // Client mode does not dlopen ChiMods, so fork without exec is safe here
  // (same rationale as test_client_crash_putblob.cc).
  pid_t child = fork();
  REQUIRE(child >= 0);
  if (child == 0) ExitClientMain();  // never returns

  // Poll rather than block: a blocking waitpid on the defect would hang the
  // TEST too, turning a clear failure into a ctest timeout with no message.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(kExitBudgetSec);
  int status = 0;
  bool exited = false;
  while (std::chrono::steady_clock::now() < deadline) {
    if (waitpid(child, &status, WNOHANG) == child) { exited = true; break; }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (!exited) {
    kill(child, SIGKILL);
    waitpid(child, &status, 0);
    FAIL("child did not exit within " << kExitBudgetSec << "s: a task "
         "submitted after ClientFinalize is parked in an unbounded wait "
         "(issue #970)");
  }

  REQUIRE(WIFEXITED(status));
  const int code = WEXITSTATUS(status);
  INFO("child exit code: " << code);
  REQUIRE(code != kInitFailed);      // never reached a runtime
  REQUIRE(code != kCteInitFailed);   // no CTE pool
  REQUIRE(code != kPutFailed);       // runtime unhealthy
  REQUIRE(code != kAtexitRegFailed);
  REQUIRE(code == kOk);

  server.Stop();
}

int main(int argc, char **argv) {
  // Allows driving the child role by hand for debugging:
  //   ./clio_run_client_exit_wait_test exit-client
  if (argc > 1 && std::string(argv[1]) == "exit-client") {
    ExitClientMain();  // never returns
  }
  int result = SimpleTest::run_all_tests(argc > 1 ? argv[1] : "");
  SIMPLE_TEST_PROCESS_EXIT(result);
  if (SimpleTest::g_test_finalize) SimpleTest::g_test_finalize();
  return result;
}
