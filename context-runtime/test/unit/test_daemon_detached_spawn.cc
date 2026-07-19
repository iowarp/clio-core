/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Console-less daemon spawn test (issue #721).
 *
 * On Windows a runtime daemon spawned with DETACHED_PROCESS failed to initialize
 * Winsock ("Successful WSASTARTUP not yet performed [10093]"); its ZeroMQ ROUTER
 * then failed every socket op and the daemon stayed alive but UNREACHABLE — a
 * client just timed out after 30 s. This test spawns the daemon console-less
 * (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP on Windows; POSIX_SPAWN_SETSID on
 * POSIX) and then connects a TCP client, which MUST reach the daemon's ROUTER —
 * exactly the socket that #721 left dead. A successful connect proves the
 * transport initialized regardless of console.
 *
 * Cross-platform on purpose (no fork()): the Windows path is the one that
 * regressed, so this test builds and runs on Windows CI too. It uses only
 * RuntimeServer (whose spawn/wait/kill go through ctp::SystemInfo, so no
 * <windows.h> leaks in to clash with the client's <winsock2.h>) and an
 * in-process client.
 */
#include "../simple_test.h"
#include "../runtime_server.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/ipc_manager.h"

namespace fs = std::filesystem;
using namespace clio::run;

namespace {
std::string ReadWholeFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
}  // namespace

TEST_CASE("DaemonDetachedSpawn - transport initializes without a console (#721)",
          "[runtime][detached][transport]") {
  constexpr unsigned kPort = 10615;
  const std::string log_path =
      (fs::temp_directory_path() / "clio_detached_spawn.log").string();
  ::remove(log_path.c_str());
  test::SetEnvVar("CLIO_TEST_SERVER_LOG", log_path);
  test::SetEnvVar("CTP_LOG_LEVEL", "info");
  test::SetEnvVar("CLIO_PORT", std::to_string(kPort));

  // Spawn the daemon with NO controlling console (the #721 failure mode).
  test::RuntimeServer server;
  REQUIRE(server.Start(kPort, "127.0.0.1", /*ephemeral=*/true, /*detached=*/true));
  REQUIRE(server.WaitForReady());

  // Connect a TCP client: its DEALER must reach the daemon's ROUTER (the socket
  // that WSAStartup failure left dead). Bound the wait so a dead transport fails
  // fast instead of hanging the whole 30 s default.
  test::SetEnvVar("CLIO_WITH_RUNTIME", "0");
  test::SetEnvVar("CLIO_IPC_MODE", "tcp");
  test::SetEnvVar("CLIO_WAIT_SERVER", "15");
  bool connected = CLIO_INIT(RuntimeMode::kClient, false);
  REQUIRE(connected);  // #721: a dead ROUTER would make this time out / fail

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());

  // The daemon must NOT have logged the Winsock-init failure.
  const std::string log = ReadWholeFile(log_path);
  REQUIRE(log.find("WSASTARTUP") == std::string::npos);
  REQUIRE(log.find("Successful WSASTARTUP not yet performed") ==
          std::string::npos);

  test::UnsetEnvVar("CLIO_TEST_SERVER_LOG");
  // server stopped by RuntimeServer destructor (RAII)
}

SIMPLE_TEST_MAIN()
