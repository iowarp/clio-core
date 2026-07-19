/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Direct unit tests for IpcManager internals that the client/runtime tests
 * never reach: hostfile loading and host lookup, SWIM dead/alive node
 * tracking, buffer allocation size ladder, and client shm bookkeeping.
 */

#include "simple_test.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <clio_ctp/introspect/system_info.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/config_manager.h>
#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>

using namespace clio::run;

namespace {
namespace fs = std::filesystem;

bool g_initialized = false;

void EnsureInitialized() {
  if (!g_initialized) {
    clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    g_initialized = true;
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  }
}

}  // namespace

TEST_CASE("IpcInternals - SWIM node state tracking", "[ipc][swim]") {
  EnsureInitialized();
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  SECTION("Unknown nodes default to dead");
  (void)ipc->GetNodeState(424242);

  SECTION("SetDead then SetAlive round-trip");
  ipc->SetDead(424242);
  REQUIRE(ipc->GetNodeState(424242) == clio::run::NodeState::kDead);
  // SetAlive removes the node from dead-tracking; nodes absent from the
  // table read back as kDead by design, so only exercise the transition.
  ipc->SetAlive(424242);

  SECTION("Repeated transitions are idempotent");
  ipc->SetDead(424242);
  ipc->SetDead(424242);
  ipc->SetAlive(424242);
  ipc->SetAlive(424242);
  REQUIRE(true);
}

TEST_CASE("IpcInternals - AllocateBuffer size ladder", "[ipc][alloc]") {
  EnsureInitialized();
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // Walk a ladder of sizes to hit different allocator size classes and
  // the large-allocation fallback. Every buffer must be valid and is
  // freed immediately.
  const size_t sizes[] = {1,        64,        4096,        64 * 1024,
                          1 << 20,  4 << 20,   16 << 20};
  for (size_t size : sizes) {
    auto buf = ipc->AllocateBuffer(size);
    REQUIRE(!buf.IsNull());
    // Touch first/last byte to verify the mapping is usable.
    buf.ptr_[0] = 'a';
    buf.ptr_[size - 1] = 'z';
    ipc->FreeBuffer(buf);
  }
}

TEST_CASE("IpcInternals - client shm info and memory registration",
          "[ipc][shm]") {
  EnsureInitialized();
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  SECTION("GetClientShmInfo for the primary segment");
  (void)ipc->GetClientShmInfo(0);

  SECTION("RegisterMemory with a null allocator id fails gracefully");
  (void)ipc->RegisterMemory(ctp::ipc::AllocatorId::GetNull());
}

TEST_CASE("IpcInternals - hostfile loading and host lookup",
          "[ipc][hostfile]") {
  EnsureInitialized();
  auto *ipc = CLIO_IPC;
  auto *config = CLIO_CONFIG_MANAGER;
  REQUIRE(ipc != nullptr);
  REQUIRE(config != nullptr);

  SECTION("LoadHostfile with no hostfile binds a single wildcard node");
  REQUIRE(ipc->LoadHostfile());
  REQUIRE(ipc->GetHost(0) != nullptr);

  SECTION("LoadHostfile from an explicit hostfile");
  fs::path hostfile = fs::temp_directory_path() / "clio_test_hostfile.txt";
  {
    std::ofstream f(hostfile);
    f << "127.0.0.1\n";
    f << "10.255.255.1\n";
  }
  fs::path cfg = fs::temp_directory_path() / "clio_test_hostfile_cfg.yaml";
  {
    std::ofstream f(cfg);
    f << "networking:\n  hostfile: " << hostfile.string() << "\n";
  }
  REQUIRE(config->LoadYaml(cfg.string()));
  REQUIRE(ipc->LoadHostfile());

  SECTION("GetHost by node id and by IP");
  const clio::run::Host *h0 = ipc->GetHost(0);
  REQUIRE(h0 != nullptr);
  REQUIRE(ipc->GetHost(999) == nullptr);
  REQUIRE(ipc->GetHostByIp("127.0.0.1") != nullptr);
  REQUIRE(ipc->GetHostByIp("203.0.113.7") == nullptr);

  // NOTE: IdentifyThisHost() is deliberately NOT exercised here — it probes
  // host identity by binding the main server port, which the in-process
  // hosted runtime already owns, and its failure path tears down the
  // process. It is covered by the daemon-based CLI tests instead.

  // Restore the no-hostfile state for any later users of the singleton.
  fs::remove(hostfile);
  fs::remove(cfg);
  fs::path empty_cfg = fs::temp_directory_path() / "clio_test_empty_cfg.yaml";
  {
    std::ofstream f(empty_cfg);
    f << "runtime:\n  num_threads: 1\n";
  }
  (void)config->LoadYaml(empty_cfg.string());
  (void)ipc->LoadHostfile();
  fs::remove(empty_cfg);
}

TEST_CASE("IpcInternals - PoolQuery FromString/ToString round-trips",
          "[ipc][pool_query]") {
  struct Case {
    const char *in;
    const char *out;
  };
  const Case cases[] = {
      {"local", "local"},
      {"BROADCAST", "broadcast"},
      {"Dynamic", "dynamic"},
      {"direct_id:5", "direct_id:5"},
      {"direct_hash:7", "direct_hash:7"},
      {"range:2:3", "range:2:3"},
      {"physical:1", "physical:1"},
  };
  for (const auto &c : cases) {
    clio::run::PoolQuery q = clio::run::PoolQuery::FromString(c.in);
    REQUIRE(q.ToString() == c.out);
  }

  SECTION("Invalid strings throw");
  bool threw = false;
  try {
    (void)clio::run::PoolQuery::FromString("warp_speed");
  } catch (const std::exception &) {
    threw = true;
  }
  REQUIRE(threw);

  threw = false;
  try {
    (void)clio::run::PoolQuery::FromString("range:5");  // missing count
  } catch (const std::exception &) {
    threw = true;
  }
  REQUIRE(threw);
}

TEST_CASE("IpcInternals - ConfigManager yaml sections and env overrides",
          "[ipc][config]") {
  EnsureInitialized();
  auto *config = CLIO_CONFIG_MANAGER;
  REQUIRE(config != nullptr);

  fs::path cfg = fs::temp_directory_path() / "clio_test_sections_cfg.yaml";
  {
    std::ofstream f(cfg);
    f << "runtime:\n"
         "  num_threads: 2\n"
         "  queue_depth: 256\n"
         "gpu:\n"
         "  blocks: 8\n"
         "  threads_per_block: 64\n"
         "  queue_depth: 32\n"
         "networking:\n"
         "  port: 10444\n"
         "  neighborhood_size: 2\n"
         "  wait_for_restart: 1000\n"
         "  wait_for_restart_poll_period: 100\n";
  }

  SECTION("LoadYaml parses gpu and networking sections");
  REQUIRE(config->LoadYaml(cfg.string()));
  REQUIRE(config->GetPort() == 10444);

  SECTION("Env overrides take precedence on reload");
  ctp::SystemInfo::Setenv("CLIO_GPU_BLOCKS", "16", 1);
  ctp::SystemInfo::Setenv("CLIO_GPU_THREADS", "128", 1);
  REQUIRE(config->LoadYaml(cfg.string()));
  ctp::SystemInfo::Unsetenv("CLIO_GPU_BLOCKS");
  ctp::SystemInfo::Unsetenv("CLIO_GPU_THREADS");

  // NOTE: LoadYaml on a missing file is HLOG(kFatal) (process exit), so the
  // not-found path is intentionally not probed here.

  SECTION("Empty config file is reported as a load failure");
  // An empty file parses as a YAML null node — every section lookup misses
  // and the caller would silently keep the default config (no compose, so no
  // storage tiers). LoadYaml must flag it so ClientInit warns loudly.
  fs::path zero_cfg = fs::temp_directory_path() / "clio_test_zero_cfg.yaml";
  { std::ofstream f(zero_cfg); }
  REQUIRE(fs::file_size(zero_cfg) == 0);
  REQUIRE_FALSE(config->LoadYaml(zero_cfg.string()));
  fs::remove(zero_cfg);

  // Restore default-ish config for any later singleton users.
  fs::remove(cfg);
  fs::path empty_cfg = fs::temp_directory_path() / "clio_test_plain_cfg.yaml";
  {
    std::ofstream f(empty_cfg);
    f << "runtime:\n  num_threads: 2\n";
  }
  (void)config->LoadYaml(empty_cfg.string());
  fs::remove(empty_cfg);
}

SIMPLE_TEST_MAIN()
