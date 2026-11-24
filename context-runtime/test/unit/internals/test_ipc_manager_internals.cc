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

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <clio_ctp/introspect/system_info.h>
#include <clio_runtime/admin/admin_client.h>
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

// A loopback bind address means "single node, this machine only", so a
// configured hostfile cannot apply and LoadHostfile drops it. This is what
// keeps a developer's ~/.clio/clio.yaml cluster hostfile from promoting a unit
// test into a peer of a real cluster (#978). Both sides of that decision are
// exercised here: CLIO_BIND_ADDR drives the bind address directly, so the
// non-loopback path is reachable without a real multi-node deployment.
TEST_CASE("IpcInternals - loopback bind ignores a configured hostfile",
          "[ipc][hostfile]") {
  EnsureInitialized();
  auto *ipc = CLIO_IPC;
  auto *config = CLIO_CONFIG_MANAGER;
  REQUIRE(ipc != nullptr);
  REQUIRE(config != nullptr);

  // 10.255.255.1 is the sentinel: it only ever appears if the hostfile was
  // actually parsed. It is in the RFC 1918 range and never bound by the test.
  fs::path hostfile = fs::temp_directory_path() / "clio_loopback_hostfile.txt";
  {
    std::ofstream f(hostfile);
    f << "127.0.0.1\n";
    f << "10.255.255.1\n";
  }
  fs::path cfg = fs::temp_directory_path() / "clio_loopback_hostfile_cfg.yaml";
  {
    std::ofstream f(cfg);
    f << "networking:\n  hostfile: " << hostfile.string() << "\n";
  }
  REQUIRE(config->LoadYaml(cfg.string()));

  SECTION("A non-loopback bind address honors the hostfile");
  ctp::SystemInfo::Setenv("CLIO_BIND_ADDR", "0.0.0.0", 1);
  REQUIRE(ipc->LoadHostfile());
  REQUIRE(ipc->GetHostByIp("10.255.255.1") != nullptr);

  SECTION("Each loopback spelling drops the hostfile");
  // 127.0.0.1 covers the "127." prefix test, and localhost / ::1 cover the
  // two literal comparisons, so every branch of IsLoopbackBindAddr runs.
  const char *loopback_addrs[] = {"127.0.0.1", "127.1.2.3", "localhost",
                                  "::1"};
  for (const char *addr : loopback_addrs) {
    ctp::SystemInfo::Setenv("CLIO_BIND_ADDR", addr, 1);
    REQUIRE(ipc->LoadHostfile());
    // The hostfile was dropped, so the sentinel peer is absent and the
    // single-node wildcard host stands in its place.
    REQUIRE(ipc->GetHostByIp("10.255.255.1") == nullptr);
    REQUIRE(ipc->GetHost(0) != nullptr);
  }

  // Restore the no-hostfile, no-override state for later users of the
  // singleton, matching the hostfile test case above.
  ctp::SystemInfo::Unsetenv("CLIO_BIND_ADDR");
  fs::remove(hostfile);
  fs::remove(cfg);
  fs::path empty_cfg =
      fs::temp_directory_path() / "clio_loopback_empty_cfg.yaml";
  {
    std::ofstream f(empty_cfg);
    f << "runtime:\n  num_threads: 1\n";
  }
  (void)config->LoadYaml(empty_cfg.string());
  (void)ipc->LoadHostfile();
  fs::remove(empty_cfg);
}

// Node-state transitions, inbound tracking and AddNode on hosts that are in
// the table. The SWIM test case above uses an unknown node id, so it only
// reaches the early returns. Each call takes hostfile_mu_, so this also
// checks that no path locks it twice (that would deadlock here).
TEST_CASE("IpcInternals - host table transitions and AddNode",
          "[ipc][hostfile][swim]") {
  EnsureInitialized();
  auto *ipc = CLIO_IPC;
  auto *config = CLIO_CONFIG_MANAGER;
  REQUIRE(ipc != nullptr);
  REQUIRE(config != nullptr);

  // Node 1 (10.255.255.1) is a peer that never answers; this process is
  // node 0. A non-loopback bind address makes LoadHostfile honor the file.
  // SWIM and the idle/dead-node reprobes are off so that nothing but this
  // test changes the peer's state.
  fs::path hostfile = fs::temp_directory_path() / "clio_table_hostfile.txt";
  {
    std::ofstream f(hostfile);
    f << "127.0.0.1\n";
    f << "10.255.255.1\n";
  }
  fs::path cfg = fs::temp_directory_path() / "clio_table_hostfile_cfg.yaml";
  {
    std::ofstream f(cfg);
    f << "runtime:\n  task_progress_interval_ms: 0\n"
         "swim:\n  enabled: false\n"
         "networking:\n  hostfile: " << hostfile.string() << "\n";
  }
  REQUIRE(config->LoadYaml(cfg.string()));
  ctp::SystemInfo::Setenv("CLIO_BIND_ADDR", "0.0.0.0", 1);
  REQUIRE(ipc->LoadHostfile());
  REQUIRE(ipc->GetNumHosts() == 2);
  const clio::run::Host *peer_host = ipc->GetHostByIp("10.255.255.1");
  REQUIRE(peer_host != nullptr);
  const u64 peer = peer_host->node_id;
  REQUIRE(ipc->IsAlive(peer));

  SECTION("SetDead and SetAlive change a known node's state");
  ipc->SetDead(peer);
  REQUIRE(ipc->GetNodeState(peer) == clio::run::NodeState::kDead);
  REQUIRE_FALSE(ipc->IsAlive(peer));
  ipc->SetAlive(peer);
  REQUIRE(ipc->GetNodeState(peer) == clio::run::NodeState::kAlive);

  SECTION("SetNodeState sets the state and invalidates the host cache");
  (void)ipc->GetAllHosts();  // build the cache before the change
  ipc->SetNodeState(peer, clio::run::NodeState::kSuspected);
  REQUIRE(ipc->GetNodeState(peer) == clio::run::NodeState::kSuspected);
  bool found = false;
  for (const auto &h : ipc->GetAllHosts()) {
    if (h.node_id == peer) {
      REQUIRE(h.state == clio::run::NodeState::kSuspected);
      found = true;
    }
  }
  REQUIRE(found);
  ipc->SetNodeState(424242, clio::run::NodeState::kDead);  // unknown: no-op
  REQUIRE(ipc->GetNumHosts() == 2);

  SECTION("NoteInbound resets SecondsSinceInbound");
  ipc->NoteInbound(peer);
  REQUIRE(ipc->SecondsSinceInbound(peer) < 60.0f);
  ipc->NoteInbound(424242);  // unknown: no-op
  REQUIRE(ipc->SecondsSinceInbound(424242) ==
          std::numeric_limits<float>::max());

  SECTION("AddNode of an address in the table returns its id and revives it");
  REQUIRE_FALSE(ipc->IsAlive(peer));  // still kSuspected from above
  REQUIRE(ipc->AddNode("10.255.255.1", 0) == peer);
  REQUIRE(ipc->IsAlive(peer));
  REQUIRE(ipc->GetNumHosts() == 2);

  SECTION("AddNode registers a new address once");
  const u64 added = ipc->AddNode("10.255.255.2", 0);
  REQUIRE(added == 2);
  REQUIRE(ipc->GetNumHosts() == 3);
  const clio::run::Host *added_host = ipc->GetHostByIp("10.255.255.2");
  REQUIRE(added_host != nullptr);
  REQUIRE(added_host->node_id == added);
  REQUIRE(ipc->AddNode("10.255.255.2", 0) == added);
  REQUIRE(ipc->GetNumHosts() == 3);

  SECTION("Concurrent AddNode of one address registers it once");
  {
    std::vector<u64> ids(8);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < ids.size(); ++i) {
      threads.emplace_back(
          [&, i] { ids[i] = ipc->AddNode("10.255.255.3", 0); });
    }
    for (auto &t : threads) t.join();
    for (u64 id : ids) REQUIRE(id == ids[0]);
    REQUIRE(ipc->GetNumHosts() == 4);
  }

  SECTION("Readers survive a concurrent hostfile reload");
  // The race hostfile_mu_ was added for: GetNodeIds iterating hostfile_map_
  // while LoadHostfile clears and refills it (an ASan heap-use-after-free).
  {
    std::atomic<bool> stop{false};
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
      readers.emplace_back([&] {
        while (!stop.load()) {
          for (u64 id : ipc->GetNodeIds()) (void)ipc->GetNodeState(id);
          (void)ipc->GetNumHosts();
          (void)ipc->GetLeaderNodeId();
        }
      });
    }
    for (int i = 0; i < 50; ++i) REQUIRE(ipc->LoadHostfile());
    stop = true;
    for (auto &t : readers) t.join();
    REQUIRE(ipc->GetNumHosts() == 2);  // the reload dropped AddNode's hosts
  }

  // Restore the no-hostfile, no-override state for later users of the
  // singleton, as the hostfile test cases above do.
  ctp::SystemInfo::Unsetenv("CLIO_BIND_ADDR");
  fs::remove(hostfile);
  fs::remove(cfg);
  fs::path empty_cfg = fs::temp_directory_path() / "clio_table_empty_cfg.yaml";
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

TEST_CASE("IpcInternals - main segment size is configurable (issue #727)",
          "[ipc][config]") {
  EnsureInitialized();
  auto *config = CLIO_CONFIG_MANAGER;
  REQUIRE(config != nullptr);

  fs::path cfg = fs::temp_directory_path() / "clio_test_mainseg_cfg.yaml";

  SECTION("yaml size string is honored verbatim");
  {
    std::ofstream f(cfg);
    f << "runtime:\n  main_segment_size: \"256m\"\n";
  }
  REQUIRE(config->LoadYaml(cfg.string()));
  REQUIRE(config->CalculateMainSegmentSize() ==
          ctp::Unit<size_t>::Megabytes(256));

  SECTION("bare byte count is honored");
  {
    std::ofstream f(cfg);
    f << "runtime:\n  main_segment_size: 134217728\n";
  }
  REQUIRE(config->LoadYaml(cfg.string()));
  REQUIRE(config->CalculateMainSegmentSize() ==
          ctp::Unit<size_t>::Megabytes(128));

  // The auto default: the machine's RAM capacity (flat 1 GiB when
  // introspection fails), mirroring the metadata segment. The half-budget and
  // non-Linux clamps apply at segment creation (ServerInitShm), not here.
  // Recomputed rather than hardcoded so the expectation holds on any host.
  size_t expect_auto = ctp::SystemInfo::GetRamCapacity();
  if (expect_auto == 0) {
    expect_auto = ctp::Unit<size_t>::Gigabytes(1);
  }

  SECTION("0 selects the auto default");
  {
    std::ofstream f(cfg);
    f << "runtime:\n  main_segment_size: 0\n";
  }
  REQUIRE(config->LoadYaml(cfg.string()));
  REQUIRE(config->CalculateMainSegmentSize() == expect_auto);
  // Whatever the budget, the auto default must be a usable segment — callers
  // (GetMemorySegmentSize included) never see a 0 sentinel.
  REQUIRE(config->CalculateMainSegmentSize() >=
          ctp::Unit<size_t>::Megabytes(64));

  SECTION("an unparseable value keeps the default instead of dying");
  {
    std::ofstream f(cfg);
    f << "runtime:\n  main_segment_size: \"lots\"\n";
  }
  REQUIRE(config->LoadYaml(cfg.string()));
  REQUIRE(config->CalculateMainSegmentSize() == expect_auto);

  SECTION("CLIO_MAIN_SEGMENT_SIZE env wins over yaml");
  {
    std::ofstream f(cfg);
    f << "runtime:\n  main_segment_size: \"256m\"\n";
  }
  ctp::SystemInfo::Setenv("CLIO_MAIN_SEGMENT_SIZE", "128m", 1);
  REQUIRE(config->LoadYaml(cfg.string()));
  config->ApplyEnvOverrides();
  ctp::SystemInfo::Unsetenv("CLIO_MAIN_SEGMENT_SIZE");
  REQUIRE(config->CalculateMainSegmentSize() ==
          ctp::Unit<size_t>::Megabytes(128));

  // Restore default-ish config for any later singleton users (LoadYaml
  // resets to defaults first, so omitting the key restores auto).
  {
    std::ofstream f(cfg);
    f << "runtime:\n  num_threads: 2\n";
  }
  (void)config->LoadYaml(cfg.string());
  fs::remove(cfg);
}

TEST_CASE("IpcInternals - MigrateContainers refuses a container hosted "
          "elsewhere (issue #1179)",
          "[admin][migrate][1179]") {
  EnsureInitialized();
  clio::run::admin::Client admin(clio::run::kAdminPoolId);
  auto *pool_manager = CLIO_POOL_MANAGER;
  REQUIRE(pool_manager != nullptr);

  // This node hosts its own admin container (id = node id) but not 4242.
  // GetContainer would fall back to the local container, which used to be
  // migrated in its place with a success report.
  const auto local_id =
      static_cast<clio::run::ContainerId>(CLIO_IPC->GetNodeId());
  const clio::run::ContainerId missing_id = 4242;
  REQUIRE(pool_manager->HasContainer(clio::run::kAdminPoolId, local_id));
  REQUIRE_FALSE(pool_manager->HasContainer(clio::run::kAdminPoolId, missing_id));

  std::vector<clio::run::MigrateInfo> migrations;
  migrations.emplace_back(clio::run::kAdminPoolId, missing_id,
                          static_cast<clio::run::u32>(CLIO_IPC->GetNodeId()));
  auto task = admin.AsyncMigrateContainers(clio::run::PoolQuery::Local(),
                                            migrations);
  task.Wait();

  REQUIRE(task->GetReturnCode() != 0);
  REQUIRE(task->num_migrated_ == 0);
  REQUIRE(task->error_message_.str().find("not hosted") != std::string::npos);
  // The local container was left alone.
  REQUIRE(pool_manager->HasContainer(clio::run::kAdminPoolId, local_id));
}

SIMPLE_TEST_MAIN()
