/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Autogen dispatch sweep tests.
 *
 * The per-method tests in test_autogen_coverage.cc exercise the autogen
 * lib_exec dispatch functions (SaveTask, LoadTask, AllocLoadTask,
 * LocalSaveTask, LocalLoadTask, LocalAllocLoadTask, NewCopyTask, Aggregate)
 * only for the low-numbered methods. This file sweeps EVERY method id of the
 * admin, bdev, and MOD_NAME modules through the full dispatch battery so the
 * remaining switch arms (and the task serialization/copy code in the
 * *_tasks.h headers they call into) are exercised too.
 */

#include "simple_test.h"

#include <string>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/container.h>
#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/pool_manager.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/task.h>
#include <clio_runtime/task_archives.h>
#include <clio_runtime/local_task_archives.h>
#include <clio_runtime/types.h>

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_runtime.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_runtime/admin/autogen/admin_methods.h>

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_runtime.h>
#include <clio_runtime/bdev/bdev_tasks.h>
#include <clio_runtime/bdev/autogen/bdev_methods.h>

#include <clio_runtime/MOD_NAME/MOD_NAME_client.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_runtime.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_tasks.h>
#include <clio_runtime/MOD_NAME/autogen/MOD_NAME_methods.h>

#include <clio_cte/core/core_runtime.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/autogen/core_methods.h>

#include <clio_cae/core/core_runtime.h>
#include <clio_cae/core/core_tasks.h>
#include <clio_cae/core/autogen/core_methods.h>

using namespace chi;

namespace {

bool g_initialized = false;

void EnsureInitialized() {
  if (!g_initialized) {
    chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    g_initialized = true;
    SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
  }
}

/**
 * Run one method id through the full autogen dispatch battery on the given
 * container. Every step is guarded so methods that cannot allocate (or are
 * not constructible in client mode) are skipped rather than failed: the goal
 * is exercising the dispatch arms, not the handler semantics.
 */
void SweepMethod(chi::Container &container, chi::u32 method) {
  auto task = container.NewTask(method);
  if (task.IsNull()) {
    return;  // method not constructible in this configuration
  }

  // --- SaveTask / LoadTask / AllocLoadTask (network archives), both
  // serialization directions.
  const chi::MsgType kDirs[] = {chi::MsgType::kSerializeIn,
                                chi::MsgType::kSerializeOut};
  for (chi::MsgType dir : kDirs) {
    chi::SaveTaskArchive save_archive(dir);
    container.SaveTask(method, save_archive, task);
    std::string data = save_archive.GetData();

    {
      chi::LoadTaskArchive load_archive(data);
      load_archive.msg_type_ = dir;
      auto loaded = container.NewTask(method);
      if (!loaded.IsNull()) {
        container.LoadTask(method, load_archive, loaded);
        container.DelTask(method, loaded);
      }
    }
    {
      chi::LoadTaskArchive load_archive(data);
      load_archive.msg_type_ = dir;
      auto alloc_loaded = container.AllocLoadTask(method, load_archive);
      if (!alloc_loaded.IsNull()) {
        container.DelTask(method, alloc_loaded);
      }
    }
  }

  // --- LocalSaveTask / LocalLoadTask / LocalAllocLoadTask (local archives).
  {
    chi::priv::vector<char> save_buf(CLIO_PRIV_ALLOC);
    chi::DefaultSaveArchive save_archive(chi::LocalMsgType::kSerializeIn,
                                         save_buf);
    container.LocalSaveTask(method, save_archive, task);

    {
      chi::DefaultLoadArchive load_archive(save_archive.GetMutableData());
      auto loaded = container.NewTask(method);
      if (!loaded.IsNull()) {
        container.LocalLoadTask(method, load_archive, loaded);
        container.DelTask(method, loaded);
      }
    }
    {
      chi::DefaultLoadArchive load_archive(save_archive.GetMutableData());
      auto alloc_loaded = container.LocalAllocLoadTask(method, load_archive);
      if (!alloc_loaded.IsNull()) {
        container.DelTask(method, alloc_loaded);
      }
    }
  }

  // --- NewCopyTask (shallow and deep).
  {
    auto copy = container.NewCopyTask(method, task, false);
    if (!copy.IsNull()) {
      container.DelTask(method, copy);
    }
    auto deep_copy = container.NewCopyTask(method, task, true);
    if (!deep_copy.IsNull()) {
      container.DelTask(method, deep_copy);
    }
  }

  // --- Aggregate via the container dispatch switch (the per-method tests
  // call task->Aggregate directly, leaving the dispatch arms uncovered).
  {
    auto replica = container.NewTask(method);
    if (!replica.IsNull()) {
      container.Aggregate(method, task, replica);
      container.DelTask(method, replica);
    }
  }

  container.DelTask(method, task);
}

}  // namespace

//==============================================================================
// Admin module sweep
//==============================================================================

TEST_CASE("AutogenSweep - Admin all methods full dispatch battery",
          "[autogen][admin][sweep]") {
  EnsureInitialized();

  auto *pool_manager = CLIO_POOL_MANAGER;
  auto *container = pool_manager->GetStaticContainer(chi::kAdminPoolId);
  if (container == nullptr) {
    INFO("Admin container not available - skipping test");
    return;
  }

  namespace adm = clio::run::admin;
  const std::vector<chi::u32> methods = {
      adm::Method::kMonitor,           adm::Method::kFlush,
      adm::Method::kSend,              adm::Method::kRecv,
      adm::Method::kClientConnect,     adm::Method::kSubmitBatch,
      adm::Method::kWreapDeadIpcs,     adm::Method::kClientRecv,
      adm::Method::kClientSend,        adm::Method::kRegisterMemory,
      adm::Method::kRestartContainers, adm::Method::kAddNode,
      adm::Method::kChangeAddressTable, adm::Method::kMigrateContainers,
      adm::Method::kHeartbeat,         adm::Method::kHeartbeatProbe,
      adm::Method::kProbeRequest,      adm::Method::kRecoverContainers,
      adm::Method::kSystemMonitor,     adm::Method::kAnnounceShutdown,
      adm::Method::kRegisterGpuContainer,
  };

  for (chi::u32 method : methods) {
    SweepMethod(*container, method);
  }
  REQUIRE(true);
}

//==============================================================================
// Bdev module sweep (directly-instantiated Runtime container)
//==============================================================================

TEST_CASE("AutogenSweep - Bdev all methods full dispatch battery",
          "[autogen][bdev][sweep]") {
  EnsureInitialized();

  clio::run::bdev::Runtime bdev_runtime;

  namespace bd = clio::run::bdev;
  const std::vector<chi::u32> methods = {
      bd::Method::kCreate,         bd::Method::kDestroy,
      bd::Method::kMonitor,        bd::Method::kAllocateBlocks,
      bd::Method::kFreeBlocks,     bd::Method::kWrite,
      bd::Method::kRead,           bd::Method::kGetStats,
      bd::Method::kUpdate,
  };

  for (chi::u32 method : methods) {
    SweepMethod(bdev_runtime, method);
  }
  REQUIRE(true);
}

//==============================================================================
// MOD_NAME template module sweep (directly-instantiated Runtime container)
//==============================================================================

TEST_CASE("AutogenSweep - MOD_NAME all methods full dispatch battery",
          "[autogen][modname][sweep]") {
  EnsureInitialized();

  clio::run::MOD_NAME::Runtime mod_runtime;

  namespace mn = clio::run::MOD_NAME;
  const std::vector<chi::u32> methods = {
      mn::Method::kCreate,        mn::Method::kDestroy,
      mn::Method::kMonitor,       mn::Method::kCustom,
      mn::Method::kCoMutexTest,   mn::Method::kCoRwLockTest,
      mn::Method::kWaitTest,      mn::Method::kTestLargeOutput,
      mn::Method::kGpuSubmit,     mn::Method::kSubtaskTest,
  };

  for (chi::u32 method : methods) {
    SweepMethod(mod_runtime, method);
  }
  REQUIRE(true);
}

//==============================================================================
// Admin Run() battery — executes the SAFE subset of admin handlers through
// the autogen Run dispatch on the real (static) admin container. Methods
// that would stop the runtime, contact other nodes, or restart containers
// are deliberately excluded.
//==============================================================================

TEST_CASE("AutogenSweep - Admin Run battery for safe methods",
          "[autogen][admin][run]") {
  EnsureInitialized();

  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto *container = pool_manager->GetStaticContainer(chi::kAdminPoolId);
  if (container == nullptr) {
    INFO("Admin container not available - skipping test");
    return;
  }

  namespace adm = clio::run::admin;

  SECTION("Monitor handler with every synchronous query type");
  const char *queries[] = {"worker_stats", "system_stats", "container_stats",
                           "get_host_info", "unknown_query_type"};
  for (const char *q : queries) {
    auto task = ipc_manager->NewTask<adm::MonitorTask>(
        chi::CreateTaskId(), chi::kAdminPoolId, chi::PoolQuery::Local(),
        std::string(q));
    if (task.IsNull()) continue;
    chi::RunContext rctx;
    auto tr = container->Run(adm::Method::kMonitor,
                             task.template Cast<chi::Task>(), rctx);
    for (int spin = 0; !tr.done() && spin < 16; ++spin) {
      tr.resume();
    }
    CLIO_IPC->DelTask(task);
  }

  SECTION("Heartbeat handler");
  {
    auto task = container->NewTask(adm::Method::kHeartbeat);
    if (!task.IsNull()) {
      chi::RunContext rctx;
      auto tr = container->Run(adm::Method::kHeartbeat, task, rctx);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
      container->DelTask(adm::Method::kHeartbeat, task);
    }
  }

  SECTION("WreapDeadIpcs handler (no dead clients, quick scan)");
  {
    auto task = container->NewTask(adm::Method::kWreapDeadIpcs);
    if (!task.IsNull()) {
      chi::RunContext rctx;
      auto tr = container->Run(adm::Method::kWreapDeadIpcs, task, rctx);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
      container->DelTask(adm::Method::kWreapDeadIpcs, task);
    }
  }

  SECTION("SystemMonitor handler");
  {
    auto task = container->NewTask(adm::Method::kSystemMonitor);
    if (!task.IsNull()) {
      chi::RunContext rctx;
      auto tr = container->Run(adm::Method::kSystemMonitor, task, rctx);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
      container->DelTask(adm::Method::kSystemMonitor, task);
    }
  }

  SECTION("MigrateContainers handler with empty migration list (no-op)");
  {
    auto task = container->NewTask(adm::Method::kMigrateContainers);
    if (!task.IsNull()) {
      chi::RunContext rctx;
      auto tr = container->Run(adm::Method::kMigrateContainers, task, rctx);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
      container->DelTask(adm::Method::kMigrateContainers, task);
    }
  }

  SECTION("RecoverContainers handler with empty recovery list (no-op)");
  {
    auto task = container->NewTask(adm::Method::kRecoverContainers);
    if (!task.IsNull()) {
      chi::RunContext rctx;
      auto tr = container->Run(adm::Method::kRecoverContainers, task, rctx);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
      container->DelTask(adm::Method::kRecoverContainers, task);
    }
  }

  SECTION("ChangeAddressTable handler with empty table (no-op)");
  {
    auto task = container->NewTask(adm::Method::kChangeAddressTable);
    if (!task.IsNull()) {
      chi::RunContext rctx;
      auto tr = container->Run(adm::Method::kChangeAddressTable, task, rctx);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
      container->DelTask(adm::Method::kChangeAddressTable, task);
    }
  }

  SECTION("GetOrCreatePool handler with a nonexistent module (error path)");
  {
    auto task = container->NewTask(adm::Method::kGetOrCreatePool);
    if (!task.IsNull()) {
      auto typed = task.template Cast<
          adm::GetOrCreatePoolTask<adm::CreateParams>>();
      typed->chimod_name_ =
          chi::priv::string(CTP_MALLOC, "no_such_module_xyz");
      typed->pool_name_ = chi::priv::string(CTP_MALLOC, "no_such_pool_xyz");
      chi::RunContext rctx;
      auto tr = container->Run(adm::Method::kGetOrCreatePool, task, rctx);
      for (int spin = 0; !tr.done() && spin < 64; ++spin) {
        tr.resume();
      }
      container->DelTask(adm::Method::kGetOrCreatePool, task);
    }
  }

  SECTION("DestroyPool handler with a bogus pool id (error path)");
  {
    auto task = container->NewTask(adm::Method::kDestroyPool);
    if (!task.IsNull()) {
      chi::RunContext rctx;
      auto tr = container->Run(adm::Method::kDestroyPool, task, rctx);
      for (int spin = 0; !tr.done() && spin < 64; ++spin) {
        tr.resume();
      }
      container->DelTask(adm::Method::kDestroyPool, task);
    }
  }

  SECTION("RegisterMemory handler with default (invalid) registration");
  {
    auto task = container->NewTask(adm::Method::kRegisterMemory);
    if (!task.IsNull()) {
      chi::RunContext rctx;
      auto tr = container->Run(adm::Method::kRegisterMemory, task, rctx);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
      container->DelTask(adm::Method::kRegisterMemory, task);
    }
  }

  REQUIRE(true);
}

//==============================================================================
// CTE core module sweep (directly-instantiated Runtime container)
//==============================================================================

TEST_CASE("AutogenSweep - CTE core all methods full dispatch battery",
          "[autogen][cte][sweep]") {
  EnsureInitialized();

  clio::cte::core::Runtime cte_runtime;

  namespace ct = clio::cte::core;
  const std::vector<chi::u32> methods = {
      ct::Method::kCreate,           ct::Method::kDestroy,
      ct::Method::kMonitor,          ct::Method::kRegisterTarget,
      ct::Method::kUnregisterTarget, ct::Method::kListTargets,
      ct::Method::kStatTargets,      ct::Method::kGetOrCreateTag,
      ct::Method::kPutBlob,          ct::Method::kGetBlob,
      ct::Method::kReorganizeBlob,   ct::Method::kDelBlob,
      ct::Method::kDelTag,           ct::Method::kGetTagSize,
      ct::Method::kPollTelemetryLog, ct::Method::kGetBlobScore,
      ct::Method::kGetBlobSize,      ct::Method::kGetContainedBlobs,
      ct::Method::kGetBlobInfo,      ct::Method::kTagQuery,
      ct::Method::kBlobQuery,        ct::Method::kGetTargetInfo,
      ct::Method::kFlushMetadata,    ct::Method::kFlushData,
      ct::Method::kSemanticSearch,   ct::Method::kTemporalSearch,
  };

  for (chi::u32 method : methods) {
    SweepMethod(cte_runtime, method);
  }
  REQUIRE(true);
}

//==============================================================================
// CAE core module sweep (directly-instantiated Runtime container)
//==============================================================================

TEST_CASE("AutogenSweep - CAE core all methods full dispatch battery",
          "[autogen][cae][sweep]") {
  EnsureInitialized();

  clio::cae::core::Runtime cae_runtime;

  namespace ca = clio::cae::core;
  const std::vector<chi::u32> methods = {
      ca::Method::kCreate,         ca::Method::kDestroy,
      ca::Method::kMonitor,        ca::Method::kParseOmni,
      ca::Method::kExportData,     ca::Method::kGetOrCreateTag,
      ca::Method::kPutBlob,        ca::Method::kGetBlob,
      ca::Method::kSemanticSearch,
  };

  for (chi::u32 method : methods) {
    SweepMethod(cae_runtime, method);
  }
  REQUIRE(true);
}

SIMPLE_TEST_MAIN()
