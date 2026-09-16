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
 * LocalSaveTask, LocalLoadTask, LocalAllocLoadTask, NewCopyTask, AggregateOut)
 * only for the low-numbered methods. This file sweeps EVERY method id of the
 * admin, bdev, and MOD_NAME modules through the full dispatch battery so the
 * remaining switch arms (and the task serialization/copy code in the
 * *_tasks.h headers they call into) are exercised too.
 */

#include "simple_test.h"

#include <cstring>
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

using namespace clio::run;

namespace {

bool g_initialized = false;

void EnsureInitialized() {
  if (!g_initialized) {
    clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    g_initialized = true;
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  }
}

/**
 * Run one method id through the full autogen dispatch battery on the given
 * container. Every step is guarded so methods that cannot allocate (or are
 * not constructible in client mode) are skipped rather than failed: the goal
 * is exercising the dispatch arms, not the handler semantics.
 */
void SweepMethod(clio::run::Container &container, clio::run::u32 method) {
  auto task = container.NewTask(method);
  if (task.IsNull()) {
    return;  // method not constructible in this configuration
  }

  // --- SaveTask / LoadTask / AllocLoadTask (network archives), both
  // serialization directions.
  const clio::run::MsgType kDirs[] = {clio::run::MsgType::kSerializeIn,
                                clio::run::MsgType::kSerializeOut};
  for (clio::run::MsgType dir : kDirs) {
    clio::run::SaveTaskArchive save_archive(dir);
    container.SaveTask(method, save_archive, task);
    std::string data = save_archive.GetData();

    {
      clio::run::LoadTaskArchive load_archive(data);
      load_archive.msg_type_ = dir;
      auto loaded = container.NewTask(method);
      if (!loaded.IsNull()) {
        container.LoadTask(method, load_archive, loaded);
      }
    }
    {
      clio::run::LoadTaskArchive load_archive(data);
      load_archive.msg_type_ = dir;
      auto alloc_loaded = container.AllocLoadTask(method, load_archive);
      if (!alloc_loaded.IsNull()) {
      }
    }
  }

  // --- LocalSaveTask / LocalLoadTask / LocalAllocLoadTask (local archives).
  {
    clio::run::priv::vector<char> save_buf(CLIO_PRIV_ALLOC);
    clio::run::DefaultSaveArchive save_archive(clio::run::LocalMsgType::kSerializeIn,
                                         save_buf);
    container.LocalSaveTask(method, save_archive, task);

    {
      clio::run::DefaultLoadArchive load_archive(save_archive.GetMutableData());
      auto loaded = container.NewTask(method);
      if (!loaded.IsNull()) {
        container.LocalLoadTask(method, load_archive, loaded);
      }
    }
    {
      clio::run::DefaultLoadArchive load_archive(save_archive.GetMutableData());
      auto alloc_loaded = container.LocalAllocLoadTask(method, load_archive);
      if (!alloc_loaded.IsNull()) {
      }
    }
  }

  // --- NewCopyTask (shallow and deep).
  {
    auto copy = container.NewCopyTask(method, task, false);
    if (!copy.IsNull()) {
    }
    auto deep_copy = container.NewCopyTask(method, task, true);
    if (!deep_copy.IsNull()) {
    }
  }

  // --- AggregateOut via the container dispatch switch (the per-method tests
  // call task->AggregateOut directly, leaving the dispatch arms uncovered).
  //
  // This is also the identity regression for issue #915. AggregateOut merges
  // a REPLICA's OUT fields into the ORIGIN; it must NEVER touch the origin's
  // identity. The 77 implementations that delegated to Copy() ran Task::Copy
  // and made the origin adopt the replica's task_id_/pool_query_/method_
  // mid-flight while send_map_, the replica accounting and the completion
  // path still keyed off the origin's — the `free(): invalid pointer` that
  // killed leader-election recovery (#856). Stamping the origin and the
  // replica with DIFFERENT identities makes any whole-task copy fail here
  // instead of corrupting a live runtime.
  {
    auto replica = container.NewTask(method);
    if (!replica.IsNull()) {
      task->task_id_ =
          clio::run::TaskId(1111, 2222, 3333, /*replica_id=*/0, 4444, 55, 6666);
      task->pool_query_ = clio::run::PoolQuery::Local();
      task->method_ = method;

      replica->task_id_ =
          clio::run::TaskId(7777, 8888, 9999, /*replica_id=*/1, 1010, 11, 1212);
      replica->pool_query_ = clio::run::PoolQuery::Broadcast();
      replica->method_ = method;

      const clio::run::TaskId id_before = task->task_id_;
      const clio::run::PoolQuery query_before = task->pool_query_;
      const clio::run::u32 method_before = task->method_;

      container.AggregateOut(method, task, replica);

      // PoolQuery has no operator== and is deliberately trivially copyable
      // (it is raw-byte compared elsewhere in the tree), so memcmp is the
      // right test here.
      if (!(task->task_id_ == id_before) ||
          memcmp(&task->pool_query_, &query_before,
                 sizeof(clio::run::PoolQuery)) != 0 ||
          task->method_ != method_before) {
        FAIL("AggregateOut changed the ORIGIN task's identity for method "
             << method
             << ": its AggregateOut is copying the whole replica (almost "
                "certainly `Copy(other_base...)`) instead of merging OUT "
                "fields only. See issue #915.");
      }
    }
  }

}

}  // namespace

//==============================================================================
// Admin module sweep
//==============================================================================

TEST_CASE("AutogenSweep - Admin all methods full dispatch battery",
          "[autogen][admin][sweep]") {
  EnsureInitialized();

  auto *pool_manager = CLIO_POOL_MANAGER;
  auto container = pool_manager->GetStaticContainer(clio::run::kAdminPoolId).get();
  if (container == nullptr) {
    INFO("Admin container not available - skipping test");
    return;
  }

  namespace adm = clio::run::admin;
  const std::vector<clio::run::u32> methods = {
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

  for (clio::run::u32 method : methods) {
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
  const std::vector<clio::run::u32> methods = {
      bd::Method::kCreate,         bd::Method::kDestroy,
      bd::Method::kMonitor,        bd::Method::kAllocateBlocks,
      bd::Method::kFreeBlocks,     bd::Method::kWrite,
      bd::Method::kRead,           bd::Method::kGetStats,
      bd::Method::kUpdate,
  };

  for (clio::run::u32 method : methods) {
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
  const std::vector<clio::run::u32> methods = {
      mn::Method::kCreate,        mn::Method::kDestroy,
      mn::Method::kMonitor,       mn::Method::kCustom,
      mn::Method::kCoMutexTest,   mn::Method::kCoRwLockTest,
      mn::Method::kWaitTest,      mn::Method::kTestLargeOutput,
      mn::Method::kGpuSubmit,     mn::Method::kSubtaskTest,
  };

  for (clio::run::u32 method : methods) {
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
  auto container = pool_manager->GetStaticContainer(clio::run::kAdminPoolId).get();
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
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, clio::run::PoolQuery::Local(),
        std::string(q));
    if (task.IsNull()) continue;
    task.template Cast<clio::run::Task>()->BeginRunContext();
    clio::run::SetCurrentTask(task.template Cast<clio::run::Task>());
    auto tr = container->Run(adm::Method::kMonitor,
                             task.template Cast<clio::run::Task>());
    for (int spin = 0; !tr.done() && spin < 16; ++spin) {
      tr.resume();
    }
    task.reset();
  }

  SECTION("Heartbeat handler");
  {
    auto task = container->NewTask(adm::Method::kHeartbeat);
    if (!task.IsNull()) {
      task->BeginRunContext();
      clio::run::SetCurrentTask(task);
      auto tr = container->Run(adm::Method::kHeartbeat, task);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
    }
  }

  SECTION("WreapDeadIpcs handler (no dead clients, quick scan)");
  {
    auto task = container->NewTask(adm::Method::kWreapDeadIpcs);
    if (!task.IsNull()) {
      task->BeginRunContext();
      clio::run::SetCurrentTask(task);
      auto tr = container->Run(adm::Method::kWreapDeadIpcs, task);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
    }
  }

  SECTION("SystemMonitor handler");
  {
    auto task = container->NewTask(adm::Method::kSystemMonitor);
    if (!task.IsNull()) {
      task->BeginRunContext();
      clio::run::SetCurrentTask(task);
      auto tr = container->Run(adm::Method::kSystemMonitor, task);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
    }
  }

  SECTION("MigrateContainers handler with empty migration list (no-op)");
  {
    auto task = container->NewTask(adm::Method::kMigrateContainers);
    if (!task.IsNull()) {
      task->BeginRunContext();
      clio::run::SetCurrentTask(task);
      auto tr = container->Run(adm::Method::kMigrateContainers, task);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
    }
  }

  SECTION("RecoverContainers handler with empty recovery list (no-op)");
  {
    auto task = container->NewTask(adm::Method::kRecoverContainers);
    if (!task.IsNull()) {
      task->BeginRunContext();
      clio::run::SetCurrentTask(task);
      auto tr = container->Run(adm::Method::kRecoverContainers, task);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
    }
  }

  SECTION("ChangeAddressTable handler with empty table (no-op)");
  {
    auto task = container->NewTask(adm::Method::kChangeAddressTable);
    if (!task.IsNull()) {
      task->BeginRunContext();
      clio::run::SetCurrentTask(task);
      auto tr = container->Run(adm::Method::kChangeAddressTable, task);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
    }
  }

  SECTION("GetOrCreatePool handler with a nonexistent module (error path)");
  {
    auto task = container->NewTask(adm::Method::kGetOrCreatePool);
    if (!task.IsNull()) {
      auto typed = task.template Cast<
          adm::GetOrCreatePoolTask<adm::CreateParams>>();
      typed->chimod_name_ =
          clio::run::priv::string(CTP_MALLOC, "no_such_module_xyz");
      typed->pool_name_ = clio::run::priv::string(CTP_MALLOC, "no_such_pool_xyz");
      task->BeginRunContext();
      clio::run::SetCurrentTask(task);
      auto tr = container->Run(adm::Method::kGetOrCreatePool, task);
      for (int spin = 0; !tr.done() && spin < 64; ++spin) {
        tr.resume();
      }
    }
  }

  SECTION("DestroyPool handler with a bogus pool id (error path)");
  {
    auto task = container->NewTask(adm::Method::kDestroyPool);
    if (!task.IsNull()) {
      task->BeginRunContext();
      clio::run::SetCurrentTask(task);
      auto tr = container->Run(adm::Method::kDestroyPool, task);
      for (int spin = 0; !tr.done() && spin < 64; ++spin) {
        tr.resume();
      }
    }
  }

  SECTION("RegisterMemory handler with default (invalid) registration");
  {
    auto task = container->NewTask(adm::Method::kRegisterMemory);
    if (!task.IsNull()) {
      task->BeginRunContext();
      clio::run::SetCurrentTask(task);
      auto tr = container->Run(adm::Method::kRegisterMemory, task);
      for (int spin = 0; !tr.done() && spin < 16; ++spin) {
        tr.resume();
      }
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
  const std::vector<clio::run::u32> methods = {
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

  for (clio::run::u32 method : methods) {
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
  const std::vector<clio::run::u32> methods = {
      ca::Method::kCreate,         ca::Method::kDestroy,
      ca::Method::kMonitor,        ca::Method::kParseOmni,
      ca::Method::kExportData,     ca::Method::kImportData,
      ca::Method::kGetOrCreateTag, ca::Method::kPutBlob,
      ca::Method::kGetBlob,        ca::Method::kSemanticSearch,
  };

  for (clio::run::u32 method : methods) {
    SweepMethod(cae_runtime, method);
  }
  REQUIRE(true);
}

//==============================================================================
// Issue #915 — AggregateOut must merge OUT fields only.
//
// The per-method sweep above stamps distinct identities on origin and replica
// for EVERY method of every module linked here, so a whole-task copy fails
// immediately. This case pins the two specifics the issue calls out, on the
// task where they actually bit: an shm-allocated IN/INOUT string, and the
// OUT-field merge semantics.
//
// PutBlobTask::blob_name_ is a priv::string owned by the CLIENT's allocator.
// A whole-task copy re-assigns it from the replica's segment, freeing the
// client's buffer through the wrong allocator — the `free(): invalid pointer`
// abort of #856/#500. AggregateOut must leave it untouched.
//==============================================================================

TEST_CASE("AggregateOut merges OUT fields only, never the whole task (#915)",
          "[autogen][cte][aggregate_out][contract]") {
  EnsureInitialized();

  clio::cte::core::Runtime cte_runtime;
  namespace ct = clio::cte::core;

  SECTION("PutBlobTask: shm string IN field and identity survive");
  {
    auto origin_base = cte_runtime.NewTask(ct::Method::kPutBlob);
    auto replica_base = cte_runtime.NewTask(ct::Method::kPutBlob);
    REQUIRE(!origin_base.IsNull());
    REQUIRE(!replica_base.IsNull());

    auto *origin = static_cast<ct::PutBlobTask *>(origin_base.get());
    auto *replica = static_cast<ct::PutBlobTask *>(replica_base.get());

    origin->blob_name_ = "origin-blob";
    origin->task_id_ =
        clio::run::TaskId(4242, 4343, 4444, /*replica_id=*/0, 4545, 7, 0xABCD);
    origin->pool_query_ = clio::run::PoolQuery::Broadcast();
    origin->context_.emulated_time_ns_ = 0;

    // A REPLICA looks exactly like the subtask that comes back off the wire:
    // same method, different task_id_ (carrying replica_id/net_key), its own
    // shm string, and the OUT state the remote handler produced.
    replica->blob_name_ = "replica-blob";
    replica->task_id_ =
        clio::run::TaskId(9999, 9898, 9797, /*replica_id=*/1, 9696, 3, 0x1234);
    replica->pool_query_ = clio::run::PoolQuery::Physical(3);
    replica->context_.emulated_time_ns_ = 12345;
    replica->context_.transform_flags_ = 0x2;

    const clio::run::TaskId id_before = origin->task_id_;
    const clio::run::PoolQuery query_before = origin->pool_query_;
    const clio::run::u32 method_before = origin->method_;

    cte_runtime.AggregateOut(ct::Method::kPutBlob, origin_base, replica_base);

    // Identity is untouched.
    REQUIRE(origin->task_id_ == id_before);
    REQUIRE(memcmp(&origin->pool_query_, &query_before,
                   sizeof(clio::run::PoolQuery)) == 0);
    REQUIRE(origin->method_ == method_before);

    // The client's shm string is untouched: a whole-task copy would have made
    // this "replica-blob" and freed the origin's buffer through the replica's
    // allocator.
    REQUIRE(origin->blob_name_.str() == "origin-blob");

    // The OUT state DID come across (this is what AggregateOut is for).
    REQUIRE(origin->context_.emulated_time_ns_ == 12345);
    REQUIRE((origin->context_.transform_flags_ & 0x2) != 0);
  }

  SECTION("FlushDataTask: OUT counters SUM across replicas");
  {
    auto origin_base = cte_runtime.NewTask(ct::Method::kFlushData);
    auto r1 = cte_runtime.NewTask(ct::Method::kFlushData);
    auto r2 = cte_runtime.NewTask(ct::Method::kFlushData);
    REQUIRE(!origin_base.IsNull());
    REQUIRE(!r1.IsNull());
    REQUIRE(!r2.IsNull());

    auto *origin = static_cast<ct::FlushDataTask *>(origin_base.get());
    auto *rep1 = static_cast<ct::FlushDataTask *>(r1.get());
    auto *rep2 = static_cast<ct::FlushDataTask *>(r2.get());

    origin->bytes_flushed_ = 0;
    origin->blobs_flushed_ = 0;
    rep1->bytes_flushed_ = 100;
    rep1->blobs_flushed_ = 1;
    rep2->bytes_flushed_ = 250;
    rep2->blobs_flushed_ = 4;

    cte_runtime.AggregateOut(ct::Method::kFlushData, origin_base, r1);
    cte_runtime.AggregateOut(ct::Method::kFlushData, origin_base, r2);

    // Last-replica-wins (what the whole-task copy did) would leave 250/4.
    REQUIRE(origin->bytes_flushed_ == 350);
    REQUIRE(origin->blobs_flushed_ == 5);
  }
}

SIMPLE_TEST_MAIN()
