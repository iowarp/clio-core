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
 * Issue #956: the per-pool task-stat model.
 *
 * Covers the three properties the scheduler analysis depends on:
 *   1. a pool has a dedicated static container, distinct from the container
 *      that runs its methods, and that static container owns the model;
 *   2. every container of the pool reads and writes THAT model, so what the
 *      monitor reports is what the scheduler used;
 *   3. the weights survive a save/restore, matched by method NAME so a
 *      renumbered method enum cannot graft one method's model onto another.
 */

#include "../simple_test.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/container.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/task_stat_model.h"

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>

using namespace clio::run;

namespace {

/** Bring up the in-process runtime once for the whole binary. */
bool InitRuntimeOnce() {
  static const bool ok = [] {
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    if (!success) {
      return false;
    }
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return true;
  }();
  return ok;
}

/** Create a small RAM bdev pool and return its (resolved) PoolId. */
PoolId CreateRamBdevPool(const std::string &pool_name, u32 major) {
  PoolId pool_id(major, 0);
  clio::run::bdev::Client client(pool_id);
  auto create_task = client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), pool_name, pool_id,
      clio::run::bdev::BdevType::kRam, 16 * 1024 * 1024);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);
  return create_task->new_pool_id_;
}

bool NearlyEqual(float a, float b) { return std::fabs(a - b) < 1e-4f; }

}  // namespace

TEST_CASE("TaskStatModel: snapshot survives a YAML round trip",
          "[task_stat_model]") {
  std::error_code ec;
  std::filesystem::path dir =
      std::filesystem::temp_directory_path(ec) / "clio_model_956";
  std::filesystem::create_directories(dir, ec);
  const std::string path = (dir / "roundtrip.yaml").string();
  std::filesystem::remove(path, ec);

  TaskStatModelSnapshot saved;
  saved.chimod_name_ = "bdev";
  saved.pool_name_ = "/mnt/nvme/scratch";
  saved.learning_rate_ = 0.2f;
  saved.methods_["Write"] = MethodStatWeights{2.5f, 0.125f, 7.5f, 0.25f};
  saved.methods_["Read"] = MethodStatWeights{1.5f, 0.0625f, 3.25f, 0.5f};
  REQUIRE(saved.Save(path));

  TaskStatModelSnapshot loaded;
  REQUIRE(loaded.Load(path));
  REQUIRE(loaded.chimod_name_ == "bdev");
  REQUIRE(loaded.pool_name_ == "/mnt/nvme/scratch");
  REQUIRE(loaded.methods_.size() == 2);
  REQUIRE(NearlyEqual(loaded.methods_["Write"].cpu_coef_, 2.5f));
  REQUIRE(NearlyEqual(loaded.methods_["Write"].cpu_mape_, 0.125f));
  REQUIRE(NearlyEqual(loaded.methods_["Write"].wall_coef_, 7.5f));
  REQUIRE(NearlyEqual(loaded.methods_["Write"].wall_mape_, 0.25f));
  REQUIRE(NearlyEqual(loaded.methods_["Read"].cpu_coef_, 1.5f));

  // A missing file is the normal first-run case, not an error to propagate.
  TaskStatModelSnapshot absent;
  REQUIRE_FALSE(absent.Load((dir / "does_not_exist.yaml").string()));
  REQUIRE(absent.Empty());

  // Garbage must degrade to the seed model rather than take the runtime down.
  const std::string bad_path = (dir / "corrupt.yaml").string();
  {
    std::ofstream ofs(bad_path, std::ios::trunc);
    ofs << "methods: [this is not: a map\n";
  }
  TaskStatModelSnapshot corrupt;
  corrupt.Load(bad_path);  // must not throw
  REQUIRE(corrupt.Empty());
}

TEST_CASE("TaskStatModel: pool has a dedicated static container that owns "
          "the model", "[task_stat_model]") {
  REQUIRE(InitRuntimeOnce());
  PoolId pool_id = CreateRamBdevPool("model_static_956", 9956);

  auto *pool_manager = CLIO_POOL_MANAGER;
  REQUIRE(pool_manager != nullptr);
  u32 node_id = CLIO_IPC->GetNodeId();

  DynamicContainer static_dc = pool_manager->GetStaticContainer(pool_id);
  DynamicContainer local_dc = pool_manager->GetContainer(pool_id, node_id);
  REQUIRE(static_dc.IsValid());
  REQUIRE(local_dc.IsValid());
  // The static container is its own instance: it never ran Create, so it must
  // not be the container that serves the pool's tasks.
  REQUIRE(static_dc.get() != local_dc.get());
  REQUIRE(static_dc.get()->container_id_ == kStaticContainerId);
  REQUIRE(static_dc.get()->IsModelOwner());
  REQUIRE_FALSE(local_dc.get()->IsModelOwner());

  // A write through the serving container lands on the static container's
  // model, and inference through the serving container reads it back.
  const u32 method = clio::run::bdev::Method::kWrite;
  local_dc.get()->SetMethodWallCoef(method, 12.5f);
  REQUIRE(NearlyEqual(static_dc.get()->GetMethodModelWall()[method], 12.5f));

  TaskStat stat;
  stat.wall_time_ = 3.0f;
  REQUIRE(NearlyEqual(local_dc.get()->InferWallClockTime(method, stat),
                      12.5f * 4.0f));

  // Reinforcement from a completed task moves the shared model, not a private
  // per-container copy.
  TaskStat cpu_stat;
  cpu_stat.compute_ = 9;
  float before = static_dc.get()->GetMethodModel()[method];
  local_dc.get()->ReinforceCpuModel(method, /*pred=*/100.0f, /*real=*/10.0f,
                                    cpu_stat);
  float after = static_dc.get()->GetMethodModel()[method];
  REQUIRE(after < before);  // over-prediction must shrink the coefficient
  REQUIRE(NearlyEqual(local_dc.get()->GetMethodModel()[method], after));
}

TEST_CASE("TaskStatModel: weights persist and restore by method name",
          "[task_stat_model]") {
  REQUIRE(InitRuntimeOnce());
  PoolId pool_id = CreateRamBdevPool("model_persist_956", 9957);

  auto *pool_manager = CLIO_POOL_MANAGER;
  u32 node_id = CLIO_IPC->GetNodeId();
  DynamicContainer static_dc = pool_manager->GetStaticContainer(pool_id);
  REQUIRE(static_dc.IsValid());

  const u32 write_id = clio::run::bdev::Method::kWrite;
  const u32 read_id = clio::run::bdev::Method::kRead;
  static_dc.get()->SetMethodCpuCoef(write_id, 4.25f);
  static_dc.get()->SetMethodWallCoef(read_id, 6.75f);

  // Flush every pool's model, then read this pool's file back.
  pool_manager->FlushModels(/*force=*/true);
  const PoolInfo *info = pool_manager->GetPoolInfo(pool_id);
  REQUIRE(info != nullptr);
  const std::string path =
      TaskStatModelPath(info->chimod_name_, "model_persist_956", node_id);
  REQUIRE_FALSE(path.empty());

  TaskStatModelSnapshot on_disk;
  REQUIRE(on_disk.Load(path));
  REQUIRE(NearlyEqual(on_disk.methods_["Write"].cpu_coef_, 4.25f));
  REQUIRE(NearlyEqual(on_disk.methods_["Read"].wall_coef_, 6.75f));

  // Restoring is what a reloaded container does: overwrite the seeded table
  // from the file. Only names the binary still defines are applied; a method
  // absent from the file keeps its seed.
  static_dc.get()->SetMethodCpuCoef(write_id, 1.0f);
  TaskStatModelSnapshot restore;
  restore.methods_["Write"] = MethodStatWeights{4.25f, 0.5f, 1.0f, 0.0f};
  restore.methods_["MethodThatNoLongerExists"] =
      MethodStatWeights{99.0f, 99.0f, 99.0f, 99.0f};
  size_t restored = static_dc.get()->ImportModel(restore);
  REQUIRE(restored == 1);  // the unknown name is ignored, not misapplied
  REQUIRE(NearlyEqual(static_dc.get()->GetMethodModel()[write_id], 4.25f));
  // Read was not in the snapshot, so it keeps the value it already had.
  REQUIRE(NearlyEqual(static_dc.get()->GetMethodModelWall()[read_id], 6.75f));
}

SIMPLE_TEST_MAIN()
