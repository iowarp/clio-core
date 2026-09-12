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
 * Covers the properties the scheduler analysis depends on (issues #956, #994):
 *   1. a pool has a dedicated static container, distinct from the container
 *      that runs its methods, and the model is NOT on it: every container owns
 *      and reinforces its own per-method statistics, so a write through one
 *      container never moves another container's coefficients;
 *   2. the weights survive a save/restore per container id, matched by method
 *      NAME so a renumbered method enum cannot graft one method's model onto
 *      another, and a container registered later restores its own file.
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
  saved.container_id_ = 3;
  saved.learning_rate_ = 0.2f;
  saved.methods_["Write"] = MethodStatWeights{2.5f, 0.125f, 7.5f, 0.25f};
  saved.methods_["Read"] = MethodStatWeights{1.5f, 0.0625f, 3.25f, 0.5f};
  REQUIRE(saved.Save(path));

  TaskStatModelSnapshot loaded;
  REQUIRE(loaded.Load(path));
  REQUIRE(loaded.chimod_name_ == "bdev");
  REQUIRE(loaded.pool_name_ == "/mnt/nvme/scratch");
  REQUIRE(loaded.container_id_ == 3);
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

/**
 * Build a second real container for `pool_id` on this node with container id
 * `container_id`, Init it exactly as CreatePool does (identity + model table
 * + method names), and register it. This is what recovery does when a dead
 * peer's container is re-hosted here (#856), and it is the only way to get
 * two containers of one pool onto a single-node test runtime.
 */
DynamicContainer RegisterExtraContainer(PoolId pool_id, u32 container_id) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  const PoolInfo *info = pool_manager->GetPoolInfo(pool_id);
  REQUIRE(info != nullptr);
  DynamicContainer dc(info->chimod_name_, pool_id, info->pool_name_);
  REQUIRE(dc.IsValid());
  dc.get()->Init(pool_id, info->pool_name_, container_id);
  REQUIRE(pool_manager->RegisterContainer(pool_id, container_id, dc));
  return dc;
}

TEST_CASE("TaskStatModel: every container owns its own model; the static "
          "container owns none",
          "[task_stat_model]") {
  REQUIRE(InitRuntimeOnce());
  PoolId pool_id = CreateRamBdevPool("model_per_container_994", 9956);

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

  // A second container of the same pool on this node (what recovery produces).
  const u32 other_id = node_id + 41;
  DynamicContainer other_dc = RegisterExtraContainer(pool_id, other_id);
  REQUIRE(pool_manager->GetContainer(pool_id, other_id).get() ==
          other_dc.get());
  REQUIRE(pool_manager->GetLocalContainers(pool_id).size() == 2);

  const u32 method = clio::run::bdev::Method::kWrite;
  const float static_wall_before =
      static_dc.get()->GetMethodModelWall()[method];
  const float other_wall_before = other_dc.get()->GetMethodModelWall()[method];

  // A seed written through the serving container lands on THAT container only:
  // neither the static container nor the pool's other container sees it.
  local_dc.get()->SetMethodWallCoef(method, 12.5f);
  REQUIRE(NearlyEqual(local_dc.get()->GetMethodModelWall()[method], 12.5f));
  REQUIRE(NearlyEqual(static_dc.get()->GetMethodModelWall()[method],
                      static_wall_before));
  REQUIRE(NearlyEqual(other_dc.get()->GetMethodModelWall()[method],
                      other_wall_before));

  // Inference reads the container's own weights.
  TaskStat stat;
  stat.wall_time_ = 3.0f;
  REQUIRE(NearlyEqual(local_dc.get()->InferWallClockTime(method, stat),
                      12.5f * 4.0f));
  REQUIRE(NearlyEqual(other_dc.get()->InferWallClockTime(method, stat),
                      other_wall_before * 4.0f));

  // Reinforcement from a completed task moves ONLY the container it ran on.
  TaskStat cpu_stat;
  cpu_stat.compute_ = 9;
  const float local_before = local_dc.get()->GetMethodModel()[method];
  const float other_before = other_dc.get()->GetMethodModel()[method];
  const float static_before = static_dc.get()->GetMethodModel()[method];
  local_dc.get()->ReinforceCpuModel(method, /*pred=*/100.0f, /*real=*/10.0f,
                                    cpu_stat);
  const float local_after = local_dc.get()->GetMethodModel()[method];
  REQUIRE(local_after < local_before);  // over-prediction shrinks the coef
  REQUIRE(NearlyEqual(other_dc.get()->GetMethodModel()[method], other_before));
  REQUIRE(NearlyEqual(static_dc.get()->GetMethodModel()[method],
                      static_before));
  REQUIRE(local_dc.get()->IsModelDirty());
  REQUIRE_FALSE(other_dc.get()->IsModelDirty());
  REQUIRE_FALSE(static_dc.get()->IsModelDirty());

  // And the other way round: learning on the second container leaves the
  // serving container's coefficient exactly where it was.
  other_dc.get()->ReinforceCpuModel(method, /*pred=*/10.0f, /*real=*/100.0f,
                                    cpu_stat);
  REQUIRE(other_dc.get()->GetMethodModel()[method] > other_before);
  REQUIRE(NearlyEqual(local_dc.get()->GetMethodModel()[method], local_after));
}

TEST_CASE("TaskStatModel: weights persist and restore per container by "
          "method name",
          "[task_stat_model]") {
  REQUIRE(InitRuntimeOnce());
  PoolId pool_id = CreateRamBdevPool("model_persist_994", 9957);

  auto *pool_manager = CLIO_POOL_MANAGER;
  u32 node_id = CLIO_IPC->GetNodeId();
  DynamicContainer local_dc = pool_manager->GetContainer(pool_id, node_id);
  REQUIRE(local_dc.IsValid());
  const u32 other_id = node_id + 42;
  DynamicContainer other_dc = RegisterExtraContainer(pool_id, other_id);
  const PoolInfo *info = pool_manager->GetPoolInfo(pool_id);
  REQUIRE(info != nullptr);
  const std::string chimod_name = info->chimod_name_;

  const u32 write_id = clio::run::bdev::Method::kWrite;
  const u32 read_id = clio::run::bdev::Method::kRead;
  local_dc.get()->SetMethodCpuCoef(write_id, 4.25f);
  local_dc.get()->SetMethodWallCoef(read_id, 6.75f);
  other_dc.get()->SetMethodCpuCoef(write_id, 8.5f);
  other_dc.get()->SetMethodWallCoef(read_id, 13.5f);

  // Flush every container's model, then read each container's file back:
  // one file per container id, each holding that container's own weights.
  pool_manager->FlushModels(/*force=*/true);
  const std::string local_path =
      TaskStatModelPath(chimod_name, "model_persist_994", node_id, node_id);
  const std::string other_path =
      TaskStatModelPath(chimod_name, "model_persist_994", node_id, other_id);
  REQUIRE_FALSE(local_path.empty());
  REQUIRE(local_path != other_path);

  TaskStatModelSnapshot local_disk;
  REQUIRE(local_disk.Load(local_path));
  REQUIRE(local_disk.container_id_ == node_id);
  REQUIRE(NearlyEqual(local_disk.methods_["Write"].cpu_coef_, 4.25f));
  REQUIRE(NearlyEqual(local_disk.methods_["Read"].wall_coef_, 6.75f));
  TaskStatModelSnapshot other_disk;
  REQUIRE(other_disk.Load(other_path));
  REQUIRE(other_disk.container_id_ == other_id);
  REQUIRE(NearlyEqual(other_disk.methods_["Write"].cpu_coef_, 8.5f));
  REQUIRE(NearlyEqual(other_disk.methods_["Read"].wall_coef_, 13.5f));
  // The static container has nothing to persist and gets no file.
  REQUIRE_FALSE(std::filesystem::exists(TaskStatModelPath(
      chimod_name, "model_persist_994", node_id, kStaticContainerId)));

  // Restoring overwrites the seeded table from the file. Only names the binary
  // still defines are applied; a method absent from the file keeps its seed.
  local_dc.get()->SetMethodCpuCoef(write_id, 1.0f);
  TaskStatModelSnapshot restore;
  restore.methods_["Write"] = MethodStatWeights{4.25f, 0.5f, 1.0f, 0.0f};
  restore.methods_["MethodThatNoLongerExists"] =
      MethodStatWeights{99.0f, 99.0f, 99.0f, 99.0f};
  size_t restored = local_dc.get()->ImportModel(restore);
  REQUIRE(restored == 1);  // the unknown name is ignored, not misapplied
  REQUIRE(NearlyEqual(local_dc.get()->GetMethodModel()[write_id], 4.25f));
  // Read was not in the snapshot, so it keeps the value it already had.
  REQUIRE(NearlyEqual(local_dc.get()->GetMethodModelWall()[read_id], 6.75f));

  // A container registered later restores ITS OWN file (matched by container
  // id), not its neighbour's: pre-write a model for a third id, register a
  // container with that id, and check the weights arrived — while the file the
  // other container wrote is left alone.
  const u32 late_id = node_id + 43;
  TaskStatModelSnapshot late_snapshot;
  late_snapshot.methods_["Write"] = MethodStatWeights{21.0f, 0.0f, 22.0f, 0.0f};
  REQUIRE(late_snapshot.Save(
      TaskStatModelPath(chimod_name, "model_persist_994", node_id, late_id)));
  DynamicContainer late_dc = RegisterExtraContainer(pool_id, late_id);
  REQUIRE(NearlyEqual(late_dc.get()->GetMethodModel()[write_id], 21.0f));
  REQUIRE(NearlyEqual(late_dc.get()->GetMethodModelWall()[write_id], 22.0f));
  REQUIRE_FALSE(late_dc.get()->IsModelDirty());  // a restore is not learning
  REQUIRE(NearlyEqual(other_dc.get()->GetMethodModel()[write_id], 8.5f));
}

SIMPLE_TEST_MAIN()
