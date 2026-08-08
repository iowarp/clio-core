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
 * Persistence for the per-pool task-stat model (issue #956).
 *
 * The scheduler routes every task on Container::InferCpuTime /
 * InferWallClockTime, whose coefficients are learned by SGD from completed
 * tasks. Without persistence those coefficients reset to the 1.0 seed on every
 * restart, so a short run's routing decisions are dominated by an untrained
 * model and are not comparable between runs. This file saves and restores them.
 */

#include "clio_runtime/task_stat_model.h"

#include <yaml-cpp/yaml.h>

#include <cctype>
#include <filesystem>
#include <fstream>

#include "clio_runtime/config_manager.h"
#include "clio_runtime/container.h"

namespace clio::run {

namespace {

/** Replace anything that is not [A-Za-z0-9._-] with '_' so a pool name that is
 *  a filesystem path (CTE names its pools "/mnt/hdd/cte") cannot escape the
 *  models directory or create nested paths. */
std::string SanitizeForFilename(const std::string &name) {
  std::string out;
  out.reserve(name.size());
  for (char c : name) {
    unsigned char uc = static_cast<unsigned char>(c);
    if (std::isalnum(uc) || c == '.' || c == '-') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out;
}

}  // namespace

std::string TaskStatModelPath(const std::string &chimod_name,
                              const std::string &pool_name, u32 node_id) {
  auto *config_manager = CLIO_CONFIG_MANAGER;
  if (!config_manager) {
    return std::string();
  }
  return config_manager->GetConfDir() + "/models/" +
         SanitizeForFilename(chimod_name) + "." + SanitizeForFilename(pool_name) +
         "." + std::to_string(node_id) + ".yaml";
}

bool TaskStatModelSnapshot::Save(const std::string &path) const {
  if (path.empty()) {
    return false;
  }
  std::error_code ec;
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path(), ec);
  if (ec) {
    HLOG(kError, "TaskStatModel: cannot create model directory for {}: {}",
         path, ec.message());
    return false;
  }

  YAML::Emitter out;
  out << YAML::BeginMap;
  out << YAML::Key << "chimod_name" << YAML::Value << chimod_name_;
  out << YAML::Key << "pool_name" << YAML::Value << pool_name_;
  out << YAML::Key << "learning_rate" << YAML::Value << learning_rate_;
  out << YAML::Key << "methods" << YAML::Value << YAML::BeginMap;
  for (const auto &kv : methods_) {
    out << YAML::Key << kv.first << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "cpu_coef" << YAML::Value << kv.second.cpu_coef_;
    out << YAML::Key << "cpu_mape" << YAML::Value << kv.second.cpu_mape_;
    out << YAML::Key << "wall_coef" << YAML::Value << kv.second.wall_coef_;
    out << YAML::Key << "wall_mape" << YAML::Value << kv.second.wall_mape_;
    out << YAML::EndMap;
  }
  out << YAML::EndMap;
  out << YAML::EndMap;

  // Write to a temp file and rename: a crash (or a kill during the periodic
  // flush) must not be able to leave a half-written file that the next startup
  // would happily load as the learned model.
  const std::string tmp_path = path + ".tmp";
  {
    std::ofstream ofs(tmp_path, std::ios::trunc);
    if (!ofs.is_open()) {
      HLOG(kError, "TaskStatModel: failed to open {} for writing", tmp_path);
      return false;
    }
    ofs << out.c_str() << "\n";
    if (!ofs.good()) {
      HLOG(kError, "TaskStatModel: failed to write {}", tmp_path);
      return false;
    }
  }
  std::filesystem::rename(tmp_path, path, ec);
  if (ec) {
    HLOG(kError, "TaskStatModel: failed to install {}: {}", path, ec.message());
    std::filesystem::remove(tmp_path, ec);
    return false;
  }
  return true;
}

bool TaskStatModelSnapshot::Load(const std::string &path) {
  methods_.clear();
  if (path.empty()) {
    return false;
  }
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    return false;  // first run for this pool — not an error
  }
  try {
    YAML::Node root = YAML::LoadFile(path);
    if (!root || !root.IsMap()) {
      HLOG(kWarning, "TaskStatModel: {} is not a YAML map; ignoring", path);
      return false;
    }
    if (root["chimod_name"]) {
      chimod_name_ = root["chimod_name"].as<std::string>();
    }
    if (root["pool_name"]) {
      pool_name_ = root["pool_name"].as<std::string>();
    }
    if (root["learning_rate"]) {
      learning_rate_ = root["learning_rate"].as<float>();
    }
    YAML::Node methods = root["methods"];
    if (methods && methods.IsMap()) {
      for (const auto &kv : methods) {
        MethodStatWeights w;
        const YAML::Node &node = kv.second;
        if (!node.IsMap()) {
          continue;
        }
        if (node["cpu_coef"]) w.cpu_coef_ = node["cpu_coef"].as<float>();
        if (node["cpu_mape"]) w.cpu_mape_ = node["cpu_mape"].as<float>();
        if (node["wall_coef"]) w.wall_coef_ = node["wall_coef"].as<float>();
        if (node["wall_mape"]) w.wall_mape_ = node["wall_mape"].as<float>();
        methods_[kv.first.as<std::string>()] = w;
      }
    }
  } catch (const std::exception &e) {
    // A corrupt model file must never take the runtime down: the model is an
    // optimization, and starting from the seed is a correct (just slower)
    // fallback.
    HLOG(kWarning, "TaskStatModel: failed to parse {} ({}); using seed model",
         path, e.what());
    methods_.clear();
    return false;
  }
  return true;
}

//=============================================================================
// Container <-> snapshot
//=============================================================================

TaskStatModelSnapshot Container::ExportModel() const {
  const Container &owner = ModelOwner();
  TaskStatModelSnapshot snap;
  snap.pool_name_ = owner.pool_name_;
  snap.learning_rate_ = owner.learning_rate_;
  // Names come from THIS container: the static container is Init'd by the same
  // module, so the tables match, but reading them from the caller keeps the
  // export usable even if a model owner was never given names.
  const std::vector<std::string> &names =
      method_names_.empty() ? owner.method_names_ : method_names_;
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i].empty()) {
      continue;
    }
    MethodStatWeights w;
    if (i < owner.method_model_.size()) w.cpu_coef_ = owner.method_model_[i];
    if (i < owner.method_mape_.size()) w.cpu_mape_ = owner.method_mape_[i];
    if (i < owner.method_model_wall_.size()) {
      w.wall_coef_ = owner.method_model_wall_[i];
    }
    if (i < owner.method_mape_wall_.size()) {
      w.wall_mape_ = owner.method_mape_wall_[i];
    }
    snap.methods_[names[i]] = w;
  }
  return snap;
}

size_t Container::ImportModel(const TaskStatModelSnapshot &snapshot) {
  Container &owner = ModelOwner();
  const std::vector<std::string> &names =
      method_names_.empty() ? owner.method_names_ : method_names_;
  size_t restored = 0;
  for (size_t i = 0; i < names.size(); ++i) {
    auto it = snapshot.methods_.find(names[i]);
    if (it == snapshot.methods_.end()) {
      continue;  // new method, or one this pool never exercised: keep the seed
    }
    const MethodStatWeights &w = it->second;
    if (i < owner.method_model_.size()) owner.method_model_[i] = w.cpu_coef_;
    if (i < owner.method_mape_.size()) owner.method_mape_[i] = w.cpu_mape_;
    if (i < owner.method_model_wall_.size()) {
      owner.method_model_wall_[i] = w.wall_coef_;
    }
    if (i < owner.method_mape_wall_.size()) {
      owner.method_mape_wall_[i] = w.wall_mape_;
    }
    ++restored;
  }
  // A restore is not new learning: leaving the flag clear keeps the periodic
  // flush from rewriting a file identical to the one just read.
  owner.model_dirty_.store(false, std::memory_order_relaxed);
  return restored;
}

}  // namespace clio::run
