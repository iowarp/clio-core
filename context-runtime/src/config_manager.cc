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
 * Configuration manager implementation
 */

#include "chimaera/config_manager.h"
#include "chimaera/task.h"
#include "chimaera/ipc_manager.h"
#include <cstdlib>
#include <filesystem>

// Global pointer variable definition for Configuration manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chi::ConfigManager, g_config_manager);

namespace chi {

// Constructor and destructor removed - handled by HSHM singleton pattern

bool ConfigManager::ClientInit() {
  if (is_initialized_) {
    return true;
  }

  // Get configuration file path from environment
  config_file_path_ = GetServerConfigPath();
  HLOG(kInfo, "Config at: {}", config_file_path_);

  // Load YAML configuration if path is provided
  if (!config_file_path_.empty()) {
    if (!LoadYaml(config_file_path_)) {
      HLOG(kError,
            "Warning: Failed to load configuration from {}, using defaults",
            config_file_path_);
    }
  }

  // Check CHI_PORT env var (overrides YAML and default)
  std::string port_env = hshm::SystemInfo::Getenv("CHI_PORT");
  if (!port_env.empty()) {
    port_ = std::stoul(port_env);
  }

  // Check CHI_SERVER_ADDR env var (overrides default 127.0.0.1)
  std::string addr_env = hshm::SystemInfo::Getenv("CHI_SERVER_ADDR");
  if (!addr_env.empty()) {
    server_addr_ = addr_env;
  }

  is_initialized_ = true;
  return true;
}

bool ConfigManager::ServerInit() {
  // Configuration is needed by both client and server, so same implementation
  return ClientInit();
}

bool ConfigManager::LoadYaml(const std::string &config_path) {
  try {
    // Use HSHM BaseConfig methods
    LoadFromFile(config_path, true);
    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

std::string ConfigManager::GetServerConfigPath() const {
  // Check CHI_SERVER_CONF first (primary)
  const char *chi_env_path = std::getenv("CHI_SERVER_CONF");
  if (chi_env_path) {
    return std::string(chi_env_path);
  }

  // Fall back to WRP_RUNTIME_CONF (secondary)
  const char *wrp_env_path = std::getenv("WRP_RUNTIME_CONF");
  if (wrp_env_path) {
    return std::string(wrp_env_path);
  }

  // Fall back to ~/.chimaera/chimaera.yaml (tertiary)
  std::string home_config =
      hshm::ConfigParse::ExpandPath("${HOME}/.chimaera/chimaera.yaml");
  if (std::filesystem::exists(home_config)) {
    return home_config;
  }

  return std::string();
}

size_t ConfigManager::GetMemorySegmentSize(MemorySegment segment) const {
  switch (segment) {
  case kMainSegment:
    return CalculateMainSegmentSize();
  case kClientDataSegment:
    return client_data_segment_size_;
  default:
    return 0;
  }
}

u32 ConfigManager::GetPort() const { return port_; }

std::string ConfigManager::GetServerAddr() const { return server_addr_; }

u32 ConfigManager::GetNeighborhoodSize() const { return neighborhood_size_; }

std::string
ConfigManager::GetSharedMemorySegmentName(MemorySegment segment) const {
  std::string segment_name;

  switch (segment) {
  case kMainSegment:
    segment_name = main_segment_name_;
    break;
  case kClientDataSegment:
    segment_name = client_data_segment_name_;
    break;
  default:
    return "";
  }

  // Use HSHM's ExpandPath to resolve environment variables
  return hshm::ConfigParse::ExpandPath(segment_name);
}

std::string ConfigManager::GetHostfilePath() const {
  if (hostfile_path_.empty()) {
    return "";
  }

  // Use HSHM's ExpandPath to resolve environment variables in hostfile path
  return hshm::ConfigParse::ExpandPath(hostfile_path_);
}

bool ConfigManager::IsValid() const { return is_initialized_; }

void ConfigManager::LoadDefault() {
  // Set default configuration values
  num_threads_ = 4;
  queue_depth_ = 1024;

  main_segment_size_ = 0;                         // 0 means auto-calculate
  client_data_segment_size_ = 512 * 1024 * 1024;  // 512MB

  port_ = 9413;
  neighborhood_size_ = 32;

  // Set default shared memory segment names with environment variables
  main_segment_name_ = "chi_main_segment_${USER}";
  client_data_segment_name_ = "chi_client_data_segment_${USER}";

  // Set default hostfile path (empty means no networking/distributed mode)
  hostfile_path_ = "";

  // Set default network retry configuration
  wait_for_restart_timeout_ = 30;      // 30 seconds
  wait_for_restart_poll_period_ = 1;   // 1 second

  // Set default worker sleep configuration (in microseconds)
  first_busy_wait_ = 50;               // 50us busy wait
  max_sleep_ = 50000;                  // 50000us (50ms) maximum sleep
}

void ConfigManager::ParseYAML(YAML::Node &yaml_conf) {
  // Parse runtime configuration (consolidated worker threads and runtime parameters)
  // This section now includes worker thread configuration previously in 'workers' section
  if (yaml_conf["runtime"]) {
    auto runtime = yaml_conf["runtime"];

    // New unified worker thread configuration
    if (runtime["num_threads"]) {
      num_threads_ = runtime["num_threads"].as<u32>();
    }

    // Queue depth configuration
    if (runtime["queue_depth"]) {
      queue_depth_ = runtime["queue_depth"].as<u32>();
    }

    // Local task scheduler
    if (runtime["local_sched"]) {
      local_sched_ = runtime["local_sched"].as<std::string>();
    }

    // Worker sleep configuration
    if (runtime["first_busy_wait"]) {
      first_busy_wait_ = runtime["first_busy_wait"].as<u32>();
    }

    // Configuration directory for persistent runtime config
    if (runtime["conf_dir"]) {
      conf_dir_ = runtime["conf_dir"].as<std::string>();
    }

    // Note: stack_size parameter removed (was never used)
    // Note: heartbeat_interval parsing removed (not used by runtime)
  }

  // Parse networking
  if (yaml_conf["networking"]) {
    auto networking = yaml_conf["networking"];
    if (networking["port"]) {
      port_ = networking["port"].as<u32>();
    }
    if (networking["neighborhood_size"]) {
      neighborhood_size_ = networking["neighborhood_size"].as<u32>();
    }
    if (networking["hostfile"]) {
      hostfile_path_ = networking["hostfile"].as<std::string>();
    }
    if (networking["wait_for_restart"]) {
      wait_for_restart_timeout_ = networking["wait_for_restart"].as<u32>();
    }
    if (networking["wait_for_restart_poll_period"]) {
      wait_for_restart_poll_period_ = networking["wait_for_restart_poll_period"].as<u32>();
    }
  }

  // Segment names are hardcoded and expanded in ipc_manager.cc
  // No configuration needed here

  // Note: Runtime section parsing is done at the beginning of ParseYAML
  // to consolidate worker thread configuration with other runtime parameters

  // Parse compose section
  if (yaml_conf["compose"]) {
    auto compose_list = yaml_conf["compose"];
    if (compose_list.IsSequence()) {
      for (const auto& pool_node : compose_list) {
        PoolConfig pool_config;

        // Extract required fields
        if (pool_node["mod_name"]) {
          pool_config.mod_name_ = pool_node["mod_name"].as<std::string>();
        }
        if (pool_node["pool_name"]) {
          pool_config.pool_name_ = pool_node["pool_name"].as<std::string>();
        }
        if (pool_node["pool_id"]) {
          std::string pool_id_str = pool_node["pool_id"].as<std::string>();
          pool_config.pool_id_ = PoolId::FromString(pool_id_str);
        }
        if (pool_node["pool_query"]) {
          std::string query_str = pool_node["pool_query"].as<std::string>();
          pool_config.pool_query_ = PoolQuery::FromString(query_str);
        }

        // Store entire YAML node as config string for module-specific parsing
        YAML::Emitter emitter;
        emitter << pool_node;
        pool_config.config_ = emitter.c_str();

        // Parse restart field if present
        if (pool_node["restart"]) {
          pool_config.restart_ = pool_node["restart"].as<bool>();
        }

        // Add to compose config
        compose_config_.pools_.push_back(pool_config);
      }
    }
  }
}

size_t ConfigManager::CalculateMainSegmentSize() const {
  // If main_segment_size is explicitly set (non-zero), use it
  if (main_segment_size_ > 0) {
    return main_segment_size_;
  }

  // Auto-calculate based on queue_depth and num_threads using exact ring_buffer sizes
  constexpr size_t BASE_OVERHEAD = 32 * 1024 * 1024;  // 32MB for metadata
  constexpr u32 NUM_PRIORITIES = 2;                    // normal + resumed

  // Calculate total workers: num_threads + 1 network worker
  u32 total_workers = num_threads_ + 1;

  // Calculate worker task queues size: TaskQueue with total_workers lanes
  size_t worker_queues_size = TaskQueue::CalculateSize(
      total_workers,      // num_lanes
      NUM_PRIORITIES,     // num_priorities
      queue_depth_);      // depth per queue

  // Calculate network queue size: NetQueue with 1 lane
  size_t net_queue_size = NetQueue::CalculateSize(
      1,                  // num_lanes
      NUM_PRIORITIES,     // num_priorities
      queue_depth_);      // depth per queue

  // Total size: BASE_OVERHEAD + queues_size * num_workers
  size_t queues_size = worker_queues_size + net_queue_size;
  size_t calculated = BASE_OVERHEAD + (queues_size * total_workers);

  return calculated;
}

} // namespace chi