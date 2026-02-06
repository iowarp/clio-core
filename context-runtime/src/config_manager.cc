/**
 * Configuration manager implementation
 */

#include "chimaera/config_manager.h"
#include <cstdlib>

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

  return std::string();
}

u32 ConfigManager::GetWorkerThreadCount(ThreadType thread_type) const {
  switch (thread_type) {
  case kSchedWorker:
    return sched_workers_;
  case kProcessReaper:
    return process_reaper_workers_;
  default:
    return 0;
  }
}

size_t ConfigManager::GetMemorySegmentSize(MemorySegment segment) const {
  switch (segment) {
  case kMainSegment:
    return main_segment_size_;
  case kClientDataSegment:
    return client_data_segment_size_;
  default:
    return 0;
  }
}

u32 ConfigManager::GetPort() const { return port_; }

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
  sched_workers_ = 4;
  slow_threads_ = 4;
  process_reaper_workers_ = 1;

  main_segment_size_ = 1024 * 1024 * 1024;        // 1GB
  client_data_segment_size_ = 512 * 1024 * 1024;  // 512MB

  port_ = 5555;
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

    // Worker thread configuration (new location)
    if (runtime["sched_threads"]) {
      sched_workers_ = runtime["sched_threads"].as<u32>();
    }
    if (runtime["slow_threads"]) {
      slow_threads_ = runtime["slow_threads"].as<u32>();
    }
    if (runtime["process_reaper_threads"]) {
      process_reaper_workers_ = runtime["process_reaper_threads"].as<u32>();
    }

    // Local task scheduler
    if (runtime["local_sched"]) {
      local_sched_ = runtime["local_sched"].as<std::string>();
    }

    // Worker sleep configuration
    if (runtime["first_busy_wait"]) {
      first_busy_wait_ = runtime["first_busy_wait"].as<u32>();
    }
    if (runtime["max_sleep"]) {
      max_sleep_ = runtime["max_sleep"].as<u32>();
    }

    // Other runtime parameters (for future use or backward compatibility)
    if (runtime["stack_size"]) {
      // Stack size parameter exists but is not currently used by the runtime
    }
    if (runtime["queue_depth"]) {
      // Queue depth parameter exists but is not currently used by the runtime
    }
    if (runtime["heartbeat_interval"]) {
      // Heartbeat interval parameter exists but is not currently used by the runtime
    }
  }

  // Backward compatibility: support old 'workers' section if runtime section didn't set these
  if (yaml_conf["workers"]) {
    auto workers = yaml_conf["workers"];
    if (workers["sched_threads"] && !yaml_conf["runtime"]["sched_threads"]) {
      sched_workers_ = workers["sched_threads"].as<u32>();
    }
    if (workers["slow_threads"] && !yaml_conf["runtime"]["slow_threads"]) {
      slow_threads_ = workers["slow_threads"].as<u32>();
    }
    if (workers["process_reaper_threads"] && !yaml_conf["runtime"]["process_reaper_threads"]) {
      process_reaper_workers_ = workers["process_reaper_threads"].as<u32>();
    }
  }

  // Parse memory segments
  if (yaml_conf["memory"]) {
    auto memory = yaml_conf["memory"];
    if (memory["main_segment_size"]) {
      main_segment_size_ = hshm::ConfigParse::ParseSize(
          memory["main_segment_size"].as<std::string>());
    }
    if (memory["client_data_segment_size"]) {
      client_data_segment_size_ = hshm::ConfigParse::ParseSize(
          memory["client_data_segment_size"].as<std::string>());
    }
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

        // Add to compose config
        compose_config_.pools_.push_back(pool_config);
      }
    }
  }
}

} // namespace chi