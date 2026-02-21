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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_CONFIG_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_CONFIG_MANAGER_H_

#include <string>
#include <vector>

#include "chimaera/types.h"
#include "chimaera/pool_query.h"

namespace chi {

/**
 * Configuration for a single pool in the compose section
 */
struct PoolConfig {
  std::string mod_name_;     /**< Module name (e.g., "chimaera_bdev") */
  std::string pool_name_;    /**< Pool name or identifier */
  PoolId pool_id_;           /**< Pool ID for this module */
  PoolQuery pool_query_;     /**< Pool query routing (Dynamic or Local) */
  std::string config_;       /**< Remaining YAML configuration as string */
  bool restart_ = false;     /**< If true, store compose file for crash-restart */

  PoolConfig() = default;

  /**
   * Constructor with allocator (for compatibility with CreateParams pattern)
   * The allocator is not used since PoolConfig uses std::string
   */
  template <typename AllocT>
  explicit PoolConfig(const AllocT& alloc) {
    (void)alloc;  // Suppress unused parameter warning
  }

  /**
   * Cereal serialization support
   * @param ar Archive for serialization
   */
  template <class Archive>
  void serialize(Archive& ar) {
    ar(mod_name_, pool_name_, pool_id_, pool_query_, config_, restart_);
  }
};

/**
 * Configuration for the compose section containing multiple pool configs
 */
struct ComposeConfig {
  std::vector<PoolConfig> pools_; /**< List of pool configurations */

  ComposeConfig() = default;
};

/**
 * Configuration manager singleton
 *
 * Inherits from hshm BaseConfig and manages YAML configuration parsing.
 * Config lookup: CHI_SERVER_CONF env -> WRP_RUNTIME_CONF env ->
 * ~/.chimaera/chimaera.yaml -> bare minimum defaults.
 * Uses HSHM global cross pointer variable singleton pattern.
 */
class ConfigManager : public hshm::BaseConfig {
 public:
  /**
   * Initialize configuration manager (generic wrapper)
   * Loads configuration from environment variables and files
   * @return true if initialization successful, false otherwise
   */
  bool Init() { return ClientInit(); }

  /**
   * Initialize configuration manager (client mode)
   * Loads configuration from environment variables and files
   * @return true if initialization successful, false otherwise
   */
  bool ClientInit();

  /**
   * Initialize configuration manager (server/runtime mode)
   * Same as ClientInit since config is needed by both
   * @return true if initialization successful, false otherwise
   */
  bool ServerInit();

  /**
   * Load configuration from YAML file
   * @param config_path Path to YAML configuration file
   * @return true if loading successful, false otherwise
   */
  bool LoadYaml(const std::string& config_path);

  /**
   * Get server configuration file path
   * Lookup order:
   *   1. CHI_SERVER_CONF env var
   *   2. WRP_RUNTIME_CONF env var
   *   3. ~/.chimaera/chimaera.yaml (if it exists)
   *   4. Empty string (bare minimum defaults, no compose)
   * @return Configuration file path or empty string if no config found
   */
  std::string GetServerConfigPath() const;

  /**
   * Get number of worker threads
   * @return Number of worker threads for task execution
   */
  u32 GetNumThreads() const { return num_threads_; }

  /**
   * Get task queue depth per worker
   * @return Queue depth (number of tasks per worker queue)
   */
  u32 GetQueueDepth() const { return queue_depth_; }

  /**
   * Calculate main segment size based on queue_depth and num_threads
   * @return Calculated size in bytes, or explicit size if main_segment_size_ > 0
   */
  size_t CalculateMainSegmentSize() const;

  /**
   * Get memory segment size
   * @param segment Memory segment identifier
   * @return Size in bytes
   */
  size_t GetMemorySegmentSize(MemorySegment segment) const;

  /**
   * Get networking port
   * @return Port number for networking
   */
  u32 GetPort() const;

  /**
   * Get server address for client connections
   * @return Server address (default: "127.0.0.1", overridden by CHI_SERVER_ADDR)
   */
  std::string GetServerAddr() const;

  /**
   * Get neighborhood size for range query splitting
   * @return Maximum number of queries when splitting range queries
   */
  u32 GetNeighborhoodSize() const;

  /**
   * Get shared memory segment names
   * @param segment Memory segment identifier`
   * @return Expanded segment name with environment variables resolved
   */
  std::string GetSharedMemorySegmentName(MemorySegment segment) const;

  /**
   * Get hostfile path for distributed scheduling
   * @return Path to hostfile with list of available nodes
   */
  std::string GetHostfilePath() const;

  /**
   * Check if configuration is valid
   * @return true if configuration is valid and loaded
   */
  bool IsValid() const;

  /**
   * Get local task scheduler name
   * @return Scheduler name (default: "default")
   */
  std::string GetLocalSched() const { return local_sched_; }

  /**
   * Get compose configuration
   * @return Compose configuration with all pool definitions
   */
  const ComposeConfig& GetComposeConfig() const { return compose_config_; }

  /**
   * Get configuration directory for persistent runtime config
   * @return Directory path for storing persistent runtime configuration
   */
  std::string GetConfDir() const { return conf_dir_; }

  /**
   * Get wait_for_restart timeout in seconds
   * @return Maximum time to wait for remote connection during system boot (default: 30 seconds)
   */
  u32 GetWaitForRestartTimeout() const { return wait_for_restart_timeout_; }

  /**
   * Get wait_for_restart polling period in seconds
   * @return Time between connection retry attempts (default: 1 second)
   */
  u32 GetWaitForRestartPollPeriod() const { return wait_for_restart_poll_period_; }

  /**
   * Get first busy wait duration in microseconds
   * @return Duration to busy wait before sleeping when there is no work (default: 10000us = 10ms)
   */
  u32 GetFirstBusyWait() const { return first_busy_wait_; }

  /**
   * Get maximum sleep duration in microseconds
   * @return Maximum sleep duration cap (default: 50000us = 50ms)
   */
  u32 GetMaxSleep() const { return max_sleep_; }

 private:
  /**
   * Set default configuration values (implements hshm::BaseConfig)
   */
  void LoadDefault() override;

  /**
   * Parse YAML configuration (implements hshm::BaseConfig)
   */
  void ParseYAML(YAML::Node& yaml_conf) override;

  bool is_initialized_ = false;
  std::string config_file_path_;

  // Configuration parameters
  u32 num_threads_ = 4;
  u32 queue_depth_ = 1024;

  size_t main_segment_size_ = hshm::Unit<size_t>::Gigabytes(1);
  size_t client_data_segment_size_ = hshm::Unit<size_t>::Megabytes(256);

  u32 port_ = 9413;
  std::string server_addr_ = "127.0.0.1";
  u32 neighborhood_size_ = 32;

  // Shared memory segment names with environment variable support
  std::string main_segment_name_ = "chi_main_segment_${USER}";
  std::string client_data_segment_name_ = "chi_client_data_segment_${USER}";

  // Networking configuration
  std::string hostfile_path_ = "";

  // Local task scheduler
  std::string local_sched_ = "default";

  // Network retry configuration for system boot
  u32 wait_for_restart_timeout_ = 30;        // Default: 30 seconds
  u32 wait_for_restart_poll_period_ = 1;     // Default: 1 second

  // Worker sleep configuration (in microseconds)
  u32 first_busy_wait_ = 10000;              // Default: 10000us (10ms) busy wait
  u32 max_sleep_ = 50000;                    // Default: 50000us (50ms) maximum sleep

  // Compose configuration
  ComposeConfig compose_config_;

  // Configuration directory for persistent runtime config
  std::string conf_dir_ = "/tmp/chimaera";
};

}  // namespace chi

// Global pointer variable declaration for Configuration manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_H(chi::ConfigManager, g_config_manager);

// Macro for accessing the Configuration manager singleton using global pointer variable
#define CHI_CONFIG_MANAGER HSHM_GET_GLOBAL_PTR_VAR(::chi::ConfigManager, g_config_manager)

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_CONFIG_MANAGER_H_