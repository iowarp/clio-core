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

#ifndef WRPCTE_CORE_CONFIG_H_
#define WRPCTE_CORE_CONFIG_H_

#include <chimaera/chimaera.h>
#include <hermes_shm/util/config_parse.h>
#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

namespace wrp_cte::core {

/**
 * Performance configuration for CTE Core operations
 */
struct PerformanceConfig {
  chi::u32
      target_stat_interval_ms_;  // Interval for updating target stats (legacy)
  chi::u32 stat_targets_period_ms_;     // Period for periodic StatTargets calls
                                        // (default 50ms)
  chi::u32 max_concurrent_operations_;  // Max concurrent I/O operations
  float score_threshold_;               // Threshold for blob reorganization
  float score_difference_threshold_;    // Minimum score difference for
                                        // reorganization

  chi::u32 flush_metadata_period_ms_;  // Period for periodic metadata flush
                                       // (default 5s)
  std::string metadata_log_path_;   // Path for metadata log (empty = disabled)
  chi::u32 flush_data_period_ms_;   // Period for data flush (default 10s)
  int flush_data_min_persistence_;  // Min persistence level to flush to
                                    // (1=temp-nonvolatile)
  chi::u64
      transaction_log_capacity_bytes_;  // Total WAL capacity (default 32MB)

  PerformanceConfig()
      : target_stat_interval_ms_(5000),
        stat_targets_period_ms_(50),
        max_concurrent_operations_(64),
        score_threshold_(0.7f),
        score_difference_threshold_(0.05f),
        flush_metadata_period_ms_(5000),
        metadata_log_path_(""),
        flush_data_period_ms_(10000),
        flush_data_min_persistence_(1),
        transaction_log_capacity_bytes_(32ULL * 1024ULL * 1024ULL) {}
};

/**
 * Target management configuration
 */
struct TargetConfig {
  chi::u32 neighborhood_;  // Number of targets (nodes CTE can buffer to)
  chi::u32 default_target_timeout_ms_;  // Default timeout for target operations
  chi::u32 poll_period_ms_;  // Period to rescan targets for statistics

  TargetConfig()
      : neighborhood_(4),
        default_target_timeout_ms_(30000),
        poll_period_ms_(5000) {}
};

/**
 * Storage block device configuration entry
 */
struct StorageDeviceConfig {
  std::string path_;       // Directory path for the block device
  std::string bdev_type_;  // Block device type ("file", "ram", etc.)
  chi::u64
      capacity_limit_;  // Capacity limit in bytes (parsed from size string)
  float score_;  // Optional manual score (0.0-1.0), -1.0 means use automatic
                 // scoring
  std::string persistence_level_;  // "volatile", "temporary", "long_term"

  StorageDeviceConfig()
      : capacity_limit_(0), score_(-1.0f), persistence_level_("volatile") {}
  StorageDeviceConfig(const std::string &path, const std::string &bdev_type,
                      chi::u64 capacity, float score = -1.0f,
                      const std::string &persistence_level = "volatile")
      : path_(path),
        bdev_type_(bdev_type),
        capacity_limit_(capacity),
        score_(score),
        persistence_level_(persistence_level) {}
};

/**
 * Storage configuration section
 */
struct StorageConfig {
  std::vector<StorageDeviceConfig> devices_;  // List of storage devices

  StorageConfig() = default;
};

/**
 * Data Placement Engine configuration
 */
struct DpeConfig {
  std::string
      dpe_type_;  // DPE algorithm type ("random", "round_robin", "max_bw")

  DpeConfig() : dpe_type_("max_bw") {}
  explicit DpeConfig(const std::string &dpe_type) : dpe_type_(dpe_type) {}
};

/**
 * CTE Core Configuration Manager
 * Provides YAML parsing and validation for CTE Core configuration
 */
class Config {
 public:
  /**
   * Performance settings
   */
  PerformanceConfig performance_;

  /**
   * Target management settings
   */
  TargetConfig targets_;

  /**
   * Storage configuration
   */
  StorageConfig storage_;

  /**
   * Data Placement Engine configuration
   */
  DpeConfig dpe_;

  /**
   * Default constructor
   */
  Config() = default;

  /**
   * Constructor with allocator (for compatibility)
   */
  explicit Config(void *alloc) {
    (void)alloc;  // Suppress unused variable warning
  }

  /**
   * Load configuration from YAML file
   * @param config_file_path Path to YAML configuration file
   * @return true if successful, false otherwise
   */
  bool LoadFromFile(const std::string &config_file_path);

  /**
   * Load configuration from YAML string
   * @param yaml_string YAML configuration content as string
   * @return true if successful, false otherwise
   */
  bool LoadFromString(const std::string &yaml_string);

  /**
   * Load configuration from environment variables
   * Falls back to config file specified in environment variable
   * @return true if successful, false otherwise
   */
  bool LoadFromEnvironment();

  /**
   * Save configuration to YAML file
   * @param config_file_path Path to output YAML file
   * @return true if successful, false otherwise
   */
  bool SaveToFile(const std::string &config_file_path) const;

  /**
   * Validate configuration parameters
   * @return true if configuration is valid, false otherwise
   */
  bool Validate() const;

  /**
   * Get configuration parameter as string for debugging
   * @param param_name Parameter name
   * @return Parameter value as string, empty if not found
   */
  std::string GetParameterString(const std::string &param_name) const;

  /**
   * Set configuration parameter from string
   * @param param_name Parameter name
   * @param value Parameter value as string
   * @return true if successful, false if parameter not found or invalid
   */
  bool SetParameterFromString(const std::string &param_name,
                              const std::string &value);

 protected:
  /**
   * Parse YAML node and populate configuration
   * @param node YAML node to parse
   * @return true if successful, false otherwise
   */
  bool ParseYamlNode(const YAML::Node &node);

  /**
   * Generate YAML representation of configuration
   * @param emitter YAML emitter to populate with configuration data
   */
  void EmitYaml(YAML::Emitter &emitter) const;

 private:
  /**
   * Environment variable name for configuration file path
   */
  std::string config_env_var_ = "WRP_RUNTIME_CONF";

  /**
   * Parse performance configuration from YAML
   * @param node YAML node containing performance config
   * @return true if successful, false otherwise
   */
  bool ParsePerformanceConfig(const YAML::Node &node);

  /**
   * Parse target configuration from YAML
   * @param node YAML node containing target config
   * @return true if successful, false otherwise
   */
  bool ParseTargetConfig(const YAML::Node &node);

  /**
   * Parse storage configuration from YAML
   * @param node YAML node containing storage config
   * @return true if successful, false otherwise
   */
  bool ParseStorageConfig(const YAML::Node &node);

  /**
   * Parse DPE configuration from YAML
   * @param node YAML node containing DPE config
   * @return true if successful, false otherwise
   */
  bool ParseDpeConfig(const YAML::Node &node);

  /**
   * Parse size string to bytes (e.g., "1GB", "512MB", "2TB")
   * @param size_str Size string to parse
   * @param size_bytes Output size in bytes
   * @return true if successful, false otherwise
   */
  bool ParseSizeString(const std::string &size_str, chi::u64 &size_bytes) const;

  /**
   * Format size in bytes to human-readable string (e.g., "1GB", "512MB")
   * @param size_bytes Size in bytes
   * @return Formatted size string
   */
  std::string FormatSizeBytes(chi::u64 size_bytes) const;
};

}  // namespace wrp_cte::core

#endif  // WRPCTE_CORE_CONFIG_H_