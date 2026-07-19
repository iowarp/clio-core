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

#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/config_parse.h>
#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

namespace clio::cte::core {

/**
 * Performance configuration for CTE Core operations
 */
struct PerformanceConfig {
  clio::run::u32
      target_stat_interval_ms_;  // Interval for updating target stats (legacy)
  clio::run::u32 stat_targets_period_ms_;     // Period for periodic StatTargets calls
                                        // (default 50ms)
  clio::run::u32 max_concurrent_operations_;  // Max concurrent I/O operations
  float score_threshold_;               // Threshold for blob reorganization
  float score_difference_threshold_;    // Minimum score difference for
                                        // reorganization

  clio::run::u32 flush_metadata_period_ms_;  // Period for periodic metadata flush
                                       // (default 5s)
  std::string metadata_log_path_;   // Path for metadata log (empty = disabled)
  clio::run::u32 flush_data_period_ms_;   // Period for data flush (default 10s)
  int flush_data_min_persistence_;  // Min persistence level to flush to
                                    // (1=temp-nonvolatile)
  clio::run::u64
      transaction_log_capacity_bytes_;  // Total WAL capacity (default 32MB)

  PerformanceConfig()
      : target_stat_interval_ms_(5000),
        // 5000 ms — fires 12×/min so target stats stay fresh without
        // saturating the bdev pool's worker (50 ms default fired
        // 20×/s and starved real PutBlob/GetBlob traffic on a single-
        // worker setup).
        stat_targets_period_ms_(5000),
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
  clio::run::u32 neighborhood_;  // Number of targets (nodes CTE can buffer to)
  clio::run::u32 default_target_timeout_ms_;  // Default timeout for target operations
  clio::run::u32 poll_period_ms_;  // Period to rescan targets for statistics

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
  clio::run::u64
      capacity_limit_;  // Capacity limit in bytes (parsed from size string)
  float score_;  // Optional manual score (0.0-1.0), -1.0 means use automatic
                 // scoring
  std::string persistence_level_;  // "volatile", "temporary", "long_term"

  // When set (major_ != 0), this storage target binds to an ALREADY-EXISTING
  // pool (e.g. a safe-bdev pool composed elsewhere) instead of having CTE
  // create its own bdev. The pool must implement the bdev task interface
  // (AllocateBlocks/FreeBlocks/Write/Read/GetStats). When attaching, path_,
  // bdev_type_, and capacity_limit_ become optional; score_ still applies.
  clio::run::PoolId existing_pool_id_;
  // Optional module name of the existing pool, for logging/validation only
  // (e.g. "clio_safe_bdev"). Not used to route — routing is purely by pool id.
  std::string existing_pool_module_;

  StorageDeviceConfig()
      : capacity_limit_(0),
        score_(-1.0f),
        persistence_level_("volatile"),
        existing_pool_id_(clio::run::PoolId::GetNull()) {}
  StorageDeviceConfig(const std::string &path, const std::string &bdev_type,
                      clio::run::u64 capacity, float score = -1.0f,
                      const std::string &persistence_level = "volatile")
      : path_(path),
        bdev_type_(bdev_type),
        capacity_limit_(capacity),
        score_(score),
        persistence_level_(persistence_level),
        existing_pool_id_(clio::run::PoolId::GetNull()) {}

  // True when this target should bind to an existing pool rather than create
  // its own bdev. Pool-id major 0 is reserved for "not set" (CTE/admin pools
  // never use major 0 for storage targets here).
  bool HasExistingPool() const { return existing_pool_id_.major_ != 0; }
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
 * Data organizer configuration.
 *
 * The organizer is the CTE's built-in, periodically-driven reorganization
 * engine (issue #738). When enabled, Create() spawns `organizer_tasks_`
 * periodic DynamicReorganize tasks; each invokes the configured
 * DataOrganizer (e.g. "frecency"), which rescores blobs and moves them
 * between tiers through the server's ReorganizeBlob logic. "none" (the
 * default) disables periodic reorganization entirely.
 */
struct OrganizerConfig {
  std::string name_;         // Organizer name ("none", "frecency")
  clio::run::u32 organizer_tasks_;  // Periodic task replicas to spawn
                                    // (parallelizes the organization work)
  clio::run::u32 period_ms_;        // Period between organizer invocations

  OrganizerConfig() : name_("none"), organizer_tasks_(1), period_ms_(5000) {}
};

/**
 * GPU metadata cache configuration.
 *
 * When enabled, CTE Core allocates a chunk of GPU-accessible memory
 * (managed/shared USM) at Create time and projects a fixed-capacity
 * snapshot of tag and blob metadata into it. The cache is updated by
 * GPU kernels launched from the CTE Core server side as PutBlob /
 * GetOrCreateTag / DelBlob / DelTag run on the CPU; downstream GPU
 * kernels can read entries directly to discover what the CTE has
 * placed in DRAM-tier (GPU-reachable) storage.
 *
 * Sizing: capacity_bytes_ caps the memory footprint of the cache. The
 * actual map slot counts (max_blobs_ / max_tags_) are derived from
 * capacity_bytes_ / sizeof(GpuBlobEntry|GpuTagEntry) in the Create path.
 */
struct GpuMetadataCacheConfig {
  bool enabled_;             /**< Master switch — off by default. */
  clio::run::u64 capacity_bytes_;  /**< Total GPU memory footprint cap. */
  clio::run::u32 max_blobs_;       /**< Hash slot count for blob entries.  */
  clio::run::u32 max_tags_;        /**< Hash slot count for tag entries.   */

  GpuMetadataCacheConfig()
      : enabled_(false),
        capacity_bytes_(64ULL * 1024ULL * 1024ULL),  // 64 MB default
        max_blobs_(8192),
        max_tags_(1024) {}
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
   * Data organizer configuration (periodic reorganization; off by default)
   */
  OrganizerConfig organizer_;

  /**
   * GPU metadata cache (optional, off by default).
   */
  GpuMetadataCacheConfig gpu_metadata_cache_;

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
  std::string config_env_var_ = "CLIO_SERVER_CONF";

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
  bool ParseSizeString(const std::string &size_str, clio::run::u64 &size_bytes) const;

  /**
   * Format size in bytes to human-readable string (e.g., "1GB", "512MB")
   * @param size_bytes Size in bytes
   * @return Formatted size string
   */
  std::string FormatSizeBytes(clio::run::u64 size_bytes) const;
};

}  // namespace clio::cte::core

#endif  // WRPCTE_CORE_CONFIG_H_