#ifndef WRPCTE_CORE_CONFIG_H_
#define WRPCTE_CORE_CONFIG_H_

#include <chimaera/chimaera.h>
#include <yaml-cpp/yaml.h>
#include <hermes_shm/util/config_parse.h>
#include <string>
#include <vector>

namespace wrp_cte::core {

/**
 * Performance configuration for CTE Core operations
 */
struct PerformanceConfig {
  chi::u32 target_stat_interval_ms_;    // Interval for updating target stats
  chi::u32 max_concurrent_operations_;  // Max concurrent I/O operations
  float score_threshold_;               // Threshold for blob reorganization
  float score_difference_threshold_;    // Minimum score difference for reorganization

  PerformanceConfig()
      : target_stat_interval_ms_(5000),
        max_concurrent_operations_(64),
        score_threshold_(0.7f),
        score_difference_threshold_(0.05f) {}
};

/**
 * Target management configuration
 */
struct TargetConfig {
  chi::u32 neighborhood_;               // Number of targets (nodes CTE can buffer to)
  chi::u32 default_target_timeout_ms_;  // Default timeout for target operations
  chi::u32 poll_period_ms_;             // Period to rescan targets for statistics

  TargetConfig()
      : neighborhood_(4),
        default_target_timeout_ms_(30000),
        poll_period_ms_(5000) {}
};

/**
 * Storage block device configuration entry
 */
struct StorageDeviceConfig {
  std::string path_;          // Directory path for the block device
  std::string bdev_type_;     // Block device type ("file", "ram", etc.)
  chi::u64 capacity_limit_;   // Capacity limit in bytes (parsed from size string)
  float score_;               // Optional manual score (0.0-1.0), -1.0 means use automatic scoring
  
  StorageDeviceConfig() : capacity_limit_(0), score_(-1.0f) {}
  StorageDeviceConfig(const std::string& path, const std::string& bdev_type, chi::u64 capacity, float score = -1.0f)
      : path_(path), bdev_type_(bdev_type), capacity_limit_(capacity), score_(score) {}
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
  std::string dpe_type_;  // DPE algorithm type ("random", "round_robin", "max_bw")

  DpeConfig() : dpe_type_("max_bw") {}
  explicit DpeConfig(const std::string& dpe_type) : dpe_type_(dpe_type) {}
};

/**
 * Compression and DNN Monitoring configuration
 */
struct CompressionConfig {
  chi::u32 monitor_interval_ms_;         // Interval for collecting target capacities and stats (default 5ms)
  std::string dnn_model_weights_path_;   // Path to DNN model weights JSON file (deprecated, use qtable)
  std::string qtable_model_path_;        // Path to Q-table model directory (contains qtable.json, binning_params.json)
  chi::u32 dnn_samples_before_reinforce_; // Number of samples to collect before reinforcing DNN
  std::string trace_folder_path_;        // Path to folder for CTE trace logs
  float qtable_learning_rate_;           // Learning rate for Q-table online updates (default 0.3)

  CompressionConfig()
      : monitor_interval_ms_(5),
        dnn_model_weights_path_(""),
        qtable_model_path_(""),
        dnn_samples_before_reinforce_(1000),
        trace_folder_path_(""),
        qtable_learning_rate_(0.3f) {}
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
   * Compression and DNN Monitoring configuration
   */
  CompressionConfig compression_;

  /**
   * Default constructor
   */
  Config() = default;

  /**
   * Constructor with allocator (for compatibility)
   */
  explicit Config(void *alloc) {
    (void)alloc; // Suppress unused variable warning
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
   * Parse compression configuration from YAML
   * @param node YAML node containing compression config
   * @return true if successful, false otherwise
   */
  bool ParseCompressionConfig(const YAML::Node &node);

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