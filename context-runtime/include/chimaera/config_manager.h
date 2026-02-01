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
    ar(mod_name_, pool_name_, pool_id_, pool_query_, config_);
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
 * Reads configuration from CHI_SERVER_CONF or WRP_RUNTIME_CONF environment variables.
 * CHI_SERVER_CONF is checked first; WRP_RUNTIME_CONF is used as fallback.
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
   * Get server configuration file path from environment
   * Checks CHI_SERVER_CONF first, then falls back to WRP_RUNTIME_CONF
   * @return Configuration file path or empty string if neither is set
   */
  std::string GetServerConfigPath() const;

  /**
   * Get number of worker threads for given type
   * @param thread_type Type of worker thread
   * @return Number of threads to spawn
   */
  u32 GetWorkerThreadCount(ThreadType thread_type) const;

  /**
   * Get number of scheduler worker threads
   * @return Number of scheduler worker threads
   */
  u32 GetSchedulerWorkerCount() const { return sched_workers_; }

  /**
   * Get number of slow worker threads
   * @return Number of slow worker threads
   */
  u32 GetSlowWorkerCount() const { return slow_threads_; }

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
  u32 sched_workers_ = 4;
  u32 slow_threads_ = 4;
  u32 process_reaper_workers_ = 1;

  size_t main_segment_size_ = hshm::Unit<size_t>::Gigabytes(1);
  size_t client_data_segment_size_ = hshm::Unit<size_t>::Megabytes(256);

  u32 port_ = 5555;
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
};

}  // namespace chi

// Global pointer variable declaration for Configuration manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_H(chi::ConfigManager, g_config_manager);

// Macro for accessing the Configuration manager singleton using global pointer variable
#define CHI_CONFIG_MANAGER HSHM_GET_GLOBAL_PTR_VAR(::chi::ConfigManager, g_config_manager)

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_CONFIG_MANAGER_H_