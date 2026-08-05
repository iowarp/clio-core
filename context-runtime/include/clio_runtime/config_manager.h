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

#ifndef CLIO_RUNTIME_INCLUDE_MANAGERS_CONFIG_MANAGER_H_
#define CLIO_RUNTIME_INCLUDE_MANAGERS_CONFIG_MANAGER_H_

#include "clio_ctp/introspect/system_info.h"
#include <string>
#include <unordered_map>
#include <vector>

#include "clio_runtime/types.h"
#include "clio_runtime/pool_query.h"

namespace clio::run {

/**
 * Configuration for a single pool in the compose section
 */
struct PoolConfig {
  std::string mod_name_;     /**< Module name (e.g., "clio_bdev") */
  std::string pool_name_;    /**< Pool name or identifier */
  PoolId pool_id_;           /**< Pool ID for this module */
  PoolQuery pool_query_;     /**< Pool query routing (Dynamic or Local) */
  std::string config_;       /**< Remaining YAML configuration as string */
  bool restart_ = false;     /**< If true, store compose file for crash-restart */
  /** Default RPC visibility for the container: 0 = public (default),
   *  1 = private. A private container rejects RPCs from external user clients
   *  (runtime-internal callers are always allowed). */
  u32 container_visibility_ = 0;
  /** Per-RPC visibility overrides, keyed by RPC method NAME -> 0 public /
   *  1 private. Methods absent here inherit container_visibility_. */
  std::unordered_map<std::string, u32> rpc_acl_;

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
    ar(mod_name_, pool_name_, pool_id_, pool_query_, config_, restart_,
       container_visibility_, rpc_acl_);
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
 * Inherits from ctp BaseConfig and manages YAML configuration parsing.
 * Config lookup, first hit wins:
 *   1. CLIO_SERVER_CONF env (or legacy CLIO_SERVER_CONF via env_compat)
 *   2. ~/.clio/clio.yaml
 *   3. ~/.clio/clio.yaml
 *   4. ~/.clio/clio.yaml
 *   5. ~/.clio/clio.yaml
 *   6. Bare-minimum defaults (no compose)
 * Uses CTP global cross pointer variable singleton pattern.
 */
class ConfigManager : public ctp::BaseConfig {
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
   * Get server configuration file path.
   * Lookup order (first hit wins):
   *   1. CLIO_SERVER_CONF env var (or legacy CLIO_SERVER_CONF via env_compat)
   *   2. ~/.clio/clio.yaml         (new canonical user config)
   *   3. ~/.clio/clio.yaml     (legacy filename in new dir)
   *   4. ~/.clio/clio.yaml     (new filename in legacy dir)
   *   5. ~/.clio/clio.yaml (legacy)
   *   6. Empty string (bare-minimum defaults, no compose)
   * Both per-user directories are seeded with identical content by
   * `make install` and by the iowarp_core pip wheel's _setup() hook.
   * @return Configuration file path or empty string if no config found
   */
  std::string GetServerConfigPath() const;

  /**
   * Get number of worker threads
   * @return Number of worker threads for task execution
   */
  u32 GetNumThreads() const { return num_threads_; }

  /** issue #807: number of parallel inbound SHM rings + drain threads. */
  u32 GetShmInShards() const { return shm_in_shards_; }

  /** issue #807: if true, SHM responses are handed to a background sender thread
   *  (nonblocking, never stalls a worker on a full client ring). Default false:
   *  measured ~3x latency regression on latency-bound workloads because the
   *  thread handoff is pure overhead when the client ring is not full, which is
   *  the common case. Enable for overload/back-pressure-heavy deployments. */
  bool GetShmAsyncSend() const { return shm_async_send_; }

  /** issue #807/#784: microseconds a SHM client waiter spins polling for its
   *  response before parking on its EventManager. On SHM the round-trip is
   *  microseconds, so a brief spin catches the completion without a park/wake +
   *  signal syscall — the dominant cost at low latency. Larger values trade
   *  throughput for latency (#784: 1ms spin = 2.2x latency, -45% throughput), so
   *  keep it small. Default 50us. */
  u32 GetShmClientSpinUs() const { return shm_client_spin_us_; }

  /**
   * issue #785: extra task lanes reserved for elastic replacement workers.
   * Lanes are indexed by worker id, so replacements spawned by a stall rescue
   * need lanes allocated up front or they can never be routed to. Matches the
   * spawn budget in WorkOrchestrator (one replacement per core).
   */
  u32 GetElasticLaneHeadroom() const {
    // Bounded deliberately. Each reserved lane costs queue_depth x priorities
    // slots in the queue segment, so scaling this with core count would add
    // ~128 lanes of segment on a large machine purely to hold replacements
    // that may never be spawned. Rescues are rare and replacements are
    // recycled, so a modest fixed ceiling is enough: the pool only ever needs
    // as many replacements as there are workers wedged AT ONCE.
    static constexpr u32 kMaxElasticLanes = 8;
    int ncpu = ctp::SystemInfo::GetCpuCount();
    u32 want = (ncpu > 0) ? static_cast<u32>(ncpu) : 8;
    return want < kMaxElasticLanes ? want : kMaxElasticLanes;
  }

  /**
   * Get task queue depth per worker
   * @return Queue depth (number of tasks per worker queue)
   */
  u32 GetQueueDepth() const { return queue_depth_; }

  /**
   * Size of the main task-data segment (issue #727).
   * @return The explicit size when main_segment_size_ > 0 (yaml
   *         `runtime: main_segment_size` or CLIO_MAIN_SEGMENT_SIZE), else the
   *         auto default: the machine's RAM capacity (1 GiB if introspection
   *         fails). The reservation is sparse; creation-time clamps in
   *         ipc_manager.cc bound what the segment actually gets. Never 0.
   */
  size_t CalculateMainSegmentSize() const;

  /**
   * Calculate queue segment size needed for TaskQueue and NetQueue ring buffers
   * @return Calculated size in bytes
   */
  size_t CalculateQueueSegmentSize() const;

  /**
   * Calculate the runtime-wide metadata segment size (issue #783).
   *
   * Backs the CTE shared-memory metadata cache. The reservation is large and
   * sparse: it is never pre-faulted, so untouched pages cost nothing. Sizing is
   * therefore about the address-space reservation, not RAM.
   *
   * @return Calculated size in bytes, or the explicit size if one is configured
   */
  size_t CalculateMetadataSegmentSize() const;

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
   * @return true if this runtime is ephemeral (started with --ephemeral /
   * CLIO_EPHEMERAL): it skips the default compose section and starts bare
   * (admin only), to be composed explicitly.
   */
  bool IsEphemeral() const;

  /**
   * Get server address for client connections
   * @return Server address (default: "127.0.0.1", overridden by CLIO_SERVER_ADDR)
   */
  std::string GetServerAddr() const;

  /**
   * Get neighborhood size for range query splitting
   * @return Maximum number of queries when splitting range queries
   */
  u32 GetNeighborhoodSize() const;

  /**
   * Get shared memory segment names. The name is suffixed with the runtime port
   * so that multiple runtimes sharing one node + ${USER} (the fallback-runtime
   * topology) each own a distinct segment instead of colliding. Pass an explicit
   * non-zero @p port to compute another runtime's segment name (used by the
   * fallback client to attach the main runtime's segments).
   * @param segment Memory segment identifier
   * @param port Port to key the name on; 0 = this runtime's configured port
   * @return Expanded segment name with environment variables resolved
   */
  std::string GetSharedMemorySegmentName(MemorySegment segment,
                                         u32 port = 0) const;

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
   * Get the task-progress validity-check interval in milliseconds.
   *
   * How long an outstanding cross-node task may sit quietly before the origin
   * runs a QueryTaskProgress liveness check on it (issue #628), then re-checks
   * each interval. This is the anti-hang mechanism that works even when a
   * task's ttl is infinite. 0 disables the periodic check.
   * Overridable via env CLIO_TASK_PROGRESS_INTERVAL_MS.
   * @return Interval in ms (default: 0 = disabled)
   */
  u32 GetTaskProgressIntervalMs() const { return task_progress_interval_ms_; }

  /**
   * Get maximum sleep duration in microseconds
   * @return Maximum sleep duration cap (default: 50000us = 50ms)
   */
  u32 GetMaxSleep() const { return max_sleep_; }

  /**
   * Get SGD learning rate for task load prediction model
   * @return Learning rate (default: 0.2)
   */
  float GetLearningRate() const { return learning_rate_; }

  /**
   * Get number of GPU blocks for GPU orchestrator
   * @return Number of blocks (default: 32)
   */
  u32 GetGpuBlocks() const { return gpu_blocks_; }

  /**
   * Get number of threads per block for GPU orchestrator
   * @return Threads per block (default: 32)
   */
  u32 GetGpuThreadsPerBlock() const { return gpu_threads_per_block_; }

  /**
   * Get GPU queue depth for GPU orchestrator task queues
   * @return Queue depth (default: 16)
   */
  u32 GetGpuQueueDepth() const { return gpu_queue_depth_; }

  /**
   * Whether SWIM membership detection is enabled.
   * When false: HeartbeatProbe periodic is a no-op (no direct/indirect
   *   probes, no suspicion timeouts, no SetDead, no recovery). Suitable
   *   for bring-up + perf debugging where we want to rule out SWIM as
   *   the cause of mid-run failures.
   * When true (default): SWIM runs with the timeouts below.
   */
  bool GetSwimEnabled() const { return swim_enabled_; }

  /** Direct-probe timeout (seconds). Used by HeartbeatProbe. */
  float GetSwimDirectProbeTimeoutSec() const {
    return swim_direct_probe_timeout_sec_;
  }

  /** Indirect-probe timeout (seconds). Used by HeartbeatProbe. */
  float GetSwimIndirectProbeTimeoutSec() const {
    return swim_indirect_probe_timeout_sec_;
  }

  /**
   * Suspicion timeout (seconds): how long a node stays in kSuspected
   * before being promoted to kDead.
   */
  float GetSwimSuspicionTimeoutSec() const {
    return swim_suspicion_timeout_sec_;
  }

 private:
  /**
   * Set default configuration values (implements ctp::BaseConfig)
   */
  void LoadDefault() override;

  /**
   * Parse YAML configuration (implements ctp::BaseConfig)
   */
  void ParseYAML(YAML::Node& yaml_conf) override;

 public:
  /**
   * Apply environment-variable overrides (CLIO_PORT, CLIO_SERVER_ADDR, etc.).
   * Must run after every (re)load: LoadFromFile resets to defaults via
   * LoadDefault, so a config reload (e.g. parsing a compose file into this
   * manager) would otherwise silently drop the env-configured port — which on
   * the force-net path retargets every forward at the wrong port. Public so
   * external reloaders (and tests) can restore env semantics after a bare
   * LoadYaml.
   */
  void ApplyEnvOverrides();

 private:
  bool is_initialized_ = false;
  std::string config_file_path_;

  // Configuration parameters
  u32 num_threads_ = 4;
  // issue #807: parallel inbound SHM rings, ENABLED by default. 0 = auto (=
  // worker count, so every worker drains its own shard). Since the drain is
  // done by the EXISTING workers in their poll loop (not dedicated threads),
  // more shards spreads ingest across the pool with no extra threads — measured
  // worker-driven S=1 1.48x and S=4 1.68x over dev. Capped at the worker count
  // in ServerInitShm (worker w owns shard w). CLIO_SHM_IN_SHARDS overrides;
  // set 1 to restore a single ingest ring.
  u32 shm_in_shards_ = 0;
  bool shm_async_send_ = false;  // issue #807: deferred SHM response send (opt-in)
  u32 shm_client_spin_us_ = 50;  // issue #807/#784: waiter spin before park
  u32 queue_depth_ = 1024;

  // issue #727: 0 = auto (the machine's RAM capacity — a sparse reservation;
  // see CalculateMainSegmentSize() and the creation-time clamps in
  // ipc_manager.cc). Settable via yaml `runtime: main_segment_size` or
  // CLIO_MAIN_SEGMENT_SIZE; LoadDefault() also installs 0, this initializer
  // just matches it.
  size_t main_segment_size_ = 0;
  size_t client_data_segment_size_ = ctp::Unit<size_t>::Megabytes(256);
  // issue #783: metadata segment. Deliberately huge and never pre-faulted --
  // shm_open + ftruncate + mmap is lazily populated, so the reservation
  // (default: the machine's RAM capacity, so metadata scales with the host)
  // costs only the pages actually touched. Creation-time safety clamps live in
  // ipc_manager.cc (half the cgroup-aware budget on Linux; 1 GB file cap
  // elsewhere). NOTE: on a host whose /dev/shm is smaller than the live set,
  // touching past the tmpfs limit raises SIGBUS, not ENOMEM. See
  // CalculateMetadataSegmentSize().
  // 0 means "auto-calculate" — CalculateMetadataSegmentSize() supplies the
  // default. Initialising this to a non-zero size would make the explicit
  // override branch there permanently true and the documented sentinel a lie;
  // it only worked because LoadDefault() resets it to 0. Set via the
  // `metadata_segment_size` key under `runtime` in the config file.
  size_t metadata_segment_size_ = 0;

  u32 port_ = 9413;
  // If true (CLIO_EPHEMERAL / --ephemeral), skip the default compose at startup.
  bool ephemeral_ = false;
  std::string server_addr_ = "127.0.0.1";
  u32 neighborhood_size_ = 32;

  // Shared memory segment names with environment variable support
  std::string main_segment_name_ = "chi_main_segment_${USER}";
  std::string client_data_segment_name_ = "chi_client_data_segment_${USER}";
  std::string queue_segment_name_ = "chi_queue_segment_${USER}";
  std::string metadata_segment_name_ = "chi_metadata_segment_${USER}";

  // Networking configuration
  std::string hostfile_path_ = "";

  // Local task scheduler
  std::string local_sched_ = "default";

  // Network retry configuration for system boot
  u32 wait_for_restart_timeout_ = 30;        // Default: 30 seconds
  u32 wait_for_restart_poll_period_ = 1;     // Default: 1 second

  // Worker sleep configuration (in microseconds)
  u32 first_busy_wait_ = 10000;              // Default: 10000us (10ms) busy wait
  // #628 task-progress probe (0 = disabled). ON by default (issue #774): this
  // probe is the ONLY recovery for a cross-node task whose response was lost
  // (e.g. the SendOut retry queue timing out under sustained back-pressure) —
  // with it disabled, the origin task leaks in send_map_ and the client's
  // Future::Wait() hangs forever. 5s keeps the scan cheap (throttled, probes
  // only replicas outstanding beyond the interval) while bounding how long a
  // lost response can park a client.
  u32 task_progress_interval_ms_ = 5000;
  u32 max_sleep_ = 50000;                    // Default: 50000us (50ms) maximum sleep

  // Task load prediction model
  float learning_rate_ = 0.2f;               // Default: 0.2 SGD learning rate

  // GPU orchestrator configuration
  u32 gpu_blocks_ = 1;                       // Default: 1 block
  u32 gpu_threads_per_block_ = 32;           // Default: 32 threads per block
  u32 gpu_queue_depth_ = 16;                 // Default: 16 tasks per queue

  // SWIM membership-detection configuration.
  // Defaults match the prior hard-coded constants in admin_runtime.cc so
  // existing deployments behave identically when these fields are absent
  // from the YAML.
  bool swim_enabled_ = true;
  float swim_direct_probe_timeout_sec_ = 30.0f;
  float swim_indirect_probe_timeout_sec_ = 15.0f;
  float swim_suspicion_timeout_sec_ = 60.0f;

  // Compose configuration
  ComposeConfig compose_config_;

  // Configuration directory for persistent runtime config
  std::string conf_dir_ = "/tmp/clio";
};

}  // namespace clio::run

// Global pointer variable declaration for Configuration manager singleton
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_H(clio::run::ConfigManager, g_config_manager);

// Macro for accessing the Configuration manager singleton using global pointer variable
#define CLIO_CONFIG_MANAGER CTP_GET_GLOBAL_PTR_VAR(::clio::run::ConfigManager, g_config_manager)
// Backward-compat alias (clio_run rebrand). External code that still
// uses the legacy CHI_* spelling keeps working unchanged.

#endif  // CLIO_RUNTIME_INCLUDE_MANAGERS_CONFIG_MANAGER_H_