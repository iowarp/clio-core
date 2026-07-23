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

#include "clio_runtime/config_manager.h"
#include "clio_runtime/task.h"
#include "clio_runtime/ipc_manager.h"
#include <cctype>
#include <cstdlib>
#include <filesystem>

// Global pointer variable definition for Configuration manager singleton
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_CC(clio::run::ConfigManager, g_config_manager);

namespace clio::run {

namespace {

/**
 * Parse a shared-memory segment size from YAML.
 *
 * Accepts either a bare byte count (`metadata_segment_size: 1073741824`) or a
 * size string with a unit suffix (`metadata_segment_size: "1g"`), matching how
 * the CTE spells storage capacities. A value of 0 means "use the built-in
 * default" — the same sentinel LoadDefault() installs.
 *
 * Returns false and leaves `out` untouched when the value cannot be parsed, so
 * the caller keeps its default. Deliberately does NOT delegate an unknown
 * suffix to ctp::ConfigParse::ParseSize, which calls exit(1) on one: a typo in
 * a config file should not take the runtime down, and the recognised suffixes
 * are checked here first.
 */
bool ParseSegmentSizeNode(const YAML::Node &node, const char *key,
                          size_t &out) {
  std::string text;
  try {
    text = node.as<std::string>();
  } catch (const std::exception &) {
    HLOG(kError, "Config: {} is not a scalar value; ignoring", key);
    return false;
  }
  if (text.empty()) {
    HLOG(kError, "Config: {} is empty; ignoring", key);
    return false;
  }

  // Locate the unit suffix (first character that is not part of the number).
  size_t i = 0;
  if (text[i] == '+' || text[i] == '-') ++i;
  size_t digits_begin = i;
  while (i < text.size() && ((text[i] >= '0' && text[i] <= '9') ||
                             text[i] == '.')) {
    ++i;
  }
  if (i == digits_begin) {
    HLOG(kError, "Config: {} = '{}' has no numeric part; ignoring", key, text);
    return false;
  }
  if (text[0] == '-') {
    HLOG(kError, "Config: {} = '{}' is negative; ignoring", key, text);
    return false;
  }

  std::string suffix;
  for (size_t j = i; j < text.size(); ++j) {
    if (!std::isspace(static_cast<unsigned char>(text[j]))) {
      suffix += static_cast<char>(std::tolower(
          static_cast<unsigned char>(text[j])));
    }
  }
  // Bare number, or a unit ParseSize understands (it keys off the first
  // character, so "gb"/"gigabytes" resolve the same as "g").
  const bool suffix_ok =
      suffix.empty() || suffix[0] == 'b' || suffix[0] == 'k' ||
      suffix[0] == 'm' || suffix[0] == 'g' || suffix[0] == 't' ||
      suffix[0] == 'p';
  if (!suffix_ok) {
    HLOG(kError, "Config: {} = '{}' has an unrecognised unit '{}'; ignoring "
         "(expected one of b, k, m, g, t, p)", key, text, suffix);
    return false;
  }

  // "b"/"bytes" is a bare byte count; ParseSize treats an unrecognised
  // leading 'b' as a fatal unit, so strip it and let the empty-suffix path
  // handle it.
  const std::string for_parse =
      (!suffix.empty() && suffix[0] == 'b') ? text.substr(0, i) : text;
  out = static_cast<size_t>(ctp::ConfigParse::ParseSize(for_parse));
  return true;
}

}  // namespace

// Constructor and destructor removed - handled by CTP singleton pattern

bool ConfigManager::ClientInit() {
  if (is_initialized_) {
    return true;
  }

  // Get configuration file path from environment
  config_file_path_ = GetServerConfigPath();
  HLOG(kInfo, "Config at: {}", config_file_path_);

  // Load YAML configuration if path is provided. LoadYaml re-applies the env
  // overrides itself; do them explicitly here too so a missing/empty config
  // path still honors CLIO_PORT et al.
  if (!config_file_path_.empty()) {
    if (!LoadYaml(config_file_path_)) {
      HLOG(kError,
            "Warning: Failed to load configuration from {}, using defaults",
            config_file_path_);
    }
  }
  ApplyEnvOverrides();

  is_initialized_ = true;
  return true;
}

void ConfigManager::ApplyEnvOverrides() {
  // Check CLIO_PORT env var (overrides YAML and default).
  if (const char *env = clio::run::env::GetCompat("PORT")) {
    std::string port_env(env);
    if (!port_env.empty()) {
      port_ = std::stoul(port_env);
    }
  }

  // Check CLIO_EPHEMERAL env var (set by `clio_run start --ephemeral`). An
  // ephemeral runtime skips the default compose section — it starts bare (just
  // admin) and is composed explicitly.
  if (const char *env = clio::run::env::GetCompat("EPHEMERAL")) {
    std::string e(env);
    ephemeral_ = !e.empty() && e != "0";
  }

  // Check CLIO_SERVER_ADDR env var (overrides default 127.0.0.1).
  if (const char *env = clio::run::env::GetCompat("SERVER_ADDR")) {
    std::string addr_env(env);
    if (!addr_env.empty()) {
      server_addr_ = addr_env;
    }
  }

  // CLIO_NUM_THREADS overrides the configured worker-thread count (last word,
  // after any config file). Useful for forcing a single worker, e.g. to test
  // whether a failure depends on cross-thread task migration.
  if (const char *env = clio::run::env::GetCompat("NUM_THREADS")) {
    char *end = nullptr;
    unsigned long n = std::strtoul(env, &end, 10);
    if (end != env && n >= 1) {
      num_threads_ = static_cast<u32>(n);
    }
  }

  // issue #807: number of parallel inbound SHM rings (each with its own drain
  // thread). CLIO_SHM_IN_SHARDS overrides. Default 4 spreads the MPSC tail
  // contention and the deserialize+route work across cores without oversubscribing
  // a small box; 1 restores the single-ring behaviour.
  if (const char *env = clio::run::env::GetCompat("SHM_IN_SHARDS")) {
    char *end = nullptr;
    unsigned long n = std::strtoul(env, &end, 10);
    if (end != env && n >= 1) {
      shm_in_shards_ = static_cast<u32>(n);
    }
  }

  // issue #807: CLIO_SHM_ASYNC_SEND=1 defers SHM response send to a background
  // thread. Off by default (see GetShmAsyncSend — 3x latency regression on
  // latency-bound workloads).
  if (const char *env = clio::run::env::GetCompat("SHM_ASYNC_SEND")) {
    shm_async_send_ = (env[0] == '1' || env[0] == 't' || env[0] == 'T');
  }

  // issue #807/#784: CLIO_SHM_CLIENT_SPIN_US — waiter spin-before-park budget.
  if (const char *env = clio::run::env::GetCompat("SHM_CLIENT_SPIN_US")) {
    char *end = nullptr;
    unsigned long n = std::strtoul(env, &end, 10);
    if (end != env) {
      shm_client_spin_us_ = static_cast<u32>(n);
    }
  }
}

bool ConfigManager::ServerInit() {
  // Configuration is needed by both client and server, so same implementation
  return ClientInit();
}

bool ConfigManager::LoadYaml(const std::string &config_path) {
  try {
    // An empty file yields a YAML null node: every section lookup in
    // ParseYAML misses and the runtime silently comes up with the default
    // config (default port, default workers, empty compose — so no storage
    // tiers). A caller that explicitly pointed CLIO_SERVER_CONF at a file
    // almost certainly did not mean that, and the resulting failures surface
    // far downstream (e.g. PutBlob out-of-space on the very first write).
    // Report it as a load failure so ClientInit warns loudly.
    std::error_code ec;
    const std::string real_path = ctp::ConfigParse::ExpandPath(config_path);
    const auto size = std::filesystem::file_size(real_path, ec);
    if (!ec && size == 0) {
      HLOG(kError, "Config file {} exists but is empty", real_path);
      return false;
    }

    // Parse the YAML as-is (yaml port/settings win). Env overrides (CLIO_PORT
    // et al.) are applied by ClientInit/ServerInit via ApplyEnvOverrides(), not
    // here — a bare LoadYaml must reflect the file so callers parsing arbitrary
    // configs (e.g. a compose file into a local ConfigManager) get the file's
    // values, and the runtime's operational config is never silently retargeted
    // by a config reload.
    LoadFromFile(config_path, true);
    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

std::string ConfigManager::GetServerConfigPath() const {
  // Check env var first: CLIO_SERVER_CONF preferred, CLIO_SERVER_CONF legacy.
  const char *env_path = clio::run::env::GetCompat("SERVER_CONF");
  if (env_path) {
    return std::string(env_path);
  }

  // Fall back to a per-user config file. Lookup order, first hit wins:
  //   1. ~/.clio/clio.yaml      (new canonical name)
  //   2. ~/.clio/clio.yaml  (legacy filename in the new dir)
  //   3. ~/.clio/clio.yaml  (new filename in the legacy dir)
  //   4. ~/.clio/clio.yaml  (legacy)
  // All four are supported; installers seed both ~/.clio/ AND ~/.clio/
  // with identical content so either layout works in the wild.
  const char *kCandidates[] = {
      "${HOME}/.clio/clio.yaml",
      "${HOME}/.clio/clio.yaml",
      "${HOME}/.clio/clio.yaml",
      "${HOME}/.clio/clio.yaml",
  };
  for (const char *tmpl : kCandidates) {
    std::string path = ctp::ConfigParse::ExpandPath(tmpl);
    if (std::filesystem::exists(path)) {
      return path;
    }
  }

  return std::string();
}

size_t ConfigManager::GetMemorySegmentSize(MemorySegment segment) const {
  switch (segment) {
  case kMainSegment:
    return CalculateMainSegmentSize();
  case kClientDataSegment:
    return client_data_segment_size_;
  case kQueueSegment:
    return CalculateQueueSegmentSize();
  default:
    return 0;
  }
}

u32 ConfigManager::GetPort() const { return port_; }

bool ConfigManager::IsEphemeral() const { return ephemeral_; }

std::string ConfigManager::GetServerAddr() const { return server_addr_; }

u32 ConfigManager::GetNeighborhoodSize() const { return neighborhood_size_; }

std::string
ConfigManager::GetSharedMemorySegmentName(MemorySegment segment,
                                          u32 port) const {
  std::string segment_name;

  switch (segment) {
  case kMainSegment:
    segment_name = main_segment_name_;
    break;
  case kClientDataSegment:
    segment_name = client_data_segment_name_;
    break;
  case kQueueSegment:
    segment_name = queue_segment_name_;
    break;
  case kMetadataSegment:
    segment_name = metadata_segment_name_;
    break;
  default:
    return "";
  }

  // Suffix with the port so multiple runtimes on one node + ${USER} (the
  // fallback-runtime topology) own distinct segments rather than colliding.
  // port == 0 means "this runtime"; a non-zero port names another runtime's
  // segment (the fallback client attaching the main runtime's segments).
  u32 name_port = (port != 0) ? port : port_;
  // Use CTP's ExpandPath to resolve environment variables
  return ctp::ConfigParse::ExpandPath(segment_name) + "_" +
         std::to_string(name_port);
}

std::string ConfigManager::GetHostfilePath() const {
  if (hostfile_path_.empty()) {
    return "";
  }

  // Use CTP's ExpandPath to resolve environment variables in hostfile path
  return ctp::ConfigParse::ExpandPath(hostfile_path_);
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
  metadata_segment_name_ = "chi_metadata_segment_${USER}";
  metadata_segment_size_ = 0;  // 0 means auto-calculate

  // Set default hostfile path (empty means no networking/distributed mode)
  hostfile_path_ = "";

  // Set default network retry configuration
  wait_for_restart_timeout_ = 30;      // 30 seconds
  wait_for_restart_poll_period_ = 1;   // 1 second

  // Set default worker sleep configuration (in microseconds)
  first_busy_wait_ = 1000;             // 1000us busy wait
  max_sleep_ = 50000;                  // 50000us (50ms) maximum sleep

  // Set default task load prediction model learning rate
  learning_rate_ = 0.2f;
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

    // Periodic cross-node task-validity check interval (issue #628)
    if (runtime["task_progress_interval_ms"]) {
      task_progress_interval_ms_ =
          runtime["task_progress_interval_ms"].as<u32>();
    }

    // Configuration directory for persistent runtime config
    if (runtime["conf_dir"]) {
      conf_dir_ = runtime["conf_dir"].as<std::string>();
    }

    // Task load prediction model learning rate
    if (runtime["learning_rate"]) {
      learning_rate_ = runtime["learning_rate"].as<float>();
    }

    // Size of the runtime-wide metadata segment (issue #783), which backs the
    // CTE's shared-memory tag/blob maps. Accepts a byte count or a size string
    // ("8g", "512MB"); 0 restores the built-in default.
    //
    // This needs to be tunable because the 8 GB default is more than some
    // hosts can back. On Windows CI, CreateFileMapping cannot reserve it and
    // the runtime falls back to the no-cache path, silently disabling the
    // feature — with no way to ask for a smaller segment instead.
    if (runtime["metadata_segment_size"]) {
      size_t parsed = 0;
      if (ParseSegmentSizeNode(runtime["metadata_segment_size"],
                               "metadata_segment_size", parsed)) {
        metadata_segment_size_ = parsed;
      }
    }

    // Note: stack_size parameter removed (was never used)
    // Note: heartbeat_interval parsing removed (not used by runtime)
  }

  // Env override for the task-progress validity-check interval (issue #628):
  // env wins over yaml so it can be tuned per-run without editing a config.
  if (const char *env = std::getenv("CLIO_TASK_PROGRESS_INTERVAL_MS")) {
    task_progress_interval_ms_ = static_cast<u32>(std::atoi(env));
  }

  // Parse GPU orchestrator configuration
  if (yaml_conf["gpu"]) {
    auto gpu = yaml_conf["gpu"];
    if (gpu["blocks"]) {
      gpu_blocks_ = gpu["blocks"].as<u32>();
    }
    if (gpu["threads_per_block"]) {
      gpu_threads_per_block_ = gpu["threads_per_block"].as<u32>();
    }
    if (gpu["queue_depth"]) {
      gpu_queue_depth_ = gpu["queue_depth"].as<u32>();
    }
  }
  // Environment variable overrides for GPU config (higher priority than YAML).
  // Allows benchmarks to set the partition count dynamically from their
  // thread parameters before CLIO_INIT().
  if (const char *env = clio::run::env::GetCompat("GPU_BLOCKS")) {
    gpu_blocks_ = static_cast<u32>(std::stoul(env));
  }
  if (const char *env = clio::run::env::GetCompat("GPU_THREADS")) {
    gpu_threads_per_block_ = static_cast<u32>(std::stoul(env));
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

  // Parse SWIM membership-detection configuration. All fields optional;
  // unspecified fields keep their compile-time defaults (matches the
  // prior hard-coded constants in admin_runtime.cc).
  if (yaml_conf["swim"]) {
    auto swim = yaml_conf["swim"];
    if (swim["enabled"]) {
      swim_enabled_ = swim["enabled"].as<bool>();
    }
    if (swim["direct_probe_timeout_sec"]) {
      swim_direct_probe_timeout_sec_ =
          swim["direct_probe_timeout_sec"].as<float>();
    }
    if (swim["indirect_probe_timeout_sec"]) {
      swim_indirect_probe_timeout_sec_ =
          swim["indirect_probe_timeout_sec"].as<float>();
    }
    if (swim["suspicion_timeout_sec"]) {
      swim_suspicion_timeout_sec_ =
          swim["suspicion_timeout_sec"].as<float>();
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

        // Optional RPC access control. container_visibility sets the default
        // visibility for every RPC (public|private, default public); a private
        // container rejects RPCs from external user clients (runtime-internal
        // callers are always allowed).
        if (pool_node["container_visibility"]) {
          std::string vis = pool_node["container_visibility"].as<std::string>();
          pool_config.container_visibility_ = (vis == "private") ? 1u : 0u;
        }
        // container_rpc_acl is a map of RPC method NAME -> public|private that
        // overrides container_visibility per method.
        if (pool_node["container_rpc_acl"] &&
            pool_node["container_rpc_acl"].IsMap()) {
          for (const auto& kv : pool_node["container_rpc_acl"]) {
            std::string name = kv.first.as<std::string>();
            std::string v = kv.second.as<std::string>();
            pool_config.rpc_acl_[name] = (v == "private") ? 1u : 0u;
          }
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

  // Main segment holds task data (FutureShm, BuddyAllocator metadata) — no queues
  // Use 1 GB default for task/data allocations
  return ctp::Unit<size_t>::Gigabytes(1);
}

size_t ConfigManager::CalculateMetadataSegmentSize() const {
  // If metadata_segment_size is explicitly set (non-zero), use it
  if (metadata_segment_size_ > 0) {
    return metadata_segment_size_;
  }

  // Default 8 GB. This is an address-space reservation, not an allocation: the
  // segment is never pre-faulted, so only pages the runtime actually writes
  // consume RAM. Sized generously because the CTE metadata cache (issue #783)
  // holds one entry per live tag and blob, and growing the segment later would
  // invalidate every client's mapping.
  return ctp::Unit<size_t>::Gigabytes(8);
}

size_t ConfigManager::CalculateQueueSegmentSize() const {
  // Queue segment holds TaskQueue and NetQueue ring buffers (ArenaAllocator)
  constexpr size_t BASE_OVERHEAD = 4 * 1024 * 1024;  // 4MB for allocator metadata
  constexpr u32 NUM_PRIORITIES = 2;                   // normal + resumed

  // issue #785: size for the elastic replacements too. Lanes are indexed by
  // worker id (RouteTask resolves GetLane(dest_worker_id, 0)), so a worker
  // spawned later has no lane at all unless one was reserved up front — and a
  // worker with no lane can neither be routed to nor receive redistributed
  // work, which is the constraint that made head-of-line blocking
  // unrecoverable under saturation.
  u32 total_workers = num_threads_ + 1 + GetElasticLaneHeadroom();

  // Calculate worker task queues size: TaskQueue with total_workers lanes
  size_t worker_queues_size = TaskQueue::CalculateSize(
      total_workers,      // num_lanes
      NUM_PRIORITIES,     // num_priorities
      queue_depth_);      // depth per queue

  // Calculate network queue size: NetQueue with 1 lane, 4 priorities
  size_t net_queue_size = NetQueue::CalculateSize(
      1,                  // num_lanes
      4,                  // num_priorities: SendIn, SendOut, ClientSendTcp, ClientSendIpc
      queue_depth_);      // depth per queue

  return BASE_OVERHEAD + worker_queues_size + net_queue_size;
}

}  // namespace clio::run