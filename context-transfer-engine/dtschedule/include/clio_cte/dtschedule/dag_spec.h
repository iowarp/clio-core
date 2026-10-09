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

#ifndef CLIO_CTE_DTSCHEDULE_DAG_SPEC_H_
#define CLIO_CTE_DTSCHEDULE_DAG_SPEC_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <cstdint>

namespace clio::run {
class IpcManager;
}

namespace clio::cte::dtschedule {

/**
 * File metadata extracted from a DAG spec (placement.json).
 *
 * Tracks producer node, consumer nodes, and file size for workflow-aware
 * placement decisions (replication, colocation, tiering).
 */
struct FileInfo {
  uint32_t producer_node = UINT32_MAX;      ///< Node ID of producer task
  std::vector<uint32_t> consumer_nodes;     ///< Node IDs of consumer tasks
  uint64_t size = 0;                        ///< File size in bytes
};

/**
 * Loader for WfCommons + task-node-map placement.json format.
 *
 * Parses DAG spec files to build file -> (producer, consumers, size) mappings
 * and enables workflow-aware blob placement decisions. Resolves hostnames
 * to node IDs at Initialize time.
 *
 * Usage:
 * auto loader = std::make_unique<DagSpecLoader>();
 * bool ok = loader->Initialize(
 *   "placement.json",
 *   ipc_manager,
 *   true,    // colocate_fanin
 *   16,      // replicate_fanout_min
 *   2        // replicate_max
 * );
 *
 * auto info = loader->LookupFile("data/mAdd_00000018.fits");
 * if (info.producer_node != UINT32_MAX) {
 *   // Found the file
 * }
 */
class DagSpecLoader {
 public:
  DagSpecLoader() = default;
  ~DagSpecLoader() = default;

  // Disable copy operations
  DagSpecLoader(const DagSpecLoader &) = delete;
  DagSpecLoader &operator=(const DagSpecLoader &) = delete;

  /**
   * Initialize the DAG spec from a placement.json file.
   *
   * Loads the WfCommons JSON + task->node map, builds file -> (producer, consumers, size)
   * mappings, and resolves hostnames to node IDs using the host list from IpcManager.
   * Unknown hostnames are logged once and ignored (their tasks are skipped).
   *
   * @param dag_path Path to placement.json file
   * @param ipc_manager Pointer to IpcManager for node ID lookups (can be nullptr)
   * @param colocate_fanin When true, all outputs of one downstream task share that task's node
   * @param replicate_fanout_min Replication threshold (copies when consumer count >= this)
   * @param replicate_max Maximum number of replicas to push
   * @return true on success, false on parse/file errors
   */
  bool Initialize(const std::string &dag_path,
                  clio::run::IpcManager *ipc_manager,
                  bool colocate_fanin,
                  uint32_t replicate_fanout_min,
                  uint32_t replicate_max);

  /**
   * Look up a file in the DAG spec by tag/blob name.
   *
   * Tries to match by:
   * 1. Exact relative path in the spec (e.g., "data/output.fits" from placement.json)
   * 2. Basename only (e.g., "output.fits")
   *
   * Returns {producer_node, {consumer1, consumer2, ...}, size} or {UINT32_MAX, {}, 0}
   * if not found.
   *
   * @param tag_or_blob_name Name to look up (can be a path like "data/mAdd.fits" or just "mAdd.fits")
   * @return FileInfo with producer/consumers/size, or {UINT32_MAX, {}, 0} if not found
   */
  FileInfo LookupFile(const std::string &tag_or_blob_name) const;

  /**
   * Get decision statistics for tests.
   *
   * @param[out] dag_files Count of files in the spec
   * @param[out] dag_nodes Count of unique producer/consumer nodes
   */
  void GetStats(uint32_t &dag_files, uint32_t &dag_nodes) const;

  /**
   * Check if the spec is loaded and ready.
   *
   * @return true if Initialize succeeded and data is available
   */
  bool IsLoaded() const { return is_loaded_; }
  /** Hostname references that did not resolve at the last Initialize. */
  uint32_t Unresolved() const { return unresolved_; }

  /**
   * Get the replicate_max configuration parameter.
   *
   * @return Maximum number of replicas to push (0 if not initialized)
   */
  uint32_t ReplicateMax() const { return replicate_max_; }

  /**
   * Get the replicate_fanout_min configuration parameter.
   *
   * @return Replication threshold for consumer count (0 if not initialized)
   */
  uint32_t ReplicateFanoutMin() const { return replicate_fanout_min_; }

  /**
   * Get the colocate_fanin configuration parameter.
   *
   * @return true if fanin colocation is enabled
   */
  bool ColocateFanin() const { return colocate_fanin_; }

  /**
   * Fan-in degree of a node: how many nodes (itself included) produce files it consumes.
   * Those producers write to it concurrently, sharing its link and tiers.
   *
   * @param node Consumer node ID
   * @return Number of distinct producer nodes (at least 1)
   */
  uint32_t FaninDegree(uint32_t node) const {
    auto it = fanin_.find(node);
    return it == fanin_.end() || it->second == 0 ? 1u : it->second;
  }


 private:
  /**
   * Extract basename from a file path.
   *
   * @param path Full or relative path (e.g., "data/output.fits")
   * @return Basename (e.g., "output.fits")
   */
  static std::string GetBasename(const std::string &path);

  /**
   * Map a hostname to a node ID using the hostlist from IpcManager.
   *
   * Unknown hostnames are logged once per unique hostname.
   *
   * @param hostname The hostname to map
   * @param ipc_manager IpcManager to query for node IDs
   * @return Node ID (0-based) or UINT32_MAX if not found
   */
  uint32_t ResolveHostname(const std::string &hostname,
                           clio::run::IpcManager *ipc_manager);
  /** IPv4 text of a host name via getaddrinfo, or \"\" when unresolvable. */
  static std::string ResolveIpv4(const std::string &name);

  // Data members
  std::unordered_map<std::string, FileInfo> file_by_path_;   ///< Full path -> FileInfo
  std::unordered_map<std::string, FileInfo> file_by_basename_;  ///< Basename -> FileInfo
  std::set<std::string> unknown_hostnames_;                   ///< Logged unknown hostnames
  bool is_loaded_ = false;                                    ///< Successfully loaded
  uint32_t unresolved_ = 0;                                   ///< Unresolved hostname refs
  bool colocate_fanin_ = false;                               ///< Colocation config
  uint32_t replicate_fanout_min_ = 0;                         ///< Replication threshold
  uint32_t replicate_max_ = 0;                                ///< Max replicas
  std::unordered_map<uint32_t, uint32_t> fanin_;              ///< Consumer node -> producer nodes
};

}  // namespace clio::cte::dtschedule

#endif  // CLIO_CTE_DTSCHEDULE_DAG_SPEC_H_
