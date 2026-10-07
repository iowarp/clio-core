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

#include <clio_cte/dtschedule/dag_spec.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/logging.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <sys/socket.h>

using json = nlohmann::json;

namespace clio::cte::dtschedule {

std::string DagSpecLoader::GetBasename(const std::string &path) {
  /**
   * Extract the filename from a path.
   *
   * Handles both forward slashes (Unix/relative paths) and backslashes
   * (Windows). Returns the full path if no separator is found.
   */
  size_t slash_pos = path.find_last_of("/\\");
  if (slash_pos != std::string::npos) {
    return path.substr(slash_pos + 1);
  }
  return path;
}

uint32_t DagSpecLoader::ResolveHostname(const std::string &hostname,
                                        clio::run::IpcManager *ipc_manager) {
  // The hostfile entry a DAG node refers to may be a hostname, a short
  // hostname, an IP, or "host:port". Match the DAG's name against each
  // entry's string form (exact, short form, or resolved IPv4), and map
  // "localhost" to this runtime's own id.
  if (hostname.empty()) {
    return UINT32_MAX;
  }
  if (ipc_manager == nullptr) {
    return UINT32_MAX;
  }
  if (hostname == "localhost" || hostname == "127.0.0.1" || hostname == "0") {
    return static_cast<uint32_t>(ipc_manager->GetNodeId());
  }
  // This machine's own name needs no host table (single-node runtimes have
  // none, and the table is filled after pools are composed).
  {
    char self_host[256] = {0};
    if (gethostname(self_host, sizeof(self_host) - 1) == 0) {
      const std::string me(self_host);
      const size_t d1 = me.find('.'), d2 = hostname.find('.');
      const std::string me_short = d1 == std::string::npos ? me : me.substr(0, d1);
      const std::string in_short = d2 == std::string::npos ? hostname : hostname.substr(0, d2);
      if (hostname == me || in_short == me_short) {
        return static_cast<uint32_t>(ipc_manager->GetNodeId());
      }
    }
  }
  auto short_of = [](const std::string &h) {
    const size_t dot = h.find('.');
    return dot == std::string::npos ? h : h.substr(0, dot);
  };
  const std::string want_short = short_of(hostname);
  const std::string want_ip = ResolveIpv4(hostname);
  for (const auto &host : ipc_manager->GetAllHosts()) {
    std::string entry = host.ip_address;
    const size_t colon = entry.find(':');
    if (colon != std::string::npos && entry.find(':', colon + 1) == std::string::npos) {
      entry = entry.substr(0, colon);  // strip "host:port" (keep IPv6 intact)
    }
    if (entry == hostname || short_of(entry) == want_short ||
        (!want_ip.empty() && (entry == want_ip || ResolveIpv4(entry) == want_ip))) {
      return static_cast<uint32_t>(host.node_id);
    }
  }
  unresolved_++;
  if (unknown_hostnames_.find(hostname) == unknown_hostnames_.end()) {
    unknown_hostnames_.insert(hostname);
    HLOG(kWarning, "dtschedule dag: hostname '{}' is not in the hostfile; "
                   "its tasks are ignored for placement", hostname);
  }
  return UINT32_MAX;
}

std::string DagSpecLoader::ResolveIpv4(const std::string &name) {
  struct addrinfo hints {};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo *res = nullptr;
  if (getaddrinfo(name.c_str(), nullptr, &hints, &res) != 0 || res == nullptr) {
    return "";
  }
  char buf[INET_ADDRSTRLEN] = {0};
  auto *sin = reinterpret_cast<struct sockaddr_in *>(res->ai_addr);
  inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf));
  freeaddrinfo(res);
  return buf;
}

bool DagSpecLoader::Initialize(const std::string &dag_path,
                               clio::run::IpcManager *ipc_manager,
                               bool colocate_fanin,
                               uint32_t replicate_fanout_min,
                               uint32_t replicate_max) {
  /**
   * Load and parse a placement.json DAG spec file.
   *
   * JSON format (WfCommons + translator):
   * {
   *   "nodes": ["host1", "host2"],
   *   "tasks": {
   *     "task_name": {
   *       "node": "host1",
   *       "outputs": ["data/file1.dat"],
   *       "output_bytes": {"data/file1.dat": 1024}
   *     }
   *   }
   * }
   *
   * Builds file_by_path_ and file_by_basename_ lookup tables.
   * Returns false on file I/O or JSON parse errors.
   */
  is_loaded_ = false;

  // Store configuration
  colocate_fanin_ = colocate_fanin;
  replicate_fanout_min_ = replicate_fanout_min;
  replicate_max_ = replicate_max;

  // Read JSON file
  std::ifstream file(dag_path);
  if (!file.is_open()) {
    HLOG(kError, "dtschedule dag: failed to open {}", dag_path);
    return false;
  }

  json j;
  try {
    file >> j;
  } catch (const json::exception &e) {
    HLOG(kError, "dtschedule dag: JSON parse error in {}: {}", dag_path, e.what());
    return false;
  }
  file.close();

  // Parse and build mappings
  if (!j.contains("tasks")) {
    HLOG(kError, "dtschedule dag: missing 'tasks' key in {}", dag_path);
    return false;
  }
  if (!j.contains("nodes")) {
    HLOG(kWarning, "dtschedule dag: missing 'nodes' key in {}", dag_path);
  }

  // Build task metadata maps
  std::unordered_map<std::string, std::vector<std::string>> task_outputs;  // task -> outputs
  std::unordered_map<std::string, std::string> task_node;                   // task -> hostname
  std::unordered_map<std::string, uint64_t> file_size;                      // file -> size
  std::unordered_map<std::string, std::vector<std::string>> file_consumers; // file -> consuming tasks

  const auto &tasks_obj = j["tasks"];
  if (!tasks_obj.is_object()) {
    HLOG(kError, "dtschedule dag: 'tasks' must be an object");
    return false;
  }

  // First pass: collect producer info
  for (const auto &[task_name, task_data] : tasks_obj.items()) {
    if (!task_data.is_object()) {
      continue;
    }

    // Get task's node
    if (task_data.contains("node") && task_data["node"].is_string()) {
      task_node[task_name] = task_data["node"].get<std::string>();
    }

    // Get output files and sizes
    if (task_data.contains("outputs") && task_data["outputs"].is_array()) {
      const auto &outputs = task_data["outputs"];
      for (const auto &output : outputs) {
        if (output.is_string()) {
          task_outputs[task_name].push_back(output.get<std::string>());

          // Get file size
          if (task_data.contains("output_bytes") &&
              task_data["output_bytes"].is_object() &&
              task_data["output_bytes"].contains(output)) {
            auto sz = task_data["output_bytes"][output];
            if (sz.is_number()) {
              file_size[output] = sz.get<uint64_t>();
            }
          }
        }
      }
    }
  }

  // Second pass: collect consumer info
  for (const auto &[task_name, task_data] : tasks_obj.items()) {
    if (!task_data.is_object()) {
      continue;
    }

    if (task_data.contains("inputs") && task_data["inputs"].is_array()) {
      const auto &inputs = task_data["inputs"];
      for (const auto &input : inputs) {
        if (input.is_string()) {
          file_consumers[input].push_back(task_name);
        }
      }
    }
  }

  // Third pass: build FileInfo for each file
  // task -> outputs tells us each file's producer; task -> inputs its
  // consumers. The translator also writes a `files` block with the same
  // information already resolved to hostnames; it wins when present.
  std::unordered_map<std::string, std::string> file_producer;  // file -> task
  for (const auto &[task_name, outputs] : task_outputs) {
    for (const auto &f : outputs) {
      file_producer[f] = task_name;
    }
  }
  std::set<std::string> all_files;
  for (const auto &[f, t] : file_producer) all_files.insert(f);
  for (const auto &[f, c] : file_consumers) all_files.insert(f);
  const bool has_files = j.contains("files") && j["files"].is_object();
  if (has_files) {
    for (const auto &[f, v] : j["files"].items()) all_files.insert(f);
  }
  for (const std::string &file_path : all_files) {
    FileInfo info;
    std::set<uint32_t> unique_consumers;
    const bool in_block = has_files && j["files"].contains(file_path);
    if (in_block) {
      const auto &fb = j["files"][file_path];
      if (fb.contains("producer_node") && fb["producer_node"].is_string()) {
        info.producer_node =
            ResolveHostname(fb["producer_node"].get<std::string>(), ipc_manager);
      }
      if (fb.contains("consumer_nodes") && fb["consumer_nodes"].is_array()) {
        for (const auto &n : fb["consumer_nodes"]) {
          if (!n.is_string()) continue;
          uint32_t id = ResolveHostname(n.get<std::string>(), ipc_manager);
          if (id != UINT32_MAX) unique_consumers.insert(id);
        }
      }
      if (fb.contains("size") && fb["size"].is_number()) {
        info.size = fb["size"].get<uint64_t>();
      }
    }
    if (info.producer_node == UINT32_MAX) {
      auto p = file_producer.find(file_path);
      if (p != file_producer.end() && task_node.count(p->second)) {
        info.producer_node = ResolveHostname(task_node[p->second], ipc_manager);
      }
    }
    if (unique_consumers.empty()) {
      auto c = file_consumers.find(file_path);
      if (c != file_consumers.end()) {
        for (const auto &consumer_task : c->second) {
          if (!task_node.count(consumer_task)) continue;
          uint32_t id = ResolveHostname(task_node[consumer_task], ipc_manager);
          if (id != UINT32_MAX) unique_consumers.insert(id);
        }
      }
    }
    info.consumer_nodes.assign(unique_consumers.begin(), unique_consumers.end());
    if (info.size == 0 && file_size.count(file_path)) {
      info.size = file_size[file_path];
    }
    file_by_path_[file_path] = info;
    file_by_basename_[GetBasename(file_path)] = info;
  }
  is_loaded_ = true;
  HLOG(kInfo, "dtschedule dag: loaded {} files from {}", file_by_path_.size(),
       dag_path);
  return true;
}

FileInfo DagSpecLoader::LookupFile(const std::string &tag_or_blob_name) const {
  /**
   * Look up a file by its name.
   *
   * Tries exact path match first, then basename match.
   * Returns default {UINT32_MAX, {}, 0} if not found.
   */
  // Try exact path match
  auto it = file_by_path_.find(tag_or_blob_name);
  if (it != file_by_path_.end()) {
    return it->second;
  }

  // Try basename match
  auto basename_it = file_by_basename_.find(tag_or_blob_name);
  if (basename_it != file_by_basename_.end()) {
    return basename_it->second;
  }

  // Try matching against the basename of the input
  std::string basename = GetBasename(tag_or_blob_name);
  basename_it = file_by_basename_.find(basename);
  if (basename_it != file_by_basename_.end()) {
    return basename_it->second;
  }

  // Flat-namespace names ("<run>__<file>", see the WfCommons translator):
  // match the part after the last "__".
  size_t sep = basename.rfind("__");
  if (sep != std::string::npos && sep + 2 < basename.size()) {
    basename_it = file_by_basename_.find(basename.substr(sep + 2));
    if (basename_it != file_by_basename_.end()) {
      return basename_it->second;
    }
  }

  return FileInfo();  // Default: {UINT32_MAX, {}, 0}
}

void DagSpecLoader::GetStats(uint32_t &dag_files, uint32_t &dag_nodes) const {
  /**
   * Return statistics about the loaded DAG.
   *
   * dag_files: total number of files in the spec
   * dag_nodes: count of unique producer and consumer node IDs
   */
  dag_files = file_by_path_.size();

  // Count unique nodes
  std::set<uint32_t> unique_nodes;
  for (const auto &[path, info] : file_by_path_) {
    if (info.producer_node != UINT32_MAX) {
      unique_nodes.insert(info.producer_node);
    }
    for (auto consumer : info.consumer_nodes) {
      if (consumer != UINT32_MAX) {
        unique_nodes.insert(consumer);
      }
    }
  }
  dag_nodes = unique_nodes.size();
}

}  // namespace clio::cte::dtschedule
