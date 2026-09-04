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
 * The admin ChiMod's web-dashboard endpoints (issue #990).
 *
 * Two kinds of route live here:
 *
 *   - node-local reads (topology, pools, config, routes), answered straight out
 *     of the manager singletons; and
 *   - Monitor forwards (workers, system stats, containers, bdevs, and the
 *     generic per-pool escape hatch), which submit a MonitorTask and wait for
 *     it, converting the msgpack payload each ChiMod's Monitor() produces into
 *     JSON for the browser.
 *
 * Handlers run on the dashboard's HTTP thread pool, not on a worker, so
 * Future::Wait() here is an ordinary in-process wait (the same thing a FUSE
 * thread or a client process does) rather than a worker-blocking call. Every
 * wait is bounded so a dead peer cannot pin an HTTP thread.
 */

#include <clio_ctp/serialize/msgpack_wrapper.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdlib>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clio_runtime/admin/admin_runtime.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/module_manager.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/viz/viz_json.h"
#include "clio_runtime/viz/viz_server.h"
#include "clio_runtime/work_orchestrator.h"

namespace clio::run::admin {

namespace {

using clio::run::viz::JsonWriter;
using clio::run::viz::Request;
using clio::run::viz::Response;

/** How long a dashboard request waits on a MonitorTask before giving up. A
 *  cross-node query to a node that is gone must not pin an HTTP thread. */
constexpr float kMonitorTimeoutSec = 5.0f;

/** How long a create-pool request waits. Creation composes nested pools (a CTE
 *  create composes its bdev tiers), so it gets a bigger budget than a read. */
constexpr float kCreateTimeoutSec = 30.0f;

/** ChiMod name the admin routes are attributed to. */
constexpr const char *kAdminModName = "clio_admin";

//===========================================================================
// msgpack -> JSON
//===========================================================================

/** Append one msgpack object to @p w as the equivalent JSON value. */
void ObjToJson(const msgpack::object &obj, JsonWriter &w) {
  switch (obj.type) {
    case msgpack::type::NIL:
      w.Null();
      break;
    case msgpack::type::BOOLEAN:
      w.Value(obj.via.boolean);
      break;
    case msgpack::type::POSITIVE_INTEGER:
      w.Value(obj.via.u64);
      break;
    case msgpack::type::NEGATIVE_INTEGER:
      w.Value(obj.via.i64);
      break;
    case msgpack::type::FLOAT32:
    case msgpack::type::FLOAT64:
      w.Value(obj.via.f64);
      break;
    case msgpack::type::STR:
      w.Value(std::string(obj.via.str.ptr, obj.via.str.size));
      break;
    case msgpack::type::BIN:
      // Nothing the dashboard reads packs BIN; describe it rather than emitting
      // bytes that would not survive JSON.
      w.Value(std::string("<binary ") + std::to_string(obj.via.bin.size) +
              " bytes>");
      break;
    case msgpack::type::ARRAY:
      w.BeginArray();
      for (uint32_t i = 0; i < obj.via.array.size; ++i) {
        ObjToJson(obj.via.array.ptr[i], w);
      }
      w.EndArray();
      break;
    case msgpack::type::MAP:
      w.BeginObject();
      for (uint32_t i = 0; i < obj.via.map.size; ++i) {
        const auto &kv = obj.via.map.ptr[i];
        if (kv.key.type == msgpack::type::STR) {
          w.Key(std::string(kv.key.via.str.ptr, kv.key.via.str.size));
        } else if (kv.key.type == msgpack::type::POSITIVE_INTEGER) {
          w.Key(std::to_string(kv.key.via.u64));
        } else {
          w.Key("?");
        }
        ObjToJson(kv.val, w);
      }
      w.EndObject();
      break;
    default:
      w.Null();
      break;
  }
}

/**
 * Flatten every container's msgpack payload into one JSON array.
 *
 * A ChiMod's Monitor() packs either an ARRAY of records (worker_stats,
 * container_stats, ...) or a single MAP (get_host_info); both shapes flatten to
 * a list of records, which is what the dashboard's tables want. This mirrors
 * what the Python visualizer's API layer did per endpoint.
 */
std::string ResultsToJsonArray(
    const std::unordered_map<clio::run::ContainerId, std::string> &results) {
  JsonWriter w;
  w.BeginArray();
  for (const auto &kv : results) {
    if (kv.second.empty()) {
      continue;
    }
    try {
      msgpack::object_handle oh =
          msgpack::unpack(kv.second.data(), kv.second.size());
      const msgpack::object &obj = oh.get();
      if (obj.type == msgpack::type::ARRAY) {
        for (uint32_t i = 0; i < obj.via.array.size; ++i) {
          ObjToJson(obj.via.array.ptr[i], w);
        }
      } else {
        ObjToJson(obj, w);
      }
    } catch (const std::exception &e) {
      HLOG(kWarning, "Viz: undecodable monitor payload from container {}: {}",
           kv.first, e.what());
    }
  }
  w.EndArray();
  return w.Str();
}

/** Keep every container's payload separate, keyed by container id. Used by the
 *  generic per-pool endpoint, where the caller asked for a raw view. */
std::string ResultsToJsonMap(
    const std::unordered_map<clio::run::ContainerId, std::string> &results) {
  JsonWriter w;
  w.BeginObject();
  for (const auto &kv : results) {
    w.Key(std::to_string(kv.first));
    if (kv.second.empty()) {
      w.Null();
      continue;
    }
    try {
      msgpack::object_handle oh =
          msgpack::unpack(kv.second.data(), kv.second.size());
      ObjToJson(oh.get(), w);
    } catch (const std::exception &) {
      w.Null();
    }
  }
  w.EndObject();
  return w.Str();
}

//===========================================================================
// Helpers
//===========================================================================

/**
 * Resolve a `{node}` path variable to a PoolQuery.
 *
 * "local" (or an empty value) and this node's own id both route locally, which
 * keeps the common single-node case off the network entirely.
 * @return false if @p node_var is neither "local" nor a number.
 */
bool NodeQuery(const std::string &node_var, clio::run::PoolQuery *out,
               clio::run::u64 *node_id_out) {
  auto *ipc_manager = CLIO_IPC;
  const clio::run::u64 self = ipc_manager ? ipc_manager->GetNodeId() : 0;
  if (node_var.empty() || node_var == "local" || node_var == "self") {
    *out = clio::run::PoolQuery::Local();
    *node_id_out = self;
    return true;
  }
  if (node_var.find_first_not_of("0123456789") != std::string::npos) {
    return false;
  }
  clio::run::u64 node_id = std::strtoull(node_var.c_str(), nullptr, 10);
  *node_id_out = node_id;
  if (node_id == self) {
    *out = clio::run::PoolQuery::Local();
    return true;
  }
  // Reject a node this cluster does not have. Physical routing to an unknown id
  // does not fail -- it lands somewhere -- so without this check the dashboard
  // would happily label another node's numbers as node 9's.
  bool known = false;
  if (ipc_manager) {
    for (const clio::run::Host &host : ipc_manager->GetAllHosts()) {
      if (host.node_id == node_id) {
        known = true;
        break;
      }
    }
  }
  if (!known) {
    return false;
  }
  *out = clio::run::PoolQuery::Physical(static_cast<clio::run::u32>(node_id));
  return true;
}

/** @return a SWIM node state as a short string for the dashboard. */
const char *NodeStateName(clio::run::NodeState state) {
  switch (state) {
    case clio::run::NodeState::kAlive: return "alive";
    case clio::run::NodeState::kProbeFailed: return "probe_failed";
    case clio::run::NodeState::kSuspected: return "suspected";
    case clio::run::NodeState::kDead: return "dead";
  }
  return "unknown";
}

}  // namespace

//===========================================================================
// Monitor forwarding
//===========================================================================

namespace {

/**
 * Run a Monitor query on behalf of a dashboard request and collect the raw
 * per-container payloads.
 *
 * Called from an HTTP thread, never from a worker: it submits a MonitorTask
 * and waits on the Future with a bounded timeout, the same way a client
 * process would. Deliberately a free function over a LOCAL admin client --
 * RegisterViz runs on a prototype container at module-load time, so no
 * handler may capture container state (see Container::RegisterViz).
 */
bool VizMonitor(
    const clio::run::PoolQuery &pool_query, const std::string &query,
    std::unordered_map<clio::run::ContainerId, std::string> *results,
    std::string *error) {
  Client admin_client(clio::run::kAdminPoolId);
  auto future = admin_client.AsyncMonitor(pool_query, query);
  if (!future.Wait(kMonitorTimeoutSec)) {
    *error = "monitor '" + query + "' timed out after " +
             std::to_string(static_cast<int>(kMonitorTimeoutSec)) + "s";
    return false;
  }
  if (future->GetReturnCode() != 0) {
    *error = "monitor '" + query + "' failed with rc=" +
             std::to_string(future->GetReturnCode());
    return false;
  }
  *results = future->results_;
  return true;
}

}  // namespace

//===========================================================================
// Route registration
//===========================================================================

void Runtime::RegisterViz(clio::run::viz::VizServer &viz,
                          const std::string &mod_name) {
  (void)mod_name;  // routes below are the admin's own, named explicitly

  // The admin ChiMod owns the dashboard shell, so "/" lands on its index page.
  viz.SetHome("/viz/clio_admin/index.html");

  // ---- Liveness ----------------------------------------------------------
  viz.AddRoute({"GET", "/api/health", kAdminModName,
                "Dashboard liveness plus this node's identity",
                [](const Request &, Response &resp) {
                  auto *ipc_manager = CLIO_IPC;
                  JsonWriter w;
                  w.BeginObject();
                  w.Field("ok", true);
                  if (ipc_manager) {
                    w.Field("node_id", ipc_manager->GetNodeId());
                    w.Field("hostname", ipc_manager->GetCurrentHostname());
                    w.Field("ip_address", ipc_manager->GetThisHost().ip_address);
                    w.Field("is_leader", ipc_manager->IsLeader());
                  }
                  w.EndObject();
                  resp.Json(w.Str());
                }});

  // ---- Cluster topology --------------------------------------------------
  // Straight out of the local host table and SWIM state: no probing, no
  // broadcast, so a dead peer costs nothing to report.
  viz.AddRoute({"GET", "/api/topology", kAdminModName,
                "Cluster membership and SWIM state as this node sees it",
                [](const Request &, Response &resp) {
                  auto *ipc_manager = CLIO_IPC;
                  if (!ipc_manager) {
                    resp.Error(503, "IPC manager unavailable");
                    return;
                  }
                  const clio::run::u64 self = ipc_manager->GetNodeId();
                  const clio::run::u64 leader = ipc_manager->GetLeaderNodeId();
                  JsonWriter w;
                  w.BeginObject();
                  w.Field("self_node_id", self);
                  w.Field("leader_node_id", leader);
                  w.Field("hostname", ipc_manager->GetCurrentHostname());
                  w.Key("nodes").BeginArray();
                  for (const clio::run::Host &host :
                       ipc_manager->GetAllHosts()) {
                    w.BeginObject();
                    w.Field("node_id", host.node_id);
                    w.Field("ip_address", host.ip_address);
                    w.Field("state", NodeStateName(host.state));
                    w.Field("alive",
                            host.state == clio::run::NodeState::kAlive);
                    w.Field("is_leader", host.node_id == leader);
                    w.Field("is_self", host.node_id == self);
                    w.EndObject();
                  }
                  w.EndArray();
                  w.EndObject();
                  resp.Json(w.Str());
                }});

  // ---- Pools -------------------------------------------------------------
  viz.AddRoute({"GET", "/api/pools", kAdminModName,
                "Pools composed on this node, from the pool manager",
                [](const Request &, Response &resp) {
                  auto *pool_manager = CLIO_POOL_MANAGER;
                  if (!pool_manager) {
                    resp.Error(503, "pool manager unavailable");
                    return;
                  }
                  JsonWriter w;
                  w.BeginObject();
                  w.Key("pools").BeginArray();
                  for (const auto &pid : pool_manager->GetAllPoolIds()) {
                    const auto *info = pool_manager->GetPoolInfo(pid);
                    if (!info) continue;
                    auto serving =
                        pool_manager->GetRealOrStaticContainer(pid).get();
                    w.BeginObject();
                    w.Field("pool_id", pid.ToString());
                    w.Field("pool_name", info->pool_name_);
                    w.Field("chimod_name", info->chimod_name_);
                    w.Field("container_id",
                            serving ? serving->container_id_ : 0u);
                    w.EndObject();
                  }
                  w.EndArray();
                  w.EndObject();
                  resp.Json(w.Str());
                }});

  // ---- Effective configuration ------------------------------------------
  viz.AddRoute({"GET", "/api/config", kAdminModName,
                "The runtime settings this daemon actually came up with",
                [vizp = &viz](const Request &, Response &resp) {
                  auto *config = CLIO_CONFIG_MANAGER;
                  auto *orchestrator = CLIO_WORK_ORCHESTRATOR;
                  auto *module_manager = CLIO_MODULE_MANAGER;
                  if (!config) {
                    resp.Error(503, "config manager unavailable");
                    return;
                  }
                  JsonWriter w;
                  w.BeginObject();
                  w.Field("port", config->GetPort());
                  w.Field("server_addr", config->GetServerAddr());
                  w.Field("num_threads", config->GetNumThreads());
                  w.Field("worker_count",
                          orchestrator
                              ? static_cast<clio::run::u64>(
                                    orchestrator->GetWorkerCount())
                              : 0);
                  w.Field("local_sched", config->GetLocalSched());
                  w.Field("hostfile", config->GetHostfilePath());
                  w.Field("conf_dir", config->GetConfDir());
                  w.Field("ephemeral", config->IsEphemeral());
                  w.Field("learning_rate", config->GetLearningRate());
                  w.Field("swim_enabled", config->GetSwimEnabled());
                  // The port ACTUALLY bound, not the configured one: with
                  // `viz: port: 0` (bind anything free) they differ, and the
                  // number a dashboard shows has to be the one you can reach.
                  w.Field("viz_port", vizp->IsRunning() ? vizp->GetPort()
                                                        : config->GetVizPort());
                  w.Field("viz_bind", vizp->IsRunning()
                                          ? vizp->GetBindAddr()
                                          : config->GetVizBindAddr());
                  w.Key("chimods").BeginArray();
                  if (module_manager) {
                    for (const std::string &name :
                         module_manager->GetLoadedChiMods()) {
                      w.Value(name);
                    }
                  }
                  w.EndArray();
                  w.EndObject();
                  resp.Json(w.Str());
                }});

  // ---- Dashboard self-description ---------------------------------------
  // What makes the dashboard discoverable: every ChiMod's registered routes and
  // mounted pages, so a page can list the modules that shipped a UI.
  viz.AddRoute({"GET", "/api/routes", kAdminModName,
                "Every route and asset mount registered by any ChiMod",
                [vizp = &viz](const Request &, Response &resp) {
                  JsonWriter w;
                  w.BeginObject();
                  w.Key("routes").BeginArray();
                  for (const auto &route : vizp->GetRoutes()) {
                    w.BeginObject();
                    w.Field("method", route.method);
                    w.Field("path", route.path);
                    w.Field("mod_name", route.mod_name);
                    w.Field("doc", route.doc);
                    w.EndObject();
                  }
                  w.EndArray();
                  w.Key("mounts").BeginArray();
                  for (const auto &mount : vizp->GetMounts()) {
                    w.BeginObject();
                    w.Field("mod_name", mount.mod_name);
                    w.Field("url_prefix", mount.url_prefix);
                    w.Field("dir", mount.dir);
                    w.EndObject();
                  }
                  w.EndArray();
                  w.EndObject();
                  resp.Json(w.Str());
                }});

  // ---- Per-node Monitor forwards ----------------------------------------
  // {node} is a node id or "local". Each of these is one MonitorTask to the
  // admin pool on that node, flattened into a JSON array.
  struct NodeRoute {
    const char *path;
    const char *key;
    const char *query;
    const char *doc;
  };
  static const NodeRoute kNodeRoutes[] = {
      {"/api/nodes/{node}/workers", "workers", "worker_stats",
       "Per-worker queue depth, load and task counts"},
      {"/api/nodes/{node}/containers", "containers", "container_stats",
       "Per-pool containers and their learned task-cost models"},
      {"/api/nodes/{node}/bdevs", "devices", "bdev_stats",
       "Every block device on the node, with capacity and throughput"},
  };
  for (const NodeRoute &nr : kNodeRoutes) {
    const std::string key = nr.key;
    const std::string query = nr.query;
    viz.AddRoute({"GET", nr.path, kAdminModName, nr.doc,
                  [key, query](const Request &req, Response &resp) {
                    clio::run::PoolQuery pool_query;
                    clio::run::u64 node_id = 0;
                    if (!NodeQuery(req.Var("node"), &pool_query, &node_id)) {
                      resp.Error(400, "unknown or malformed node id: " + req.Var("node") +
                                        " (use a node id from /api/topology, or 'local')");
                      return;
                    }
                    std::unordered_map<clio::run::ContainerId, std::string>
                        results;
                    std::string error;
                    if (!VizMonitor(pool_query, query, &results, &error)) {
                      resp.Error(503, error);
                      return;
                    }
                    JsonWriter w;
                    w.BeginObject();
                    w.Field("node_id", node_id);
                    w.RawField(key, ResultsToJsonArray(results));
                    w.EndObject();
                    resp.Json(w.Str());
                  }});
  }

  // system_stats takes a cursor, so it gets its own handler.
  viz.AddRoute(
      {"GET", "/api/nodes/{node}/system_stats", kAdminModName,
       "Sampled CPU / RAM / GPU utilization ring, newer than ?min_event_id",
       [](const Request &req, Response &resp) {
         clio::run::PoolQuery pool_query;
         clio::run::u64 node_id = 0;
         if (!NodeQuery(req.Var("node"), &pool_query, &node_id)) {
           resp.Error(400, "unknown or malformed node id: " + req.Var("node") +
                                        " (use a node id from /api/topology, or 'local')");
           return;
         }
         const std::string cursor = req.Param("min_event_id", "0");
         if (cursor.find_first_not_of("0123456789") != std::string::npos) {
           resp.Error(400, "min_event_id must be a number");
           return;
         }
         std::unordered_map<clio::run::ContainerId, std::string> results;
         std::string error;
         if (!VizMonitor(pool_query, "system_stats:" + cursor, &results,
                         &error)) {
           resp.Error(503, error);
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("node_id", node_id);
         w.RawField("entries", ResultsToJsonArray(results));
         w.EndObject();
         resp.Json(w.Str());
       }});

  // ---- Generic per-pool Monitor ------------------------------------------
  // The escape hatch that makes the dashboard extensible without new C++: any
  // ChiMod's Monitor() query, on any pool, from a page's fetch(). Routed through
  // the admin's pool_stats:// forwarder, so it reaches the pool's own container.
  viz.AddRoute(
      {"GET", "/api/pools/{pool}/monitor", kAdminModName,
       "Forward ?query to a pool's own Monitor() (?routing=local|broadcast|...)",
       [](const Request &req, Response &resp) {
         const std::string pool = req.Var("pool");
         const std::string query = req.Param("query");
         const std::string routing = req.Param("routing", "local");
         if (query.empty()) {
           resp.Error(400, "missing ?query");
           return;
         }
         // The pieces are spliced into a pool_stats:// URI that the admin's
         // Monitor parses on ':' boundaries, so anything exotic in them would
         // reinterpret the URI rather than reach a ChiMod. Keep them to the
         // characters real pool ids, routing modes and selectors use.
         const std::string kAllowed =
             "abcdefghijklmnopqrstuvwxyz"
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-";
         if (pool.find_first_not_of(kAllowed) != std::string::npos ||
             query.find_first_not_of(kAllowed) != std::string::npos ||
             routing.find_first_not_of(kAllowed + ":") != std::string::npos) {
           resp.Error(400, "pool, query and routing must be alphanumeric");
           return;
         }
         std::unordered_map<clio::run::ContainerId, std::string> results;
         std::string error;
         if (!VizMonitor(clio::run::PoolQuery::Local(),
                         "pool_stats://" + pool + ":" + routing + ":" + query,
                         &results, &error)) {
           resp.Error(503, error);
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("pool_id", pool);
         w.Field("query", query);
         w.RawField("results", ResultsToJsonMap(results));
         w.EndObject();
         resp.Json(w.Str());
       }});

  // ---- Pool shutdown -------------------------------------------------------
  // The Pools tab's per-card "x". Refuses the admin pool: destroying it is not
  // pool management, it is killing the runtime out from under every client --
  // the CLI's `clio_run stop` is the honest spelling of that intent.
  viz.AddRoute(
      {"POST", "/api/pools/{pool}/destroy", kAdminModName,
       "Destroy a pool and its containers on this node",
       [](const Request &req, Response &resp) {
         const std::string pool_str = req.Var("pool");
         clio::run::PoolId pool_id;
         try {
           pool_id = clio::run::PoolId::FromString(pool_str);
         } catch (const std::exception &e) {
           resp.Error(400, "bad pool id '" + pool_str + "': " + e.what());
           return;
         }
         auto *pool_manager = CLIO_POOL_MANAGER;
         const auto *info =
             pool_manager ? pool_manager->GetPoolInfo(pool_id) : nullptr;
         if (!info) {
           resp.Error(404, "no pool with id " + pool_str);
           return;
         }
         if (pool_id == clio::run::kAdminPoolId) {
           resp.Error(403,
                      "refusing to destroy the admin pool -- that is the "
                      "runtime itself; use `clio_run stop`");
           return;
         }
         const std::string pool_name = info->pool_name_;

         Client admin_client(clio::run::kAdminPoolId);
         auto future = admin_client.AsyncDestroyPool(
             clio::run::PoolQuery::Local(), pool_id);
         if (!future.Wait(kCreateTimeoutSec)) {
           resp.Error(503, "DestroyPool timed out");
           return;
         }
         HLOG(kInfo, "Viz: destroyed pool {} ({}) from the dashboard",
              pool_str, pool_name);
         if (future->GetReturnCode() != 0) {
           resp.Error(500, "DestroyPool failed with rc=" +
                               std::to_string(future->GetReturnCode()));
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("ok", true);
         w.Field("pool_id", pool_str);
         w.Field("pool_name", pool_name);
         w.EndObject();
         resp.Json(w.Str());
       }});

  // ---- Module inventory for the "Add Pool" flow ---------------------------
  // Every loaded ChiMod, plus whether it registered a create form
  // (GET /api/mod/<mod>/create) and whether it mounted pages. The Pools page
  // uses this to offer a searchable module list: modules with a form get it,
  // the rest fall back to the generic compose editor below.
  viz.AddRoute({"GET", "/api/chimods", kAdminModName,
                "Loaded ChiMods and their dashboard capabilities",
                [vizp = &viz](const Request &, Response &resp) {
                  auto *module_manager = CLIO_MODULE_MANAGER;
                  if (!module_manager) {
                    resp.Error(503, "module manager unavailable");
                    return;
                  }
                  std::set<std::string> creators;
                  for (const auto &route : vizp->GetRoutes()) {
                    // The convention: a module's create form lives at
                    // GET /api/mod/<mod>/create.
                    if (route.method == "GET" &&
                        route.path == "/api/mod/" + route.mod_name + "/create") {
                      creators.insert(route.mod_name);
                    }
                  }
                  std::set<std::string> mounted;
                  for (const auto &mount : vizp->GetMounts()) {
                    mounted.insert(mount.mod_name);
                  }
                  JsonWriter w;
                  w.BeginObject();
                  w.Key("chimods").BeginArray();
                  for (const std::string &name :
                       module_manager->GetLoadedChiMods()) {
                    w.BeginObject();
                    w.Field("name", name);
                    w.Field("has_create", creators.count(name) > 0);
                    w.Field("has_pages", mounted.count(name) > 0);
                    w.EndObject();
                  }
                  w.EndArray();
                  w.EndObject();
                  resp.Json(w.Str());
                }});

  // ---- Generic pool creation (the compose path) ---------------------------
  // Creates a pool of ANY loaded ChiMod, the same way `clio_run compose` does:
  // the module-specific parameters travel as the compose entry's YAML, so a
  // module needs no dashboard code at all to be creatable. Modules that DO
  // register /api/mod/<mod>/create give the user a real form instead; this is
  // the fallback (and the escape hatch for exotic parameters).
  //
  // action=validate checks everything and creates nothing, which is how the
  // form's Validate button works. Note get-or-create semantics on the create
  // itself: a pool_name that already exists returns the existing pool.
  viz.AddRoute(
      {"POST", "/api/pools/compose", kAdminModName,
       "Create a pool of any ChiMod (fields: mod_name, pool_name, pool_id, "
       "pool_query, config[, action=validate])",
       [](const Request &req, Response &resp) {
         const std::string mod_name = req.Param("mod_name");
         const std::string pool_name = req.Param("pool_name");
         const std::string pool_id_str = req.Param("pool_id");
         const std::string pool_query_str = req.Param("pool_query", "local");
         const std::string config = req.Param("config");
         const bool validate_only = req.Param("action") == "validate";

         // Collect every field error, so one round trip fixes the whole form.
         JsonWriter errors;
         errors.BeginObject();
         size_t error_count = 0;
         auto fail = [&errors, &error_count](const std::string &field,
                                             const std::string &message) {
           errors.Field(field, message);
           ++error_count;
         };

         auto *module_manager = CLIO_MODULE_MANAGER;
         if (mod_name.empty()) {
           fail("mod_name", "required");
         } else if (module_manager && !module_manager->GetChiMod(mod_name)) {
           fail("mod_name", "no ChiMod named '" + mod_name + "' is loaded");
         }
         if (pool_name.empty()) {
           fail("pool_name", "required");
         }
         clio::run::PoolId pool_id;
         if (pool_id_str.empty()) {
           fail("pool_id", "required (e.g. \"" +
                               std::to_string(viz::SuggestFreePoolMajor(800)) +
                               ".0\")");
         } else {
           try {
             pool_id = clio::run::PoolId::FromString(pool_id_str);
             if (pool_id.IsNull()) {
               fail("pool_id", "must not be null");
             }
           } catch (const std::exception &e) {
             fail("pool_id", std::string("unparseable: ") + e.what());
           }
         }
         clio::run::PoolQuery pool_query;
         try {
           pool_query = clio::run::PoolQuery::FromString(pool_query_str);
         } catch (const std::exception &e) {
           fail("pool_query", std::string("unparseable: ") + e.what());
         }
         // The module parses config_ as YAML at Create time, where a syntax
         // error surfaces as an opaque create failure -- so catch it here.
         YAML::Node config_node(YAML::NodeType::Map);
         if (!config.empty()) {
           try {
             config_node = YAML::Load(config);
             if (!config_node.IsMap()) {
               fail("config", "must be a YAML mapping (key: value lines)");
             }
           } catch (const std::exception &e) {
             fail("config", std::string("invalid YAML: ") + e.what());
           }
         }
         errors.EndObject();

         if (error_count > 0) {
           JsonWriter w;
           w.BeginObject();
           w.Field("ok", false);
           w.RawField("errors", errors.Str());
           w.EndObject();
           resp.status = 400;
           resp.Json(w.Str());
           return;
         }
         if (validate_only) {
           resp.Json("{\"ok\":true}");
           return;
         }

         // Build the compose entry exactly as ConfigManager::ParseYAML would
         // have stored it: identity keys merged over the module-specific YAML,
         // the whole node re-emitted into PoolConfig::config_.
         config_node["mod_name"] = mod_name;
         config_node["pool_name"] = pool_name;
         config_node["pool_id"] = pool_id_str;
         config_node["pool_query"] = pool_query_str;
         YAML::Emitter emitter;
         emitter << config_node;

         clio::run::PoolConfig pool_config;
         pool_config.mod_name_ = mod_name;
         pool_config.pool_name_ = pool_name;
         pool_config.pool_id_ = pool_id;
         pool_config.pool_query_ = pool_query;
         pool_config.config_ = emitter.c_str();

         Client admin_client(clio::run::kAdminPoolId);
         auto future = admin_client.AsyncCompose(pool_config);
         if (!future.Wait(kCreateTimeoutSec)) {
           resp.Error(503, "pool creation timed out");
           return;
         }
         if (future->GetReturnCode() != 0) {
           resp.Error(500, "pool creation failed with rc=" +
                               std::to_string(future->GetReturnCode()));
           return;
         }
         JsonWriter w;
         w.BeginObject();
         w.Field("ok", true);
         w.Field("pool_name", pool_name);
         w.Field("pool_id", future->new_pool_id_.ToString());
         w.EndObject();
         resp.Json(w.Str());
       }});
}

}  // namespace clio::run::admin
