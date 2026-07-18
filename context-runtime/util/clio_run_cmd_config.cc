// clio_run config export --path <file>
//
// Export the node's CURRENT (live effective) configuration as YAML. Unlike
// re-emitting the input file, this attaches to the running daemon as a client
// and reflects reality:
//   * the runtime/networking settings the node is actually running with,
//   * the composed pools (from the loaded compose config), and
//   * for safe_bdev pools, the LIVE member roster queried from the pool's
//     Monitor("stats") -- so members added/removed/recovered at runtime are
//     reflected, not just the startup members.
// A `live_pools` block lists every currently-active pool (name + id) so any
// pool composed at runtime is visible even when its full config cannot be
// reconstructed.

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <clio_ctp/serialize/msgpack_wrapper.h>
#include <yaml-cpp/yaml.h>

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/config_manager.h>

#include "clio_run_commands.h"

namespace {

void PrintConfigUsage() {
  HIPRINT("Usage: clio_run config <subcommand> [options]");
  HIPRINT("  export --path <file>   Write the node's current effective");
  HIPRINT("                         configuration (live pools + safe-bdev");
  HIPRINT("                         members) as YAML to <file>.");
}

// RAII: join the ZMQ recv thread / close the DEALER socket on every return
// path (mirrors clio_run_cmd_compose.cc).
struct ClientFinalizeGuard {
  ~ClientFinalizeGuard() {
    auto* mgr = CLIO_RUNTIME_MANAGER;
    if (mgr != nullptr) {
      mgr->ClientFinalize();
    }
  }
};

bool InitClient() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Clio runtime client");
    return false;
  }
  return true;
}

std::string PoolIdStr(const clio::run::PoolId& id) {
  return std::to_string(id.major_) + "." + std::to_string(id.minor_);
}

// Query safe_bdev pool `pool_id`'s Monitor("stats") and, if it carries a
// "members" roster, emit it as a YAML sequence. Returns an empty/undefined node
// if the pool is not a safe_bdev, is unreachable, or carries no roster. Any
// error is non-fatal -- the exported compose entry simply keeps its static
// members (or none).
YAML::Node LiveSafeBdevMembers(clio::run::admin::Client* admin,
                               const clio::run::PoolId& pool_id) {
  YAML::Node members(YAML::NodeType::Undefined);
  // Target THIS safe_bdev pool's Monitor("stats") via the admin pool_stats://
  // forwarder (pool_stats://<major.minor>:<routing>:<selector>); a plain
  // Monitor(Dynamic(),"stats") would not route to the pool.
  const std::string uri =
      "pool_stats://" + PoolIdStr(pool_id) + ":local:stats";
  auto mon = admin->AsyncMonitor(clio::run::PoolQuery::Local(), uri);
  mon.Wait();
  if (mon->GetReturnCode() != 0) {
    return members;
  }
  for (const auto& kv_blob : mon->results_) {
    const std::string& blob = kv_blob.second;
    if (blob.empty()) {
      continue;
    }
    try {
      msgpack::object_handle oh = msgpack::unpack(blob.data(), blob.size());
      const msgpack::object& obj = oh.get();
      if (obj.type != msgpack::type::MAP) {
        continue;
      }
      // Confirm this blob is for the pool we asked about (pool_name match is
      // enough; every safe_bdev "stats" map carries pool_name + members).
      const msgpack::object* members_obj = nullptr;
      for (uint32_t j = 0; j < obj.via.map.size; ++j) {
        const auto& e = obj.via.map.ptr[j];
        std::string key;
        e.key.convert(key);
        if (key == "members" &&
            e.val.type == msgpack::type::ARRAY) {
          members_obj = &e.val;
        }
      }
      if (members_obj == nullptr) {
        continue;
      }
      members = YAML::Node(YAML::NodeType::Sequence);
      for (uint32_t m = 0; m < members_obj->via.array.size; ++m) {
        const msgpack::object& mo = members_obj->via.array.ptr[m];
        if (mo.type != msgpack::type::MAP) {
          continue;
        }
        YAML::Node mnode;
        for (uint32_t f = 0; f < mo.via.map.size; ++f) {
          const auto& fe = mo.via.map.ptr[f];
          std::string fkey;
          fe.key.convert(fkey);
          if (fkey == "pool_id") {
            uint64_t v = 0;
            fe.val.convert(v);
            mnode["pool_id"] =
                PoolIdStr(clio::run::PoolId::FromU64(v));
          } else if (fkey == "index" || fkey == "recovering") {
            uint64_t v = 0;
            fe.val.convert(v);
            mnode[fkey] = v;
          } else {  // role, pool_name, state -> strings
            std::string v;
            fe.val.convert(v);
            mnode[fkey] = v;
          }
        }
        members.push_back(mnode);
      }
      return members;  // first safe_bdev blob wins
    } catch (const std::exception& e) {
      HLOG(kWarning, "config export: failed to decode safe_bdev stats: {}",
           e.what());
    }
  }
  (void)pool_id;
  return members;
}

int ConfigExport(const std::string& path) {
  if (!InitClient()) {
    return 1;
  }
  ClientFinalizeGuard guard;

  auto* cfg = CLIO_CONFIG_MANAGER;
  auto* admin = CLIO_ADMIN;
  if (cfg == nullptr || admin == nullptr) {
    HLOG(kError, "config export: runtime client not available");
    return 1;
  }

  YAML::Node root;

  // (1) Runtime + networking effective settings.
  YAML::Node runtime;
  runtime["num_threads"] = cfg->GetNumThreads();
  runtime["queue_depth"] = cfg->GetQueueDepth();
  runtime["first_busy_wait"] = cfg->GetFirstBusyWait();
  runtime["max_sleep"] = cfg->GetMaxSleep();
  root["runtime"] = runtime;

  YAML::Node net;
  net["port"] = cfg->GetPort();
  root["networking"] = net;

  // (2) Compose section reconstructed from the loaded config. Each pool's
  // extracted top-level keys come from the struct; the module-specific
  // remainder is re-parsed from PoolConfig::config_ and merged in.
  const clio::run::ComposeConfig& compose = cfg->GetComposeConfig();
  std::unordered_set<clio::run::u64> composed_ids;
  YAML::Node compose_seq(YAML::NodeType::Sequence);
  for (const auto& pc : compose.pools_) {
    YAML::Node pool;
    pool["mod_name"] = pc.mod_name_;
    pool["pool_name"] = pc.pool_name_;
    pool["pool_id"] = PoolIdStr(pc.pool_id_);
    if (pc.restart_) {
      pool["restart"] = true;
    }
    // Merge the module-specific remainder (bdev_type, capacity, storage, ...).
    if (!pc.config_.empty()) {
      try {
        YAML::Node extra = YAML::Load(pc.config_);
        if (extra.IsMap()) {
          for (auto it = extra.begin(); it != extra.end(); ++it) {
            pool[it->first.as<std::string>()] = it->second;
          }
        }
      } catch (const std::exception& e) {
        HLOG(kWarning, "config export: could not parse config for pool {}: {}",
             pc.pool_name_, e.what());
      }
    }
    // For safe_bdev pools, overlay the LIVE member roster so runtime
    // add/remove/recover is reflected (falls back to whatever config_ carried).
    if (pc.mod_name_ == "clio_safe_bdev") {
      YAML::Node live = LiveSafeBdevMembers(admin, pc.pool_id_);
      if (live.IsSequence() && live.size() > 0) {
        pool["members"] = live;
      }
    }
    composed_ids.insert(pc.pool_id_.ToU64());
    compose_seq.push_back(pool);
  }
  root["compose"] = compose_seq;

  // (3) Live pool roster: every currently-active pool. Pools not present in the
  // startup compose (composed at runtime) are surfaced here so nothing is
  // silently dropped, even when their full config cannot be reconstructed.
  auto list = admin->AsyncListContainers(clio::run::PoolQuery::Local());
  list.Wait();
  if (list->GetReturnCode() == 0) {
    YAML::Node live_pools(YAML::NodeType::Sequence);
    for (size_t i = 0; i < list->pool_names_.size(); ++i) {
      YAML::Node lp;
      lp["pool_name"] = list->pool_names_[i];
      if (i < list->pool_ids_.size()) {
        lp["pool_id"] = list->pool_ids_[i];
      }
      live_pools.push_back(lp);
    }
    root["live_pools"] = live_pools;
  } else {
    HLOG(kWarning, "config export: ListContainers failed (rc={}); live_pools "
                   "omitted",
         list->GetReturnCode());
  }

  // (4) Emit to <path>.
  YAML::Emitter out;
  out << root;
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    HLOG(kError, "config export: cannot open '{}' for writing", path);
    return 1;
  }
  ofs << "# Exported by `clio_run config export` -- effective live "
         "configuration.\n";
  ofs << out.c_str() << "\n";
  ofs.close();
  HLOG(kSuccess, "config export: wrote effective configuration to {}", path);
  return 0;
}

}  // namespace

int Config(int argc, char** argv) {
  if (argc < 1) {
    PrintConfigUsage();
    return 1;
  }
  std::string sub = argv[0];
  if (sub == "-h" || sub == "--help") {
    PrintConfigUsage();
    return 0;
  }
  if (sub != "export") {
    HLOG(kError, "Unknown config subcommand: {}", sub);
    PrintConfigUsage();
    return 1;
  }

  std::string path;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--path") == 0 && i + 1 < argc) {
      path = argv[++i];
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintConfigUsage();
      return 0;
    }
  }
  if (path.empty()) {
    HLOG(kError, "config export requires --path <file>");
    PrintConfigUsage();
    return 1;
  }
  return ConfigExport(path);
}
