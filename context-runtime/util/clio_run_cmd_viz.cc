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
 * `clio-run viz <start|stop|status>`: control the local runtime's web
 * dashboard without restarting the runtime.
 *
 * The dashboard lives inside the runtime process, so every action is an admin
 * Monitor query ("viz_start" / "viz_stop" / "viz_status") answered by the
 * admin container on this node; see Runtime::MonitorViz.
 */

#include <clio_ctp/serialize/msgpack_wrapper.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "clio_ctp/introspect/system_info.h"
#include "clio_runtime/admin/admin_client.h"
#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/pool_query.h"
#include "clio_runtime/types.h"
#include "clio_run_commands.h"
#include "clio_run_local_runtime.h"

namespace {

void PrintVizCmdUsage() {
  HIPRINT("Usage: clio-run viz <start|stop|status> [options]");
  HIPRINT("  Control the web dashboard of the Clio runtime on this node.");
  HIPRINT("");
  HIPRINT("  start   Start the dashboard unless it is already running, then");
  HIPRINT("          print its address.");
  HIPRINT("            --port <n>     TCP port (0 = any free port)");
  HIPRINT("            --bind <addr>  bind address: loopback, or the address");
  HIPRINT("                           the runtime's configuration names");
  HIPRINT("          Without options the runtime's configured endpoint is used");
  HIPRINT("          (viz: port/bind, CLIO_VIZ_PORT/CLIO_VIZ_BIND; default");
  HIPRINT("          127.0.0.1:8080).");
  HIPRINT("  stop    Stop the dashboard. The runtime keeps running.");
  HIPRINT("  status  Print the dashboard's address, or that it is not running.");
  HIPRINT("");
  HIPRINT("  Exit codes: 0 = dashboard running (start/status) or stopped (stop);");
  HIPRINT("              1 = not running / failed; 2 = runtime not reachable.");
}

/** Dashboard state as reported by the runtime. */
struct VizState {
  bool running = false;
  std::string bind;
  clio::run::u32 port = 0;
  std::string error;
};

/** Decode Runtime::MonitorViz's reply. @return false if no reply decoded. */
bool DecodeVizState(const clio::run::admin::MonitorTask &task, VizState *out) {
  for (const auto &[container_id, blob] : task.results_) {
    (void)container_id;
    if (blob.empty()) {
      continue;
    }
    try {
      msgpack::object_handle oh = msgpack::unpack(blob.data(), blob.size());
      const msgpack::object &obj = oh.get();
      if (obj.type != msgpack::type::MAP) {
        continue;
      }
      for (uint32_t i = 0; i < obj.via.map.size; ++i) {
        const auto &kv = obj.via.map.ptr[i];
        std::string key;
        kv.key.convert(key);
        if (key == "running") kv.val.convert(out->running);
        else if (key == "bind") kv.val.convert(out->bind);
        else if (key == "port") kv.val.convert(out->port);
        else if (key == "error") kv.val.convert(out->error);
      }
      return true;
    } catch (const std::exception &) {
      continue;
    }
  }
  return false;
}

/** The URL an operator should open. A wildcard bind is reachable on every
 *  interface, so name this machine instead of printing 0.0.0.0. The query is
 *  node-local (PoolQuery::Local), so the client's hostname IS the runtime's;
 *  the runtime's own idea of its hostname can be a bind address. */
std::string VizUrl(const VizState &st) {
  std::string host = st.bind;
  if (host.empty() || host == "0.0.0.0" || host == "::") {
    host = ctp::SystemInfo::GetHostname();
    if (host.empty()) {
      host = "localhost";
    }
  }
  if (host.find(':') != std::string::npos) {
    host = "[" + host + "]";  // IPv6 literal
  }
  return "http://" + host + ":" + std::to_string(st.port) + "/";
}

}  // namespace

int Viz(int argc, char *argv[]) {
  if (argc < 1 || std::strcmp(argv[0], "--help") == 0 ||
      std::strcmp(argv[0], "-h") == 0) {
    PrintVizCmdUsage();
    return argc < 1 ? 1 : 0;
  }
  const std::string action = argv[0];
  if (action != "start" && action != "stop" && action != "status") {
    HLOG(kError, "Unknown viz subcommand: {}", action);
    PrintVizCmdUsage();
    return 1;
  }

  std::string query = "viz_" + action;
  std::string start_args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintVizCmdUsage();
      return 0;
    }
    if (action == "start" && (arg == "--port" || arg == "--bind") &&
        i + 1 < argc) {
      std::string val = argv[++i];
      bool bad = val.empty() || val.find_first_of("&?=") != std::string::npos;
      if (!bad && arg == "--port") {
        char *end = nullptr;
        unsigned long port = std::strtoul(val.c_str(), &end, 10);
        bad = *end != '\0' || port > 65535;
      }
      if (bad) {
        HLOG(kError, "Invalid value for {}: '{}'", arg, val);
        return 1;
      }
      start_args += (start_args.empty() ? "" : "&") + arg.substr(2) + "=" + val;
      continue;
    }
    HLOG(kError, "Unknown or incomplete option for viz {}: {}", action, arg);
    PrintVizCmdUsage();
    return 1;
  }
  if (!start_args.empty()) {
    query += "?" + start_args;
  }

  // Never leave this process's own IPC entries behind (see RuntimeStatus).
  struct OwnEntriesGuard {
    ~OwnEntriesGuard() {
      if (auto *ipc = CLIO_IPC) {
        ipc->UnlinkOwnPidEntries();
      }
    }
  } own_entries_guard;

  // Answer promptly when no runtime is up. An explicit CLIO_WAIT_SERVER wins.
  if (!clio::run::env::GetCompat("WAIT_SERVER")) {
#ifdef _WIN32
    _putenv_s("CLIO_WAIT_SERVER", "2");
#else
    setenv("CLIO_WAIT_SERVER", "2", 1);
#endif
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    // Resolve the port only now: the config is loaded by the init attempt.
    HIPRINT("viz: not running (no Clio runtime on port {}; start one with "
            "`clio-run start`)", clio::run::cli::ResolveRuntimePort());
    return 2;
  }
  const clio::run::u32 rt_port = clio::run::cli::ResolveRuntimePort();
  struct ClientFinalizeGuard {
    ~ClientFinalizeGuard() {
      if (auto *mgr = CLIO_RUNTIME_MANAGER) {
        mgr->ClientFinalize();
      }
    }
  } finalize_guard;

  VizState st;
  try {
    clio::run::admin::Client admin(clio::run::kAdminPoolId);
    auto task = admin.AsyncMonitor(clio::run::PoolQuery::Local(), query);
    // Generous: a stop waits up to the server's stop budget (3 s default).
    if (task.IsNull() || !task.Wait(15.0f) || task->GetReturnCode() != 0 ||
        !DecodeVizState(*task, &st)) {
      HIPRINT("viz: no answer from the Clio runtime on port {} (is it an "
              "older runtime without viz control?)", rt_port);
      return 2;
    }
  } catch (const std::exception &e) {
    HIPRINT("viz: request failed: {}", e.what());
    return 2;
  }

  if (!st.error.empty()) {
    HLOG(kError, "viz {}: {}", action, st.error);
  }
  if (action == "stop") {
    if (st.running) {
      HIPRINT("viz: still running at {}", VizUrl(st));
      return 1;
    }
    HIPRINT("viz: stopped");
    return 0;
  }
  if (st.running) {
    HIPRINT("viz: running at {}", VizUrl(st));
    return 0;
  }
  HIPRINT("viz: not running");
  return 1;
}
