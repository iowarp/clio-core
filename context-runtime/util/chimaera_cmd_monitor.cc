#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <thread>
#include <iomanip>

#include "chimaera/chimaera.h"
#include "chimaera/singletons.h"
#include "chimaera/types.h"
#include "chimaera/admin/admin_client.h"
#include "chimaera_commands.h"

namespace {

volatile bool g_monitor_running = true;

struct MonitorOptions {
  int interval_sec = 1;
  bool once = false;
  bool json_output = false;
  bool verbose = false;
};

void PrintMonitorUsage() {
  HIPRINT("Usage: chimaera monitor [OPTIONS]");
  HIPRINT("");
  HIPRINT("Options:");
  HIPRINT("  -h, --help        Show this help message");
  HIPRINT("  -i, --interval N  Set monitoring interval in seconds (default: 1)");
  HIPRINT("  -o, --once        Run once and exit (default: continuous monitoring)");
  HIPRINT("  -j, --json        Output raw JSON format");
  HIPRINT("  -v, --verbose     Enable verbose output");
}

bool ParseMonitorArgs(int argc, char* argv[], MonitorOptions& opts) {
  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      PrintMonitorUsage();
      return false;
    } else if (arg == "-i" || arg == "--interval") {
      if (i + 1 < argc) {
        opts.interval_sec = std::atoi(argv[++i]);
        if (opts.interval_sec < 1) {
          HLOG(kError, "Interval must be >= 1 second");
          return false;
        }
      } else {
        HLOG(kError, "-i/--interval requires an argument");
        return false;
      }
    } else if (arg == "-o" || arg == "--once") {
      opts.once = true;
    } else if (arg == "-j" || arg == "--json") {
      opts.json_output = true;
    } else if (arg == "-v" || arg == "--verbose") {
      opts.verbose = true;
    } else {
      HLOG(kError, "Unknown option: {}", arg);
      PrintMonitorUsage();
      return false;
    }
  }
  return true;
}

void PrintStats(const chimaera::admin::MonitorTask& task) {
  HIPRINT("\033[2J\033[H");

  auto now = std::chrono::system_clock::now();
  auto now_t = std::chrono::system_clock::to_time_t(now);
  std::ostringstream time_ss;
  time_ss << std::put_time(std::localtime(&now_t), "%Y-%m-%d %H:%M:%S");
  HIPRINT("==================================================");
  HIPRINT("        Chimaera Worker Monitor");
  HIPRINT("        {}", time_ss.str());
  HIPRINT("==================================================");
  HIPRINT("");

  chi::u32 total_queued = 0;
  chi::u32 total_blocked = 0;
  chi::u32 total_periodic = 0;

  for (const auto& stats : task.info_) {
    total_queued += stats.num_queued_tasks_;
    total_blocked += stats.num_blocked_tasks_;
    total_periodic += stats.num_periodic_tasks_;
  }

  HIPRINT("Summary:");
  HIPRINT("  Total Workers:        {}", task.info_.size());
  HIPRINT("  Total Queued Tasks:   {}", total_queued);
  HIPRINT("  Total Blocked Tasks:  {}", total_blocked);
  HIPRINT("  Total Periodic Tasks: {}", total_periodic);
  HIPRINT("");

  std::ostringstream header;
  header << std::setw(6) << "ID"
         << std::setw(10) << "Running"
         << std::setw(10) << "Active"
         << std::setw(12) << "Idle Iters"
         << std::setw(10) << "Queued"
         << std::setw(10) << "Blocked"
         << std::setw(10) << "Periodic"
         << std::setw(15) << "Suspend (us)";
  HIPRINT("Worker Details:");
  HIPRINT("{}", header.str());
  HIPRINT("{}", std::string(83, '-'));

  for (const auto& stats : task.info_) {
    std::ostringstream row;
    row << std::setw(6) << stats.worker_id_
        << std::setw(10) << (stats.is_running_ ? "Yes" : "No")
        << std::setw(10) << (stats.is_active_ ? "Yes" : "No")
        << std::setw(12) << stats.idle_iterations_
        << std::setw(10) << stats.num_queued_tasks_
        << std::setw(10) << stats.num_blocked_tasks_
        << std::setw(10) << stats.num_periodic_tasks_
        << std::setw(15) << stats.suspend_period_us_;
    HIPRINT("{}", row.str());
  }

  HIPRINT("");
  HIPRINT("Press Ctrl+C to exit");
}

}  // namespace

int Monitor(int argc, char* argv[]) {
  MonitorOptions opts;
  if (!ParseMonitorArgs(argc, argv, opts)) {
    return 0;
  }

  if (opts.verbose) {
    HLOG(kInfo, "Initializing Chimaera client...");
  }

  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Chimaera client");
    HLOG(kError, "Make sure the Chimaera runtime is running");
    return 1;
  }

  HLOG(kInfo, "Getting admin client...");
  auto* admin_client = CHI_ADMIN;
  if (!admin_client) {
    HLOG(kError, "Failed to get admin client");
    return 1;
  }

  HLOG(kInfo, "Admin client obtained successfully");

  while (g_monitor_running) {
    try {
      HLOG(kInfo, "Sending AsyncMonitor request...");
      auto future = admin_client->AsyncMonitor(chi::PoolQuery::Local());
      future.Wait();

      if (future->GetReturnCode() != 0) {
        HLOG(kError, "Monitor task failed with return code {}",
             future->GetReturnCode());
        break;
      }

      if (opts.json_output) {
        std::ostringstream json;
        json << "{\"workers\":[";
        bool first = true;
        for (const auto& stats : future->info_) {
          if (!first) json << ",";
          first = false;
          json << "{"
               << "\"worker_id\":" << stats.worker_id_ << ","
               << "\"is_running\":" << (stats.is_running_ ? "true" : "false") << ","
               << "\"is_active\":" << (stats.is_active_ ? "true" : "false") << ","
               << "\"idle_iterations\":" << stats.idle_iterations_ << ","
               << "\"num_queued_tasks\":" << stats.num_queued_tasks_ << ","
               << "\"num_blocked_tasks\":" << stats.num_blocked_tasks_ << ","
               << "\"num_periodic_tasks\":" << stats.num_periodic_tasks_ << ","
               << "\"suspend_period_us\":" << stats.suspend_period_us_ << "}";
        }
        json << "]}";
        HIPRINT("{}", json.str());
      } else {
        PrintStats(*future);
      }

      if (opts.once) break;

      for (int i = 0; i < opts.interval_sec && g_monitor_running; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

    } catch (const std::exception& e) {
      HLOG(kError, "Exception during monitoring: {}", e.what());
      break;
    }
  }

  return 0;
}
