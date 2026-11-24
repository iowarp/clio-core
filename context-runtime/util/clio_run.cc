#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "clio_ctp/introspect/system_info.h"
#include "clio_run_commands.h"

// Injected by CMake from project(iowarp-core VERSION ...) through the
// clio_core_version interface target; the fallback only covers a hand build.
#ifndef CLIO_CORE_VERSION_STRING
#define CLIO_CORE_VERSION_STRING "unknown"
#endif

namespace {
// Program name to display in usage messages: derived from argv[0]'s basename
// so the canonical `clio-run` and the legacy alias `clio_run` each print their
// own name. Set by main().
const char* g_progname = "clio-run";

void PrintUsage() {
  std::cerr << "Usage: " << g_progname << " [--local] <command> [options]\n"
            << "\n"
            << "Global options:\n"
            << "  --local         Single-host mode: bind and connect on\n"
            << "                  127.0.0.1 only (sets CLIO_BIND_ADDR and\n"
            << "                  CLIO_SERVER_ADDR). Never listens on 0.0.0.0,\n"
            << "                  so no firewall prompt. Accepted anywhere.\n"
            << "\n"
            << "Commands:\n"
            << "  start           Start the Clio runtime server; it recovers\n"
            << "                  all persistent state (--fresh: start empty)\n"
            << "  stop            Stop the Clio runtime server\n"
            << "                  (--force: immediate ungraceful stop;\n"
            << "                   --grace-period <ms>: drain budget)\n"
            << "  status          Report runtime state (running/stopped/stale)\n"
            << "  migrate         Migrate a container to a different node\n"
            << "  monitor         Monitor worker statistics\n"
            << "  compose         Manage pools from a compose config:\n"
            << "                  compose <start|stop|rm|list> (see compose --help)\n"
            << "  pool            Manage one pool of the running runtime:\n"
            << "                  pool <start|stop|rm> <pool> [--force]\n"
            << "                  (see pool --help)\n"
            << "  config          Export node configuration:\n"
            << "                  config export --path <file>\n"
            << "  refresh         Autogenerate ChiMod method files\n"
            << "  viz             Control the web dashboard:\n"
            << "                  viz <start|stop|status> (see viz --help)\n"
            << "  version         Print the version (also --version, -V)\n"
            << "\n"
            << "Legacy nested forms (still supported):\n"
            << "  runtime <start|stop>\n"
            << "  repo refresh\n"
            << "\n"
            << "Run '" << g_progname
            << " <command> --help' for more information on a command.\n"
            << "(clio_run is a legacy alias of clio-run.)\n";
}
/**
 * `restart` was removed: `start` always recovers persistent state now.
 * @return 1, after saying what to run instead
 */
int RestartRemoved() {
  std::cerr << "`" << g_progname << " restart` was removed: `" << g_progname
            << " start` always recovers persistent state. Use `" << g_progname
            << " start` (recover) or `" << g_progname
            << " start --fresh` (discard this node's state).\n";
  return 1;
}

/**
 * Remove every `--local` from the argument list and, if any was present,
 * pin the runtime and its clients to loopback.
 *
 * Sets CLIO_BIND_ADDR (what every server socket binds) and CLIO_SERVER_ADDR
 * (what clients connect to) to 127.0.0.1 in this process's environment, so a
 * runtime started from here -- in process or as a spawned daemon, which
 * inherits the environment -- never listens on 0.0.0.0. A single-host
 * runtime then needs no hostfile and trips no firewall prompt. Loopback
 * binding also makes the runtime ignore any configured hostfile (see
 * IpcManager's single-node path).
 *
 * @param argc in/out: argument count; reduced by the number of flags removed
 * @param argv in/out: argument vector, compacted in place (argv[0] kept)
 * @return true if `--local` was given
 */
bool ConsumeLocalFlag(int &argc, char *argv[]) {
  int out = 0;
  bool local = false;
  for (int i = 0; i < argc; ++i) {
    if (i > 0 && argv[i] != nullptr && std::strcmp(argv[i], "--local") == 0) {
      local = true;
      continue;
    }
    argv[out++] = argv[i];
  }
  argc = out;
  if (local) {
    ctp::SystemInfo::Setenv("CLIO_BIND_ADDR", std::string("127.0.0.1"), 1);
    ctp::SystemInfo::Setenv("CLIO_SERVER_ADDR", std::string("127.0.0.1"), 1);
  }
  return local;
}
}  // namespace

int main(int argc, char* argv[]) {
  // Set the displayed program name from argv[0]'s basename so usage messages
  // match whichever symlink (clio_run or clio) the user invoked.
  if (argc > 0 && argv[0] != nullptr && argv[0][0] != '\0') {
    const char* base = std::strrchr(argv[0], '/');
    g_progname = base ? base + 1 : argv[0];
  }

  // Global, position-independent: strip it before dispatch so no subcommand
  // parser sees it.
  ConsumeLocalFlag(argc, argv);

  if (argc < 2) {
    PrintUsage();
    return 1;
  }

  std::string cmd = argv[1];

  if (cmd == "--help" || cmd == "-h") {
    PrintUsage();
    return 0;
  }

  if (cmd == "--version" || cmd == "-V" || cmd == "version") {
    std::cout << g_progname << " " << CLIO_CORE_VERSION_STRING << "\n";
    return 0;
  }

  // Flat dispatch (canonical form):
  //   <progname> start | stop | refresh | migrate | monitor | compose | pool
  // Strip "<progname> <cmd>" from argv. Each handler sees only its own args.
  {
    int new_argc = argc - 2;
    char** new_argv = argv + 2;

    if (cmd == "start") {
      return RuntimeStart(new_argc, new_argv);
    } else if (cmd == "restart") {
      return RestartRemoved();
    } else if (cmd == "stop") {
      return RuntimeStop(new_argc, new_argv);
    } else if (cmd == "pool") {
      return Pool(new_argc, new_argv);
    } else if (cmd == "status") {
      return RuntimeStatus(new_argc, new_argv);
    } else if (cmd == "refresh") {
      return RefreshRepo(new_argc, new_argv);
    } else if (cmd == "migrate") {
      return Migrate(new_argc, new_argv);
    } else if (cmd == "monitor") {
      return Monitor(new_argc, new_argv);
    } else if (cmd == "compose") {
      return Compose(new_argc, new_argv);
    } else if (cmd == "config") {
      return Config(new_argc, new_argv);
    } else if (cmd == "viz") {
      return Viz(new_argc, new_argv);
    }
  }

  // Legacy nested forms (kept working for backward compat):
  //   <progname> runtime <start|stop>
  //   <progname> repo refresh
  if (cmd == "runtime") {
    if (argc < 3) {
      std::cerr << "Usage: " << g_progname
                << " runtime <start|stop> [options]\n"
                << "Hint: the canonical flat form is `" << g_progname
                << " <start|stop>`.\n";
      return 1;
    }

    std::string subcmd = argv[2];
    // Strip "<progname> runtime <subcmd>" from argv
    int new_argc = argc - 3;
    char** new_argv = argv + 3;

    if (subcmd == "start") {
      return RuntimeStart(new_argc, new_argv);
    } else if (subcmd == "restart") {
      return RestartRemoved();
    } else if (subcmd == "stop") {
      return RuntimeStop(new_argc, new_argv);
    } else if (subcmd == "status") {
      return RuntimeStatus(new_argc, new_argv);
    } else {
      std::cerr << "Unknown runtime subcommand: " << subcmd << "\n";
      std::cerr << "Usage: " << g_progname
                << " runtime <start|stop> [options]\n";
      return 1;
    }
  }

  if (cmd == "repo") {
    if (argc < 3) {
      std::cerr << "Usage: " << g_progname << " repo <refresh> [options]\n"
                << "Hint: the canonical flat form is `" << g_progname
                << " refresh`.\n";
      return 1;
    }

    std::string subcmd = argv[2];
    // Strip "<progname> repo <subcmd>" from argv
    int new_argc = argc - 3;
    char** new_argv = argv + 3;

    if (subcmd == "refresh") {
      return RefreshRepo(new_argc, new_argv);
    } else {
      std::cerr << "Unknown repo subcommand: " << subcmd << "\n";
      std::cerr << "Usage: " << g_progname << " repo <refresh> [options]\n";
      return 1;
    }
  }

  std::cerr << "Unknown command: " << cmd << "\n";
  PrintUsage();
  return 1;
}
