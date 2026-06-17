#include <cstring>
#include <iostream>
#include <string>

#include "clio_run_commands.h"

namespace {
// Program name to display in usage messages: derived from argv[0]'s basename
// so the symlink invocation `clio_run` prints "Usage: clio_run ..." while
// the legacy `chimaera` invocation keeps the historical text. Set by main().
const char* g_progname = "clio_run";

void PrintUsage() {
  std::cerr << "Usage: " << g_progname << " <command> [options]\n"
            << "\n"
            << "Commands:\n"
            << "  start           Start the Clio runtime server\n"
            << "  restart         Restart the Clio runtime (WAL replay)\n"
            << "  stop            Stop the Clio runtime server\n"
            << "  migrate         Migrate a container to a different node\n"
            << "  monitor         Monitor worker statistics\n"
            << "  compose         Manage pools from a compose config:\n"
            << "                  compose <start|stop|rm|list> (see compose --help)\n"
            << "  refresh         Autogenerate ChiMod method files\n"
            << "\n"
            << "Legacy nested forms (still supported):\n"
            << "  runtime <start|restart|stop>\n"
            << "  repo refresh\n"
            << "\n"
            << "Run '" << g_progname
            << " <command> --help' for more information on a command.\n";
}
}  // namespace

int main(int argc, char* argv[]) {
  // Set the displayed program name from argv[0]'s basename so usage messages
  // match whichever symlink (clio_run or chimaera) the user invoked.
  if (argc > 0 && argv[0] != nullptr && argv[0][0] != '\0') {
    const char* base = std::strrchr(argv[0], '/');
    g_progname = base ? base + 1 : argv[0];
  }

  if (argc < 2) {
    PrintUsage();
    return 1;
  }

  std::string cmd = argv[1];

  if (cmd == "--help" || cmd == "-h") {
    PrintUsage();
    return 0;
  }

  // Flat dispatch (canonical form):
  //   <progname> start | restart | stop | refresh | migrate | monitor | compose
  // Strip "<progname> <cmd>" from argv. Each handler sees only its own args.
  {
    int new_argc = argc - 2;
    char** new_argv = argv + 2;

    if (cmd == "start") {
      return RuntimeStart(new_argc, new_argv);
    } else if (cmd == "restart") {
      return RuntimeRestart(new_argc, new_argv);
    } else if (cmd == "stop") {
      return RuntimeStop(new_argc, new_argv);
    } else if (cmd == "refresh") {
      return RefreshRepo(new_argc, new_argv);
    } else if (cmd == "migrate") {
      return Migrate(new_argc, new_argv);
    } else if (cmd == "monitor") {
      return Monitor(new_argc, new_argv);
    } else if (cmd == "compose") {
      return Compose(new_argc, new_argv);
    }
  }

  // Legacy nested forms (kept working for backward compat):
  //   <progname> runtime <start|restart|stop>
  //   <progname> repo refresh
  if (cmd == "runtime") {
    if (argc < 3) {
      std::cerr << "Usage: " << g_progname
                << " runtime <start|restart|stop> [options]\n"
                << "Hint: the canonical flat form is `" << g_progname
                << " <start|restart|stop>`.\n";
      return 1;
    }

    std::string subcmd = argv[2];
    // Strip "<progname> runtime <subcmd>" from argv
    int new_argc = argc - 3;
    char** new_argv = argv + 3;

    if (subcmd == "start") {
      return RuntimeStart(new_argc, new_argv);
    } else if (subcmd == "restart") {
      return RuntimeRestart(new_argc, new_argv);
    } else if (subcmd == "stop") {
      return RuntimeStop(new_argc, new_argv);
    } else {
      std::cerr << "Unknown runtime subcommand: " << subcmd << "\n";
      std::cerr << "Usage: " << g_progname
                << " runtime <start|restart|stop> [options]\n";
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
