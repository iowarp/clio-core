#include <cstring>
#include <iostream>
#include <string>

#include "chimaera_commands.h"

namespace {
void PrintUsage() {
  std::cerr << "Usage: chimaera <command> [options]\n"
            << "\n"
            << "Commands:\n"
            << "  runtime start   Start the Chimaera runtime server\n"
            << "  runtime stop    Stop the Chimaera runtime server\n"
            << "  migrate         Migrate a container to a different node\n"
            << "  monitor         Monitor worker statistics\n"
            << "  compose         Create/destroy pools from compose config\n"
            << "  repo refresh    Autogenerate ChiMod method files\n"
            << "\n"
            << "Run 'chimaera <command> --help' for more information on a command.\n";
}
}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    PrintUsage();
    return 1;
  }

  std::string cmd = argv[1];

  if (cmd == "--help" || cmd == "-h") {
    PrintUsage();
    return 0;
  }

  // Handle "runtime start" and "runtime stop" subcommands
  if (cmd == "runtime") {
    if (argc < 3) {
      std::cerr << "Usage: chimaera runtime <start|stop> [options]\n";
      return 1;
    }

    std::string subcmd = argv[2];
    // Strip "chimaera runtime <subcmd>" from argv
    int new_argc = argc - 3;
    char** new_argv = argv + 3;

    if (subcmd == "start") {
      return RuntimeStart(new_argc, new_argv);
    } else if (subcmd == "stop") {
      return RuntimeStop(new_argc, new_argv);
    } else {
      std::cerr << "Unknown runtime subcommand: " << subcmd << "\n";
      std::cerr << "Usage: chimaera runtime <start|stop> [options]\n";
      return 1;
    }
  }

  // Handle "repo refresh" subcommand
  if (cmd == "repo") {
    if (argc < 3) {
      std::cerr << "Usage: chimaera repo <refresh> [options]\n";
      return 1;
    }

    std::string subcmd = argv[2];
    // Strip "chimaera repo <subcmd>" from argv
    int new_argc = argc - 3;
    char** new_argv = argv + 3;

    if (subcmd == "refresh") {
      return RefreshRepo(new_argc, new_argv);
    } else {
      std::cerr << "Unknown repo subcommand: " << subcmd << "\n";
      std::cerr << "Usage: chimaera repo <refresh> [options]\n";
      return 1;
    }
  }

  // Strip "chimaera <cmd>" from argv
  int new_argc = argc - 2;
  char** new_argv = argv + 2;

  if (cmd == "migrate") {
    return Migrate(new_argc, new_argv);
  } else if (cmd == "monitor") {
    return Monitor(new_argc, new_argv);
  } else if (cmd == "compose") {
    return Compose(new_argc, new_argv);
  } else {
    std::cerr << "Unknown command: " << cmd << "\n";
    PrintUsage();
    return 1;
  }
}
