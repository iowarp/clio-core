#include "chimaera_commands.h"

int main(int argc, char* argv[]) {
  // Thin wrapper: forward to RuntimeStop with args after program name
  return RuntimeStop(argc - 1, argv + 1);
}
