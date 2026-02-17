#include "chimaera_commands.h"

int main(int argc, char* argv[]) {
  // Thin wrapper: forward to RuntimeStart with args after program name
  return RuntimeStart(argc - 1, argv + 1);
}
