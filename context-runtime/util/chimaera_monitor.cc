#include "chimaera_commands.h"

int main(int argc, char* argv[]) {
  // Thin wrapper: forward to Monitor with args after program name
  return Monitor(argc - 1, argv + 1);
}
