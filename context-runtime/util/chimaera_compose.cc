#include "chimaera_commands.h"

int main(int argc, char** argv) {
  // Thin wrapper: forward to Compose with args after program name
  return Compose(argc - 1, argv + 1);
}
