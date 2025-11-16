/**
 * Main Chimaera initialization and global functions
 */

#include "chimaera/chimaera.h"
#include "chimaera/container.h"
#include "chimaera/work_orchestrator.h"
#include <cstdlib>
#include <cstring>

namespace chi {

bool CHIMAERA_INIT(ChimaeraMode mode, bool default_with_runtime) {
  // Static guard to prevent double initialization
  static bool s_initialized = false;
  if (s_initialized) {
    return true;  // Already initialized, return success
  }

  auto* chimaera_manager = CHI_CHIMAERA_MANAGER;

  // Check environment variable CHIMAERA_WITH_RUNTIME
  bool with_runtime = default_with_runtime;
  const char* env_val = std::getenv("CHIMAERA_WITH_RUNTIME");
  if (env_val != nullptr) {
    // Environment variable overrides default
    with_runtime = (std::strcmp(env_val, "1") == 0 ||
                   std::strcmp(env_val, "true") == 0 ||
                   std::strcmp(env_val, "TRUE") == 0);
  }

  // Determine what to initialize based on mode and with_runtime flag
  bool init_runtime = false;
  bool init_client = false;

  if (mode == ChimaeraMode::kServer || mode == ChimaeraMode::kRuntime) {
    // Server/Runtime mode: always start runtime
    init_runtime = true;
    init_client = true;  // Runtime also needs client components
  } else {
    // Client mode
    init_client = true;
    init_runtime = with_runtime;
  }

  // Initialize runtime first if needed
  if (init_runtime) {
    if (!chimaera_manager->ServerInit()) {
      return false;
    }
  }

  // Initialize client components
  if (init_client) {
    if (!chimaera_manager->ClientInit()) {
      return false;
    }
  }

  // Mark as initialized on success
  s_initialized = true;
  return true;
}

}  // namespace chi