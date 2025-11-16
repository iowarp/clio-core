#ifndef CHIMAERA_INCLUDE_CHIMAERA_CHIMAERA_H_
#define CHIMAERA_INCLUDE_CHIMAERA_CHIMAERA_H_

/**
 * Main header file for Chimaera distributed task execution framework
 *
 * This header provides the primary interface for both runtime and client
 * applications using the Chimaera framework.
 */

#include "chimaera/pool_query.h"
#include "chimaera/singletons.h"
#include "chimaera/task.h"
#include "chimaera/task_archives.h"
#include "chimaera/types.h"
#include "chimaera/worker.h"

namespace chi {

/**
 * Chimaera initialization mode
 */
enum class ChimaeraMode {
  kClient,   /**< Client mode - connects to existing runtime */
  kServer,   /**< Server mode - starts runtime components */
  kRuntime = kServer  /**< Alias for kServer */
};

/**
 * Global initialization functions
 */

/**
 * Initialize Chimaera with specified mode
 *
 * @param mode Initialization mode (kClient or kServer/kRuntime)
 * @param default_with_runtime Default behavior if CHIMAERA_WITH_RUNTIME env var not set
 *        If true, will start runtime in addition to client initialization
 *        If false, will only initialize client components
 * @return true if initialization successful, false otherwise
 *
 * Environment variable:
 *   CHIMAERA_WITH_RUNTIME=1 - Start runtime regardless of mode
 *   CHIMAERA_WITH_RUNTIME=0 - Don't start runtime (client only)
 *   If not set, uses default_with_runtime parameter
 */
bool CHIMAERA_INIT(ChimaeraMode mode, bool default_with_runtime = false);

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_CHIMAERA_H_