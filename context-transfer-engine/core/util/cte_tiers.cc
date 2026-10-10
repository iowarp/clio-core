/**
 * cte_tiers — show where the CTE's bytes are: one line per storage target.
 *
 * For every registered target prints its score and remaining space (see
 * cte_tiers_report.h). Answers "did my data land on the tier I scored
 * highest?" without reading the runtime log.
 *
 * Usage:
 *   cte_tiers
 *
 * Environment:
 *   CLIO_SERVER_CONF — path to runtime YAML config (as for cte_search)
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

#include <chrono>
#include <iostream>
#include <thread>

#include "cte_tiers_report.h"

int main() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    std::cerr << "error: failed to initialize Clio runtime\n";
    return 1;
  }
  struct Finalizer {
    ~Finalizer() {
      auto *mgr = CLIO_RUNTIME_MANAGER;
      if (mgr != nullptr) { mgr->ClientFinalize(); }
    }
  } finalizer;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::cerr << "error: failed to initialize CTE client\n";
    return 1;
  }
  return clio::cte::core::util::WriteTierReport(CLIO_CTE_CLIENT, std::cout);
}
