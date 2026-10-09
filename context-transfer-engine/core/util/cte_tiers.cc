/**
 * cte_tiers — show where the CTE's bytes are: one line per storage target.
 *
 * For every registered target (ListTargets) prints its score and remaining
 * space (GetTargetInfo). Answers "did my data land on the tier I scored
 * highest?" without reading the runtime log. (GetTargetInfo's bytes_written_ /
 * bytes_read_ are not printed: they read 0 on targets that demonstrably hold
 * data, so free space is the trustworthy signal.)
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
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

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
  auto *client = CLIO_CTE_CLIENT;

  auto list = client->AsyncListTargets(clio::run::PoolQuery::Local());
  list.Wait();
  if (list->GetReturnCode() != 0) {
    std::cerr << "error: ListTargets failed (rc " << list->GetReturnCode()
              << ")\n";
    return 1;
  }
  std::printf("%-60s %6s %14s\n", "target", "score", "free_MB");
  for (const std::string &name : list->target_names_) {
    auto info =
        client->AsyncGetTargetInfo(name, clio::run::PoolQuery::Local());
    info.Wait();
    if (info->GetReturnCode() != 0) {
      std::printf("%-60s (GetTargetInfo rc %u)\n", name.c_str(),
                  info->GetReturnCode());
      continue;
    }
    std::printf("%-60s %6.2f %14.1f\n", name.c_str(), info->target_score_,
                info->remaining_space_ / 1e6);
  }
  return 0;
}
