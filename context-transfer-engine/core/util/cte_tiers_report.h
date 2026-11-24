/**
 * WriteTierReport — the body of `cte_tiers`: one line per CTE storage target
 * with its score and remaining space.
 *
 * Kept out of cte_tiers.cc so a unit test can call it against an in-process
 * runtime (test_cte_tiers); the executable is only CLIO_INIT plus this call.
 */

#ifndef CLIO_CTE_CORE_UTIL_CTE_TIERS_REPORT_H_
#define CLIO_CTE_CORE_UTIL_CTE_TIERS_REPORT_H_

#include <clio_cte/core/core_client.h>

#include <cstdio>
#include <ostream>
#include <string>

namespace clio::cte::core::util {

/**
 * Write the tier report for every target the local CTE pool has registered.
 *
 * Free space is refreshed (StatTargets) before it is read, so it is current.
 * GetTargetInfo's bytes_written_ / bytes_read_ are not printed: they read 0 on
 * targets that demonstrably hold data, so free space is the trustworthy
 * signal of where a workload's bytes landed.
 *
 * @param client CTE client of the pool to report on
 * @param out    stream the report is written to
 * @return 0 on success, 1 if the target list could not be read
 */
inline int WriteTierReport(Client *client, std::ostream &out) {
  // GetTargetInfo answers from a per-target snapshot that the periodic
  // StatTargets sweep refreshes every few seconds; refresh it now so the
  // report reflects the puts that just happened, not the last sweep.
  auto stat = client->AsyncStatTargets(clio::run::PoolQuery::Local());
  stat.Wait();
  auto list = client->AsyncListTargets(clio::run::PoolQuery::Local());
  list.Wait();
  if (list->GetReturnCode() != 0) {
    out << "error: ListTargets failed (rc " << list->GetReturnCode() << ")\n";
    return 1;
  }
  char line[512];
  std::snprintf(line, sizeof(line), "%-60s %6s %14s\n", "target", "score",
                "free_MB");
  out << line;
  for (const std::string &name : list->target_names_) {
    auto info =
        client->AsyncGetTargetInfo(name, clio::run::PoolQuery::Local());
    info.Wait();
    if (info->GetReturnCode() != 0) {
      std::snprintf(line, sizeof(line), "%-60s (GetTargetInfo rc %u)\n",
                    name.c_str(), info->GetReturnCode());
    } else {
      std::snprintf(line, sizeof(line), "%-60s %6.2f %14.1f\n", name.c_str(),
                    info->target_score_, info->remaining_space_ / 1e6);
    }
    out << line;
  }
  return 0;
}

}  // namespace clio::cte::core::util

#endif  // CLIO_CTE_CORE_UTIL_CTE_TIERS_REPORT_H_
