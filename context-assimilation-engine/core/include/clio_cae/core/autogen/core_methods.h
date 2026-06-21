#ifndef CLIO_CAE_CORE_AUTOGEN_METHODS_H_
#define CLIO_CAE_CORE_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for core
 */

namespace clio::cae::core {

namespace Method {
// Inherited methods
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// core-specific methods
GLOBAL_CROSS_CONST clio::run::u32 kParseOmni = 10;
GLOBAL_CROSS_CONST clio::run::u32 kProcessHdf5Dataset = 11;
GLOBAL_CROSS_CONST clio::run::u32 kExportData = 12;

// CTE interceptor methods. IDs MUST match clio::cte::core::Method::k*
// so that a CTE-built task (whose constructor stamps the CTE method id)
// dispatches to the matching CAE handler when routed to a CAE pool.
// Keep these in sync with context-transfer-engine/core/clio_mod.yaml.
GLOBAL_CROSS_CONST clio::run::u32 kGetOrCreateTag = 14;
GLOBAL_CROSS_CONST clio::run::u32 kPutBlob = 15;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlob = 16;
GLOBAL_CROSS_CONST clio::run::u32 kSemanticSearch = 35;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 36;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "ParseOmni";
    v[11] = "ProcessHdf5Dataset";
    v[12] = "ExportData";
    v[14] = "GetOrCreateTag";
    v[15] = "PutBlob";
    v[16] = "GetBlob";
    v[35] = "SemanticSearch";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cae::core

#endif  // CORE_AUTOGEN_METHODS_H_
