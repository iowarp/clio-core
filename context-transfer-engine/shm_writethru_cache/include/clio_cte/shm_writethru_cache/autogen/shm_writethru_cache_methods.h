#ifndef CLIO_CTE_SHM_WRITETHRU_CACHE_AUTOGEN_METHODS_H_
#define CLIO_CTE_SHM_WRITETHRU_CACHE_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/autogen/core_methods.h>

#include <string>
#include <vector>

/**
 * Method ids of the shm_writethru_cache chimod (see clio_mod.yaml).
 * Interposed core verbs reuse the core's ids and task structs.
 */
namespace clio::cte::shm_wtc {

namespace Method {
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

GLOBAL_CROSS_CONST clio::run::u32 kGetOrCreateTag =
    clio::cte::core::Method::kGetOrCreateTag;
GLOBAL_CROSS_CONST clio::run::u32 kPutBlob = clio::cte::core::Method::kPutBlob;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlob = clio::cte::core::Method::kGetBlob;
GLOBAL_CROSS_CONST clio::run::u32 kDelBlob = clio::cte::core::Method::kDelBlob;
GLOBAL_CROSS_CONST clio::run::u32 kDelTag = clio::cte::core::Method::kDelTag;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlobSize =
    clio::cte::core::Method::kGetBlobSize;
GLOBAL_CROSS_CONST clio::run::u32 kMultiPutBlob =
    clio::cte::core::Method::kMultiPutBlob;
GLOBAL_CROSS_CONST clio::run::u32 kSyncTag = clio::cte::core::Method::kSyncTag;

GLOBAL_CROSS_CONST clio::run::u32 kFlushLog = 100;
GLOBAL_CROSS_CONST clio::run::u32 kInvalidate = 101;
GLOBAL_CROSS_CONST clio::run::u32 kHomeFlush = 102;
GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 103;

/** Method names indexed by id. */
inline const std::vector<std::string> &GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[kCreate] = "Create";
    v[kDestroy] = "Destroy";
    v[kMonitor] = "Monitor";
    v[kGetOrCreateTag] = "GetOrCreateTag";
    v[kPutBlob] = "PutBlob";
    v[kGetBlob] = "GetBlob";
    v[kDelBlob] = "DelBlob";
    v[kDelTag] = "DelTag";
    v[kGetBlobSize] = "GetBlobSize";
    v[kMultiPutBlob] = "MultiPutBlob";
    v[kSyncTag] = "SyncTag";
    v[kFlushLog] = "FlushLog";
    v[kInvalidate] = "Invalidate";
    v[kHomeFlush] = "HomeFlush";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::shm_wtc

#endif  // CLIO_CTE_SHM_WRITETHRU_CACHE_AUTOGEN_METHODS_H_
