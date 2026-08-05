/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_CACHE_AUTOGEN_METHODS_H_
#define CLIO_CTE_CACHE_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/autogen/core_methods.h>
#include <string>
#include <vector>

/**
 * Method ids for the cache chimod. Hand-maintained; same interposition rules
 * as the replication and compressor chimods: the core data verbs carry the
 * CORE's ids and task structs, every other core id is forwarded verbatim by
 * the dispatch defaults, and module verbs live ABOVE the core's id space.
 */
namespace clio::cte::cache {

namespace Method {
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// Interposed core data verbs — same ids, same task structs as the core.
GLOBAL_CROSS_CONST clio::run::u32 kPutBlob = clio::cte::core::Method::kPutBlob;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlob = clio::cte::core::Method::kGetBlob;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlobSize =
    clio::cte::core::Method::kGetBlobSize;
GLOBAL_CROSS_CONST clio::run::u32 kMultiPutBlob =
    clio::cte::core::Method::kMultiPutBlob;

// No cache-specific methods beyond Create/Destroy/Monitor: the data verbs
// are the core's, and asynchronous write-through needs no sweep. The id
// space above the core's stays reserved for future module verbs.
GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 101;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[kPutBlob] = "PutBlob";
    v[kGetBlob] = "GetBlob";
    v[kGetBlobSize] = "GetBlobSize";
    v[kMultiPutBlob] = "MultiPutBlob";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::cache

#endif  // CLIO_CTE_CACHE_AUTOGEN_METHODS_H_
