/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_CHECKPOINT_AUTOGEN_METHODS_H_
#define CLIO_CTE_CHECKPOINT_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/autogen/core_methods.h>
#include <string>
#include <vector>

/**
 * Method ids for the checkpoint chimod. Hand-maintained; same interposition
 * rules as the cache/compressor/replication chimods: the core data verb
 * carries the CORE's id and task struct, every other core id is forwarded
 * verbatim by the dispatch defaults, and module verbs live ABOVE the core's
 * id space.
 */
namespace clio::cte::checkpoint {

namespace Method {
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// The one interposed core verb: the core's fault dispatch sends its own
// GetBlobTask here when a fault-handled tag misses a blob.
GLOBAL_CROSS_CONST clio::run::u32 kGetBlob = clio::cte::core::Method::kGetBlob;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 101;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[kGetBlob] = "GetBlob";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::checkpoint

#endif  // CLIO_CTE_CHECKPOINT_AUTOGEN_METHODS_H_
