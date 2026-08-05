/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_REPLICATION_AUTOGEN_METHODS_H_
#define CLIO_CTE_REPLICATION_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/autogen/core_methods.h>
#include <string>
#include <vector>

/**
 * Method ids for the replication chimod. Hand-maintained (same as the CTE
 * core, compressor and filesystem chimods). Keep in sync with clio_mod.yaml
 * and the switch cases in autogen/replication_lib_exec.cc.
 *
 * This chimod INTERPOSES on the CTE core's own task interface: a
 * clio::cte::core::Client pointed at this pool keeps working unchanged.
 * kPutBlob/kGetBlob therefore MUST carry the core's method ids (the client
 * sends core ids), and every other core method id is forwarded verbatim to
 * the next pool by the dispatch defaults — which is why the module's own
 * verbs are numbered ABOVE the core's method space: a module id that
 * collided with a core id would shadow the forwarded verb.
 */
namespace clio::cte::replication {

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

// replication-specific methods (above the core's id space; see header note)
GLOBAL_CROSS_CONST clio::run::u32 kReplicateBlob = 100;
GLOBAL_CROSS_CONST clio::run::u32 kFlushTag = 101;
GLOBAL_CROSS_CONST clio::run::u32 kReplicateSweep = 102;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 103;

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
    v[100] = "ReplicateBlob";
    v[101] = "FlushTag";
    v[102] = "ReplicateSweep";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::replication

#endif  // CLIO_CTE_REPLICATION_AUTOGEN_METHODS_H_
