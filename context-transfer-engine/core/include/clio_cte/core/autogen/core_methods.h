#ifndef CLIO_CTE_CORE_AUTOGEN_METHODS_H_
#define CLIO_CTE_CORE_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for core
 */

namespace clio::cte::core {

namespace Method {
// Inherited methods
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// core-specific methods
GLOBAL_CROSS_CONST clio::run::u32 kRegisterTarget = 10;
GLOBAL_CROSS_CONST clio::run::u32 kUnregisterTarget = 11;
GLOBAL_CROSS_CONST clio::run::u32 kListTargets = 12;
GLOBAL_CROSS_CONST clio::run::u32 kStatTargets = 13;
GLOBAL_CROSS_CONST clio::run::u32 kGetOrCreateTag = 14;
GLOBAL_CROSS_CONST clio::run::u32 kPutBlob = 15;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlob = 16;
GLOBAL_CROSS_CONST clio::run::u32 kReorganizeBlob = 17;
GLOBAL_CROSS_CONST clio::run::u32 kDelBlob = 18;
GLOBAL_CROSS_CONST clio::run::u32 kDelTag = 19;
GLOBAL_CROSS_CONST clio::run::u32 kGetTagSize = 20;
GLOBAL_CROSS_CONST clio::run::u32 kPollTelemetryLog = 21;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlobScore = 22;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlobSize = 23;
GLOBAL_CROSS_CONST clio::run::u32 kGetContainedBlobs = 24;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlobInfo = 25;
GLOBAL_CROSS_CONST clio::run::u32 kTagQuery = 30;
GLOBAL_CROSS_CONST clio::run::u32 kBlobQuery = 31;
GLOBAL_CROSS_CONST clio::run::u32 kGetTargetInfo = 32;
GLOBAL_CROSS_CONST clio::run::u32 kFlushMetadata = 33;
GLOBAL_CROSS_CONST clio::run::u32 kFlushData = 34;
GLOBAL_CROSS_CONST clio::run::u32 kSemanticSearch = 35;
GLOBAL_CROSS_CONST clio::run::u32 kTemporalSearch = 36;
GLOBAL_CROSS_CONST clio::run::u32 kTruncateBlob = 37;
GLOBAL_CROSS_CONST clio::run::u32 kRenameTag = 38;
GLOBAL_CROSS_CONST clio::run::u32 kGetOrCreateTagAlias = 39;
GLOBAL_CROSS_CONST clio::run::u32 kGetTagName = 40;
GLOBAL_CROSS_CONST clio::run::u32 kGetCapacity = 41;
GLOBAL_CROSS_CONST clio::run::u32 kGetNumAliases = 42;

// Fully-POD, GPU-compatible blob methods (fixed_string<32>, no SSO/SVO fixup).
GLOBAL_CROSS_CONST clio::run::u32 kPodPutBlob = 43;
GLOBAL_CROSS_CONST clio::run::u32 kPodGetBlob = 44;
GLOBAL_CROSS_CONST clio::run::u32 kPodReorganizeBlob = 45;

// Periodic internal data-organizer driver (issue #738)
GLOBAL_CROSS_CONST clio::run::u32 kDynamicReorganize = 46;

// Score-driven capacity eviction: free the lowest-score blobs off a tier
// (targets with score >= min) until a byte budget is reclaimed.
GLOBAL_CROSS_CONST clio::run::u32 kEvict = 47;

// Batched multi-blob put (issue #862): one task carrying up to ~64 puts to
// DIFFERENT blobs, executed as nested PutBlob calls — amortizes per-task
// scheduling/completion and staging allocation across the batch.
GLOBAL_CROSS_CONST clio::run::u32 kMultiPutBlob = 48;

// Register a node as holding a cached/replicated copy of one blob (issue
// #886 distributed coherence). The blob's OWNER container records the node
// in BlobInfo::replica_nodes_; the next primary write invalidates every
// registered node's copy (write-invalidate) before completing.
GLOBAL_CROSS_CONST clio::run::u32 kRegisterReplicaContainer = 49;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 50;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "RegisterTarget";
    v[11] = "UnregisterTarget";
    v[12] = "ListTargets";
    v[13] = "StatTargets";
    v[14] = "GetOrCreateTag";
    v[15] = "PutBlob";
    v[16] = "GetBlob";
    v[17] = "ReorganizeBlob";
    v[18] = "DelBlob";
    v[19] = "DelTag";
    v[20] = "GetTagSize";
    v[21] = "PollTelemetryLog";
    v[22] = "GetBlobScore";
    v[23] = "GetBlobSize";
    v[24] = "GetContainedBlobs";
    v[25] = "GetBlobInfo";
    v[30] = "TagQuery";
    v[31] = "BlobQuery";
    v[32] = "GetTargetInfo";
    v[33] = "FlushMetadata";
    v[34] = "FlushData";
    v[35] = "SemanticSearch";
    v[36] = "TemporalSearch";
    v[37] = "TruncateBlob";
    v[38] = "RenameTag";
    v[39] = "GetOrCreateTagAlias";
    v[40] = "GetTagName";
    v[41] = "GetCapacity";
    v[42] = "GetNumAliases";
    v[43] = "PodPutBlob";
    v[44] = "PodGetBlob";
    v[45] = "PodReorganizeBlob";
    v[46] = "DynamicReorganize";
    v[47] = "Evict";
    v[48] = "MultiPutBlob";
    v[49] = "RegisterReplicaContainer";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::core

#endif  // CORE_AUTOGEN_METHODS_H_
