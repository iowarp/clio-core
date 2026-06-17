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
GLOBAL_CROSS_CONST chi::u32 kCreate = 0;
GLOBAL_CROSS_CONST chi::u32 kDestroy = 1;
GLOBAL_CROSS_CONST chi::u32 kMonitor = 9;

// core-specific methods
GLOBAL_CROSS_CONST chi::u32 kRegisterTarget = 10;
GLOBAL_CROSS_CONST chi::u32 kUnregisterTarget = 11;
GLOBAL_CROSS_CONST chi::u32 kListTargets = 12;
GLOBAL_CROSS_CONST chi::u32 kStatTargets = 13;
GLOBAL_CROSS_CONST chi::u32 kGetOrCreateTag = 14;
GLOBAL_CROSS_CONST chi::u32 kPutBlob = 15;
GLOBAL_CROSS_CONST chi::u32 kGetBlob = 16;
GLOBAL_CROSS_CONST chi::u32 kReorganizeBlob = 17;
GLOBAL_CROSS_CONST chi::u32 kDelBlob = 18;
GLOBAL_CROSS_CONST chi::u32 kDelTag = 19;
GLOBAL_CROSS_CONST chi::u32 kGetTagSize = 20;
GLOBAL_CROSS_CONST chi::u32 kPollTelemetryLog = 21;
GLOBAL_CROSS_CONST chi::u32 kGetBlobScore = 22;
GLOBAL_CROSS_CONST chi::u32 kGetBlobSize = 23;
GLOBAL_CROSS_CONST chi::u32 kGetContainedBlobs = 24;
GLOBAL_CROSS_CONST chi::u32 kGetBlobInfo = 25;
GLOBAL_CROSS_CONST chi::u32 kTagQuery = 30;
GLOBAL_CROSS_CONST chi::u32 kBlobQuery = 31;
GLOBAL_CROSS_CONST chi::u32 kGetTargetInfo = 32;
GLOBAL_CROSS_CONST chi::u32 kFlushMetadata = 33;
GLOBAL_CROSS_CONST chi::u32 kFlushData = 34;
GLOBAL_CROSS_CONST chi::u32 kSemanticSearch = 35;
GLOBAL_CROSS_CONST chi::u32 kTemporalSearch = 36;
GLOBAL_CROSS_CONST chi::u32 kTruncateBlob = 37;
GLOBAL_CROSS_CONST chi::u32 kRenameTag = 38;
GLOBAL_CROSS_CONST chi::u32 kGetOrCreateTagAlias = 39;
GLOBAL_CROSS_CONST chi::u32 kGetTagName = 40;

GLOBAL_CROSS_CONST chi::u32 kMaxMethodId = 41;

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
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::core

#endif  // CORE_AUTOGEN_METHODS_H_
