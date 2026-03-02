#ifndef WRP_CTE_CORE_AUTOGEN_METHODS_H_
#define WRP_CTE_CORE_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for core
 */

namespace wrp_cte::core {

namespace Method {
// Inherited methods
GLOBAL_CONST chi::u32 kCreate = 0;
GLOBAL_CONST chi::u32 kDestroy = 1;
GLOBAL_CONST chi::u32 kMonitor = 9;

// core-specific methods
GLOBAL_CONST chi::u32 kRegisterTarget = 10;
GLOBAL_CONST chi::u32 kUnregisterTarget = 11;
GLOBAL_CONST chi::u32 kListTargets = 12;
GLOBAL_CONST chi::u32 kStatTargets = 13;
GLOBAL_CONST chi::u32 kGetOrCreateTag = 14;
GLOBAL_CONST chi::u32 kPutBlob = 15;
GLOBAL_CONST chi::u32 kGetBlob = 16;
GLOBAL_CONST chi::u32 kReorganizeBlob = 17;
GLOBAL_CONST chi::u32 kDelBlob = 18;
GLOBAL_CONST chi::u32 kDelTag = 19;
GLOBAL_CONST chi::u32 kGetTagSize = 20;
GLOBAL_CONST chi::u32 kPollTelemetryLog = 21;
GLOBAL_CONST chi::u32 kGetBlobScore = 22;
GLOBAL_CONST chi::u32 kGetBlobSize = 23;
GLOBAL_CONST chi::u32 kGetContainedBlobs = 24;
GLOBAL_CONST chi::u32 kGetBlobInfo = 25;
GLOBAL_CONST chi::u32 kTagQuery = 30;
GLOBAL_CONST chi::u32 kBlobQuery = 31;
GLOBAL_CONST chi::u32 kGetTargetInfo = 32;
GLOBAL_CONST chi::u32 kFlushMetadata = 33;
GLOBAL_CONST chi::u32 kFlushData = 34;

GLOBAL_CONST chi::u32 kMaxMethodId = 35;

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
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace wrp_cte::core

#endif  // CORE_AUTOGEN_METHODS_H_
