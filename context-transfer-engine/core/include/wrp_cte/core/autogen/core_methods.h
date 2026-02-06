#ifndef WRP_CTE_CORE_AUTOGEN_METHODS_H_
#define WRP_CTE_CORE_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>

/**
 * Auto-generated method definitions for core
 */

namespace wrp_cte::core {

namespace Method {
// Inherited methods
GLOBAL_CONST chi::u32 kCreate = 0;
GLOBAL_CONST chi::u32 kDestroy = 1;

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
GLOBAL_CONST chi::u32 kTagQuery = 30;
GLOBAL_CONST chi::u32 kBlobQuery = 31;
GLOBAL_CONST chi::u32 kGetTargetInfo = 32;
}  // namespace Method

}  // namespace wrp_cte::core

#endif  // CORE_AUTOGEN_METHODS_H_
