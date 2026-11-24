#ifndef CHIMAERA_SAFE_BDEV_AUTOGEN_METHODS_H_
#define CHIMAERA_SAFE_BDEV_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for safe_bdev.
 * Updated in issue #1120 to avoid method ID collisions with bdev module.
 */

namespace clio::run::safe_bdev {

namespace Method {
// Inherited methods
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// Methods reused from bdev (same IDs so reused task types' method_ match)
GLOBAL_CROSS_CONST clio::run::u32 kAllocateBlocks = 10;
GLOBAL_CROSS_CONST clio::run::u32 kFreeBlocks = 11;
GLOBAL_CROSS_CONST clio::run::u32 kWrite = 12;
GLOBAL_CROSS_CONST clio::run::u32 kRead = 13;
GLOBAL_CROSS_CONST clio::run::u32 kGetStats = 14;
GLOBAL_CROSS_CONST clio::run::u32 kSync = 18;  // Same as bdev

// safe_bdev-specific methods (erasure-coding management, renumbered to start at 21)
GLOBAL_CROSS_CONST clio::run::u32 kAddBdev = 21;
GLOBAL_CROSS_CONST clio::run::u32 kRemoveBdev = 22;
GLOBAL_CROSS_CONST clio::run::u32 kRecoverBdev = 23;
GLOBAL_CROSS_CONST clio::run::u32 kBuildParity = 24;
GLOBAL_CROSS_CONST clio::run::u32 kFlushAllocLog = 25;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 26;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "AllocateBlocks";
    v[11] = "FreeBlocks";
    v[12] = "Write";
    v[13] = "Read";
    v[14] = "GetStats";
    v[18] = "Sync";
    v[21] = "AddBdev";
    v[22] = "RemoveBdev";
    v[23] = "RecoverBdev";
    v[24] = "BuildParity";
    v[25] = "FlushAllocLog";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::run::safe_bdev

#endif  // CHIMAERA_SAFE_BDEV_AUTOGEN_METHODS_H_
