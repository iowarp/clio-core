#ifndef CHIMAERA_SAFE_BDEV_AUTOGEN_METHODS_H_
#define CHIMAERA_SAFE_BDEV_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for safe_bdev
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

// safe_bdev-specific methods (erasure-coding management)
GLOBAL_CROSS_CONST clio::run::u32 kAddBdev = 16;
GLOBAL_CROSS_CONST clio::run::u32 kRemoveBdev = 17;
GLOBAL_CROSS_CONST clio::run::u32 kRecoverBdev = 18;
GLOBAL_CROSS_CONST clio::run::u32 kBuildParity = 19;
GLOBAL_CROSS_CONST clio::run::u32 kFlushAllocLog = 20;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 21;

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
    v[16] = "AddBdev";
    v[17] = "RemoveBdev";
    v[18] = "RecoverBdev";
    v[19] = "BuildParity";
    v[20] = "FlushAllocLog";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::run::safe_bdev

#endif  // CHIMAERA_SAFE_BDEV_AUTOGEN_METHODS_H_
