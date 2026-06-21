#ifndef CLIO_BDEV_AUTOGEN_METHODS_H_
#define CLIO_BDEV_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for bdev
 */

namespace clio::run::bdev {

namespace Method {
// Inherited methods
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// bdev-specific methods
GLOBAL_CROSS_CONST clio::run::u32 kAllocateBlocks = 10;
GLOBAL_CROSS_CONST clio::run::u32 kFreeBlocks = 11;
GLOBAL_CROSS_CONST clio::run::u32 kWrite = 12;
GLOBAL_CROSS_CONST clio::run::u32 kRead = 13;
GLOBAL_CROSS_CONST clio::run::u32 kGetStats = 14;
GLOBAL_CROSS_CONST clio::run::u32 kUpdate = 15;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 16;

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
    v[15] = "Update";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::run::bdev

#endif  // BDEV_AUTOGEN_METHODS_H_
