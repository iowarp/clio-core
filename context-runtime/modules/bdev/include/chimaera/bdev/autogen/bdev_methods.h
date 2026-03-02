#ifndef CHIMAERA_BDEV_AUTOGEN_METHODS_H_
#define CHIMAERA_BDEV_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for bdev
 */

namespace chimaera::bdev {

namespace Method {
// Inherited methods
GLOBAL_CONST chi::u32 kCreate = 0;
GLOBAL_CONST chi::u32 kDestroy = 1;
GLOBAL_CONST chi::u32 kMonitor = 9;

// bdev-specific methods
GLOBAL_CONST chi::u32 kAllocateBlocks = 10;
GLOBAL_CONST chi::u32 kFreeBlocks = 11;
GLOBAL_CONST chi::u32 kWrite = 12;
GLOBAL_CONST chi::u32 kRead = 13;
GLOBAL_CONST chi::u32 kGetStats = 14;

GLOBAL_CONST chi::u32 kMaxMethodId = 15;

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
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace chimaera::bdev

#endif  // BDEV_AUTOGEN_METHODS_H_
