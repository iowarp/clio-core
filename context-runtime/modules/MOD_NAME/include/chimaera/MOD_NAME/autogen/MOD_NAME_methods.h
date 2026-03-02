#ifndef CHIMAERA_MOD_NAME_AUTOGEN_METHODS_H_
#define CHIMAERA_MOD_NAME_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for MOD_NAME
 */

namespace chimaera::MOD_NAME {

namespace Method {
// Inherited methods
GLOBAL_CONST chi::u32 kCreate = 0;
GLOBAL_CONST chi::u32 kDestroy = 1;
GLOBAL_CONST chi::u32 kMonitor = 9;

// MOD_NAME-specific methods
GLOBAL_CONST chi::u32 kCustom = 10;
GLOBAL_CONST chi::u32 kCoMutexTest = 20;
GLOBAL_CONST chi::u32 kCoRwLockTest = 21;
GLOBAL_CONST chi::u32 kWaitTest = 23;
GLOBAL_CONST chi::u32 kTestLargeOutput = 24;
GLOBAL_CONST chi::u32 kGpuSubmit = 25;

GLOBAL_CONST chi::u32 kMaxMethodId = 26;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "Custom";
    v[20] = "CoMutexTest";
    v[21] = "CoRwLockTest";
    v[23] = "WaitTest";
    v[24] = "TestLargeOutput";
    v[25] = "GpuSubmit";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace chimaera::MOD_NAME

#endif  // MOD_NAME_AUTOGEN_METHODS_H_
