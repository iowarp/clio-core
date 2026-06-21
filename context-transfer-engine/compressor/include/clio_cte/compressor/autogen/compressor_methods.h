#ifndef CLIO_CTE_COMPRESSOR_AUTOGEN_METHODS_H_
#define CLIO_CTE_COMPRESSOR_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for compressor
 */

namespace clio::cte::compressor {

namespace Method {
// Inherited methods
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// compressor-specific methods
GLOBAL_CROSS_CONST clio::run::u32 kDynamicSchedule = 10;
GLOBAL_CROSS_CONST clio::run::u32 kCompress = 11;
GLOBAL_CROSS_CONST clio::run::u32 kDecompress = 12;
GLOBAL_CROSS_CONST clio::run::u32 kPollNodeLoad = 13;
GLOBAL_CROSS_CONST clio::run::u32 kPollConsumers = 14;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 15;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "DynamicSchedule";
    v[11] = "Compress";
    v[12] = "Decompress";
    v[13] = "PollNodeLoad";
    v[14] = "PollConsumers";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::compressor

#endif  // COMPRESSOR_AUTOGEN_METHODS_H_
