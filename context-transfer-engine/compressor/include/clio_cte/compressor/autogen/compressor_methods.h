#ifndef CLIO_CTE_COMPRESSOR_AUTOGEN_METHODS_H_
#define CLIO_CTE_COMPRESSOR_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/autogen/core_methods.h>
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
// Interposed core data verbs (issue #886 interposition): same ids, same
// task structs as the CTE core — a clio::cte::core::Client pointed at this
// pool keeps working, and every other core id is forwarded to next_pool_id
// by the dispatch defaults.
GLOBAL_CROSS_CONST clio::run::u32 kPutBlob = clio::cte::core::Method::kPutBlob;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlob = clio::cte::core::Method::kGetBlob;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlobSize =
    clio::cte::core::Method::kGetBlobSize;
GLOBAL_CROSS_CONST clio::run::u32 kMultiPutBlob =
    clio::cte::core::Method::kMultiPutBlob;
// POD put/get: what the gpu_vector cache manager submits from DEVICE code
// (no SHM strings). Interposed for the same reason as kPutBlob/kGetBlob —
// without these the device-submitted page evictions and page faults would
// fall through to the forwarding default and land on the core uncompressed.
GLOBAL_CROSS_CONST clio::run::u32 kPodPutBlob =
    clio::cte::core::Method::kPodPutBlob;
GLOBAL_CROSS_CONST clio::run::u32 kPodGetBlob =
    clio::cte::core::Method::kPodGetBlob;

// compressor-specific methods, numbered ABOVE the core's method-id space —
// a collision would shadow a forwarded core verb (they previously sat on
// 10-14 = RegisterTarget..GetOrCreateTag).
GLOBAL_CROSS_CONST clio::run::u32 kDynamicSchedule = 100;
GLOBAL_CROSS_CONST clio::run::u32 kCompress = 101;
GLOBAL_CROSS_CONST clio::run::u32 kDecompress = 102;
GLOBAL_CROSS_CONST clio::run::u32 kPollNodeLoad = 103;
GLOBAL_CROSS_CONST clio::run::u32 kPollConsumers = 104;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 105;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[kPutBlob] = "PutBlob";
    v[kGetBlob] = "GetBlob";
    v[kGetBlobSize] = "GetBlobSize";
    v[kMultiPutBlob] = "MultiPutBlob";
    v[kPodPutBlob] = "PodPutBlob";
    v[kPodGetBlob] = "PodGetBlob";
    v[100] = "DynamicSchedule";
    v[101] = "Compress";
    v[102] = "Decompress";
    v[103] = "PollNodeLoad";
    v[104] = "PollConsumers";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::compressor

#endif  // COMPRESSOR_AUTOGEN_METHODS_H_
