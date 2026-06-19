#ifndef CLIO_CTE_FILESYSTEM_AUTOGEN_METHODS_H_
#define CLIO_CTE_FILESYSTEM_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <string>
#include <vector>

/**
 * Method ids for the filesystem chimod. Hand-maintained (same as the CTE core
 * and compressor chimods). Keep in sync with clio_mod.yaml and the switch
 * cases in autogen/filesystem_lib_exec.cc.
 */
namespace clio::cte::filesystem {

namespace Method {
GLOBAL_CROSS_CONST chi::u32 kCreate = 0;
GLOBAL_CROSS_CONST chi::u32 kDestroy = 1;
GLOBAL_CROSS_CONST chi::u32 kMonitor = 9;

GLOBAL_CROSS_CONST chi::u32 kOpen = 10;
GLOBAL_CROSS_CONST chi::u32 kClose = 11;
GLOBAL_CROSS_CONST chi::u32 kRead = 12;
GLOBAL_CROSS_CONST chi::u32 kWrite = 13;
GLOBAL_CROSS_CONST chi::u32 kGetattr = 14;
GLOBAL_CROSS_CONST chi::u32 kTruncate = 15;
GLOBAL_CROSS_CONST chi::u32 kAppend = 16;
GLOBAL_CROSS_CONST chi::u32 kReaddir = 17;
GLOBAL_CROSS_CONST chi::u32 kMkdir = 18;
GLOBAL_CROSS_CONST chi::u32 kRmdir = 19;
GLOBAL_CROSS_CONST chi::u32 kUnlink = 20;
GLOBAL_CROSS_CONST chi::u32 kRename = 21;
GLOBAL_CROSS_CONST chi::u32 kStatSize = 22;
GLOBAL_CROSS_CONST chi::u32 kLink = 23;
// Deferred-append pipeline (collective, log-structured appends):
GLOBAL_CROSS_CONST chi::u32 kAppendSequence = 24;   // periodic local queue drain
GLOBAL_CROSS_CONST chi::u32 kAppendCollect = 25;    // ManyToOne collect (synchronous)
GLOBAL_CROSS_CONST chi::u32 kAppendExecution = 26;  // merge a plan slice into pages
GLOBAL_CROSS_CONST chi::u32 kAppendPlan = 27;       // sort+plan+dispatch (suspendable)

GLOBAL_CROSS_CONST chi::u32 kMaxMethodId = 28;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "Open";
    v[11] = "Close";
    v[12] = "Read";
    v[13] = "Write";
    v[14] = "Getattr";
    v[15] = "Truncate";
    v[16] = "Append";
    v[17] = "Readdir";
    v[18] = "Mkdir";
    v[19] = "Rmdir";
    v[20] = "Unlink";
    v[21] = "Rename";
    v[22] = "StatSize";
    v[23] = "Link";
    v[24] = "AppendSequence";
    v[25] = "AppendCollect";
    v[26] = "AppendExecution";
    v[27] = "AppendPlan";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_AUTOGEN_METHODS_H_
