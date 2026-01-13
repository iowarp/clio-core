#ifndef CHIMAERA_ADMIN_AUTOGEN_METHODS_H_
#define CHIMAERA_ADMIN_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>

/**
 * Auto-generated method definitions for admin
 */

namespace chimaera::admin {

namespace Method {
// Inherited methods
GLOBAL_CONST chi::u32 kCreate = 0;
GLOBAL_CONST chi::u32 kDestroy = 1;

// admin-specific methods
GLOBAL_CONST chi::u32 kGetOrCreatePool = 10;
GLOBAL_CONST chi::u32 kDestroyPool = 11;
GLOBAL_CONST chi::u32 kStopRuntime = 12;
GLOBAL_CONST chi::u32 kFlush = 13;
GLOBAL_CONST chi::u32 kSend = 14;
GLOBAL_CONST chi::u32 kRecv = 15;
GLOBAL_CONST chi::u32 kHeartbeat = 16;
}  // namespace Method

}  // namespace chimaera::admin

#endif  // ADMIN_AUTOGEN_METHODS_H_
