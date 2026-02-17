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
GLOBAL_CONST chi::u32 kClientConnect = 16;
GLOBAL_CONST chi::u32 kMonitor = 17;
GLOBAL_CONST chi::u32 kSubmitBatch = 18;
GLOBAL_CONST chi::u32 kWreapDeadIpcs = 19;
GLOBAL_CONST chi::u32 kClientRecv = 20;
GLOBAL_CONST chi::u32 kClientSend = 21;
GLOBAL_CONST chi::u32 kRegisterMemory = 22;
GLOBAL_CONST chi::u32 kRestartContainers = 23;
GLOBAL_CONST chi::u32 kAddNode = 24;
GLOBAL_CONST chi::u32 kChangeAddressTable = 25;
GLOBAL_CONST chi::u32 kMigrateContainers = 26;
GLOBAL_CONST chi::u32 kHeartbeat = 27;
GLOBAL_CONST chi::u32 kHeartbeatProbe = 28;
GLOBAL_CONST chi::u32 kProbeRequest = 29;
GLOBAL_CONST chi::u32 kRecoverContainers = 30;
}  // namespace Method

}  // namespace chimaera::admin

#endif  // ADMIN_AUTOGEN_METHODS_H_
