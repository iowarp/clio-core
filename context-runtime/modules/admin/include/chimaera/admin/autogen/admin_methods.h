#ifndef CHIMAERA_ADMIN_AUTOGEN_METHODS_H_
#define CHIMAERA_ADMIN_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for admin
 */

namespace chimaera::admin {

namespace Method {
// Inherited methods
GLOBAL_CONST chi::u32 kCreate = 0;
GLOBAL_CONST chi::u32 kDestroy = 1;
GLOBAL_CONST chi::u32 kMonitor = 9;

// admin-specific methods
GLOBAL_CONST chi::u32 kGetOrCreatePool = 10;
GLOBAL_CONST chi::u32 kDestroyPool = 11;
GLOBAL_CONST chi::u32 kStopRuntime = 12;
GLOBAL_CONST chi::u32 kFlush = 13;
GLOBAL_CONST chi::u32 kSend = 14;
GLOBAL_CONST chi::u32 kRecv = 15;
GLOBAL_CONST chi::u32 kClientConnect = 16;
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
GLOBAL_CONST chi::u32 kSystemMonitor = 31;
GLOBAL_CONST chi::u32 kAnnounceShutdown = 32;

GLOBAL_CONST chi::u32 kMaxMethodId = 33;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "GetOrCreatePool";
    v[11] = "DestroyPool";
    v[12] = "StopRuntime";
    v[13] = "Flush";
    v[14] = "Send";
    v[15] = "Recv";
    v[16] = "ClientConnect";
    v[18] = "SubmitBatch";
    v[19] = "WreapDeadIpcs";
    v[20] = "ClientRecv";
    v[21] = "ClientSend";
    v[22] = "RegisterMemory";
    v[23] = "RestartContainers";
    v[24] = "AddNode";
    v[25] = "ChangeAddressTable";
    v[26] = "MigrateContainers";
    v[27] = "Heartbeat";
    v[28] = "HeartbeatProbe";
    v[29] = "ProbeRequest";
    v[30] = "RecoverContainers";
    v[31] = "SystemMonitor";
    v[32] = "AnnounceShutdown";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace chimaera::admin

#endif  // ADMIN_AUTOGEN_METHODS_H_
