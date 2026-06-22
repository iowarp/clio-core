/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CLIO_RUNTIME_INCLUDE_TYPES_H_
#define CLIO_RUNTIME_INCLUDE_TYPES_H_

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

// Main CTP include
#include <clio_ctp/clio_ctp.h>
#include <clio_ctp/memory/allocator/malloc_allocator.h>
#include <clio_ctp/memory/allocator/round_robin_allocator.h>
#include <clio_ctp/memory/allocator/thread_allocator.h>
#include <clio_ctp/util/env_compat.h>
// CLIO_RUN_API + CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_{H,CC} macros (per-DLL
// export decoration so data globals work across DLL boundaries on Windows).
#include "clio_runtime/api.h"

/**
 * Core type definitions for CLIO Runtime distributed task execution framework
 */

// Canonical runtime namespace: clio::run.

namespace clio::run {

// Backward-compat env-var helper alias. Canonical lives in clio_ctp
// (ctp::env::GetCompat) so ctp-internal code can use it without an upward
// dependency on clio_runtime; this alias makes clio::run::env::GetCompat reachable
// from every clio_runtime TU that already includes <clio_runtime/types.h>.
// See clio_ctp/util/env_compat.h for the implementation.
namespace env {
using ctp::env::GetCompat;
}  // namespace env

// Basic type aliases using CTP types
using u32 = ctp::u32;
using u64 = ctp::u64;
using i64 = ctp::i64;
using ibitfield = ctp::ibitfield;

// Node state for SWIM-inspired failure detection
enum class NodeState : u32 {
  kAlive = 0,
  kProbeFailed = 1,  // Direct probe failed, indirect probing in progress
  kSuspected = 2,    // All indirect probes failed, likely dead
  kDead = 3          // Confirmed dead after suspicion timeout
};

// Time unit constants for period conversions (divisors from nanoseconds)
constexpr double kNano = 1.0;           // 1 nanosecond
constexpr double kMicro = 1000.0;       // 1000 nanoseconds = 1 microsecond
constexpr double kMilli = 1000000.0;    // 1,000,000 nanoseconds = 1 millisecond
constexpr double kSec = 1000000000.0;   // 1,000,000,000 nanoseconds = 1 second
constexpr double kMin = 60000000000.0;  // 60 seconds = 1 minute
constexpr double kHour = 3600000000000.0;  // 3600 seconds = 1 hour

// Forward declarations
class Task;
class PoolQuery;
class Worker;
class WorkOrchestrator;
class PoolManager;
class IpcManager;
class ConfigManager;
class ModuleManager;
class RuntimeManager;

/**
 * Host structure for hostfile management
 * Contains IP address and corresponding 64-bit node ID
 */
struct Host {
  std::string ip_address;  // IP address as string (IPv4 or IPv6)
  u64 node_id;             // 64-bit representation of IP address
  NodeState state;         // SWIM failure detection state
  std::chrono::steady_clock::time_point state_changed_at;

  /**
   * Default constructor
   */
  Host()
      : node_id(0),
        state(NodeState::kAlive),
        state_changed_at(std::chrono::steady_clock::now()) {}

  /**
   * Constructor with IP address and node ID (required)
   * Node IDs are assigned based on linear offset in hostfile
   * @param ip IP address string
   * @param id Node ID (typically offset in hostfile)
   */
  Host(const std::string &ip, u64 id)
      : ip_address(ip),
        node_id(id),
        state(NodeState::kAlive),
        state_changed_at(std::chrono::steady_clock::now()) {}

  bool IsAlive() const { return state == NodeState::kAlive; }

  /**
   * Stream output operator for Host
   * @param os Output stream
   * @param host Host object to print
   * @return Reference to output stream
   */
  friend std::ostream &operator<<(std::ostream &os, const Host &host) {
    const char *state_name = "unknown";
    switch (host.state) {
      case NodeState::kAlive:
        state_name = "alive";
        break;
      case NodeState::kProbeFailed:
        state_name = "probe_failed";
        break;
      case NodeState::kSuspected:
        state_name = "suspected";
        break;
      case NodeState::kDead:
        state_name = "dead";
        break;
    }
    os << "Host(ip=" << host.ip_address << ", node_id=" << host.node_id
       << ", state=" << state_name << ")";
    return os;
  }
};

/**
 * Unique identifier with major and minor components
 * Serializable and supports null values
 */
struct UniqueId {
  u32 major_;
  u32 minor_;

  CTP_CROSS_FUN constexpr UniqueId() : major_(0), minor_(0) {}
  CTP_CROSS_FUN constexpr UniqueId(u32 major, u32 minor)
      : major_(major), minor_(minor) {}

  // Equality operators
  CTP_CROSS_FUN bool operator==(const UniqueId &other) const {
    return major_ == other.major_ && minor_ == other.minor_;
  }

  CTP_CROSS_FUN bool operator!=(const UniqueId &other) const {
    return !(*this == other);
  }

  // Comparison operators for ordering
  CTP_CROSS_FUN bool operator<(const UniqueId &other) const {
    if (major_ != other.major_) return major_ < other.major_;
    return minor_ < other.minor_;
  }

  // Convert to u64 for compatibility and hashing
  CTP_CROSS_FUN u64 ToU64() const {
    return (static_cast<u64>(major_) << 32) | static_cast<u64>(minor_);
  }

  // Create from u64
  CTP_CROSS_FUN static UniqueId FromU64(u64 value) {
    return UniqueId(static_cast<u32>(value >> 32),
                    static_cast<u32>(value & 0xFFFFFFFF));
  }

  /**
   * Parse UniqueId from string format "major.minor"
   * @param str String representation of ID (e.g., "200.0")
   * @return Parsed UniqueId
   */
  static UniqueId FromString(const std::string &str);

  /**
   * Convert UniqueId to string format "major.minor"
   * @return String representation (e.g., "200.0")
   */
  std::string ToString() const;

  // Get null/invalid instance
  CTP_CROSS_FUN static constexpr UniqueId GetNull() { return UniqueId(0, 0); }

  // Check if this is a null/invalid ID
  CTP_CROSS_FUN bool IsNull() const { return major_ == 0 && minor_ == 0; }

  // Serialization support
  template <typename Ar>
  CTP_CROSS_FUN void serialize(Ar &ar) {
    ar.range(major_, minor_);
  }
};

/**
 * Pool identifier - typedef of UniqueId for semantic clarity
 */
using PoolId = UniqueId;

// Stream output operator for PoolId (typedef of UniqueId)
inline std::ostream &operator<<(std::ostream &os, const PoolId &pool_id) {
  os << "PoolId(major:" << pool_id.major_ << ", minor:" << pool_id.minor_
     << ")";
  return os;
}

/**
 * Task identifier containing process, thread, and sequence information
 */
struct TaskId {
  u32 pid_;    ///< Process ID
  u32 tid_;    ///< Thread ID
  u32 major_;  ///< Major sequence number (monotonically increasing per thread)
  u32 replica_id_;  ///< Replica identifier (for replicated tasks)
  u32 unique_;      ///< Unique identifier incremented for both root tasks and
                    ///< subtasks
  u32 node_id_;     ///< Node identifier for distributed execution
  size_t net_key_;  ///< Network key for send/recv map lookup (pointer-based)

  CTP_CROSS_FUN TaskId()
      : pid_(0),
        tid_(0),
        major_(0),
        replica_id_(0),
        unique_(0),
        node_id_(0),
        net_key_(0) {}
  CTP_CROSS_FUN TaskId(u32 pid, u32 tid, u32 major, u32 replica_id = 0,
                       u32 unique = 0, u64 node_id = 0, size_t net_key = 0)
      : pid_(pid),
        tid_(tid),
        major_(major),
        replica_id_(replica_id),
        unique_(unique),
        node_id_(node_id),
        net_key_(net_key) {}

  // Equality operators
  CTP_CROSS_FUN bool operator==(const TaskId &other) const {
    return pid_ == other.pid_ && tid_ == other.tid_ && major_ == other.major_ &&
           replica_id_ == other.replica_id_ && unique_ == other.unique_ &&
           node_id_ == other.node_id_ && net_key_ == other.net_key_;
  }

  bool operator!=(const TaskId &other) const { return !(*this == other); }

  // Convert to u64 for hashing (combine all fields)
  CTP_CROSS_FUN u64 ToU64() const {
    // Combine multiple fields using XOR and shifts for better distribution
    u64 hash1 = (static_cast<u64>(pid_) << 32) | static_cast<u64>(tid_);
    u64 hash2 =
        (static_cast<u64>(major_) << 32) | static_cast<u64>(replica_id_);
    u64 hash3 = (static_cast<u64>(unique_) << 32) |
                static_cast<u64>(node_id_ & 0xFFFFFFFF);
    return hash1 ^ hash2 ^ hash3;
  }

  // Serialization support
  template <typename Ar>
  CTP_CROSS_FUN void serialize(Ar &ar) {
    ar.range(pid_, tid_, major_, replica_id_, unique_, node_id_, net_key_);
  }
};

// Stream output operator for TaskId
inline std::ostream &operator<<(std::ostream &os, const TaskId &task_id) {
  os << "TaskId(pid:" << task_id.pid_ << ", tid:" << task_id.tid_
     << ", major:" << task_id.major_ << ", replica:" << task_id.replica_id_
     << ", unique:" << task_id.unique_ << ", node:" << task_id.node_id_
     << ", net_key:" << task_id.net_key_ << ")";
  return os;
}

/**
 * Identity for cooperative lock ownership tracking.
 * Subtasks share [pid, tid, major, node_id] with their parent,
 * so reentrancy is detected when all fields match.
 */
struct LockOwnerId {
  u32 worker_id_;
  u32 pid_;
  u32 tid_;
  u32 major_;
  u64 node_id_;

  CTP_CROSS_FUN LockOwnerId()
      : worker_id_(0), pid_(0), tid_(0), major_(0), node_id_(0) {}

  CTP_CROSS_FUN LockOwnerId(u32 worker_id, u32 pid, u32 tid, u32 major,
                            u64 node_id)
      : worker_id_(worker_id),
        pid_(pid),
        tid_(tid),
        major_(major),
        node_id_(node_id) {}

  CTP_CROSS_FUN bool IsNull() const {
    return worker_id_ == 0 && pid_ == 0 && tid_ == 0 && major_ == 0 &&
           node_id_ == 0;
  }

  CTP_CROSS_FUN bool operator==(const LockOwnerId &other) const {
    if (IsNull() || other.IsNull()) return false;
    return worker_id_ == other.worker_id_ && pid_ == other.pid_ &&
           tid_ == other.tid_ && major_ == other.major_ &&
           node_id_ == other.node_id_;
  }

  CTP_CROSS_FUN bool operator!=(const LockOwnerId &other) const {
    return !(*this == other);
  }

  CTP_CROSS_FUN void Clear() {
    worker_id_ = 0;
    pid_ = 0;
    tid_ = 0;
    major_ = 0;
    node_id_ = 0;
  }
};

// Host implementation lives in manager.cc. Under any device pass
// (CUDA/ROCm/SYCL) return a default-constructed sentinel — chimods that
// would call this from device code (corwlock helpers traced by DPC++) get
// a parseable inline body instead of an unresolved external reference.
#if !CTP_IS_DEVICE_PASS
LockOwnerId GetCurrentLockOwnerId();
#else
inline CTP_CROSS_FUN LockOwnerId GetCurrentLockOwnerId() {
  return LockOwnerId{};
}
#endif

using MethodId = u32;

// Worker and Lane identifiers
using WorkerId = u32;
using LaneId = u32;
using ContainerId = u32;
static constexpr ContainerId kInvalidContainerId = static_cast<ContainerId>(-1);
using MinorId = u32;

// Container addressing system types
using GroupId = u32;

/**
 * Predefined container groups
 */
namespace Group {
static constexpr GroupId kPhysical =
    0; /**< Physical address wrapper around node_id */
static constexpr GroupId kLocal = 1;  /**< Containers on THIS node */
static constexpr GroupId kGlobal = 2; /**< All containers in the pool */
}  // namespace Group

/**
 * Container address containing pool, group, and minor ID components
 *
 * Addresses have three components:
 * - PoolId: The pool the address is for
 * - GroupId: The container group (Physical, Local, or Global)
 * - MinorId: The unique ID within the group (node_id or container_id)
 */
struct Address {
  PoolId pool_id_;
  GroupId group_id_;
  MinorId minor_id_;

  Address() : pool_id_(), group_id_(Group::kLocal), minor_id_(0) {}
  Address(PoolId pool_id, GroupId group_id, MinorId minor_id)
      : pool_id_(pool_id), group_id_(group_id), minor_id_(minor_id) {}

  // Equality operator
  bool operator==(const Address &other) const {
    return pool_id_ == other.pool_id_ && group_id_ == other.group_id_ &&
           minor_id_ == other.minor_id_;
  }

  // Inequality operator
  bool operator!=(const Address &other) const { return !(*this == other); }

  // Cereal serialization support
  template <class Archive>
  void serialize(Archive &ar) {
    ar(pool_id_, group_id_, minor_id_);
  }
};

// Hash function for Address to use in std::unordered_map
struct AddressHash {
  std::size_t operator()(const Address &addr) const {
    std::size_t h1 = std::hash<u64>{}(addr.pool_id_.ToU64());
    std::size_t h2 = std::hash<GroupId>{}(addr.group_id_);
    std::size_t h3 = std::hash<MinorId>{}(addr.minor_id_);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

// Task flags using CTP BIT_OPT macro
#define TASK_PERIODIC BIT_OPT(clio::run::u32, 0)
#define TASK_ROUTED BIT_OPT(clio::run::u32, 1)
#define TASK_DATA_OWNER BIT_OPT(clio::run::u32, 2)
#define TASK_REMOTE BIT_OPT(clio::run::u32, 3)
// Bit 4 was TASK_FORCE_NET — removed in favor of the CLIO_FORCE_NET env
// variable (read by IpcManager::ServerInit, stored as force_net_, checked
// in IsTaskLocal). The per-task flag was redundant and noisier: every
// task site had to remember to set/clear it, vs. a single startup switch
// that flips the runtime's routing decision once. Bit position kept
// reserved so existing on-wire/persisted flag values keep their bit
// numbering.
#define TASK_STARTED \
  BIT_OPT(clio::run::u32, 5)  ///< Task execution has been started (set in BeginTask,
                        ///< unset in ReschedulePeriodicTask)
#define TASK_RUN_CTX_EXISTS \
  BIT_OPT(clio::run::u32, 6)  ///< RunContext has been allocated for this task (set in
                        ///< BeginTask, prevents duplicate BeginTask calls when
                        ///< task is forwarded between workers)
#define TASK_FIRE_AND_FORGET \
  BIT_OPT(clio::run::u32, 7)  ///< Task does not need a response. Wait/co_await return
                        ///< instantly; SendOut, ClientSend, and
                        ///< IpcManager::SendRuntime are skipped.
#define TASK_BATCH_AGGREGATE \
  BIT_OPT(clio::run::u32, 8)  ///< ManyToOne: this is the synthetic aggregate task the
                        ///< neighborhood leader runs for a batch. On completion,
                        ///< its OUT is broadcast to the batched original tasks.
#define TASK_EXTERNAL_CLIENT \
  BIT_OPT(clio::run::u32, 9)  ///< Task ingressed from an external user client (set in
                        ///< the IpcCpu2Cpu / IpcCpu2CpuZmq client-receive paths,
                        ///< never on runtime-internal self-sends). Serialized in
                        ///< SerializeIn so it rides a remote hop to the container
                        ///< owner, which enforces per-RPC access control
                        ///< (private methods reject external callers).

// Bulk transfer flags are defined in clio_ctp/lightbeam/lightbeam.h:
// - BULK_EXPOSE: Bulk is exposed (sender exposes for reading)
// - BULK_XFER: Bulk is exposed for writing (receiver)

// Lane mapping policies for task distribution
enum class LaneMapPolicy {
  kMapByPidTid = 0,  ///< Map tasks to lanes by hashing PID+TID (ensures
                     ///< per-thread affinity)
  kRoundRobin =
      1,  ///< Map tasks to lanes using round-robin (static counter, default)
  kRandom = 2  ///< Map tasks to lanes randomly
};

// Special pool IDs
constexpr PoolId kAdminPoolId =
    UniqueId(1, 0);  // Admin ChiMod pool ID (reserved)

// Allocator type aliases using CTP conventions
//
// CLIO_QUEUE_ALLOC_T: BuddyAllocator on both CPU and GPU (queue ring buffers)
//
// CLIO_TASK_ALLOC_T: allocator for task data and clio::run::priv structures
//   CPU: MultiProcessAllocator — shared across processes in the main SHM
//   segment GPU: BuddyAllocator       — per-thread GPU allocator
//
// CLIO_PRIV_ALLOC_T / CLIO_PRIV_ALLOC: private data allocator and instance
//   CPU: MallocAllocator / CTP_MALLOC
//   GPU: CLIO_TASK_ALLOC_T  / CLIO_IPC->GetMainAllocator()
//        (task constructors are never called from GPU kernels, so the GPU
//         CLIO_PRIV_ALLOC is only used for dynamic clio::run::priv operations in
//         kernels)
#define CLIO_QUEUE_ALLOC_T ctp::ipc::BuddyAllocator

/** Allocator scope for NewObj: private (warp-local) or shared (cross-warp) */
enum class AllocScope { kPrivate, kShared };

// Use the host allocators when compiling host CPU code under any compiler;
// any device pass (CUDA/ROCm/SYCL) takes the GPU branch below. Switching
// from `CTP_IS_HOST` to `!CTP_IS_DEVICE_PASS` is what lets DPC++'s SYCL
// device pass — where CTP_IS_HOST=1 because no __CUDA_ARCH__ is set —
// avoid the host-only MallocAllocatorSingleton (whose function-local
// static is rejected by SYCL kernels).
#if !CTP_IS_DEVICE_PASS
#define CLIO_TASK_ALLOC_T ctp::ipc::BuddyAllocator
#define CLIO_PRIV_ALLOC_T ctp::ipc::MallocAllocator
#define CLIO_PRIV_ALLOC CTP_MALLOC
// On CPU, shared == private (single-threaded MallocAllocator)
#define CLIO_PRIV_SHARED_ALLOC_T ctp::ipc::MallocAllocator
#define CLIO_PRIV_SHARED_ALLOC CTP_MALLOC
#else
#define CLIO_TASK_ALLOC_T ctp::ipc::BuddyAllocator
// GPU: CLIO_PRIV_ALLOC uses a cached PrivateBuddyAllocator* per warp.
// The pointer is resolved once during GPU init (from PartitionedAllocator) and
// cached in IpcManager::warp_priv_alloc_[], eliminating the per-allocation
// PartitionedAllocator indirection (GetAutoTid + LazyInitThread +
// GetThreadBlock).
#define CLIO_PRIV_ALLOC_T ctp::ipc::PrivateBuddyAllocator
/**
 * Get the GPU private allocator (cached PrivateBuddyAllocator for this warp).
 * Declared here; defined in ipc_manager.h after CLIO_IPC is available.
 *
 * @return Pointer to the warp's cached BuddyAllocator
 */
CTP_GPU_FUN ctp::ipc::PrivateBuddyAllocator *GetPrivAllocGpu();
#define CLIO_PRIV_ALLOC (::clio::run::GetPrivAllocGpu())
// GPU: CLIO_PRIV_SHARED_ALLOC returns the PartitionedAllocator
// (PartitionedAllocator) which dispatches allocations to the calling warp's
// partition via GetAutoTid(). Use for cross-warp data structures (shared maps,
// vectors) where multiple warps may allocate/free concurrently.
#define CLIO_PRIV_SHARED_ALLOC_T ctp::ipc::RoundRobinAllocator
CTP_GPU_FUN ctp::ipc::RoundRobinAllocator *GetSharedAllocGpu();
#define CLIO_PRIV_SHARED_ALLOC (::clio::run::GetSharedAllocGpu())
#endif

// ---------------------------------------------------------------------------
// Historical macro aliases
//
// `CLIO_TASK_ALLOC_T` was previously named `CLIO_MAIN_ALLOC_T`. The old name
// kept appearing in downstream snapshots of the runtime header (e.g.
// vendored copies inside chimods that haven't pulled in a fresh runtime
// sync). Re-exporting both spellings here lets that code compile without
// any source changes.
#define CLIO_MAIN_ALLOC_T CLIO_TASK_ALLOC_T

// Backward-compat aliases (clio_run rebrand). External code that still
// uses the legacy CHI_* spelling keeps working unchanged. These are
// text-substitution aliases — they resolve to whichever CLIO_* def is
// active for the current compilation pass (host vs device).

// Memory segment identifiers
enum MemorySegment {
  kMainSegment = 0,
  kClientDataSegment = 1,
  kQueueSegment = 2
};

// Input/Output parameter macros
#define IN
#define OUT
#define INOUT
#define TEMP

// CTP Thread-local storage keys
extern CLIO_RUN_API ctp::ThreadLocalKey chi_cur_worker_key_;
extern CLIO_RUN_API bool chi_cur_worker_key_created_;
extern CLIO_RUN_API ctp::ThreadLocalKey chi_task_counter_key_;
extern CLIO_RUN_API ctp::ThreadLocalKey chi_is_client_thread_key_;

/**
 * Thread-local task counter for generating unique TaskId major and unique
 * numbers
 */
struct TaskCounter {
  u32 counter_;

  TaskCounter() : counter_(0) {}

  u32 GetNext() { return ++counter_; }
};

/**
 * Create a new TaskId with current process/thread info and next major counter
 * In runtime mode: copies current task's TaskId and increments unique (keeps
 * replica_id_ same) In client mode: creates new TaskId with fresh major
 * counter, replica_id_ = 0, and unique from counter
 * @return TaskId with pid, tid, major, replica_id_, unique, and node_id
 * populated
 */
#if !CTP_IS_DEVICE_PASS
TaskId CreateTaskId();  // Host implementation in manager.cc
#else
// Device-pass inline implementation — simplified version. Used under
// CUDA/ROCm/SYCL device passes; CTP_IS_DEVICE_PASS makes the SYCL pass
// pick the inline body instead of the unresolved host declaration.
inline CTP_CROSS_FUN TaskId CreateTaskId() {
  TaskId id;
  id.pid_ = 0;
  id.tid_ = 0;
  id.major_ = 1;
  id.replica_id_ = 0;
  id.unique_ = 1;
  id.node_id_ = 0;
  return id;
}
#endif

// Template aliases for full pointers using CTP
template <typename T>
using FullPtr = ctp::ipc::FullPtr<T>;

/**
 * Migration descriptor for container migration between nodes
 */
struct MigrateInfo {
  PoolId pool_id_;
  ContainerId container_id_;
  u32 dest_;  // destination node ID

  MigrateInfo() : pool_id_(), container_id_(0), dest_(0) {}
  MigrateInfo(PoolId pool_id, ContainerId container_id, u32 dest)
      : pool_id_(pool_id), container_id_(container_id), dest_(dest) {}

  template <typename Ar>
  void serialize(Ar &ar) {
    ar(pool_id_, container_id_, dest_);
  }
};

/**
 * Recovery descriptor for recreating containers from dead nodes
 */
struct RecoveryAssignment {
  PoolId pool_id_;
  std::string chimod_name_;
  std::string pool_name_;
  std::string chimod_params_;
  ContainerId container_id_;
  u32 dest_node_id_;
  u32 dead_node_id_;

  RecoveryAssignment() : container_id_(0), dest_node_id_(0), dead_node_id_(0) {}

  template <typename Ar>
  void serialize(Ar &ar) {
    ar(pool_id_, chimod_name_, pool_name_, chimod_params_, container_id_,
       dest_node_id_, dead_node_id_);
  }
};

/** IPC transport mode for client-to-runtime communication */
enum class IpcMode : u32 {
  kTcp = 0,  ///< ZMQ tcp:// (default, always available)
  kIpc = 1,  ///< ZMQ ipc:// (Unix Domain Socket)
  kShm = 2,  ///< Shared memory (existing behavior)
};

}  // namespace clio::run

namespace clio::run::priv {
typedef ctp::priv::string<CLIO_PRIV_ALLOC_T> string;

template <typename T>
using vector = ctp::priv::vector<T, CLIO_PRIV_ALLOC_T>;

template <typename Key, typename T>
using unordered_map = ctp::priv::unordered_map_ll<Key, T, CLIO_PRIV_ALLOC_T>;

// Shared-scope types for cross-warp GPU data structures.
// On CPU these are identical to the private types above.
// On GPU, AllocT = PartitionedAllocator which dispatches to the correct
// warp partition, avoiding concurrent access to a single BuddyAllocator.
typedef ctp::priv::string<CLIO_PRIV_SHARED_ALLOC_T> shared_string;

template <typename T>
using shared_vector = ctp::priv::vector<T, CLIO_PRIV_SHARED_ALLOC_T>;

template <typename Key, typename T>
using shared_unordered_map =
    ctp::priv::unordered_map_ll<Key, T, CLIO_PRIV_SHARED_ALLOC_T>;
}  // namespace clio::run::priv

namespace clio::run::ipc {
// Queue structures use CLIO_QUEUE_ALLOC_T (BuddyAllocator)
template <typename T>
using multi_mpsc_ring_buffer =
    ctp::ipc::multi_mpsc_ring_buffer<T, CLIO_QUEUE_ALLOC_T>;

template <typename T>
using mpsc_ring_buffer = ctp::ipc::mpsc_ring_buffer<T, CLIO_QUEUE_ALLOC_T>;

template <typename T>
using vector = ctp::ipc::vector<T, CLIO_QUEUE_ALLOC_T>;
}  // namespace clio::run::ipc

// Hash function specializations for std::unordered_map
namespace std {
template <>
struct hash<clio::run::UniqueId> {
  size_t operator()(const clio::run::UniqueId &id) const {
    return hash<clio::run::u32>()(id.major_) ^ (hash<clio::run::u32>()(id.minor_) << 1);
  }
};

template <>
struct hash<clio::run::TaskId> {
  size_t operator()(const clio::run::TaskId &id) const {
    return hash<clio::run::u64>()(id.ToU64());
  }
};

}  // namespace std

#endif  // CLIO_RUNTIME_INCLUDE_TYPES_H_