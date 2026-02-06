#ifndef CHIMAERA_INCLUDE_CHIMAERA_TYPES_H_
#define CHIMAERA_INCLUDE_CHIMAERA_TYPES_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

// Main HSHM include
#include <hermes_shm/hermes_shm.h>
#include <hermes_shm/memory/allocator/malloc_allocator.h>

/**
 * Core type definitions for Chimaera distributed task execution framework
 */

namespace chi {

// Basic type aliases using HSHM types
using u32 = hshm::u32;
using u64 = hshm::u64;
using i64 = hshm::i64;
using ibitfield = hshm::ibitfield;

// Time unit constants for period conversions (divisors from nanoseconds)
constexpr double kNano = 1.0;          // 1 nanosecond
constexpr double kMicro = 1000.0;      // 1000 nanoseconds = 1 microsecond
constexpr double kMilli = 1000000.0;   // 1,000,000 nanoseconds = 1 millisecond
constexpr double kSec = 1000000000.0;  // 1,000,000,000 nanoseconds = 1 second
constexpr double kMin = 60000000000.0; // 60 seconds = 1 minute
constexpr double kHour = 3600000000000.0; // 3600 seconds = 1 hour

// Forward declarations
class Task;
class PoolQuery;
class Worker;
class WorkOrchestrator;
class PoolManager;
class IpcManager;
class ConfigManager;
class ModuleManager;
class Chimaera;

/**
 * Unique identifier with major and minor components
 * Serializable and supports null values
 */
struct UniqueId {
  u32 major_;
  u32 minor_;

  constexpr UniqueId() : major_(0), minor_(0) {}
  constexpr UniqueId(u32 major, u32 minor) : major_(major), minor_(minor) {}

  // Equality operators
  bool operator==(const UniqueId &other) const {
    return major_ == other.major_ && minor_ == other.minor_;
  }

  bool operator!=(const UniqueId &other) const { return !(*this == other); }

  // Comparison operators for ordering
  bool operator<(const UniqueId &other) const {
    if (major_ != other.major_)
      return major_ < other.major_;
    return minor_ < other.minor_;
  }

  // Convert to u64 for compatibility and hashing
  u64 ToU64() const {
    return (static_cast<u64>(major_) << 32) | static_cast<u64>(minor_);
  }

  // Create from u64
  static UniqueId FromU64(u64 value) {
    return UniqueId(static_cast<u32>(value >> 32),
                    static_cast<u32>(value & 0xFFFFFFFF));
  }

  /**
   * Parse UniqueId from string format "major.minor"
   * @param str String representation of ID (e.g., "200.0")
   * @return Parsed UniqueId
   */
  static UniqueId FromString(const std::string& str);

  // Get null/invalid instance
  static constexpr UniqueId GetNull() { return UniqueId(0, 0); }

  // Check if this is a null/invalid ID
  bool IsNull() const { return major_ == 0 && minor_ == 0; }

  // Serialization support
  template <typename Ar> void serialize(Ar &ar) { ar(major_, minor_); }
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
  u32 pid_;   ///< Process ID
  u32 tid_;   ///< Thread ID
  u32 major_; ///< Major sequence number (monotonically increasing per thread)
  u32 replica_id_; ///< Replica identifier (for replicated tasks)
  u32 unique_;     ///< Unique identifier incremented for both root tasks and
                   ///< subtasks
  u64 node_id_;    ///< Node identifier for distributed execution
  size_t net_key_; ///< Network key for send/recv map lookup (pointer-based)

  TaskId()
      : pid_(0), tid_(0), major_(0), replica_id_(0), unique_(0), node_id_(0),
        net_key_(0) {}
  TaskId(u32 pid, u32 tid, u32 major, u32 replica_id = 0, u32 unique = 0,
         u64 node_id = 0, size_t net_key = 0)
      : pid_(pid), tid_(tid), major_(major), replica_id_(replica_id),
        unique_(unique), node_id_(node_id), net_key_(net_key) {}

  // Equality operators
  bool operator==(const TaskId &other) const {
    return pid_ == other.pid_ && tid_ == other.tid_ && major_ == other.major_ &&
           replica_id_ == other.replica_id_ && unique_ == other.unique_ &&
           node_id_ == other.node_id_ && net_key_ == other.net_key_;
  }

  bool operator!=(const TaskId &other) const { return !(*this == other); }

  // Convert to u64 for hashing (combine all fields)
  u64 ToU64() const {
    // Combine multiple fields using XOR and shifts for better distribution
    u64 hash1 = (static_cast<u64>(pid_) << 32) | static_cast<u64>(tid_);
    u64 hash2 =
        (static_cast<u64>(major_) << 32) | static_cast<u64>(replica_id_);
    u64 hash3 = (static_cast<u64>(unique_) << 32) |
                static_cast<u64>(node_id_ & 0xFFFFFFFF);
    return hash1 ^ hash2 ^ hash3;
  }

  // Serialization support
  template <typename Ar> void serialize(Ar &ar) {
    ar(pid_, tid_, major_, replica_id_, unique_, node_id_, net_key_);
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

using MethodId = u32;

// Worker and Lane identifiers
using WorkerId = u32;
using LaneId = u32;
using ContainerId = u32;
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
} // namespace Group

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
  template <class Archive> void serialize(Archive &ar) {
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

// Task flags using HSHM BIT_OPT macro
#define TASK_PERIODIC BIT_OPT(chi::u32, 0)
#define TASK_ROUTED BIT_OPT(chi::u32, 1)
#define TASK_DATA_OWNER BIT_OPT(chi::u32, 2)
#define TASK_REMOTE BIT_OPT(chi::u32, 3)
#define TASK_FORCE_NET                                                         \
  BIT_OPT(chi::u32,                                                            \
          4) ///< Force task through network code even for local execution
#define TASK_STARTED                                                           \
  BIT_OPT(chi::u32, 5) ///< Task execution has been started (set in BeginTask,
                       ///< unset in ReschedulePeriodicTask)

// Bulk transfer flags are defined in hermes_shm/lightbeam/lightbeam.h:
// - BULK_EXPOSE: Bulk is exposed (sender exposes for reading)
// - BULK_XFER: Bulk is exposed for writing (receiver)

// Thread types for work orchestrator
enum ThreadType {
  kSchedWorker = 0,    ///< Scheduler worker for fast tasks (EstCpuTime < 50us)
  kSlow = 1,           ///< Slow worker for long-running tasks (EstCpuTime >= 50us)
  kProcessReaper = 2,  ///< Process reaper thread
  kNetWorker = 3       ///< Network worker for Send/Recv tasks
};

// Lane mapping policies for task distribution
enum class LaneMapPolicy {
  kMapByPidTid = 0, ///< Map tasks to lanes by hashing PID+TID (ensures
                    ///< per-thread affinity)
  kRoundRobin =
      1, ///< Map tasks to lanes using round-robin (static counter, default)
  kRandom = 2 ///< Map tasks to lanes randomly
};

// Special pool IDs
constexpr PoolId kAdminPoolId =
    UniqueId(1, 0); // Admin ChiMod pool ID (reserved)

// Allocator type aliases using HSHM conventions
#define CHI_MAIN_ALLOC_T hipc::MultiProcessAllocator
#define CHI_CDATA_ALLOC_T hipc::MultiProcessAllocator

// Memory segment identifiers
enum MemorySegment {
  kMainSegment = 0,
  kClientDataSegment = 1
};

// Input/Output parameter macros
#define IN
#define OUT
#define INOUT
#define TEMP

// HSHM Thread-local storage keys
extern hshm::ThreadLocalKey chi_cur_worker_key_;
extern hshm::ThreadLocalKey chi_task_counter_key_;

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
TaskId CreateTaskId();

// Template aliases for full pointers using HSHM
template <typename T> using FullPtr = hipc::FullPtr<T>;

}  // namespace chi

namespace chi::priv {
// Private data structures use MallocAllocator (heap memory, not shared)
typedef hshm::priv::string<hipc::MallocAllocator> string;

template<typename T>
using vector = hshm::priv::vector<T, hipc::MallocAllocator>;
}  // namespace chi::priv

namespace chi::ipc {
template <typename T>
using multi_mpsc_ring_buffer =
    hipc::multi_mpsc_ring_buffer<T, CHI_MAIN_ALLOC_T>;

template <typename T>
using mpsc_ring_buffer = hipc::mpsc_ring_buffer<T, CHI_MAIN_ALLOC_T>;

template <typename T>
using vector = hipc::vector<T, CHI_MAIN_ALLOC_T>;
}  // namespace chi::ipc

// Hash function specializations for std::unordered_map
namespace std {
template <> struct hash<chi::UniqueId> {
  size_t operator()(const chi::UniqueId &id) const {
    return hash<chi::u32>()(id.major_) ^ (hash<chi::u32>()(id.minor_) << 1);
  }
};

template <> struct hash<chi::TaskId> {
  size_t operator()(const chi::TaskId &id) const {
    return hash<chi::u64>()(id.ToU64());
  }
};

} // namespace std

#endif // CHIMAERA_INCLUDE_CHIMAERA_TYPES_H_