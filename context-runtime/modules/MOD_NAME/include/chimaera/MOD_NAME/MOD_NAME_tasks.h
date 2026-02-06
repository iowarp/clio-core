#ifndef MOD_NAME_TASKS_H_
#define MOD_NAME_TASKS_H_

#include <chimaera/chimaera.h>
#include "autogen/MOD_NAME_methods.h"
// Include admin tasks for BaseCreateTask
#include <chimaera/admin/admin_tasks.h>

/**
 * Task struct definitions for MOD_NAME
 * 
 * Defines the tasks for Create and Custom methods.
 */

namespace chimaera::MOD_NAME {

/**
 * CreateParams for MOD_NAME chimod
 * Contains configuration parameters for MOD_NAME container creation
 */
struct CreateParams {
  // MOD_NAME-specific parameters (primitives only for cereal compatibility)
  chi::u32 worker_count_;
  chi::u32 config_flags_;

  // Required: chimod library name for module manager
  static constexpr const char* chimod_lib_name = "chimaera_MOD_NAME";

  // Constructor with parameters (also serves as default)
  CreateParams(chi::u32 worker_count = 1, chi::u32 config_flags = 0)
      : worker_count_(worker_count), config_flags_(config_flags) {
  }

  // Serialization support for cereal
  template<class Archive>
  void serialize(Archive& ar) {
    ar(worker_count_, config_flags_);
  }
};

/**
 * CreateTask - Initialize the MOD_NAME container
 * Type alias for GetOrCreatePoolTask with CreateParams (uses kGetOrCreatePool method)
 * Non-admin modules should use GetOrCreatePoolTask instead of BaseCreateTask
 */
using CreateTask = chimaera::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * CustomTask - Example custom operation
 */
struct CustomTask : public chi::Task {
  // Task-specific data
  INOUT chi::priv::string data_;
  IN chi::u32 operation_id_;

  /** SHM default constructor */
  CustomTask()
      : chi::Task(),
        data_(HSHM_MALLOC), operation_id_(0) {}

  /** Emplace constructor */
  explicit CustomTask(
      const chi::TaskId &task_node,
      const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query,
      const std::string &data,
      chi::u32 operation_id)
      : chi::Task(task_node, pool_id, pool_query, 10),
        data_(HSHM_MALLOC, data), operation_id_(operation_id) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kCustom;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   * This includes: data_, operation_id_
   */
  template<typename Archive>
  void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(data_, operation_id_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: data_
   */
  template<typename Archive>
  void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar(data_);
  }

  /**
   * Copy from another CustomTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<CustomTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy CustomTask-specific fields
    data_ = other->data_;
    operation_id_ = other->operation_id_;
  }

  /**
   * Aggregate replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void Aggregate(const hipc::FullPtr<CustomTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * CoMutexTestTask - Test CoMutex functionality
 */
struct CoMutexTestTask : public chi::Task {
  IN chi::u32 test_id_;         // Test identifier
  IN chi::u32 hold_duration_ms_; // How long to hold the mutex

  /** SHM default constructor */
  CoMutexTestTask()
      : chi::Task(), test_id_(0), hold_duration_ms_(0) {}

  /** Emplace constructor */
  explicit CoMutexTestTask(
      const chi::TaskId &task_node,
      const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query,
      chi::u32 test_id,
      chi::u32 hold_duration_ms)
      : chi::Task(task_node, pool_id, pool_query, 20),
        test_id_(test_id), hold_duration_ms_(hold_duration_ms) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kCoMutexTest;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template<typename Archive>
  void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(test_id_, hold_duration_ms_);
  }

  template<typename Archive>
  void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    // No output parameters for this task
  }

  /**
   * Copy from another CoMutexTestTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<CoMutexTestTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy CoMutexTestTask-specific fields
    test_id_ = other->test_id_;
    hold_duration_ms_ = other->hold_duration_ms_;
  }

  /**
   * Aggregate replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void Aggregate(const hipc::FullPtr<CoMutexTestTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * CoRwLockTestTask - Test CoRwLock functionality
 */
struct CoRwLockTestTask : public chi::Task {
  IN chi::u32 test_id_;         // Test identifier
  IN bool is_writer_;           // True for write lock, false for read lock
  IN chi::u32 hold_duration_ms_; // How long to hold the lock

  /** SHM default constructor */
  CoRwLockTestTask()
      : chi::Task(), test_id_(0), is_writer_(false), hold_duration_ms_(0) {}

  /** Emplace constructor */
  explicit CoRwLockTestTask(
      const chi::TaskId &task_node,
      const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query,
      chi::u32 test_id,
      bool is_writer,
      chi::u32 hold_duration_ms)
      : chi::Task(task_node, pool_id, pool_query, 21),
        test_id_(test_id), is_writer_(is_writer), hold_duration_ms_(hold_duration_ms) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kCoRwLockTest;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template<typename Archive>
  void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(test_id_, is_writer_, hold_duration_ms_);
  }

  template<typename Archive>
  void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    // No output parameters for this task
  }

  /**
   * Copy from another CoRwLockTestTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<CoRwLockTestTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy CoRwLockTestTask-specific fields
    test_id_ = other->test_id_;
    is_writer_ = other->is_writer_;
    hold_duration_ms_ = other->hold_duration_ms_;
  }

  /**
   * Aggregate replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void Aggregate(const hipc::FullPtr<CoRwLockTestTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * WaitTestTask - Test recursive task->Wait() functionality
 * This task calls itself recursively "depth" times to test nested Wait() calls
 */
struct WaitTestTask : public chi::Task {
  IN chi::u32 depth_;              // Number of recursive calls to make
  IN chi::u32 test_id_;            // Test identifier for tracking
  INOUT chi::u32 current_depth_;   // Current recursion level (starts at 0)

  /** SHM default constructor */
  WaitTestTask()
      : chi::Task(), depth_(0), test_id_(0), current_depth_(0) {}

  /** Emplace constructor */
  explicit WaitTestTask(
      const chi::TaskId &task_node,
      const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query,
      chi::u32 depth,
      chi::u32 test_id)
      : chi::Task(task_node, pool_id, pool_query, 23),
        depth_(depth), test_id_(test_id), current_depth_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kWaitTest;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template<typename Archive>
  void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(depth_, test_id_, current_depth_);
  }

  template<typename Archive>
  void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar(current_depth_);  // Return the final depth reached
  }

  /**
   * Copy from another WaitTestTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<WaitTestTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy WaitTestTask-specific fields
    depth_ = other->depth_;
    test_id_ = other->test_id_;
    current_depth_ = other->current_depth_;
  }

  /**
   * Aggregate replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void Aggregate(const hipc::FullPtr<WaitTestTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * Standard DestroyTask for MOD_NAME
 * All ChiMods should use the same DestroyTask structure from admin
 */
using DestroyTask = chimaera::admin::DestroyTask;

} // namespace chimaera::MOD_NAME

#endif // MOD_NAME_TASKS_H_