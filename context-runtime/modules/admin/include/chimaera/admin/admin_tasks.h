#ifndef ADMIN_TASKS_H_
#define ADMIN_TASKS_H_

#include <chimaera/chimaera.h>
#include <chimaera/config_manager.h>
#include <yaml-cpp/yaml.h>

#include "autogen/admin_methods.h"

/**
 * Task struct definitions for Admin ChiMod
 *
 * Critical ChiMod for managing ChiPools and runtime lifecycle.
 * Responsible for pool creation/destruction and runtime shutdown.
 */

namespace chimaera::admin {

/**
 * CreateParams for admin chimod
 * Contains configuration parameters for admin container creation
 */
struct CreateParams {
  // Admin-specific parameters can be added here
  // For now, admin doesn't need special parameters beyond the base ones

  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "chimaera_admin";

  // Default constructor
  CreateParams() = default;

  // Serialization support for cereal
  template <class Archive> void serialize(Archive &ar) {
    // No additional fields to serialize for admin
  }

  /**
   * Load configuration from PoolConfig (for compose mode)
   * @param pool_config Pool configuration from compose section
   */
  void LoadConfig(const chi::PoolConfig &pool_config) {
    // Admin doesn't have additional configuration fields
    // YAML config parsing would go here for modules with config fields
    (void)pool_config; // Suppress unused parameter warning
  }
};

/**
 * BaseCreateTask - Templated base class for all ChiMod CreateTasks
 * @tparam CreateParamsT The parameter structure containing chimod-specific
 * configuration
 * @tparam MethodId The method ID for this task type
 * @tparam IS_ADMIN Whether this is an admin operation (sets volatile variable)
 * @tparam DO_COMPOSE Whether this task is called from compose (minimal error
 * checking)
 */
template <typename CreateParamsT, chi::u32 MethodId = Method::kCreate,
          bool IS_ADMIN = false, bool DO_COMPOSE = false>
struct BaseCreateTask : public chi::Task {
  // Pool operation parameters
  INOUT chi::priv::string chimod_name_;
  IN chi::priv::string pool_name_;
  INOUT chi::priv::string
      chimod_params_; // Serialized parameters for the specific ChiMod
  INOUT chi::PoolId new_pool_id_;

  // Results for pool operations
  OUT chi::priv::string error_message_;

  // Flags set by template parameters (must be serialized for remote execution)
  bool is_admin_;
  bool do_compose_;

  // Client pointer for PostWait callback (not serialized)
  chi::ContainerClient *client_;

  /** SHM default constructor */
  BaseCreateTask()
      : chi::Task(), chimod_name_(CHI_IPC->GetMainAlloc()), pool_name_(CHI_IPC->GetMainAlloc()),
        chimod_params_(CHI_IPC->GetMainAlloc()), new_pool_id_(chi::PoolId::GetNull()),
        error_message_(CHI_IPC->GetMainAlloc()), is_admin_(IS_ADMIN), do_compose_(DO_COMPOSE),
        client_(nullptr) {
    HLOG(kDebug, "BaseCreateTask default constructor: IS_ADMIN={}, DO_COMPOSE={}, do_compose_={}",
         IS_ADMIN, DO_COMPOSE, do_compose_);
  }

  /** Emplace constructor with CreateParams arguments */
  template <typename... CreateParamsArgs>
  explicit BaseCreateTask(const chi::TaskId &task_node,
                          const chi::PoolId &task_pool_id,
                          const chi::PoolQuery &pool_query,
                          const std::string &chimod_name,
                          const std::string &pool_name,
                          const chi::PoolId &target_pool_id,
                          chi::ContainerClient *client,
                          CreateParamsArgs &&...create_params_args)
      : chi::Task(task_node, task_pool_id, pool_query, 0),
        chimod_name_(CHI_IPC->GetMainAlloc(), chimod_name), pool_name_(CHI_IPC->GetMainAlloc(), pool_name),
        chimod_params_(CHI_IPC->GetMainAlloc()), new_pool_id_(target_pool_id),
        error_message_(CHI_IPC->GetMainAlloc()), is_admin_(IS_ADMIN), do_compose_(DO_COMPOSE),
        client_(client) {
    // Initialize base task
    task_id_ = task_node;
    method_ = MethodId;
    task_flags_.Clear();
    pool_query_ = pool_query;

    // In compose mode, skip CreateParams construction - PoolConfig will be set
    // via SetParams
    if (!do_compose_) {
      // Create and serialize the CreateParams with provided arguments
      CreateParamsT params(
          std::forward<CreateParamsArgs>(create_params_args)...);
      chi::Task::Serialize(CHI_IPC->GetMainAlloc(), chimod_params_, params);
    }
  }

  /** Compose constructor - takes PoolConfig directly */
  explicit BaseCreateTask(const chi::TaskId &task_node,
                          const chi::PoolId &task_pool_id,
                          const chi::PoolQuery &pool_query,
                          const chi::PoolConfig &pool_config)
      : chi::Task(task_node, task_pool_id, pool_query, 0),
        chimod_name_(CHI_IPC->GetMainAlloc(), pool_config.mod_name_),
        pool_name_(CHI_IPC->GetMainAlloc(), pool_config.pool_name_), chimod_params_(CHI_IPC->GetMainAlloc()),
        new_pool_id_(pool_config.pool_id_), error_message_(CHI_IPC->GetMainAlloc()),
        is_admin_(IS_ADMIN), do_compose_(DO_COMPOSE), client_(nullptr) {
    HLOG(kDebug, "BaseCreateTask COMPOSE constructor: IS_ADMIN={}, DO_COMPOSE={}, do_compose_={}, pool_name={}",
         IS_ADMIN, DO_COMPOSE, do_compose_, pool_config.pool_name_);
    // Initialize base task
    task_id_ = task_node;
    method_ = MethodId;
    task_flags_.Clear();
    pool_query_ = pool_query;

    // Serialize PoolConfig directly into chimod_params_
    chi::Task::Serialize(CHI_IPC->GetMainAlloc(), chimod_params_, pool_config);
  }

  /**
   * Set parameters by serializing them to chimod_params_
   * Does nothing if do_compose_ is true (compose mode)
   */
  template <typename... Args>
  void SetParams(AllocT* alloc,
                 Args &&...args) {
    if (do_compose_) {
      return; // Skip SetParams in compose mode
    }
    CreateParamsT params(std::forward<Args>(args)...);
    chi::Task::Serialize(alloc, chimod_params_, params);
  }

  /**
   * Get the CreateParams by deserializing from chimod_params_
   * In compose mode (do_compose_=true), deserializes PoolConfig and calls
   * LoadConfig
   */
  CreateParamsT
  GetParams(AllocT* alloc) const {
    if (do_compose_) {
      // Compose mode: deserialize PoolConfig and load into CreateParams
      chi::PoolConfig pool_config =
          chi::Task::Deserialize<chi::PoolConfig>(chimod_params_);
      CreateParamsT params;
      params.LoadConfig(pool_config);
      return params;
    } else {
      // Normal mode: deserialize CreateParams directly
      return chi::Task::Deserialize<CreateParamsT>(chimod_params_);
    }
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   * This includes: chimod_name_, pool_name_, chimod_params_, new_pool_id_, is_admin_, do_compose_
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    HLOG(kDebug, "BaseCreateTask::SerializeIn BEFORE: do_compose_={}, is_admin_={}", do_compose_, is_admin_);
    Task::SerializeIn(ar);
    ar(chimod_name_, pool_name_, chimod_params_, new_pool_id_, is_admin_, do_compose_);
    HLOG(kDebug, "BaseCreateTask::SerializeIn AFTER: do_compose_={}, is_admin_={}", do_compose_, is_admin_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: chimod_name_, chimod_params_, new_pool_id_, error_message_, is_admin_, do_compose_
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(chimod_name_, chimod_params_, new_pool_id_, error_message_, is_admin_, do_compose_);
  }

  /**
   * Copy from another BaseCreateTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<BaseCreateTask> &other) {
    HLOG(kDebug, "BaseCreateTask::Copy() BEFORE: this->do_compose_={}, other->do_compose_={}",
         do_compose_, other->do_compose_);
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy BaseCreateTask-specific fields
    chimod_name_ = other->chimod_name_;
    pool_name_ = other->pool_name_;
    chimod_params_ = other->chimod_params_;
    new_pool_id_ = other->new_pool_id_;
    error_message_ = other->error_message_;
    is_admin_ = other->is_admin_;
    do_compose_ = other->do_compose_;
    HLOG(kDebug, "BaseCreateTask::Copy() AFTER: this->do_compose_={}", do_compose_);
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<BaseCreateTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }

  /**
   * Post-wait callback called after task completion
   * Sets client_->pool_id_ and client_->return_code_ from task results
   */
  void PostWait() {
    if (client_ != nullptr) {
      client_->pool_id_ = new_pool_id_;
      client_->return_code_ = return_code_;
    }
  }
};

/**
 * CreateTask - Admin container creation task
 * Uses MethodId=kCreate and IS_ADMIN=true
 */
using CreateTask = BaseCreateTask<CreateParams, Method::kCreate, true>;

/**
 * GetOrCreatePoolTask - Template typedef for pool creation by external ChiMods
 * Other ChiMods should inherit this to create their pool creation tasks
 * @tparam CreateParamsT The parameter structure for the specific ChiMod
 */
template <typename CreateParamsT>
using GetOrCreatePoolTask =
    BaseCreateTask<CreateParamsT, Method::kGetOrCreatePool, false>;

/**
 * ComposeTask - Typedef for compose-based creation with minimal error checking
 * Used when creating pools from compose configuration
 * Uses kGetOrCreatePool method and IS_ADMIN=false to create pools in other
 * ChiMods
 * @tparam CreateParamsT The parameter structure for the specific ChiMod
 */
template <typename CreateParamsT>
using ComposeTask =
    BaseCreateTask<CreateParamsT, Method::kGetOrCreatePool, false, true>;

/**
 * DestroyPoolTask - Destroy an existing ChiPool
 */
struct DestroyPoolTask : public chi::Task {
  // Pool destruction parameters
  IN chi::PoolId target_pool_id_; ///< ID of pool to destroy
  IN chi::u32 destruction_flags_; ///< Flags controlling destruction behavior

  // Output results
  OUT chi::priv::string error_message_; ///< Error description if destruction failed

  /** SHM default constructor */
  DestroyPoolTask()
      : chi::Task(), target_pool_id_(), destruction_flags_(0),
        error_message_(CHI_IPC->GetMainAlloc()) {}

  /** Emplace constructor */
  explicit DestroyPoolTask(const chi::TaskId &task_node,
                           const chi::PoolId &pool_id,
                           const chi::PoolQuery &pool_query,
                           chi::PoolId target_pool_id,
                           chi::u32 destruction_flags = 0)
      : chi::Task(task_node, pool_id, pool_query, 10),
        target_pool_id_(target_pool_id), destruction_flags_(destruction_flags),
        error_message_(CHI_IPC->GetMainAlloc()) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kDestroyPool;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   * This includes: target_pool_id_, destruction_flags_
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(target_pool_id_, destruction_flags_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: error_message_
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(error_message_);
  }

  /**
   * Copy from another DestroyPoolTask (assumes this task is already
   * constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<DestroyPoolTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy DestroyPoolTask-specific fields
    target_pool_id_ = other->target_pool_id_;
    destruction_flags_ = other->destruction_flags_;
    error_message_ = other->error_message_;
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<DestroyPoolTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * StopRuntimeTask - Stop the entire Chimaera runtime
 */
struct StopRuntimeTask : public chi::Task {
  // Runtime shutdown parameters
  IN chi::u32 shutdown_flags_;  ///< Flags controlling shutdown behavior
  IN chi::u32 grace_period_ms_; ///< Grace period for clean shutdown

  // Output results
  OUT chi::priv::string error_message_; ///< Error description if shutdown failed

  /** SHM default constructor */
  StopRuntimeTask()
      : chi::Task(), shutdown_flags_(0), grace_period_ms_(5000),
        error_message_(CHI_IPC->GetMainAlloc()) {}

  /** Emplace constructor */
  explicit StopRuntimeTask(const chi::TaskId &task_node,
                           const chi::PoolId &pool_id,
                           const chi::PoolQuery &pool_query,
                           chi::u32 shutdown_flags = 0,
                           chi::u32 grace_period_ms = 5000)
      : chi::Task(task_node, pool_id, pool_query, 10),
        shutdown_flags_(shutdown_flags), grace_period_ms_(grace_period_ms),
        error_message_(CHI_IPC->GetMainAlloc()) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kStopRuntime;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   * This includes: shutdown_flags_, grace_period_ms_
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(shutdown_flags_, grace_period_ms_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: error_message_
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(error_message_);
  }

  /**
   * Copy from another StopRuntimeTask (assumes this task is already
   * constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<StopRuntimeTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy StopRuntimeTask-specific fields
    shutdown_flags_ = other->shutdown_flags_;
    grace_period_ms_ = other->grace_period_ms_;
    error_message_ = other->error_message_;
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<StopRuntimeTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * FlushTask - Flush administrative operations
 * Simple task with no additional inputs beyond basic task parameters
 */
struct FlushTask : public chi::Task {
  // Output results
  OUT chi::u64 total_work_done_; ///< Total amount of work remaining across all
                                 ///< containers

  /** SHM default constructor */
  FlushTask()
      : chi::Task(), total_work_done_(0) {}

  /** Emplace constructor */
  explicit FlushTask(const chi::TaskId &task_node, const chi::PoolId &pool_id,
                     const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, 10),
        total_work_done_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kFlush;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   * No additional parameters for FlushTask
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    // No additional parameters to serialize for flush
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: total_work_done_
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(total_work_done_);
  }

  /**
   * Copy from another FlushTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<FlushTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy FlushTask-specific fields
    total_work_done_ = other->total_work_done_;
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<FlushTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * Standard DestroyTask for reuse by all ChiMods
 * All ChiMods should use this same DestroyTask structure
 */
using DestroyTask = DestroyPoolTask;

/**
 * SendTask - Periodic task for sending queued tasks over network
 * Polls net_queue_ for tasks and sends them to remote nodes
 * This is a periodic task similar to RecvTask
 */
struct SendTask : public chi::Task {
  // Network transfer parameters
  IN chi::u32 transfer_flags_; ///< Flags controlling transfer behavior

  // Results
  OUT chi::priv::string error_message_; ///< Error description if transfer failed

  /** SHM default constructor */
  SendTask()
      : chi::Task(), transfer_flags_(0), error_message_(CHI_IPC->GetMainAlloc()) {}

  /** Emplace constructor */
  explicit SendTask(const chi::TaskId &task_node, const chi::PoolId &pool_id,
                    const chi::PoolQuery &pool_query,
                    chi::u32 transfer_flags = 0)
      : chi::Task(task_node, pool_id, pool_query, Method::kSend),
        transfer_flags_(transfer_flags), error_message_(CHI_IPC->GetMainAlloc()) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kSend;
    task_flags_.Clear();
    pool_query_ = pool_query;
    stat_.io_size_ = 1024 * 1024; // 1MB
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(transfer_flags_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(error_message_);
  }

  /**
   * Copy from another SendTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<SendTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy SendTask-specific fields
    transfer_flags_ = other->transfer_flags_;
    error_message_ = other->error_message_;
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<SendTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * RecvTask - Unified task for receiving task inputs or outputs from network
 * Replaces ServerRecvTaskIn and ClientRecvTaskOut
 * This is a periodic task that polls for incoming network messages
 */
struct RecvTask : public chi::Task {
  // Network transfer parameters
  IN chi::u32 transfer_flags_; ///< Flags controlling transfer behavior

  // Results
  OUT chi::priv::string error_message_; ///< Error description if transfer failed

  /** SHM default constructor */
  RecvTask()
      : chi::Task(), transfer_flags_(0), error_message_(CHI_IPC->GetMainAlloc()) {}

  /** Emplace constructor */
  explicit RecvTask(const chi::TaskId &task_node, const chi::PoolId &pool_id,
                    const chi::PoolQuery &pool_query,
                    chi::u32 transfer_flags = 0)
      : chi::Task(task_node, pool_id, pool_query, Method::kRecv),
        transfer_flags_(transfer_flags), error_message_(CHI_IPC->GetMainAlloc()) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kRecv;
    task_flags_.Clear();
    pool_query_ = pool_query;
    stat_.io_size_ = 1024 * 1024; // 1MB
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(transfer_flags_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(error_message_);
  }

  /**
   * Copy from another RecvTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<RecvTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy RecvTask-specific fields
    transfer_flags_ = other->transfer_flags_;
    error_message_ = other->error_message_;
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<RecvTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * HeartbeatTask - Runtime health check
 * Used to verify runtime is alive and responding
 * Returns 0 on success to indicate runtime is healthy
 */
struct HeartbeatTask : public chi::Task {
  // Heartbeat response
  OUT int32_t response_; ///< 0 = success, non-zero = error

  /** SHM default constructor */
  HeartbeatTask()
      : chi::Task(), response_(-1) {}

  /** Emplace constructor */
  explicit HeartbeatTask(const chi::TaskId &task_node, const chi::PoolId &pool_id,
                         const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, Method::kHeartbeat),
        response_(-1) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kHeartbeat;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   * No additional parameters for HeartbeatTask
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    // No additional parameters to serialize for heartbeat
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: response_
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(response_);
  }

  /**
   * Copy from another HeartbeatTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<HeartbeatTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy HeartbeatTask-specific fields
    response_ = other->response_;
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<HeartbeatTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

} // namespace chimaera::admin

#endif // ADMIN_TASKS_H_