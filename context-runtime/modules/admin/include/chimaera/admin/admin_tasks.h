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

#ifndef ADMIN_TASKS_H_
#define ADMIN_TASKS_H_

#include <chimaera/chimaera.h>
#include <chimaera/config_manager.h>
#include <hermes_shm/memory/allocator/malloc_allocator.h>
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
  template <class Archive>
  void serialize(Archive &ar) {
    // No additional fields to serialize for admin
  }

  /**
   * Load configuration from PoolConfig (for compose mode)
   * @param pool_config Pool configuration from compose section
   */
  void LoadConfig(const chi::PoolConfig &pool_config) {
    // Admin doesn't have additional configuration fields
    // YAML config parsing would go here for modules with config fields
    (void)pool_config;  // Suppress unused parameter warning
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
      chimod_params_;  // Serialized parameters for the specific ChiMod
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
      : chi::Task(),
        chimod_name_(HSHM_MALLOC),
        pool_name_(HSHM_MALLOC),
        chimod_params_(HSHM_MALLOC),
        new_pool_id_(chi::PoolId::GetNull()),
        error_message_(HSHM_MALLOC),
        is_admin_(IS_ADMIN),
        do_compose_(DO_COMPOSE),
        client_(nullptr) {
    HLOG(kDebug,
         "BaseCreateTask default constructor: IS_ADMIN={}, DO_COMPOSE={}, "
         "do_compose_={}",
         IS_ADMIN, DO_COMPOSE, do_compose_);
  }

  /** Emplace constructor with CreateParams arguments */
  template <typename... CreateParamsArgs>
  explicit BaseCreateTask(
      const chi::TaskId &task_node, const chi::PoolId &task_pool_id,
      const chi::PoolQuery &pool_query, const std::string &chimod_name,
      const std::string &pool_name, const chi::PoolId &target_pool_id,
      chi::ContainerClient *client, CreateParamsArgs &&...create_params_args)
      : chi::Task(task_node, task_pool_id, pool_query, 0),
        chimod_name_(HSHM_MALLOC, chimod_name),
        pool_name_(HSHM_MALLOC, pool_name),
        chimod_params_(HSHM_MALLOC),
        new_pool_id_(target_pool_id),
        error_message_(HSHM_MALLOC),
        is_admin_(IS_ADMIN),
        do_compose_(DO_COMPOSE),
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
      chi::Task::Serialize(HSHM_MALLOC, chimod_params_, params);
    }
  }

  /** Compose constructor - takes PoolConfig directly */
  explicit BaseCreateTask(const chi::TaskId &task_node,
                          const chi::PoolId &task_pool_id,
                          const chi::PoolQuery &pool_query,
                          const chi::PoolConfig &pool_config)
      : chi::Task(task_node, task_pool_id, pool_query, 0),
        chimod_name_(HSHM_MALLOC, pool_config.mod_name_),
        pool_name_(HSHM_MALLOC, pool_config.pool_name_),
        chimod_params_(HSHM_MALLOC),
        new_pool_id_(pool_config.pool_id_),
        error_message_(HSHM_MALLOC),
        is_admin_(IS_ADMIN),
        do_compose_(DO_COMPOSE),
        client_(nullptr) {
    HLOG(kDebug,
         "BaseCreateTask COMPOSE constructor: IS_ADMIN={}, DO_COMPOSE={}, "
         "do_compose_={}, pool_name={}",
         IS_ADMIN, DO_COMPOSE, do_compose_, pool_config.pool_name_);
    // Initialize base task
    task_id_ = task_node;
    method_ = MethodId;
    task_flags_.Clear();
    pool_query_ = pool_query;

    // Serialize PoolConfig directly into chimod_params_
    chi::Task::Serialize(HSHM_MALLOC, chimod_params_, pool_config);
  }

  /**
   * Set parameters by serializing them to chimod_params_
   * Does nothing if do_compose_ is true (compose mode)
   */
  template <typename... Args>
  void SetParams(Args &&...args) {
    if (do_compose_) {
      return;  // Skip SetParams in compose mode
    }
    CreateParamsT params(std::forward<Args>(args)...);
    chi::Task::Serialize(HSHM_MALLOC, chimod_params_, params);
  }

  /**
   * Get the CreateParams by deserializing from chimod_params_
   * In compose mode (do_compose_=true), deserializes PoolConfig and calls
   * LoadConfig
   */
  CreateParamsT GetParams() const {
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
   * This includes: chimod_name_, pool_name_, chimod_params_, new_pool_id_,
   * is_admin_, do_compose_
   */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    HLOG(kDebug,
         "BaseCreateTask::SerializeIn BEFORE: do_compose_={}, is_admin_={}",
         do_compose_, is_admin_);
    Task::SerializeIn(ar);
    ar(chimod_name_, pool_name_, chimod_params_, new_pool_id_, is_admin_,
       do_compose_);
    HLOG(kDebug,
         "BaseCreateTask::SerializeIn AFTER: do_compose_={}, is_admin_={}",
         do_compose_, is_admin_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: chimod_name_, chimod_params_, new_pool_id_, error_message_,
   * is_admin_, do_compose_
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(chimod_name_, chimod_params_, new_pool_id_, error_message_, is_admin_,
       do_compose_);
  }

  /**
   * Copy from another BaseCreateTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<BaseCreateTask> &other) {
    HLOG(kDebug,
         "BaseCreateTask::Copy() BEFORE: this->do_compose_={}, "
         "other->do_compose_={}",
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
    HLOG(kDebug, "BaseCreateTask::Copy() AFTER: this->do_compose_={}",
         do_compose_);
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
  IN chi::PoolId target_pool_id_;  ///< ID of pool to destroy
  IN chi::u32 destruction_flags_;  ///< Flags controlling destruction behavior

  // Output results
  OUT chi::priv::string
      error_message_;  ///< Error description if destruction failed

  /** SHM default constructor */
  DestroyPoolTask()
      : chi::Task(),
        target_pool_id_(),
        destruction_flags_(0),
        error_message_(HSHM_MALLOC) {}

  /** Emplace constructor */
  explicit DestroyPoolTask(const chi::TaskId &task_node,
                           const chi::PoolId &pool_id,
                           const chi::PoolQuery &pool_query,
                           chi::PoolId target_pool_id,
                           chi::u32 destruction_flags = 0)
      : chi::Task(task_node, pool_id, pool_query, 10),
        target_pool_id_(target_pool_id),
        destruction_flags_(destruction_flags),
        error_message_(HSHM_MALLOC) {
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
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(target_pool_id_, destruction_flags_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: error_message_
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
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
  IN chi::u32 shutdown_flags_;   ///< Flags controlling shutdown behavior
  IN chi::u32 grace_period_ms_;  ///< Grace period for clean shutdown

  // Output results
  OUT chi::priv::string
      error_message_;  ///< Error description if shutdown failed

  /** SHM default constructor */
  StopRuntimeTask()
      : chi::Task(),
        shutdown_flags_(0),
        grace_period_ms_(5000),
        error_message_(HSHM_MALLOC) {}

  /** Emplace constructor */
  explicit StopRuntimeTask(const chi::TaskId &task_node,
                           const chi::PoolId &pool_id,
                           const chi::PoolQuery &pool_query,
                           chi::u32 shutdown_flags = 0,
                           chi::u32 grace_period_ms = 5000)
      : chi::Task(task_node, pool_id, pool_query, 10),
        shutdown_flags_(shutdown_flags),
        grace_period_ms_(grace_period_ms),
        error_message_(HSHM_MALLOC) {
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
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(shutdown_flags_, grace_period_ms_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: error_message_
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
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
  OUT chi::u64 total_work_done_;  ///< Total amount of work remaining across all
                                  ///< containers

  /** SHM default constructor */
  FlushTask() : chi::Task(), total_work_done_(0) {}

  /** Emplace constructor */
  explicit FlushTask(const chi::TaskId &task_node, const chi::PoolId &pool_id,
                     const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, 10), total_work_done_(0) {
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
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    // No additional parameters to serialize for flush
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: total_work_done_
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
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
  IN chi::u32 transfer_flags_;  ///< Flags controlling transfer behavior

  // Results
  OUT chi::priv::string
      error_message_;  ///< Error description if transfer failed

  /** SHM default constructor */
  SendTask() : chi::Task(), transfer_flags_(0), error_message_(HSHM_MALLOC) {}

  /** Emplace constructor */
  explicit SendTask(const chi::TaskId &task_node, const chi::PoolId &pool_id,
                    const chi::PoolQuery &pool_query,
                    chi::u32 transfer_flags = 0)
      : chi::Task(task_node, pool_id, pool_query, Method::kSend),
        transfer_flags_(transfer_flags),
        error_message_(HSHM_MALLOC) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kSend;
    task_flags_.Clear();
    pool_query_ = pool_query;
    stat_.io_size_ = 1024 * 1024;  // 1MB
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(transfer_flags_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
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
  IN chi::u32 transfer_flags_;  ///< Flags controlling transfer behavior

  // Results
  OUT chi::priv::string
      error_message_;  ///< Error description if transfer failed

  /** SHM default constructor */
  RecvTask() : chi::Task(), transfer_flags_(0), error_message_(HSHM_MALLOC) {}

  /** Emplace constructor */
  explicit RecvTask(const chi::TaskId &task_node, const chi::PoolId &pool_id,
                    const chi::PoolQuery &pool_query,
                    chi::u32 transfer_flags = 0)
      : chi::Task(task_node, pool_id, pool_query, Method::kRecv),
        transfer_flags_(transfer_flags),
        error_message_(HSHM_MALLOC) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kRecv;
    task_flags_.Clear();
    pool_query_ = pool_query;
    stat_.io_size_ = 1024 * 1024;  // 1MB
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(transfer_flags_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
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
 * ClientConnectTask - Client connection handshake
 * Received via lightbeam transport and responds with success
 * Returns 0 on success to indicate runtime is healthy
 */
struct ClientConnectTask : public chi::Task {
  // Connect response
  OUT int32_t response_;  ///< 0 = success, non-zero = error
  OUT chi::u64 server_generation_;  ///< Server's generation counter for restart detection

  /** SHM default constructor */
  ClientConnectTask() : chi::Task(), response_(-1), server_generation_(0) {}

  /** Emplace constructor */
  explicit ClientConnectTask(const chi::TaskId &task_node,
                             const chi::PoolId &pool_id,
                             const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, Method::kClientConnect),
        response_(-1),
        server_generation_(0) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kClientConnect;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(response_, server_generation_);
  }

  void Copy(const hipc::FullPtr<ClientConnectTask> &other) {
    Task::Copy(other.template Cast<Task>());
    response_ = other->response_;
    server_generation_ = other->server_generation_;
  }

  void Aggregate(const hipc::FullPtr<ClientConnectTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * ClientRecvTask - Receive tasks from ZMQ clients (TCP/IPC)
 * Periodic task that polls ZMQ ROUTER sockets for client task submissions
 */
struct ClientRecvTask : public chi::Task {
  OUT chi::u32 tasks_received_;

  /** SHM default constructor */
  ClientRecvTask() : chi::Task(), tasks_received_(0) {}

  /** Emplace constructor */
  explicit ClientRecvTask(const chi::TaskId &task_node,
                          const chi::PoolId &pool_id,
                          const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, Method::kClientRecv),
        tasks_received_(0) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kClientRecv;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(tasks_received_);
  }

  void Copy(const hipc::FullPtr<ClientRecvTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tasks_received_ = other->tasks_received_;
  }

  void Aggregate(const hipc::FullPtr<ClientRecvTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * ClientSendTask - Send completed task outputs to ZMQ clients
 * Periodic task that polls net_queue_ kClientSendTcp/kClientSendIpc priorities
 */
struct ClientSendTask : public chi::Task {
  OUT chi::u32 tasks_sent_;

  /** SHM default constructor */
  ClientSendTask() : chi::Task(), tasks_sent_(0) {}

  /** Emplace constructor */
  explicit ClientSendTask(const chi::TaskId &task_node,
                          const chi::PoolId &pool_id,
                          const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, Method::kClientSend),
        tasks_sent_(0) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kClientSend;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(tasks_sent_);
  }

  void Copy(const hipc::FullPtr<ClientSendTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tasks_sent_ = other->tasks_sent_;
  }

  void Aggregate(const hipc::FullPtr<ClientSendTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * WreapDeadIpcsTask - Periodic task to reap shared memory from dead processes
 *
 * This task periodically calls IpcManager::WreapDeadIpcs() to clean up
 * shared memory segments belonging to processes that have terminated.
 * Scheduled by default every second during admin container creation.
 */
struct WreapDeadIpcsTask : public chi::Task {
  // Output: Number of segments reaped in this invocation
  OUT chi::u64 reaped_count_;

  /** SHM default constructor */
  WreapDeadIpcsTask() : chi::Task(), reaped_count_(0) {}

  /** Emplace constructor */
  explicit WreapDeadIpcsTask(const chi::TaskId &task_node,
                             const chi::PoolId &pool_id,
                             const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, Method::kWreapDeadIpcs),
        reaped_count_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kWreapDeadIpcs;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   * No additional parameters for WreapDeadIpcsTask
   */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    // No additional parameters to serialize
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: reaped_count_
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(reaped_count_);
  }

  /**
   * Copy from another WreapDeadIpcsTask (assumes this task is already
   * constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<WreapDeadIpcsTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy WreapDeadIpcsTask-specific fields
    reaped_count_ = other->reaped_count_;
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<WreapDeadIpcsTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * MonitorTask - Monitor runtime and worker statistics
 *
 * This task collects statistics from all workers in the runtime including:
 * - Number of queued, blocked, and periodic tasks
 * - Worker idle status and suspend periods
 * - Overall system load and utilization
 */
struct MonitorTask : public chi::Task {
  /** Output: Vector of worker statistics */
  OUT std::vector<chi::WorkerStats> info_;

  /**
   * SHM default constructor
   */
  MonitorTask() : chi::Task(), info_() {}

  /**
   * Emplace constructor - create new MonitorTask
   * @param task_node Unique task identifier
   * @param pool_id Pool this task belongs to
   * @param pool_query Query for routing this task
   */
  explicit MonitorTask(const chi::TaskId &task_node, const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, Method::kMonitor), info_() {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kMonitor;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   * No additional parameters for MonitorTask
   */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    // No additional parameters to serialize
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: info_ (vector of WorkerStats)
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(info_);
  }

  /**
   * Copy from another MonitorTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<MonitorTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy MonitorTask-specific fields
    info_ = other->info_;
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<MonitorTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * TaskBatch - Container for batch task submission
 * Stores task metadata and serialized task data for efficient batch submission
 */
class TaskBatch {
 private:
  std::vector<chi::LocalTaskInfo>
      task_infos_;                    /**< Task metadata for deserialization */
  std::vector<char> serialized_data_; /**< Serialized task data */

 public:
  /**
   * Default constructor
   */
  TaskBatch() = default;

  /**
   * Add a task to the batch
   * @tparam TaskT Task type to add
   * @tparam Args Constructor argument types
   * @param args Arguments to pass to task constructor
   */
  template <typename TaskT, typename... Args>
  void Add(Args &&...args) {
    // Create new task in IPC
    auto task = CHI_IPC->NewTask<TaskT>(std::forward<Args>(args)...);

    // Serialize task inputs using LocalSaveTaskArchive
    chi::LocalSaveTaskArchive archive(chi::LocalMsgType::kSerializeIn);
    archive << (*task);

    // Record task info
    const auto &task_infos = archive.GetTaskInfos();
    if (!task_infos.empty()) {
      task_infos_.insert(task_infos_.end(), task_infos.begin(),
                         task_infos.end());
    }

    // Append serialized data
    const auto &data = archive.GetData();
    serialized_data_.insert(serialized_data_.end(), data.begin(), data.end());
  }

  /**
   * Get task infos
   * @return Vector of task information
   */
  const std::vector<chi::LocalTaskInfo> &GetTaskInfos() const {
    return task_infos_;
  }

  /**
   * Get serialized data
   * @return Vector of serialized task data
   */
  const std::vector<char> &GetSerializedData() const {
    return serialized_data_;
  }

  /**
   * Get number of tasks in batch
   * @return Number of tasks
   */
  size_t GetTaskCount() const { return task_infos_.size(); }

  /**
   * Serialize for cereal
   * @tparam Archive Archive type
   * @param ar Archive instance
   */
  template <typename Archive>
  void serialize(Archive &ar) {
    ar(task_infos_, serialized_data_);
  }
};

/**
 * SubmitBatchTask - Submit a batch of tasks in a single RPC
 * Allows efficient submission of multiple tasks with minimal network overhead
 */
struct SubmitBatchTask : public chi::Task {
  // Batch task data
  IN std::vector<chi::LocalTaskInfo> task_infos_;  ///< Task metadata
  IN std::vector<char> serialized_data_;           ///< Serialized task data

  // Results
  OUT chi::u32 tasks_completed_;         ///< Number of tasks completed
  OUT chi::priv::string error_message_;  ///< Error description if failed

  /**
   * SHM default constructor
   */
  SubmitBatchTask()
      : chi::Task(),
        task_infos_(),
        serialized_data_(),
        tasks_completed_(0),
        error_message_(HSHM_MALLOC) {}

  /**
   * Emplace constructor
   */
  explicit SubmitBatchTask(const chi::TaskId &task_node,
                           const chi::PoolId &pool_id,
                           const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, Method::kSubmitBatch),
        task_infos_(),
        serialized_data_(),
        tasks_completed_(0),
        error_message_(HSHM_MALLOC) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kSubmitBatch;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Constructor with batch data
   */
  explicit SubmitBatchTask(const chi::TaskId &task_node,
                           const chi::PoolId &pool_id,
                           const chi::PoolQuery &pool_query,
                           const TaskBatch &batch)
      : chi::Task(task_node, pool_id, pool_query, Method::kSubmitBatch),
        task_infos_(batch.GetTaskInfos()),
        serialized_data_(batch.GetSerializedData()),
        tasks_completed_(0),
        error_message_(HSHM_MALLOC) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kSubmitBatch;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters for network transfer
   * This includes: task_infos_, serialized_data_
   */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(task_infos_, serialized_data_);
  }

  /**
   * Serialize OUT and INOUT parameters for network transfer
   * This includes: tasks_completed_, error_message_
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(tasks_completed_, error_message_);
  }

  /**
   * Copy from another SubmitBatchTask
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<SubmitBatchTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy SubmitBatchTask-specific fields
    task_infos_ = other->task_infos_;
    serialized_data_ = other->serialized_data_;
    tasks_completed_ = other->tasks_completed_;
    error_message_ = other->error_message_;
  }

  /**
   * Aggregate replica results into this task
   */
  void Aggregate(const hipc::FullPtr<SubmitBatchTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * RegisterAcceleratorMemoryTask - Register GPU accelerator memory with runtime
 *
 * This task is called from GPU kernels to register a GPU memory backend
 * with the Chimaera runtime. The runtime can then use this memory for
 * allocations within GPU kernels.
 */
// TODO: RegisterAcceleratorMemoryTask - incomplete, needs Method::kRegisterAcceleratorMemory defined
// struct RegisterAcceleratorMemoryTask : public chi::Task {
//   // Backend information for GPU memory
//   IN chi::u64 backend_id_;           ///< Backend ID
//   IN chi::u64 data_capacity_;        ///< GPU memory capacity in bytes
//   IN chi::u32 gpu_id_;               ///< GPU device ID
//
//   // Results
//   OUT chi::priv::string error_message_;  ///< Error description if registration failed
//
//   /** SHM default constructor */
//   RegisterAcceleratorMemoryTask()
//       : chi::Task(),
//         backend_id_(0),
//         data_capacity_(0),
//         gpu_id_(0),
//         error_message_(HSHM_MALLOC) {}
//
//   /** Emplace constructor */
//   explicit RegisterAcceleratorMemoryTask(const chi::TaskId &task_node,
//                                          const chi::PoolId &pool_id,
//                                          const chi::PoolQuery &pool_query,
//                                          chi::u64 backend_id,
//                                          chi::u64 data_capacity,
//                                          chi::u32 gpu_id)
//       : chi::Task(task_node, pool_id, pool_query, Method::kRegisterAcceleratorMemory),
//         backend_id_(backend_id),
//         data_capacity_(data_capacity),
//         gpu_id_(gpu_id),
//         error_message_(HSHM_MALLOC) {
//     // Initialize task
//     task_id_ = task_node;
//     pool_id_ = pool_id;
//     method_ = Method::kRegisterAcceleratorMemory;
//     task_flags_.Clear();
//     pool_query_ = pool_query;
//   }
//
//   /**
//    * Serialize IN and INOUT parameters for network transfer
//    * This includes: backend_id_, data_capacity_, gpu_id_
//    */
//   template <typename Archive>
//   void SerializeIn(Archive &ar) {
//     Task::SerializeIn(ar);
//     ar(backend_id_, data_capacity_, gpu_id_);
//   }
//
//   /**
//    * Serialize OUT and INOUT parameters for network transfer
//    * This includes: error_message_
//    */
//   template <typename Archive>
//   void SerializeOut(Archive &ar) {
//     Task::SerializeOut(ar);
//     ar(error_message_);
//   }
//
//   /**
//    * Copy from another RegisterAcceleratorMemoryTask
//    * @param other Pointer to the source task to copy from
//    */
//   void Copy(const hipc::FullPtr<RegisterAcceleratorMemoryTask> &other) {
//     // Copy base Task fields
//     Task::Copy(other.template Cast<Task>());
//     // Copy RegisterAcceleratorMemoryTask-specific fields
//     backend_id_ = other->backend_id_;
//     data_capacity_ = other->data_capacity_;
//     gpu_id_ = other->gpu_id_;
//     error_message_ = other->error_message_;
//   }
//
//   /** Aggregate replica results into this task */
//   void Aggregate(const hipc::FullPtr<RegisterAcceleratorMemoryTask> &other) {
//     Task::Aggregate(other.template Cast<Task>());
//     Copy(other);
//   }
// };

/**
 * RegisterMemoryTask - Register client shared memory with runtime
 *
 * When a SHM-mode client creates a new shared memory segment via
 * IncreaseMemory(), it sends this task over TCP to tell the runtime
 * server to attach to the new segment.
 */
struct RegisterMemoryTask : public chi::Task {
  IN chi::u32 alloc_major_;  ///< AllocatorId major (pid)
  IN chi::u32 alloc_minor_;  ///< AllocatorId minor (index)
  OUT bool success_;

  /** SHM default constructor */
  RegisterMemoryTask()
      : chi::Task(), alloc_major_(0), alloc_minor_(0), success_(false) {}

  /** Emplace constructor */
  explicit RegisterMemoryTask(const chi::TaskId &task_node,
                              const chi::PoolId &pool_id,
                              const chi::PoolQuery &pool_query,
                              const hipc::AllocatorId &alloc_id)
      : chi::Task(task_node, pool_id, pool_query, Method::kRegisterMemory),
        alloc_major_(alloc_id.major_),
        alloc_minor_(alloc_id.minor_),
        success_(false) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kRegisterMemory;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(alloc_major_, alloc_minor_);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(success_);
  }

  void Copy(const hipc::FullPtr<RegisterMemoryTask> &other) {
    Task::Copy(other.template Cast<Task>());
    alloc_major_ = other->alloc_major_;
    alloc_minor_ = other->alloc_minor_;
    success_ = other->success_;
  }

  void Aggregate(const hipc::FullPtr<RegisterMemoryTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * RestartContainersTask - Restart containers from saved compose configs
 * Reads conf_dir/restart/ directory and re-creates pools from saved YAML files
 */
struct RestartContainersTask : public chi::Task {
  OUT chi::u32 containers_restarted_;
  OUT chi::priv::string error_message_;

  /** SHM default constructor */
  RestartContainersTask()
      : chi::Task(),
        containers_restarted_(0),
        error_message_(HSHM_MALLOC) {}

  /** Emplace constructor */
  explicit RestartContainersTask(const chi::TaskId &task_node,
                                 const chi::PoolId &pool_id,
                                 const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, Method::kRestartContainers),
        containers_restarted_(0),
        error_message_(HSHM_MALLOC) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kRestartContainers;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(containers_restarted_, error_message_);
  }

  void Copy(const hipc::FullPtr<RestartContainersTask> &other) {
    Task::Copy(other.template Cast<Task>());
    containers_restarted_ = other->containers_restarted_;
    error_message_ = other->error_message_;
  }

  void Aggregate(const hipc::FullPtr<RestartContainersTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * AddNodeTask - Register a new node with all existing nodes in the cluster
 * Broadcasts to all nodes to update their internal hostfile
 */
struct AddNodeTask : public chi::Task {
  IN chi::priv::string new_node_ip_;
  IN chi::u32 new_node_port_;
  OUT chi::u64 new_node_id_;
  OUT chi::priv::string error_message_;

  /** SHM default constructor */
  AddNodeTask()
      : chi::Task(),
        new_node_ip_(HSHM_MALLOC),
        new_node_port_(0),
        new_node_id_(0),
        error_message_(HSHM_MALLOC) {}

  /** Emplace constructor */
  explicit AddNodeTask(const chi::TaskId &task_node,
                       const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query,
                       const std::string &new_node_ip,
                       chi::u32 new_node_port)
      : chi::Task(task_node, pool_id, pool_query, Method::kAddNode),
        new_node_ip_(HSHM_MALLOC, new_node_ip),
        new_node_port_(new_node_port),
        new_node_id_(0),
        error_message_(HSHM_MALLOC) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kAddNode;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(new_node_ip_, new_node_port_);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(new_node_id_, error_message_);
  }

  void Copy(const hipc::FullPtr<AddNodeTask> &other) {
    Task::Copy(other.template Cast<Task>());
    new_node_ip_ = other->new_node_ip_;
    new_node_port_ = other->new_node_port_;
    new_node_id_ = other->new_node_id_;
    error_message_ = other->error_message_;
  }

  void Aggregate(const hipc::FullPtr<AddNodeTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * ChangeAddressTableTask - Update ContainerId->NodeId mapping on a node
 * Broadcasts to all nodes to update their address table for a pool
 */
struct ChangeAddressTableTask : public chi::Task {
  IN chi::PoolId target_pool_id_;
  IN chi::ContainerId container_id_;
  IN chi::u32 new_node_id_;
  OUT chi::priv::string error_message_;

  /** SHM default constructor */
  ChangeAddressTableTask()
      : chi::Task(),
        target_pool_id_(),
        container_id_(0),
        new_node_id_(0),
        error_message_(HSHM_MALLOC) {}

  /** Emplace constructor */
  explicit ChangeAddressTableTask(const chi::TaskId &task_node,
                                   const chi::PoolId &pool_id,
                                   const chi::PoolQuery &pool_query,
                                   const chi::PoolId &target_pool_id,
                                   chi::ContainerId container_id,
                                   chi::u32 new_node_id)
      : chi::Task(task_node, pool_id, pool_query, Method::kChangeAddressTable),
        target_pool_id_(target_pool_id),
        container_id_(container_id),
        new_node_id_(new_node_id),
        error_message_(HSHM_MALLOC) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kChangeAddressTable;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(target_pool_id_, container_id_, new_node_id_);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(error_message_);
  }

  void Copy(const hipc::FullPtr<ChangeAddressTableTask> &other) {
    Task::Copy(other.template Cast<Task>());
    target_pool_id_ = other->target_pool_id_;
    container_id_ = other->container_id_;
    new_node_id_ = other->new_node_id_;
    error_message_ = other->error_message_;
  }

  void Aggregate(const hipc::FullPtr<ChangeAddressTableTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * MigrateContainersTask - Orchestrate container migration
 * Processes a list of MigrateInfo entries to move containers between nodes
 */
struct MigrateContainersTask : public chi::Task {
  IN chi::priv::string migrations_json_;
  OUT chi::u32 num_migrated_;
  OUT chi::priv::string error_message_;

  /** SHM default constructor */
  MigrateContainersTask()
      : chi::Task(),
        migrations_json_(HSHM_MALLOC),
        num_migrated_(0),
        error_message_(HSHM_MALLOC) {}

  /** Emplace constructor */
  explicit MigrateContainersTask(const chi::TaskId &task_node,
                                  const chi::PoolId &pool_id,
                                  const chi::PoolQuery &pool_query,
                                  const std::string &migrations_json)
      : chi::Task(task_node, pool_id, pool_query, Method::kMigrateContainers),
        migrations_json_(HSHM_MALLOC, migrations_json),
        num_migrated_(0),
        error_message_(HSHM_MALLOC) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kMigrateContainers;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(migrations_json_);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(num_migrated_, error_message_);
  }

  void Copy(const hipc::FullPtr<MigrateContainersTask> &other) {
    Task::Copy(other.template Cast<Task>());
    migrations_json_ = other->migrations_json_;
    num_migrated_ = other->num_migrated_;
    error_message_ = other->error_message_;
  }

  void Aggregate(const hipc::FullPtr<MigrateContainersTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * HeartbeatTask - Liveness probe that just returns success
 * No input or output fields beyond base Task
 */
struct HeartbeatTask : public chi::Task {
  /** SHM default constructor */
  HeartbeatTask() : chi::Task() {}

  /** Emplace constructor */
  explicit HeartbeatTask(const chi::TaskId &task_node,
                         const chi::PoolId &pool_id,
                         const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, Method::kHeartbeat) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kHeartbeat;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }

  void Copy(const hipc::FullPtr<HeartbeatTask> &other) {
    Task::Copy(other.template Cast<Task>());
  }

  void Aggregate(const hipc::FullPtr<HeartbeatTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * HeartbeatProbeTask - Periodic SWIM failure detector
 * Local periodic task, no extra fields needed
 */
struct HeartbeatProbeTask : public chi::Task {
  /** SHM default constructor */
  HeartbeatProbeTask() : chi::Task() {}

  /** Emplace constructor */
  explicit HeartbeatProbeTask(const chi::TaskId &task_node,
                              const chi::PoolId &pool_id,
                              const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, Method::kHeartbeatProbe) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kHeartbeatProbe;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }

  void Copy(const hipc::FullPtr<HeartbeatProbeTask> &other) {
    Task::Copy(other.template Cast<Task>());
  }

  void Aggregate(const hipc::FullPtr<HeartbeatProbeTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * ProbeRequestTask - Indirect probe request to helper node
 * Remote task: asks a helper node to probe a target on our behalf
 */
struct ProbeRequestTask : public chi::Task {
  IN chi::u64 target_node_id_;   // node to probe on behalf of requester
  OUT int32_t probe_result_;     // 0 = alive, -1 = unreachable

  /** SHM default constructor */
  ProbeRequestTask() : chi::Task(), target_node_id_(0), probe_result_(-1) {}

  /** Emplace constructor */
  explicit ProbeRequestTask(const chi::TaskId &task_node,
                            const chi::PoolId &pool_id,
                            const chi::PoolQuery &pool_query,
                            chi::u64 target_node_id)
      : chi::Task(task_node, pool_id, pool_query, Method::kProbeRequest),
        target_node_id_(target_node_id),
        probe_result_(-1) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kProbeRequest;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(target_node_id_);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(probe_result_);
  }

  void Copy(const hipc::FullPtr<ProbeRequestTask> &other) {
    Task::Copy(other.template Cast<Task>());
    target_node_id_ = other->target_node_id_;
    probe_result_ = other->probe_result_;
  }

  void Aggregate(const hipc::FullPtr<ProbeRequestTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * RecoverContainersTask - Leader broadcasts recovery plan to surviving nodes
 * Each node updates address_map_, only dest node creates the container
 */
struct RecoverContainersTask : public chi::Task {
  IN chi::priv::string assignments_data_;  // Serialized vector<RecoveryAssignment>
  IN chi::u64 dead_node_id_;
  OUT chi::u32 num_recovered_;
  OUT chi::priv::string error_message_;

  /** SHM default constructor */
  RecoverContainersTask()
      : chi::Task(),
        assignments_data_(HSHM_MALLOC),
        dead_node_id_(0),
        num_recovered_(0),
        error_message_(HSHM_MALLOC) {}

  /** Emplace constructor */
  explicit RecoverContainersTask(const chi::TaskId &task_node,
                                  const chi::PoolId &pool_id,
                                  const chi::PoolQuery &pool_query,
                                  const std::string &assignments_data,
                                  chi::u64 dead_node_id)
      : chi::Task(task_node, pool_id, pool_query, Method::kRecoverContainers),
        assignments_data_(HSHM_MALLOC, assignments_data),
        dead_node_id_(dead_node_id),
        num_recovered_(0),
        error_message_(HSHM_MALLOC) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kRecoverContainers;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(assignments_data_, dead_node_id_);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(num_recovered_, error_message_);
  }

  void Copy(const hipc::FullPtr<RecoverContainersTask> &other) {
    Task::Copy(other.template Cast<Task>());
    assignments_data_ = other->assignments_data_;
    dead_node_id_ = other->dead_node_id_;
    num_recovered_ = other->num_recovered_;
    error_message_ = other->error_message_;
  }

  void Aggregate(const hipc::FullPtr<RecoverContainersTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

}  // namespace chimaera::admin

#endif  // ADMIN_TASKS_H_