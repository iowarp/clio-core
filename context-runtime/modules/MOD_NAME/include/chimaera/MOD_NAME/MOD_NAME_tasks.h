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
        data_(HSHM_MALLOC), operation_id_(0) {
  }

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

  /** Destructor */
  ~CustomTask() {
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
    HLOG(kInfo, "CustomTask::Copy() - ENTRY, this={}, other={}, this.data_.data()={}, other.data_.data()={}",
         (void*)this, (void*)other.ptr_, (void*)data_.data(), (void*)other->data_.data());
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy CustomTask-specific fields
    data_ = other->data_;
    operation_id_ = other->operation_id_;
    HLOG(kInfo, "CustomTask::Copy() - EXIT, this={}, this.data_.data()={}",
         (void*)this, (void*)data_.data());
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
 * TestLargeOutputTask - Test large output streaming (1MB)
 * Tests streaming mechanism for large output data
 */
struct TestLargeOutputTask : public chi::Task {
  // Task-specific data
  OUT std::vector<uint8_t> data_;  // Output: 1MB of test data

  /** SHM default constructor */
  TestLargeOutputTask()
      : chi::Task() {}

  /** Emplace constructor */
  explicit TestLargeOutputTask(
      const chi::TaskId &task_node,
      const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, 24) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kTestLargeOutput;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template<typename Archive>
  void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    // No input parameters for this task
  }

  template<typename Archive>
  void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar(data_);  // Return the 1MB output data
  }

  /**
   * Copy from another TestLargeOutputTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<TestLargeOutputTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy TestLargeOutputTask-specific fields
    this->data_ = other->data_;
  }

  /**
   * Aggregate replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void Aggregate(const hipc::FullPtr<TestLargeOutputTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * GpuSubmitTask - GPU-compatible task for testing Part 3
 * This task can be created and submitted from GPU kernels
 */
struct GpuSubmitTask : public chi::Task {
  IN chi::u32 gpu_id_;          // GPU ID that submitted the task
  IN chi::u32 test_value_;      // Test value to verify correct execution
  INOUT chi::u32 result_value_; // Result computed by the task

  /** SHM default constructor */
  HSHM_CROSS_FUN GpuSubmitTask()
      : chi::Task(), gpu_id_(0), test_value_(0), result_value_(0) {}

  /** Emplace constructor */
  HSHM_CROSS_FUN explicit GpuSubmitTask(
      const chi::TaskId &task_node,
      const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query,
      chi::u32 gpu_id,
      chi::u32 test_value)
      : chi::Task(task_node, pool_id, pool_query, 25),
        gpu_id_(gpu_id), test_value_(test_value), result_value_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kGpuSubmit;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template<typename Archive>
  HSHM_CROSS_FUN void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(gpu_id_, test_value_, result_value_);
  }

  template<typename Archive>
  HSHM_CROSS_FUN void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar(result_value_);  // Return the computed result
  }

  /**
   * Copy from another GpuSubmitTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  HSHM_CROSS_FUN void Copy(const hipc::FullPtr<GpuSubmitTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy GpuSubmitTask-specific fields
    gpu_id_ = other->gpu_id_;
    test_value_ = other->test_value_;
    result_value_ = other->result_value_;
  }

  /**
   * Aggregate replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  HSHM_CROSS_FUN void Aggregate(const hipc::FullPtr<GpuSubmitTask> &other) {
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