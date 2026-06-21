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

#include <clio_runtime/clio_runtime.h>
#include "autogen/MOD_NAME_methods.h"
// Include admin tasks for BaseCreateTask
#include <clio_runtime/admin/admin_tasks.h>

/**
 * Task struct definitions for MOD_NAME
 * 
 * Defines the tasks for Create and Custom methods.
 */

namespace clio::run::MOD_NAME {

using MonitorTask = clio::run::admin::MonitorTask;

/**
 * CreateParams for MOD_NAME chimod
 * Contains configuration parameters for MOD_NAME container creation
 */
struct CreateParams {
  // MOD_NAME-specific parameters (primitives only for cereal compatibility)
  clio::run::u32 worker_count_;
  clio::run::u32 config_flags_;

  // Required: chimod library name for module manager
  static constexpr const char* chimod_lib_name = "clio_MOD_NAME";

  // Constructor with parameters (also serves as default)
  CreateParams(clio::run::u32 worker_count = 1, clio::run::u32 config_flags = 0)
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
using CreateTask = clio::run::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * CustomTask - Example custom operation
 */
struct CustomTask : public clio::run::Task {
  // Task-specific data
  INOUT clio::run::priv::string data_;
  IN clio::run::u32 operation_id_;

  /** SHM default constructor */
  CustomTask()
      : clio::run::Task(),
        data_(CLIO_PRIV_ALLOC), operation_id_(0) {
  }

  /** Emplace constructor */
  explicit CustomTask(
      const clio::run::TaskId &task_node,
      const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query,
      const std::string &data,
      clio::run::u32 operation_id)
      : clio::run::Task(task_node, pool_id, pool_query, 10),
        data_(CLIO_PRIV_ALLOC, data), operation_id_(operation_id) {
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
  CTP_CROSS_FUN void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(data_, operation_id_);
  }

  template<typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar(data_);
  }

  /**
   * Copy from another CustomTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const ctp::ipc::FullPtr<CustomTask> &other) {
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
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<CustomTask>());
  }
};

/**
 * ManyToOneSumTask - collective sum, used to exercise AggregateIn for the
 * PoolQuery::ManyToOne path.
 *
 * Each submitter contributes value_. On the neighborhood leader the batch is
 * folded into one aggregate task via AggregateIn (summing value_), which runs
 * once and echoes the combined total into sum_; that single result is then
 * broadcast (SerializeOut copy-back) to every submitter. After completion all
 * submitters' sum_ equals the sum of the whole batch's value_.
 */
struct ManyToOneSumTask : public clio::run::Task {
  IN clio::run::u64 value_;  // this submitter's contribution
  OUT clio::run::u64 sum_;   // collective total (broadcast to all members)

  /** SHM default constructor */
  ManyToOneSumTask() : clio::run::Task(), value_(0), sum_(0) {}

  /** Emplace constructor */
  explicit ManyToOneSumTask(const clio::run::TaskId &task_node,
                            const clio::run::PoolId &pool_id,
                            const clio::run::PoolQuery &pool_query, clio::run::u64 value)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kManyToOneSum),
        value_(value), sum_(0) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kManyToOneSum;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  ~ManyToOneSumTask() {}

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(value_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(sum_);
  }

  void Copy(const ctp::ipc::FullPtr<ManyToOneSumTask> &other) {
    Task::Copy(other.template Cast<Task>());
    value_ = other->value_;
    sum_ = other->sum_;
  }

  /** AggregateOut: gather replica OUT — sum partial totals (N->1). */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    sum_ += other_base.template Cast<ManyToOneSumTask>()->sum_;
  }

  /** AggregateIn: fold a batched member's input contribution into this. */
  void AggregateIn(const ctp::ipc::FullPtr<clio::run::Task> &member_base) {
    value_ += member_base.template Cast<ManyToOneSumTask>()->value_;
  }
};

/**
 * CoMutexTestTask - Test CoMutex functionality
 */
struct CoMutexTestTask : public clio::run::Task {
  IN clio::run::u32 test_id_;         // Test identifier
  IN clio::run::u32 hold_duration_ms_; // How long to hold the mutex

  /** SHM default constructor */
  CoMutexTestTask()
      : clio::run::Task(), test_id_(0), hold_duration_ms_(0) {}

  /** Emplace constructor */
  explicit CoMutexTestTask(
      const clio::run::TaskId &task_node,
      const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query,
      clio::run::u32 test_id,
      clio::run::u32 hold_duration_ms)
      : clio::run::Task(task_node, pool_id, pool_query, 20),
        test_id_(test_id), hold_duration_ms_(hold_duration_ms) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kCoMutexTest;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template<typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(test_id_, hold_duration_ms_);
  }

  template<typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    // No output parameters for this task
  }

  /**
   * Copy from another CoMutexTestTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const ctp::ipc::FullPtr<CoMutexTestTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy CoMutexTestTask-specific fields
    test_id_ = other->test_id_;
    hold_duration_ms_ = other->hold_duration_ms_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<CoMutexTestTask>());
  }
};

/**
 * CoRwLockTestTask - Test CoRwLock functionality
 */
struct CoRwLockTestTask : public clio::run::Task {
  IN clio::run::u32 test_id_;         // Test identifier
  IN bool is_writer_;           // True for write lock, false for read lock
  IN clio::run::u32 hold_duration_ms_; // How long to hold the lock

  /** SHM default constructor */
  CoRwLockTestTask()
      : clio::run::Task(), test_id_(0), is_writer_(false), hold_duration_ms_(0) {}

  /** Emplace constructor */
  explicit CoRwLockTestTask(
      const clio::run::TaskId &task_node,
      const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query,
      clio::run::u32 test_id,
      bool is_writer,
      clio::run::u32 hold_duration_ms)
      : clio::run::Task(task_node, pool_id, pool_query, 21),
        test_id_(test_id), is_writer_(is_writer), hold_duration_ms_(hold_duration_ms) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kCoRwLockTest;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template<typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(test_id_, is_writer_, hold_duration_ms_);
  }

  template<typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    // No output parameters for this task
  }

  /**
   * Copy from another CoRwLockTestTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const ctp::ipc::FullPtr<CoRwLockTestTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy CoRwLockTestTask-specific fields
    test_id_ = other->test_id_;
    is_writer_ = other->is_writer_;
    hold_duration_ms_ = other->hold_duration_ms_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<CoRwLockTestTask>());
  }
};

/**
 * WaitTestTask - Test recursive task->Wait() functionality
 * This task calls itself recursively "depth" times to test nested Wait() calls
 */
struct WaitTestTask : public clio::run::Task {
  IN clio::run::u32 depth_;              // Number of recursive calls to make
  IN clio::run::u32 test_id_;            // Test identifier for tracking
  INOUT clio::run::u32 current_depth_;   // Current recursion level (starts at 0)

  /** SHM default constructor */
  WaitTestTask()
      : clio::run::Task(), depth_(0), test_id_(0), current_depth_(0) {}

  /** Emplace constructor */
  explicit WaitTestTask(
      const clio::run::TaskId &task_node,
      const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query,
      clio::run::u32 depth,
      clio::run::u32 test_id)
      : clio::run::Task(task_node, pool_id, pool_query, 23),
        depth_(depth), test_id_(test_id), current_depth_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kWaitTest;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template<typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(depth_, test_id_, current_depth_);
  }

  template<typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar(current_depth_);  // Return the final depth reached
  }

  /**
   * Copy from another WaitTestTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const ctp::ipc::FullPtr<WaitTestTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy WaitTestTask-specific fields
    depth_ = other->depth_;
    test_id_ = other->test_id_;
    current_depth_ = other->current_depth_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<WaitTestTask>());
  }
};

/**
 * TestLargeOutputTask - Test large output streaming (1MB)
 * Tests streaming mechanism for large output data
 */
struct TestLargeOutputTask : public clio::run::Task {
  // Task-specific data
  OUT std::vector<uint8_t> data_;  // Output: 1MB of test data

  /** SHM default constructor */
  TestLargeOutputTask()
      : clio::run::Task() {}

  /** Emplace constructor */
  explicit TestLargeOutputTask(
      const clio::run::TaskId &task_node,
      const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_node, pool_id, pool_query, 24) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kTestLargeOutput;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template<typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
  }

  template<typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar(data_);
  }

  /**
   * Copy from another TestLargeOutputTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const ctp::ipc::FullPtr<TestLargeOutputTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy TestLargeOutputTask-specific fields
    this->data_ = other->data_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<TestLargeOutputTask>());
  }
};

/**
 * GpuSubmitTask - GPU-compatible task for testing Part 3
 * This task can be created and submitted from GPU kernels
 */
struct GpuSubmitTask : public clio::run::Task {
  IN clio::run::u32 gpu_id_;          // GPU ID that submitted the task
  IN clio::run::u32 test_value_;      // Test value to verify correct execution
  INOUT clio::run::u32 result_value_; // Result computed by the task
  OUT clio::run::u32 counter_value_;  // Atomic counter: number of lanes that executed

  /** SHM default constructor */
  CTP_CROSS_FUN GpuSubmitTask()
      : clio::run::Task(), gpu_id_(0), test_value_(0), result_value_(0),
        counter_value_(0) {}

  /** Emplace constructor */
  CTP_CROSS_FUN explicit GpuSubmitTask(
      const clio::run::TaskId &task_node,
      const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query,
      clio::run::u32 gpu_id,
      clio::run::u32 test_value)
      : clio::run::Task(task_node, pool_id, pool_query, 25),
        gpu_id_(gpu_id), test_value_(test_value), result_value_(0),
        counter_value_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kGpuSubmit;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template<typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(gpu_id_, test_value_, result_value_);
  }

  template<typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar(result_value_, counter_value_);
  }

  /**
   * Copy from another GpuSubmitTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  CTP_CROSS_FUN void Copy(const ctp::ipc::FullPtr<GpuSubmitTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy GpuSubmitTask-specific fields
    gpu_id_ = other->gpu_id_;
    test_value_ = other->test_value_;
    result_value_ = other->result_value_;
    counter_value_ = other->counter_value_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  CTP_CROSS_FUN void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<GpuSubmitTask>());
  }
};

/**
 * SubtaskTestTask - GPU coroutine subtask spawning test.
 * The GPU implementation co_awaits GpuSubmit on itself.
 */
struct SubtaskTestTask : public clio::run::Task {
  IN clio::run::u32 test_value_;
  IN clio::run::u32 num_subtasks_;  /**< Number of subtasks to spawn (for benchmarking) */
  OUT clio::run::u32 result_value_;

  CTP_CROSS_FUN SubtaskTestTask()
      : clio::run::Task(), test_value_(0), num_subtasks_(1), result_value_(0) {}

  CTP_CROSS_FUN explicit SubtaskTestTask(
      const clio::run::TaskId &task_node,
      const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query,
      clio::run::u32 test_value,
      clio::run::u32 num_subtasks = 1)
      : clio::run::Task(task_node, pool_id, pool_query, 10),
        test_value_(test_value), num_subtasks_(num_subtasks), result_value_(0) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kSubtaskTest;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(test_value_);
    ar(num_subtasks_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(result_value_);
  }

  void Copy(const ctp::ipc::FullPtr<SubtaskTestTask> &other) {
    Task::Copy(other.template Cast<Task>());
    test_value_ = other->test_value_;
    num_subtasks_ = other->num_subtasks_;
    result_value_ = other->result_value_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<SubtaskTestTask>());
  }
};

/**
 * Standard DestroyTask for MOD_NAME
 * All ChiMods should use the same DestroyTask structure from admin
 */
using DestroyTask = clio::run::admin::DestroyTask;

} // namespace clio::run::MOD_NAME

#endif // MOD_NAME_TASKS_H_