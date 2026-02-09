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

#ifndef SIMPLE_MOD_TASKS_H_
#define SIMPLE_MOD_TASKS_H_

#include <chimaera/chimaera.h>
#include <chimaera/admin/admin_tasks.h>

#include "autogen/simple_mod_methods.h"

/**
 * Task struct definitions for Simple Mod ChiMod
 *
 * Minimal ChiMod for testing external development patterns.
 * Demonstrates basic task structure for external ChiMod development.
 */

namespace external_test::simple_mod {

/**
 * CreateParams for simple_mod chimod
 * Contains configuration parameters for simple_mod container creation
 */
struct CreateParams {
  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "external_test_simple_mod";

  // Default constructor
  CreateParams() = default;


  // Serialization support for cereal
  template <class Archive>
  void serialize(Archive &ar) {
    // No additional fields to serialize for simple_mod
    (void)ar;
  }
};

/**
 * CreateTask - Simple mod container creation task
 * Uses the standard BaseCreateTask template from admin module
 */
using CreateTask = chimaera::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * Standard DestroyTask for simple_mod
 * Uses the reusable DestroyTask from admin module
 */
using DestroyTask = chimaera::admin::DestroyTask;

/**
 * FlushTask - Simple flush task for simple_mod
 * Minimal task with no additional inputs beyond basic task parameters
 */
struct FlushTask : public chi::Task {
  // Output results
  OUT chi::u64 total_work_done_;  ///< Total amount of work completed

  /** SHM default constructor */
  explicit FlushTask(AllocT* alloc)
      : chi::Task(alloc), total_work_done_(0) {}

  /** Emplace constructor */
  explicit FlushTask(AllocT* alloc,
                     const chi::TaskId &task_node, const chi::PoolId &pool_id,
                     const chi::PoolQuery &pool_query)
      : chi::Task(alloc, task_node, pool_id, pool_query, 10),
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
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    // No parameters to serialize for flush
    (void)ar;
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
   * Copy from another FlushTask
   */
  void Copy(const hipc::FullPtr<FlushTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    total_work_done_ = other->total_work_done_;
  }

  /**
   * Aggregate replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void Aggregate(const hipc::FullPtr<FlushTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

}  // namespace external_test::simple_mod

#endif  // SIMPLE_MOD_TASKS_H_