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

#ifndef WRP_CAE_CORE_TASKS_H_
#define WRP_CAE_CORE_TASKS_H_

#include <chimaera/admin/admin_tasks.h>
#include <chimaera/chimaera.h>
#include <wrp_cae/core/autogen/core_methods.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>
#include <vector>

namespace wrp_cae::core {

/**
 * CreateParams for core chimod
 * Contains configuration parameters for core container creation
 */
struct CreateParams {
  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "wrp_cae_core";

  // Default constructor
  CreateParams() {}

  // Copy constructor (for BaseCreateTask)
  CreateParams(const CreateParams &other) {}

  // Serialization support for cereal
  template <class Archive>
  void serialize(Archive &ar) {
    // No members to serialize
  }
};

/**
 * CreateTask - Initialize the core container
 * Type alias for GetOrCreatePoolTask with CreateParams
 */
using CreateTask = chimaera::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * DestroyTask - Destroy the core container
 */
using DestroyTask = chi::Task;  // Simple task for destruction

/**
 * ParseOmniTask - Parse OMNI YAML file and schedule assimilation tasks
 */
struct ParseOmniTask : public chi::Task {
  // Task-specific data using HSHM macros
  IN chi::priv::string
      serialized_ctx_;  // Input: Serialized AssimilationCtx (internal use)
  OUT chi::u32
      num_tasks_scheduled_;   // Output: Number of assimilation tasks scheduled
  OUT chi::u32 result_code_;  // Output: Result code (0 = success)
  OUT chi::priv::string error_message_;  // Output: Error message if failed

  // SHM constructor
  ParseOmniTask()
      : chi::Task(),
        serialized_ctx_(HSHM_MALLOC),
        num_tasks_scheduled_(0),
        result_code_(0),
        error_message_(HSHM_MALLOC) {}

  // Emplace constructor - accepts vector of AssimilationCtx and serializes
  // internally
  explicit ParseOmniTask(
      const chi::TaskId &task_node, const chi::PoolId &pool_id,
      const chi::PoolQuery &pool_query,
      const std::vector<wrp_cae::core::AssimilationCtx> &contexts)
      : chi::Task(task_node, pool_id, pool_query, Method::kParseOmni),
        serialized_ctx_(HSHM_MALLOC),
        num_tasks_scheduled_(0),
        result_code_(0),
        error_message_(HSHM_MALLOC) {
    task_id_ = task_node;
    method_ = Method::kParseOmni;
    task_flags_.Clear();
    pool_query_ = pool_query;

    // Serialize the vector of contexts transparently using cereal
    std::stringstream ss;
    {
      cereal::BinaryOutputArchive ar(ss);
      ar(contexts);
    }
    serialized_ctx_ = chi::priv::string(HSHM_MALLOC, ss.str());
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(serialized_ctx_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(num_tasks_scheduled_, result_code_, error_message_);
  }

  // Copy method for distributed execution (optional)
  void Copy(const hipc::FullPtr<ParseOmniTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    serialized_ctx_ = other->serialized_ctx_;
    num_tasks_scheduled_ = other->num_tasks_scheduled_;
    result_code_ = other->result_code_;
    error_message_ = other->error_message_;
  }

  /**
   * Aggregate replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void Aggregate(const hipc::FullPtr<ParseOmniTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * ProcessHdf5DatasetTask - Process a single HDF5 dataset
 * Used for distributed processing where each dataset can be routed to different
 * nodes
 */
struct ProcessHdf5DatasetTask : public chi::Task {
  // Task-specific data
  IN chi::priv::string file_path_;       // HDF5 file path
  IN chi::priv::string dataset_path_;    // Dataset path within HDF5 file
  IN chi::priv::string tag_prefix_;      // Tag prefix for CTE storage
  OUT chi::u32 result_code_;             // Result code (0 = success)
  OUT chi::priv::string error_message_;  // Error message if failed

  // SHM constructor
  ProcessHdf5DatasetTask()
      : chi::Task(),
        file_path_(HSHM_MALLOC),
        dataset_path_(HSHM_MALLOC),
        tag_prefix_(HSHM_MALLOC),
        result_code_(0),
        error_message_(HSHM_MALLOC) {}

  // Emplace constructor
  explicit ProcessHdf5DatasetTask(const chi::TaskId &task_node,
                                  const chi::PoolId &pool_id,
                                  const chi::PoolQuery &pool_query,
                                  const std::string &file_path,
                                  const std::string &dataset_path,
                                  const std::string &tag_prefix)
      : chi::Task(task_node, pool_id, pool_query, Method::kProcessHdf5Dataset),
        file_path_(HSHM_MALLOC, file_path),
        dataset_path_(HSHM_MALLOC, dataset_path),
        tag_prefix_(HSHM_MALLOC, tag_prefix),
        result_code_(0),
        error_message_(HSHM_MALLOC) {
    task_id_ = task_node;
    method_ = Method::kProcessHdf5Dataset;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(file_path_, dataset_path_, tag_prefix_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(result_code_, error_message_);
  }

  // Copy method for distributed execution
  void Copy(const hipc::FullPtr<ProcessHdf5DatasetTask> &other) {
    Task::Copy(other.template Cast<Task>());
    file_path_ = other->file_path_;
    dataset_path_ = other->dataset_path_;
    tag_prefix_ = other->tag_prefix_;
    result_code_ = other->result_code_;
    error_message_ = other->error_message_;
  }

  /**
   * Aggregate replica results into this task
   */
  void Aggregate(const hipc::FullPtr<ProcessHdf5DatasetTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    // Keep the first error if any
    if (result_code_ == 0 && other->result_code_ != 0) {
      result_code_ = other->result_code_;
      error_message_ = other->error_message_;
    }
  }
};

}  // namespace wrp_cae::core

#endif  // WRP_CAE_CORE_TASKS_H_
