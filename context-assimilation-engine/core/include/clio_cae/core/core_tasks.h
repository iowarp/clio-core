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

#ifndef CLIO_CAE_CORE_TASKS_H_
#define CLIO_CAE_CORE_TASKS_H_

#include <clio_runtime/admin/admin_tasks.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/autogen/core_methods.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cte/core/core_tasks.h>

#include "clio_ctp/data_structures/serialization/global_serialize.h"
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace clio::cae::core {

using MonitorTask = clio::run::admin::MonitorTask;

/**
 * CreateParams for core chimod
 * Contains configuration parameters for core container creation
 */
struct CreateParams {
  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "clio_cae_core";

  // Optional: pool ID of the next module in the pipeline (e.g., CTE core at
  // 513.0) when CAE is configured as a transparent interceptor in front of
  // CTE. When null, the CAE forwarding handlers fall back to the global CTE
  // pool ID (kCtePoolId). Mirrors compressor's CompressorConfig::next_pool_id_.
  //
  // Transparent LLM summarization used to be configured here too
  // (label_endpoint / label_prompts / label_matches). It now lives in its own
  // interposer chimod: see context-assimilation-engine/summarizer. Compose it
  // between this pool and CTE and move those keys onto its entry.
  clio::run::PoolId next_pool_id_;

  // Default constructor
  CreateParams() : next_pool_id_(clio::run::PoolId::GetNull()) {}

  // Copy constructor (for BaseCreateTask)
  CreateParams(const CreateParams &other)
      : next_pool_id_(other.next_pool_id_) {}

  // Compose pool-id ctor (matches compressor pattern)
  CreateParams(const clio::run::PoolId &pool_id, const CreateParams &other)
      : next_pool_id_(other.next_pool_id_) {
    (void)pool_id;
  }

  // Serialization support
  template <class Archive>
  void serialize(Archive &ar) {
    ar(next_pool_id_);
  }

  /**
   * Load configuration from compose YAML. Parses:
   *   - `next_pool_id`        ("major.minor")
   * @param pool_config The compose entry for this pool.
   */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (pool_config.config_.empty()) return;
    try {
      YAML::Node node = YAML::Load(pool_config.config_);
      if (node["next_pool_id"]) {
        std::string next_str = node["next_pool_id"].as<std::string>();
        auto dot = next_str.find('.');
        if (dot != std::string::npos) {
          clio::run::u32 major = std::stoul(next_str.substr(0, dot));
          clio::run::u32 minor = std::stoul(next_str.substr(dot + 1));
          next_pool_id_ = clio::run::PoolId(major, minor);
        }
      }
    } catch (...) {
      // best-effort
    }
  }
};

/**
 * CreateTask - Initialize the core container
 * Type alias for GetOrCreatePoolTask with CreateParams
 */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * DestroyTask - Destroy the core container
 */
using DestroyTask = clio::run::Task;  // Simple task for destruction

/**
 * ParseOmniTask - Parse OMNI YAML file and schedule assimilation tasks
 */
struct ParseOmniTask : public clio::run::Task {
  // Task-specific data using CTP macros
  IN clio::run::priv::string
      serialized_ctx_;  // Input: Serialized AssimilationCtx (internal use)
  OUT clio::run::u32
      num_tasks_scheduled_;   // Output: Number of assimilation tasks scheduled
  OUT clio::run::u32 result_code_;  // Output: Result code (0 = success)
  OUT clio::run::priv::string error_message_;  // Output: Error message if failed

  // SHM constructor
  ParseOmniTask()
      : clio::run::Task(),
        serialized_ctx_(CTP_MALLOC),
        num_tasks_scheduled_(0),
        result_code_(0),
        error_message_(CTP_MALLOC) {}

  // Emplace constructor - accepts vector of AssimilationCtx and serializes
  // internally
  CTP_CROSS_FUN explicit ParseOmniTask(
      const clio::run::TaskId &task_node, const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query,
      const std::vector<clio::cae::core::AssimilationCtx> &contexts)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kParseOmni),
        serialized_ctx_(CTP_MALLOC),
        num_tasks_scheduled_(0),
        result_code_(0),
        error_message_(CTP_MALLOC) {
    task_id_ = task_node;
    method_ = Method::kParseOmni;
    task_flags_.Clear();
    pool_query_ = pool_query;

    // Serialize the vector of contexts using GlobalSerialize
    std::vector<char> buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> ar(buf);
      ar(contexts);
      ar.Finalize();
    }
    serialized_ctx_ = clio::run::priv::string(CTP_MALLOC, std::string(buf.begin(), buf.end()));
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(serialized_ctx_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(num_tasks_scheduled_, result_code_, error_message_);
  }

  // Copy method for distributed execution (optional)
  void Copy(const ctp::ipc::FullPtr<ParseOmniTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    serialized_ctx_ = other->serialized_ctx_;
    num_tasks_scheduled_ = other->num_tasks_scheduled_;
    result_code_ = other->result_code_;
    error_message_ = other->error_message_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    // OUT fields ONLY -- never Copy() (issue #915): a whole-task assignment
    // destroys this ORIGIN's identity and re-assigns IN shm members across
    // allocator segments. See Task::AggregateOut for the full contract.
    auto replica = other_base.template Cast<ParseOmniTask>();
    // Each replica schedules its own share of the work, so the count is the
    // SUM. result_code_ keeps the FIRST failure so a later success cannot mask
    // it.
    num_tasks_scheduled_ += replica->num_tasks_scheduled_;
    if (result_code_ == 0) {
      result_code_ = replica->result_code_;
    }
    // error_message_ is diagnostic: keep the FIRST replica that reported one,
    // so the failure that set the collective return code is the one the caller
    // sees (last-replica-wins would hide it behind a later success).
    if (error_message_.size() == 0 && replica->error_message_.size() > 0) {
      error_message_ = replica->error_message_;
    }
  }
};

/**
 * ProcessHdf5DatasetTask - Process a single HDF5 dataset
 * Used for distributed processing where each dataset can be routed to different
 * nodes
 */
struct ProcessHdf5DatasetTask : public clio::run::Task {
  // Task-specific data
  IN clio::run::priv::string file_path_;       // HDF5 file path
  IN clio::run::priv::string dataset_path_;    // Dataset path within HDF5 file
  IN clio::run::priv::string tag_prefix_;      // Tag prefix for CTE storage
  OUT clio::run::u32 result_code_;             // Result code (0 = success)
  OUT clio::run::priv::string error_message_;  // Error message if failed

  // SHM constructor
  ProcessHdf5DatasetTask()
      : clio::run::Task(),
        file_path_(CTP_MALLOC),
        dataset_path_(CTP_MALLOC),
        tag_prefix_(CTP_MALLOC),
        result_code_(0),
        error_message_(CTP_MALLOC) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit ProcessHdf5DatasetTask(const clio::run::TaskId &task_node,
                                  const clio::run::PoolId &pool_id,
                                  const clio::run::PoolQuery &pool_query,
                                  const std::string &file_path,
                                  const std::string &dataset_path,
                                  const std::string &tag_prefix)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kProcessHdf5Dataset),
        file_path_(CTP_MALLOC, file_path),
        dataset_path_(CTP_MALLOC, dataset_path),
        tag_prefix_(CTP_MALLOC, tag_prefix),
        result_code_(0),
        error_message_(CTP_MALLOC) {
    task_id_ = task_node;
    method_ = Method::kProcessHdf5Dataset;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(file_path_, dataset_path_, tag_prefix_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(result_code_, error_message_);
  }

  // Copy method for distributed execution
  void Copy(const ctp::ipc::FullPtr<ProcessHdf5DatasetTask> &other) {
    Task::Copy(other.template Cast<Task>());
    file_path_ = other->file_path_;
    dataset_path_ = other->dataset_path_;
    tag_prefix_ = other->tag_prefix_;
    result_code_ = other->result_code_;
    error_message_ = other->error_message_;
  }

  /**
   * AggregateOut replica results into this task
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<ProcessHdf5DatasetTask>();
    // Keep the first error if any
    if (result_code_ == 0 && other->result_code_ != 0) {
      result_code_ = other->result_code_;
      error_message_ = other->error_message_;
    }
  }
};

/**
 * ExportDataTask - Export blobs from CTE to a file
 * Iterates over all blobs in a tag and writes them to the output path.
 */
struct ExportDataTask : public clio::run::Task {
  IN clio::run::priv::string tag_name_;      // Tag to export from CTE
  IN clio::run::priv::string output_path_;   // Destination file path
  IN clio::run::priv::string format_;        // Export format: "hdf5" or "binary"
  OUT clio::run::u32 result_code_;           // 0 = success
  OUT clio::run::priv::string error_message_;
  OUT clio::run::u64 bytes_exported_;

  // SHM constructor
  ExportDataTask()
      : clio::run::Task(),
        tag_name_(CTP_MALLOC),
        output_path_(CTP_MALLOC),
        format_(CTP_MALLOC),
        result_code_(0),
        error_message_(CTP_MALLOC),
        bytes_exported_(0) {}

  // Emplace constructor
  explicit ExportDataTask(const clio::run::TaskId &task_node,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query,
                          const std::string &tag_name,
                          const std::string &output_path,
                          const std::string &format)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kExportData),
        tag_name_(CTP_MALLOC, tag_name),
        output_path_(CTP_MALLOC, output_path),
        format_(CTP_MALLOC, format),
        result_code_(0),
        error_message_(CTP_MALLOC),
        bytes_exported_(0) {
    task_id_ = task_node;
    method_ = Method::kExportData;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_name_, output_path_, format_);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(result_code_, error_message_, bytes_exported_);
  }

  void Copy(const ctp::ipc::FullPtr<ExportDataTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_name_ = other->tag_name_;
    output_path_ = other->output_path_;
    format_ = other->format_;
    result_code_ = other->result_code_;
    error_message_ = other->error_message_;
    bytes_exported_ = other->bytes_exported_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    // OUT fields ONLY -- never Copy() (issue #915): a whole-task assignment
    // destroys this ORIGIN's identity and re-assigns IN shm members across
    // allocator segments. See Task::AggregateOut for the full contract.
    auto replica = other_base.template Cast<ExportDataTask>();
    // Each replica exports its own share, so the byte count is the SUM;
    // result_code_ keeps the FIRST failure.
    bytes_exported_ += replica->bytes_exported_;
    if (result_code_ == 0) {
      result_code_ = replica->result_code_;
    }
    // error_message_ is diagnostic: keep the FIRST replica that reported one,
    // so the failure that set the collective return code is the one the caller
    // sees (last-replica-wins would hide it behind a later success).
    if (error_message_.size() == 0 && replica->error_message_.size() > 0) {
      error_message_ = replica->error_message_;
    }
  }
};

/**
 * ImportDataTask - Import an HDF5 dataset from a file into a CTE tag.
 * The inverse of ExportDataTask: reads the dataset at the in-file path equal to
 * `tag_name_` and writes it into that tag in the kvhdf5 store form (a __meta
 * blob + one raw chunk blob per coordinate).
 */
struct ImportDataTask : public clio::run::Task {
  IN clio::run::priv::string tag_name_;      // Tag to import into (== in-file path)
  IN clio::run::priv::string input_path_;    // Source file path
  IN clio::run::priv::string format_;        // Import format: "hdf5"
  OUT clio::run::u32 result_code_;           // 0 = success
  OUT clio::run::priv::string error_message_;
  OUT clio::run::u64 bytes_imported_;

  // SHM constructor
  ImportDataTask()
      : clio::run::Task(),
        tag_name_(CTP_MALLOC),
        input_path_(CTP_MALLOC),
        format_(CTP_MALLOC),
        result_code_(0),
        error_message_(CTP_MALLOC),
        bytes_imported_(0) {}

  // Emplace constructor
  explicit ImportDataTask(const clio::run::TaskId &task_node,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query,
                          const std::string &tag_name,
                          const std::string &input_path,
                          const std::string &format)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kImportData),
        tag_name_(CTP_MALLOC, tag_name),
        input_path_(CTP_MALLOC, input_path),
        format_(CTP_MALLOC, format),
        result_code_(0),
        error_message_(CTP_MALLOC),
        bytes_imported_(0) {
    task_id_ = task_node;
    method_ = Method::kImportData;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_name_, input_path_, format_);
  }

  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(result_code_, error_message_, bytes_imported_);
  }

  void Copy(const ctp::ipc::FullPtr<ImportDataTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_name_ = other->tag_name_;
    input_path_ = other->input_path_;
    format_ = other->format_;
    result_code_ = other->result_code_;
    error_message_ = other->error_message_;
    bytes_imported_ = other->bytes_imported_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    // OUT fields ONLY -- never Copy() (issue #915): a whole-task assignment
    // destroys this ORIGIN's identity and re-assigns IN shm members across
    // allocator segments. See Task::AggregateOut for the full contract.
    auto replica = other_base.template Cast<ImportDataTask>();
    // Each replica imports its own share, so the byte count is the SUM;
    // result_code_ keeps the FIRST failure.
    bytes_imported_ += replica->bytes_imported_;
    if (result_code_ == 0) {
      result_code_ = replica->result_code_;
    }
    // error_message_ is diagnostic: keep the FIRST replica that reported one,
    // so the failure that set the collective return code is the one the caller
    // sees (last-replica-wins would hide it behind a later success).
    if (error_message_.size() == 0 && replica->error_message_.size() > 0) {
      error_message_ = replica->error_message_;
    }
  }
};

// ---------------------------------------------------------------------------
// CTE interceptor task typedefs. CAE forwards these tasks transparently to
// the configured `next_pool_id` CTE core. The struct layout AND the
// dispatching method id are inherited from clio::cte::core — see
// autogen/core_methods.h for the matching kPutBlob / kGetBlob /
// kGetOrCreateTag constants. The static_asserts below guard the
// invariant that lets a CTE-built task dispatch to a CAE handler.
// ---------------------------------------------------------------------------
using PutBlobTask = clio::cte::core::PutBlobTask;
using GetBlobTask = clio::cte::core::GetBlobTask;
using GetOrCreateTagTask =
    clio::cte::core::GetOrCreateTagTask<clio::cte::core::CreateParams>;
using SemanticSearchTask = clio::cte::core::SemanticSearchTask;

static_assert(Method::kPutBlob == clio::cte::core::Method::kPutBlob,
              "CAE kPutBlob must match clio::cte::core::Method::kPutBlob "
              "for transparent CTE→CAE task dispatch");
static_assert(Method::kGetBlob == clio::cte::core::Method::kGetBlob,
              "CAE kGetBlob must match clio::cte::core::Method::kGetBlob "
              "for transparent CTE→CAE task dispatch");
static_assert(
    Method::kGetOrCreateTag == clio::cte::core::Method::kGetOrCreateTag,
    "CAE kGetOrCreateTag must match clio::cte::core::Method::kGetOrCreateTag "
    "for transparent CTE→CAE task dispatch");
static_assert(
    Method::kSemanticSearch == clio::cte::core::Method::kSemanticSearch,
    "CAE kSemanticSearch must match clio::cte::core::Method::kSemanticSearch "
    "for transparent CTE→CAE task dispatch");

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_TASKS_H_
