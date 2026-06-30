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

#ifndef CLIO_CAE_CORE_CLIENT_H_
#define CLIO_CAE_CORE_CLIENT_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/core_tasks.h>

namespace clio::cae::core {

class Client : public clio::run::ContainerClient {
 public:
  Client() = default;
  explicit Client(const clio::run::PoolId& pool_id) { Init(pool_id); }

  /**
   * Asynchronous Create - returns immediately
   * After Wait(), caller should:
   *   1. Update client pool_id_: client.Init(task->new_pool_id_)
   * Note: Task is automatically freed when Future goes out of scope
   */
  clio::run::Future<CreateTask> AsyncCreate(
      const clio::run::PoolQuery& pool_query,
      const std::string& pool_name,
      const clio::run::PoolId& custom_pool_id,
      const CreateParams& params = CreateParams()) {
    auto* ipc_manager = CLIO_IPC;

    // CRITICAL: CreateTask MUST use admin pool for GetOrCreatePool processing
    // Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(
        clio::run::CreateTaskId(),
        clio::run::kAdminPoolId,  // Always use admin pool for CreateTask
        pool_query,
        CreateParams::chimod_lib_name,  // ChiMod name from CreateParams
        pool_name,                       // Pool name
        custom_pool_id,                  // Target pool ID
        this,                            // Client pointer for PostWait
        params);                         // CreateParams with configuration

    // Submit to runtime
    return ipc_manager->Send(task);
  }

  /**
   * Monitor container state - asynchronous
   */
  clio::run::Future<MonitorTask> AsyncMonitor(const clio::run::PoolQuery &pool_query,
                                        const std::string &query) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<MonitorTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, query);
    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous ParseOmni - Parse OMNI YAML file and schedule assimilation tasks
   * Accepts vector of AssimilationCtx and serializes it transparently in the task constructor
   * After Wait(), access results via task->num_tasks_scheduled_ and task->result_code_
   */
  clio::run::Future<ParseOmniTask> AsyncParseOmni(
      const std::vector<AssimilationCtx>& contexts) {
    auto* ipc_manager = CLIO_IPC;

    auto task = ipc_manager->NewTask<ParseOmniTask>(
        clio::run::CreateTaskId(),
        pool_id_,
        clio::run::PoolQuery::Local(),
        contexts);

    return ipc_manager->Send(task);
  }

  /**
   * Asynchronous ProcessHdf5Dataset - Process a single HDF5 dataset
   * Can be routed to a specific node for distributed processing
   * @param pool_query Pool query for routing (use PoolQuery::Physical(node_id) for specific node)
   * @param file_path Path to the HDF5 file
   * @param dataset_path Path to the dataset within the HDF5 file
   * @param tag_prefix Tag prefix for CTE storage
   */
  /**
   * Asynchronous ExportData - export all blobs in a CTE tag to a file
   * @param tag_name  Name of the CTE tag to export
   * @param output_path  Destination file path
   * @param format  Export format: "hdf5" or "binary"
   * @param pool_query  Pool query for routing (default: Local)
   */
  clio::run::Future<ExportDataTask> AsyncExportData(
      const std::string &tag_name,
      const std::string &output_path,
      const std::string &format,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local()) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<ExportDataTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query,
        tag_name, output_path, format);
    return ipc_manager->Send(task);
  }

  clio::run::Future<ProcessHdf5DatasetTask> AsyncProcessHdf5Dataset(
      const clio::run::PoolQuery& pool_query,
      const std::string& file_path,
      const std::string& dataset_path,
      const std::string& tag_prefix) {
    auto* ipc_manager = CLIO_IPC;

    HLOG(kInfo, "AsyncProcessHdf5Dataset: Creating task for pool_id={}, file={}, dataset={}",
         pool_id_, file_path, dataset_path);

    auto task = ipc_manager->NewTask<ProcessHdf5DatasetTask>(
        clio::run::CreateTaskId(),
        pool_id_,
        pool_query,
        file_path,
        dataset_path,
        tag_prefix);

    if (task.IsNull()) {
      HLOG(kError, "AsyncProcessHdf5Dataset: NewTask returned null!");
    } else {
      HLOG(kInfo, "AsyncProcessHdf5Dataset: Task created, method={}, calling Send",
           task->method_);
    }

    return ipc_manager->Send(task);
  }

  // --------------------------------------------------------------------
  // CTE interceptor passthroughs.
  //
  // These mirror clio::cte::core::Client::AsyncPutBlob / AsyncGetBlob /
  // AsyncGetOrCreateTag and target the CAE pool. The CAE runtime forwards
  // each task to its configured next_pool_id (the real CTE core), enabling
  // a CTE client pointed at the CAE pool to transparently flow data
  // through CAE first. Analogous to clio::cte::compressor::Client.
  //
  // The typedef'd task structs (PutBlobTask etc.) inherit their method id
  // from clio::cte::core::Method, so we re-stamp method_ after NewTask so
  // dispatch at CAE lands in the right Run case.
  // --------------------------------------------------------------------
  clio::run::Future<PutBlobTask> AsyncPutBlob(
      const clio::cte::core::TagId &tag_id, const char *blob_name,
      clio::run::u64 offset, clio::run::u64 size, ctp::ipc::ShmPtr<> blob_data,
      float score = -1.0f,
      const clio::cte::core::Context &context = clio::cte::core::Context(),
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<PutBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
        offset, size, blob_data, score, context, flags);
    task->method_ = Method::kPutBlob;
    return ipc_manager->Send(task);
  }

  clio::run::Future<PutBlobTask> AsyncPutBlob(
      const clio::cte::core::TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size, ctp::ipc::ShmPtr<> blob_data,
      float score = -1.0f,
      const clio::cte::core::Context &context = clio::cte::core::Context(),
      clio::run::u32 flags = 0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncPutBlob(tag_id, blob_name.c_str(), offset, size, blob_data,
                        score, context, flags, pool_query);
  }

  clio::run::Future<GetBlobTask> AsyncGetBlob(
      const clio::cte::core::TagId &tag_id, const char *blob_name,
      clio::run::u64 offset, clio::run::u64 size, clio::run::u32 flags,
      ctp::ipc::ShmPtr<> blob_data,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<GetBlobTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, blob_name,
        offset, size, flags, blob_data);
    task->method_ = Method::kGetBlob;
    return ipc_manager->Send(task);
  }

  clio::run::Future<GetBlobTask> AsyncGetBlob(
      const clio::cte::core::TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size, clio::run::u32 flags,
      ctp::ipc::ShmPtr<> blob_data,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    return AsyncGetBlob(tag_id, blob_name.c_str(), offset, size, flags,
                        blob_data, pool_query);
  }

  clio::run::Future<GetOrCreateTagTask> AsyncGetOrCreateTag(
      const std::string &tag_name,
      const clio::cte::core::TagId &tag_id = clio::cte::core::TagId::GetNull(),
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic()) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<GetOrCreateTagTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_name, tag_id);
    task->method_ = Method::kGetOrCreateTag;
    return ipc_manager->Send(task);
  }

  /**
   * Forward SemanticSearch (BM25 keyword search) to the CTE core
   * behind the CAE interceptor. Same signature/semantics as
   * clio::cte::core::Client::AsyncSemanticSearch — see that for the
   * matching/scoring rules.
   */
  clio::run::Future<SemanticSearchTask> AsyncSemanticSearch(
      const std::string &tag_regex, const std::string &blob_regex,
      const std::string &query_text, clio::run::u32 k = 10,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local()) {
    auto *ipc_manager = CLIO_IPC;
    auto task = ipc_manager->NewTask<SemanticSearchTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_regex, blob_regex,
        query_text, k);
    task->method_ = Method::kSemanticSearch;
    return ipc_manager->Send(task);
  }
};

}  // namespace clio::cae::core

// Global pointer-based singleton for CAE client with lazy initialization
CTP_DEFINE_GLOBAL_PTR_VAR_H(clio::cae::core::Client, g_cae_client);

/**
 * Initialize CAE client singleton
 * Calls CLIO_CTE_CLIENT_INIT internally to ensure CTE is initialized
 * Creates and initializes a global CAE client singleton
 *
 * @param config_path Path to configuration file (optional)
 * @param pool_query Pool query for CAE pool creation (default: Dynamic)
 * @return true on success, false on failure
 */
bool CLIO_CAE_CLIENT_INIT(const std::string &config_path = "",
                         const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Dynamic());

/**
 * Global CAE client singleton accessor macro
 * Returns pointer to the global CAE client instance
 */
#define CLIO_CAE_CLIENT                                                         \
  (&(*CTP_GET_GLOBAL_PTR_VAR(clio::cae::core::Client,                         \
                              g_cae_client)))

// Intermediate `clio_cae` namespace-spelling alias. In-tree code uses
// `clio::cae::core::*`; downstream that migrated to the `clio_cae::*` waypoint
// keeps compiling via this alias. Safe to use the simple `namespace X = Y;`
// form because no external chimod opens `namespace clio_cae::xxx {}`.
// (The wrp_cae/ forwarder shim tree and the wrp_cae::/WRP_CAE_* compat aliases
// were removed; downstream must use the clio_cae / clio::cae names.)
namespace clio_cae = clio::cae;

#endif  // CLIO_CAE_CORE_CLIENT_H_
