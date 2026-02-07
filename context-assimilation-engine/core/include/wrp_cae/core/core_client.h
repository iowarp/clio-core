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

#ifndef WRP_CAE_CORE_CLIENT_H_
#define WRP_CAE_CORE_CLIENT_H_

#include <chimaera/chimaera.h>
#include <wrp_cae/core/core_tasks.h>

namespace wrp_cae::core {

class Client : public chi::ContainerClient {
 public:
  Client() = default;
  explicit Client(const chi::PoolId& pool_id) { Init(pool_id); }

  /**
   * Asynchronous Create - returns immediately
   * After Wait(), caller should:
   *   1. Update client pool_id_: client.Init(task->new_pool_id_)
   * Note: Task is automatically freed when Future goes out of scope
   */
  chi::Future<CreateTask> AsyncCreate(
      const chi::PoolQuery& pool_query,
      const std::string& pool_name,
      const chi::PoolId& custom_pool_id,
      const CreateParams& params = CreateParams()) {
    auto* ipc_manager = CHI_IPC;

    // CRITICAL: CreateTask MUST use admin pool for GetOrCreatePool processing
    // Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Always use admin pool for CreateTask
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
   * Asynchronous ParseOmni - Parse OMNI YAML file and schedule assimilation tasks
   * Accepts vector of AssimilationCtx and serializes it transparently in the task constructor
   * After Wait(), access results via task->num_tasks_scheduled_ and task->result_code_
   */
  chi::Future<ParseOmniTask> AsyncParseOmni(
      const std::vector<AssimilationCtx>& contexts) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<ParseOmniTask>(
        chi::CreateTaskId(),
        pool_id_,
        chi::PoolQuery::Local(),
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
  chi::Future<ProcessHdf5DatasetTask> AsyncProcessHdf5Dataset(
      const chi::PoolQuery& pool_query,
      const std::string& file_path,
      const std::string& dataset_path,
      const std::string& tag_prefix) {
    auto* ipc_manager = CHI_IPC;

    HLOG(kInfo, "AsyncProcessHdf5Dataset: Creating task for pool_id={}, file={}, dataset={}",
         pool_id_, file_path, dataset_path);

    auto task = ipc_manager->NewTask<ProcessHdf5DatasetTask>(
        chi::CreateTaskId(),
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

};

}  // namespace wrp_cae::core

// Global pointer-based singleton for CAE client with lazy initialization
HSHM_DEFINE_GLOBAL_PTR_VAR_H(wrp_cae::core::Client, g_cae_client);

/**
 * Initialize CAE client singleton
 * Calls WRP_CTE_CLIENT_INIT internally to ensure CTE is initialized
 * Creates and initializes a global CAE client singleton
 *
 * @param config_path Path to configuration file (optional)
 * @param pool_query Pool query for CAE pool creation (default: Dynamic)
 * @return true on success, false on failure
 */
bool WRP_CAE_CLIENT_INIT(const std::string &config_path = "",
                         const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic());

/**
 * Global CAE client singleton accessor macro
 * Returns pointer to the global CAE client instance
 */
#define WRP_CAE_CLIENT                                                         \
  (&(*HSHM_GET_GLOBAL_PTR_VAR(wrp_cae::core::Client,                         \
                              g_cae_client)))

#endif  // WRP_CAE_CORE_CLIENT_H_
