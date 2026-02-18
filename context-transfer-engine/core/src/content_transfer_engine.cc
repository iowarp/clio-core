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

#include <chimaera/chimaera.h>
#include <wrp_cte/core/content_transfer_engine.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_config.h>
#include <string>

// Define global pointer variable in source file (outside namespace)
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(wrp_cte::core::ContentTransferEngine,
                              g_cte_manager);

namespace wrp_cte::core {

bool ContentTransferEngine::ClientInit(const chi::PoolQuery &pool_query) {
  // Check for race conditions - if already initialized or initializing
  if (is_initialized_) {
    return true;
  }
  if (is_initializing_) {
    return true;
  }
  auto *chimaera_manager = CHI_CHIMAERA_MANAGER;
  if (chimaera_manager->IsInitializing()) {
    return true;
  }

  // Set initializing flag
  is_initializing_ = true;

  // Initialize Chimaera client
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    is_initializing_ = false;
    return false;
  }

  // Initialize CTE core client
  auto *cte_client = WRP_CTE_CLIENT;

  // Create CreateParams without config - configuration is now provided via chimaera compose
  CreateParams params;

  // Create CTE Core container using constants from core_tasks.h and specified pool_query
  auto create_task = cte_client->AsyncCreate(pool_query,
                                              wrp_cte::core::kCtePoolName,
                                              wrp_cte::core::kCtePoolId,
                                              params);
  create_task.Wait();

  // Check if Create operation succeeded
  chi::u32 return_code = create_task->GetReturnCode();
  if (return_code != 0) {
    HLOG(kError, "CTE ClientInit: Failed to create CTE pool '{}' with return code: {}",
          wrp_cte::core::kCtePoolName, return_code);
    is_initializing_ = false;
    return false;
  }

  // Update client pool_id_ with the actual pool ID from the task
  cte_client->pool_id_ = create_task->new_pool_id_;

  // Delete the create task

  // Mark as successfully initialized
  is_initialized_ = true;
  is_initializing_ = false;

  return true;
}

std::vector<std::string> ContentTransferEngine::TagQuery(
    const std::string &tag_re,
    chi::u32 max_tags,
    const chi::PoolQuery &pool_query) {
  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncTagQuery(tag_re, max_tags, pool_query);
  task.Wait();

  std::vector<std::string> results = task->results_;
  return results;
}

std::vector<std::pair<std::string, std::string>> ContentTransferEngine::BlobQuery(
    const std::string &tag_re,
    const std::string &blob_re,
    chi::u32 max_blobs,
    const chi::PoolQuery &pool_query) {
  auto *cte_client = WRP_CTE_CLIENT;
  auto task = cte_client->AsyncBlobQuery(tag_re, blob_re, max_blobs, pool_query);
  task.Wait();

  std::vector<std::pair<std::string, std::string>> results;
  for (size_t i = 0; i < task->tag_names_.size(); ++i) {
    results.emplace_back(task->tag_names_[i], task->blob_names_[i]);
  }

  return results;
}

} // namespace wrp_cte::core