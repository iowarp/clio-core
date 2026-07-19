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

#include <clio_cae/core/constants.h>
#include <clio_cte/core/core_client.h>  // For CLIO_CTE_CLIENT_INIT

// Must include clio_cae core_client.h last to avoid circular dependency
#include <clio_cae/core/core_client.h>

// Global CAE client singleton definition
CTP_DEFINE_GLOBAL_PTR_VAR_CC(clio::cae::core::Client, g_cae_client);

/**
 * Initialize CAE client singleton
 * Calls CLIO_CTE_CLIENT_INIT internally to ensure CTE is initialized
 */
bool CLIO_CAE_CLIENT_INIT(const std::string &config_path,
                         const clio::run::PoolQuery &pool_query) {
  // Note: Default parameters are defined in the header

  // Static guard to prevent multiple initializations
  static bool is_initialized = false;
  if (is_initialized) {
    return true;
  }

  // Suppress unused parameter warning
  (void)config_path;

  // First, ensure CTE client is initialized (CAE depends on CTE)
  // LCOV_EXCL_START: defensive init-failure returns. In a healthy test
  // environment CTE init and singleton allocation always succeed; these
  // branches only fire on a corrupted runtime that cannot be staged here.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT(config_path, pool_query)) {
    return false;
  }
  // LCOV_EXCL_STOP

  // Get or create the CAE client singleton
  auto *cae_client = CTP_GET_GLOBAL_PTR_VAR(clio::cae::core::Client, g_cae_client);
  if (!cae_client) {
    return false;  // LCOV_EXCL_LINE: singleton allocation never fails in tests
  }

  // Create the CAE pool
  clio::cae::core::CreateParams params;
  auto create_task = cae_client->AsyncCreate(
      pool_query,
      "cae_client_pool",
      clio::cae::core::kCaePoolId,
      params);
  create_task.Wait();

  // Update client pool_id_ with the actual pool ID from the task
  cae_client->pool_id_ = create_task->new_pool_id_;
  cae_client->return_code_ = create_task->return_code_;

  // Check if creation was successful
  if (create_task->GetReturnCode() != 0) {
    return false;  // LCOV_EXCL_LINE: pool creation does not fail in tests
  }

  // Mark as initialized
  is_initialized = true;

  return true;
}
