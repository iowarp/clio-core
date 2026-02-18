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

#ifndef WRP_CTE_CORE_CONTENT_TRANSFER_ENGINE_H_
#define WRP_CTE_CORE_CONTENT_TRANSFER_ENGINE_H_

#include "hermes_shm/util/singleton.h"
#include <chimaera/chimaera.h>

namespace wrp_cte::core {

/**
 * Main Content Transfer Engine manager class
 * 
 * Central coordinator for the CTE system initialization and state management.
 * Manages initialization state to prevent race conditions during startup.
 * Uses HSHM global pointer variable singleton pattern.
 */
class ContentTransferEngine {
public:
  /**
   * Constructor
   */
  ContentTransferEngine() : is_initializing_(false), is_initialized_(false) {}

  /**
   * Destructor - handles automatic finalization
   */
  ~ContentTransferEngine() = default;

  /**
   * Initialize client components
   * Configuration is now provided via chimaera compose using WRP_RUNTIME_CONF
   * @param pool_query Pool query type for CTE container creation
   * @return true if initialization successful, false otherwise
   */
  bool ClientInit(const chi::PoolQuery &pool_query = chi::PoolQuery::Dynamic());

  /**
   * Check if CTE is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const {
    return is_initialized_ && !is_initializing_;
  }

  /**
   * Check if CTE is currently in the process of initializing
   * @return true if initialization is in progress, false otherwise
   */
  bool IsInitializing() const {
    return is_initializing_;
  }

  /**
   * Query tags by regex pattern
   * @param tag_re Tag regex pattern
   * @param max_tags Maximum number of tags to return (0 = no limit)
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Vector of matching tag names
   */
  std::vector<std::string> TagQuery(const std::string &tag_re,
                                     chi::u32 max_tags = 0,
                                     const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast());

  /**
   * Query blobs by tag and blob regex patterns
   * @param tag_re Tag regex pattern
   * @param blob_re Blob regex pattern
   * @param max_blobs Maximum number of blobs to return (0 = no limit)
   * @param pool_query Pool query for routing (default: Broadcast)
   * @return Vector of pairs (tag_name, blob_name) for matching blobs
   */
  std::vector<std::pair<std::string, std::string>> BlobQuery(const std::string &tag_re,
                                      const std::string &blob_re,
                                      chi::u32 max_blobs = 0,
                                      const chi::PoolQuery &pool_query = chi::PoolQuery::Broadcast());

private:
  bool is_initializing_;  /**< True during initialization process */
  bool is_initialized_;   /**< True when fully initialized */
};

}  // namespace wrp_cte::core

// Global pointer variable declaration for ContentTransferEngine singleton (outside namespace)
HSHM_DEFINE_GLOBAL_PTR_VAR_H(wrp_cte::core::ContentTransferEngine, g_cte_manager);

// Macro for accessing the ContentTransferEngine singleton using global pointer variable
#define CTE_MANAGER (HSHM_GET_GLOBAL_PTR_VAR(wrp_cte::core::ContentTransferEngine, g_cte_manager))

#endif  // WRP_CTE_CORE_CONTENT_TRANSFER_ENGINE_H_