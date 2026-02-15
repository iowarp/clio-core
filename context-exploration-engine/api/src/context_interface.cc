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

#include <wrp_cee/api/context_interface.h>
#include <wrp_cae/core/core_client.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cae/core/constants.h>
#include <chimaera/chimaera.h>
#include <iostream>

namespace iowarp {

ContextInterface::ContextInterface() : is_initialized_(false) {
  // Lazy initialization - defer Chimaera/CAE/CTE init until first operation
  // This allows object construction without requiring a running runtime
}

bool ContextInterface::EnsureInitialized() {
  if (is_initialized_) {
    return true;
  }

  // Cache initialization failure to avoid repeatedly creating/destroying
  // ZMQ contexts, which corrupts mimalloc's internal state in Python 3.13
  static bool init_failed = false;
  if (init_failed) {
    return false;
  }

  // Initialize Chimaera as a client for the context interface
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    std::cerr << "Error: Failed to initialize Chimaera client" << std::endl;
    init_failed = true;
    return false;
  }

  // Initialize CAE client (which initializes CTE internally)
  if (!WRP_CAE_CLIENT_INIT()) {
    std::cerr << "Error: Failed to initialize CAE client" << std::endl;
    return false;
  }

  // Verify Chimaera IPC is available
  auto* ipc_manager = CHI_IPC;
  if (!ipc_manager) {
    std::cerr << "Error: Chimaera IPC not initialized. Is the runtime running?" << std::endl;
    return false;
  }

  is_initialized_ = true;
  return true;
}

ContextInterface::~ContextInterface() {
  // Cleanup if needed
}

int ContextInterface::ContextBundle(
    const std::vector<wrp_cae::core::AssimilationCtx> &bundle) {
  if (!EnsureInitialized()) {
    std::cerr << "Error: ContextInterface failed to initialize" << std::endl;
    return 1;
  }

  if (bundle.empty()) {
    std::cerr << "Warning: Empty bundle provided to ContextBundle" << std::endl;
    return 0;
  }

  try {
    // Connect to CAE core container using the standard pool ID
    wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);

    // Call AsyncParseOmni with vector of contexts and wait for completion
    auto task = cae_client.AsyncParseOmni(bundle);
    task.Wait();

    chi::u32 result = task->result_code_;
    chi::u32 num_tasks_scheduled = task->num_tasks_scheduled_;

    if (result != 0) {
      std::cerr << "Error: ParseOmni failed with result code " << result << std::endl;
      return static_cast<int>(result);
    }

    std::cout << "ContextBundle completed successfully!" << std::endl;
    std::cout << "  Tasks scheduled: " << num_tasks_scheduled << std::endl;

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error in ContextBundle: " << e.what() << std::endl;
    return 1;
  }
}

std::vector<std::string> ContextInterface::ContextQuery(
    const std::string &tag_re,
    const std::string &blob_re,
    unsigned int max_results) {
  if (!EnsureInitialized()) {
    std::cerr << "Error: ContextInterface failed to initialize" << std::endl;
    return std::vector<std::string>();
  }

  try {
    // Get the CTE client singleton
    auto* cte_client = WRP_CTE_CLIENT;
    if (!cte_client) {
      std::cerr << "Error: CTE client not initialized" << std::endl;
      return std::vector<std::string>();
    }

    // Call AsyncBlobQuery with tag and blob regex patterns
    // Use Broadcast to query across all nodes
    auto task = cte_client->AsyncBlobQuery(
        tag_re,
        blob_re,
        max_results,  // max_blobs (0 = unlimited)
        chi::PoolQuery::Broadcast());
    task.Wait();

    // Extract results from task - blob names only
    std::vector<std::string> results;
    for (size_t i = 0; i < task->blob_names_.size(); ++i) {
      results.push_back(task->blob_names_[i]);
    }

    return results;

  } catch (const std::exception& e) {
    std::cerr << "Error in ContextQuery: " << e.what() << std::endl;
    return std::vector<std::string>();
  }
}

std::vector<std::string> ContextInterface::ContextRetrieve(
    const std::string &tag_re,
    const std::string &blob_re,
    unsigned int max_results,
    size_t max_context_size,
    unsigned int batch_size) {
  if (!EnsureInitialized()) {
    std::cerr << "Error: ContextInterface failed to initialize" << std::endl;
    return std::vector<std::string>();
  }

  try {
    // Get the CTE client singleton
    auto* cte_client = WRP_CTE_CLIENT;
    if (!cte_client) {
      std::cerr << "Error: CTE client not initialized" << std::endl;
      return std::vector<std::string>();
    }

    // Get IPC manager for buffer allocation
    auto* ipc_manager = CHI_IPC;
    if (!ipc_manager) {
      std::cerr << "Error: Chimaera IPC not initialized" << std::endl;
      return std::vector<std::string>();
    }

    // Use AsyncBlobQuery to get list of blobs matching the pattern
    auto query_task = cte_client->AsyncBlobQuery(
        tag_re,
        blob_re,
        max_results,
        chi::PoolQuery::Broadcast());
    query_task.Wait();

    // Build query_results from separate tag_names_ and blob_names_ vectors
    std::vector<std::pair<std::string, std::string>> query_results;
    size_t result_count = std::min(query_task->tag_names_.size(), query_task->blob_names_.size());
    for (size_t i = 0; i < result_count; ++i) {
      query_results.emplace_back(query_task->tag_names_[i], query_task->blob_names_[i]);
    }

    if (query_results.empty()) {
      std::cout << "ContextRetrieve: No blobs found matching patterns" << std::endl;
      return std::vector<std::string>();
    }

    std::cout << "ContextRetrieve: Found " << query_results.size() << " matching blobs" << std::endl;

    // Allocate buffer for packed context
    hipc::FullPtr<char> context_buffer = ipc_manager->AllocateBuffer(max_context_size);
    if (context_buffer.IsNull()) {
      std::cerr << "Error: Failed to allocate context buffer" << std::endl;
      return std::vector<std::string>();
    }

    size_t buffer_offset = 0;  // Current offset in context buffer
    std::vector<std::string> results;

    // Process blobs in batches
    for (size_t batch_start = 0; batch_start < query_results.size(); batch_start += batch_size) {
      size_t batch_end = std::min(batch_start + batch_size, query_results.size());
      size_t batch_count = batch_end - batch_start;

      // Schedule AsyncGetBlob operations for this batch
      std::vector<chi::Future<wrp_cte::core::GetBlobTask>> tasks;
      tasks.reserve(batch_count);

      for (size_t i = batch_start; i < batch_end; ++i) {
        const auto& [tag_name, blob_name] = query_results[i];

        // Get or create tag to get tag_id
        auto tag_task = cte_client->AsyncGetOrCreateTag(tag_name);
        tag_task.Wait();
        wrp_cte::core::TagId tag_id = tag_task->tag_id_;
        if (tag_id.IsNull()) {
          std::cerr << "Warning: Failed to get tag '" << tag_name << "', skipping blob" << std::endl;
          continue;
        }

        // Get blob size first
        auto size_task = cte_client->AsyncGetBlobSize(tag_id, blob_name);
        size_task.Wait();
        chi::u64 blob_size = size_task->size_;
        if (blob_size == 0) {
          std::cerr << "Warning: Blob '" << blob_name << "' has zero size, skipping" << std::endl;
          continue;
        }

        // Check if blob fits in buffer
        if (buffer_offset + blob_size > max_context_size) {
          std::cout << "ContextRetrieve: Not enough space for blob '" << blob_name
                    << "' (" << blob_size << " bytes), stopping" << std::endl;
          break;
        }

        // Calculate buffer pointer for this blob
        hipc::ShmPtr<> blob_buffer_ptr;
        blob_buffer_ptr.alloc_id_ = context_buffer.shm_.alloc_id_;
        blob_buffer_ptr.off_ = context_buffer.shm_.off_.load() + buffer_offset;

        // Schedule AsyncGetBlob
        auto task = cte_client->AsyncGetBlob(
            tag_id,
            blob_name,
            0,              // offset within blob
            blob_size,      // size to read
            0,              // flags
            blob_buffer_ptr);

        tasks.push_back(task);
        buffer_offset += blob_size;
      }

      // Wait for all tasks in this batch to complete
      for (auto& task : tasks) {
        task.Wait();
        if (task->return_code_ != 0) {
          std::cerr << "Warning: GetBlob failed for a blob in batch" << std::endl;
        }
      }

      // Task cleanup is handled by Future destructors when 'tasks'
      // goes out of scope. Manual DelTask here causes a double-free.
    }

    // Convert buffer to std::string
    std::string packed_context;
    if (buffer_offset > 0) {
      packed_context.assign(context_buffer.ptr_, buffer_offset);
      std::cout << "ContextRetrieve: Retrieved " << buffer_offset
                << " bytes of packed context" << std::endl;
    }

    // Free the allocated buffer
    ipc_manager->FreeBuffer(context_buffer);

    // Return the packed context as a vector with single string
    if (!packed_context.empty()) {
      results.push_back(packed_context);
    }

    return results;

  } catch (const std::exception& e) {
    std::cerr << "Error in ContextRetrieve: " << e.what() << std::endl;
    return std::vector<std::string>();
  }
}

int ContextInterface::ContextSplice(
    const std::string &new_ctx,
    const std::string &tag_re,
    const std::string &blob_re) {
  (void)new_ctx;  // Suppress unused parameter warning
  (void)tag_re;   // Suppress unused parameter warning
  (void)blob_re;  // Suppress unused parameter warning

  // Not yet implemented
  std::cerr << "Warning: ContextSplice is not yet implemented" << std::endl;
  return 1;
}

int ContextInterface::ContextDestroy(
    const std::vector<std::string> &context_names) {
  if (!EnsureInitialized()) {
    std::cerr << "Error: ContextInterface failed to initialize" << std::endl;
    return 1;
  }

  if (context_names.empty()) {
    std::cerr << "Warning: Empty context_names list provided to ContextDestroy" << std::endl;
    return 0;
  }

  try {
    // Get the CTE client singleton
    auto* cte_client = WRP_CTE_CLIENT;
    if (!cte_client) {
      std::cerr << "Error: CTE client not initialized" << std::endl;
      return 1;
    }

    // Iterate over each context name and delete the corresponding tag
    int error_count = 0;
    for (const auto& context_name : context_names) {
      auto task = cte_client->AsyncDelTag(context_name);
      task.Wait();
      bool result = (task->return_code_ == 0);
      if (!result) {
        std::cerr << "Error: Failed to delete context '" << context_name << "'" << std::endl;
        error_count++;
      } else {
        std::cout << "Successfully deleted context: " << context_name << std::endl;
      }
    }

    if (error_count > 0) {
      std::cerr << "ContextDestroy completed with " << error_count << " error(s)" << std::endl;
      return 1;
    }

    std::cout << "ContextDestroy completed successfully!" << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error in ContextDestroy: " << e.what() << std::endl;
    return 1;
  }
}

}  // namespace iowarp
