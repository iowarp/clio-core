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

#include <clio_cee/api/context_interface.h>
#include <clio_cae/core/core_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cae/core/constants.h>
#include <clio_runtime/clio_runtime.h>
#include <iostream>
#include <clio_ctp/util/logging.h>

namespace iowarp {

ContextInterface::ContextInterface() : is_initialized_(false) {
  // Lazy initialization - defer CLIO Runtime/CAE/CTE init until first operation
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

  // Initialize CLIO Runtime as a client for the context interface
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Clio client");
    init_failed = true;
    return false;
  }

  // Initialize CAE client (which initializes CTE internally)
  if (!CLIO_CAE_CLIENT_INIT()) {
    HLOG(kError, "Failed to initialize CAE client");
    return false;
  }

  // Verify CLIO Runtime IPC is available
  auto* ipc_manager = CLIO_IPC;
  if (!ipc_manager) {
    HLOG(kError, "Clio IPC not initialized. Is the runtime running?");
    return false;
  }

  is_initialized_ = true;
  return true;
}

ContextInterface::~ContextInterface() {
  // Cleanup if needed
}

int ContextInterface::ContextBundle(
    const std::vector<clio::cae::core::AssimilationCtx> &bundle) {
  if (!EnsureInitialized()) {
    HLOG(kError, "ContextInterface failed to initialize");
    return 1;
  }

  if (bundle.empty()) {
    HLOG(kWarning, "Empty bundle provided to ContextBundle");
    return 0;
  }

  try {
    // Connect to CAE core container using the standard pool ID
    clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);

    // Call AsyncParseOmni with vector of contexts and wait for completion
    auto task = cae_client.AsyncParseOmni(bundle);
    task.Wait();

    clio::run::u32 result = task->result_code_;
    clio::run::u32 num_tasks_scheduled = task->num_tasks_scheduled_;

    if (result != 0) {
      HLOG(kError, "ParseOmni failed with result code {}", result);
      return static_cast<int>(result);
    }

    HLOG(kSuccess, "ContextBundle completed successfully!");
    HLOG(kInfo, "  Tasks scheduled: {}", num_tasks_scheduled);

    return 0;

  } catch (const std::exception& e) {
    HLOG(kError, "Error in ContextBundle: {}", e.what());
    return 1;
  }
}

std::vector<std::string> ContextInterface::ContextQuery(
    const std::string &tag_re,
    const std::string &blob_re,
    unsigned int max_results,
    const std::string &prompt,
    uint64_t time_begin,
    uint64_t time_end) {
  if (!EnsureInitialized()) {
    HLOG(kError, "ContextInterface failed to initialize");
    return std::vector<std::string>();
  }

  try {
    auto* cte_client = CLIO_CTE_CLIENT;
    if (!cte_client) {
      HLOG(kError, "CTE client not initialized");
      return std::vector<std::string>();
    }

    // Mode 1: temporal — any non-zero bound routes here.
    if (time_begin != 0 || time_end != 0) {
      auto task = cte_client->AsyncTemporalSearch(
          tag_re, blob_re,
          time_begin, time_end,
          static_cast<clio::run::u32>(max_results),
          clio::run::PoolQuery::Broadcast());
      task.Wait();

      std::vector<std::string> results;
      results.reserve(task->results_.size());
      for (const auto& r : task->results_) {
        results.push_back(r.blob_name_);
      }
      return results;
    }

    // Mode 2: semantic — BM25 keyword search.
    if (!prompt.empty()) {
      // max_results=0 means "unlimited" for the regex path, but BM25 needs
      // a positive top-k; fall back to the SemanticSearchTask default (10).
      clio::run::u32 k = max_results > 0 ? max_results : 10;
      auto task = cte_client->AsyncSemanticSearch(
          tag_re, blob_re, prompt, k, clio::run::PoolQuery::Broadcast());
      task.Wait();

      std::vector<std::string> results;
      results.reserve(task->results_.size());
      for (const auto& r : task->results_) {
        results.push_back(r.blob_name_);
      }
      return results;
    }

    // Mode 3: regex-only via BlobQuery.
    auto task = cte_client->AsyncBlobQuery(
        tag_re, blob_re,
        max_results,
        clio::run::PoolQuery::Broadcast());
    task.Wait();

    std::vector<std::string> results;
    for (size_t i = 0; i < task->blob_names_.size(); ++i) {
      results.push_back(task->blob_names_[i]);
    }
    return results;

  } catch (const std::exception& e) {
    HLOG(kError, "Error in ContextQuery: {}", e.what());
    return std::vector<std::string>();
  }
}

std::vector<std::string> ContextInterface::ContextRetrieve(
    const std::string &tag_re,
    const std::string &blob_re,
    unsigned int max_results,
    size_t max_context_size,
    unsigned int batch_size,
    const std::string &prompt,
    uint64_t time_begin,
    uint64_t time_end) {
  if (!EnsureInitialized()) {
    HLOG(kError, "ContextInterface failed to initialize");
    return std::vector<std::string>();
  }

  try {
    // Get the CTE client singleton
    auto* cte_client = CLIO_CTE_CLIENT;
    if (!cte_client) {
      HLOG(kError, "CTE client not initialized");
      return std::vector<std::string>();
    }

    // Get IPC manager for buffer allocation
    auto* ipc_manager = CLIO_IPC;
    if (!ipc_manager) {
      HLOG(kError, "Clio IPC not initialized");
      return std::vector<std::string>();
    }

    // Same three-mode dispatch as ContextQuery to resolve matching blob names.
    std::vector<std::pair<std::string, std::string>> query_results;

    if (time_begin != 0 || time_end != 0) {
      // Mode 1: temporal search
      auto task = cte_client->AsyncTemporalSearch(
          tag_re, blob_re,
          time_begin, time_end,
          static_cast<clio::run::u32>(max_results),
          clio::run::PoolQuery::Broadcast());
      task.Wait();
      query_results.reserve(task->results_.size());
      for (const auto& r : task->results_) {
        query_results.emplace_back(r.tag_name_, r.blob_name_);
      }
    } else if (!prompt.empty()) {
      // Mode 2: BM25 semantic search
      clio::run::u32 k = max_results > 0 ? max_results : 10;
      auto task = cte_client->AsyncSemanticSearch(
          tag_re, blob_re, prompt, k, clio::run::PoolQuery::Broadcast());
      task.Wait();
      query_results.reserve(task->results_.size());
      for (const auto& r : task->results_) {
        query_results.emplace_back(r.tag_name_, r.blob_name_);
      }
    } else {
      // Mode 3: regex-only via BlobQuery
      auto query_task = cte_client->AsyncBlobQuery(
          tag_re, blob_re, max_results, clio::run::PoolQuery::Broadcast());
      query_task.Wait();
      size_t result_count = std::min(query_task->tag_names_.size(),
                                     query_task->blob_names_.size());
      query_results.reserve(result_count);
      for (size_t i = 0; i < result_count; ++i) {
        query_results.emplace_back(query_task->tag_names_[i],
                                   query_task->blob_names_[i]);
      }
    }

    if (query_results.empty()) {
      HLOG(kInfo, "ContextRetrieve: No blobs found matching patterns");
      return std::vector<std::string>();
    }

    HLOG(kInfo, "ContextRetrieve: Found {} matching blobs", query_results.size());

    // Allocate buffer for packed context
    ctp::ipc::FullPtr<char> context_buffer = ipc_manager->AllocateBuffer(max_context_size);
    if (context_buffer.IsNull()) {
      HLOG(kError, "Failed to allocate context buffer");
      return std::vector<std::string>();
    }

    size_t buffer_offset = 0;  // Current offset in context buffer
    std::vector<std::string> results;

    // Process blobs in batches
    for (size_t batch_start = 0; batch_start < query_results.size(); batch_start += batch_size) {
      size_t batch_end = std::min(batch_start + batch_size, query_results.size());
      size_t batch_count = batch_end - batch_start;

      // Schedule AsyncGetBlob operations for this batch
      std::vector<clio::run::Future<clio::cte::core::GetBlobTask>> tasks;
      tasks.reserve(batch_count);

      for (size_t i = batch_start; i < batch_end; ++i) {
        const auto& [tag_name, blob_name] = query_results[i];

        // Get or create tag to get tag_id
        auto tag_task = cte_client->AsyncGetOrCreateTag(tag_name);
        tag_task.Wait();
        clio::cte::core::TagId tag_id = tag_task->tag_id_;
        if (tag_id.IsNull()) {
          HLOG(kWarning, "Failed to get tag '{}', skipping blob", tag_name);
          continue;
        }

        // Get blob size first
        auto size_task = cte_client->AsyncGetBlobSize(tag_id, blob_name);
        size_task.Wait();
        clio::run::u64 blob_size = size_task->size_;
        if (blob_size == 0) {
          HLOG(kWarning, "Blob '{}' has zero size, skipping", blob_name);
          continue;
        }

        // Check if blob fits in buffer
        if (buffer_offset + blob_size > max_context_size) {
          HLOG(kInfo, "ContextRetrieve: Not enough space for blob '{}' ({} bytes), stopping",
               blob_name, blob_size);
          break;
        }

        // Calculate buffer pointer for this blob
        ctp::ipc::ShmPtr<> blob_buffer_ptr;
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
          HLOG(kWarning, "GetBlob failed for a blob in batch");
        }
      }

      // Task cleanup is handled by Future destructors when 'tasks'
      // goes out of scope. Manual DelTask here causes a double-free.
    }

    // Convert buffer to std::string
    std::string packed_context;
    if (buffer_offset > 0) {
      packed_context.assign(context_buffer.ptr_, buffer_offset);
      HLOG(kSuccess, "ContextRetrieve: Retrieved {} bytes of packed context", buffer_offset);
    }

    // Free the allocated buffer
    ipc_manager->FreeBuffer(context_buffer);

    // Return the packed context as a vector with single string
    if (!packed_context.empty()) {
      results.push_back(packed_context);
    }

    return results;

  } catch (const std::exception& e) {
    HLOG(kError, "Error in ContextRetrieve: {}", e.what());
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
  HLOG(kWarning, "ContextSplice is not yet implemented");
  return 1;
}

int ContextInterface::ContextDestroy(
    const std::vector<std::string> &context_names) {
  if (!EnsureInitialized()) {
    HLOG(kError, "ContextInterface failed to initialize");
    return 1;
  }

  if (context_names.empty()) {
    HLOG(kWarning, "Empty context_names list provided to ContextDestroy");
    return 0;
  }

  try {
    // Get the CTE client singleton
    auto* cte_client = CLIO_CTE_CLIENT;
    if (!cte_client) {
      HLOG(kError, "CTE client not initialized");
      return 1;
    }

    // Iterate over each context name and delete the corresponding tag
    int error_count = 0;
    for (const auto& context_name : context_names) {
      auto task = cte_client->AsyncDelTag(context_name);
      task.Wait();
      bool result = (task->return_code_ == 0);
      if (!result) {
        HLOG(kError, "Failed to delete context '{}'", context_name);
        error_count++;
      } else {
        HLOG(kSuccess, "Successfully deleted context: {}", context_name);
      }
    }

    if (error_count > 0) {
      HLOG(kError, "ContextDestroy completed with {} error(s)", error_count);
      return 1;
    }

    HLOG(kSuccess, "ContextDestroy completed successfully!");
    return 0;

  } catch (const std::exception& e) {
    HLOG(kError, "Error in ContextDestroy: {}", e.what());
    return 1;
  }
}

}  // namespace iowarp
