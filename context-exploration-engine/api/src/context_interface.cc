#include <wrp_cee/api/context_interface.h>
#include <wrp_cae/core/core_client.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cae/core/constants.h>
#include <chimaera/chimaera.h>
#include <iostream>

namespace iowarp {

ContextInterface::ContextInterface() : is_initialized_(false) {
  // Initialize CAE client (which initializes CTE and Chimaera internally)
  if (!WRP_CAE_CLIENT_INIT()) {
    std::cerr << "Error: Failed to initialize CAE client" << std::endl;
    return;
  }

  // Verify Chimaera IPC is available
  auto* ipc_manager = CHI_IPC;
  if (!ipc_manager) {
    std::cerr << "Error: Chimaera IPC not initialized. Is the runtime running?" << std::endl;
    return;
  }

  is_initialized_ = true;
}

ContextInterface::~ContextInterface() {
  // Cleanup if needed
}

int ContextInterface::ContextBundle(
    const std::vector<wrp_cae::core::AssimilationCtx> &bundle) {
  if (!is_initialized_) {
    std::cerr << "Error: ContextInterface not initialized" << std::endl;
    return 1;
  }

  if (bundle.empty()) {
    std::cerr << "Warning: Empty bundle provided to ContextBundle" << std::endl;
    return 0;
  }

  try {
    // Connect to CAE core container using the standard pool ID
    wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);

    // Call ParseOmni with vector of contexts
    chi::u32 num_tasks_scheduled = 0;
    chi::u32 result = cae_client.ParseOmni(HSHM_MCTX, bundle, num_tasks_scheduled);

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
  if (!is_initialized_) {
    std::cerr << "Error: ContextInterface not initialized" << std::endl;
    return std::vector<std::string>();
  }

  try {
    // Get the CTE client singleton
    auto* cte_client = WRP_CTE_CLIENT;
    if (!cte_client) {
      std::cerr << "Error: CTE client not initialized" << std::endl;
      return std::vector<std::string>();
    }

    // Call BlobQuery with tag and blob regex patterns
    // Use Broadcast to query across all nodes
    auto query_results = cte_client->BlobQuery(
        HSHM_MCTX,
        tag_re,
        blob_re,
        max_results,  // max_blobs (0 = unlimited)
        chi::PoolQuery::Broadcast());

    // Convert pair<string, string> results to just blob names
    std::vector<std::string> results;
    for (const auto& pair : query_results) {
      results.push_back(pair.second);  // pair.second is the blob name
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
  if (!is_initialized_) {
    std::cerr << "Error: ContextInterface not initialized" << std::endl;
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

    // Use BlobQuery to get list of blobs matching the pattern
    auto query_results = cte_client->BlobQuery(
        HSHM_MCTX,
        tag_re,
        blob_re,
        max_results,
        chi::PoolQuery::Broadcast());

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
      std::vector<hipc::FullPtr<wrp_cte::core::GetBlobTask>> tasks;
      tasks.reserve(batch_count);

      for (size_t i = batch_start; i < batch_end; ++i) {
        const auto& [tag_name, blob_name] = query_results[i];

        // Get or create tag to get tag_id
        wrp_cte::core::TagId tag_id = cte_client->GetOrCreateTag(HSHM_MCTX, tag_name);
        if (tag_id.IsNull()) {
          std::cerr << "Warning: Failed to get tag '" << tag_name << "', skipping blob" << std::endl;
          continue;
        }

        // Get blob size first
        chi::u64 blob_size = cte_client->GetBlobSize(HSHM_MCTX, tag_id, blob_name);
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
        hipc::Pointer blob_buffer_ptr = context_buffer.shm_;
        blob_buffer_ptr.off_ = blob_buffer_ptr.off_ + buffer_offset;

        // Schedule AsyncGetBlob
        auto task = cte_client->AsyncGetBlob(
            HSHM_MCTX,
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
        task->Wait();
        if (task->return_code_.load() != 0) {
          std::cerr << "Warning: GetBlob failed for a blob in batch" << std::endl;
        }
      }

      // Delete all tasks in this batch
      for (auto& task : tasks) {
        ipc_manager->DelTask(task);
      }
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
  if (!is_initialized_) {
    std::cerr << "Error: ContextInterface not initialized" << std::endl;
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
      bool result = cte_client->DelTag(HSHM_MCTX, context_name);
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
