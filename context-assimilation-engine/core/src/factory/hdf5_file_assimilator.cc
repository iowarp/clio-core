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
#include <chimaera/ipc_manager.h>
#ifndef _WIN32
#include <fnmatch.h>
#include <sys/stat.h>
#endif
#include <wrp_cae/core/constants.h>  // For kCaePoolId
#include <wrp_cae/core/core_client.h>
#include <wrp_cae/core/core_tasks.h>
#include <wrp_cae/core/factory/hdf5_file_assimilator.h>

#include <algorithm>
#include <cstring>
#include <vector>

// Include wrp_cte headers after closing any wrp_cae namespace to avoid Method
// namespace collision
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

namespace wrp_cae::core {

Hdf5FileAssimilator::Hdf5FileAssimilator(
    std::shared_ptr<wrp_cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

chi::TaskResume Hdf5FileAssimilator::Schedule(const AssimilationCtx& ctx,
                                              int& error_code) {
  HLOG(kDebug, "Hdf5FileAssimilator::Schedule() - ENTRY");
  HLOG(kDebug, "  ctx.src: '{}'", ctx.src);
  HLOG(kDebug, "  ctx.dst: '{}'", ctx.dst);
  HLOG(kDebug, "  ctx.format: '{}'", ctx.format);

  // Validate destination protocol
  std::string dst_protocol = GetUrlProtocol(ctx.dst);
  HLOG(kDebug, "Hdf5FileAssimilator: Extracted destination protocol: '{}'",
       dst_protocol);
  if (dst_protocol != "iowarp") {
    HLOG(kError,
         "Hdf5FileAssimilator: Destination protocol must be 'iowarp', got '{}'",
         dst_protocol);
    error_code = -1;
    co_return;
  }

  // Extract tag prefix from destination URL (remove iowarp:: prefix)
  std::string tag_prefix = GetUrlPath(ctx.dst);
  HLOG(kDebug, "Hdf5FileAssimilator: Extracted tag prefix: '{}'", tag_prefix);
  if (tag_prefix.empty()) {
    HLOG(kError,
         "Hdf5FileAssimilator: Invalid destination URL, no tag name found");
    error_code = -2;
    co_return;
  }

  // Handle dependency-based scheduling
  if (!ctx.depends_on.empty()) {
    // TODO: Implement dependency handling
    // For now, log that dependencies are not yet supported
    HLOG(kDebug,
         "Hdf5FileAssimilator: Dependency handling not yet implemented "
         "(depends_on: {})",
         ctx.depends_on);
    error_code = 0;
    co_return;
  }

  // Extract source file path
  std::string src_path = GetUrlPath(ctx.src);
  HLOG(kDebug, "Hdf5FileAssimilator: Extracted source file path: '{}'",
       src_path);
  if (src_path.empty()) {
    HLOG(kError, "Hdf5FileAssimilator: Invalid source URL, no file path found");
    error_code = -3;
    co_return;
  }

  // Open HDF5 file
  HLOG(kDebug, "Hdf5FileAssimilator: Opening HDF5 file...");
  hid_t file_id = OpenHdf5File(src_path);
  if (file_id < 0) {
    HLOG(kError, "Hdf5FileAssimilator: Failed to open HDF5 file '{}'",
         src_path);
    error_code = -4;
    co_return;
  }
  HLOG(kDebug,
       "Hdf5FileAssimilator: HDF5 file opened successfully (file_id: {})",
       file_id);

  // Discover all datasets in the file
  HLOG(kDebug, "Hdf5FileAssimilator: Discovering datasets...");
  std::vector<std::string> dataset_paths;
  int discover_result = DiscoverDatasets(file_id, dataset_paths);
  if (discover_result != 0) {
    HLOG(kError, "Hdf5FileAssimilator: Failed to discover datasets in '{}'",
         src_path);
    CloseHdf5File(file_id);
    error_code = -5;
    co_return;
  }

  HLOG(kDebug, "Hdf5FileAssimilator: Discovered {} dataset(s) in '{}'",
       dataset_paths.size(), src_path);

  // Apply dataset filtering if patterns are specified
  std::vector<std::string> filtered_paths;
  if (!ctx.include_patterns.empty() || !ctx.exclude_patterns.empty()) {
    HLOG(kDebug, "Hdf5FileAssimilator: Applying dataset filters...");
    HLOG(kDebug, "  Include patterns: {}", ctx.include_patterns.size());
    for (const auto& pattern : ctx.include_patterns) {
      HLOG(kDebug, "    - '{}'", pattern);
    }
    HLOG(kDebug, "  Exclude patterns: {}", ctx.exclude_patterns.size());
    for (const auto& pattern : ctx.exclude_patterns) {
      HLOG(kDebug, "    - '{}'", pattern);
    }

    HLOG(kDebug,
         "Hdf5FileAssimilator: Checking {} discovered datasets against filters",
         dataset_paths.size());
    // Print first 5 dataset paths for debugging
    for (size_t i = 0; i < std::min(size_t(5), dataset_paths.size()); ++i) {
      HLOG(kDebug, "  Sample dataset {}: '{}'", i + 1, dataset_paths[i]);
    }

    for (const auto& dataset_path : dataset_paths) {
      if (MatchesFilter(dataset_path, ctx.include_patterns,
                        ctx.exclude_patterns)) {
        filtered_paths.push_back(dataset_path);
      }
    }
    HLOG(kDebug, "Hdf5FileAssimilator: Filtered to {} dataset(s) (from {})",
         filtered_paths.size(), dataset_paths.size());
  } else {
    HLOG(kDebug,
         "Hdf5FileAssimilator: No dataset filters specified, processing all "
         "datasets");
    filtered_paths = dataset_paths;
  }

  // Get distributed processing info from CTE/IPC manager
  size_t num_nodes = 1;
  auto* ipc_manager = CHI_IPC;
  std::vector<chi::Host> all_hosts;
  if (ipc_manager) {
    all_hosts = ipc_manager->GetAllHosts();
    num_nodes = all_hosts.size();
    if (num_nodes > 1) {
      HLOG(kDebug,
           "Hdf5FileAssimilator: CTE distributed mode - {} nodes available",
           num_nodes);
    } else {
      HLOG(kDebug, "Hdf5FileAssimilator: Single node mode");
    }
  }

  // Close file before distributing tasks (each task will open it independently)
  HLOG(kDebug,
       "Hdf5FileAssimilator: Closing HDF5 file before distributed "
       "processing...");
  CloseHdf5File(file_id);

  // Distribute datasets across nodes using CTE task routing
  int total_errors = 0;
  if (num_nodes > 1 && !all_hosts.empty()) {
    // Distributed mode: create per-dataset tasks routed to specific nodes
    HLOG(kDebug,
         "Hdf5FileAssimilator: Creating {} distributed tasks across {} nodes",
         filtered_paths.size(), num_nodes);

    // Create a local CAE client with the correct pool_id for distributed tasks
    // Do NOT use WRP_CAE_CLIENT global singleton as it may not be properly initialized
    // with the correct pool_id from the runtime's compose configuration
    wrp_cae::core::Client cae_client(kCaePoolId);
    HLOG(kInfo, "Hdf5FileAssimilator: Created CAE client with pool_id {} for distributed tasks",
          kCaePoolId);

    // Create futures for all dataset tasks
    std::vector<chi::Future<ProcessHdf5DatasetTask>> futures;
    futures.reserve(filtered_paths.size());

    for (size_t i = 0; i < filtered_paths.size(); ++i) {
      const auto& dataset_path = filtered_paths[i];
      // Round-robin distribution to nodes using direct hash
      chi::u32 target_node = static_cast<chi::u32>(i % num_nodes);
      auto pool_query = chi::PoolQuery::DirectHash(target_node);

      HLOG(kDebug, "Hdf5FileAssimilator: Routing dataset {}/{} '{}' to node {}",
           i + 1, filtered_paths.size(), dataset_path, target_node);

      HLOG(kInfo, "Hdf5FileAssimilator: Calling AsyncProcessHdf5Dataset for pool_id={}",
            kCaePoolId);

      auto future = cae_client.AsyncProcessHdf5Dataset(
          pool_query, src_path, dataset_path, tag_prefix);

      HLOG(kInfo, "Hdf5FileAssimilator: AsyncProcessHdf5Dataset returned, task_ptr IsNull={}",
            future.GetTaskPtr().IsNull());

      futures.push_back(std::move(future));
    }

    // Wait for all tasks to complete using co_await (not blocking Wait!)
    // Using co_await allows the coroutine to yield, letting other tasks run
    HLOG(kDebug,
         "Hdf5FileAssimilator: Waiting for {} distributed tasks to complete...",
         futures.size());
    for (size_t i = 0; i < futures.size(); ++i) {
      co_await futures[i];  // Yield coroutine, allow other tasks to run
      auto task = futures[i].get();
      if (task->result_code_ != 0) {
        HLOG(kError, "Hdf5FileAssimilator: Dataset {} failed (error: {})",
             filtered_paths[i], task->result_code_);
        total_errors++;
      } else {
        HLOG(kDebug, "Hdf5FileAssimilator: Dataset {} completed successfully",
             filtered_paths[i]);
      }
    }
  } else {
    // Single node mode: process locally
    // Re-open the file for local processing
    file_id = OpenHdf5File(src_path);
    if (file_id < 0) {
      HLOG(kError,
           "Hdf5FileAssimilator: Failed to re-open HDF5 file for local "
           "processing");
      error_code = -8;
      co_return;
    }

    for (size_t i = 0; i < filtered_paths.size(); ++i) {
      const auto& dataset_path = filtered_paths[i];
      HLOG(kDebug, "Hdf5FileAssimilator: Processing dataset {}/{}: '{}'", i + 1,
           filtered_paths.size(), dataset_path);
      int result = 0;
      co_await ProcessDataset(file_id, dataset_path, tag_prefix, result);
      if (result != 0) {
        HLOG(kError,
             "Hdf5FileAssimilator: Failed to process dataset '{}' (error code: "
             "{})",
             dataset_path, result);
        total_errors++;
      } else {
        HLOG(kDebug, "Hdf5FileAssimilator: Successfully processed dataset '{}'",
             dataset_path);
      }
    }

    HLOG(kDebug, "Hdf5FileAssimilator: Closing HDF5 file...");
    CloseHdf5File(file_id);
  }

  HLOG(kDebug, "Hdf5FileAssimilator: HDF5 file closed");

  if (total_errors > 0) {
    HLOG(kError,
         "Hdf5FileAssimilator: Completed with {} error(s) out of {} dataset(s)",
         total_errors, filtered_paths.size());
    error_code = -6;
    co_return;
  }

  HLOG(
      kDebug,
      "Hdf5FileAssimilator: Successfully processed all {} dataset(s) from '{}'",
      filtered_paths.size(), src_path);
  HLOG(kDebug, "Hdf5FileAssimilator::Schedule() - EXIT (success)");

  error_code = 0;
  co_return;
}

hid_t Hdf5FileAssimilator::OpenHdf5File(const std::string& file_path) {
  HLOG(kDebug, "OpenHdf5File: Checking if file exists: '{}'", file_path);
  // Check if file exists
  struct stat st;
  if (stat(file_path.c_str(), &st) != 0) {
    HLOG(kError, "Hdf5FileAssimilator: File does not exist: '{}'", file_path);
    return -1;
  }
  HLOG(kDebug, "OpenHdf5File: File exists, size: {} bytes", st.st_size);

  // Open HDF5 file for reading (serial, read-only)
  HLOG(kDebug, "OpenHdf5File: Calling H5Fopen...");
  hid_t file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    HLOG(kError, "Hdf5FileAssimilator: H5Fopen failed for file '{}'",
         file_path);
    return -1;
  }
  HLOG(kDebug, "OpenHdf5File: H5Fopen succeeded, file_id: {}", file_id);

  return file_id;
}

void Hdf5FileAssimilator::CloseHdf5File(hid_t file_id) {
  HLOG(kDebug, "CloseHdf5File: Closing file_id: {}", file_id);
  if (file_id >= 0) {
    H5Fclose(file_id);
    HLOG(kDebug, "CloseHdf5File: H5Fclose completed");
  }
}

herr_t Hdf5FileAssimilator::VisitCallback(hid_t loc_id, const char* name,
                                          const H5L_info_t* info,
                                          void* operator_data) {
  // Cast operator data to vector of dataset paths
  auto* dataset_paths = static_cast<std::vector<std::string>*>(operator_data);

  // Check if this is a dataset
  H5O_info_t obj_info;
  // Handle HDF5 API version differences for H5Oget_info_by_name
#if H5_VERSION_GE(1, 12, 0)
  // HDF5 1.12+ API: includes fields parameter for selective info retrieval
  herr_t status =
      H5Oget_info_by_name(loc_id, name, &obj_info, H5O_INFO_BASIC, H5P_DEFAULT);
#else
  // HDF5 1.10 API: no fields parameter
  herr_t status = H5Oget_info_by_name(loc_id, name, &obj_info, H5P_DEFAULT);
#endif
  if (status < 0) {
    return 0;  // Continue iteration even on error
  }

  // If it's a dataset, add it to the list with leading slash
  // (to match h5dump/h5ls output and user expectations)
  if (obj_info.type == H5O_TYPE_DATASET) {
    dataset_paths->push_back("/" + std::string(name));
  }

  return 0;  // Continue iteration
}

int Hdf5FileAssimilator::DiscoverDatasets(
    hid_t file_id, std::vector<std::string>& dataset_paths) {
  HLOG(kDebug, "DiscoverDatasets: Starting dataset discovery for file_id: {}",
       file_id);
  dataset_paths.clear();

  // Use H5Lvisit to visit all links in the file recursively (including nested
  // groups) H5Lvisit traverses the entire HDF5 hierarchy, unlike H5Literate
  // which only visits root level
  HLOG(kDebug, "DiscoverDatasets: Calling H5Lvisit...");
  herr_t status = H5Lvisit(file_id, H5_INDEX_NAME, H5_ITER_NATIVE,
                           VisitCallback, &dataset_paths);

  if (status < 0) {
    HLOG(kError, "Hdf5FileAssimilator: H5Literate failed");
    return -1;
  }
  HLOG(kDebug, "DiscoverDatasets: H5Literate completed, found {} datasets",
       dataset_paths.size());

  return 0;
}

chi::TaskResume Hdf5FileAssimilator::ProcessDataset(
    hid_t file_id, const std::string& dataset_path,
    const std::string& tag_prefix, int& error_code) {
  HLOG(kDebug, "ProcessDataset: ENTRY - dataset: '{}', tag_prefix: '{}'",
       dataset_path, tag_prefix);

  // Open dataset
  HLOG(kDebug, "ProcessDataset: Opening dataset '{}'...", dataset_path);
  hid_t dataset_id = H5Dopen2(file_id, dataset_path.c_str(), H5P_DEFAULT);
  if (dataset_id < 0) {
    HLOG(kError, "Hdf5FileAssimilator: Failed to open dataset '{}'",
         dataset_path);
    error_code = -1;
    co_return;
  }
  HLOG(kDebug, "ProcessDataset: Dataset opened, dataset_id: {}", dataset_id);

  // Get dataspace and datatype
  HLOG(kDebug, "ProcessDataset: Getting dataspace...");
  hid_t dataspace_id = H5Dget_space(dataset_id);
  if (dataspace_id < 0) {
    HLOG(kError,
         "Hdf5FileAssimilator: Failed to get dataspace for dataset '{}'",
         dataset_path);
    H5Dclose(dataset_id);
    error_code = -2;
    co_return;
  }
  HLOG(kDebug, "ProcessDataset: Got dataspace, dataspace_id: {}", dataspace_id);

  HLOG(kDebug, "ProcessDataset: Getting datatype...");
  hid_t datatype_id = H5Dget_type(dataset_id);
  if (datatype_id < 0) {
    HLOG(kError, "Hdf5FileAssimilator: Failed to get datatype for dataset '{}'",
         dataset_path);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    error_code = -3;
    co_return;
  }
  HLOG(kDebug, "ProcessDataset: Got datatype, datatype_id: {}", datatype_id);

  // Get dimensions
  HLOG(kDebug, "ProcessDataset: Getting dimensions...");
  int rank = H5Sget_simple_extent_ndims(dataspace_id);
  if (rank < 0) {
    HLOG(kError, "Hdf5FileAssimilator: Failed to get rank for dataset '{}'",
         dataset_path);
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    error_code = -4;
    co_return;
  }
  HLOG(kDebug, "ProcessDataset: Rank: {}", rank);

  std::vector<hsize_t> dims(rank);
  H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);

  // Calculate total dataset size in bytes
  size_t type_size = H5Tget_size(datatype_id);
  size_t total_elements = 1;
  for (hsize_t dim : dims) {
    total_elements *= dim;
  }
  size_t total_bytes = total_elements * type_size;
  HLOG(kDebug,
       "ProcessDataset: Dataset info - rank: {}, total_elements: {}, "
       "type_size: {}, total_bytes: {}",
       rank, total_elements, type_size, total_bytes);

  // Create globally unique tag name: tag_prefix/dataset_path
  // Remove leading slash from dataset_path if present
  std::string dataset_path_clean = dataset_path;
  if (!dataset_path_clean.empty() && dataset_path_clean[0] == '/') {
    dataset_path_clean = dataset_path_clean.substr(1);
  }
  std::string tag_name = tag_prefix + "/" + dataset_path_clean;
  HLOG(kDebug, "ProcessDataset: Creating tag: '{}'", tag_name);

  // Get or create the tag in CTE
  HLOG(kDebug, "ProcessDataset: Calling GetOrCreateTag for '{}'...", tag_name);
  auto tag_task = cte_client_->AsyncGetOrCreateTag(tag_name);
  co_await tag_task;
  wrp_cte::core::TagId tag_id = tag_task->tag_id_;
  if (tag_id.IsNull()) {
    HLOG(kError, "Hdf5FileAssimilator: Failed to get or create tag '{}'",
         tag_name);
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    error_code = -5;
    co_return;
  }
  HLOG(kDebug, "ProcessDataset: Tag created/retrieved, tag_id: {}", tag_id);

  // Create and store tensor description as "description" blob
  std::string description = FormatTensorDescription(datatype_id, dims);
  HLOG(kDebug, "ProcessDataset: Tensor description: '{}'", description);
  size_t desc_size = description.size();
  auto desc_buffer = CHI_IPC->AllocateBuffer(desc_size);
  std::memcpy(desc_buffer.ptr_, description.c_str(), desc_size);

  HLOG(kDebug,
       "ProcessDataset: Submitting description blob (size: {} bytes)...",
       desc_size);
  auto desc_task =
      cte_client_->AsyncPutBlob(tag_id, "description", 0, desc_size,
                                desc_buffer.shm_.template Cast<void>(), 1.0f,
                                wrp_cte::core::Context(), 0);
  HLOG(kDebug, "ProcessDataset: Waiting for description blob task...");
  co_await desc_task;

  if (desc_task->return_code_ != 0) {
    HLOG(kError,
         "Hdf5FileAssimilator: Failed to store description for dataset '{}', "
         "return_code: {}",
         dataset_path, desc_task->return_code_);
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    error_code = -6;
    co_return;
  }
  HLOG(kDebug, "ProcessDataset: Description blob stored successfully");

  HLOG(kDebug, "Hdf5FileAssimilator: Stored description for '{}': {}", tag_name,
       description);

  // Define chunking parameters
  // Note: kMaxChunkSize must be < 2MB due to hermes_shm MultiProcessAllocator
  // thread_unit_ limit (2MB). Using 1.5MB to leave room for allocator overhead.
  static constexpr size_t kMaxChunkSize = 1536 * 1024;  // 1.5 MB
  static constexpr size_t kMaxParallelTasks = 32;

  // Calculate number of chunks
  size_t num_chunks = (total_bytes + kMaxChunkSize - 1) / kMaxChunkSize;
  HLOG(kDebug,
       "ProcessDataset: Starting data transfer - total_bytes: {}, num_chunks: "
       "{}, kMaxChunkSize: {}",
       total_bytes, num_chunks, kMaxChunkSize);

  // Allocate buffer for reading entire dataset using malloc (not IPC allocator)
  // This avoids the 2MB allocator limit for HDF5 read operations
  HLOG(kDebug,
       "ProcessDataset: Allocating read buffer of size: {} bytes (malloc)",
       total_bytes);
  char* dataset_buffer = static_cast<char*>(malloc(total_bytes));
  if (!dataset_buffer) {
    HLOG(kError,
         "Hdf5FileAssimilator: Failed to allocate {} bytes for dataset '{}'",
         total_bytes, dataset_path);
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    error_code = -7;
    co_return;
  }

  // Read the entire dataset in one H5Dread call
  HLOG(kDebug, "ProcessDataset: Reading entire dataset with H5Dread...");
  herr_t read_status = H5Dread(dataset_id, datatype_id, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, dataset_buffer);
  if (read_status < 0) {
    HLOG(kError, "Hdf5FileAssimilator: H5Dread failed for dataset '{}'",
         dataset_path);
    free(dataset_buffer);
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    error_code = -8;
    co_return;
  }
  HLOG(kDebug, "ProcessDataset: Dataset read successfully ({} bytes)",
       total_bytes);

  // Process data in chunks and send to CTE
  size_t chunk_idx = 0;
  size_t bytes_processed = 0;
  std::vector<chi::Future<wrp_cte::core::PutBlobTask>> active_tasks;

  HLOG(kDebug, "ProcessDataset: Starting chunk transfer loop...");
  while (bytes_processed < total_bytes) {
    HLOG(kDebug,
         "ProcessDataset: Loop iteration - bytes_processed: {}/{}, "
         "active_tasks: {}",
         bytes_processed, total_bytes, active_tasks.size());

    // Submit tasks up to the parallel limit
    while (active_tasks.size() < kMaxParallelTasks &&
           bytes_processed < total_bytes) {
      size_t current_chunk_size =
          std::min(kMaxChunkSize, total_bytes - bytes_processed);

      // Create blob name with chunk index
      std::string blob_name = "chunk_" + std::to_string(chunk_idx);
      HLOG(kDebug,
           "ProcessDataset: Creating blob '{}' with size: {} bytes (chunk "
           "{}/{})",
           blob_name, current_chunk_size, chunk_idx + 1, num_chunks);

      // Allocate IPC buffer for this chunk and copy data
      auto chunk_buffer = CHI_IPC->AllocateBuffer(current_chunk_size);
      std::memcpy(chunk_buffer.ptr_, dataset_buffer + bytes_processed,
                  current_chunk_size);

      // Submit PutBlob task asynchronously
      auto task =
          cte_client_->AsyncPutBlob(tag_id, blob_name, 0, current_chunk_size,
                                    chunk_buffer.shm_.template Cast<void>(),
                                    1.0f, wrp_cte::core::Context(), 0);

      active_tasks.push_back(task);

      bytes_processed += current_chunk_size;
      chunk_idx++;
    }

    // Wait for at least one task to complete before continuing
    if (!active_tasks.empty()) {
      auto& first_task = active_tasks.front();
      co_await first_task;

      if (first_task->return_code_ != 0) {
        HLOG(kError, "Hdf5FileAssimilator: PutBlob task failed with code {}",
             first_task->return_code_);
        CHI_IPC->FreeBuffer(first_task->blob_data_.template Cast<char>());
        free(dataset_buffer);
        H5Tclose(datatype_id);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        error_code = -9;
        co_return;
      }

      CHI_IPC->FreeBuffer(first_task->blob_data_.template Cast<char>());
      active_tasks.erase(active_tasks.begin());
    }
  }

  // Wait for all remaining tasks to complete
  HLOG(kDebug, "ProcessDataset: Waiting for {} remaining tasks to complete...",
       active_tasks.size());
  for (auto& task : active_tasks) {
    co_await task;
    if (task->return_code_ != 0) {
      HLOG(kError, "Hdf5FileAssimilator: PutBlob task failed with code {}",
           task->return_code_);
      CHI_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
      free(dataset_buffer);
      H5Tclose(datatype_id);
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
      error_code = -10;
      co_return;
    }
    CHI_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
  }

  HLOG(kDebug, "ProcessDataset: All tasks completed, cleaning up resources...");
  free(dataset_buffer);
  HLOG(kDebug, "ProcessDataset: Buffer freed");
  H5Tclose(datatype_id);
  HLOG(kDebug, "ProcessDataset: Datatype closed");
  H5Sclose(dataspace_id);
  HLOG(kDebug, "ProcessDataset: Dataspace closed");
  H5Dclose(dataset_id);
  HLOG(kDebug, "ProcessDataset: Dataset closed");

  HLOG(kDebug,
       "Hdf5FileAssimilator: Successfully transferred {} chunk(s) ({} bytes) "
       "for dataset '{}'",
       num_chunks, total_bytes, tag_name);
  HLOG(kDebug, "ProcessDataset: EXIT - success");

  error_code = 0;
  co_return;
}

std::string Hdf5FileAssimilator::GetTypeName(hid_t datatype) {
  H5T_class_t type_class = H5Tget_class(datatype);
  size_t type_size = H5Tget_size(datatype);

  switch (type_class) {
    case H5T_INTEGER:
      if (type_size == 4) {
        return "int32";
      } else if (type_size == 8) {
        return "int64";
      }
      return "unknown";

    case H5T_FLOAT:
      if (type_size == 4) {
        return "float32";
      } else if (type_size == 8) {
        return "float64";
      }
      return "unknown";

    default:
      return "unknown";
  }
}

std::string Hdf5FileAssimilator::FormatTensorDescription(
    hid_t datatype, const std::vector<hsize_t>& dims) {
  std::string type_name = GetTypeName(datatype);
  std::string description = "tensor<" + type_name;

  for (hsize_t dim : dims) {
    description += ", " + std::to_string(dim);
  }

  description += ">";
  return description;
}

std::string Hdf5FileAssimilator::GetUrlProtocol(const std::string& url) {
  size_t pos = url.find("::");
  if (pos == std::string::npos) {
    return "";
  }
  return url.substr(0, pos);
}

std::string Hdf5FileAssimilator::GetUrlPath(const std::string& url) {
  size_t pos = url.find("::");
  if (pos == std::string::npos) {
    return "";
  }
  return url.substr(pos + 2);
}

bool Hdf5FileAssimilator::MatchGlobPattern(const std::string& str,
                                           const std::string& pattern) {
  // Use fnmatch for glob pattern matching (supports *, ?, [])
  // FNM_PATHNAME: '/' must be matched explicitly
  // FNM_PERIOD: leading '.' must be matched explicitly
  int result = fnmatch(pattern.c_str(), str.c_str(), 0);
  return result == 0;
}

bool Hdf5FileAssimilator::MatchesFilter(
    const std::string& dataset_path,
    const std::vector<std::string>& include_patterns,
    const std::vector<std::string>& exclude_patterns) {
  // First check exclude patterns - if any match, exclude the dataset
  for (const auto& exclude_pattern : exclude_patterns) {
    if (MatchGlobPattern(dataset_path, exclude_pattern)) {
      HLOG(kDebug, "Dataset '{}' excluded by pattern '{}'", dataset_path,
           exclude_pattern);
      return false;
    }
  }

  // If no include patterns specified, include all (that weren't excluded)
  if (include_patterns.empty()) {
    return true;
  }

  // Check if dataset matches any include pattern
  for (const auto& include_pattern : include_patterns) {
    bool matches = MatchGlobPattern(dataset_path, include_pattern);
    HLOG(kDebug, "Checking dataset '{}' against pattern '{}': {}", dataset_path,
         include_pattern, matches ? "MATCH" : "NO MATCH");
    if (matches) {
      HLOG(kDebug, "Dataset '{}' included by pattern '{}'", dataset_path,
           include_pattern);
      return true;
    }
  }

  // No include pattern matched, so exclude the dataset
  HLOG(kDebug, "Dataset '{}' does not match any include pattern", dataset_path);
  return false;
}

}  // namespace wrp_cae::core
