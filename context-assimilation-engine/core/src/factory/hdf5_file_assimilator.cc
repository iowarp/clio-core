#include <wrp_cae/core/factory/hdf5_file_assimilator.h>
#include <chimaera/chimaera.h>
#include <sys/stat.h>
#include <algorithm>
#include <vector>
#include <cstring>

// Include wrp_cte headers after closing any wrp_cae namespace to avoid Method namespace collision
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

namespace wrp_cae::core {

Hdf5FileAssimilator::Hdf5FileAssimilator(std::shared_ptr<wrp_cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

int Hdf5FileAssimilator::Schedule(const AssimilationCtx& ctx) {
  HILOG(kInfo, "Hdf5FileAssimilator::Schedule() - ENTRY");
  HILOG(kInfo, "  ctx.src: '{}'", ctx.src);
  HILOG(kInfo, "  ctx.dst: '{}'", ctx.dst);
  HILOG(kInfo, "  ctx.format: '{}'", ctx.format);

  // Validate destination protocol
  std::string dst_protocol = GetUrlProtocol(ctx.dst);
  HILOG(kInfo, "Hdf5FileAssimilator: Extracted destination protocol: '{}'", dst_protocol);
  if (dst_protocol != "iowarp") {
    HELOG(kError, "Hdf5FileAssimilator: Destination protocol must be 'iowarp', got '{}'",
          dst_protocol);
    return -1;
  }

  // Extract tag prefix from destination URL (remove iowarp:: prefix)
  std::string tag_prefix = GetUrlPath(ctx.dst);
  HILOG(kInfo, "Hdf5FileAssimilator: Extracted tag prefix: '{}'", tag_prefix);
  if (tag_prefix.empty()) {
    HELOG(kError, "Hdf5FileAssimilator: Invalid destination URL, no tag name found");
    return -2;
  }

  // Handle dependency-based scheduling
  if (!ctx.depends_on.empty()) {
    // TODO: Implement dependency handling
    // For now, log that dependencies are not yet supported
    HILOG(kInfo, "Hdf5FileAssimilator: Dependency handling not yet implemented (depends_on: {})",
          ctx.depends_on);
    return 0;
  }

  // Extract source file path
  std::string src_path = GetUrlPath(ctx.src);
  HILOG(kInfo, "Hdf5FileAssimilator: Extracted source file path: '{}'", src_path);
  if (src_path.empty()) {
    HELOG(kError, "Hdf5FileAssimilator: Invalid source URL, no file path found");
    return -3;
  }

  // Open HDF5 file
  HILOG(kInfo, "Hdf5FileAssimilator: Opening HDF5 file...");
  hid_t file_id = OpenHdf5File(src_path);
  if (file_id < 0) {
    HELOG(kError, "Hdf5FileAssimilator: Failed to open HDF5 file '{}'", src_path);
    return -4;
  }
  HILOG(kInfo, "Hdf5FileAssimilator: HDF5 file opened successfully (file_id: {})", file_id);

  // Discover all datasets in the file
  HILOG(kInfo, "Hdf5FileAssimilator: Discovering datasets...");
  std::vector<std::string> dataset_paths;
  int discover_result = DiscoverDatasets(file_id, dataset_paths);
  if (discover_result != 0) {
    HELOG(kError, "Hdf5FileAssimilator: Failed to discover datasets in '{}'", src_path);
    CloseHdf5File(file_id);
    return -5;
  }

  HILOG(kInfo, "Hdf5FileAssimilator: Discovered {} dataset(s) in '{}'",
        dataset_paths.size(), src_path);

  // Process each dataset
  int total_errors = 0;
  for (size_t i = 0; i < dataset_paths.size(); ++i) {
    const auto& dataset_path = dataset_paths[i];
    HILOG(kInfo, "Hdf5FileAssimilator: Processing dataset {}/{}: '{}'",
          i + 1, dataset_paths.size(), dataset_path);
    int result = ProcessDataset(file_id, dataset_path, tag_prefix);
    if (result != 0) {
      HELOG(kError, "Hdf5FileAssimilator: Failed to process dataset '{}' (error code: {})",
            dataset_path, result);
      total_errors++;
    } else {
      HILOG(kInfo, "Hdf5FileAssimilator: Successfully processed dataset '{}'", dataset_path);
    }
  }

  HILOG(kInfo, "Hdf5FileAssimilator: Closing HDF5 file...");
  CloseHdf5File(file_id);
  HILOG(kInfo, "Hdf5FileAssimilator: HDF5 file closed");

  if (total_errors > 0) {
    HELOG(kError, "Hdf5FileAssimilator: Completed with {} error(s) out of {} dataset(s)",
          total_errors, dataset_paths.size());
    return -6;
  }

  HILOG(kInfo, "Hdf5FileAssimilator: Successfully processed all {} dataset(s) from '{}'",
        dataset_paths.size(), src_path);
  HILOG(kInfo, "Hdf5FileAssimilator::Schedule() - EXIT (success)");

  return 0;
}

hid_t Hdf5FileAssimilator::OpenHdf5File(const std::string& file_path) {
  HILOG(kInfo, "OpenHdf5File: Checking if file exists: '{}'", file_path);
  // Check if file exists
  struct stat st;
  if (stat(file_path.c_str(), &st) != 0) {
    HELOG(kError, "Hdf5FileAssimilator: File does not exist: '{}'", file_path);
    return -1;
  }
  HILOG(kInfo, "OpenHdf5File: File exists, size: {} bytes", st.st_size);

  // Open HDF5 file for reading (serial, read-only)
  HILOG(kInfo, "OpenHdf5File: Calling H5Fopen...");
  hid_t file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    HELOG(kError, "Hdf5FileAssimilator: H5Fopen failed for file '{}'", file_path);
    return -1;
  }
  HILOG(kInfo, "OpenHdf5File: H5Fopen succeeded, file_id: {}", file_id);

  return file_id;
}

void Hdf5FileAssimilator::CloseHdf5File(hid_t file_id) {
  HILOG(kInfo, "CloseHdf5File: Closing file_id: {}", file_id);
  if (file_id >= 0) {
    H5Fclose(file_id);
    HILOG(kInfo, "CloseHdf5File: H5Fclose completed");
  }
}

herr_t Hdf5FileAssimilator::VisitCallback(hid_t loc_id, const char* name,
                                          const H5L_info_t* info, void* operator_data) {
  // Cast operator data to vector of dataset paths
  auto* dataset_paths = static_cast<std::vector<std::string>*>(operator_data);

  // Check if this is a dataset
  H5O_info_t obj_info;
  // Handle HDF5 API version differences for H5Oget_info_by_name
#if H5_VERSION_GE(1, 12, 0)
  // HDF5 1.12+ API: includes fields parameter for selective info retrieval
  herr_t status = H5Oget_info_by_name(loc_id, name, &obj_info, H5O_INFO_BASIC, H5P_DEFAULT);
#else
  // HDF5 1.10 API: no fields parameter
  herr_t status = H5Oget_info_by_name(loc_id, name, &obj_info, H5P_DEFAULT);
#endif
  if (status < 0) {
    return 0;  // Continue iteration even on error
  }

  // If it's a dataset, add it to the list
  if (obj_info.type == H5O_TYPE_DATASET) {
    dataset_paths->push_back(std::string(name));
  }

  return 0;  // Continue iteration
}

int Hdf5FileAssimilator::DiscoverDatasets(hid_t file_id,
                                          std::vector<std::string>& dataset_paths) {
  HILOG(kInfo, "DiscoverDatasets: Starting dataset discovery for file_id: {}", file_id);
  dataset_paths.clear();

  // Use H5Literate to visit all links in the file recursively
  // Starting from root group, index 0, with VisitCallback
  hsize_t idx = 0;
  HILOG(kInfo, "DiscoverDatasets: Calling H5Literate...");
  herr_t status = H5Literate(file_id, H5_INDEX_NAME, H5_ITER_NATIVE,
                             &idx, VisitCallback, &dataset_paths);

  if (status < 0) {
    HELOG(kError, "Hdf5FileAssimilator: H5Literate failed");
    return -1;
  }
  HILOG(kInfo, "DiscoverDatasets: H5Literate completed, found {} datasets", dataset_paths.size());

  return 0;
}

int Hdf5FileAssimilator::ProcessDataset(hid_t file_id,
                                        const std::string& dataset_path,
                                        const std::string& tag_prefix) {
  HILOG(kInfo, "ProcessDataset: ENTRY - dataset: '{}', tag_prefix: '{}'",
        dataset_path, tag_prefix);

  // Open dataset
  HILOG(kInfo, "ProcessDataset: Opening dataset '{}'...", dataset_path);
  hid_t dataset_id = H5Dopen2(file_id, dataset_path.c_str(), H5P_DEFAULT);
  if (dataset_id < 0) {
    HELOG(kError, "Hdf5FileAssimilator: Failed to open dataset '{}'", dataset_path);
    return -1;
  }
  HILOG(kInfo, "ProcessDataset: Dataset opened, dataset_id: {}", dataset_id);

  // Get dataspace and datatype
  HILOG(kInfo, "ProcessDataset: Getting dataspace...");
  hid_t dataspace_id = H5Dget_space(dataset_id);
  if (dataspace_id < 0) {
    HELOG(kError, "Hdf5FileAssimilator: Failed to get dataspace for dataset '{}'",
          dataset_path);
    H5Dclose(dataset_id);
    return -2;
  }
  HILOG(kInfo, "ProcessDataset: Got dataspace, dataspace_id: {}", dataspace_id);

  HILOG(kInfo, "ProcessDataset: Getting datatype...");
  hid_t datatype_id = H5Dget_type(dataset_id);
  if (datatype_id < 0) {
    HELOG(kError, "Hdf5FileAssimilator: Failed to get datatype for dataset '{}'",
          dataset_path);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    return -3;
  }
  HILOG(kInfo, "ProcessDataset: Got datatype, datatype_id: {}", datatype_id);

  // Get dimensions
  HILOG(kInfo, "ProcessDataset: Getting dimensions...");
  int rank = H5Sget_simple_extent_ndims(dataspace_id);
  if (rank < 0) {
    HELOG(kError, "Hdf5FileAssimilator: Failed to get rank for dataset '{}'", dataset_path);
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    return -4;
  }
  HILOG(kInfo, "ProcessDataset: Rank: {}", rank);

  std::vector<hsize_t> dims(rank);
  H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);

  // Calculate total dataset size in bytes
  size_t type_size = H5Tget_size(datatype_id);
  size_t total_elements = 1;
  for (hsize_t dim : dims) {
    total_elements *= dim;
  }
  size_t total_bytes = total_elements * type_size;
  HILOG(kInfo, "ProcessDataset: Dataset info - rank: {}, total_elements: {}, type_size: {}, total_bytes: {}",
        rank, total_elements, type_size, total_bytes);

  // Create globally unique tag name: tag_prefix/dataset_path
  // Remove leading slash from dataset_path if present
  std::string dataset_path_clean = dataset_path;
  if (!dataset_path_clean.empty() && dataset_path_clean[0] == '/') {
    dataset_path_clean = dataset_path_clean.substr(1);
  }
  std::string tag_name = tag_prefix + "/" + dataset_path_clean;
  HILOG(kInfo, "ProcessDataset: Creating tag: '{}'", tag_name);

  // Get or create the tag in CTE
  hipc::MemContext mctx;
  HILOG(kInfo, "ProcessDataset: Calling GetOrCreateTag for '{}'...", tag_name);
  wrp_cte::core::TagId tag_id = cte_client_->GetOrCreateTag(mctx, tag_name);
  if (tag_id.IsNull()) {
    HELOG(kError, "Hdf5FileAssimilator: Failed to get or create tag '{}'", tag_name);
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    return -5;
  }
  HILOG(kInfo, "ProcessDataset: Tag created/retrieved, tag_id: {}", tag_id);

  // Create and store tensor description as "description" blob
  std::string description = FormatTensorDescription(datatype_id, dims);
  HILOG(kInfo, "ProcessDataset: Tensor description: '{}'", description);
  size_t desc_size = description.size();
  auto desc_buffer = CHI_IPC->AllocateBuffer(desc_size);
  std::memcpy(desc_buffer.ptr_, description.c_str(), desc_size);

  HILOG(kInfo, "ProcessDataset: Submitting description blob (size: {} bytes)...", desc_size);
  auto desc_task = cte_client_->AsyncPutBlob(
      mctx, tag_id, "description", 0, desc_size, desc_buffer.shm_, 1.0f, 0);
  HILOG(kInfo, "ProcessDataset: Waiting for description blob task...");
  desc_task->Wait();

  if (desc_task->return_code_.load() != 0) {
    HELOG(kError, "Hdf5FileAssimilator: Failed to store description for dataset '{}', return_code: {}",
          dataset_path, desc_task->return_code_.load());
    CHI_IPC->DelTask(desc_task);
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    return -6;
  }
  CHI_IPC->DelTask(desc_task);
  HILOG(kInfo, "ProcessDataset: Description blob stored successfully");

  HILOG(kInfo, "Hdf5FileAssimilator: Stored description for '{}': {}", tag_name, description);

  // Define chunking parameters
  static constexpr size_t kMaxChunkSize = 1024 * 1024;  // 1 MB
  static constexpr size_t kMaxParallelTasks = 32;

  // Calculate number of chunks
  size_t num_chunks = (total_bytes + kMaxChunkSize - 1) / kMaxChunkSize;
  HILOG(kInfo, "ProcessDataset: Starting data transfer - total_bytes: {}, num_chunks: {}, kMaxChunkSize: {}",
        total_bytes, num_chunks, kMaxChunkSize);

  // Allocate buffer for reading chunks
  size_t buffer_size = std::min(kMaxChunkSize, total_bytes);
  HILOG(kInfo, "ProcessDataset: Allocating read buffer of size: {} bytes", buffer_size);
  auto read_buffer = CHI_IPC->AllocateBuffer(buffer_size);
  char* buffer = read_buffer.ptr_;

  // Process chunks in batches
  size_t chunk_idx = 0;
  size_t bytes_processed = 0;
  std::vector<hipc::FullPtr<wrp_cte::core::PutBlobTask>> active_tasks;

  HILOG(kInfo, "ProcessDataset: Starting chunk processing loop...");
  while (bytes_processed < total_bytes) {
    HILOG(kInfo, "ProcessDataset: Loop iteration - bytes_processed: {}/{}, active_tasks: {}",
          bytes_processed, total_bytes, active_tasks.size());
    // Submit tasks up to the parallel limit
    while (active_tasks.size() < kMaxParallelTasks && bytes_processed < total_bytes) {
      HILOG(kInfo, "ProcessDataset: Preparing chunk {} (bytes_processed: {}/{})",
            chunk_idx, bytes_processed, total_bytes);
      // Calculate chunk size
      size_t current_chunk_size = std::min(kMaxChunkSize, total_bytes - bytes_processed);

      // Calculate hyperslab parameters for this chunk
      // We read the dataset linearly, treating it as a flat array
      size_t element_offset = bytes_processed / type_size;
      size_t elements_to_read = current_chunk_size / type_size;

      // For simplicity, we'll read the entire dataset into memory for small datasets
      // For large datasets, we use hyperslab selection
      if (total_bytes <= kMaxChunkSize) {
        // Small dataset - read entire dataset in one go
        HILOG(kInfo, "ProcessDataset: Reading small dataset (entire dataset in one go)...");
        herr_t read_status = H5Dread(dataset_id, datatype_id, H5S_ALL, H5S_ALL,
                                     H5P_DEFAULT, buffer);
        if (read_status < 0) {
          HELOG(kError, "Hdf5FileAssimilator: Failed to read dataset '{}'", dataset_path);
          CHI_IPC->FreeBuffer(read_buffer);
          H5Tclose(datatype_id);
          H5Sclose(dataspace_id);
          H5Dclose(dataset_id);
          return -7;
        }
        HILOG(kInfo, "ProcessDataset: Small dataset read successfully");
      } else {
        // Large dataset - use hyperslab selection
        // Create memory dataspace for this chunk
        hsize_t mem_dims[1] = {elements_to_read};
        hid_t memspace = H5Screate_simple(1, mem_dims, nullptr);
        if (memspace < 0) {
          HELOG(kError, "Hdf5FileAssimilator: Failed to create memory space for chunk {}",
                chunk_idx);
          CHI_IPC->FreeBuffer(read_buffer);
          H5Tclose(datatype_id);
          H5Sclose(dataspace_id);
          H5Dclose(dataset_id);
          return -8;
        }

        // Select hyperslab in file dataspace
        // Flatten the dataset to 1D for easier chunking
        hsize_t start[1] = {element_offset};
        hsize_t count[1] = {elements_to_read};
        hsize_t stride[1] = {1};
        hsize_t block[1] = {1};

        // Create a flattened dataspace
        hsize_t total_elements_hsize = total_elements;
        hid_t flat_space = H5Screate_simple(1, &total_elements_hsize, nullptr);
        if (flat_space < 0) {
          HELOG(kError, "Hdf5FileAssimilator: Failed to create flat space for chunk {}",
                chunk_idx);
          H5Sclose(memspace);
          CHI_IPC->FreeBuffer(read_buffer);
          H5Tclose(datatype_id);
          H5Sclose(dataspace_id);
          H5Dclose(dataset_id);
          return -9;
        }

        herr_t select_status = H5Sselect_hyperslab(flat_space, H5S_SELECT_SET,
                                                   start, stride, count, block);
        if (select_status < 0) {
          HELOG(kError, "Hdf5FileAssimilator: Failed to select hyperslab for chunk {}",
                chunk_idx);
          H5Sclose(flat_space);
          H5Sclose(memspace);
          CHI_IPC->FreeBuffer(read_buffer);
          H5Tclose(datatype_id);
          H5Sclose(dataspace_id);
          H5Dclose(dataset_id);
          return -10;
        }

        // Read the hyperslab
        herr_t read_status = H5Dread(dataset_id, datatype_id, memspace,
                                    flat_space, H5P_DEFAULT, buffer);
        H5Sclose(flat_space);
        H5Sclose(memspace);

        if (read_status < 0) {
          HELOG(kError, "Hdf5FileAssimilator: Failed to read chunk {} from dataset '{}'",
                chunk_idx, dataset_path);
          CHI_IPC->FreeBuffer(read_buffer);
          H5Tclose(datatype_id);
          H5Sclose(dataspace_id);
          H5Dclose(dataset_id);
          return -11;
        }
      }

      // Create blob name with chunk index
      std::string blob_name = "chunk_" + std::to_string(chunk_idx);
      HILOG(kInfo, "ProcessDataset: Creating blob '{}' with size: {} bytes", blob_name, current_chunk_size);

      // Allocate a new buffer for this chunk (since we need to keep reading)
      auto chunk_buffer = CHI_IPC->AllocateBuffer(current_chunk_size);
      std::memcpy(chunk_buffer.ptr_, buffer, current_chunk_size);

      // Submit PutBlob task asynchronously
      HILOG(kInfo, "ProcessDataset: Submitting AsyncPutBlob for chunk {}...", chunk_idx);
      auto task = cte_client_->AsyncPutBlob(
          mctx, tag_id, blob_name, 0, current_chunk_size,
          chunk_buffer.shm_, 1.0f, 0);

      active_tasks.push_back(task);
      HILOG(kInfo, "ProcessDataset: Task submitted for chunk {}, active_tasks count: {}",
            chunk_idx, active_tasks.size());

      bytes_processed += current_chunk_size;
      chunk_idx++;

      // For small datasets, we're done after one chunk
      if (total_bytes <= kMaxChunkSize) {
        HILOG(kInfo, "ProcessDataset: Small dataset - exiting submission loop");
        break;
      }
    }

    // Wait for at least one task to complete before continuing
    if (!active_tasks.empty()) {
      HILOG(kInfo, "ProcessDataset: Waiting for first task to complete...");
      auto& first_task = active_tasks.front();
      first_task->Wait();

      if (first_task->return_code_.load() != 0) {
        HELOG(kError, "Hdf5FileAssimilator: PutBlob task failed with code {}",
              first_task->return_code_.load());
        // Free the buffer before deleting the task
        CHI_IPC->FreeBuffer(first_task->blob_data_);
        CHI_IPC->DelTask(first_task);
        CHI_IPC->FreeBuffer(read_buffer);
        H5Tclose(datatype_id);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        return -12;
      }

      HILOG(kInfo, "ProcessDataset: First task completed successfully");
      // Free the buffer before deleting the task
      CHI_IPC->FreeBuffer(first_task->blob_data_);
      CHI_IPC->DelTask(first_task);
      active_tasks.erase(active_tasks.begin());
    }
  }

  // Wait for all remaining tasks to complete
  HILOG(kInfo, "ProcessDataset: Waiting for {} remaining tasks to complete...",
        active_tasks.size());
  for (auto& task : active_tasks) {
    task->Wait();
    if (task->return_code_.load() != 0) {
      HELOG(kError, "Hdf5FileAssimilator: PutBlob task failed with code {}",
            task->return_code_.load());
      // Free the buffer before deleting the task
      CHI_IPC->FreeBuffer(task->blob_data_);
      CHI_IPC->DelTask(task);
      CHI_IPC->FreeBuffer(read_buffer);
      H5Tclose(datatype_id);
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
      return -12;
    }
    // Free the buffer before deleting the task
    CHI_IPC->FreeBuffer(task->blob_data_);
    CHI_IPC->DelTask(task);
  }

  HILOG(kInfo, "ProcessDataset: All tasks completed, cleaning up resources...");
  CHI_IPC->FreeBuffer(read_buffer);
  HILOG(kInfo, "ProcessDataset: Buffer freed");
  H5Tclose(datatype_id);
  HILOG(kInfo, "ProcessDataset: Datatype closed");
  H5Sclose(dataspace_id);
  HILOG(kInfo, "ProcessDataset: Dataspace closed");
  H5Dclose(dataset_id);
  HILOG(kInfo, "ProcessDataset: Dataset closed");

  HILOG(kInfo, "Hdf5FileAssimilator: Successfully transferred {} chunk(s) ({} bytes) for dataset '{}'",
        num_chunks, total_bytes, tag_name);
  HILOG(kInfo, "ProcessDataset: EXIT - success");

  return 0;
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

}  // namespace wrp_cae::core
