#include <wrp_cae/core/factory/binary_file_assimilator.h>
#include <chimaera/chimaera.h>
#include <sys/stat.h>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cstring>

// Include wrp_cte headers after closing any wrp_cae namespace to avoid Method namespace collision
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

namespace wrp_cae::core {

BinaryFileAssimilator::BinaryFileAssimilator(std::shared_ptr<wrp_cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

int BinaryFileAssimilator::Schedule(const AssimilationCtx& ctx) {
  HILOG(kInfo, "BinaryFileAssimilator::Schedule ENTRY: src='{}', dst='{}', range_off={}, range_size={}",
        ctx.src, ctx.dst, ctx.range_off, ctx.range_size);

  // Validate destination protocol
  std::string dst_protocol = GetUrlProtocol(ctx.dst);
  HILOG(kInfo, "BinaryFileAssimilator: Extracted dst_protocol='{}'", dst_protocol);

  if (dst_protocol != "iowarp") {
    HELOG(kError, "BinaryFileAssimilator: Destination protocol must be 'iowarp', got '{}'",
          dst_protocol);
    return -1;
  }

  // Extract tag name from destination URL
  std::string tag_name = GetUrlPath(ctx.dst);
  HILOG(kInfo, "BinaryFileAssimilator: Extracted tag_name='{}'", tag_name);

  if (tag_name.empty()) {
    HELOG(kError, "BinaryFileAssimilator: Invalid destination URL, no tag name found");
    return -2;
  }

  // Get or create the tag in CTE
  HILOG(kInfo, "BinaryFileAssimilator: Getting or creating tag '{}'", tag_name);
    wrp_cte::core::TagId tag_id = cte_client_->GetOrCreateTag(tag_name);
  if (tag_id.IsNull()) {
    HELOG(kError, "BinaryFileAssimilator: Failed to get or create tag '{}'", tag_name);
    return -3;
  }
  HILOG(kInfo, "BinaryFileAssimilator: Tag '{}' obtained/created successfully", tag_name);

  // Handle dependency-based scheduling
  if (!ctx.depends_on.empty()) {
    // TODO: Implement dependency handling
    // For now, log that dependencies are not yet supported
    HILOG(kInfo, "BinaryFileAssimilator: Dependency handling not yet implemented (depends_on: {})",
          ctx.depends_on);
    return 0;
  }

  // Extract source file path
  std::string src_path = GetUrlPath(ctx.src);
  HILOG(kInfo, "BinaryFileAssimilator: Extracted src_path='{}'", src_path);

  if (src_path.empty()) {
    HELOG(kError, "BinaryFileAssimilator: Invalid source URL, no file path found");
    return -4;
  }

  // Determine file size and chunk parameters
  size_t file_size;
  size_t chunk_offset;
  size_t total_size;

  if (ctx.range_size > 0) {
    HILOG(kInfo, "BinaryFileAssimilator: Using range mode - offset={}, size={}",
          ctx.range_off, ctx.range_size);
    // Use the specified range
    chunk_offset = ctx.range_off;
    total_size = ctx.range_size;
    file_size = GetFileSize(src_path);
    if (file_size == 0) {
      HELOG(kError, "BinaryFileAssimilator: Failed to get file size for '{}'", src_path);
      return -5;
    }
    HILOG(kInfo, "BinaryFileAssimilator: File size={} bytes", file_size);
    // Validate range
    if (chunk_offset + total_size > file_size) {
      HELOG(kError, "BinaryFileAssimilator: Range exceeds file size (offset: {}, size: {}, file_size: {})",
            chunk_offset, total_size, file_size);
      return -6;
    }
  } else {
    HILOG(kInfo, "BinaryFileAssimilator: Using full file mode");
    // Use entire file
    file_size = GetFileSize(src_path);
    if (file_size == 0) {
      HELOG(kError, "BinaryFileAssimilator: Failed to get file size for '{}'", src_path);
      return -5;
    }
    HILOG(kInfo, "BinaryFileAssimilator: File size={} bytes", file_size);
    chunk_offset = 0;
    total_size = file_size;
  }

  // Store file metadata as "description" blob
  std::string description = "binary<size=" + std::to_string(total_size) +
                           ", offset=" + std::to_string(chunk_offset) + ">";
  size_t desc_size = description.size();
  auto desc_buffer = CHI_IPC->AllocateBuffer(desc_size);
  std::memcpy(desc_buffer.ptr_, description.c_str(), desc_size);

  HILOG(kInfo, "BinaryFileAssimilator: Storing description blob: '{}'", description);
  auto desc_task = cte_client_->AsyncPutBlob(
      tag_id, "description", 0, desc_size, desc_buffer.shm_.template Cast<void>(), 1.0f, 0);
  desc_task->Wait();

  if (desc_task->return_code_.load() != 0) {
    HELOG(kError, "BinaryFileAssimilator: Failed to store description for tag '{}', return_code: {}",
          tag_name, desc_task->return_code_.load());
    CHI_IPC->DelTask(desc_task);
    return -9;
  }
  CHI_IPC->DelTask(desc_task);
  HILOG(kInfo, "BinaryFileAssimilator: Description blob stored successfully");

  // Define chunking parameters
  static constexpr size_t kMaxChunkSize = 1024 * 1024;  // 1 MB
  static constexpr size_t kMaxParallelTasks = 32;

  // Calculate number of chunks
  size_t num_chunks = (total_size + kMaxChunkSize - 1) / kMaxChunkSize;
  HILOG(kInfo, "BinaryFileAssimilator: Will transfer {} bytes in {} chunks (max {} parallel)",
        total_size, num_chunks, kMaxParallelTasks);

  // Open file for reading
  HILOG(kInfo, "BinaryFileAssimilator: Opening file '{}'", src_path);
  std::ifstream file(src_path, std::ios::binary);
  if (!file.is_open()) {
    HELOG(kError, "BinaryFileAssimilator: Failed to open file '{}'", src_path);
    return -7;
  }

  // Seek to the starting offset
  HILOG(kInfo, "BinaryFileAssimilator: Seeking to offset {}", chunk_offset);
  file.seekg(chunk_offset, std::ios::beg);
  if (!file.good()) {
    HELOG(kError, "BinaryFileAssimilator: Failed to seek to offset {} in file '{}'",
          chunk_offset, src_path);
    return -8;
  }

  // Process chunks in batches
  HILOG(kInfo, "BinaryFileAssimilator: Starting chunk processing");
  size_t chunk_idx = 0;
  size_t bytes_processed = 0;
  std::vector<hipc::FullPtr<wrp_cte::core::PutBlobTask>> active_tasks;

  while (bytes_processed < total_size) {
    // Submit tasks up to the parallel limit
    while (active_tasks.size() < kMaxParallelTasks && bytes_processed < total_size) {
      // Calculate chunk size
      size_t current_chunk_size = std::min(kMaxChunkSize, total_size - bytes_processed);

      // Allocate buffer in shared memory
      auto buffer_ptr = CHI_IPC->AllocateBuffer(current_chunk_size);
      char* buffer = buffer_ptr.ptr_;

      // Read chunk from file
      HILOG(kInfo, "BinaryFileAssimilator: About to read chunk {} ({} bytes) at file position {}",
            chunk_idx, current_chunk_size, file.tellg());

      file.read(buffer, current_chunk_size);
      std::streamsize bytes_read = file.gcount();

      HILOG(kInfo, "BinaryFileAssimilator: After read - bytes_read={}, eof={}, fail={}, bad={}, good={}",
            bytes_read, file.eof(), file.fail(), file.bad(), file.good());

      if (bytes_read != static_cast<std::streamsize>(current_chunk_size)) {
        // Check if this is a legitimate short read at end of file
        if (file.eof() && bytes_read > 0) {
          // This is OK - we read partial data at the end
          HILOG(kInfo, "BinaryFileAssimilator: Chunk {} partial read: {} bytes (expected {})",
                chunk_idx, bytes_read, current_chunk_size);
          current_chunk_size = static_cast<size_t>(bytes_read);
        } else if (file.fail() || bytes_read == 0) {
          HELOG(kError, "BinaryFileAssimilator: Failed to read chunk {} from file '{}' (bytes_read={}, eof={}, fail={}, bad={})",
                chunk_idx, src_path, bytes_read, file.eof(), file.fail(), file.bad());
          HELOG(kError, "BinaryFileAssimilator: File position: {}, bytes_processed: {}, total_size: {}",
                file.tellg(), bytes_processed, total_size);
          CHI_IPC->FreeBuffer(buffer_ptr);
          return -9;
        }
      }

      HILOG(kInfo, "BinaryFileAssimilator: Read chunk {} successfully ({} bytes)",
            chunk_idx, bytes_read);

      // Create blob name with chunk index
      std::string blob_name = "chunk_" + std::to_string(chunk_idx);

      HILOG(kInfo, "BinaryFileAssimilator: Submitting chunk {} (offset={}, size={}, blob='{}')",
            chunk_idx, chunk_offset + bytes_processed, current_chunk_size, blob_name);

      // Submit PutBlob task asynchronously
      HILOG(kInfo, "BinaryFileAssimilator: About to call AsyncPutBlob for chunk {}", chunk_idx);
      auto task = cte_client_->AsyncPutBlob(
          tag_id, blob_name, 0,
          current_chunk_size, buffer_ptr.shm_.template Cast<void>(), 1.0f, 0);

      active_tasks.push_back(task);
      
      bytes_processed += current_chunk_size;
      chunk_idx++;
    }

    // Wait for at least one task to complete before continuing
    if (!active_tasks.empty()) {
      // Wait for the first task to complete
      auto& first_task = active_tasks.front();
      first_task->Wait();

      if (first_task->return_code_.load() != 0) {
        HELOG(kError, "BinaryFileAssimilator: PutBlob task failed with code {}",
              first_task->return_code_.load());
        // Free the buffer before deleting the task
        CHI_IPC->FreeBuffer(first_task->blob_data_.template Cast<char>());
        CHI_IPC->DelTask(first_task);
        return -10;
      }

      // Free the buffer before deleting the task
      CHI_IPC->FreeBuffer(first_task->blob_data_.template Cast<char>());
      CHI_IPC->DelTask(first_task);
      active_tasks.erase(active_tasks.begin());
    }
  }

  // Wait for all remaining tasks to complete
  HILOG(kInfo, "BinaryFileAssimilator: Waiting for {} remaining tasks to complete", active_tasks.size());
  for (auto& task : active_tasks) {
    task->Wait();
    if (task->return_code_.load() != 0) {
      HELOG(kError, "BinaryFileAssimilator: PutBlob task failed with code {}",
            task->return_code_.load());
      // Free the buffer before deleting the task
      CHI_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
      CHI_IPC->DelTask(task);
      return -10;
    }
    // Free the buffer before deleting the task
    CHI_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
    CHI_IPC->DelTask(task);
  }

  file.close();

  HILOG(kInfo, "BinaryFileAssimilator: Successfully scheduled {} chunks for file '{}' to tag '{}'",
        num_chunks, src_path, tag_name);
  HILOG(kInfo, "BinaryFileAssimilator::Schedule EXIT: Success");

  return 0;
}

std::string BinaryFileAssimilator::GetUrlProtocol(const std::string& url) {
  size_t pos = url.find("::");
  if (pos == std::string::npos) {
    return "";
  }
  return url.substr(0, pos);
}

std::string BinaryFileAssimilator::GetUrlPath(const std::string& url) {
  size_t pos = url.find("::");
  if (pos == std::string::npos) {
    return "";
  }
  return url.substr(pos + 2);
}

size_t BinaryFileAssimilator::GetFileSize(const std::string& file_path) {
  struct stat st;
  if (stat(file_path.c_str(), &st) != 0) {
    return 0;
  }
  return static_cast<size_t>(st.st_size);
}

}  // namespace wrp_cae::core
