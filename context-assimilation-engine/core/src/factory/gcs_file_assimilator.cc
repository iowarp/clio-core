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

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/factory/gcs_file_assimilator.h>

// google-cloud-cpp pulls in abseil, whose <absl/base/log_severity.h> declares
//   enum class LogSeverity : int { kInfo, kWarning, kError, kFatal };
// CLIO's logging.h defines those same names as object-like macros
// (kInfo=1, kWarning=3, kError=4, kFatal=5), which textually mangle the enum and
// produce a cascade of syntax errors. Neutralize the macros across just the
// google-cloud includes, then restore them so HLOG(kInfo, ...) keeps working in
// the rest of this translation unit. (S3 never hit this — the AWS SDK has no abseil.)
#pragma push_macro("kInfo")
#pragma push_macro("kWarning")
#pragma push_macro("kError")
#pragma push_macro("kFatal")
#undef kInfo
#undef kWarning
#undef kError
#undef kFatal
#include <google/cloud/credentials.h>
#include <google/cloud/options.h>
#include <google/cloud/storage/client.h>
#include <google/cloud/storage/options.h>
#pragma pop_macro("kFatal")
#pragma pop_macro("kError")
#pragma pop_macro("kWarning")
#pragma pop_macro("kInfo")

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <string>
#include <vector>

// Include clio_cte headers after the clio_cae includes to avoid Method
// namespace collision (same ordering as BinaryFileAssimilator).
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

namespace gcs = google::cloud::storage;
namespace gc = google::cloud;

namespace clio::cae::core {

GcsFileAssimilator::GcsFileAssimilator(
    std::shared_ptr<clio::cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

clio::run::TaskResume GcsFileAssimilator::Schedule(const AssimilationCtx& ctx,
                                                   int& error_code) {
#ifdef __NVCOMPILER
  thread_local clio::run::RunContext _fb_rctx;
  clio::run::RunContext* _fp = clio::run::GetCurrentRunContextFromWorker();
  clio::run::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug,
       "GcsFileAssimilator::Schedule ENTRY: src='{}', dst='{}', range_off={}, "
       "range_size={}",
       ctx.src, ctx.dst, ctx.range_off, ctx.range_size);

  // Validate destination protocol
  std::string dst_protocol = GetUrlProtocol(ctx.dst);
  if (dst_protocol != "iowarp") {
    HLOG(kError,
         "GcsFileAssimilator: Destination protocol must be 'iowarp', got '{}'",
         dst_protocol);
    error_code = -1;
    CLIO_CO_RETURN;
  }

  // Extract tag name from destination URL
  std::string tag_name = GetUrlPath(ctx.dst);
  if (tag_name.empty()) {
    HLOG(kError,
         "GcsFileAssimilator: Invalid destination URL, no tag name found");
    error_code = -2;
    CLIO_CO_RETURN;
  }

  // Get or create the tag in CTE
  auto tag_task = cte_client_->AsyncGetOrCreateTag(tag_name);
  CLIO_CO_AWAIT(tag_task);
  clio::cte::core::TagId tag_id = tag_task->tag_id_;
  if (tag_id.IsNull()) {
    HLOG(kError, "GcsFileAssimilator: Failed to get or create tag '{}'",
         tag_name);
    error_code = -3;
    CLIO_CO_RETURN;
  }

  // Dependency-based scheduling is not yet supported (mirrors binary backend).
  if (!ctx.depends_on.empty()) {
    HLOG(kDebug,
         "GcsFileAssimilator: Dependency handling not yet implemented "
         "(depends_on: {})",
         ctx.depends_on);
    error_code = 0;
    CLIO_CO_RETURN;
  }

  // Parse the GCS source URL into bucket + object
  std::string bucket;
  std::string object;
  if (!ParseGcsUrl(ctx.src, bucket, object)) {
    HLOG(kError, "GcsFileAssimilator: Invalid GCS source URL '{}'", ctx.src);
    error_code = -4;
    CLIO_CO_RETURN;
  }

  // Build a GCS client. Credentials default to Application Default
  // Credentials; an optional GCS_ENDPOINT (e.g. fake-gcs-server) switches to
  // that endpoint with anonymous credentials.
  gc::Options options;
  const char* endpoint_env = std::getenv("GCS_ENDPOINT");
  if (endpoint_env && *endpoint_env) {
    options.set<gcs::RestEndpointOption>(endpoint_env);
    options.set<gc::UnifiedCredentialsOption>(gc::MakeInsecureCredentials());
  }
  auto client = gcs::Client(std::move(options));

  // Open the object stream: ranged when range_size>0, otherwise the whole
  // object (size resolved from object metadata for the description blob).
  size_t total_size;
  size_t chunk_offset;
  gcs::ObjectReadStream stream;
  if (ctx.range_size > 0) {
    chunk_offset = ctx.range_off;
    total_size = ctx.range_size;
    stream = client.ReadObject(
        bucket, object,
        gcs::ReadRange(static_cast<std::int64_t>(ctx.range_off),
                       static_cast<std::int64_t>(ctx.range_off + ctx.range_size)));
  } else {
    auto metadata = client.GetObjectMetadata(bucket, object);
    if (!metadata) {
      HLOG(kError, "GcsFileAssimilator: HEAD gs://{}/{} failed: {}", bucket,
           object, metadata.status().message());
      error_code = -5;
      CLIO_CO_RETURN;
    }
    total_size = static_cast<size_t>(metadata->size());
    chunk_offset = 0;
    stream = client.ReadObject(bucket, object);
  }
  if (!stream.status().ok()) {
    HLOG(kError, "GcsFileAssimilator: ReadObject gs://{}/{} failed: {}", bucket,
         object, stream.status().message());
    error_code = -7;
    CLIO_CO_RETURN;
  }
  HLOG(kDebug, "GcsFileAssimilator: gs://{}/{} -> {} bytes (offset {})", bucket,
       object, total_size, chunk_offset);

  // Store object metadata as the "description" blob (mirrors binary backend).
  std::string description = "binary<size=" + std::to_string(total_size) +
                            ", offset=" + std::to_string(chunk_offset) + ">";
  size_t desc_size = description.size();
  auto desc_buffer = CLIO_IPC->AllocateBuffer(desc_size);
  std::memcpy(desc_buffer.ptr_, description.c_str(), desc_size);
  auto desc_task =
      cte_client_->AsyncPutBlob(tag_id, "description", 0, desc_size,
                                desc_buffer.shm_.template Cast<void>(), 1.0f,
                                clio::cte::core::Context(), 0);
  CLIO_CO_AWAIT(desc_task);
  if (desc_task->return_code_ != 0) {
    HLOG(kError,
         "GcsFileAssimilator: Failed to store description for tag '{}' (code {})",
         tag_name, desc_task->return_code_);
    error_code = -9;
    CLIO_CO_RETURN;
  }

  // Stream the object body into CTE in chunks, keeping up to kMaxParallelTasks
  // PutBlob tasks in flight (identical wait-and-drain shape to binary backend).
  static constexpr size_t kMaxChunkSize = 1024 * 1024;  // 1 MB
  static constexpr size_t kMaxParallelTasks = 32;
  size_t chunk_idx = 0;
  size_t bytes_processed = 0;
  std::vector<clio::run::Future<clio::cte::core::PutBlobTask>> active_tasks;

  while (bytes_processed < total_size) {
    while (active_tasks.size() < kMaxParallelTasks &&
           bytes_processed < total_size) {
      size_t current_chunk_size =
          std::min(kMaxChunkSize, total_size - bytes_processed);
      auto buffer_ptr = CLIO_IPC->AllocateBuffer(current_chunk_size);
      char* buffer = buffer_ptr.ptr_;

      stream.read(buffer, current_chunk_size);
      std::streamsize bytes_read = stream.gcount();
      if (bytes_read != static_cast<std::streamsize>(current_chunk_size)) {
        if (stream.eof() && bytes_read > 0) {
          current_chunk_size = static_cast<size_t>(bytes_read);
        } else {
          HLOG(kError,
               "GcsFileAssimilator: Short read on chunk {} from gs://{}/{} "
               "(bytes_read={}, eof={}, fail={})",
               chunk_idx, bucket, object, bytes_read, stream.eof(),
               stream.fail());
          CLIO_IPC->FreeBuffer(buffer_ptr);
          error_code = -9;
          CLIO_CO_RETURN;
        }
      }

      std::string blob_name = "chunk_" + std::to_string(chunk_idx);
      auto task =
          cte_client_->AsyncPutBlob(tag_id, blob_name, 0, current_chunk_size,
                                    buffer_ptr.shm_.template Cast<void>(), 1.0f,
                                    clio::cte::core::Context(), 0);
      active_tasks.push_back(task);
      bytes_processed += current_chunk_size;
      chunk_idx++;
    }

    if (!active_tasks.empty()) {
      auto& first_task = active_tasks.front();
      CLIO_CO_AWAIT(first_task);
      if (first_task->return_code_ != 0) {
        HLOG(kError, "GcsFileAssimilator: PutBlob task failed with code {}",
             first_task->return_code_);
        CLIO_IPC->FreeBuffer(first_task->blob_data_.template Cast<char>());
        error_code = -10;
        CLIO_CO_RETURN;
      }
      CLIO_IPC->FreeBuffer(first_task->blob_data_.template Cast<char>());
      active_tasks.erase(active_tasks.begin());
    }
  }

  // Drain any remaining in-flight tasks.
  for (auto& task : active_tasks) {
    CLIO_CO_AWAIT(task);
    if (task->return_code_ != 0) {
      HLOG(kError, "GcsFileAssimilator: PutBlob task failed with code {}",
           task->return_code_);
      CLIO_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
      error_code = -10;
      CLIO_CO_RETURN;
    }
    CLIO_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
  }

  HLOG(kDebug,
       "GcsFileAssimilator: Imported gs://{}/{} ({} chunks) into tag '{}'",
       bucket, object, chunk_idx, tag_name);
  error_code = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

std::string GcsFileAssimilator::GetUrlProtocol(const std::string& url) {
  size_t pos_standard = url.find("://");
  if (pos_standard != std::string::npos) {
    return url.substr(0, pos_standard);
  }
  size_t pos_custom = url.find("::");
  if (pos_custom != std::string::npos) {
    return url.substr(0, pos_custom);
  }
  return "";
}

std::string GcsFileAssimilator::GetUrlPath(const std::string& url) {
  size_t pos_standard = url.find("://");
  if (pos_standard != std::string::npos) {
    return url.substr(pos_standard + 3);
  }
  size_t pos_custom = url.find("::");
  if (pos_custom != std::string::npos) {
    return url.substr(pos_custom + 2);
  }
  return "";
}

bool GcsFileAssimilator::ParseGcsUrl(const std::string& url, std::string& bucket,
                                     std::string& object) {
  // Strip the scheme (`gs://`, `gcs://`, or the `::` forms) -> "bucket/object".
  std::string path = GetUrlPath(url);
  if (path.empty()) {
    return false;
  }
  size_t slash = path.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= path.size()) {
    return false;  // need both a bucket and a non-empty object name
  }
  bucket = path.substr(0, slash);
  object = path.substr(slash + 1);
  return true;
}

}  // namespace clio::cae::core
