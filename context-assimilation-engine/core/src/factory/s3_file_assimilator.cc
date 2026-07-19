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
#include <clio_cae/core/factory/s3_file_assimilator.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Include clio_cte headers after the clio_cae includes to avoid Method
// namespace collision (same ordering as BinaryFileAssimilator).
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

namespace clio::cae::core {

namespace {

// The S3 download is performed by a SEPARATE process (`cae_s3_tool`), never by
// linking the AWS SDK into this runtime. Loading libaws-cpp-sdk-core.so into the
// CLIO runtime process corrupts runtime startup (its load-time global ctors /
// static-baked s2n collide with CLIO's init), so the SDK is fully isolated in the
// helper and reached via fork+exec — the same approach the Globus assimilator uses
// to shell out to curl.

/**
 * Resolve the path/name of the cae_s3_tool helper executable.
 *
 * Honors the CAE_S3_TOOL environment override (used by the test harness to point
 * at the build-tree binary); otherwise returns the bare name, resolved via PATH.
 *
 * @return Helper executable path or name.
 */
std::string ResolveS3Tool() {
  const char* override_path = std::getenv("CAE_S3_TOOL");
  if (override_path && *override_path) {
    return override_path;
  }
  return "cae_s3_tool";
}

/**
 * Fork + exec a process and wait for it to finish.
 *
 * @param args Full argv (args[0] is the program, resolved via PATH).
 * @return The child's exit status (0 = success), or -1 if fork/exec failed.
 */
int RunProcess(const std::vector<std::string>& args) {
  pid_t pid = fork();
  if (pid == -1) {
    return -1;
  }
  if (pid == 0) {
    std::vector<const char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& a : args) {
      argv.push_back(a.c_str());
    }
    argv.push_back(nullptr);
    execvp(argv[0], const_cast<char* const*>(argv.data()));
    _exit(127);  // exec failed
  }
  int status = 0;
  waitpid(pid, &status, 0);
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return -1;
}

/**
 * Create a unique temporary file and return its path. The file is created empty.
 *
 * @param out_path Output: the created temp file path (unchanged on failure).
 * @return true on success, false if the temp file could not be created.
 */
bool MakeTempFile(std::string& out_path) {
  const char* tmpdir = std::getenv("TMPDIR");
  std::string tmpl = (tmpdir && *tmpdir) ? tmpdir : "/tmp";
  tmpl += "/cae_s3_XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  int fd = mkstemp(buf.data());
  if (fd == -1) {
    return false;
  }
  close(fd);  // helper reopens by name; keep only the path
  out_path.assign(buf.data());
  return true;
}

}  // namespace

S3FileAssimilator::S3FileAssimilator(
    std::shared_ptr<clio::cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

clio::run::TaskResume S3FileAssimilator::Schedule(const AssimilationCtx& ctx,
                                                  int& error_code) {
#ifdef __NVCOMPILER
  thread_local clio::run::RunContext _fb_rctx;
  clio::run::RunContext* _fp = clio::run::GetCurrentRunContextFromWorker();
  clio::run::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug,
       "S3FileAssimilator::Schedule ENTRY: src='{}', dst='{}', range_off={}, "
       "range_size={}",
       ctx.src, ctx.dst, ctx.range_off, ctx.range_size);

  // Validate destination protocol
  std::string dst_protocol = GetUrlProtocol(ctx.dst);
  if (dst_protocol != "iowarp") {
    HLOG(kError,
         "S3FileAssimilator: Destination protocol must be 'iowarp', got '{}'",
         dst_protocol);
    error_code = -1;
    CLIO_CO_RETURN;
  }

  // Extract tag name from destination URL
  std::string tag_name = GetUrlPath(ctx.dst);
  if (tag_name.empty()) {
    HLOG(kError,
         "S3FileAssimilator: Invalid destination URL, no tag name found");
    error_code = -2;
    CLIO_CO_RETURN;
  }

  // Get or create the tag in CTE
  auto tag_task = cte_client_->AsyncGetOrCreateTag(tag_name);
  CLIO_CO_AWAIT(tag_task);
  clio::cte::core::TagId tag_id = tag_task->tag_id_;
  if (tag_id.IsNull()) {
    HLOG(kError, "S3FileAssimilator: Failed to get or create tag '{}'",
         tag_name);
    error_code = -3;
    CLIO_CO_RETURN;
  }

  // Dependency-based scheduling is not yet supported (mirrors binary backend).
  if (!ctx.depends_on.empty()) {
    HLOG(kDebug,
         "S3FileAssimilator: Dependency handling not yet implemented "
         "(depends_on: {})",
         ctx.depends_on);
    error_code = 0;
    CLIO_CO_RETURN;
  }

  // Parse the S3 source URL into bucket + key
  std::string bucket;
  std::string key;
  if (!ParseS3Url(ctx.src, bucket, key)) {
    HLOG(kError, "S3FileAssimilator: Invalid S3 source URL '{}'", ctx.src);
    error_code = -4;
    CLIO_CO_RETURN;
  }

  // Download the object to a temp file via the out-of-process cae_s3_tool helper
  // (the AWS SDK must not be loaded into this runtime process). The helper reads
  // credentials / region / endpoint from the standard AWS environment.
  std::string tmp_path;
  if (!MakeTempFile(tmp_path)) {
    HLOG(kError, "S3FileAssimilator: Failed to create temp file for download");
    error_code = -5;
    CLIO_CO_RETURN;
  }
  std::vector<std::string> args = {ResolveS3Tool(), "get", bucket, key,
                                   tmp_path};
  if (ctx.range_size > 0) {
    args.push_back(std::to_string(ctx.range_off));
    args.push_back(std::to_string(ctx.range_size));
  }
  int tool_rc = RunProcess(args);
  if (tool_rc != 0) {
    HLOG(kError,
         "S3FileAssimilator: cae_s3_tool get failed (rc={}) for s3://{}/{}",
         tool_rc, bucket, key);
    ::unlink(tmp_path.c_str());
    error_code = -7;
    CLIO_CO_RETURN;
  }

  // Open the downloaded file and stream it into CTE. Unlinking the open file
  // makes cleanup automatic on every return path below (POSIX: the bytes stay
  // readable until the stream closes).
  std::ifstream body(tmp_path, std::ios::binary);
  ::unlink(tmp_path.c_str());
  if (!body) {
    HLOG(kError, "S3FileAssimilator: Failed to open downloaded object '{}'",
         tmp_path);
    error_code = -8;
    CLIO_CO_RETURN;
  }
  body.seekg(0, std::ios::end);
  size_t total_size = static_cast<size_t>(body.tellg());
  body.seekg(0, std::ios::beg);
  size_t chunk_offset = (ctx.range_size > 0) ? ctx.range_off : 0;
  HLOG(kDebug, "S3FileAssimilator: s3://{}/{} -> {} bytes (offset {})", bucket,
       key, total_size, chunk_offset);

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
         "S3FileAssimilator: Failed to store description for tag '{}' (code {})",
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

      body.read(buffer, current_chunk_size);
      std::streamsize bytes_read = body.gcount();
      if (bytes_read != static_cast<std::streamsize>(current_chunk_size)) {
        if (body.eof() && bytes_read > 0) {
          current_chunk_size = static_cast<size_t>(bytes_read);
        } else {
          HLOG(kError,
               "S3FileAssimilator: Short read on chunk {} from s3://{}/{} "
               "(bytes_read={}, eof={}, fail={})",
               chunk_idx, bucket, key, bytes_read, body.eof(), body.fail());
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
        HLOG(kError, "S3FileAssimilator: PutBlob task failed with code {}",
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
      HLOG(kError, "S3FileAssimilator: PutBlob task failed with code {}",
           task->return_code_);
      CLIO_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
      error_code = -10;
      CLIO_CO_RETURN;
    }
    CLIO_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
  }

  HLOG(kDebug,
       "S3FileAssimilator: Imported s3://{}/{} ({} chunks) into tag '{}'",
       bucket, key, chunk_idx, tag_name);
  error_code = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

std::string S3FileAssimilator::GetUrlProtocol(const std::string& url) {
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

std::string S3FileAssimilator::GetUrlPath(const std::string& url) {
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

bool S3FileAssimilator::ParseS3Url(const std::string& url, std::string& bucket,
                                   std::string& key) {
  // Strip the scheme (`s3://` or `s3::`) -> "bucket/key".
  std::string path = GetUrlPath(url);
  if (path.empty()) {
    return false;
  }
  size_t slash = path.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= path.size()) {
    return false;  // need both a bucket and a non-empty key
  }
  bucket = path.substr(0, slash);
  key = path.substr(slash + 1);
  return true;
}

}  // namespace clio::cae::core
