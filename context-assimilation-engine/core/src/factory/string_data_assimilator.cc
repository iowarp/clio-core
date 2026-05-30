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
#include <clio_cae/core/factory/string_data_assimilator.h>

#include <cstring>
#include <string>

#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

namespace clio::cae::core {

namespace {

// Parse "protocol::path" — returns the path part, or empty if malformed.
std::string GetUrlPath(const std::string& url) {
  size_t pos = url.find("::");
  if (pos == std::string::npos) return "";
  return url.substr(pos + 2);
}

std::string GetUrlProtocol(const std::string& url) {
  size_t pos = url.find("::");
  if (pos == std::string::npos) return "";
  return url.substr(0, pos);
}

}  // namespace

StringDataAssimilator::StringDataAssimilator(
    std::shared_ptr<clio::cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

chi::TaskResume StringDataAssimilator::Schedule(const AssimilationCtx& ctx,
                                                int& error_code) {
#ifdef __NVCOMPILER
  thread_local chi::RunContext _fb_rctx;
  chi::RunContext* _fp = chi::GetCurrentRunContextFromWorker();
  chi::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug,
       "StringDataAssimilator::Schedule ENTRY: src='{}', dst='{}', "
       "src_data_size={}",
       ctx.src, ctx.dst, ctx.src_data.size());

  // src must be "string::<blob_name>"
  if (GetUrlProtocol(ctx.src) != "string") {
    HLOG(kError,
         "StringDataAssimilator: Source protocol must be 'string', got '{}'",
         ctx.src);
    error_code = -1;
    CLIO_CO_RETURN;
  }
  std::string blob_name = GetUrlPath(ctx.src);
  if (blob_name.empty()) {
    HLOG(kError,
         "StringDataAssimilator: Invalid source URL — no blob name after "
         "'string::'");
    error_code = -2;
    CLIO_CO_RETURN;
  }

  // dst must be "iowarp::<tag_name>"
  if (GetUrlProtocol(ctx.dst) != "iowarp") {
    HLOG(kError,
         "StringDataAssimilator: Destination protocol must be 'iowarp', got "
         "'{}'",
         ctx.dst);
    error_code = -3;
    CLIO_CO_RETURN;
  }
  std::string tag_name = GetUrlPath(ctx.dst);
  if (tag_name.empty()) {
    HLOG(kError,
         "StringDataAssimilator: Invalid destination URL — no tag name after "
         "'iowarp::'");
    error_code = -4;
    CLIO_CO_RETURN;
  }

  // Get or create the tag derived from dst
  auto tag_task = cte_client_->AsyncGetOrCreateTag(tag_name);
  CLIO_CO_AWAIT(tag_task);
  clio::cte::core::TagId tag_id = tag_task->tag_id_;
  if (tag_id.IsNull()) {
    HLOG(kError, "StringDataAssimilator: Failed to get or create tag '{}'",
         tag_name);
    error_code = -5;
    CLIO_CO_RETURN;
  }

  // CTE's PutBlob rejects size=0, so refuse empty payloads up front
  // rather than letting the failure surface as an opaque rc downstream.
  // The agentic-loop use case always has bytes to push; an empty
  // src_data is almost certainly a caller bug.
  size_t data_size = ctx.src_data.size();
  if (data_size == 0) {
    HLOG(kError,
         "StringDataAssimilator: ctx.src_data is empty for src='{}' — CTE "
         "does not store zero-byte blobs",
         ctx.src);
    error_code = -7;
    CLIO_CO_RETURN;
  }
  auto buffer = CLIO_IPC->AllocateBuffer(data_size);
  std::memcpy(buffer.ptr_, ctx.src_data.data(), data_size);

  auto put_task = cte_client_->AsyncPutBlob(
      tag_id, blob_name, /*offset=*/0, data_size,
      buffer.shm_.template Cast<void>(), /*score=*/1.0f,
      clio::cte::core::Context(), /*flags=*/0);
  CLIO_CO_AWAIT(put_task);

  if (put_task->return_code_ != 0) {
    HLOG(kError,
         "StringDataAssimilator: PutBlob failed for tag='{}' blob='{}' rc={}",
         tag_name, blob_name, put_task->return_code_);
    CLIO_IPC->FreeBuffer(put_task->blob_data_.template Cast<char>());
    error_code = -6;
    CLIO_CO_RETURN;
  }

  CLIO_IPC->FreeBuffer(put_task->blob_data_.template Cast<char>());

  HLOG(kDebug,
       "StringDataAssimilator: Stored {} bytes as tag='{}' blob='{}'",
       data_size, tag_name, blob_name);
  error_code = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cae::core
