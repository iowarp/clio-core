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

#include <clio_cae/core/factory/csv_file_assimilator.h>
#include <clio_cae/core/factory/glob_match.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/ipc_manager.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

// Include clio_cte headers after closing any clio_cae namespace to avoid Method
// namespace collision
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

namespace clio::cae::core {

namespace {

/** Chunk size; must stay < 2 MB (MultiProcessAllocator thread unit). */
constexpr size_t kCsvMaxChunkSize = 1536 * 1024;
/** PutBlob tasks in flight per column. */
constexpr size_t kCsvMaxParallelTasks = 32;

/**
 * Columns a dataset_filter selects, in file order.
 * @param names Header names.
 * @param ctx Context holding include/exclude patterns.
 * @param out Receives the indices of the selected columns.
 */
void SelectColumns(const std::vector<std::string>& names,
                   const AssimilationCtx& ctx, std::vector<size_t>* out) {
  out->clear();
  for (size_t c = 0; c < names.size(); ++c) {
    if (MatchesDatasetFilter(CsvFileAssimilator::DatasetName(names[c]),
                             ctx.include_patterns, ctx.exclude_patterns)) {
      out->push_back(c);
    }
  }
}

}  // namespace

CsvFileAssimilator::CsvFileAssimilator(
    std::shared_ptr<clio::cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

clio::run::TaskResume CsvFileAssimilator::Schedule(const AssimilationCtx& ctx,
                                                   int& error_code) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  if (UrlProtocol(ctx.dst) != "iowarp") {
    HLOG(kError, "CsvFileAssimilator: destination must be iowarp::, got '{}'",
         ctx.dst);
    error_code = -1;
    CLIO_CO_RETURN;
  }
  std::string tag_prefix = UrlPath(ctx.dst);
  std::string src_path = UrlPath(ctx.src);
  if (tag_prefix.empty() || src_path.empty()) {
    HLOG(kError, "CsvFileAssimilator: bad src '{}' or dst '{}'", ctx.src,
         ctx.dst);
    error_code = tag_prefix.empty() ? -2 : -3;
    CLIO_CO_RETURN;
  }
  std::string text;
  if (!ReadFile(src_path, &text)) {
    HLOG(kError, "CsvFileAssimilator: cannot read '{}'", src_path);
    error_code = -4;
    CLIO_CO_RETURN;
  }
  CsvTable table;
  std::string parse_err;
  if (!ParseCsv(text, &table, &parse_err)) {
    HLOG(kError, "CsvFileAssimilator: '{}': {}", src_path, parse_err);
    error_code = -5;
    CLIO_CO_RETURN;
  }
  text.clear();
  text.shrink_to_fit();
  std::vector<size_t> selected;
  SelectColumns(table.names, ctx, &selected);
  HLOG(kInfo, "CsvFileAssimilator: '{}': {} column(s) x {} row(s), {} selected",
       src_path, table.names.size(), table.NumRows(), selected.size());
  if (selected.empty() && !ctx.include_patterns.empty()) {
    // Same contract as the HDF5 assimilator: an include list that selects
    // nothing is a mistake, not an empty success.
    HLOG(kError,
         "CsvFileAssimilator: NO column matched the {} include pattern(s); "
         "columns are matched as '/<header>'",
         ctx.include_patterns.size());
    error_code = -9;
    CLIO_CO_RETURN;
  }
  int failures = 0;
  for (size_t k = 0; k < selected.size(); ++k) {
    const size_t c = selected[k];
    CsvColumnTensor tensor;
    EncodeCsvColumn(table.fields[c], &tensor);
    table.fields[c].clear();
    table.fields[c].shrink_to_fit();
    int rc = 0;
    CLIO_CO_AWAIT(StoreColumn(tag_prefix + "/" + table.names[c], tensor, rc));
    if (rc != 0) {
      HLOG(kError, "CsvFileAssimilator: column '{}' failed ({})",
           table.names[c], rc);
      ++failures;
    }
  }
  error_code = failures == 0 ? 0 : -6;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CsvFileAssimilator::StoreColumn(
    const std::string& tag_name, const CsvColumnTensor& tensor,
    int& error_code) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  auto tag_task = cte_client_->AsyncGetOrCreateTag(tag_name);
  CLIO_CO_AWAIT(tag_task);
  clio::cte::core::TagId tag_id = tag_task->tag_id_;
  if (tag_id.IsNull()) {
    error_code = -1;
    CLIO_CO_RETURN;
  }
  const size_t desc_size = tensor.description.size();
  auto desc_buf = CLIO_IPC->AllocateBuffer(desc_size);
  std::memcpy(desc_buf.ptr_, tensor.description.data(), desc_size);
  auto desc_task = cte_client_->AsyncPutBlob(
      tag_id, "description", 0, desc_size, desc_buf.shm_.template Cast<void>(),
      1.0f, clio::cte::core::Context(), 0);
  CLIO_CO_AWAIT(desc_task);
  CLIO_IPC->FreeBuffer(desc_buf);
  if (desc_task->return_code_ != 0) {
    error_code = -2;
    CLIO_CO_RETURN;
  }
  const size_t total = tensor.bytes.size();
  size_t done = 0;
  size_t chunk_idx = 0;
  int rc = 0;
  std::vector<clio::run::Future<clio::cte::core::PutBlobTask>> active;
  while ((done < total || !active.empty())) {
    while (rc == 0 && active.size() < kCsvMaxParallelTasks && done < total) {
      const size_t n = std::min(kCsvMaxChunkSize, total - done);
      auto buf = CLIO_IPC->AllocateBuffer(n);
      std::memcpy(buf.ptr_, tensor.bytes.data() + done, n);
      active.push_back(cte_client_->AsyncPutBlob(
          tag_id, "chunk_" + std::to_string(chunk_idx), 0, n,
          buf.shm_.template Cast<void>(), 1.0f, clio::cte::core::Context(), 0));
      done += n;
      ++chunk_idx;
    }
    if (active.empty()) break;  // rc != 0 and nothing left in flight
    CLIO_CO_AWAIT(active.front());
    if (active.front()->return_code_ != 0) rc = -3;
    CLIO_IPC->FreeBuffer(active.front()->blob_data_.template Cast<char>());
    active.erase(active.begin());
  }
  HLOG(kDebug, "CsvFileAssimilator: '{}' {} -> {} chunk(s), {} bytes", tag_name,
       tensor.description, chunk_idx, total);
  error_code = rc;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

bool CsvFileAssimilator::ReadFile(const std::string& path, std::string* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  if (in.bad()) return false;
  *out = ss.str();
  return true;
}

std::string CsvFileAssimilator::UrlPath(const std::string& url) {
  size_t pos = url.find("::");
  return pos == std::string::npos ? std::string() : url.substr(pos + 2);
}

std::string CsvFileAssimilator::UrlProtocol(const std::string& url) {
  size_t pos = url.find("::");
  return pos == std::string::npos ? std::string() : url.substr(0, pos);
}

}  // namespace clio::cae::core
