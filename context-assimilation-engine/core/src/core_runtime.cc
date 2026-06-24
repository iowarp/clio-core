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

#include <clio_cae/core/core_runtime.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cae/core/factory/assimilator_factory.h>
#include <clio_cae/core/label_client.h>
#ifdef CLIO_CAE_ENABLE_HDF5
#include <hdf5.h>
#include <clio_cae/core/factory/hdf5_file_assimilator.h>
#endif

#include "clio_ctp/data_structures/serialization/global_serialize.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <regex>
#include <string_view>
#include <vector>

// Include clio_cte headers before opening namespace to avoid Method namespace
// collision
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::cae::core::Runtime)

namespace clio::cae::core {

clio::run::TaskResume Runtime::Monitor(ctp::ipc::FullPtr<MonitorTask> task,
                                 clio::run::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  task->SetReturnCode(0);
  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Create(ctp::ipc::FullPtr<CreateTask> task, clio::run::RunContext& rctx) {
  CLIO_TASK_BODY_BEGIN
  // Container is already initialized via Init() before Create is called
  // Do NOT call Init() here

  // Load create-time configuration. In compose mode this materializes the
  // PoolConfig stored on the task into a CreateParams via LoadConfig.
  CreateParams params = task->GetParams();
  next_pool_id_ = params.next_pool_id_;
  label_matches_ = std::move(params.label_matches_);
  label_prompts_ = std::move(params.label_prompts_);
  label_endpoint_ = std::move(params.label_endpoint_);

  // Initialize CTE client. When CAE is wired as a transparent interceptor
  // (compose YAML sets next_pool_id), forward to that pool instead of the
  // hard-coded kCtePoolId; otherwise fall back to the legacy default so
  // existing CAE deployments keep working.
  clio::run::PoolId cte_pool = !next_pool_id_.IsNull()
                            ? next_pool_id_
                            : clio::cte::core::kCtePoolId;
  cte_client_ = std::make_shared<clio::cte::core::Client>(cte_pool);

  HLOG(kInfo,
       "Core container created and initialized for pool: {} (ID: {}), "
       "CTE next_pool_id={}, label_matches={}, label_endpoint='{}'",
       pool_name_, pool_id_, cte_pool, label_matches_.size(),
       label_endpoint_);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::PoolId Runtime::ResolveNextPoolId() const {
  if (!next_pool_id_.IsNull()) return next_pool_id_;
  return clio::cte::core::kCtePoolId;
}

clio::run::PoolQuery Runtime::ScheduleTask(
    const ctp::ipc::FullPtr<clio::run::Task> &task) {
  // Interceptor methods always run locally — they immediately forward
  // synchronously to the configured CTE core pool, so there is no value
  // in bouncing the task across nodes. Mirrors compressor_runtime.cc
  // which routes non-Compress Dynamic methods to Local.
  switch (task->method_) {
    case Method::kPutBlob:
    case Method::kGetBlob:
    case Method::kGetOrCreateTag:
    case Method::kSemanticSearch:
      return clio::run::PoolQuery::Local();
    default:
      return clio::run::PoolQuery::Local();
  }
}

/**
 * Walk label_matches_ and return the first rule (if any) whose tag_re_
 * matches `tag_name` and blob_re_ matches `blob_name`. std::regex_search
 * (not _match) so ".*\\.txt" matches any name ending in .txt without
 * needing an explicit anchor. An invalid regex in either side disables
 * that rule (logged once at kWarning).
 */
static const LabelMatch *FindLabelMatch(
    const std::vector<LabelMatch> &rules, const std::string &tag_name,
    const std::string &blob_name) {
  for (const auto &rule : rules) {
    try {
      std::regex tag_rx(rule.tag_re_);
      std::regex blob_rx(rule.blob_re_);
      if (std::regex_search(tag_name, tag_rx) &&
          std::regex_search(blob_name, blob_rx)) {
        return &rule;
      }
    } catch (const std::regex_error &e) {
      HLOG(kWarning,
           "FindLabelMatch: invalid regex in label rule (tag='{}' blob='{}'): {}",
           rule.tag_re_, rule.blob_re_, e.what());
    }
  }
  return nullptr;
}

clio::run::TaskResume Runtime::PutBlob(ctp::ipc::FullPtr<PutBlobTask> task,
                                 clio::run::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  if (!cte_client_) {
    cte_client_ = std::make_shared<clio::cte::core::Client>(ResolveNextPoolId());
  }
  // 1. Forward the original blob through to CTE first so the user's
  //    semantic is preserved regardless of labeling outcome. CO_AWAIT
  //    yields the worker while the downstream CTE task runs — Wait()
  //    would block this worker and deadlock when only a single worker is
  //    available (or when this and the downstream task happen to land on
  //    the same worker).
  auto fwd = cte_client_->AsyncPutBlob(
      task->tag_id_, task->blob_name_.str(), task->offset_, task->size_,
      task->blob_data_, task->score_, task->context_, task->flags_,
      clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(fwd);
  task->context_ = fwd->context_;
  task->SetReturnCode(fwd->GetReturnCode());

  // 2. Transparent labeling. Skipped silently when no rules are
  //    configured or no rule matches; labeling failures are logged but
  //    must not flip the PutBlob return code.
  if (label_matches_.empty() || fwd->GetReturnCode() != 0) {
    CLIO_CO_RETURN;
  }
  std::string tag_name;
  {
    std::lock_guard<std::mutex> lock(tag_names_mu_);
    auto it = tag_names_.find(task->tag_id_);
    if (it != tag_names_.end()) tag_name = it->second;
  }
  std::string blob_name = task->blob_name_.str();
  const LabelMatch *rule = FindLabelMatch(label_matches_, tag_name, blob_name);
  if (!rule) {
    CLIO_CO_RETURN;
  }

  // 3. Resolve the prompt template (named in the rule).
  auto pit = label_prompts_.find(rule->prompt_);
  if (pit == label_prompts_.end()) {
    HLOG(kWarning,
         "CAE::PutBlob: label rule references unknown prompt '{}', skipping",
         rule->prompt_);
    CLIO_CO_RETURN;
  }

  // 4. Snapshot the blob payload off shared memory into a plain string.
  //    The label_client uses libcurl synchronously so we want a stable
  //    buffer that doesn't share lifetime with the inbound ShmPtr.
  std::string payload;
  if (!task->blob_data_.IsNull() && task->size_ > 0) {
    auto fullptr =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    if (fullptr.ptr_) {
      payload.assign(fullptr.ptr_, task->size_);
    }
  }
  const std::string &prompt_template = pit->second;

  // 5. Decide whether to chunk. The Ollama API counts both prompt and
  //    generated tokens against num_ctx. Reserve ~25% of context for
  //    the prompt template + the response budget; the remaining 75% is
  //    available for blob payload. Convert tokens to bytes via a
  //    conservative ~3 bytes/token English ratio (binary blobs run
  //    closer to 1 byte/token, so this errs on splitting *more*).
  //
  //    context_length_<=0 disables chunking entirely — caller takes
  //    Ollama's default 2048 and accepts whatever truncation it does.
  const int ctx_tokens = rule->context_length_;
  std::vector<std::string_view> chunks;
  if (ctx_tokens <= 0 || payload.empty()) {
    chunks.emplace_back(payload);
  } else {
    size_t budget_tokens = static_cast<size_t>(ctx_tokens) * 3 / 4;
    size_t budget_bytes = budget_tokens * 3;
    if (budget_bytes > prompt_template.size() + 256) {
      budget_bytes -= prompt_template.size();
    }
    if (budget_bytes == 0) budget_bytes = 256;  // sanity floor
    for (size_t off = 0; off < payload.size(); off += budget_bytes) {
      size_t take = std::min(budget_bytes, payload.size() - off);
      chunks.emplace_back(payload.data() + off, take);
    }
  }

  // 6. Inference per chunk, then concatenate. A labeling failure on any
  //    one chunk doesn't abort the whole label — we skip the chunk and
  //    log; the user still gets a partial label. A real production
  //    deploy would dispatch each chunk to a dedicated labeling worker
  //    pool; v1 keeps everything inline.
  std::string label_text;
  size_t successful_chunks = 0;
  for (size_t i = 0; i < chunks.size(); ++i) {
    std::string full_prompt = prompt_template;
    full_prompt.append("\n\n");
    full_prompt.append(chunks[i].data(), chunks[i].size());

    std::string chunk_label;
    bool ok = OllamaGenerate(label_endpoint_, rule->model_, full_prompt,
                             ctx_tokens, rule->num_predict_, chunk_label);
    if (!ok || chunk_label.empty()) {
      HLOG(kWarning,
           "CAE::PutBlob: chunk {} of {} failed for tag='{}' blob='{}' "
           "model='{}'",
           i + 1, chunks.size(), tag_name, blob_name, rule->model_);
      continue;
    }
    if (!label_text.empty()) label_text.append("\n\n");
    label_text.append(chunk_label);
    ++successful_chunks;
  }
  if (successful_chunks == 0 || label_text.empty()) {
    HLOG(kWarning,
         "CAE::PutBlob: labeling produced no output for tag='{}' blob='{}'",
         tag_name, blob_name);
    CLIO_CO_RETURN;
  }

  // 7. Store the concatenated label as "{blob_name}_label" in the same
  //    tag, via the same CTE forwarding path so it lands on the same
  //    backend as the original blob.
  std::string label_blob_name = blob_name + "_label";
  auto *ipc = CLIO_IPC;
  auto label_buf = ipc->AllocateBuffer(label_text.size());
  if (label_buf.IsNull()) {
    HLOG(kWarning, "CAE::PutBlob: label SHM allocation failed");
    CLIO_CO_RETURN;
  }
  std::memcpy(label_buf.ptr_, label_text.data(), label_text.size());
  ctp::ipc::ShmPtr<> label_shm = label_buf.shm_.template Cast<void>();
  clio::cte::core::Context label_ctx;
  auto label_fut = cte_client_->AsyncPutBlob(
      task->tag_id_, label_blob_name, 0,
      static_cast<clio::run::u64>(label_text.size()), label_shm, task->score_,
      label_ctx, 0, clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(label_fut);
  ipc->FreeBuffer(label_buf);
  if (label_fut->GetReturnCode() != 0) {
    HLOG(kWarning,
         "CAE::PutBlob: failed to store label blob '{}' (rc={})",
         label_blob_name, label_fut->GetReturnCode());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlob(ctp::ipc::FullPtr<GetBlobTask> task,
                                 clio::run::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  if (!cte_client_) {
    cte_client_ = std::make_shared<clio::cte::core::Client>(ResolveNextPoolId());
  }
  auto fwd = cte_client_->AsyncGetBlob(
      task->tag_id_, task->blob_name_.str(), task->offset_, task->size_,
      task->flags_, task->blob_data_, clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(fwd);
  task->SetReturnCode(fwd->GetReturnCode());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetOrCreateTag(
    ctp::ipc::FullPtr<GetOrCreateTagTask> task, clio::run::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  if (!cte_client_) {
    cte_client_ = std::make_shared<clio::cte::core::Client>(ResolveNextPoolId());
  }
  std::string tag_name = task->tag_name_.str();
  auto fwd = cte_client_->AsyncGetOrCreateTag(
      tag_name, task->tag_id_, clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(fwd);
  task->tag_id_ = fwd->tag_id_;
  task->SetReturnCode(fwd->GetReturnCode());

  // Remember tag_id → tag_name so PutBlob can later match against
  // LabelMatch::tag_re_ without re-querying CTE. Only the resolved
  // (non-null) tag id is cached.
  if (!task->tag_id_.IsNull() && !tag_name.empty()) {
    std::lock_guard<std::mutex> lock(tag_names_mu_);
    tag_names_[task->tag_id_] = std::move(tag_name);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SemanticSearch(
    ctp::ipc::FullPtr<SemanticSearchTask> task, clio::run::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  if (!cte_client_) {
    cte_client_ = std::make_shared<clio::cte::core::Client>(ResolveNextPoolId());
  }
  // Forward verbatim. CAE doesn't add any semantic-search logic on
  // top of CTE today — the only reason this handler exists is so a
  // CTE client pointed at the CAE entrypoint pool keeps working
  // without any client-side rewiring.
  auto fwd = cte_client_->AsyncSemanticSearch(
      task->tag_regex_.str(), task->blob_regex_.str(),
      task->query_text_.str(), task->k_, clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(fwd);
  task->results_ = fwd->results_;
  task->SetReturnCode(fwd->GetReturnCode());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::u64 Runtime::GetWorkRemaining() const {
  // CAE doesn't currently track work remaining
  // Return 0 to indicate no pending work
  return 0;
}

clio::run::TaskResume Runtime::ParseOmni(ctp::ipc::FullPtr<ParseOmniTask> task,
                                   clio::run::RunContext& rctx) {
  CLIO_TASK_BODY_BEGIN
  HLOG(kInfo, "ParseOmni called with {} bytes of serialized data",
       task->serialized_ctx_.size());

  // Deserialize the vector of AssimilationCtx
  std::vector<AssimilationCtx> assimilation_contexts;
  try {
    std::string data = task->serialized_ctx_.str();
    std::vector<char> buf(data.begin(), data.end());
    ctp::ipc::GlobalDeserialize<std::vector<char>> ar(buf);
    ar(assimilation_contexts);
  } catch (const std::exception& e) {
    HLOG(kError, "ParseOmni: Failed to deserialize AssimilationCtx vector: {}",
         e.what());
    task->result_code_ = -1;
    task->error_message_ = e.what();
    task->num_tasks_scheduled_ = 0;
    CLIO_CO_RETURN;
  }

  HLOG(kInfo, "ParseOmni: Processing {} assimilation contexts",
       assimilation_contexts.size());

  // Process each assimilation context
  clio::run::u32 tasks_scheduled = 0;
  AssimilatorFactory factory(cte_client_);

  for (size_t i = 0; i < assimilation_contexts.size(); ++i) {
    const auto& assimilation_ctx = assimilation_contexts[i];

    HLOG(kInfo,
         "ParseOmni: Processing context {}/{} - src: {}, dst: {}, format: {}",
         i + 1, assimilation_contexts.size(), assimilation_ctx.src,
         assimilation_ctx.dst, assimilation_ctx.format);

    // Get appropriate assimilator for this context
    auto assimilator = factory.Get(assimilation_ctx.src);

    if (!assimilator) {
      HLOG(kError, "ParseOmni: No assimilator found for source: {}",
           assimilation_ctx.src);
      task->result_code_ = -2;
      task->error_message_ =
          "No assimilator found for source: " + assimilation_ctx.src;
      task->num_tasks_scheduled_ = tasks_scheduled;
      CLIO_CO_RETURN;
    }

    // Schedule the assimilation using co_await
    int result = 0;
    CLIO_CO_AWAIT(assimilator->Schedule(assimilation_ctx, result));
    if (result != 0) {
      HLOG(
          kError,
          "ParseOmni: Assimilator failed for context {}/{} with error code: {}",
          i + 1, assimilation_contexts.size(), result);
      task->result_code_ = result;
      task->error_message_ = std::string("Assimilator failed");
      task->num_tasks_scheduled_ = tasks_scheduled;
      CLIO_CO_RETURN;
    }

    tasks_scheduled++;
  }

  // Success
  task->result_code_ = 0;
  task->error_message_ = "";
  task->num_tasks_scheduled_ = tasks_scheduled;

  HLOG(kInfo, "ParseOmni: Successfully scheduled {} assimilations",
       tasks_scheduled);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ProcessHdf5Dataset(
    ctp::ipc::FullPtr<ProcessHdf5DatasetTask> task, clio::run::RunContext& rctx) {
  CLIO_TASK_BODY_BEGIN
#ifdef CLIO_CAE_ENABLE_HDF5
  HLOG(kInfo, "ProcessHdf5Dataset: file='{}', dataset='{}', tag_prefix='{}'",
       task->file_path_.str(), task->dataset_path_.str(),
       task->tag_prefix_.str());

  // Open the HDF5 file
  hid_t file_id =
      H5Fopen(task->file_path_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    HLOG(kError, "ProcessHdf5Dataset: Failed to open HDF5 file: {}",
         task->file_path_.str());
    task->result_code_ = -1;
    task->error_message_ =
        clio::run::priv::string("Failed to open HDF5 file", CTP_MALLOC);
    CLIO_CO_RETURN;
  }

  // Create assimilator and process the dataset
  clio::cae::core::Hdf5FileAssimilator assimilator(cte_client_);
  int result = 0;
  CLIO_CO_AWAIT(assimilator.ProcessDataset(file_id, task->dataset_path_.str(),
                                      task->tag_prefix_.str(), result));

  // Close the HDF5 file
  H5Fclose(file_id);

  if (result != 0) {
    HLOG(kError,
         "ProcessHdf5Dataset: Failed to process dataset '{}' (error: {})",
         task->dataset_path_.str(), result);
    task->result_code_ = result;
    task->error_message_ =
        clio::run::priv::string("Dataset processing failed", CTP_MALLOC);
  } else {
    HLOG(kInfo, "ProcessHdf5Dataset: Successfully processed dataset '{}'",
         task->dataset_path_.str());
    task->result_code_ = 0;
  }
#else
  task->result_code_ = -1;
  task->error_message_ =
      clio::run::priv::string("HDF5 support not compiled in", CTP_MALLOC);
#endif
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ExportData(ctp::ipc::FullPtr<ExportDataTask> task,
                                    clio::run::RunContext& rctx) {
  CLIO_TASK_BODY_BEGIN
  task->result_code_ = 0;
  task->bytes_exported_ = 0;

  const std::string tag_name = task->tag_name_.str();
  const std::string output_path = task->output_path_.str();
  const std::string format = task->format_.str();

  HLOG(kInfo, "ExportData: tag='{}', output='{}', format='{}'",
       tag_name, output_path, format);

  // Step 1: resolve the tag ID
  auto tag_future = cte_client_->AsyncGetOrCreateTag(tag_name);
  CLIO_CO_AWAIT(tag_future);
  const auto &tag_id = tag_future->tag_id_;
  if (tag_id.IsNull()) {
    HLOG(kError, "ExportData: tag '{}' not found", tag_name);
    task->result_code_ = -1;
    task->error_message_ = clio::run::priv::string("Tag not found", CTP_MALLOC);
    CLIO_CO_RETURN;
  }

  // Step 2: list all blobs in the tag
  auto blobs_future = cte_client_->AsyncGetContainedBlobs(tag_id);
  CLIO_CO_AWAIT(blobs_future);
  const auto &blob_names = blobs_future->blob_names_;

  if (blob_names.empty()) {
    HLOG(kInfo, "ExportData: tag '{}' has no blobs", tag_name);
    CLIO_CO_RETURN;
  }

  if (format == "hdf5") {
#ifdef CLIO_CAE_ENABLE_HDF5
    hid_t file_id = H5Fcreate(output_path.c_str(), H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
      HLOG(kError, "ExportData: failed to create HDF5 file '{}'", output_path);
      task->result_code_ = -2;
      task->error_message_ =
          clio::run::priv::string("Failed to create HDF5 file", CTP_MALLOC);
      CLIO_CO_RETURN;
    }

    for (const auto &blob_name : blob_names) {
      // Get blob size
      auto size_future = cte_client_->AsyncGetBlobSize(tag_id, blob_name);
      CLIO_CO_AWAIT(size_future);
      clio::run::u64 blob_size = size_future->size_;
      if (blob_size == 0) continue;

      // Allocate buffer and read blob
      auto *ipc_manager = CLIO_IPC;
      ctp::ipc::FullPtr<char> buf = ipc_manager->AllocateBuffer(blob_size);
      if (buf.IsNull()) {
        HLOG(kError, "ExportData: allocation failed for blob '{}'", blob_name);
        continue;
      }
      ctp::ipc::ShmPtr<> shm_ptr(buf.shm_);
      auto get_future = cte_client_->AsyncGetBlob(tag_id, blob_name, 0,
                                                   blob_size, 0, shm_ptr);
      CLIO_CO_AWAIT(get_future);

      if (get_future->GetReturnCode() == 0) {
        hsize_t dims[1] = {static_cast<hsize_t>(blob_size)};
        hid_t space = H5Screate_simple(1, dims, nullptr);
        hid_t ds = H5Dcreate2(file_id, blob_name.c_str(), H5T_NATIVE_UINT8,
                               space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds >= 0) {
          H5Dwrite(ds, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   buf.ptr_);
          H5Dclose(ds);
          task->bytes_exported_ += blob_size;
        }
        H5Sclose(space);
      }
      ipc_manager->FreeBuffer(buf);
    }

    H5Fclose(file_id);
    HLOG(kInfo, "ExportData: wrote {} bytes to HDF5 '{}'",
         task->bytes_exported_, output_path);
#else
    task->result_code_ = -3;
    task->error_message_ =
        clio::run::priv::string("HDF5 support not compiled in", CTP_MALLOC);
#endif
  } else {
    // Binary format: sequential blob data with a simple header per blob
    std::ofstream ofs(output_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
      HLOG(kError, "ExportData: failed to open '{}' for writing", output_path);
      task->result_code_ = -2;
      task->error_message_ =
          clio::run::priv::string("Failed to open output file", CTP_MALLOC);
      CLIO_CO_RETURN;
    }

    for (const auto &blob_name : blob_names) {
      auto size_future = cte_client_->AsyncGetBlobSize(tag_id, blob_name);
      CLIO_CO_AWAIT(size_future);
      clio::run::u64 blob_size = size_future->size_;
      if (blob_size == 0) continue;

      auto *ipc_manager = CLIO_IPC;
      ctp::ipc::FullPtr<char> buf = ipc_manager->AllocateBuffer(blob_size);
      if (buf.IsNull()) {
        HLOG(kError, "ExportData: allocation failed for blob '{}'", blob_name);
        continue;
      }
      ctp::ipc::ShmPtr<> shm_ptr(buf.shm_);
      auto get_future = cte_client_->AsyncGetBlob(tag_id, blob_name, 0,
                                                   blob_size, 0, shm_ptr);
      CLIO_CO_AWAIT(get_future);

      if (get_future->GetReturnCode() == 0) {
        // Header: name length (u32) + name + data length (u64) + data
        uint32_t name_len = static_cast<uint32_t>(blob_name.size());
        ofs.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
        ofs.write(blob_name.data(), name_len);
        ofs.write(reinterpret_cast<const char *>(&blob_size), sizeof(blob_size));
        ofs.write(buf.ptr_, static_cast<std::streamsize>(blob_size));
        task->bytes_exported_ += blob_size;
      }
      ipc_manager->FreeBuffer(buf);
    }

    HLOG(kInfo, "ExportData: wrote {} bytes to binary '{}'",
         task->bytes_exported_, output_path);
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cae::core
