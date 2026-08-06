/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/factory/model_weights_assimilator.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

// Include clio_cte headers after any clio_cae namespace to avoid Method
// namespace collision (matches binary_file_assimilator.cc).
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

namespace clio::cae::core {

ModelWeightsAssimilator::ModelWeightsAssimilator(
    std::shared_ptr<clio::cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

size_t ModelWeightsAssimilator::ParseFormatParam(const std::string& format,
                                                 const std::string& key,
                                                 size_t def) {
  // format looks like "modelweights;page=2097152;nblocks=8"
  std::string needle = key + "=";
  size_t pos = format.find(needle);
  if (pos == std::string::npos) return def;
  pos += needle.size();
  size_t end = format.find(';', pos);
  std::string val = format.substr(pos, end == std::string::npos
                                           ? std::string::npos
                                           : end - pos);
  if (val.empty()) return def;
  errno = 0;
  char* endp = nullptr;
  unsigned long long parsed = std::strtoull(val.c_str(), &endp, 10);
  if (endp == val.c_str() || parsed == 0ULL) return def;
  return static_cast<size_t>(parsed);
}

clio::run::TaskResume ModelWeightsAssimilator::Schedule(
    const AssimilationCtx& ctx, int& error_code) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug,
       "ModelWeightsAssimilator::Schedule ENTRY: src='{}', dst='{}', "
       "format='{}', range_off={}, range_size={}",
       ctx.src, ctx.dst, ctx.format, ctx.range_off, ctx.range_size);

  // ----- Validate destination (iowarp::<tag>) -----
  if (GetUrlProtocol(ctx.dst) != "iowarp") {
    HLOG(kError, "ModelWeightsAssimilator: dst protocol must be 'iowarp', got '{}'",
         ctx.dst);
    error_code = -1;
    CLIO_CO_RETURN;
  }
  std::string tag_name = GetUrlPath(ctx.dst);
  if (tag_name.empty()) {
    HLOG(kError, "ModelWeightsAssimilator: empty tag name in dst '{}'", ctx.dst);
    error_code = -2;
    CLIO_CO_RETURN;
  }

  // ----- Source file -----
  std::string src_path = GetUrlPath(ctx.src);
  if (src_path.empty()) {
    HLOG(kError, "ModelWeightsAssimilator: empty src path in '{}'", ctx.src);
    error_code = -4;
    CLIO_CO_RETURN;
  }
  size_t file_size = GetFileSize(src_path);
  if (file_size == 0) {
    HLOG(kError, "ModelWeightsAssimilator: cannot size file '{}'", src_path);
    error_code = -5;
    CLIO_CO_RETURN;
  }

  size_t range_off = ctx.range_off;
  size_t total_size = ctx.range_size > 0 ? ctx.range_size : (file_size - range_off);
  if (range_off + total_size > file_size) {
    HLOG(kError,
         "ModelWeightsAssimilator: range [{},{}) exceeds file size {}",
         range_off, range_off + total_size, file_size);
    error_code = -6;
    CLIO_CO_RETURN;
  }

  // ----- Geometry (must match the mapping Vector) -----
  const size_t kDefaultPage = 2ULL * 1024 * 1024;  // 2 MiB
  size_t page_size = ParseFormatParam(ctx.format, "page", kDefaultPage);
  size_t nblocks = ParseFormatParam(ctx.format, "nblocks", 1);
  if (page_size == 0) page_size = kDefaultPage;
  if (nblocks == 0) nblocks = 1;

  size_t num_pages = (total_size + page_size - 1) / page_size;
  size_t pages_per_block = (num_pages + nblocks - 1) / nblocks;
  if (pages_per_block == 0) pages_per_block = 1;

  HLOG(kInfo,
       "ModelWeightsAssimilator: tag='{}' total={}B page={}B nblocks={} "
       "num_pages={} pages/block={}",
       tag_name, total_size, page_size, nblocks, num_pages, pages_per_block);

  // ----- Get or create the tag -----
  auto tag_task = cte_client_->AsyncGetOrCreateTag(tag_name);
  CLIO_CO_AWAIT(tag_task);
  clio::cte::core::TagId tag_id = tag_task->tag_id_;
  if (tag_id.IsNull()) {
    HLOG(kError, "ModelWeightsAssimilator: failed to get/create tag '{}'", tag_name);
    error_code = -3;
    CLIO_CO_RETURN;
  }

  // ----- Manifest blob (geometry; ignored by the vector) -----
  {
    std::string manifest = "modelweights\n";
    manifest += "total_size=" + std::to_string(total_size) + "\n";
    manifest += "page_size=" + std::to_string(page_size) + "\n";
    manifest += "nblocks=" + std::to_string(nblocks) + "\n";
    manifest += "num_pages=" + std::to_string(num_pages) + "\n";
    auto mbuf = CLIO_IPC->AllocateBuffer(manifest.size());
    std::memcpy(mbuf.ptr_, manifest.data(), manifest.size());
    auto mtask = cte_client_->AsyncPutBlob(tag_id, "manifest", 0, manifest.size(),
                                           mbuf.shm_.template Cast<void>(), 1.0f,
                                           clio::cte::core::Context(), 0);
    CLIO_CO_AWAIT(mtask);
    if (mtask->return_code_ != 0) {
      HLOG(kError, "ModelWeightsAssimilator: manifest PutBlob failed rc={}",
           mtask->return_code_);
      CLIO_IPC->FreeBuffer(mtask->blob_data_.template Cast<char>());
      error_code = -9;
      CLIO_CO_RETURN;
    }
    CLIO_IPC->FreeBuffer(mtask->blob_data_.template Cast<char>());
  }

  // Optional: route the PAGE PutBlobs through a storage pool (the
  // cte_compressor entrypoint) so the weight pages are stored COMPRESSED, while
  // the tag + manifest above stay on the CTE core (cte_client_). The compressor
  // is a data-path-only forwarder — it has no tag ops — so tag creation must NOT
  // be routed through it (mirrors how gpu_vector::Vector always creates the tag
  // on kCtePoolId and only routes page traffic to its storage_pool_id).
  // CLIO_MW_STORAGE_POOL=<major> selects it; unset keeps pages on the core.
  clio::cte::core::Client *page_client = cte_client_.get();
  std::unique_ptr<clio::cte::core::Client> storage_client;
  // `pool=<major>` in the omni format string selects the storage pool in code
  // (e.g. "modelweights;page=2097152;nblocks=8;pool=600" routes pages through
  // the compressor pool the compose file created). The env var remains as a
  // fallback for operator experiments only.
  size_t pool_param = ParseFormatParam(ctx.format, "pool", 0);
  {
    unsigned long m = pool_param;
    if (m == 0) {
      if (const char *sp = std::getenv("CLIO_MW_STORAGE_POOL")) {
        m = std::strtoul(sp, nullptr, 10);
      }
    }
    if (m != 0) {
      storage_client = std::make_unique<clio::cte::core::Client>(
          clio::run::PoolId(static_cast<clio::run::u32>(m), 0));
      page_client = storage_client.get();
      HLOG(kInfo,
           "ModelWeightsAssimilator: routing page PutBlobs through storage "
           "pool {} (compressed store); tag+manifest stay on the core",
           m);
    }
  }

  // ----- Open and stream the weight file into page blobs -----
  std::ifstream file(src_path, std::ios::binary);
  if (!file.is_open()) {
    HLOG(kError, "ModelWeightsAssimilator: cannot open '{}'", src_path);
    error_code = -7;
    CLIO_CO_RETURN;
  }
  file.seekg(static_cast<std::streamoff>(range_off), std::ios::beg);
  if (!file.good()) {
    HLOG(kError, "ModelWeightsAssimilator: seek to {} failed in '{}'",
         range_off, src_path);
    error_code = -8;
    CLIO_CO_RETURN;
  }

  static constexpr size_t kMaxParallelTasks = 32;
  std::vector<clio::run::Future<clio::cte::core::PutBlobTask>> active_tasks;

  size_t bytes_remaining = total_size;
  for (size_t p = 0; p < num_pages; ++p) {
    size_t this_bytes = std::min(page_size, bytes_remaining);
    size_t block = p / pages_per_block;

    // Full-page buffer, zero-padded tail so every blob is exactly page_size.
    auto buf = CLIO_IPC->AllocateBuffer(page_size);
    std::memset(buf.ptr_, 0, page_size);
    file.read(buf.ptr_, static_cast<std::streamsize>(this_bytes));
    std::streamsize got = file.gcount();
    if (got != static_cast<std::streamsize>(this_bytes)) {
      HLOG(kError,
           "ModelWeightsAssimilator: short read on page {} (got {} want {})",
           p, static_cast<long long>(got), this_bytes);
      CLIO_IPC->FreeBuffer(buf);
      error_code = -10;
      CLIO_CO_RETURN;
    }

    // Name the vector's GetBlob will request: "b<block>_pi<page>".
    //
    // The tag is deliberately NOT part of this. Lookups are already scoped by
    // tag_id -- HashBlobToContainer(tag_id, blob_name) keys on both -- so
    // prefixing the semantic tag transported it twice. It also had to survive
    // inside PodGetBlobTask::blob_name_, a priv::string that must stay in its
    // small-string buffer because a Pod task cannot carry a heap allocation to
    // the handler. A 44-char tag was SILENTLY truncated to 31, so every device
    // fault missed with rc=1 and the weights stayed zero with no error
    // anywhere. Dropping the prefix removes the whole failure class.
    std::string blob_name = "b" + std::to_string(block) + "_pi" +
                            std::to_string(p);
    auto task = page_client->AsyncPutBlob(tag_id, blob_name, 0, page_size,
                                          buf.shm_.template Cast<void>(), 1.0f,
                                          clio::cte::core::Context(), 0);
    active_tasks.push_back(task);
    bytes_remaining -= this_bytes;

    // Drain when the in-flight window is full.
    if (active_tasks.size() >= kMaxParallelTasks) {
      auto& front = active_tasks.front();
      CLIO_CO_AWAIT(front);
      if (front->return_code_ != 0) {
        HLOG(kError, "ModelWeightsAssimilator: PutBlob failed rc={}",
             front->return_code_);
        CLIO_IPC->FreeBuffer(front->blob_data_.template Cast<char>());
        error_code = -11;
        CLIO_CO_RETURN;
      }
      CLIO_IPC->FreeBuffer(front->blob_data_.template Cast<char>());
      active_tasks.erase(active_tasks.begin());
    }
  }

  // Drain the remaining in-flight PutBlobs.
  for (auto& task : active_tasks) {
    CLIO_CO_AWAIT(task);
    if (task->return_code_ != 0) {
      HLOG(kError, "ModelWeightsAssimilator: PutBlob failed rc={}",
           task->return_code_);
      CLIO_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
      error_code = -11;
      CLIO_CO_RETURN;
    }
    CLIO_IPC->FreeBuffer(task->blob_data_.template Cast<char>());
  }

  file.close();
  HLOG(kInfo,
       "ModelWeightsAssimilator: assimilated '{}' -> tag '{}' ({} pages)",
       src_path, tag_name, num_pages);
  error_code = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

std::string ModelWeightsAssimilator::GetUrlProtocol(const std::string& url) {
  size_t pos = url.find("::");
  if (pos == std::string::npos) return "";
  return url.substr(0, pos);
}

std::string ModelWeightsAssimilator::GetUrlPath(const std::string& url) {
  size_t pos = url.find("::");
  if (pos == std::string::npos) return "";
  return url.substr(pos + 2);
}

size_t ModelWeightsAssimilator::GetFileSize(const std::string& file_path) {
  std::ifstream f(file_path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) return 0;
  std::streamoff sz = f.tellg();
  return sz > 0 ? static_cast<size_t>(sz) : 0;
}

}  // namespace clio::cae::core
