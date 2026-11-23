/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/factory/binary_file_assimilator.h>
#include <clio_cae/core/factory/dataverse.h>
#include <clio_cae/core/factory/dataverse_file_assimilator.h>
#include <clio_cae/core/factory/path_policy.h>
#ifdef CLIO_CAE_ENABLE_HDF5
#include <clio_cae/core/factory/hdf5_file_assimilator.h>
#endif

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// clio_cte headers after the clio_cae ones (Method namespace collision; see
// binary_file_assimilator.cc).
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

namespace clio::cae::core {

namespace {

bool EndsWith(const std::string &s, const char *suf) {
  const size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

[[maybe_unused]] bool IsHdf5Path(const std::string &p) {
  return EndsWith(p, ".h5") || EndsWith(p, ".hdf5") || EndsWith(p, ".he5") ||
         EndsWith(p, ".hdf");
}

}  // namespace

DataverseFileAssimilator::DataverseFileAssimilator(
    std::shared_ptr<clio::cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

clio::run::TaskResume DataverseFileAssimilator::Schedule(
    const AssimilationCtx &ctx, int &error_code) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  DataverseSource source;
  std::string err;
  std::string stage_dir;
  std::string tag_prefix;
  bool to_cte = false;
  DataverseStageResult staged;
  size_t ingested = 0;
  size_t skipped_empty = 0;
  std::string manifest;
  std::vector<std::string> obj_inc, obj_exc, h5_inc, h5_exc;

  if (!ParseDataverseUrl(ctx.src, &source, &err)) {
    HLOG(kError, "DataverseFileAssimilator: {}", err);
    error_code = -1;
    CLIO_CO_RETURN;
  }
  {
    const size_t sep = ctx.dst.find("::");
    const std::string proto = sep == std::string::npos ? "" : ctx.dst.substr(0, sep);
    const std::string path = sep == std::string::npos ? "" : ctx.dst.substr(sep + 2);
    if (proto == "file" && !path.empty()) {
      std::string reason;
      if (!PathPolicy::Get().Allows(path, /*write=*/true, reason)) {
        HLOG(kError, "DataverseFileAssimilator: refusing destination '{}': {}",
             ctx.dst, reason);
        error_code = -3;
        CLIO_CO_RETURN;
      }
      stage_dir = path;
    } else if (proto == "iowarp" && !path.empty()) {
      to_cte = true;
      tag_prefix = path;
      stage_dir = DataverseCacheDir(source);
    } else {
      HLOG(kError,
           "DataverseFileAssimilator: destination must be file::<dir> or "
           "iowarp::<tag>, got '{}'",
           ctx.dst);
      error_code = -3;
      CLIO_CO_RETURN;
    }
  }

  // Blocking, like the http(s) download: the transfers run on curl children
  // and helper threads while this worker waits.
  HLOG(kInfo, "DataverseFileAssimilator: {} -> {} (staging in {})", ctx.src,
       ctx.dst, stage_dir);
  // dataset_filter patterns with an "hdf5:" prefix select datasets inside the
  // HDF5 files (handed to the HDF5 assimilator); the rest select objects.
  for (const auto &p : ctx.include_patterns)
    (p.compare(0, 5, "hdf5:") == 0 ? h5_inc : obj_inc)
        .push_back(p.compare(0, 5, "hdf5:") == 0 ? p.substr(5) : p);
  for (const auto &p : ctx.exclude_patterns)
    (p.compare(0, 5, "hdf5:") == 0 ? h5_exc : obj_exc)
        .push_back(p.compare(0, 5, "hdf5:") == 0 ? p.substr(5) : p);
  staged = DataverseStage(ctx.src, obj_inc, obj_exc,
                          stage_dir, ctx.src_token, /*list_only=*/false,
                          [](const std::string &m) {
                            HLOG(kInfo, "DataverseFileAssimilator: {}", m);
                          });
  if (!staged.ok) {
    HLOG(kError, "DataverseFileAssimilator: {} FAILED: {}", ctx.src,
         staged.error);
    error_code = -13;
    CLIO_CO_RETURN;
  }
  HLOG(kInfo,
       "DataverseFileAssimilator: staged {} objects ({} bytes) with {} HTTP "
       "requests, {} bytes received, in {} s",
       staged.objects.size(), staged.bytes, staged.requests, staged.fetched,
       static_cast<long long>(staged.seconds));
  if (!to_cte) {
    error_code = 0;
    CLIO_CO_RETURN;
  }

  // Ingest each staged object into <tag_prefix>/<path>.
  for (size_t i = 0; i < staged.objects.size(); ++i) {
    const DataverseObject &o = staged.objects[i];
    if (o.size == 0) {  // the binary assimilator rejects empty files
      ++skipped_empty;
      continue;
    }
    AssimilationCtx child;
    child.dst = "iowarp::" + tag_prefix + "/" + o.path;
    child.format = "binary";
    child.chunk_size = ctx.chunk_size;
    child.blob_prefix = ctx.blob_prefix;
    int rc = 0;
#ifdef CLIO_CAE_ENABLE_HDF5
    if (ctx.format == "hdf5" && IsHdf5Path(o.path)) {
      child.src = "hdf5::" + o.local;
      child.format = "hdf5";
      child.include_patterns = h5_inc;
      child.exclude_patterns = h5_exc;
      Hdf5FileAssimilator h5(cte_client_);
      CLIO_CO_AWAIT(h5.Schedule(child, rc));
    } else
#endif
    {
      child.src = "file::" + o.local;
      BinaryFileAssimilator bin(cte_client_);
      CLIO_CO_AWAIT(bin.Schedule(child, rc));
    }
    if (rc != 0) {
      HLOG(kError, "DataverseFileAssimilator: ingest of {} failed ({})",
           o.path, rc);
      error_code = -20;
      CLIO_CO_RETURN;
    }
    ++ingested;
  }
#ifndef CLIO_CAE_ENABLE_HDF5
  if (ctx.format == "hdf5") {
    HLOG(kWarning,
         "DataverseFileAssimilator: format hdf5 asked for, but HDF5 support is "
         "not compiled in; HDF5 files were ingested as binary");
  }
#endif

  // The manifest blob: what was ingested, for a client that starts from the
  // prefix tag.
  {
    std::ostringstream m;
    m << "# " << ctx.src << " (" << staged.version << ")\n"
      << "# path\tsize\tfile_id\tmember\ttag\n";
    for (const auto &o : staged.objects) {
      m << o.path << '\t' << o.size << '\t' << o.file_id << '\t' << o.member
        << '\t' << tag_prefix << '/' << o.path << '\n';
    }
    manifest = m.str();
  }
  {
    auto tag_task = cte_client_->AsyncGetOrCreateTag(tag_prefix);
    CLIO_CO_AWAIT(tag_task);
    clio::cte::core::TagId tag_id = tag_task->tag_id_;
    if (tag_id.IsNull()) {
      HLOG(kError, "DataverseFileAssimilator: cannot create tag '{}'",
           tag_prefix);
      error_code = -21;
      CLIO_CO_RETURN;
    }
    auto buf = CLIO_IPC->AllocateBuffer(manifest.size());
    std::memcpy(buf.ptr_, manifest.data(), manifest.size());
    auto put = cte_client_->AsyncPutBlob(tag_id, "manifest", 0, manifest.size(),
                                         buf.shm_.template Cast<void>(), 1.0f,
                                         clio::cte::core::Context(), 0);
    CLIO_CO_AWAIT(put);
    const int put_rc = static_cast<int>(put->return_code_);
    CLIO_IPC->FreeBuffer(buf);
    if (put_rc != 0) {
      HLOG(kError, "DataverseFileAssimilator: manifest put failed ({})",
           put_rc);
      error_code = -22;
      CLIO_CO_RETURN;
    }
  }
  HLOG(kInfo,
       "DataverseFileAssimilator: ingested {} objects under '{}' ({} empty "
       "files skipped)",
       ingested, tag_prefix, skipped_empty);
  error_code = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cae::core
