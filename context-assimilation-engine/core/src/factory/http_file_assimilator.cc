/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/factory/http_auth.h>
#include <clio_cae/core/factory/http_fetch.h>
#include <clio_cae/core/factory/http_file_assimilator.h>

#include <string>

namespace clio::cae::core {

namespace {

/** The URL without its query string, for logs (query values may be secrets). */
std::string LogUrl(const std::string& url) {
  size_t q = url.find_first_of("?#");
  return q == std::string::npos ? url : url.substr(0, q) + "?...";
}

}  // namespace

HttpFileAssimilator::HttpFileAssimilator(
    std::shared_ptr<clio::cte::core::Client> cte_client)
    : cte_client_(std::move(cte_client)) {}

int HttpFileAssimilator::Run(const AssimilationCtx& ctx) {
  const std::string src_log = LogUrl(ctx.src);
  const size_t sep = ctx.dst.find("::");
  if (sep == std::string::npos || ctx.dst.substr(0, sep) != "file" ||
      ctx.dst.size() <= sep + 2) {
    HLOG(kError,
         "HttpFileAssimilator: destination must be file::<path> for an "
         "http(s) source, got '{}'",
         ctx.dst);
    return -3;
  }
  const std::string dst_path = ctx.dst.substr(sep + 2);

  HttpAuthResult a = ResolveHttpAuth(ctx.src, ctx.src_token);
  if (!a.ok) {
    HLOG(kError, "HttpFileAssimilator: {}: {}", src_log, a.error);
    return -1;
  }
  std::string from = a.auth.sections.empty() ? "no config section" : "";
  for (const auto& s : a.auth.sections) {
    from += (from.empty() ? "[" : ", [") + s + "]";
  }
  HLOG(kInfo,
       "HttpFileAssimilator: {} -> {} ({} query params, {} headers{} from {})",
       src_log, dst_path, a.auth.num_query, a.auth.headers.size(),
       a.auth.userpwd.empty() ? "" : ", basic auth", from);

  HttpFetchRequest req;
  req.auth = a.auth;
  req.range_off = ctx.range_off;
  req.range_size = ctx.range_size;
  req.dst_path = dst_path;
  const HttpFetchConfig cfg = HttpFetchConfig::FromEnv();
  HttpFetchResult r = HttpFetch(req, cfg, [&](const std::string& m) {
    HLOG(kInfo, "HttpFileAssimilator: {}: {}", dst_path, m);
  });
  if (!r.ok) {
    HLOG(kError, "HttpFileAssimilator: {} -> {} FAILED: {}", src_log, dst_path,
         r.error);
    return r.probe.status >= 400 ? -r.probe.status : -13;
  }
  if (r.skipped) {
    HLOG(kInfo,
         "HttpFileAssimilator: {} already holds the {} bytes of {}; skipped "
         "(CAE_HTTP_OVERWRITE=1 to fetch again)",
         dst_path, r.bytes, src_log);
    return 0;
  }
  const double mbs = r.seconds > 0 ? r.bytes / 1e6 / r.seconds : 0;
  HLOG(kInfo,
       "HttpFileAssimilator: {} done: {} bytes in {} s ({} MB/s, {} parts, "
       "{} retries)",
       dst_path, r.bytes, static_cast<long long>(r.seconds),
       static_cast<long long>(mbs), r.parts, r.retries);
  return 0;
}

clio::run::TaskResume HttpFileAssimilator::Schedule(const AssimilationCtx& ctx,
                                                    int& error_code) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  // Blocking, like the Globus file download: the transfer runs on curl
  // children and helper threads while this worker waits for it.
  error_code = Run(ctx);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cae::core
