/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

// cae_http_tool: what HttpFileAssimilator does for one transfer, without a
// runtime. Same config file (~/.clio/cae_http_auth.ini or $CAE_HTTP_AUTH_FILE)
// and the same CAE_HTTP_* knobs, so a site's settings can be checked before
// an OMNI run.
//
//   cae_http_tool probe <url>                   status, size, range support
//   cae_http_tool get <url> <dst> [off [size]]  download (part of) <url>

#include <clio_cae/core/factory/http_auth.h>
#include <clio_cae/core/factory/http_fetch.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace cae = clio::cae::core;

namespace {

int Usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s probe <url>\n"
               "       %s get <url> <dst_path> [range_off [range_size]]\n"
               "config: %s\n",
               argv0, argv0, cae::HttpAuthFilePath().c_str());
  return 2;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) return Usage(argv[0]);
  const std::string cmd = argv[1];
  const std::string url = argv[2];
  const char *tok = std::getenv("CAE_HTTP_TOKEN");  // stands in for src_token
  cae::HttpAuthResult a = cae::ResolveHttpAuth(url, tok ? tok : "");
  if (!a.ok) {
    std::fprintf(stderr, "error: %s\n", a.error.c_str());
    return 1;
  }
  std::string from;
  for (const auto &s : a.auth.sections) from += " [" + s + "]";
  std::fprintf(stderr, "config: %s%s; %zu query params, %zu headers%s\n",
               cae::HttpAuthFilePath().c_str(),
               from.empty() ? " (no section matched)" : from.c_str(),
               a.auth.num_query, a.auth.headers.size(),
               a.auth.userpwd.empty() ? "" : ", basic auth");
  const cae::HttpFetchConfig cfg = cae::HttpFetchConfig::FromEnv();

  if (cmd == "probe" && argc == 3) {
    cae::HttpProbe p = cae::HttpProbeUrl(a.auth, cfg);
    std::printf("status=%d ranges=%s size=%lld type=%s%s%s\n", p.status,
                p.ranges ? "yes" : "no", static_cast<long long>(p.total),
                p.content_type.c_str(), p.error.empty() ? "" : " error=",
                p.error.c_str());
    return p.error.empty() && p.status > 0 && p.status < 400 ? 0 : 1;
  }
  if (cmd == "get" && argc >= 4 && argc <= 6) {
    cae::HttpFetchRequest req;
    req.auth = a.auth;
    req.dst_path = argv[3];
    if (argc >= 5) req.range_off = std::strtoull(argv[4], nullptr, 10);
    if (argc >= 6) req.range_size = std::strtoull(argv[5], nullptr, 10);
    cae::HttpFetchResult r = cae::HttpFetch(
        req, cfg, [](const std::string &m) { std::fprintf(stderr, "%s\n", m.c_str()); });
    if (!r.ok) {
      std::fprintf(stderr, "error: %s\n", r.error.c_str());
      return 1;
    }
    std::printf("%s %llu bytes in %.1f s (%.1f MB/s, %d parts, %d retries)%s\n",
                req.dst_path.c_str(), static_cast<unsigned long long>(r.bytes),
                r.seconds, r.seconds > 0 ? r.bytes / 1e6 / r.seconds : 0.0,
                r.parts, r.retries, r.skipped ? " [already present, skipped]" : "");
    return 0;
  }
  return Usage(argv[0]);
}
