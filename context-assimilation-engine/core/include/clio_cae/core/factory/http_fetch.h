/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#ifndef CLIO_CAE_CORE_FACTORY_HTTP_FETCH_H_
#define CLIO_CAE_CORE_FACTORY_HTTP_FETCH_H_

// Download one http(s) URL to a local file, in parallel ranged GETs when the
// server supports ranges. The HTTP work is done by curl child processes, as in
// the Globus assimilator: libcurl or Poco inside the runtime process has
// crashed there, and curl honours the proxy variables (https_proxy, no_proxy)
// that Aurora compute nodes need.
//
// Credentials and the URL with its query.* params reach curl on its standard
// input (curl -K -), never on its command line, so `ps` on a shared node does
// not show them.
//
// A site whose config section says client = wget (http_auth.h) is fetched by
// wget instead, in one stream. wget gets its credentials and headers from a
// private (0600) temporary wgetrc (--config) and the URL from a private
// temporary file (-i), both removed when it exits; again nothing secret is on
// its command line.
//
// The engine has no runtime dependencies, so cae_http_tool and the tests can
// use it as is.

#include <clio_cae/core/factory/http_auth.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace clio::cae::core {

/** Tuning, read from the environment of the process that downloads. */
struct HttpFetchConfig {
  size_t streams = 8;                  ///< CAE_HTTP_STREAMS: parallel GETs (1-64)
  uint64_t part_bytes = 64ull << 20;   ///< CAE_HTTP_PART_MB: bytes per GET
  int part_retries = 5;                ///< CAE_HTTP_PART_RETRIES: per part
  int backoff_ms = 1000;               ///< CAE_HTTP_BACKOFF_MS: first retry wait
  bool overwrite = false;              ///< CAE_HTTP_OVERWRITE=1: no skip
  int progress_s = 30;                 ///< CAE_HTTP_PROGRESS_S: 0 = quiet
  std::string curl = "curl";           ///< CAE_HTTP_CURL: curl executable
  std::string wget = "wget";           ///< CAE_HTTP_WGET: wget executable

  static HttpFetchConfig FromEnv();
};

/** What the server said to a one-byte ranged GET. */
struct HttpProbe {
  int status = 0;            ///< final HTTP status after redirects (0: none)
  bool ranges = false;       ///< server honours Range
  int64_t total = -1;        ///< size of the whole resource (-1: unknown)
  std::string content_type;  ///< of the final response
  std::string error;         ///< curl failure, if any
};

/** One download. */
struct HttpFetchRequest {
  HttpAuth auth;            ///< URL (with query params), headers, basic auth
  uint64_t range_off = 0;   ///< first byte of the source to fetch
  uint64_t range_size = 0;  ///< bytes to fetch (0: to the end)
  std::string dst_path;     ///< local file to write
};

struct HttpFetchResult {
  bool ok = false;
  bool skipped = false;     ///< dst already held exactly these bytes
  uint64_t bytes = 0;       ///< bytes written
  double seconds = 0;
  int parts = 0;            ///< ranged GETs that made up the file
  int retries = 0;          ///< GETs repeated after a failure
  HttpProbe probe;
  std::string error;
};

using HttpLog = std::function<void(const std::string &)>;

/** Send a one-byte ranged GET and report status, size and range support. */
HttpProbe HttpProbeUrl(const HttpAuth &auth, const HttpFetchConfig &cfg);

/**
 * Download req.auth.url to req.dst_path. The bytes go to "<dst>.cae-part",
 * which is renamed to dst only once every byte has arrived, so dst is never a
 * partial file. Parent directories are created.
 */
HttpFetchResult HttpFetch(const HttpFetchRequest &req,
                          const HttpFetchConfig &cfg, const HttpLog &log);

namespace http_detail {

/** curl -K config lines for this request (url, headers, user), or "" + *err. */
std::string CurlConfig(const HttpAuth &auth, std::string *err);

/**
 * wgetrc lines for this request (headers, user, password), or "" + *err.
 *
 * @param auth resolved request; userpwd is split at its first ':'
 * @param err  set when a value cannot be expressed in a wgetrc (line breaks,
 *             or leading/trailing blanks, which wget would strip)
 * @return the wgetrc text; "" with *err set on failure. A request with no
 *         credentials or headers yields a comment line, never "".
 */
std::string WgetConfig(const HttpAuth &auth, std::string *err);

/**
 * Parse the header dump (curl -D -, or wget -S, whose lines are indented) of
 * a probe. The last response wins, so redirects and a 401 challenge before
 * the authenticated retry are skipped.
 */
HttpProbe ParseProbeHeaders(const std::string &dump);

}  // namespace http_detail

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_FACTORY_HTTP_FETCH_H_
