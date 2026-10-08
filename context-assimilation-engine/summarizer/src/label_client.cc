/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include <clio_cae/summarizer/label_client.h>

#ifdef CLIO_CAE_ENABLE_SUMMARIZATION

#include <clio_ctp/util/logging.h>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <mutex>
#include <string>

namespace clio::cae::summarizer {

namespace {

// libcurl needs global init exactly once before any easy handle is used.
// Guard it with std::once so concurrent first-callers don't race.
void EnsureCurlInit() {
  static std::once_flag once;
  std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

size_t WriteToStringCb(char *ptr, size_t size, size_t nmemb, void *userdata) {
  auto *buf = static_cast<std::string *>(userdata);
  buf->append(ptr, size * nmemb);
  return size * nmemb;
}

}  // namespace

namespace detail {

// Hosts that must always be reached directly, never through an HTTP proxy.
//
// Ollama normally listens on the same host as the summarizer, so the endpoint
// is a loopback address. A site that exports http_proxy -- HPC login and
// compute nodes routinely do, to let outbound traffic reach the internet --
// otherwise sends even a 127.0.0.1 request to the proxy, which refuses to
// relay to loopback and answers with its own error page. Every label then
// comes back empty behind an HTTP 503 that has nothing to do with the model.
//
// CURLOPT_NOPROXY REPLACES libcurl's own reading of no_proxy rather than
// adding to it, so carry whatever the environment already asked to bypass:
// a site that routes some internal host directly must keep working.
std::string LoopbackNoProxyList() {
  const char *env = std::getenv("no_proxy");
  if (env == nullptr || *env == '\0') env = std::getenv("NO_PROXY");
  std::string list = "localhost,127.0.0.1,::1";
  if (env != nullptr && *env != '\0') {
    list += ",";
    list += env;
  }
  return list;
}

}  // namespace detail

bool OllamaGenerate(const std::string &endpoint_base,
                    const std::string &model,
                    const std::string &prompt,
                    int context_length,
                    int num_predict,
                    std::string &out_response) {
  out_response.clear();
  EnsureCurlInit();

  if (endpoint_base.empty() || model.empty()) {
    HLOG(kWarning,
         "OllamaGenerate: missing endpoint or model (endpoint='{}' model='{}')",
         endpoint_base, model);
    return false;
  }

  std::string url = endpoint_base;
  if (!url.empty() && url.back() == '/') url.pop_back();
  url += "/api/generate";

  nlohmann::json body = {
      {"model", model},
      {"prompt", prompt},
      {"stream", false},
  };
  // num_ctx widens Ollama's prompt+response budget. Its default (~2048)
  // silently truncates anything larger, which manifests as a label that
  // only describes the *tail* of the blob. num_predict hard-caps the
  // *output* length, useful for benchmarks where summary length is the
  // controlled variable.
  if (context_length > 0 || num_predict > 0) {
    nlohmann::json opts = nlohmann::json::object();
    if (context_length > 0) opts["num_ctx"] = context_length;
    if (num_predict > 0) opts["num_predict"] = num_predict;
    body["options"] = std::move(opts);
  }
  std::string body_str = body.dump();

  CURL *curl = curl_easy_init();
  // LCOV_EXCL_START: curl_easy_init() only fails on libcurl-internal
  // out-of-memory / init faults that cannot be provoked from a test.
  if (!curl) {
    HLOG(kWarning, "OllamaGenerate: curl_easy_init failed");
    return false;
  }
  // LCOV_EXCL_STOP

  std::string resp_buf;
  struct curl_slist *headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE,
                   static_cast<long>(body_str.size()));
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteToStringCb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp_buf);
  // Bounded so a stuck inference server doesn't wedge a worker forever.
  // 120 s is enough for a small local model on CPU; tune at the deploy
  // layer if a larger model on cold weights needs more.
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  // Outlives curl_easy_perform below: libcurl keeps the pointer, not a copy.
  const std::string no_proxy = detail::LoopbackNoProxyList();
  curl_easy_setopt(curl, CURLOPT_NOPROXY, no_proxy.c_str());

  CURLcode rc = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (rc != CURLE_OK) {
    HLOG(kWarning, "OllamaGenerate: transport error: {} ({})",
         curl_easy_strerror(rc), static_cast<int>(rc));
    return false;
  }
  if (http_code != 200) {
    HLOG(kWarning, "OllamaGenerate: HTTP {} from {} body='{}'", http_code, url,
         resp_buf.substr(0, 200));
    return false;
  }

  try {
    auto parsed = nlohmann::json::parse(resp_buf);
    if (!parsed.contains("response")) {
      HLOG(kWarning, "OllamaGenerate: response JSON missing 'response' field");
      return false;
    }
    out_response = parsed["response"].get<std::string>();
    return true;
  } catch (const std::exception &e) {
    HLOG(kWarning, "OllamaGenerate: JSON parse failed: {}", e.what());
    return false;
  }
}

}  // namespace clio::cae::summarizer

#else  // CLIO_CAE_ENABLE_SUMMARIZATION

namespace clio::cae::summarizer {

bool OllamaGenerate(const std::string &, const std::string &,
                    const std::string &, int, int, std::string &out_response) {
  out_response.clear();
  return false;
}

namespace detail {
std::string LoopbackNoProxyList() { return {}; }
}  // namespace detail

}  // namespace clio::cae::summarizer

#endif  // CLIO_CAE_ENABLE_SUMMARIZATION
