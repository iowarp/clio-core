/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include <clio_cae/summarizer/label_client.h>
#include <clio_ctp/util/config_parse.h>

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

#ifdef CLIO_CAE_ENABLE_SUMMARIZATION

#include <clio_ctp/util/logging.h>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

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

namespace {

/**
 * POST a JSON body and collect the response.
 *
 * The transport both protocols share: bounded timeouts, the loopback proxy
 * bypass, and optional extra headers (the OpenAI bearer token).
 *
 * @param caller     Function name, for log context.
 * @param url        Full request URL.
 * @param body       JSON request body.
 * @param extra_hdrs Additional "Name: value" headers.
 * @param out_body   Response body (filled even on a non-200 status).
 * @return true when the transport succeeded AND the status was 200.
 */
bool PostJson(const char *caller, const std::string &url,
              const std::string &body,
              const std::vector<std::string> &extra_hdrs,
              std::string &out_body) {
  EnsureCurlInit();
  CURL *curl = curl_easy_init();
  // LCOV_EXCL_START: curl_easy_init() only fails on libcurl-internal
  // out-of-memory / init faults that cannot be provoked from a test.
  if (!curl) {
    HLOG(kWarning, "{}: curl_easy_init failed", caller);
    return false;
  }
  // LCOV_EXCL_STOP

  struct curl_slist *headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  for (const auto &h : extra_hdrs) headers = curl_slist_append(headers, h.c_str());

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteToStringCb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out_body);
  // Bounded so a stuck inference server doesn't wedge a worker forever.
  // 120 s is enough for a small local model on CPU; tune at the deploy
  // layer if a larger model on cold weights needs more.
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  // Outlives curl_easy_perform below: libcurl keeps the pointer, not a copy.
  // Remote endpoints (ALCF) still go through the site's http(s)_proxy.
  const std::string no_proxy = detail::LoopbackNoProxyList();
  curl_easy_setopt(curl, CURLOPT_NOPROXY, no_proxy.c_str());

  CURLcode rc = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (rc != CURLE_OK) {
    HLOG(kWarning, "{}: transport error: {} ({})", caller,
         curl_easy_strerror(rc), static_cast<int>(rc));
    return false;
  }
  if (http_code != 200) {
    HLOG(kWarning, "{}: HTTP {} from {} body='{}'", caller, http_code, url,
         out_body.substr(0, 200));
    return false;
  }
  return true;
}

/**
 * Join a base URL and a path, dropping one trailing slash from the base.
 *
 * @param base Base URL as configured.
 * @param path Path beginning with '/'.
 * @return base + path.
 */
std::string JoinUrl(std::string base, const char *path) {
  if (!base.empty() && base.back() == '/') base.pop_back();
  return base + path;
}

}  // namespace

bool OllamaGenerate(const std::string &endpoint_base,
                    const std::string &model,
                    const std::string &prompt,
                    int context_length,
                    int num_predict,
                    std::string &out_response) {
  out_response.clear();
  if (endpoint_base.empty() || model.empty()) {
    HLOG(kWarning,
         "OllamaGenerate: missing endpoint or model (endpoint='{}' model='{}')",
         endpoint_base, model);
    return false;
  }

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

  std::string resp_buf;
  if (!PostJson("OllamaGenerate", JoinUrl(endpoint_base, "/api/generate"),
                body.dump(), {}, resp_buf)) {
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

bool OpenAIChatComplete(const std::string &endpoint_base,
                        const std::string &api_key,
                        const std::string &model,
                        const std::string &prompt,
                        int num_predict,
                        std::string &out_response) {
  out_response.clear();
  if (endpoint_base.empty() || model.empty()) {
    HLOG(kWarning,
         "OpenAIChatComplete: missing endpoint or model (endpoint='{}' "
         "model='{}')",
         endpoint_base, model);
    return false;
  }

  nlohmann::json body = {
      {"model", model},
      {"messages", nlohmann::json::array(
                       {{{"role", "user"}, {"content", prompt}}})},
      {"stream", false},
  };
  if (num_predict > 0) body["max_tokens"] = num_predict;

  std::vector<std::string> hdrs;
  if (!api_key.empty()) hdrs.push_back("Authorization: Bearer " + api_key);

  std::string resp_buf;
  if (!PostJson("OpenAIChatComplete",
                JoinUrl(endpoint_base, "/chat/completions"), body.dump(), hdrs,
                resp_buf)) {
    return false;
  }

  try {
    auto parsed = nlohmann::json::parse(resp_buf);
    const auto &choices = parsed.at("choices");
    if (!choices.is_array() || choices.empty()) {
      HLOG(kWarning, "OpenAIChatComplete: response has no choices");
      return false;
    }
    const auto &content = choices[0].at("message").at("content");
    // Some servers send null content for an empty completion.
    if (!content.is_string()) {
      HLOG(kWarning, "OpenAIChatComplete: message content is not a string");
      return false;
    }
    out_response = content.get<std::string>();
    return true;
  } catch (const std::exception &e) {
    HLOG(kWarning, "OpenAIChatComplete: unexpected response JSON: {}",
         e.what());
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

bool OpenAIChatComplete(const std::string &, const std::string &,
                        const std::string &, const std::string &, int,
                        std::string &out_response) {
  out_response.clear();
  return false;
}

namespace detail {
std::string LoopbackNoProxyList() { return {}; }
}  // namespace detail

}  // namespace clio::cae::summarizer

#endif  // CLIO_CAE_ENABLE_SUMMARIZATION

// Protocol selection and token resolution need neither libcurl nor JSON, so
// they build (and are unit-testable) whether or not summarization is enabled.
namespace clio::cae::summarizer {

bool ParseLabelApi(const std::string &name, LabelApi &out) {
  std::string lower;
  lower.reserve(name.size());
  for (char c : name) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (lower.empty() || lower == "ollama") {
    out = LabelApi::kOllama;
    return true;
  }
  if (lower == "openai") {
    out = LabelApi::kOpenAI;
    return true;
  }
  return false;
}

namespace {

/**
 * Strip leading and trailing whitespace.
 * @param s Input string.
 * @return s without surrounding spaces, tabs, CRs, or newlines.
 */
std::string Trim(const std::string &s) {
  const char *kWs = " \t\r\n";
  size_t first = s.find_first_not_of(kWs);
  if (first == std::string::npos) return {};
  size_t last = s.find_last_not_of(kWs);
  return s.substr(first, last - first + 1);
}

}  // namespace

std::string ResolveApiKey(const InferenceTarget &target) {
  // Each source returns its own token; a function-scope result would leave
  // only its exception cleanup on the closing brace, which gcov never counts.
  if (!target.api_key_file_.empty()) {
    // Compose pool configs are not env-expanded, so do it here: the
    // documented example is "${HOME}/.alcf_access_token".
    std::ifstream in(ctp::ConfigParse::ExpandPath(target.api_key_file_));
    if (in) {
      std::ostringstream ss;
      ss << in.rdbuf();
      std::string key = Trim(ss.str());
      if (!key.empty()) return key;
    }
  }
  // A missing, unreadable, or blank file falls through to the env var.
  if (!target.api_key_env_.empty()) {
    const char *env = std::getenv(target.api_key_env_.c_str());
    if (env != nullptr) return Trim(env);
  }
  return {};
}

bool LabelGenerate(const InferenceTarget &target, const std::string &model,
                   const std::string &prompt, int context_length,
                   int num_predict, std::string &out_response) {
  if (target.api_ == LabelApi::kOpenAI) {
    return OpenAIChatComplete(target.endpoint_, ResolveApiKey(target), model,
                              prompt, num_predict, out_response);
  }
  return OllamaGenerate(target.endpoint_, model, prompt, context_length,
                        num_predict, out_response);
}

}  // namespace clio::cae::summarizer
