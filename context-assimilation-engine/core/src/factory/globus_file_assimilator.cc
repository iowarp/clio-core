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
#include <clio_ctp/util/logging.h>
#include <clio_cae/core/factory/globus_file_assimilator.h>

#include <chrono>
#include <csignal>
#include <cstdlib>

#ifdef CAE_ENABLE_GLOBUS
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <sstream>
#endif

// Include clio_cte headers after closing any clio_cae namespace to avoid Method
// namespace collision
#include <clio_cte/core/core_client.h>

namespace clio::cae::core {

GlobusFileAssimilator::GlobusFileAssimilator(
    std::shared_ptr<clio::cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

clio::run::TaskResume GlobusFileAssimilator::Schedule(const AssimilationCtx& ctx,
                                                int& error_code) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
#ifndef CAE_ENABLE_GLOBUS
  HLOG(kError, "GlobusFileAssimilator: Globus support not compiled in");
  error_code = -20;
  CLIO_CO_RETURN;
#else
  error_code = 0;
  // Validate source is a Globus URL (either web URL or globus:// URI)
  bool is_globus_web_url = (ctx.src.find("https://app.globus.org") == 0);
  bool is_globus_uri = (ctx.src.find("globus://") == 0);

  if (!is_globus_web_url && !is_globus_uri) {
    HLOG(kError,
         "GlobusFileAssimilator: Source must be a Globus web URL or globus:// "
         "URI, got: '{}'",
         ctx.src);
    error_code = -2;
    CLIO_CO_RETURN;
  }

  // Validate destination protocol
  bool is_dst_globus_web_url = (ctx.dst.find("https://app.globus.org") == 0);
  bool is_dst_globus_uri = (ctx.dst.find("globus://") == 0);
  std::string dst_protocol = GetUrlProtocol(ctx.dst);  // for file:: format
  bool is_valid_dst = (dst_protocol == "file" || is_dst_globus_uri ||
                       is_dst_globus_web_url);

  if (!is_valid_dst) {
    HLOG(kError,
         "GlobusFileAssimilator: Destination must be file::, globus://, or "
         "Globus web URL, got: '{}'",
         ctx.dst);
    error_code = -3;
    CLIO_CO_RETURN;
  }

  // Get access token from context or environment variable
  std::string access_token;
  if (!ctx.src_token.empty()) {
    access_token = ctx.src_token;
    HLOG(kDebug, "GlobusFileAssimilator: Using access token from src_token");
  } else {
    const char* access_token_env = std::getenv("GLOBUS_ACCESS_TOKEN");
    if (!access_token_env || std::strlen(access_token_env) == 0) {
      HLOG(kError,
           "GlobusFileAssimilator: No access token provided. Set src_token in "
           "OMNI file or GLOBUS_ACCESS_TOKEN environment variable");
      error_code = -1;
      CLIO_CO_RETURN;
    }
    access_token = access_token_env;
    HLOG(kDebug,
         "GlobusFileAssimilator: Using access token from GLOBUS_ACCESS_TOKEN "
         "environment variable");
  }

  // Parse source URI (supports both globus:// URIs and https://app.globus.org
  // URLs)
  std::string src_endpoint;
  std::string src_path;

  // Check if this is a Globus web URL
  if (ctx.src.find("https://app.globus.org") == 0) {
    if (!ParseGlobusWebUrl(ctx.src, src_endpoint, src_path)) {
      HLOG(kError,
           "GlobusFileAssimilator: Failed to parse Globus web URL: '{}'",
           ctx.src);
      error_code = -4;
      CLIO_CO_RETURN;
    }
  } else {
    // Parse as standard globus:// URI
    if (!ParseGlobusUri(ctx.src, src_endpoint, src_path)) {
      HLOG(kError, "GlobusFileAssimilator: Failed to parse source URI: '{}'",
           ctx.src);
      error_code = -4;
      CLIO_CO_RETURN;
    }
  }

  HLOG(kDebug, "GlobusFileAssimilator: Source endpoint='{}', path='{}'",
       src_endpoint, src_path);

  // Handle different destination types
  if (dst_protocol == "file") {
    // Globus to local filesystem
    HLOG(kInfo, "=========================================");
    HLOG(kInfo, "Globus to Local Filesystem Transfer");
    HLOG(kInfo, "=========================================");
    HLOG(kDebug,
         "GlobusFileAssimilator: Transferring from Globus to local filesystem");

    std::string dst_path = GetUrlPath(ctx.dst);
    if (dst_path.empty()) {
      HLOG(
          kError,
          "GlobusFileAssimilator: Invalid destination URL, no file path found");
      error_code = -5;
      CLIO_CO_RETURN;
    }

    HLOG(kInfo, "Source:       {}", ctx.src);
    HLOG(kInfo, "Destination:  {}", ctx.dst);

    // HTTPS downloads require the collection-specific HTTPS token,
    // not the Transfer API token. Check dst_token, then
    // GLOBUS_HTTPS_ACCESS_TOKEN env var, then fall back to access_token.
    std::string https_token;
    if (!ctx.dst_token.empty()) {
      https_token = ctx.dst_token;
      HLOG(kDebug, "GlobusFileAssimilator: Using HTTPS token from dst_token");
    } else {
      const char* https_env = std::getenv("GLOBUS_HTTPS_ACCESS_TOKEN");
      if (https_env && std::strlen(https_env) > 0) {
        https_token = https_env;
        HLOG(kDebug, "GlobusFileAssimilator: Using HTTPS token from "
             "GLOBUS_HTTPS_ACCESS_TOKEN environment variable");
      } else {
        https_token = access_token;
        HLOG(kDebug, "GlobusFileAssimilator: No collection HTTPS token found, "
             "falling back to transfer API token");
      }
    }

    // Download file from Globus to local filesystem.
    // Note: this blocks the clio scheduler worker, but ZMQ I/O threads
    // remain active so the runtime's IPC port stays alive for heartbeats.
    // access_token = Transfer API token (for endpoint metadata)
    // https_token = Collection HTTPS token (for file download)
    signal(SIGPIPE, SIG_IGN);
    int download_result = DownloadFile(src_endpoint, src_path, dst_path,
                                       access_token, https_token);

    if (download_result != 0) {
      HLOG(kError,
           "GlobusFileAssimilator: Failed to download file from Globus (error code: {})",
           download_result);
      error_code = download_result;
      CLIO_CO_RETURN;
    }

    HLOG(kInfo, "Transfer completed successfully!");
    HLOG(kDebug,
         "GlobusFileAssimilator: Successfully downloaded file to local "
         "filesystem");
    CLIO_CO_RETURN;

  } else {
    // Globus to Globus transfer
    HLOG(kDebug, "GlobusFileAssimilator: Initiating Globus-to-Globus transfer");

    // Parse destination URI (supports both globus:// URIs and
    // https://app.globus.org URLs)
    std::string dst_endpoint;
    std::string dst_path;

    // Check if this is a Globus web URL
    if (ctx.dst.find("https://app.globus.org") == 0) {
      if (!ParseGlobusWebUrl(ctx.dst, dst_endpoint, dst_path)) {
        HLOG(kError,
             "GlobusFileAssimilator: Failed to parse destination Globus web "
             "URL: '{}'",
             ctx.dst);
        error_code = -5;
        CLIO_CO_RETURN;
      }
    } else {
      // Parse as standard globus:// URI
      if (!ParseGlobusUri(ctx.dst, dst_endpoint, dst_path)) {
        HLOG(kError,
             "GlobusFileAssimilator: Failed to parse destination URI: '{}'",
             ctx.dst);
        error_code = -5;
        CLIO_CO_RETURN;
      }
    }

    HLOG(kDebug, "GlobusFileAssimilator: Destination endpoint='{}', path='{}'",
         dst_endpoint, dst_path);

    // Get submission ID
    std::string submission_id = GetSubmissionId(access_token);
    if (submission_id.empty()) {
      HLOG(
          kError,
          "GlobusFileAssimilator: Failed to get submission ID from Globus API");
      error_code = -6;
      CLIO_CO_RETURN;
    }

    HLOG(kDebug, "GlobusFileAssimilator: Obtained submission ID: '{}'",
         submission_id);

    // Submit transfer
    std::string task_id = SubmitTransfer(src_endpoint, dst_endpoint, src_path,
                                         dst_path, access_token, submission_id);
    if (task_id.empty()) {
      HLOG(kError,
           "GlobusFileAssimilator: Failed to submit transfer to Globus API");
      error_code = -7;
      CLIO_CO_RETURN;
    }

    HLOG(
        kDebug,
        "GlobusFileAssimilator: Transfer submitted successfully, task ID: '{}'",
        task_id);

    // Poll for transfer completion
    int poll_result = PollTransferStatus(task_id, access_token);
    if (poll_result != 0) {
      HLOG(kError, "GlobusFileAssimilator: Transfer failed or timed out");
      error_code = poll_result;
      CLIO_CO_RETURN;
    }

    HLOG(kDebug, "GlobusFileAssimilator: Transfer completed successfully");
    CLIO_CO_RETURN;
  }
#endif
  CLIO_TASK_BODY_END
}

std::string GlobusFileAssimilator::GetUrlProtocol(const std::string& url) {
  size_t pos = url.find("::");
  if (pos == std::string::npos) {
    return "";
  }
  return url.substr(0, pos);
}

std::string GlobusFileAssimilator::GetUrlPath(const std::string& url) {
  size_t pos = url.find("::");
  if (pos == std::string::npos) {
    return "";
  }
  return url.substr(pos + 2);
}

bool GlobusFileAssimilator::ParseGlobusUri(const std::string& uri,
                                           std::string& endpoint_id,
                                           std::string& path) {
  // Check for globus:// prefix
  const std::string prefix = "globus://";
  if (uri.find(prefix) != 0) {
    return false;
  }

  // Remove the "globus://" prefix
  const size_t scheme_len = prefix.length();
  const std::string uri_part = uri.substr(scheme_len);

  // Find the first slash after the endpoint ID
  size_t first_slash = uri_part.find('/');

  if (first_slash == 0) {
    // Handle case where there's a leading slash after globus://
    // e.g., globus:///~/path/to/file (empty endpoint)
    endpoint_id = "";
    path = uri_part;
  } else if (first_slash == std::string::npos) {
    // No path specified, only endpoint ID
    // e.g., globus://endpoint-id
    endpoint_id = uri_part;
    path = "/";
  } else {
    // Standard case: globus://endpoint-id/path/to/file
    endpoint_id = uri_part.substr(0, first_slash);
    path = uri_part.substr(first_slash);
  }

  // Validate endpoint ID (should not be empty)
  if (endpoint_id.empty()) {
    return false;
  }

  return true;
}

bool GlobusFileAssimilator::ParseGlobusWebUrl(const std::string& url,
                                              std::string& endpoint_id,
                                              std::string& path) {
  // Parse Globus web URLs like:
  // https://app.globus.org/file-manager?origin_id=ENDPOINT_ID&origin_path=%2Fpath%2Fto%2Ffile

  // Find origin_id parameter
  size_t origin_id_pos = url.find("origin_id=");
  if (origin_id_pos == std::string::npos) {
    HLOG(kError, "GlobusFileAssimilator: No origin_id found in Globus URL");
    return false;
  }

  // Extract endpoint ID (everything between origin_id= and next & or end of
  // string)
  size_t id_start = origin_id_pos + 10;  // length of "origin_id="
  size_t id_end = url.find('&', id_start);
  if (id_end == std::string::npos) {
    endpoint_id = url.substr(id_start);
  } else {
    endpoint_id = url.substr(id_start, id_end - id_start);
  }

  // Find origin_path parameter
  size_t origin_path_pos = url.find("origin_path=");
  if (origin_path_pos == std::string::npos) {
    // No path specified, default to root
    path = "/";
    HLOG(kDebug,
         "GlobusFileAssimilator: No origin_path in URL, using root '/'");
  } else {
    // Extract path (everything between origin_path= and next & or end of
    // string)
    size_t path_start = origin_path_pos + 12;  // length of "origin_path="
    size_t path_end = url.find('&', path_start);
    std::string encoded_path;
    if (path_end == std::string::npos) {
      encoded_path = url.substr(path_start);
    } else {
      encoded_path = url.substr(path_start, path_end - path_start);
    }

    // URL decode the path
    path = UrlDecode(encoded_path);
  }

  HLOG(
      kDebug,
      "GlobusFileAssimilator: Parsed Globus web URL - endpoint='{}', path='{}'",
      endpoint_id, path);

  return !endpoint_id.empty();
}

std::string GlobusFileAssimilator::UrlDecode(const std::string& encoded) {
  std::string decoded;
  decoded.reserve(encoded.length());

  for (size_t i = 0; i < encoded.length(); ++i) {
    if (encoded[i] == '%' && i + 2 < encoded.length()) {
      // Convert hex to char
      std::string hex = encoded.substr(i + 1, 2);
      char ch = static_cast<char>(std::strtol(hex.c_str(), nullptr, 16));
      decoded += ch;
      i += 2;
    } else if (encoded[i] == '+') {
      decoded += ' ';
    } else {
      decoded += encoded[i];
    }
  }

  return decoded;
}

#ifdef CAE_ENABLE_GLOBUS

// Percent-encode a URL path component: encode everything except unreserved
// characters and the path separator '/'.
static std::string UrlEncodePath(const std::string& path) {
  static const char hex[] = "0123456789ABCDEF";
  std::string out;
  out.reserve(path.size());
  for (unsigned char c : path) {
    if (std::isalnum(c) || c == '/' || c == '-' || c == '_' || c == '.' ||
        c == '~') {
      out += static_cast<char>(c);
    } else {
      out += '%';
      out += hex[(c >> 4) & 0xF];
      out += hex[c & 0xF];
    }
  }
  return out;
}

// Fork + exec curl to perform HTTP requests.  This completely avoids the
// glibc NSS SIGSEGV that occurs when getaddrinfo() is called from inside a
// clio worker-thread context (dlopen'd module + ctp allocator + NSS
// lazy-init = null nss_action_list → segfault at address 0x2).
//
// RunCurlCapture: forks curl and returns stdout as a string.
//   Returns "" on any curl error or non-zero exit status.
// RunCurlExec: forks curl and returns its exit code (0 = success).
//   Used when the output is written to a file via -o.

static std::string RunCurlCapture(const std::vector<std::string>& args) {
  int pipefd[2];
  if (pipe(pipefd) == -1) {
    return "";
  }

  pid_t pid = fork();
  if (pid == -1) {
    close(pipefd[0]);
    close(pipefd[1]);
    return "";
  }

  if (pid == 0) {
    // Child: wire stdout → pipe write-end; let stderr through (visible in
    // runtime logs via the parent's stderr).
    close(pipefd[0]);
    if (dup2(pipefd[1], STDOUT_FILENO) == -1) { _exit(1); }
    close(pipefd[1]);

    std::vector<const char*> argv;
    argv.push_back("curl");
    for (const auto& arg : args) {
      argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);
    execvp("curl", const_cast<char* const*>(argv.data()));
    _exit(1);
  }

  // Parent: drain the read end.
  close(pipefd[1]);
  std::string result;
  char buf[8192];
  ssize_t n;
  while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0) {
    result.append(buf, static_cast<size_t>(n));
  }
  close(pipefd[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    return "";
  }
  return result;
}

static int RunCurlExec(const std::vector<std::string>& args) {
  pid_t pid = fork();
  if (pid == -1) { return -1; }

  if (pid == 0) {
    std::vector<const char*> argv;
    argv.push_back("curl");
    for (const auto& arg : args) {
      argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);
    execvp("curl", const_cast<char* const*>(argv.data()));
    _exit(1);
  }

  int status = 0;
  waitpid(pid, &status, 0);
  if (WIFEXITED(status)) { return WEXITSTATUS(status); }
  return -1;
}

std::string GlobusFileAssimilator::HttpGet(const std::string& url,
                                           const std::string& access_token) {
  // Use fork/exec curl to avoid NSS crash in clio worker-thread context.
  // -s: silent  -L: follow redirects  -k: skip cert verify  --fail: non-200
  // returns non-zero exit code (RunCurlCapture returns "" in that case)
  std::string auth = "Authorization: Bearer " + access_token;
  std::string result = RunCurlCapture({
      "-s", "-L", "-k", "--fail",
      "-H", auth,
      "-H", "Accept: application/json",
      "-H", "User-Agent: CAE-Globus-Client/1.0",
      url,
  });
  if (result.empty()) {
    HLOG(kError, "GlobusFileAssimilator: HTTP GET failed for URL: {}", url);
  }
  return result;
}

std::string GlobusFileAssimilator::HttpPost(const std::string& url,
                                            const std::string& access_token,
                                            const std::string& payload) {
  // Use fork/exec curl to avoid NSS crash in clio worker-thread context.
  std::string auth = "Authorization: Bearer " + access_token;
  std::string result = RunCurlCapture({
      "-s", "-L", "-k", "--fail",
      "-X", "POST",
      "-H", "Content-Type: application/json",
      "-H", auth,
      "-H", "Accept: application/json",
      "-H", "User-Agent: CAE-Globus-Client/1.0",
      "-d", payload,
      url,
  });
  if (result.empty()) {
    HLOG(kError, "GlobusFileAssimilator: HTTP POST failed for URL: {}", url);
  }
  return result;
}

std::string GlobusFileAssimilator::GetSubmissionId(
    const std::string& access_token) {
  std::string url = "https://transfer.api.globus.org/v0.10/submission_id";
  std::string response = HttpGet(url, access_token);

  if (response.empty()) {
    return "";
  }

  try {
    nlohmann::json json_response = nlohmann::json::parse(response);
    if (json_response.contains("value")) {
      return json_response["value"];
    } else {
      HLOG(kError,
           "GlobusFileAssimilator: No 'value' field in submission ID response");
      return "";
    }
  } catch (const std::exception& e) {
    HLOG(kError,
         "GlobusFileAssimilator: Failed to parse submission ID response: {}",
         e.what());
    return "";
  }
}

std::string GlobusFileAssimilator::SubmitTransfer(
    const std::string& src_endpoint, const std::string& dst_endpoint,
    const std::string& src_path, const std::string& dst_path,
    const std::string& access_token, const std::string& submission_id) {
  // Create the JSON payload for the transfer request
  nlohmann::json transfer_request;
  transfer_request["DATA_TYPE"] = "transfer";
  transfer_request["submission_id"] = submission_id;
  transfer_request["source_endpoint"] = src_endpoint;
  transfer_request["destination_endpoint"] = dst_endpoint;
  transfer_request["label"] = "CAE Transfer";
  transfer_request["sync_level"] = 0;
  transfer_request["verify_checksum"] = true;

  // Create the transfer item
  nlohmann::json transfer_item;
  transfer_item["DATA_TYPE"] = "transfer_item";
  transfer_item["source_path"] = src_path;
  transfer_item["destination_path"] = dst_path;
  transfer_item["recursive"] = false;

  // Add the transfer item to the DATA array
  nlohmann::json data_array = nlohmann::json::array();
  data_array.push_back(transfer_item);
  transfer_request["DATA"] = data_array;

  // Convert to JSON string
  std::string payload = transfer_request.dump(2);

  HLOG(kDebug,
       "GlobusFileAssimilator: Submitting transfer request with payload: {}",
       payload);

  // Submit the transfer
  std::string url = "https://transfer.api.globus.org/v0.10/transfer";
  std::string response = HttpPost(url, access_token, payload);

  if (response.empty()) {
    return "";
  }

  try {
    nlohmann::json json_response = nlohmann::json::parse(response);
    if (json_response.contains("code") && json_response["code"] == "Accepted") {
      if (json_response.contains("task_id")) {
        return json_response["task_id"];
      } else {
        HLOG(kError,
             "GlobusFileAssimilator: No 'task_id' field in transfer response");
        return "";
      }
    } else {
      HLOG(kError, "GlobusFileAssimilator: Transfer not accepted. Response: {}",
           response);
      return "";
    }
  } catch (const std::exception& e) {
    HLOG(kError, "GlobusFileAssimilator: Failed to parse transfer response: {}",
         e.what());
    return "";
  }
}

int GlobusFileAssimilator::PollTransferStatus(const std::string& task_id,
                                              const std::string& access_token) {
  const int max_attempts = 30;  // 30 attempts with 10s delay = 5 minutes max
  const int delay_seconds = 10;

  for (int attempt = 1; attempt <= max_attempts; ++attempt) {
    HLOG(kDebug,
         "GlobusFileAssimilator: Checking transfer status (attempt {}/{})",
         attempt, max_attempts);

    std::string url = "https://transfer.api.globus.org/v0.10/task/" + task_id;
    std::string response = HttpGet(url, access_token);

    if (response.empty()) {
      HLOG(kError, "GlobusFileAssimilator: Failed to get transfer status");
      // Continue trying rather than failing immediately
      std::this_thread::sleep_for(std::chrono::seconds(delay_seconds));
      continue;
    }

    try {
      nlohmann::json json_response = nlohmann::json::parse(response);

      if (!json_response.contains("status")) {
        HLOG(kError,
             "GlobusFileAssimilator: No 'status' field in status response");
        std::this_thread::sleep_for(std::chrono::seconds(delay_seconds));
        continue;
      }

      std::string status = json_response["status"];
      HLOG(kDebug, "GlobusFileAssimilator: Transfer status: {}", status);

      if (status == "SUCCEEDED") {
        HLOG(kDebug, "GlobusFileAssimilator: Transfer completed successfully");
        return 0;
      } else if (status == "FAILED" || status == "INACTIVE") {
        HLOG(kError, "GlobusFileAssimilator: Transfer failed with status: {}",
             status);
        if (json_response.contains("fatal_error")) {
          HLOG(kError, "GlobusFileAssimilator: Fatal error: {}",
               json_response["fatal_error"].dump());
        }
        if (json_response.contains("nice_status_details")) {
          HLOG(kError, "GlobusFileAssimilator: Details: {}",
               json_response["nice_status_details"].dump());
        }
        return -8;
      }
      // Status is ACTIVE or other intermediate state, continue polling
    } catch (const std::exception& e) {
      HLOG(kError, "GlobusFileAssimilator: Failed to parse status response: {}",
           e.what());
    }

    // Wait before polling again
    std::this_thread::sleep_for(std::chrono::seconds(delay_seconds));
  }

  HLOG(kError,
       "GlobusFileAssimilator: Transfer did not complete within {} seconds",
       max_attempts * delay_seconds);
  return -8;
}

int GlobusFileAssimilator::DownloadFile(const std::string& endpoint_id,
                                        const std::string& remote_path,
                                        const std::string& local_path,
                                        const std::string& transfer_token,
                                        const std::string& https_token) {
  HLOG(kInfo, "==========================================");
  HLOG(kInfo, "Globus File Download Starting");
  HLOG(kInfo, "==========================================");
  HLOG(kInfo, "Endpoint ID:  {}", endpoint_id);
  HLOG(kInfo, "Remote path:  {}", remote_path);
  HLOG(kInfo, "Local path:   {}", local_path);

  HLOG(kDebug, "GlobusFileAssimilator: Downloading file from Globus endpoint");
  HLOG(kDebug,
       "GlobusFileAssimilator: Endpoint: {}, Remote path: {}, Local path: {}",
       endpoint_id, remote_path, local_path);

  try {
    // Get endpoint details to find the HTTPS server
    HLOG(kInfo, "[Step 1/4] Querying Globus endpoint details...");
    std::string endpoint_url =
        "https://transfer.api.globus.org/v0.10/endpoint/" + endpoint_id;
    HLOG(kInfo, "  API URL: {}", endpoint_url);
    std::string endpoint_response = HttpGet(endpoint_url, transfer_token);

    if (endpoint_response.empty()) {
      HLOG(kError, "GlobusFileAssimilator: Failed to get endpoint details from Globus API");
      return -11;
    }
    HLOG(kInfo, "  Endpoint details retrieved successfully");

    // Parse endpoint response to get HTTPS server
    HLOG(kInfo, "[Step 2/4] Parsing endpoint configuration...");
    nlohmann::json endpoint_json = nlohmann::json::parse(endpoint_response);

    std::string https_server;
    if (endpoint_json.contains("https_server") &&
        !endpoint_json["https_server"].is_null()) {
      https_server = endpoint_json["https_server"];
    } else {
      HLOG(
          kError,
          "GlobusFileAssimilator: Endpoint does not have HTTPS server enabled");
      HLOG(kError,
           "GlobusFileAssimilator: Endpoint must have HTTPS access enabled for "
           "local downloads");
      return -12;
    }

    HLOG(kInfo, "  HTTPS server: {}", https_server);

    // Construct the download URL
    HLOG(kInfo, "[Step 3/4] Initiating HTTPS download...");
    // https_server may already include the scheme (e.g. "https://host").
    // URL-encode the path so that spaces and other special characters are
    // properly percent-encoded (e.g. "Calibration Data" → "Calibration%20Data").
    std::string encoded_path = UrlEncodePath(remote_path);
    std::string download_url;
    if (https_server.find("://") != std::string::npos) {
      download_url = https_server + encoded_path;
    } else {
      download_url = "https://" + https_server + encoded_path;
    }
    HLOG(kInfo, "  Download URL: {}", download_url);

    // Download the file using fork/exec curl (avoids NSS/POCO crash).
    // curl writes the body directly to local_path (-o), stdout is the HTTP
    // status code (-w "%{http_code}"), and stderr shows progress/errors.
    HLOG(kInfo, "  Downloading via curl...");
    std::string auth = "Authorization: Bearer " + https_token;
    std::string http_status_str = RunCurlCapture({
        "-s", "-L", "-k",
        "--max-time", "300",
        "-H", auth,
        "-H", "User-Agent: CAE-Globus-Client/1.0",
        "-o", local_path,
        "-w", "%{http_code}",
        download_url,
    });

    HLOG(kInfo, "[Step 4/4] Writing file to local filesystem...");
    HLOG(kInfo, "  Output path: {}", local_path);

    if (http_status_str.empty()) {
      HLOG(kError,
           "GlobusFileAssimilator: curl download failed (no response)");
      return -13;
    }

    int http_status = 0;
    try {
      http_status = std::stoi(http_status_str);
    } catch (...) {
      HLOG(kError,
           "GlobusFileAssimilator: curl returned unexpected status: {}",
           http_status_str);
      return -13;
    }

    if (http_status != 200) {
      HLOG(kError,
           "GlobusFileAssimilator: HTTP GET failed with status {}",
           http_status);
      return -13;
    }

    HLOG(kInfo, "  HTTP Status: {}", http_status);
    HLOG(kInfo, "  File written successfully");
    HLOG(kInfo, "==========================================");
    HLOG(kInfo, "Download Complete!");
    HLOG(kInfo, "==========================================");

    HLOG(kDebug, "GlobusFileAssimilator: File downloaded successfully");
    return 0;

  } catch (const std::exception& e) {
    HLOG(kError, "GlobusFileAssimilator: Exception in DownloadFile: {}",
         e.what());
    return -15;
  } catch (...) {
    HLOG(kError, "GlobusFileAssimilator: Unknown exception in DownloadFile");
    return -15;
  }
}
#endif  // CAE_ENABLE_GLOBUS

}  // namespace clio::cae::core
