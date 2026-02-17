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

#include <chimaera/chimaera.h>
#include <hermes_shm/util/logging.h>
#include <wrp_cae/core/factory/globus_file_assimilator.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <thread>

#ifdef CAE_ENABLE_GLOBUS
#include <Poco/Exception.h>
#include <Poco/Net/Context.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/StreamCopier.h>
#include <Poco/URI.h>

#include <nlohmann/json.hpp>
#include <sstream>
#endif

// Include wrp_cte headers after closing any wrp_cae namespace to avoid Method
// namespace collision
#include <wrp_cte/core/core_client.h>

namespace wrp_cae::core {

GlobusFileAssimilator::GlobusFileAssimilator(
    std::shared_ptr<wrp_cte::core::Client> cte_client)
    : cte_client_(cte_client) {}

int GlobusFileAssimilator::Schedule(const AssimilationCtx& ctx) {
#ifndef CAE_ENABLE_GLOBUS
  HLOG(kError, "GlobusFileAssimilator: Globus support not compiled in");
  return -20;
#else
  // Validate source is a Globus URL (either web URL or globus:// URI)
  bool is_globus_web_url = (ctx.src.find("https://app.globus.org") == 0);
  std::string src_protocol = GetUrlProtocol(ctx.src);
  bool is_globus_uri = (src_protocol == "globus");

  if (!is_globus_web_url && !is_globus_uri) {
    HLOG(kError,
         "GlobusFileAssimilator: Source must be a Globus web URL or globus:// "
         "URI, got protocol '{}'",
         src_protocol);
    return -2;
  }

  // Validate destination protocol
  bool is_dst_globus_web_url = (ctx.dst.find("https://app.globus.org") == 0);
  std::string dst_protocol = GetUrlProtocol(ctx.dst);
  bool is_valid_dst = (dst_protocol == "file" || dst_protocol == "globus" ||
                       is_dst_globus_web_url);

  if (!is_valid_dst) {
    HLOG(kError,
         "GlobusFileAssimilator: Destination must be file://, globus://, or "
         "Globus web URL, got protocol '{}'",
         dst_protocol);
    return -3;
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
      return -1;
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
      return -4;
    }
  } else {
    // Parse as standard globus:// URI
    if (!ParseGlobusUri(ctx.src, src_endpoint, src_path)) {
      HLOG(kError, "GlobusFileAssimilator: Failed to parse source URI: '{}'",
           ctx.src);
      return -4;
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
      return -5;
    }

    HLOG(kInfo, "Source:       {}", ctx.src);
    HLOG(kInfo, "Destination:  {}", ctx.dst);

    // Download file from Globus to local filesystem
    int download_result =
        DownloadFile(src_endpoint, src_path, dst_path, access_token);
    if (download_result != 0) {
      HLOG(kError,
           "GlobusFileAssimilator: Failed to download file from Globus (error code: {})",
           download_result);
      return download_result;
    }

    HLOG(kInfo, "Transfer completed successfully!");
    HLOG(kDebug,
         "GlobusFileAssimilator: Successfully downloaded file to local "
         "filesystem");
    return 0;

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
        return -5;
      }
    } else {
      // Parse as standard globus:// URI
      if (!ParseGlobusUri(ctx.dst, dst_endpoint, dst_path)) {
        HLOG(kError,
             "GlobusFileAssimilator: Failed to parse destination URI: '{}'",
             ctx.dst);
        return -5;
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
      return -6;
    }

    HLOG(kDebug, "GlobusFileAssimilator: Obtained submission ID: '{}'",
         submission_id);

    // Submit transfer
    std::string task_id = SubmitTransfer(src_endpoint, dst_endpoint, src_path,
                                         dst_path, access_token, submission_id);
    if (task_id.empty()) {
      HLOG(kError,
           "GlobusFileAssimilator: Failed to submit transfer to Globus API");
      return -7;
    }

    HLOG(
        kDebug,
        "GlobusFileAssimilator: Transfer submitted successfully, task ID: '{}'",
        task_id);

    // Poll for transfer completion
    int poll_result = PollTransferStatus(task_id, access_token);
    if (poll_result != 0) {
      HLOG(kError, "GlobusFileAssimilator: Transfer failed or timed out");
      return poll_result;
    }

    HLOG(kDebug, "GlobusFileAssimilator: Transfer completed successfully");
    return 0;
  }
#endif
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
std::string GlobusFileAssimilator::HttpGet(const std::string& url,
                                           const std::string& access_token) {
  try {
    Poco::URI uri(url);

    // Create an SSL Context for HTTPS
    Poco::Net::Context::Ptr context =
        new Poco::Net::Context(Poco::Net::Context::CLIENT_USE, "", "", "",
                               Poco::Net::Context::VERIFY_NONE, 9, false,
                               "ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");

    // Set up the session with timeout
    Poco::Net::HTTPSClientSession session(uri.getHost(), uri.getPort(),
                                          context);
    session.setTimeout(Poco::Timespan(30, 0));  // 30 second timeout

    // Prepare the request
    std::string path = uri.getPathAndQuery();
    if (path.empty()) {
      path = "/";
    }

    Poco::Net::HTTPRequest request(Poco::Net::HTTPRequest::HTTP_GET, path,
                                   Poco::Net::HTTPMessage::HTTP_1_1);
    request.set("Authorization", "Bearer " + access_token);
    request.set("Accept", "application/json");
    request.set("User-Agent", "CAE-Globus-Client/1.0");

    // Send the request
    session.sendRequest(request);

    // Get the response
    Poco::Net::HTTPResponse response;
    std::istream& rs = session.receiveResponse(response);
    std::stringstream responseBody;
    Poco::StreamCopier::copyStream(rs, responseBody);
    std::string responseStr = responseBody.str();

    if (response.getStatus() == Poco::Net::HTTPResponse::HTTP_OK) {
      return responseStr;
    } else {
      HLOG(kError, "GlobusFileAssimilator: HTTP GET failed with status {} {}",
           response.getStatus(), response.getReason());
      HLOG(kError, "GlobusFileAssimilator: Response body: {}", responseStr);
      return "";
    }
  } catch (Poco::Exception& e) {
    HLOG(kError, "GlobusFileAssimilator: POCO exception in HttpGet: {}",
         e.displayText());
    return "";
  } catch (std::exception& e) {
    HLOG(kError, "GlobusFileAssimilator: Exception in HttpGet: {}", e.what());
    return "";
  }
}

std::string GlobusFileAssimilator::HttpPost(const std::string& url,
                                            const std::string& access_token,
                                            const std::string& payload) {
  try {
    Poco::URI uri(url);

    // Create an SSL Context for HTTPS
    Poco::Net::Context::Ptr context =
        new Poco::Net::Context(Poco::Net::Context::CLIENT_USE, "", "", "",
                               Poco::Net::Context::VERIFY_NONE, 9, false,
                               "ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");

    // Set up the session
    Poco::Net::HTTPSClientSession session(uri.getHost(), uri.getPort(),
                                          context);
    session.setTimeout(Poco::Timespan(30, 0));  // 30 second timeout

    // Prepare the request
    std::string path = uri.getPathAndQuery();
    if (path.empty()) {
      path = "/";
    }

    Poco::Net::HTTPRequest request(Poco::Net::HTTPRequest::HTTP_POST, path,
                                   Poco::Net::HTTPMessage::HTTP_1_1);
    request.setContentType("application/json");
    request.set("Authorization", "Bearer " + access_token);
    request.set("Accept", "application/json");
    request.set("User-Agent", "CAE-Globus-Client/1.0");
    request.setContentLength(payload.length());

    // Send the request
    std::ostream& os = session.sendRequest(request);
    os << payload;

    // Get the response
    Poco::Net::HTTPResponse response;
    std::istream& responseStream = session.receiveResponse(response);
    std::stringstream responseBody;
    Poco::StreamCopier::copyStream(responseStream, responseBody);
    std::string responseStr = responseBody.str();

    // Check for success status codes
    if (response.getStatus() == Poco::Net::HTTPResponse::HTTP_OK ||
        response.getStatus() == Poco::Net::HTTPResponse::HTTP_ACCEPTED) {
      return responseStr;
    } else {
      HLOG(kError, "GlobusFileAssimilator: HTTP POST failed with status {} {}",
           response.getStatus(), response.getReason());
      HLOG(kError, "GlobusFileAssimilator: Response body: {}", responseStr);
      return "";
    }
  } catch (Poco::Exception& e) {
    HLOG(kError, "GlobusFileAssimilator: POCO exception in HttpPost: {}",
         e.displayText());
    return "";
  } catch (std::exception& e) {
    HLOG(kError, "GlobusFileAssimilator: Exception in HttpPost: {}", e.what());
    return "";
  }
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
                                        const std::string& access_token) {
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
    std::string endpoint_response = HttpGet(endpoint_url, access_token);

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
    std::string download_url = "https://" + https_server + remote_path;
    HLOG(kInfo, "  Download URL: {}", download_url);

    // Download the file using HTTPS
    Poco::URI uri(download_url);

    // Create an SSL Context for HTTPS
    Poco::Net::Context::Ptr context =
        new Poco::Net::Context(Poco::Net::Context::CLIENT_USE, "", "", "",
                               Poco::Net::Context::VERIFY_NONE, 9, false,
                               "ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");

    // Set up the session with extended timeout for large files
    Poco::Net::HTTPSClientSession session(uri.getHost(), uri.getPort(),
                                          context);
    session.setTimeout(
        Poco::Timespan(300, 0));  // 5 minute timeout for downloads

    // Prepare the request
    std::string path = uri.getPathAndQuery();
    if (path.empty()) {
      path = "/";
    }

    Poco::Net::HTTPRequest request(Poco::Net::HTTPRequest::HTTP_GET, path,
                                   Poco::Net::HTTPMessage::HTTP_1_1);
    request.set("Authorization", "Bearer " + access_token);
    request.set("User-Agent", "CAE-Globus-Client/1.0");

    // Send the request
    HLOG(kInfo, "  Sending HTTPS request...");
    session.sendRequest(request);

    // Get the response
    Poco::Net::HTTPResponse response;
    std::istream& rs = session.receiveResponse(response);

    if (response.getStatus() != Poco::Net::HTTPResponse::HTTP_OK) {
      HLOG(kError, "GlobusFileAssimilator: HTTP GET failed with status {} {}",
           response.getStatus(), response.getReason());
      return -13;
    }
    HLOG(kInfo, "  HTTP Status: {} {}", response.getStatus(),
         response.getReason());
    if (response.has("Content-Length")) {
      HLOG(kInfo, "  Content-Length: {} bytes", response.get("Content-Length"));
    }

    // Open output file
    HLOG(kInfo, "[Step 4/4] Writing file to local filesystem...");
    HLOG(kInfo, "  Output path: {}", local_path);
    std::ofstream output_file(local_path, std::ios::binary);
    if (!output_file) {
      HLOG(kError, "GlobusFileAssimilator: Failed to open output file: {}",
           local_path);
      return -14;
    }

    // Copy the response stream to the output file
    HLOG(kInfo, "  Downloading and writing file...");
    HLOG(kDebug, "GlobusFileAssimilator: Writing file to {}", local_path);
    Poco::StreamCopier::copyStream(rs, output_file);
    output_file.close();

    HLOG(kInfo, "  File written successfully");
    HLOG(kInfo, "==========================================");
    HLOG(kInfo, "Download Complete!");
    HLOG(kInfo, "==========================================");

    HLOG(kDebug, "GlobusFileAssimilator: File downloaded successfully");
    return 0;

  } catch (Poco::Exception& e) {
    HLOG(kError, "GlobusFileAssimilator: POCO exception in DownloadFile: {}",
         e.displayText());
    return -15;
  } catch (const std::exception& e) {
    HLOG(kError, "GlobusFileAssimilator: Exception in DownloadFile: {}",
         e.what());
    return -15;
  }
}
#endif  // CAE_ENABLE_GLOBUS

}  // namespace wrp_cae::core
