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

#ifndef CLIO_CAE_CORE_GLOBUS_FILE_ASSIMILATOR_H_
#define CLIO_CAE_CORE_GLOBUS_FILE_ASSIMILATOR_H_

#include <clio_cae/core/factory/base_assimilator.h>
#include <string>
#include <memory>

// Forward declaration
namespace clio::cte::core {
class Client;
}  // namespace clio::cte::core

namespace clio::cae::core {

/**
 * GlobusFileAssimilator - Handles assimilation of files via Globus transfer
 * Supports Globus-to-Globus and Globus-to-local transfers
 *
 * Protocol format:
 * - Source: globus://<endpoint_id>/<path>
 * - Destination: globus://<endpoint_id>/<path> OR file::/local/path
 */
class GlobusFileAssimilator : public BaseAssimilator {
 public:
  /**
   * Constructor with CTE client
   * @param cte_client Shared pointer to initialized CTE client
   */
  explicit GlobusFileAssimilator(std::shared_ptr<clio::cte::core::Client> cte_client);

  /**
   * Schedule assimilation tasks for a Globus file transfer
   * @param ctx Assimilation context with source, destination, and metadata
   * @return 0 on success, non-zero error code on failure
   *
   * Error codes:
   * -1:  Missing access token
   * -2:  Invalid source protocol (not globus)
   * -3:  Invalid destination protocol (not file or globus)
   * -4:  Failed to parse source URI
   * -5:  Failed to parse destination URI
   * -6:  Failed to get submission ID (Globus-to-Globus)
   * -7:  Failed to submit transfer (Globus-to-Globus)
   * -8:  Transfer failed or timed out (Globus-to-Globus)
   * -11: Failed to get endpoint details (Globus-to-local)
   * -12: Endpoint does not have HTTPS access enabled (Globus-to-local)
   * -13: HTTP download request failed (Globus-to-local)
   * -14: Failed to open local output file (Globus-to-local)
   * -15: Exception during download (Globus-to-local)
   * -20: Globus support not compiled in
   */
  clio::run::TaskResume Schedule(const AssimilationCtx& ctx, int& error_code) override;

 private:
  /**
   * Extract protocol from URL (part before ::)
   * @param url URL in format protocol::path
   * @return Protocol string, or empty string if no protocol found
   */
  std::string GetUrlProtocol(const std::string& url);

  /**
   * Extract path from URL (part after ::)
   * @param url URL in format protocol::path
   * @return Path string, or empty string if no protocol found
   */
  std::string GetUrlPath(const std::string& url);

  /**
   * Parse Globus URI into endpoint ID and path
   * Format: globus://<endpoint_id>/<path>
   * @param uri Globus URI to parse
   * @param endpoint_id Output: Extracted endpoint ID
   * @param path Output: Extracted path
   * @return true if parsing succeeded, false otherwise
   */
  bool ParseGlobusUri(const std::string& uri, std::string& endpoint_id,
                      std::string& path);

  /**
   * Parse Globus web URL into endpoint ID and path
   * Format: https://app.globus.org/file-manager?origin_id=<endpoint_id>&origin_path=<encoded_path>
   * @param url Globus web URL to parse
   * @param endpoint_id Output: Extracted endpoint ID
   * @param path Output: Extracted path (URL-decoded)
   * @return true if parsing succeeded, false otherwise
   */
  bool ParseGlobusWebUrl(const std::string& url, std::string& endpoint_id,
                         std::string& path);

  /**
   * URL decode a string (converts %XX to characters)
   * @param encoded URL-encoded string
   * @return Decoded string
   */
  std::string UrlDecode(const std::string& encoded);

#ifdef CAE_ENABLE_GLOBUS
  /**
   * Perform HTTP GET request using POCO
   * @param url URL to fetch
   * @param access_token Globus access token
   * @return Response body, or empty string on error
   */
  std::string HttpGet(const std::string& url, const std::string& access_token);

  /**
   * Perform HTTP POST request using POCO
   * @param url URL to post to
   * @param access_token Globus access token
   * @param payload JSON payload to send
   * @return Response body, or empty string on error
   */
  std::string HttpPost(const std::string& url, const std::string& access_token,
                       const std::string& payload);

  /**
   * Get a submission ID from Globus API
   * @param access_token Globus access token
   * @return Submission ID, or empty string on error
   */
  std::string GetSubmissionId(const std::string& access_token);

  /**
   * Submit a transfer request to Globus API
   * @param src_endpoint Source endpoint ID
   * @param dst_endpoint Destination endpoint ID
   * @param src_path Source file path
   * @param dst_path Destination file path
   * @param access_token Globus access token
   * @param submission_id Submission ID from GetSubmissionId
   * @return Task ID, or empty string on error
   */
  std::string SubmitTransfer(const std::string& src_endpoint,
                            const std::string& dst_endpoint,
                            const std::string& src_path,
                            const std::string& dst_path,
                            const std::string& access_token,
                            const std::string& submission_id);

  /**
   * Poll for transfer completion status
   * @param task_id Globus task ID
   * @param access_token Globus access token
   * @return 0 on success, -8 on failure/timeout
   */
  int PollTransferStatus(const std::string& task_id,
                         const std::string& access_token);

  /**
   * Download a file from Globus endpoint to local filesystem
   * @param endpoint_id Globus endpoint ID
   * @param remote_path Path on the Globus endpoint
   * @param local_path Local filesystem path to write the file
   * @param access_token Globus access token
   * @return 0 on success, negative error code on failure
   *
   * Error codes:
   * -11: Failed to get endpoint details
   * -12: Endpoint does not have HTTPS access enabled
   * -13: HTTP download request failed
   * -14: Failed to open local output file
   * -15: Exception during download
   */
  int DownloadFile(const std::string& endpoint_id,
                   const std::string& remote_path,
                   const std::string& local_path,
                   const std::string& transfer_token,
                   const std::string& https_token);
#endif

  std::shared_ptr<clio::cte::core::Client> cte_client_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_GLOBUS_FILE_ASSIMILATOR_H_
