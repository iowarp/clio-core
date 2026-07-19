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

#ifndef CLIO_CAE_CORE_GCS_FILE_ASSIMILATOR_H_
#define CLIO_CAE_CORE_GCS_FILE_ASSIMILATOR_H_

#include <clio_cae/core/factory/base_assimilator.h>
#include <string>
#include <memory>

// Forward declaration
namespace clio::cte::core {
class Client;
}  // namespace clio::cte::core

namespace clio::cae::core {

/**
 * GcsFileAssimilator - Imports an object straight from Google Cloud Storage
 * into CTE.
 *
 * The source URL is `gs://bucket/object` (the `gcs://bucket/object` form is
 * also accepted). Object bytes are streamed in chunks into CTE using the same
 * tag + chunked-blob flow as BinaryFileAssimilator; only the byte source
 * differs (a GCS ObjectReadStream instead of a local file).
 *
 * Credentials are resolved through Google Application Default Credentials
 * (GOOGLE_APPLICATION_CREDENTIALS, gcloud ADC, or the GCE/GKE metadata
 * server). An optional GCS-compatible endpoint (e.g. fake-gcs-server) can be
 * set via GCS_ENDPOINT, in which case anonymous credentials are used.
 */
class GcsFileAssimilator : public BaseAssimilator {
 public:
  /**
   * Constructor with CTE client
   * @param cte_client Shared pointer to initialized CTE client
   */
  explicit GcsFileAssimilator(std::shared_ptr<clio::cte::core::Client> cte_client);

  /**
   * Schedule assimilation tasks for a GCS object
   * This is a coroutine that uses co_await for async CTE operations.
   * @param ctx Assimilation context with source, destination, and metadata
   * @param error_code Output: 0 on success, non-zero error code on failure
   * @return TaskResume for coroutine suspension/resumption
   */
  clio::run::TaskResume Schedule(const AssimilationCtx& ctx, int& error_code) override;

 private:
  /**
   * Extract protocol from URL (part before :: or ://)
   * @param url URL in format protocol::path or protocol://path
   * @return Protocol string, or empty string if no protocol found
   */
  std::string GetUrlProtocol(const std::string& url);

  /**
   * Extract path from URL (part after :: or ://)
   * @param url URL in format protocol::path or protocol://path
   * @return Path string, or empty string if no protocol found
   */
  std::string GetUrlPath(const std::string& url);

  /**
   * Parse a GCS source URL into its bucket and object components.
   * Accepts `gs://bucket/object`, `gcs://bucket/object`, and the `::` forms.
   * @param url Source URL
   * @param bucket Output: bucket name
   * @param object Output: object name (everything after the first '/')
   * @return true on success, false if the URL is malformed
   */
  bool ParseGcsUrl(const std::string& url, std::string& bucket, std::string& object);

  std::shared_ptr<clio::cte::core::Client> cte_client_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_GCS_FILE_ASSIMILATOR_H_
