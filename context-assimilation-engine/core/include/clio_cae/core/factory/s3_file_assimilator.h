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

#ifndef CLIO_CAE_CORE_S3_FILE_ASSIMILATOR_H_
#define CLIO_CAE_CORE_S3_FILE_ASSIMILATOR_H_

#include <clio_cae/core/factory/base_assimilator.h>
#include <string>
#include <memory>

// Forward declaration
namespace clio::cte::core {
class Client;
}  // namespace clio::cte::core

namespace clio::cae::core {

/**
 * S3FileAssimilator - Imports an object straight from Amazon S3 (or any
 * S3-compatible endpoint, e.g. MinIO) into CTE.
 *
 * The source URL is `s3://bucket/key` (the `s3::bucket/key` form is also
 * accepted). Object bytes are streamed in chunks into CTE using the same
 * tag + chunked-blob flow as BinaryFileAssimilator; only the byte source
 * differs (an S3 GetObject response stream instead of a local file).
 *
 * Credentials, region, and an optional S3-compatible endpoint are resolved
 * from the standard AWS environment at assimilation time
 * (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN, profiles,
 * instance roles; AWS_DEFAULT_REGION; S3_ENDPOINT or AWS_ENDPOINT_URL).
 */
class S3FileAssimilator : public BaseAssimilator {
 public:
  /**
   * Constructor with CTE client
   * @param cte_client Shared pointer to initialized CTE client
   */
  explicit S3FileAssimilator(std::shared_ptr<clio::cte::core::Client> cte_client);

  /**
   * Schedule assimilation tasks for an S3 object
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
   * Parse an S3 source URL into its bucket and key components.
   * Accepts `s3://bucket/key` and `s3::bucket/key`.
   * @param url Source URL
   * @param bucket Output: bucket name
   * @param key Output: object key (everything after the first '/')
   * @return true on success, false if the URL is malformed
   */
  bool ParseS3Url(const std::string& url, std::string& bucket, std::string& key);

  std::shared_ptr<clio::cte::core::Client> cte_client_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_S3_FILE_ASSIMILATOR_H_
