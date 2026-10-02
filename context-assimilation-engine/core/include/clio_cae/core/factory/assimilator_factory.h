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

#ifndef CLIO_CAE_CORE_ASSIMILATOR_FACTORY_H_
#define CLIO_CAE_CORE_ASSIMILATOR_FACTORY_H_

#include <memory>
#include <string>
#include <clio_cae/core/factory/base_assimilator.h>

namespace clio::cae::core {

// Forward declarations
class BinaryFileAssimilator;
// Kept opaque here (defined under CLIO_ENABLE_S3_REST in s3_conn_pool.h): only
// the S3 assimilator dereferences it, and this header is compiled in all
// configurations. A null pointer means "no pool" (non-S3 callers, or an S3
// build that chose not to pool).
class S3ConnectionPool;

}  // namespace clio::cae::core

namespace clio::cte::core {
class Client;
}  // namespace clio::cte::core

namespace clio::cae::core {

/**
 * AssimilatorFactory - Factory for creating appropriate assimilator instances
 * Selects the correct assimilator based on the source URL protocol
 */
class AssimilatorFactory {
 public:
  /**
   * Constructor with CTE client and an optional S3 keep-alive connection pool.
   * @param cte_client Shared pointer to initialized CTE client
   * @param s3_pool    Long-lived S3 connection pool (owned by Runtime), or
   *                   nullptr; passed on to the S3 assimilator so cross-object
   *                   sockets are reused. Ignored by every other assimilator.
   */
  explicit AssimilatorFactory(
      std::shared_ptr<clio::cte::core::Client> cte_client,
      S3ConnectionPool* s3_pool = nullptr);

  /**
   * Get an assimilator instance for the given source URL
   * @param src Source URL (protocol::path format)
   * @return Unique pointer to an assimilator instance, or nullptr if unsupported
   */
  std::unique_ptr<BaseAssimilator> Get(const std::string& src);

 private:
  /**
   * Extract protocol from URL (part before ::)
   * @param url URL in format protocol::path
   * @return Protocol string, or empty string if no protocol found
   */
  std::string GetUrlProtocol(const std::string& url);

  std::shared_ptr<clio::cte::core::Client> cte_client_;
  S3ConnectionPool* s3_pool_ = nullptr;  ///< not owned; may be null
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_ASSIMILATOR_FACTORY_H_
