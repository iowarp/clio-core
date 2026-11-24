/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#ifndef CLIO_CAE_CORE_HTTP_FILE_ASSIMILATOR_H_
#define CLIO_CAE_CORE_HTTP_FILE_ASSIMILATOR_H_

#include <clio_cae/core/factory/base_assimilator.h>

#include <memory>
#include <string>

namespace clio::cte::core {
class Client;
}  // namespace clio::cte::core

namespace clio::cae::core {

/**
 * http:// and https:// sources, downloaded to a file:: destination.
 *
 * Per-site credentials and request parameters come from the config file that
 * http_auth.h describes (~/.clio/cae_http_auth.ini), or a bearer token from
 * the OMNI src_token. Large files arrive in parallel ranged GETs (see
 * http_fetch.h for the CAE_HTTP_* knobs). A destination that already holds
 * exactly the source's bytes is left alone, so a failed OMNI run can simply be
 * run again.
 *
 * OMNI:
 *   - src: "https://host/path/file.zip"
 *     dst: "file::/lus/flare/projects/<proj>/<user>/file.zip"
 *     format: "binary"
 */
class HttpFileAssimilator : public BaseAssimilator {
 public:
  explicit HttpFileAssimilator(
      std::shared_ptr<clio::cte::core::Client> cte_client);

  clio::run::TaskResume Schedule(const AssimilationCtx& ctx,
                                 int& error_code) override;

  /** The blocking body of Schedule, returning its error code. */
  static int Run(const AssimilationCtx& ctx);

 private:
  std::shared_ptr<clio::cte::core::Client> cte_client_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_HTTP_FILE_ASSIMILATOR_H_
