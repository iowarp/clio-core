/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#include <clio_cte/filesystem/filesystem_client.h>

// Process-wide filesystem client singleton.
CLIO_CTE_DEFINE_GLOBAL_PTR_VAR_CC(clio::cte::filesystem::Client, g_fs_client);

namespace clio::cte::filesystem {

bool CLIO_CFS_CLIENT_INIT(const std::string &config_path,
                          const chi::PoolQuery &pool_query) {
  // TODO(#552): create-or-bind the filesystem pool (over the configured CTE
  // core pool), allocate the Client, and publish it via
  // CTP_SET_GLOBAL_PTR_VAR(... g_fs_client ...). Mirror
  // clio::cte::core::CLIO_CTE_CLIENT_INIT.
  (void)config_path;
  (void)pool_query;
  return false;
}

}  // namespace clio::cte::filesystem
